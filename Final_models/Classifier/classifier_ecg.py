from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from Final_models.WGAN.preprocessing_utils import reorder_features


class ECG_Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad, dropout=0.2):
        super(ECG_Conv_Block, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, padding=pad)
        self.batch_norm = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout1d(dropout)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.max_pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, ecg):
        x = self.conv(ecg)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.max_pool(x)
        return x


class ECG_Residual_Block(nn.Module):
    def __init__(self, channels, k_size, dropout=0.2):
        super(ECG_Residual_Block, self).__init__()
        pad = (k_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout1d(dropout)
        self.leaky_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        identity = x
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.leaky_relu(out)
        return out


class Classifier(nn.Module):
    """
    ECG-ONLY classifier — num_risk_factors=0 disables the risk_mlp branch
    entirely, so predictions come purely from ECG morphology. This is a
    diagnostic run: if the MI-default bias persists with no CRF input at all,
    the bias lives in the ECG/synthetic-data pipeline rather than the risk
    factor branch. If it disappears or changes character, that points at the
    CRF branch as the dominant cause.

    Same backbone as the deepened 3-class CRF+ECG model (5 downsampling
    blocks + 2 residual blocks, large->small kernels, gradual channel ramp
    16->256) so any difference in behaviour can be attributed to the absence
    of risk factors, not a different ECG architecture.
    """

    def __init__(self, num_leads=3, num_risk_factors=0, num_classes=3,
                conv_dropout=0.3, head_dropout=0.4):
        super(Classifier, self).__init__()

        chs = [num_leads, 16, 32, 64, 128, 256]
        k_sizes = [11, 11, 9, 7, 5]
        blocks = []
        for cin, cout, k_size in zip(chs[:-1], chs[1:], k_sizes):
            pad = (k_size - 1) // 2
            blocks.append(ECG_Conv_Block(cin, cout, k_size, pad, dropout=conv_dropout))
        self.conv_blocks = nn.Sequential(*blocks)

        self.res_blocks = nn.Sequential(
            ECG_Residual_Block(256, k_size=5, dropout=conv_dropout),
            ECG_Residual_Block(256, k_size=5, dropout=conv_dropout),
        )

        ecg_feat_dim = 256

        # Forced off for this experiment regardless of what's passed in —
        # this run is specifically ECG-only
        self.risk_mlp = None
        risk_emb_dim = 0

        self.head = nn.Sequential(
            nn.Linear(ecg_feat_dim + risk_emb_dim, 128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, ecg, risk_factors=None):
        x = self.conv_blocks(ecg)
        x = self.res_blocks(x)
        x = x.mean(dim=-1)
        # risk_factors is accepted but ignored — kept in the signature so the
        # existing train/eval loop (which always passes risk) doesn't need
        # special-casing
        logits = self.head(x)
        return logits


def make_metrics(num_classes: int):
    return MetricCollection({
        "acc_micro": MulticlassAccuracy(num_classes=num_classes, average='micro'),
        "acc_macro": MulticlassAccuracy(num_classes=num_classes, average='macro'),
        "acc_weighted": MulticlassAccuracy(num_classes=num_classes, average='weighted'),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average='macro'),
        "f1_micro": MulticlassF1Score(num_classes=num_classes, average='micro'),
        "f1_weighted": MulticlassF1Score(num_classes=num_classes, average='weighted'),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average='macro'),
        "recall_micro": MulticlassRecall(num_classes=num_classes, average='micro'),
        "recall_weighted": MulticlassF1Score(num_classes=num_classes, average='weighted'),
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average='macro'),
        "precision_micro": MulticlassPrecision(num_classes=num_classes, average='micro'),
        "precision_weighted": MulticlassF1Score(num_classes=num_classes, average='weighted'),
        "f1_per_class": MulticlassF1Score(num_classes=num_classes, average='none'),
        "recall_per_class": MulticlassRecall(num_classes=num_classes, average='none'),
        "precision_per_class": MulticlassPrecision(num_classes=num_classes, average='none'),
        "conf_mat": MulticlassConfusionMatrix(num_classes=num_classes)
    })


@torch.no_grad()
def run_eval_epoch(model: nn.Module, loader: DataLoader[TensorDataset], device: torch.device, metrics: MetricCollection):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n = 0

    for i, (ecg, risk, labels) in enumerate(loader):
        ecg: torch.Tensor
        risk: torch.Tensor
        labels: torch.Tensor
        ecg = ecg.to(device)
        # risk is loaded but not sent to the model — ECG-only experiment
        labels = labels.to(device)
        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        logits = model(ecg)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        metrics.update(preds, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs

    out = metrics.compute()
    out['loss'] = torch.tensor(total_loss / max(n, 1), device=device)
    return out


def run_train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: optim.Adam, cost_function: nn.CrossEntropyLoss, metrics: MetricCollection):
    model.train()
    metrics.reset()

    total_loss = 0.0
    n = 0
    correct = 0

    for i, (ecg, risk, labels) in enumerate(loader):
        ecg: torch.Tensor
        risk: torch.Tensor
        labels: torch.Tensor
        ecg = ecg.to(device)
        labels = labels.to(device)
        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        optimizer.zero_grad()
        logits = model(ecg)
        loss = cost_function(logits, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, dim=1)
        metrics.update(preds, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        correct += (preds == labels).sum().item()

    out = metrics.compute()
    out['loss'] = torch.tensor(total_loss/max(n, 1), device=device)
    return out


def pretty_print_metrics(prefix: str, out: dict, num_classes: int):
    scalars = ["loss", "acc_micro", "acc_macro", "acc_weighted",
               "f1_macro", "f1_weighted", "recall_macro", "precision_macro"]
    msg = prefix + " | " + " | ".join(
        f"{k}={out[k].item():.4f}" for k in scalars if k in out
    )
    print(msg)

    if "f1_per_class" in out:
        f1_pc = out["f1_per_class"].detach().cpu().numpy()
        print(f"{prefix} f1_per_class: {np.round(f1_pc, 4)}")

    if "conf_mat" in out:
        cm = out["conf_mat"].detach().cpu().numpy()
        print(f"{prefix} confusion matrix:\n{cm}")


def train(classifier: nn.Module, trainloader: DataLoader, validloader: DataLoader, device: torch.device, optimizer, cost_function, model_path: str, num_epochs: int, num_classes: int, testloader_synth, testloader_real):
    train_metrics = make_metrics(num_classes).to(device)
    val_metrics = make_metrics(num_classes=num_classes).to(device)

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, num_epochs+1):
        train_out = run_train_epoch(
            classifier, trainloader, device, optimizer, cost_function, train_metrics)
        val_out = run_eval_epoch(classifier, validloader, device, val_metrics)

        pretty_print_metrics(
            f"Epoch {epoch:03d} TRAIN", train_out, num_classes)
        pretty_print_metrics(f"Epoch {epoch:03d} VAL  ", val_out, num_classes)

        val_f1_macro = val_out["f1_macro"].item()
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            os.makedirs(model_path, exist_ok=True)
            torch.save(classifier.state_dict(), os.path.join(
                model_path, "best_model.pth"))

        history.append({
            "epoch": epoch,
            "train_loss": train_out["loss"].item(),
            "train_acc_micro": train_out["acc_micro"].item(),
            "train_acc_macro": train_out["acc_macro"].item(),
            "train_acc_weighted": train_out["acc_weighted"].item(),
            "train_f1_macro": train_out["f1_macro"].item(),
            "train_f1_micro": train_out["f1_micro"].item(),
            "train_f1_weighted": train_out["f1_weighted"].item(),
            "val_loss": val_out["loss"].item(),
            "val_acc_micro": val_out["acc_micro"].item(),
            "val_acc_macro": val_out["acc_macro"].item(),
            "val_acc_weighted": val_out["acc_weighted"].item(),
            "val_f1_macro": val_out["f1_macro"].item(),
            "val_f1_weighted": val_out["f1_weighted"].item(),
        })

        print(
            f"Epoch: {epoch:03d} | Train Loss: {train_out['loss']:.4f} | Val Loss : {val_out['loss']:.4f} | Val F1(Macro) : {val_out['f1_macro']:.4f} | Val Acc(micro) : {val_out['acc_micro']:.4f} | Val Acc(weighted) : {val_out['acc_weighted']:.4f}")

    test_metrics = make_metrics(num_classes).to(device)
    synth_test_out = run_eval_epoch(
        classifier, testloader_synth, device, test_metrics)

    test_metrics2 = make_metrics(num_classes).to(device)
    real_test_out = run_eval_epoch(
        classifier, testloader_real, device, test_metrics2)

    pretty_print_metrics("SYNTH TEST", synth_test_out, num_classes)
    pretty_print_metrics("REAL  TEST", real_test_out, num_classes)

    ckpt = {
        "epochs": num_epochs,
        "model_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "synth_test": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in synth_test_out.items()},
        "real_test": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in real_test_out.items()},
        "best_val_f1_macro": best_val_f1,
    }
    torch.save(ckpt, os.path.join(model_path, "checkpoint.pth"))


def main(trainloader, validloader, num_epochs, num_classes, device, testloader_synth, testloader_real):
    classifier = Classifier(
        num_leads=3, num_risk_factors=0, num_classes=num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    cost_function = nn.CrossEntropyLoss()
    classifier_model_num = 0
    while os.path.exists(f"Final_models/Classifier/models/classifier_ecg_only_{classifier_model_num}"):
        classifier_model_num += 1
    model_path = f"Final_models/Classifier/models/classifier_ecg_only_{classifier_model_num}"
    os.makedirs(model_path)
    train(classifier, trainloader, validloader, device, optimizer, cost_function,
          model_path, num_epochs, num_classes, testloader_synth, testloader_real)


if __name__ == "__main__":
    dataset = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 30
    BATCH_SIZE = 64

    NUM_CLASSES = 3   # None, MI, Stroke — Syncope dropped, matches latest model

    df = pd.read_csv("augmented_dataset.csv")
    synth_data = torch.load(
        f"synth_datasets/{dataset}_real_synth_dataset_filtered.pth", weights_only=False)

    def filter_to_3class(dataset_obj):
        """
        Drops Syncope (label 3), keeps None/MI/Stroke (0/1/2) as-is.
        Risk factor tensor is kept in the TensorDataset for loader
        compatibility, but the model and train/eval loops above ignore it.
        """
        ecg_list, crf_list, label_list = [], [], []
        for i in range(len(dataset_obj)):
            ecg, crf, label = dataset_obj[i]
            lbl = label.item() if torch.is_tensor(label) else label
            if lbl in (0, 1, 2):
                ecg_list.append(ecg)
                crf_list.append(crf)
                label_list.append(lbl)
        ecg_t   = torch.stack(ecg_list)
        crf_t   = torch.stack(crf_list)
        label_t = torch.tensor(label_list, dtype=torch.long)
        return TensorDataset(ecg_t, crf_t, label_t)

    train_dataset_3c = filter_to_3class(synth_data['train'].dataset)
    valid_dataset_3c = filter_to_3class(synth_data['valid'].dataset)
    test_dataset_3c  = filter_to_3class(synth_data['test'].dataset)

    print(f"3-class train size: {len(train_dataset_3c)} (was {len(synth_data['train'].dataset)})")
    print(f"3-class valid size: {len(valid_dataset_3c)} (was {len(synth_data['valid'].dataset)})")
    print(f"3-class test size:  {len(test_dataset_3c)} (was {len(synth_data['test'].dataset)})")

    trainloader = DataLoader(
        train_dataset_3c, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(
        valid_dataset_3c, batch_size=BATCH_SIZE, shuffle=True)
    testloader_synth = DataLoader(
        test_dataset_3c, batch_size=BATCH_SIZE, shuffle=False)

    ecg_data = np.load("real_ecg.npy", allow_pickle=True)
    crf_data = np.load("real_crf.npy", allow_pickle=True)
    crf_data = crf_data.tolist()
    vasc_events = [val['Vascular event'] for val in crf_data]
    keys = [k for k in crf_data[0].keys() if k != 'Vascular event']
    non_vasc_features = np.array([[d[k] for k in keys] for d in crf_data])
    non_vasc_features_reordered = np.array(
        [reorder_features(row) for row in non_vasc_features])
    ecg_data = torch.tensor(ecg_data, dtype=torch.float32)
    crf_data = torch.tensor(non_vasc_features_reordered, dtype=torch.float32)
    labels = torch.tensor(vasc_events, dtype=torch.long)
    ecg_data = ecg_data.permute(0, 2, 1)
    real_test_dataset_full = TensorDataset(
        ecg_data, crf_data, labels)

    real_test_dataset = filter_to_3class(real_test_dataset_full)
    print(f"3-class real test size: {len(real_test_dataset)} (was {len(real_test_dataset_full)})")

    testloader_real = DataLoader(
        real_test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    main(num_epochs=num_epochs, trainloader=trainloader, validloader=validloader,
         device=device, num_classes=NUM_CLASSES, testloader_synth=testloader_synth, testloader_real=testloader_real)