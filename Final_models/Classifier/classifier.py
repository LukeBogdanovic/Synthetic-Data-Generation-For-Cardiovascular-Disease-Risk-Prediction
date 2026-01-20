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
from Final_models.WGAN.preprocessing_utils import reorder_features, evaluate_on_test


class ECG_Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad):
        super(ECG_Conv_Block, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, padding=pad)
        self.batch_norm = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout1d(0.2)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.max_pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, ecg):
        x = self.conv(ecg)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.max_pool(x)
        return x


class Classifier(nn.Module):

    def __init__(self, num_leads, num_risk_factors, num_classes):
        super(Classifier, self).__init__()
        chs = [num_leads, 32, 64, 128, 256]
        k_sizes = [5, 7, 9, 11]
        blocks = []
        for cin, cout, k_size in zip(chs[:-1], chs[1:], k_sizes):
            pad = (k_size-1) // 2
            blocks.append(ECG_Conv_Block(cin, cout, k_size, pad))
        self.conv_blocks = nn.Sequential(*blocks)
        ecg_feat_dim = 256
        risk_emb_dim = 64
        if num_risk_factors > 0:
            self.risk_mlp = nn.Sequential(
                nn.Linear(num_risk_factors, 128),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, risk_emb_dim),
                nn.LeakyReLU(0.3, inplace=True)
            )
        else:
            self.risk_mlp = None
            risk_emb_dim = 0

        self.head = nn.Sequential(
            nn.Linear(ecg_feat_dim + risk_emb_dim, 128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, ecg, risk_factors=None):
        x = self.conv_blocks(ecg)
        x = x.mean(dim=-1)
        if self.risk_mlp is not None:
            if risk_factors is None:
                raise ValueError("Risk factors must be provided when num_risk_factors > 0")
            r = self.risk_mlp(risk_factors)
            x = torch.cat([x, r], dim=1)
        logits = self.head(x)
        return logits


def make_metrics(num_classes : int):
    return MetricCollection({
        "acc_micro": MulticlassAccuracy(num_classes=num_classes, average='micro'),
        "acc_macro": MulticlassAccuracy(num_classes=num_classes, average='macro'),
        "acc_weighted": MulticlassAccuracy(num_classes=num_classes, average='weighted'),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average='macro'),
        "f1_weighted": MulticlassF1Score(num_classes=num_classes, average='weighted'),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average='macro'),
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average='macro'),
        "f1_per_class": MulticlassF1Score(num_classes=num_classes, average='none'),
        "recall_per_class": MulticlassRecall(num_classes=num_classes, average='none'),
        "precision_per_class": MulticlassPrecision(num_classes=num_classes, average='none'),
        "conf_mat": MulticlassConfusionMatrix(num_classes=num_classes)
    })

@torch.no_grad()
def run_eval_epoch(model : nn.Module, loader : DataLoader[TensorDataset], device : torch.device, metrics : MetricCollection):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n = 0

    for i, (ecg, risk, labels) in enumerate(loader):
        ecg : torch.Tensor
        risk : torch.Tensor
        labels: torch.Tensor
        ecg = ecg.to(device)
        risk = risk.to(device)
        labels = labels.to(device)
        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        logits = model(ecg, risk)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        metrics.update(preds, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
    
    out = metrics.compute()
    out['loss'] = torch.tensor(total_loss / max(n,1), device=device)
    return out


def run_train_epoch(model : nn.Module, loader : DataLoader, device : torch.device, optimizer : optim.Adam, cost_function: nn.CrossEntropyLoss, metrics : MetricCollection):
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
        risk = risk.to(device)
        labels = labels.to(device)
        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        optimizer.zero_grad()
        logits = model(ecg, risk)
        loss = cost_function(logits, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, dim=1)
        metrics.update(preds, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        correct += (preds == labels).sum().item()
        print(f"Step: {i+1:4d}/{len(loader)} | Loss: {total_loss/n:.4f} | Acc: {correct/n:.4f}")
    
    out = metrics.compute()
    out['loss'] = torch.tensor(total_loss/max(n,1), device=device)
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


def train(classifier : nn.Module, trainloader: DataLoader, validloader: DataLoader, device: torch.device, optimizer, cost_function, model_path: str, num_epochs: int, num_classes: int, testloader_synth, testloader_real):
    train_metrics = make_metrics(num_classes).to(device)
    val_metrics = make_metrics(num_classes=num_classes).to(device)

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, num_epochs+1):
        train_out = run_train_epoch(classifier, trainloader, device, optimizer, cost_function, train_metrics)
        val_out = run_eval_epoch(classifier, validloader, device, val_metrics)

        pretty_print_metrics(f"Epoch {epoch:03d} TRAIN", train_out, num_classes)
        pretty_print_metrics(f"Epoch {epoch:03d} VAL  ", val_out, num_classes)

        val_f1_macro = val_out["f1_macro"].item()
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            os.makedirs(model_path, exist_ok=True)
            torch.save(classifier.state_dict(), os.path.join(model_path, "best_model.pth"))
        
        history.append({
            "epoch": epoch,
            "train_loss": train_out["loss"].item(),
            "train_acc_micro": train_out["acc_micro"].item(),
            "train_acc_macro": train_out["acc_macro"].item(),
            "train_acc_weighted": train_out["acc_weighted"].item(),
            "train_f1_macro": train_out["f1_macro"].item(),
            "train_f1_weighted": train_out["f1_weighted"].item(),
            "val_loss": val_out["loss"].item(),
            "val_acc_micro": val_out["acc_micro"].item(),
            "val_acc_macro": val_out["acc_macro"].item(),
            "val_acc_weighted": val_out["acc_weighted"].item(),
            "val_f1_macro": val_out["f1_macro"].item(),
            "val_f1_weighted": val_out["f1_weighted"].item(),
        })

        print(f"Epoch: {epoch:03d} | Train Loss: {train_out['loss']:.4f} | Val Loss : {val_out['loss']:.4f} | Val F1(Macro) : {val_out['f1_macro']:.4f} | Val Acc(micro) : {val_out['acc_micro']:.4f} | Val Acc(weighted) : {val_out['acc_weighted']:.4f}")

    test_metrics = make_metrics(num_classes).to(device)
    synth_test_out = run_eval_epoch(classifier, testloader_synth, device, test_metrics)

    test_metrics2 = make_metrics(num_classes).to(device)
    real_test_out = run_eval_epoch(classifier, testloader_real, device, test_metrics2)

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


def main(trainloader, validloader, num_epochs, num_risk_factors, num_classes, device, testloader_synth, testloader_real):
    classifier = Classifier(num_leads=3,num_risk_factors=num_risk_factors, num_classes=4).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    cost_function = nn.CrossEntropyLoss()
    classifier_model_num = 0
    while os.path.exists(f"Final_models/Classifier/models/classifier{classifier_model_num}"):
        classifier_model_num += 1
    model_path = f"Final_models/Classifier/models/classifier{classifier_model_num}"
    os.makedirs(model_path)
    train(classifier,trainloader,validloader,device,optimizer,cost_function,model_path,num_epochs,num_classes, testloader_synth, testloader_real)

if __name__ == "__main__":
    dataset = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    BATCH_SIZE = 64
    df = pd.read_csv("augmented_dataset.csv")
    synth_data = torch.load(f"synth_datasets/{dataset}_real_synth_dataset.pth", weights_only=False)
    trainloader = DataLoader(synth_data['train'].dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(synth_data['valid'].dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader_synth = DataLoader(synth_data['test'].dataset, batch_size=BATCH_SIZE, shuffle=False)
    ecg_data = np.load("real_ecg.npy",allow_pickle=True)
    crf_data = np.load("real_crf.npy",allow_pickle=True)
    crf_data = crf_data.tolist()
    vasc_events = [val['Vascular event'] for val in crf_data]
    keys = [k for k in crf_data[0].keys() if k != 'Vascular event']
    non_vasc_features = np.array([[d[k] for k in keys] for d in crf_data])
    non_vasc_features_reordered = np.array([reorder_features(row) for row in non_vasc_features])
    ecg_data = torch.tensor(ecg_data, dtype=torch.float32)
    crf_data = torch.tensor(non_vasc_features_reordered, dtype=torch.float32)
    labels = torch.tensor(vasc_events, dtype=torch.long)
    ecg_data = ecg_data.permute(0, 2, 1)
    real_test_dataset = TensorDataset(ecg_data, crf_data, labels)  # Create new real dataset
    testloader_real = DataLoader(real_test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    main(num_epochs=num_epochs, trainloader=trainloader, validloader=validloader, num_risk_factors=7, device=device, num_classes=4, testloader_synth=testloader_synth, testloader_real=testloader_real)