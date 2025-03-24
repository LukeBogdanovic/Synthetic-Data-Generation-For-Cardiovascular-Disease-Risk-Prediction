'''
:File: ECG_classifier.py
:Author: Luke Bogdanovic
:Date: 12/03/2025
:Purpose: Trains the data fusion classifier 
'''
from torchmetrics import Recall, Precision, F1Score, Accuracy, ConfusionMatrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os
import numpy as np
import time
import pandas as pd
import pynvml
from preprocessing_utils import reorder_features, evaluate_on_test


class Classifier(nn.Module):
    '''
    Dual branch data fusion classifier that uses a CNN branch for feature extraction from ECGs.
    Uses DNN for feature extraction from CRFs. Concatenates features from branches into single
    feature vector used to classify the final output
    '''

    def __init__(self, ecg_length, num_risk_factors, num_classes):
        super(Classifier, self).__init__()
        # CNN branch
        self.ecg_conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64,
                      kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=7, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9,
                      padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1)
        )
        # Fully connected layer out of the CNN
        self.ecg_fc = nn.Linear(256, 128)
        # DNN branch
        self.risk_fc = nn.Sequential(
            nn.Linear(num_risk_factors, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Data fusion trunk
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, ecg, risk):
        ecg_features = self.ecg_conv(ecg)  # CNN branch
        # Average adaptive pooling for the layer in CNN
        ecg_features = torch.mean(ecg_features, dim=-1)
        ecg_features = self.ecg_fc(ecg_features)  # CNN output layer
        risk_features = self.risk_fc(risk)  # DNN branch
        # Concatenate ecg and CRF features
        fused_features = torch.cat((ecg_features, risk_features), dim=1)
        # Perform classification using data fusion trunk
        output = self.fusion_fc(fused_features)
        return output


def train_classifier(classifier, trainloader, validloader, device, optimizer, cost_function, model_path, num_epochs):
    '''
    Training loop for the data fusion classifier. Trains for the number of epochs
    specified using the optimizer and cost function provided. Accepts training data from the dataloader.
    Evaluates training using the validation set, testing set and the real data testing set.

    :param classifier: Classifier model
    :param trainloader: Training set
    :param validloader: Validation set
    :param device: Chosen device for training
    :param optimizer: Chosen optimizer for training
    :param cost_function: Chosen cost function for training
    :param model_path: Model path for saving
    :param num_epochs: Number of epochs for training
    '''
    num_classes = 4
    train_recall = Recall(
        task="multiclass", num_classes=num_classes, average="none").to(device)
    train_precision = Precision(
        task="multiclass", num_classes=num_classes, average="none").to(device)
    train_f1 = F1Score(task="multiclass",
                       num_classes=num_classes, average="none").to(device)
    train_accuracy = Accuracy(
        task="multiclass", num_classes=num_classes, average="macro").to(device)
    train_conf_mat = ConfusionMatrix(
        task="multiclass", num_classes=num_classes).to(device)
    val_recall = Recall(task="multiclass",
                        num_classes=num_classes, average="none").to(device)
    val_precision = Precision(
        task="multiclass", num_classes=num_classes, average="none").to(device)
    val_f1 = F1Score(task="multiclass", num_classes=num_classes,
                     average="none").to(device)
    val_accuracy = Accuracy(
        task="multiclass", num_classes=num_classes, average="macro").to(device)
    val_conf_mat = ConfusionMatrix(
        task="multiclass", num_classes=num_classes).to(device)
    pynvml.nvmlInit()  # Initialise the NVIDIA management library
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get GPU:0
    # Create metrics history storage dictionary
    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_recall': [],
        'train_precision': [],
        'train_f1': [],
        'train_conf_mat': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_recall': [],
        'val_precision': [],
        'val_f1': [],
        'val_conf_mat': [],
        'gpu_power_avg': []
    }
    for epoch in range(num_epochs):
        classifier.train()  # Set model to training mode
        # Take start time for epoch start to track time per epoch
        start_time_epoch = time.time()
        # Set accumulation values to 0
        running_loss = 0.0
        power_readings = []
        all_train_preds = []
        all_train_labels = []
        all_train_probs = []
        # Loop for steps per epoch and grab data from the dataloader
        for step, (ecg, risk, labels) in enumerate(trainloader):
            # Take start time for step start to track time per step
            start_time_step = time.time()
            ecg = ecg.to(device)  # Send ECG batch to GPU
            risk = risk.to(device)  # Send CRF batch to GPU
            labels = labels.to(device)  # Send labels batch to GPU
            optimizer.zero_grad()  # Set optimizer gradients to 0
            # Get logits from classifier training
            logits = classifier(ecg, risk)
            loss = cost_function(logits, labels)  # Get loss for the step
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters for optimizer
            # Add calculated values to accumulation variables
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)  # Calculate predictions
            probs = F.softmax(logits, dim=1)  # Calculate probabilities
            all_train_preds.append(preds)
            all_train_labels.append(labels)
            all_train_probs.append(probs)
            power_usage = pynvml.nvmlDeviceGetPowerUsage(
                handle) / 1000  # Gets power usage in watts
            power_readings.append(power_usage)
            end_time_step = time.time()  # Track end time for time per step
            print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {step+1}/{len(trainloader)} |"
                  f" Time: {end_time_step-start_time_step}")
        all_train_preds = torch.cat(all_train_preds)
        all_train_labels = torch.cat(all_train_labels)
        all_train_probs = torch.cat(all_train_probs)
        train_loss_epoch = running_loss / len(trainloader)
        train_acc_val = train_accuracy(all_train_preds, all_train_labels)
        train_rec_val = train_recall(all_train_preds, all_train_labels)
        train_prec_val = train_precision(all_train_preds, all_train_labels)
        train_f1_val = train_f1(all_train_preds, all_train_labels)
        train_conf_mat_val = train_conf_mat(all_train_preds, all_train_labels)
        train_recall.reset()
        train_precision.reset()
        train_f1.reset()
        train_accuracy.reset()
        train_conf_mat.reset()
        end_time_epoch = time.time()
        avg_gpu_power = sum(power_readings)/len(power_readings)
        print(
            f"Epoch time elapsed: {end_time_epoch-start_time_epoch}s | Avg GPU Power: {avg_gpu_power:.2f}W")
        classifier.eval()  # Set to inference mode for evaluation
        running_val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []
        # Same as training loop just set to not calculate gradients and from validation set instead of training set
        with torch.no_grad():
            for ecg, risk, labels in validloader:
                ecg = ecg.to(device)
                risk = risk.to(device)
                labels = labels.to(device)
                logits = classifier(ecg, risk)
                loss = cost_function(logits, labels)
                running_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                probs = F.softmax(logits, dim=1)
                all_val_preds.append(preds)
                all_val_labels.append(labels)
                all_val_probs.append(probs)
        all_val_preds = torch.cat(all_val_preds)
        all_val_labels = torch.cat(all_val_labels)
        all_val_probs = torch.cat(all_val_probs)
        val_loss_epoch = running_val_loss / len(validloader)
        val_acc_val = val_accuracy(all_val_preds, all_val_labels)
        val_rec_val = val_recall(all_val_preds, all_val_labels)
        val_prec_val = val_precision(all_val_preds, all_val_labels)
        val_f1_val = val_f1(all_val_preds, all_val_labels)
        val_conf_mat_val = val_conf_mat(all_val_preds, all_val_labels)
        val_recall.reset()
        val_precision.reset()
        val_f1.reset()
        val_accuracy.reset()
        val_conf_mat.reset()
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
        print(
            f"Train: Accuracy: {train_acc_val:.4f}, Recall: {train_rec_val}, Precision: {train_prec_val}, F1: {train_f1_val}")
        print(f"Train Confusion Matrix:\n{train_conf_mat_val}")
        print(
            f"Val:   Accuracy: {val_acc_val:.4f}, Recall: {val_rec_val}, Precision: {val_prec_val}, F1: {val_f1_val}")
        print(f"Val Confusion Matrix:\n{val_conf_mat_val}\n")
        # Add metrics to dictionary
        metrics_history['epoch'].append(epoch + 1)
        metrics_history['train_loss'].append(train_loss_epoch)
        metrics_history['train_accuracy'].append(train_acc_val.item())
        metrics_history['train_recall'].append(train_rec_val)
        metrics_history['train_precision'].append(train_prec_val)
        metrics_history['train_f1'].append(train_f1_val)
        metrics_history['train_conf_mat'].append(
            train_conf_mat_val.cpu().numpy())
        metrics_history['val_loss'].append(val_loss_epoch)
        metrics_history['val_accuracy'].append(val_acc_val.item())
        metrics_history['val_recall'].append(val_rec_val)
        metrics_history['val_precision'].append(val_prec_val)
        metrics_history['val_f1'].append(val_f1_val)
        metrics_history['val_conf_mat'].append(val_conf_mat_val.cpu().numpy())
        metrics_history['gpu_power_avg'].append(avg_gpu_power)
    test_metrics = evaluate_on_test(
        classifier, test_loader, device, num_classes=4)  # Calculate the test set metrics
    real_test_metrics = evaluate_on_test(
        classifier, testloader, device, 4)  # Calculate the real test metrics
    # Create dictionary for Pytorch to save
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics_history': metrics_history,
        'test_metrics': test_metrics,
        'real_test_metrics': real_test_metrics
    }
    # Save the model and metrics
    torch.save(
        checkpoint, f"{model_path}/{dataset}_real_synth_data_classifier.pth")
    pynvml.nvmlShutdown()  # Shutdown Nvidia management library


def main(trainloader, validloader, num_epochs, ecg_length, num_risk_factors, device):
    classifier = Classifier(ecg_length=ecg_length,
                            num_risk_factors=num_risk_factors,
                            num_classes=4).to(device)  # Create and send classifier to GPU
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    cost_function = nn.CrossEntropyLoss()
    classifier_model_num = 0
    # Create model storage path
    while os.path.exists(f"classifier_models/classifier{classifier_model_num}"):
        classifier_model_num += 1
    model_path = f"classifier_models/classifier{classifier_model_num}"
    os.makedirs(model_path)
    train_classifier(classifier=classifier, trainloader=trainloader, validloader=validloader,
                     optimizer=optimizer, cost_function=cost_function, model_path=model_path, num_epochs=num_epochs, device=device)  # Begin training


if __name__ == "__main__":
    dataset = 0.5  # Set dataset real data percentage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10  # Number of epochs
    BATCH_SIZE = 64
    df = pd.read_csv("augmented_dataset.csv")  # Read oversampled CRFs
    synth_data = torch.load(
        f"synth_datasets/{dataset}_real_synth_dataset.pth", weights_only=False)  # Load dataloaders from pytorch file
    train_loader = DataLoader(
        synth_data['train'].dataset, batch_size=BATCH_SIZE, shuffle=True)  # Set new batch size
    valid_loader = DataLoader(
        synth_data['valid'].dataset, batch_size=BATCH_SIZE, shuffle=True)  # Set new batch size
    test_loader = DataLoader(
        synth_data['test'].dataset, batch_size=BATCH_SIZE, shuffle=False)  # Set new batch size
    ecg_data = np.load("real_ecg2.npy", allow_pickle=True)  # Load real ECGs
    crf_data = np.load("real_crf2.npy", allow_pickle=True)  # Load real CRFs
    crf_data = crf_data.tolist()  # Send CRF data to list
    vascular_events = [val['Vascular event'] for val in crf_data]
    keys = [k for k in crf_data[0].keys() if k != 'Vascular event']
    # Create a 2D NumPy array where each row corresponds to the values (in the same order) of the non-vascular features
    non_vasc_features = np.array([[d[k] for k in keys] for d in crf_data])
    non_vasc_features_reordered = np.array(
        [reorder_features(row) for row in non_vasc_features])  # Reorder the vascular features to match generated features
    ecg_data = torch.tensor(ecg_data, dtype=torch.float32)  # Create ecg tensor
    crf_data = torch.tensor(non_vasc_features_reordered,
                            dtype=torch.float32)  # Create crf tensor
    # Create labels tensor
    labels = torch.tensor(vascular_events, dtype=torch.long)
    # Permute ecg data (batch, leads, timestep)
    ecg_data = ecg_data.permute(0, 2, 1)
    real_test_dataset = TensorDataset(
        ecg_data, crf_data, labels)  # Create new real dataset
    testloader = DataLoader(
        real_test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)  # Create real test loader
    main(num_epochs=num_epochs, trainloader=train_loader,
         validloader=valid_loader, num_risk_factors=7, device=device, ecg_length=128*5)
