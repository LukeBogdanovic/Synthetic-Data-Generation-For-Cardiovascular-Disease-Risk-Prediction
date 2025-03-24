'''
:File: ECG_torch_classify.py
:Author: Luke Bogdanovic
:Date: 12/03/2025
:Purpose: Uses the trained classifier specified to predict the classes of a given
    set of data.
'''
import torch
from ECG_classifier import Classifier
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from preprocessing_utils import reorder_features, evaluate_on_test

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 4         # Number of classes
ecg_length = 128 * 5      # For example, if your ECG length was 640
num_risk_factors = 7     # Replace with the actual number of CRF features

# Create an instance of the classifier and load the checkpoint
classifier = Classifier(ecg_length=ecg_length,
                        num_classes=num_classes, num_risk_factors=7)
classifier.to(device)

# Load the saved checkpoint
checkpoint = torch.load(
    "classifier_models/classifier72/classifier.pth", map_location=device, weights_only=False)  # Load model and metrics
# Load model weights and biases
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()  # Set the model to evaluation mode

# Load the real ECGs from NumPy file
ecg_data = np.load("real_ecg.npy", allow_pickle=True)
# Load the real CRFs from NumPy file
crf_data = np.load("real_crf.npy", allow_pickle=True)
crf_data = crf_data.tolist()  # Create list from CRF data
vascular_events = [val['Vascular event']
                   for val in crf_data]  # Get all vascular event labels

crf_data = [
    [value for key, value in d.items() if key.lower() != "vascular event"]
    for d in crf_data
]  # Get all CRF data from the CRFs list that isn't a vascular event
non_vasc_features_reordered = np.array(
    [reorder_features(row) for row in crf_data])  # Reorder the features so they match the generated CRFs format

# Create tensor for ECGs
ecg_data = torch.tensor(ecg_data, dtype=torch.float32)
crf_data = torch.tensor(non_vasc_features_reordered,
                        dtype=torch.float32)  # Create tensor for CRFs
# Create tensor for labels
labels = torch.tensor(vascular_events, dtype=torch.long)
# Change shape of tensor to (batch_size, leads, time)
ecg_data = ecg_data.permute(0, 2, 1)
# Create tensor dataset of the ECGs, CRFs and labels
test_dataset = TensorDataset(ecg_data, crf_data, labels)
# Create dataloader for the tensordataset
testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Evaluate the classifier using the test set
evaluate_on_test(classifier, testloader, device)
