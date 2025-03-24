'''
:File: datasetGen.py
:Author: Luke Bogdanovic
:Date: 12/3/2025
:Purpose: Script for generating new synthetic datasets.
'''
import pandas as pd
import numpy as np
import random
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import torch
from gan_scripts.CWGAN_torch import Generator
from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.model_selection import train_test_split


def generate_ecgs_for_crf(label_tensor, generator, noise_dim, batch_size, device):
    '''
    Generates ECGs from the provided CRF labels. Uses the provided generator to
    generate new 3-lead ECG signals from random latent vector. Generates signals
    in sizes of the batch size provided.

    :param label_tensor: Tensor of vascular condition labels
    :param generator: Generator model for creating ECGs
    :param noise_dim: Latent noise vector size
    :param batch_size: Set batch size
    :param device: Chosen device for training

    Returns:
        generated_signals: Tensor of all generated signals
    '''
    generator.to(device)  # Send generator model to the selected device
    generator.eval()  # Set generator to inference mode
    # Get the number of samples in the tensor
    num_samples = label_tensor.shape[0]
    generated_signals = []
    with torch.no_grad():  # Set the model to not calculate gradients
        for i in range(0, num_samples, batch_size):
            # Get current batch conditions
            batch_conditions = label_tensor[i:i + batch_size].to(device)
            # Get current batch size
            current_batch_size = batch_conditions.shape[0]
            # Create random noise vector
            noise = torch.randn(current_batch_size, noise_dim, device=device)
            batch_ecgs = generator(noise, batch_conditions)  # Generate signals
            # Append signals to end of list after sending data back to CPU
            generated_signals.append(batch_ecgs.cpu())
    # Concatenate all signals into a single tensor
    return torch.cat(generated_signals, dim=0)


real_data_fraction = 0.25  # Amount of real data to use
number_of_samples = 10000  # Amount of samples to have in final dataset
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # Device setup
BATCH_SIZE = 128  # Batch size to generate signals in
df = pd.read_csv("augmented_dataset.csv")  # Load original CRF dataset
# Load real data for ECG and CRF
real_data = np.load("fine_tune_data.npy", allow_pickle=True)
real_ecg_list = []
real_conditions_list = []
for item in real_data:  # Separate ECG and CRFs
    real_ecg_list.append(item[0])
    real_conditions_list.append(item[1])
# Map condition strings to numeric labels
condition_mapping = {
    'none': 0,
    'myocardial infarction': 1,
    'stroke': 2,
    'syncope': 3
}
# Create list for real ecg and CRFs using a tuple of (ecg, condition)
real_shuffle = list(zip(real_ecg_list, real_conditions_list))
random.shuffle(real_shuffle)  # Inplace random shuffle of the real data
# Grab the number of samples for the dataset from the shuffled dataset
samples = real_shuffle[:number_of_samples]
# Select the amount of real samples requested
first_half = samples[:int(len(samples)*real_data_fraction)]
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)  # Detect metadata from original dataset
TVAE_model = TVAESynthesizer(
    metadata=metadata, epochs=3000, cuda=True, verbose=True)  # Create the TVAE model
model = TVAE_model.load("TVAE_model.pkl")  # Load the trained TVAE model
if len(first_half) >= 1:  # Check if any real data is being used in the dataset
    real_ecg_list_split, real_crf_list_split = zip(*first_half)
    real_ecg_array = np.stack(real_ecg_list_split, axis=0)
    # Convert real ECGs to leads first (batch, leads, timestep)
    real_ecg_array = np.transpose(
        real_ecg_array, (0, 2, 1))  # shape: (batch, leads, timestep)
    # Convert to torch tensors
    synth_crf_df = model.sample(len(first_half))
    synth_crf_df['Vascular event'] = synth_crf_df['Vascular event'].str.lower().map(
        condition_mapping)
    # synthetic_labels_tensor = torch.tensor(labels)
    synthetic_groups = {}
    for code in condition_mapping.values():
        # Filter the synthetic CRF DataFrame for this vascular event
        synthetic_groups[code] = synth_crf_df[synth_crf_df['Vascular event'] == code]
    # For each real ECG sample, match a synthetic CRF with the same vascular condition.
    matched_synth_crf_list = []
    for label in real_crf_list_split:
        group = synthetic_groups.get(label)
        if group is not None and not group.empty:
            # Sample one synthetic record from the group at random
            sample = group.sample(n=1)
            matched_synth_crf_list.append(sample)
        else:
            num_features = synth_crf_df.shape[1]
            matched_synth_crf_list.append(pd.DataFrame(
                [np.zeros(num_features)], columns=synth_crf_df.columns))
    # Concatenate the matched synthetic CRF samples into one DataFrame
    matched_synth_crf_df = pd.concat(
        matched_synth_crf_list, ignore_index=True)
    matched_synth_crf_df = matched_synth_crf_df.drop(
        columns=['Vascular event'])
    synth_crf_array = matched_synth_crf_df.to_numpy()
    synth_crf_tensor = torch.tensor(synth_crf_array, dtype=torch.float32)
    # Convert real ECG data and real labels to torch tensors.
    real_ecg_tensor = torch.tensor(real_ecg_array, dtype=torch.float32)
    real_labels_tensor = torch.tensor(
        real_crf_list_split, dtype=torch.long)
if 'real_ecg_tensor' in locals():  # Check if real_ecg_tensor exists in scope
    # Get the remaining number of CRF samples for the dataset using the TVAE
    tvae_crf_samples = model.sample(number_of_samples-len(real_ecg_tensor))
else:
    # Get the number of samples for the dataset (no real data used)
    tvae_crf_samples = model.sample(number_of_samples)
if len(tvae_crf_samples) >= 1:  # Check if the TVAE has been used to generate samples
    generator = Generator(ecg_length=640, n_leads=3,
                          latent_dim=50)  # Instantiate generator
    CGAN_model = torch.load(
        "gan_scripts/gan/CWGAN_models/pretrained/CWGAN.pth", map_location=device, weights_only=False)  # Using best performing model
    # Load generator weights
    generator.load_state_dict(CGAN_model['gen_state_dict'])
    # Grab all values from class column
    labels = tvae_crf_samples['Vascular event']
    # Use mapping to convert to encoded format
    labels = labels.str.lower().map(condition_mapping)
    # Drop the class column from the CRF data
    all_crf_features = tvae_crf_samples.drop(columns=['Vascular event'])
    all_crf_features = all_crf_features.to_numpy()  # Convert the CRFs to numpy array
    synthetic_labels_tensor = torch.tensor(
        labels, dtype=torch.long)  # Convert classes to torch tensor
    synthetic_crf_tensor = torch.tensor(
        all_crf_features, dtype=torch.float32)  # Convert synthetic CRFs to torch tensor
    train_ecg = generate_ecgs_for_crf(
        synthetic_labels_tensor.unsqueeze(1), generator, 50, BATCH_SIZE, device)  # Generate ECGs for the classes
    # Change shape to be (batch, leads, timestep)
    synthetic_ecg_tensor = train_ecg.permute(0, 2, 1)
    if len(first_half) >= 1:  # Check if real data is being used
        combined_ecg_tensor = torch.cat(
            [real_ecg_tensor, synthetic_ecg_tensor], dim=0)  # Concatenate real ecg tensor and generated ecg tensor
        combined_synth_tensor = torch.cat(
            [synth_crf_tensor, synthetic_crf_tensor], dim=0)  # Concatenate oversampled CRFs tensor and generated CRFs tensor
        combined_labels_tensor = torch.cat(
            [real_labels_tensor, synthetic_labels_tensor], dim=0)  # Concatenate real labels tensor and generated labels tensor
        combined_dataset = TensorDataset(
            combined_ecg_tensor, combined_synth_tensor, combined_labels_tensor)  # Create tensor dataset with ECG, CRF, Label
    else:
        combined_dataset = TensorDataset(
            synthetic_ecg_tensor, synthetic_crf_tensor, synthetic_labels_tensor)  # Create tensor dataset with ECG, CRF, Label
    # Get all indices of the dataset
    indices = np.arange(len(combined_dataset))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=42)  # Split the dataset 80:20
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42)  # Split the 20% of dataset into valid and test set 50:50
    train_dataset = Subset(combined_dataset, train_idx)  # Get training subset
    val_dataset = Subset(combined_dataset, val_idx)  # Get validation subset
    test_dataset = Subset(combined_dataset, test_idx)  # Get testing subset
else:
    combined_dataset = TensorDataset(
        real_ecg_tensor, synth_crf_tensor, real_labels_tensor)  # Create combined real dataset
    # Get all indices of the dataset
    indices = np.arange(len(combined_dataset))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=42)  # Split the dataset 80:20
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42)  # Split the 20% of dataset into valid and test set 50:50
    train_dataset = Subset(combined_dataset, train_idx)  # Get training subset
    val_dataset = Subset(combined_dataset, val_idx)  # Get validation subset
    test_dataset = Subset(combined_dataset, test_idx)  # Get testing subset
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Create training dataloader
valid_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Create validation dataloader
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Create testing dataloader
# Add all dataloaders to a dictionary for saving
datasets = {
    "train": train_loader,
    "valid": valid_loader,
    "test": test_loader
}
torch.save(
    datasets, f"synth_datasets/{real_data_fraction}_real_synth_dataset.pth")  # Save the datasets to disk in Pytorch format
