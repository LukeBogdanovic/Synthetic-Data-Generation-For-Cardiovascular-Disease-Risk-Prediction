'''
:File: CWGAN_torch_gen.py
:Author: Luke Bogdanovic
:Date: 12/3/2025
:Purpose: Generates an ECG signal using the CWGAN generator model
'''
import torch
from preprocessing_utils import per_lead_minmax_scaling, plot_generated_sample
import os
import numpy as np
from CWGAN_torch import Generator

latent_dim = 50  # Noise vector size
ecg_length = 128 * 5  # Length of the ECG
n_leads = 3  # Number of leads to generate

if os.path.exists("fine_tune_data.npy"):  # Check for the saved numpy file
    # Load the saved numpy file
    data = np.load("fine_tune_data.npy", allow_pickle=True)
    # Collect segments from the saved data
    segments = [item[0] for item in data]
    ecg_dataset = np.stack(segments)  # Stack all segments into a single array
    normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(
        ecg_dataset=ecg_dataset)  # Normalize values in each lead between (-1,1)
    # Collect all labels from the saved data
    labels = [item[1] for item in data]
    labels = np.array(labels)  # Store all labels in a numpy array


generator = Generator(ecg_length=ecg_length,
                      n_leads=n_leads, latent_dim=latent_dim)  # Instantiate the generator model
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # Select the device to run model on
cwgan = torch.load(
    "gan/CWGAN_models/pretrained/CWGAN.pth", map_location=device, weights_only=False)  # Load the pytorch model file
# Load the weights for the model
generator.load_state_dict(cwgan['generator_state_dict'])
generator.to(device)  # Send generator to chosen device
generator.eval()  # Set model to inference mode

with torch.no_grad():  # Set to not calculate gradients
    # Create random noise vector as tensor
    noise = torch.randn(1, latent_dim, device=device)
    # Create a condition tensor with shape (1, 1)
    condition = torch.tensor(
        [3], device=device, dtype=torch.long).unsqueeze(1)
    generated_signal = generator(noise, condition)

plot_generated_sample(
    generated_signal=generated_signal, lead_maxs=lead_maxs, lead_mins=lead_mins)  # Plot the generated signal from the CWGAN
