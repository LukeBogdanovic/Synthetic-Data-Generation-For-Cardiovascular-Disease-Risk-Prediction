'''
:File: WGAN_torch_gen.py
:Author: Luke Bogdanovic
:Date: 12/3/2025
:Purpose: Generates an ECG signal using the WGAN generator model
'''
import torch
from preprocessing_utils import per_lead_minmax_scaling, plot_generated_sample
import os
import numpy as np
from WGAN_torch import Generator

latent_dim = 50  # Noise vector size
ecg_length = 128 * 5  # Length of the ECG
n_leads = 3  # Number of leads to generate

if os.path.exists("normalized_ecg_phys.npy"):  # Check for the saved numpy file
    # Load the saved numpy file
    data = np.load("normalized_ecg_phys.npy", allow_pickle=True)
    segments = [item for item in data]  # Collect segments from the saved data
    ecg_dataset = np.stack(segments)  # Stack all segments into a single array
    normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(
        ecg_dataset=ecg_dataset)  # Normalize values in each lead between (-1,1)

generator = Generator(ecg_length=ecg_length,
                      n_leads=n_leads, latent_dim=latent_dim)  # Instantiate generator model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wgan = torch.load(
    "gan_scripts/gan/WGAN_models/pretrain/CWGAN.pth", map_location=device, weights_only=False)  # Load the GAN model and metrics
# Load the generator weight and biases
generator.load_state_dict(wgan['gen_state_dict'])
generator.to(device)  # Send model to device
generator.eval()  # Set model to evaluation/inference mode

with torch.no_grad():  # Set to not calculate gradients
    # Create random noise vector as tensor
    noise = torch.randn(1, latent_dim, device=device)
    generated_signal = generator(noise)  # Generate the signal

plot_generated_sample(generated_signal=generated_signal,
                      lead_maxs=lead_maxs, lead_mins=lead_mins)  # Plot the signals per lead
