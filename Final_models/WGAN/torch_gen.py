import torch
from preprocessing_utils import per_lead_minmax_scaling, plot_generated_sample
import os
import numpy as np
from UpsampleAndConv1D import Generator

latent_dim = 100  # Noise vector size
ecg_length = 128 * 5  # Length of the ECG
n_leads = 3  # Number of leads to generate

if os.path.exists("../../biased_ptbxl_ecgs.npy"):  # Check for the saved numpy file
    # Load the saved numpy file
    data = np.load("../../biased_ptbxl_ecgs.npy", allow_pickle=True)
    segments = [item for item in data]  # Collect segments from the saved data
    ecg_dataset = np.stack(segments)  # Stack all segments into a single array
    normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(
        # Normalize values in each lead between (-1,1)
        ecg_dataset=ecg_dataset)

generator = Generator(ecg_length=ecg_length,
                      n_leads=n_leads, latent_dim=latent_dim)  # Instantiate generator model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wgan = torch.load(
    # Load the GAN model and metrics
    "models/UpsampleAndCNN_WGAN/Model_0_GP_10.0_DTW_1.0/Model.pth", map_location=device, weights_only=False)
# Load the generator weight and biases
generator.load_state_dict(wgan['gen_state_dict'])
generator.to(device)  # Send model to device
generator.eval()  # Set model to evaluation/inference mode

with torch.no_grad():  # Set to not calculate gradients
    # Create random noise vector as tensor
    noise = torch.randn(1, latent_dim, device=device)
    generated_signal = generator(noise)  # Generate the signal

plot_generated_sample(generated_signal=generated_signal,
                      # Plot the signals per lead
                      lead_maxs=lead_maxs, lead_mins=lead_mins, amount_signal=256)
