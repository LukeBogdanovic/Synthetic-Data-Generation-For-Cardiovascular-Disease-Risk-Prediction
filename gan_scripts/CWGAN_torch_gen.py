import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocessing import bandpass_filter, per_lead_minmax_scaling, per_lead_inverse_scaling
import os
import numpy as np


# Define the same Generator class and parameters as used during training.
# For instance:
latent_dim = 50
ecg_length = 128 * 5  # adjust based on your training configuration
n_leads = 3

if os.path.exists("fine_tune_data.npy"):  # Check for the saved numpy file
    data = np.load("fine_tune_data.npy", allow_pickle=True)
    segments = [item[0] for item in data]
    ecg_dataset = np.stack(segments)
    normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(
        ecg_dataset=ecg_dataset)
    labels = [item[1] for item in data]
    labels = np.array(labels)


class Generator(nn.Module):
    '''
    Generator model for the WGAN-GP-DTW model. \\
    Consists of a twin stack of bidirectional LSTMs with 75 hidden units each \\
    and 5 
    '''

    def __init__(self, ecg_length=640, n_leads=3, latent_dim=50, condition_dim=1):
        super(Generator, self).__init__()
        self.ecg_length = ecg_length
        self.latent_dim = latent_dim
        self.n_leads = n_leads
        self.condition_dim = condition_dim
        self.fc = nn.Linear(latent_dim + condition_dim, ecg_length * 32)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=7, padding=3),
            nn.ReLU()
        )
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(256)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=192,
                      kernel_size=25, padding=12, stride=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(192, affine=True),
            nn.Conv1d(in_channels=192, out_channels=128,
                      kernel_size=17, padding=(17-1)//2, stride=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(128, affine=True),
            nn.Conv1d(in_channels=128, out_channels=64,
                      kernel_size=17, padding=(17-1)//2, stride=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(64, affine=True),
            nn.Conv1d(in_channels=64, out_channels=32,
                      kernel_size=17, padding=(17-1)//2, stride=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(32, affine=True),
            nn.Conv1d(in_channels=32, out_channels=n_leads,
                      kernel_size=17, padding=(17-1)//2, stride=1),
            nn.Tanh()
        )
        self.condition_embedding = nn.Embedding(4, self.condition_dim)

    def forward(self, noise, condition):
        cond_emb = self.condition_embedding(condition.squeeze(1))
        combined = torch.cat((noise, cond_emb), dim=1)
        x = self.fc(combined)
        x = x.view(-1, 32, self.ecg_length)
        x = self.cnn1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.layer_norm(x)
        x, _ = self.lstm2(x)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)  # Convert back to (batch, channels, time)
        x = self.cnn2(x)
        x = x.permute(0, 2, 1)  # Convert back to (batch, time, leads)
        return x


generator = Generator(ecg_length=ecg_length,
                      n_leads=n_leads, latent_dim=latent_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwgan = torch.load(
    "gan_scripts/gan/CWGAN_models/cwgan_3/CWGAN.pth", map_location=device, weights_only=False)
generator.load_state_dict(cwgan['generator_state_dict'])
generator.to(device)
generator.eval()

with torch.no_grad():
    noise = torch.randn(1, latent_dim, device=device)
    # Create a condition tensor with shape (1, 1)
    condition = torch.tensor(
        [3], device=device, dtype=torch.long).unsqueeze(1)
    generated_signal = generator(noise, condition)

generated_sample = generated_signal.cpu().numpy()
generated_sample = per_lead_inverse_scaling(
    generated_sample, lead_mins, lead_maxs)
generated_sample = generated_sample.squeeze(0)
generated_sample = bandpass_filter(generated_sample, 0.5, 40, 128)

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
lead_labels = ['Lead III', 'V3', 'V5']

for i, ax in enumerate(axs):
    ax.plot(generated_sample[:, i])
    ax.set_title(lead_labels[i])
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

plt.tight_layout()
plt.show()
