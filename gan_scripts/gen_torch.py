import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gan_pretrain_preprocessing import bandpass_filter, reverse_ecg_normalization, normalize_ecg
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the same Generator class and parameters as used during training.
# For instance:
latent_dim = 50
ecg_length = 128 * 5  # adjust based on your training configuration
n_leads = 3

if os.path.exists("normalized_ecg_phys.npy"):
    data = np.load("normalized_ecg_phys.npy", allow_pickle=True)
    n_records = len(data)
    subset_size = 10000
    indices = np.random.choice(n_records, subset_size, replace=False)
    subset_ecg_dataset = [data[i] for i in indices]
    m_scaler = MinMaxScaler(feature_range=(-1, 1)
                            ).fit(np.vstack(subset_ecg_dataset))
    normalized_data = [normalize_ecg(ecg, m_scaler)
                       for ecg in subset_ecg_dataset]
else:
    from gan_pretrain_preprocessing import normalized_data
    # Assume m_scaler is defined in that module


class Generator(nn.Module):
    def __init__(self, ecg_length=640, n_leads=3, latent_dim=50):
        super(Generator, self).__init__()
        self.ecg_length = ecg_length
        self.latent_dim = latent_dim
        self.n_leads = n_leads
        # Fully-connected layer to expand noise
        self.fc = nn.Linear(latent_dim, ecg_length * 32)
        # 1D convolution (padding=2 gives “same” padding for kernel_size=5)
        self.conv = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=100,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=200, hidden_size=100,
                             num_layers=1, batch_first=True, bidirectional=True)
        # TimeDistributed dense layer implemented as a linear layer applied to each timestep
        self.fc_time = nn.Linear(200, n_leads)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, noise):
        x = self.fc(noise)                    # (batch, ecg_length*32)
        x = x.view(-1, self.ecg_length, 32)     # (batch, ecg_length, 32)
        x = x.transpose(1, 2)                 # (batch, 32, ecg_length)
        x = self.relu(self.conv(x))           # (batch, 64, ecg_length)
        x = x.transpose(1, 2)                 # (batch, ecg_length, 64)
        x, _ = self.lstm1(x)                  # (batch, ecg_length, 128)
        x, _ = self.lstm2(x)                  # (batch, ecg_length, 64)
        x = self.fc_time(x)                   # (batch, ecg_length, n_leads)
        x = self.tanh(x)
        return x


generator = Generator(ecg_length=ecg_length,
                      n_leads=n_leads, latent_dim=latent_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.load_state_dict(torch.load(
    "gan_scripts/gan/WGAN_models/wgan_9/generator.pth", map_location=device))
generator.to(device)
generator.eval()

with torch.no_grad():
    noise = torch.randn(1, latent_dim, device=device)
    generated_signal = generator(noise)

generated_sample = generated_signal.squeeze(
    0).cpu().numpy()
generated_sample = reverse_ecg_normalization(generated_sample, m_scaler)

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
