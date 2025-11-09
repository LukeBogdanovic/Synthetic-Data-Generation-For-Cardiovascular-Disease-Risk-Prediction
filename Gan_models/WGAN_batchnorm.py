import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tslearn.metrics import SoftDTWLossPyTorch
from ignite.metrics import MaximumMeanDiscrepancy
from torch.utils.data import DataLoader, TensorDataset
from preprocessing_utils import per_lead_minmax_scaling, save_generated_ecg


N_LEADS = 3
LATENT_DIM = 50
NUM_SECONDS = 5
NUM_SAMPLES = 128
ECG_LENGTH = NUM_SAMPLES * NUM_SECONDS
BATCH_SIZE = 64


class Generator(nn.Module):
    def __init__(self, ecg_length=ECG_LENGTH, n_leads=N_LEADS, latent_dim=LATENT_DIM):
        super(Generator, self).__init__()
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, ecg_length * 8)
        self.cnn1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=16,
                               kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(in_channels=16, out_channels=32,
                               kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=40,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=80, hidden_size=40,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(80)
        self.to_leads = nn.Linear(80, n_leads)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(-1, 8, self.ecg_length)
        x = self.cnn1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm2(x)
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        x = self.to_leads(x)
        x = self.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, ecg_length=ECG_LENGTH, n_leads=N_LEADS):
        super(Critic, self).__init__()
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        self.conv1 = nn.Conv1d(n_leads, 64, kernel_size=5, stride=2, padding=4)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=7, stride=2, padding=3)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=9, stride=2, padding=2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(16, 1)

    def forward(self, ecg):
        x = ecg.transpose(1, 2)
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.mean(dim=2)
        x = self.fc(x)
        return x


def flatten_ecg(x):
    return x.reshape(x.size(0), -1)


def train(generator, critic, dataloader, num_epochs, latent_dim, n_critic, g_optimizer, c_optimizer, device, image_path, model_path, lead_mins, lead_maxs):
    generator.train()
    critic.train()
    metrics_history = {
        'epoch': [],
        'gen_loss': [],
        'critic_loss': [],
        'mvdtw': [],
        'mmd': [],
    }
    softdtw = SoftDTWLossPyTorch(gamma=1, normalize=True)
    mmd = MaximumMeanDiscrepancy(var=1.0)
    subset = min(16, BATCH_SIZE)
    clip = 0.01
    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        running_g_loss = 0.0
        running_c_loss = 0.0
        running_mvdtw = 0.0
        mmd.reset()
        for i, (real_ecg,) in enumerate(dataloader):
            start_time_step = time.time()
            real_ecg = real_ecg.to(device)
            batch_size = real_ecg.size(0)
            for _ in range(n_critic):
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake_ecg = generator(noise)
                c_optimizer.zero_grad()
                critic_real = critic(real_ecg)
                critic_fake = critic(fake_ecg.detach())
                loss_critic = critic_fake.mean() - critic_real.mean()
                loss_critic.backward()
                c_optimizer.step()
                for p in critic.parameters():
                    p.data.clamp_(-clip, clip)
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_ecg = generator(noise)
            g_optimizer.zero_grad()
            critic_fake = critic(fake_ecg)
            loss_generator = -critic_fake.mean()
            loss_generator.backward()
            g_optimizer.step()
            idx = torch.randperm(batch_size, device=device)[:subset]
            mvdTW_value = softdtw(fake_ecg[idx], real_ecg[idx]).mean()
            with torch.no_grad():
                fake_flat = flatten_ecg(fake_ecg[idx])
                real_flat = flatten_ecg(real_ecg[idx])
                fake_flat = fake_flat.to(device, dtype=torch.float32)
                real_flat = real_flat.to(device, dtype=torch.float32)
                mmd.update((fake_flat, real_flat))
            running_c_loss += loss_critic.item()
            running_g_loss += loss_generator.item()
            running_mvdtw += mvdTW_value.item()
            end_time_step = time.time()
            print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {i+1}/{len(dataloader)} |"
                  f" Critic Loss: {loss_critic.item():.4f} | Generator Loss: {loss_generator.item():.4f} |"
                  f" MMD: {mmd.compute() :.4f} |mvdTW: {mvdTW_value:.4f} | Time: {end_time_step-start_time_step}")
        end_time_epoch = time.time()
        mmd_epoch = mmd.compute()
        print(
            f"Epoch time elapsed: {end_time_epoch-start_time_epoch}s | MMD: {mmd_epoch:.4f}")
        save_generated_ecg(generator, epoch,
                           device, latent_dim=latent_dim, save_path=image_path, lead_maxs=lead_maxs, lead_mins=lead_mins, num_classes=0)
        gen_loss_epoch = running_g_loss / len(dataloader)
        critic_loss_epoch = running_c_loss / len(dataloader)
        mvdTW_epoch = running_mvdtw / len(dataloader)
        metrics_history['epoch'].append(epoch+1)
        metrics_history['gen_loss'].append(gen_loss_epoch)
        metrics_history['critic_loss'].append(critic_loss_epoch)
        metrics_history['mvdtw'].append(mvdTW_epoch)
        metrics_history['mmd'].append(mmd_epoch)
    checkpoint = {
        'epoch': num_epochs,
        'gen_state_dict': generator.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'gen_optimizer_state_dict': g_optimizer.state_dict(),
        'critic_optimizer_state_dict': c_optimizer.state_dict(),
        'metrics_history': metrics_history
    }
    # Save the model and metrics
    torch.save(checkpoint, f"{model_path}/CWGAN.pth")


def main():
    if os.path.exists("../biased_ptbxl_ecgs.npy"):
        data = np.load("../biased_ptbxl_ecgs.npy", allow_pickle=True)
        normalized_data, LEAD_MINS, LEAD_MAXS = per_lead_minmax_scaling(data)
        dataset = torch.tensor(np.array(normalized_data), dtype=torch.float32)
        dataloader = DataLoader(TensorDataset(
            dataset), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    n_critic = 5
    GAN_MODEL_NUM = 0
    while os.path.exists(f"images/WGANGP_images/generated_images_wganGP{GAN_MODEL_NUM}"):
        GAN_MODEL_NUM += 1
    image_path = f"images/WGANGP_images/generated_images_wganGP{GAN_MODEL_NUM}"
    os.makedirs(image_path)
    GAN_MODEL_NUM = 0
    while os.path.exists(f"gan_scripts/gan/WGANGP_models/wganGP_{GAN_MODEL_NUM}"):
        GAN_MODEL_NUM += 1
    model_path = f"gan_scripts/gan/WGANGP_models/wganGP_{GAN_MODEL_NUM}"
    generator = Generator(ecg_length=ECG_LENGTH, n_leads=N_LEADS,
                          latent_dim=LATENT_DIM).to(device)
    critic = Critic(ecg_length=ECG_LENGTH, n_leads=N_LEADS).to(
        device)
    os.makedirs(model_path)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=[0.0, 0.9])
    c_optimizer = optim.Adam(critic.parameters(), lr=1e-4, betas=[0.0, 0.9])
    train(generator, critic, dataloader, num_epochs, LATENT_DIM, n_critic,
          g_optimizer, c_optimizer, device, image_path, model_path, LEAD_MINS, LEAD_MAXS)


if __name__ == "__main__":
    main()
