import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.signal import resample_poly
from tslearn.metrics import SoftDTWLossPyTorch
from ignite.metrics import MaximumMeanDiscrepancy
from torch.utils.data import DataLoader, TensorDataset
from preprocessing_utils import per_lead_minmax_scaling, save_generated_ecg, gradient_penalty

LATENT_DIM = 100
BATCH_SIZE = 128
LAMBDA_GP = 10
LAMBDA_DTW = 1e-3
LR_G = 2e-4
LR_C = 1e-4
N_CRITIC = 5
NUM_EPOCHS = 50
N_LEADS = 3
NUM_SECONDS = 5
NUM_SAMPLES = 100
ECG_LENGTH = NUM_SAMPLES * NUM_SECONDS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Up(nn.Module):
    def __init__(self, cin, cout, scale, k=9, p=4, act_slope=0.2):
        super().__init__()
        self.ups = nn.Upsample(scale_factor=scale, mode='nearest')
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, padding=p)
        self.res1x1 = nn.Conv1d(cin, cout, kernel_size=1)
        self.act = nn.LeakyReLU(act_slope, inplace=True)

    def forward(self, x):
        x_up = self.ups(x)
        y = self.conv(x_up)
        y = self.act(y)
        r = self.res1x1(x_up)
        return y + r


class Generator(nn.Module):
    def __init__(self,):
        super().__init__()

# class Critic(nn.Module):
#     def __init__(self, ecg_length=ECG_LENGTH, n_leads=N_LEADS):
#         super(Critic, self).__init__()
#         self.ecg_length = ecg_length
#         self.n_leads = n_leads
#         self.conv1 = nn.Conv1d(n_leads, 96, kernel_size=3, stride=2, padding=5)
#         self.conv2 = nn.Conv1d(96, 64, kernel_size=5, stride=2, padding=4)
#         self.conv3 = nn.Conv1d(64, 32, kernel_size=7, stride=2, padding=3)
#         self.conv4 = nn.Conv1d(32, 16, kernel_size=9, stride=2, padding=2)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.flatten_dim = self._get_flatten_dim()
#         self.mb_disc = MiniBatchDiscrimination(self.flatten_dim, 100, 5)
#         self.fc = nn.Linear(self.flatten_dim + 100, 1)

#     def _get_flatten_dim(self):
#         with torch.no_grad():
#             dummy = torch.zeros(1, self.n_leads, self.ecg_length)
#             x = self.leaky_relu(self.conv1(dummy))
#             x = self.leaky_relu(self.conv2(x))
#             x = self.leaky_relu(self.conv3(x))
#             x = self.leaky_relu(self.conv4(x))
#             flat_dim = x.view(1, -1).size(1)
#         return flat_dim

#     def forward(self, ecg):
#         x = ecg.transpose(1, 2)
#         x = self.leaky_relu(self.conv1(x))
#         x = self.leaky_relu(self.conv2(x))
#         x = self.leaky_relu(self.conv3(x))
#         x = self.leaky_relu(self.conv4(x))
#         x = x.view(x.size(0), -1)
#         x = self.mb_disc(x)
#         x = self.fc(x)
#         return x


def train(generator, critic, dataloader, num_epochs, latent_dim, n_critic, g_optimizer, c_optimizer, device, image_path, model_path, lead_mins, lead_maxs, lambda_gp, lambda_dtw):
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
    subset = min(32, BATCH_SIZE)
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
                gp = gradient_penalty(critic, real_ecg, fake_ecg, device)
                loss_critic = critic_fake.mean() - critic_real.mean() + lambda_gp * gp
                loss_critic.backward()
                c_optimizer.step()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_ecg = generator(noise)
            g_optimizer.zero_grad()
            critic_fake = critic(fake_ecg)
            with torch.no_grad():
                idx = torch.randperm(real_ecg.size(0), device=device)[
                    :min(32, batch_size)]
                real_sub = downsample(real_ecg[idx], factor=4)
            fake_sub = downsample(fake_ecg[idx], factor=4)
            dtw_loss = softdtw(fake_sub, real_sub.detach()).mean()
            loss_generator = -critic_fake.mean() + lambda_dtw*dtw_loss
            loss_generator.backward()
            g_optimizer.step()
            with torch.no_grad():
                idx = torch.randperm(batch_size, device=device)[
                    :min(32, batch_size)]
                real_sub = downsample(real_ecg[idx], factor=4)
                fake_sub = downsample(fake_ecg[idx], factor=4)
                mvdTW_value = softdtw(fake_sub, real_sub.detach()).mean()
                fake_flat = flatten_ecg(fake_ecg[idx])
                real_flat = flatten_ecg(real_ecg[idx])
                fake_flat = fake_flat.to(device, dtype=torch.float32)
                real_flat = real_flat.to(device, dtype=torch.float32)
                mmd.update((fake_flat, real_flat))
                w_gap = (critic_real.mean() - critic_fake.mean()).item()
            running_c_loss += loss_critic.item()
            running_g_loss += loss_generator.item()
            running_mvdtw += mvdTW_value.item()
            end_time_step = time.time()
            print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {i+1}/{len(dataloader)} |"
                  f" Critic Loss: {loss_critic.item():.4f} | Generator Loss: {loss_generator.item():.4f} |"
                  f" MMD: {mmd.compute() :.4f} | mvdTW: {mvdTW_value:.4f} | Time: {end_time_step-start_time_step} | W-gap: {w_gap:.4f} | GP: {gp:.4f}")
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


def downsample_input(input):
    downsampled = resample_poly(input, up=25, down=32, axis=0)
    target_len = 500
    cur_len = downsampled.shape[0]
    if cur_len > target_len:
        downsampled = downsampled[:, :target_len]
    elif cur_len < target_len:
        pad = target_len - cur_len
        downsampled = np.pad(downsampled, ((0, 0), (0, pad)), mode="edge")
    return downsampled


def batch_downsample(X):
    return np.stack([downsample_input(x) for x in X], axis=0)


def main():
    if os.path.exists("../biased_ptbxl_ecgs_100Hz.npy"):
        data = np.load("../biased_ptbxl_ecgs_100Hz.npy", allow_pickle=True)
        normalized_data, LEAD_MINS, LEAD_MAXS = per_lead_minmax_scaling(data)
        dataset = torch.tensor(np.array(normalized_data), dtype=torch.float32)
        dataloader = DataLoader(TensorDataset(
            dataset), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    GAN_MODEL_NUM = 0
    while os.path.exists(f"images/WGANGP_images/generated_images_wganGP{GAN_MODEL_NUM}"):
        GAN_MODEL_NUM += 1
    image_path = f"images/WGANGP_images/generated_images_wganGP{GAN_MODEL_NUM}"
    os.makedirs(image_path)
    GAN_MODEL_NUM = 0
    while os.path.exists(f"gan_scripts/gan/WGANGP_models/wganGP_{GAN_MODEL_NUM}"):
        GAN_MODEL_NUM += 1
    model_path = f"gan_scripts/gan/WGANGP_models/wganGP_{GAN_MODEL_NUM}"
    generator = Generator().to(DEVICE)
    critic = Critic().to(DEVICE)
    os.makedirs(model_path)
    g_optimizer = optim.Adam(generator.parameters(), lr=LR_G, betas=[0.0, 0.9])
    c_optimizer = optim.Adam(critic.parameters(), lr=LR_C, betas=[0.0, 0.9])
    train(generator, critic, dataloader, NUM_EPOCHS, LATENT_DIM, N_CRITIC, g_optimizer,
          c_optimizer, DEVICE, image_path, model_path, LEAD_MINS, LEAD_MAXS, LAMBDA_GP, LAMBDA_DTW)


if __name__ == "__main__":
    main()
