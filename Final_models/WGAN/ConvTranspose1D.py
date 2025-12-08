from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tslearn.metrics import SoftDTWLossPyTorch
import time
import torch.optim as optim
import torch.nn as nn
import torch
from ignite.metrics import MaximumMeanDiscrepancy
import os
from preprocessing_utils import save_generated_ecg, per_lead_minmax_scaling, gradient_penalty
import torch.nn.functional as F

latent_dim = 100
num_seconds = 5
ecg_length = 128 * num_seconds
n_leads = 3
BATCH_SIZE = 128


class DeconvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_ch, out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm = nn.InstanceNorm1d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Generator(nn.Module):
    """
    Generator using ConvTranspose1d.
    noise: (B, latent_dim)
    output: (B, 3, 640)
    """

    def __init__(
        self,
        ecg_length: int = 640,
        n_leads: int = 3,
        latent_dim: int = 128,
        L0: int = 5,
        ch0: int = 256,
    ):
        super().__init__()

        assert L0 * \
            (2 **
             7) == ecg_length, f"L0 * 2^7 must equal ECG_length {ecg_length}"
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        self.latent_dim = latent_dim
        self.L0 = L0

        self.fc = nn.Linear(latent_dim, ch0*L0)
        chs = [ch0, 192, 128, 96, 64, 48, 32, 16]
        blocks = []
        for cin, cout in zip(chs[:-1], chs[1:]):
            blocks.append(DeconvBlock1D(cin, cout))
        self.deconv = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv1d(chs[-1], 32, kernel_size=15, padding=7, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, n_leads, kernel_size=7, padding=3, bias=True),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z):
        x = self.fc(z)                      # (B, ch0 * L0)
        x = x.view(z.size(0), -1, self.L0)  # (B, ch0, L0)
        x = self.deconv(x)                  # (B, ch_last, 640)
        x = self.head(x)                    # (B, 3, 640)
        return x.permute(0, 2, 1)           # (B, 640, 3)


class Critic(nn.Module):
    def __init__(self, ecg_length=640, n_leads=3, base_ch=64):
        super().__init__()
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        layers = []
        in_ch = n_leads
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*4, base_ch*8]
        for i, out_ch in enumerate(chs):
            if i == 0:
                k1 = 15
            elif i == 1:
                k1 = 11
            else:
                k1 = 7

            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k1,
                          stride=2, padding=k1//2),
                nn.LeakyReLU(0.2, inplace=True),
            ]

            # Second conv in stage: refine features (no further downsample)
            k2 = 3
            layers += [
                nn.Conv1d(out_ch, out_ch, kernel_size=k2,
                          stride=1, padding=k2//2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Accept (B, L, C) or (B, C, L)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.shape}")
        if x.shape[1] == self.ecg_length and x.shape[2] == self.n_leads:
            x = x.permute(0, 2, 1)  # (B, C, L)
        h = self.net(x)           # (B, C', ~20-ish)
        h = self.global_pool(h)   # (B, C', 1)
        h = h.squeeze(-1)         # (B, C')
        out = self.head(h)        # (B, 1)
        return out


def flatten_ecg(x):
    return x.reshape(x.size(0), -1)


def downsample_for_dtw(x, factor=4):
    # x: (B, 640, 3) or (B, 3, 640) – you use (B, 640, 3) here
    if x.shape[1] == 640 and x.shape[2] == 3:
        x = x.permute(0, 2, 1)  # (B, 3, 640)
    # (B, 3, 640/factor)
    x = F.avg_pool1d(x, kernel_size=factor, stride=factor)
    return x.permute(0, 2, 1)  # (B, 640/factor, 3)


def train(generator, critic, dataloader, num_epochs, latent_dim, n_critic, lambda_gp, lambda_dtw, g_optimizer, c_optimizer, device, image_path, model_path, lead_maxs, lead_mins):
    generator.train()
    critic.train()
    metrics_history = {
        'epoch': [],
        'gen_loss': [],
        'critic_loss': [],
        'mvdtw': [],
        'mmd': [],
        'wgap': [],
        'gp': []
    }
    softdtw = SoftDTWLossPyTorch(gamma=0.5, normalize=True).to(device=device)
    mmd = MaximumMeanDiscrepancy(var=1.0, device=device)
    subset = min(32, BATCH_SIZE)
    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        running_g_loss = 0.0
        running_c_loss = 0.0
        running_mmd = 0.0
        running_mvdtw = 0.0
        running_wgap = 0.0
        running_gp = 0.0
        mmd.reset()
        for i, (real_ecg, ) in enumerate(dataloader):
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
                gp = gradient_penalty(
                    critic, real_ecg, fake_ecg, device=device)
                loss_critic = loss_critic + (lambda_gp*gp)
                loss_critic.backward()
                c_optimizer.step()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_ecg = generator(noise)
            g_optimizer.zero_grad()
            critic_fake = critic(fake_ecg)
            loss_generator = -critic_fake.mean()
            # idx = torch.randperm(batch_size, device=device)[:subset]
            fake_sub = downsample_for_dtw(fake_ecg, factor=4)
            real_sub = downsample_for_dtw(real_ecg, factor=4)
            mvDTW_value = softdtw(fake_sub, real_sub).mean()
            mmd.reset()
            with torch.no_grad():
                fake_flat = flatten_ecg(fake_ecg)
                real_flat = flatten_ecg(real_ecg)
                fake_flat = fake_flat.to(device, dtype=torch.float32)
                real_flat = real_flat.to(device, dtype=torch.float32)
                mmd.update((fake_flat, real_flat))
            loss_generator = loss_generator + (lambda_dtw * mvDTW_value)
            loss_generator.backward()
            g_optimizer.step()
            mmd_step = mmd.compute()
            wgap = critic_real.mean().item() - critic_fake.mean().item()
            running_c_loss += loss_critic.item()
            running_g_loss += loss_generator.item()
            running_mvdtw += mvDTW_value.item()
            running_mmd += mmd_step
            running_wgap += wgap
            running_gp += gp.item()
            end_time_step = time.time()
            print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {i+1}/{len(dataloader)} |"
                  f" Critic Loss: {loss_critic.item():.4f} | Generator Loss: {loss_generator.item():.4f} |"
                  f" MMD: {mmd_step:.4f} | mvdTW: {mvDTW_value:.4f} | W-Gap: {wgap:.4f} | GP: {gp:.4f} | Time: {end_time_step-start_time_step}")
        end_time_epoch = time.time()
        save_generated_ecg(generator, epoch,
                           # Save images of each generated lead
                           device, latent_dim=latent_dim, save_path=image_path, lead_maxs=lead_maxs, lead_mins=lead_mins, num_classes=0)
        # Calculate average metrics for epoch
        gen_loss_epoch = running_g_loss / len(dataloader)
        critic_loss_epoch = running_c_loss / len(dataloader)
        mvdTW_epoch = running_mvdtw / len(dataloader)
        mmd_epoch = running_mmd / len(dataloader)
        wgap_epoch = running_wgap / len(dataloader)
        gp_epoch = running_gp / len(dataloader)
        print(
            f"Epoch time elapsed: {end_time_epoch-start_time_epoch}s | MMD: {mmd_epoch:.4f}")
        # Add metrics to dictionary for saving
        metrics_history['epoch'].append(epoch+1)
        metrics_history['gen_loss'].append(gen_loss_epoch)
        metrics_history['critic_loss'].append(critic_loss_epoch)
        metrics_history['mvdtw'].append(mvdTW_epoch)
        metrics_history['mmd'].append(mmd_epoch)
        metrics_history['wgap'].append(wgap_epoch)
        metrics_history['gp'].append(gp_epoch)
    # Create dictionary for Pytorch to save
    checkpoint = {
        'epoch': num_epochs,
        'gen_state_dict': generator.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'gen_optimizer_state_dict': g_optimizer.state_dict(),
        'critic_optimizer_state_dict': c_optimizer.state_dict(),
        'metrics_history': metrics_history
    }
    # Save the model and metrics
    torch.save(checkpoint, f"{model_path}/Model.pth")


def main():
    if os.path.exists("../../biased_ptbxl_ecgs.npy"):
        data = np.load("../../biased_ptbxl_ecgs.npy", allow_pickle=True)
        normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(data)
    normalized_data = np.array(normalized_data)
    dataset_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(dataset_tensor),
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    n_critic = 8
    lambda_gp = 10.0
    lambda_dtw = 1.0
    GAN_model_num = 0
    generator = Generator(ecg_length=ecg_length,
                          n_leads=n_leads, latent_dim=latent_dim).to(device)
    critic = Critic(ecg_length=ecg_length, n_leads=n_leads).to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=[0.0, 0.9])
    c_optimier = optim.Adam(critic.parameters(), lr=1e-4, betas=[0.0, 0.9])
    while os.path.exists(f"images/DCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"):
        GAN_model_num += 1
    image_path = f"images/DCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"
    os.makedirs(image_path)
    GAN_model_num = 0
    while os.path.exists(f"models/DCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"):
        GAN_model_num += 1
    model_path = f"models/DCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"
    os.makedirs(model_path)
    train(generator, critic, dataloader, num_epochs, latent_dim, n_critic,
          lambda_gp, lambda_dtw, g_optimizer, c_optimier, device, image_path, model_path, lead_maxs, lead_mins)


if __name__ == "__main__":
    main()
