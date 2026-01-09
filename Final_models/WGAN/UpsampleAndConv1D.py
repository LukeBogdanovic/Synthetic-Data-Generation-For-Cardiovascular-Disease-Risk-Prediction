from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tslearn.metrics import SoftDTWLossPyTorch
import time
import torch.optim as optim
import torch.nn as nn
import torch
from ignite.metrics import MaximumMeanDiscrepancy
import os
from Final_models.WGAN.preprocessing_utils import save_generated_ecg, per_lead_minmax_scaling, gradient_penalty
import torch.nn.functional as F

latent_dim = 100
num_seconds = 5
ecg_length = 128 * num_seconds
n_leads = 3
BATCH_SIZE = 128


class NoiseInjection(nn.Module):
    """
    Adds learned per-channel noise (StyleGAN trick).
    Produces ECG-like micro-variability.
    """

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), device=x.device)
        return x + self.weight * noise


class Generator(nn.Module):
    def __init__(self, ecg_length=640, n_leads=3, latent_dim=100,
                 L0=10, ch0=256, ups_factors=(2, 2, 2, 2, 2, 2), ch_min=16):
        super().__init__()
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        self.latent_dim = latent_dim
        self.L0 = L0

        prod = 1
        for f in ups_factors:
            prod *= f
        if L0 * prod != ecg_length:
            raise ValueError(
                f"L0 * product(ups_factors) must equal ecg_length. "
                f"Got {L0} * {prod} = {L0 * prod} != {ecg_length}"
            )

        self.fc = nn.Linear(latent_dim, ch0 * L0)

        chs = [ch0]
        c = ch0
        for _ in ups_factors:
            c = max(c // 2, ch_min)
            chs.append(c)

        blocks = []
        for cin, cout, sf in zip(chs[:-1], chs[1:], ups_factors):
            blocks += [
                nn.Upsample(scale_factor=sf, mode="linear",
                            align_corners=False),
                nn.Conv1d(cin, cout, kernel_size=5, padding=2, bias=False),
                nn.InstanceNorm1d(cout),
                NoiseInjection(cout),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(cout, cout, kernel_size=3, padding=1),
                nn.InstanceNorm1d(cout),
                NoiseInjection(cout),
                nn.LeakyReLU(0.3, inplace=True)
            ]
        self.deconv = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv1d(chs[-1], chs[-1], kernel_size=7, padding=3),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(chs[-1], n_leads, kernel_size=7, padding=3),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)
        return: (B, 640, 3)
        """
        x = self.fc(z)                      # (B, ch0 * L0)
        x = x.view(z.size(0), -1, self.L0)  # (B, ch0, L0)
        x = self.deconv(x)                  # (B, ch_last, 640)
        x = self.head(x)                    # (B, 3, 640)
        return x.permute(0, 2, 1)           # (B, 640, 3)


class MiniBatchDiscrimination(nn.Module):
    '''
    Implements minibatch discrimination to help the critic to detect mode collapse.\\
    Compares each sample with other samples in the same batch to add features based\\
    on the similarity of generated signals to other signals in the batch.
    '''

    def __init__(self, input_dim, num_kernel, dim_kernel):
        '''
        Initialises the weights for the minibatch discrimination layer.

        :param input_dim: Input dimension from the last layers output
        :param num_kernel: Number of kernels to compute over
        :param dim_kernel: Dimension of kernels to compute over
        '''
        super(MiniBatchDiscrimination, self).__init__()
        self.num_kernel = num_kernel  # Number of kernel functions to use
        self.dim_kernel = dim_kernel  # Dimensionality of each kernel
        self.weight = nn.Parameter(torch.empty(
            input_dim, num_kernel * dim_kernel))  # Create a learnable matrix
        # Initialise the weights using Xavier
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Project input to a new space
        activation = torch.matmul(x, self.weight)
        activation = activation.view(-1, self.num_kernel, self.dim_kernel)
        a = activation.unsqueeze(3)
        b = activation.permute(1, 2, 0).unsqueeze(0)
        diff = torch.abs(a - b)  # Pairwise absolute difference between samples
        # L1 norm distance across kernel dimensions
        l1 = torch.sum(diff, dim=2)
        features = torch.sum(torch.exp(-l1), dim=2)  # Measure the similarity
        # Concatenate original and similarity features
        out = torch.cat([x, features], dim=1)
        return out


class CriticConvBlock(nn.Module):
    def __init__(self, ch0, ch1, ch2, kernel_size=16, stride=1, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv1d(
            ch0, ch1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(
            ch1, ch2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class Critic(nn.Module):
    '''
    Critic class for the WGAN-GP-DTW model.\\
    Consists of 3 convolution1D with spectral normalization for stability.\\
    Uses minibatch discrimination layer for prevention of mode collapse.
    '''

    def __init__(self, ecg_length=640, n_leads=3, num_kernel=50, dim_kernel=10):
        super().__init__()
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        blocks = []
        blocks.append(CriticConvBlock(n_leads, 32, 64))
        blocks.append(CriticConvBlock(64, 128, 256))
        self.conv_block = nn.Sequential(*blocks)
        with torch.no_grad():
            dummy = torch.zeros(1, n_leads, ecg_length)
            h = self.conv_block(dummy)
            flatten_dim = h.view(1, -1).size(1)
        self.minibatch = MiniBatchDiscrimination(
            input_dim=flatten_dim, num_kernel=50, dim_kernel=10)
        self.fc = nn.Linear(flatten_dim+num_kernel, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.shape[1] == self.ecg_length and x.shape[2] == self.n_leads:
            x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.minibatch(x)
        x = self.fc(x)
        return x


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
    softdtw = SoftDTWLossPyTorch(gamma=0.5, normalize=True).to(device)
    mmd = MaximumMeanDiscrepancy(var=1.0, device=device)
    subset = min(64, BATCH_SIZE)
    mvdtw_subset = min(64, BATCH_SIZE)
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
                # fake_ecg = fake_ecg.permute(0, 2, 1)
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
            # fake_ecg = fake_ecg.permute(0, 2, 1)
            critic_fake = critic(fake_ecg)
            loss_generator = -critic_fake.mean()
            # mvdtw_idx = torch.randperm(batch_size, device=device)[
            #     :mvdtw_subset]
            # idx = torch.randperm(batch_size, device=device)[:subset]
            fake_sub = downsample_for_dtw(fake_ecg, factor=4)
            real_sub = downsample_for_dtw(real_ecg, factor=4)
            mvDTW_value = softdtw(fake_sub, real_sub).mean()
            mmd.reset()
            # compute_mmd = (i % 5 == 0)
            # if compute_mmd:
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
            # mmd_value = compute_mmd(real_ecg, fake_ecg, metric_lib=metric_lib)
            wgap = critic_real.mean().item() - critic_fake.mean().item()
            # Add calculated values to accumulation variables
            running_c_loss += loss_critic.item()
            running_g_loss += loss_generator.item()
            running_mmd += mmd_step
            running_mvdtw += mvDTW_value.item()
            running_wgap += wgap
            running_gp += gp.item()
            end_time_step = time.time()
            print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {i+1}/{len(dataloader)} |"
                  f" Critic Loss: {loss_critic.item():.4f} | Generator Loss: {loss_generator.item():.4f} |"
                  f" MMD: {mmd_step:.4f} | mvdTW: {mvDTW_value:.4f} | D_Fake: {critic_fake.mean():.4f} | D_Real: {critic_real.mean():.4f} | W-Gap: {wgap:.4f} | GP: {gp:.4f} | Time: {end_time_step-start_time_step}")
        end_time_epoch = time.time()
        save_generated_ecg(generator, epoch,
                           # Save images of each generated lead
                           device, latent_dim=latent_dim, save_path=image_path, lead_maxs=lead_maxs, lead_mins=lead_mins, num_classes=0)
        # Calculate average metrics for epoch
        gen_loss_epoch = running_g_loss / len(dataloader)
        critic_loss_epoch = running_c_loss / len(dataloader)
        mmd_epoch = running_mmd / len(dataloader)
        mvdTW_epoch = running_mvdtw / len(dataloader)
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
    n_critic = 5
    lambda_gp = 10.0
    lambda_dtw = 1.0
    GAN_model_num = 0
    generator = Generator(ecg_length=ecg_length,
                          n_leads=n_leads, latent_dim=latent_dim).to(device)
    critic = Critic(ecg_length=ecg_length, n_leads=n_leads).to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=[0.0, 0.9])
    c_optimier = optim.Adam(critic.parameters(), lr=1e-4, betas=[0.0, 0.9])
    while os.path.exists(f"images/UpsampleAndCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"):
        GAN_model_num += 1
    image_path = f"images/UpsampleAndCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"
    os.makedirs(image_path)
    GAN_model_num = 0
    while os.path.exists(f"models/UpsampleAndCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"):
        GAN_model_num += 1
    model_path = f"models/UpsampleAndCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"
    os.makedirs(model_path)
    train(generator, critic, dataloader, num_epochs, latent_dim, n_critic,
          lambda_gp, lambda_dtw, g_optimizer, c_optimier, device, image_path, model_path, lead_maxs, lead_mins)


if __name__ == "__main__":
    main()
