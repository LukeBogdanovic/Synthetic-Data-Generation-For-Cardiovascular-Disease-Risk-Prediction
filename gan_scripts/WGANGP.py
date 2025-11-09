import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
from tslearn.metrics import SoftDTWLossPyTorch
import numpy as np
from torch.nn.utils import spectral_norm
from torch.utils.data import TensorDataset, DataLoader
from preprocessing_utils import save_generated_ecg, per_lead_minmax_scaling, gradient_penalty
from ignite.metrics import MaximumMeanDiscrepancy

latent_dim = 50  # Latent space/noise dimension
num_seconds = 5  # Number of seconds as input
ecg_length = 100 * num_seconds  # Length of input ECG signals
n_leads = 3  # Number of leads as input and to generate
BATCH_SIZE = 128  # Batch size for dataset


class Generator(nn.Module):
    def __init__(self, ecg_length=ecg_length, n_leads=n_leads, latent_dim=latent_dim, L0=5, ch0=256, ups_factors=(5, 5, 2, 2), ch_min=16):
        super(Generator, self).__init__()
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
                f"Got {L0} * {prod} != {ecg_length}"
            )
        self.fc = nn.Linear(latent_dim, ch0 * L0)
        chs = [ch0]
        c = ch0
        for _ in ups_factors:
            c = max(c // 2, ch_min)
            chs.append(c)
        layers = []
        for cin, cout, sf in zip(chs[:-1], chs[1:], ups_factors):
            layers += [
                nn.Upsample(scale_factor=sf, mode='linear',
                            align_corners=False),
                nn.Conv1d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            ]
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(chs[-1], 8, kernel_size=7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, n_leads, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(noise.size(0), -1, self.L0)
        x = self.deconv(x)
        x = self.head(x)
        return x.transpose(1, 2)


class Critic(nn.Module):
    def __init__(self, ecg_length=640, n_leads=3, base=64):
        super().__init__()
        chs = [n_leads, base, 96, 128, 192, 256, 384, 512]
        blocks = []
        for cin, cout in zip(chs[:-1], chs[1:]):
            blocks += [
                nn.Conv1d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(cout, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, L)
        h = self.net(x)  # (B, C', L/128 ≈ 5)
        h = h.mean(dim=-1)  # global avg pool over time
        return self.head(h)


class MiniBatchDiscrimination(nn.Module):
    def __init__(self, input_dim, num_kernel, dim_kernel):
        super(MiniBatchDiscrimination, self).__init__()
        self.num_kernel = num_kernel
        self.input_dim = input_dim
        self.dim_kernel = dim_kernel
        self.weight = nn.Parameter(torch.empty(
            self.input_dim, self.num_kernel*self.dim_kernel))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        activation = torch.matmul(x, self.weight)
        activation = activation.view(-1, self.num_kernel, self.dim_kernel)
        a = activation.unsqueeze(3)
        b = activation.permute(1, 2, 0).unsqueeze(0)
        diff = torch.abs(a-b)
        l1 = torch.sum(diff, dim=2)
        features = torch.sum(torch.exp(-l1), dim=2)
        out = torch.cat([x, features], dim=1)
        return out


def flatten_ecg(x):
    return x.reshape(x.size(0), -1)


def train(generator, critic, dataloader, num_epochs, latent_dim, n_critic, lambda_gp, lambda_dtw, g_optimizer, c_optimizer, device, image_path, model_path, lead_maxs, lead_mins):
    generator.train()  # Set generator to training mode
    critic.train()  # Set critic to training mode
    # Create metrics history storage dictionary
    metrics_history = {
        'epoch': [],
        'gen_loss': [],
        'critic_loss': [],
        'mvdtw': [],
        'mmd': [],
        'gpu_power_avg': []
    }
    softdtw = SoftDTWLossPyTorch(gamma=0.5, normalize=True)
    mmd = MaximumMeanDiscrepancy(var=1.0)
    subset = min(32, BATCH_SIZE)
    for epoch in range(num_epochs):  # Train for number of epochs
        # Take start time for epoch start to track time per epoch
        start_time_epoch = time.time()
        # Set accumulation values to 0
        running_g_loss = 0.0
        running_c_loss = 0.0
        running_mvdtw = 0.0
        power_readings = []
        mmd.reset()
        # Loop for steps per epoch and grab data from the dataloader
        for i, (real_ecg,) in enumerate(dataloader):
            # Take start time for step start to track time per step
            start_time_step = time.time()
            real_ecg = real_ecg.to(device)  # Send real sample to GPU
            # Get batch size value from the real sample shape
            batch_size = real_ecg.size(0)
            for _ in range(n_critic):  # Train critic n times
                # Create noise of shape (batch_size, latent_dim) latent_dim=50
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake_ecg = generator(noise)  # Generate fake samples
                c_optimizer.zero_grad()  # Set critic optimizer gradients to zero
                critic_real = critic(real_ecg)
                critic_fake = critic(fake_ecg.detach())
                # Critic Wasserstein loss calculation
                loss_critic = critic_fake.mean() - critic_real.mean()
                # Calculate the gradient penalty for real and fake ecg samples
                gp = gradient_penalty(
                    critic, real_ecg, fake_ecg, device=device)
                # Modify the critic loss based on the gradient penalty
                loss_critic = loss_critic + (lambda_gp * gp)
                loss_critic.backward()  # Calculate gradients for critic
                c_optimizer.step()  # Update parameters for critic
            # Create new set of noises of shape (batch_size, latent_dim) latent_dim=50
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_ecg = generator(noise)  # Generate fake samples
            g_optimizer.zero_grad()  # Set generator optimizer gradients to zero for backpropagation
            critic_fake = critic(fake_ecg)  # Use critic to train generator
            loss_generator = -critic_fake.mean()  # Generator Wasserstein loss
            idx = torch.randperm(batch_size, device=device)[:subset]
            # Compute the mvDTW loss figure
            mvdTW_value = softdtw(fake_ecg[idx], real_ecg[idx]).mean()
            # Compute the maximum mean discrepancy metric
            with torch.no_grad():
                fake_flat = flatten_ecg(fake_ecg[idx])
                real_flat = flatten_ecg(real_ecg[idx])
                fake_flat = fake_flat.to(device, dtype=torch.float32)
                real_flat = real_flat.to(device, dtype=torch.float32)
                mmd.update((fake_flat, real_flat))
            loss_generator = loss_generator + (lambda_dtw *
                                               mvdTW_value)  # + (lambda_mmd * mmd.compute())  # Calculate full generator loss
            loss_generator.backward()  # Calculate gradients for generator
            g_optimizer.step()  # Update parameters for generator
            # Add calculated values to accumulation variables
            running_c_loss += loss_critic.item()
            running_g_loss += loss_generator.item()
            running_mvdtw += mvdTW_value.item()
            end_time_step = time.time()  # Track end time for time per step
            print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {i+1}/{len(dataloader)} |"
                  f" Critic Loss: {loss_critic.item():.4f} | Generator Loss: {loss_generator.item():.4f} |"
                  f" MMD: {mmd.compute() :.4f} |mvdTW: {mvdTW_value:.4f} | Time: {end_time_step-start_time_step}")
        end_time_epoch = time.time()  # Track end time for time per epoch
        mmd_epoch = mmd.compute()
        print(
            f"Epoch time elapsed: {end_time_epoch-start_time_epoch}s | MMD: {mmd_epoch:.4f}")
        save_generated_ecg(generator, epoch,
                           device, latent_dim=latent_dim, save_path=image_path, lead_maxs=lead_maxs, lead_mins=lead_mins, num_classes=0)  # Save images of each generated lead
        # Calculate average metrics for epoch
        gen_loss_epoch = running_g_loss / len(dataloader)
        critic_loss_epoch = running_c_loss / len(dataloader)
        mvdTW_epoch = running_mvdtw / len(dataloader)
        # Add metrics to dictionary for saving
        metrics_history['epoch'].append(epoch+1)
        metrics_history['gen_loss'].append(gen_loss_epoch)
        metrics_history['critic_loss'].append(critic_loss_epoch)
        metrics_history['mvdtw'].append(mvdTW_epoch)
        metrics_history['mmd'].append(mmd_epoch)
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
    torch.save(checkpoint, f"{model_path}/CWGAN.pth")


def main():
    if os.path.exists("../biased_ptbxl_ecgs_100Hz.npy"):  # Check for the saved numpy file
        # Load the saved numpy file
        data = np.load("../biased_ptbxl_ecgs_100Hz.npy", allow_pickle=True)
        normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(data)
    # Create numpy array of each normalized ecg
    normalized_data = np.array(normalized_data)
    # Convert the numpy array to a torch tensor
    dataset_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(dataset_tensor),
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    # Set GPU device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50  # Number of epochs
    n_critic = 2  # Number of times critic is trained (default=5)
    lambda_gp = 5.0  # Gradient penalty modifier hyperparameter (default=10.0)
    # Dynamic time warping modifier hyperparameter (default=0.1)
    lambda_dtw = 0.05
    lambda_mmd = 0.1
    GAN_model_num = 0
    generator = Generator(ecg_length=ecg_length, n_leads=n_leads,
                          latent_dim=latent_dim).to(device)  # Create Generator model and send to GPU
    critic = Critic(ecg_length=ecg_length, n_leads=n_leads).to(
        device)  # Create critic model and send to GPU
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=[0.0, 0.9])
    c_optimizer = optim.Adam(critic.parameters(), lr=1e-4, betas=[0.0, 0.9])
    while os.path.exists(f"images/BiLSTM_WGAN_models/generated_images_wgan{GAN_model_num}"):
        GAN_model_num += 1
    image_path = f"images/BiLSTM_WGAN_models/generated_images_wgan{GAN_model_num}"
    os.makedirs(image_path)  # Create new folder for the images to be saved to
    GAN_model_num = 0  # Reset folder index number for model saving
    # Check for folder number availability
    while os.path.exists(f"gan_scripts/gan/BiLSTM_WGAN_models/wgan_{GAN_model_num}"):
        GAN_model_num += 1  # Increment model number for folder naming
    # Assign path for model to be saved to
    model_path = f"gan_scripts/gan/BiLSTM_WGAN_models/wgan_{GAN_model_num}"
    os.makedirs(model_path)  # Create new folder for the models to be saved to
    train(generator, critic, dataloader, num_epochs, latent_dim,
          n_critic, lambda_gp, lambda_dtw, g_optimizer, c_optimizer, device, image_path, model_path, lead_maxs, lead_mins)  # Begin training loop


if __name__ == "__main__":
    main()
