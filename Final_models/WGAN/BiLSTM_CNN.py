'''
:File: WGAN_torch.py
:Author: Luke Bogdanovic
:Date: 12/03/2025
:Purpose: Script for training the WGAN model. Saves model and metrics of the trained model.
'''
import os
import time
import numpy as np
# import ctypes
from tslearn.metrics import SoftDTWLossPyTorch as SoftDTW
from ignite.metrics import MaximumMeanDiscrepancy as MMD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocessing_utils import per_lead_minmax_scaling, save_generated_ecg, gradient_penalty
import pynvml
import torch.nn.functional as F

latent_dim = 100  # Latent space/noise dimension
num_seconds = 5  # Number of seconds as input
ecg_length = 128 * num_seconds  # Length of input ECG signals
n_leads = 3  # Number of leads as input and to generate
BATCH_SIZE = 128  # Batch size for dataset


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(
                2), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class UpsampleBlock1d(nn.Module):
    def __init__(self, ch0, ch1, ch2, kernel_size=16, stride=1, padding='same', up_scale=2, up_mode='linear'):
        super().__init__()
        self.conv1 = nn.Conv1d(
            ch0, ch1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(
            ch1, ch2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.upsample = nn.Upsample(
            scale_factor=up_scale, mode=up_mode, align_corners=False)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.noise1 = NoiseInjection(ch1)
        self.noise2 = NoiseInjection(ch2)
        self.norm1 = nn.InstanceNorm1d(ch1)
        self.norm2 = nn.InstanceNorm1d(ch2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.upsample(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim=100, base_len=160, lstm_hidden=64, feat_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_len = base_len
        self.feat_dim = feat_dim
        self.fc = nn.Linear(latent_dim, base_len * feat_dim)
        self.bilstm: nn.LSTM = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        blocks = []
        blocks.append(UpsampleBlock1d(2*lstm_hidden, 128, 64))
        blocks.append(UpsampleBlock1d(64, 32, 16))
        self.conv_block = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=7, padding=3),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, n_leads, kernel_size=7, padding=3),
            nn.Tanh(),
        )

        self.conv_out = nn.Conv1d(
            in_channels=16,
            out_channels=n_leads,
            kernel_size=16,
            stride=1,
            padding="same"
        )
        self.tanh = nn.Tanh()

    def _init_weights(self):
        # Conv / Linear
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # LSTM init (orthogonal + forget gate bias)
        for name, param in self.bilstm.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                # set forget gate bias to 1
                n = param.size(0) // 4
                param.data[n:2*n].fill_(1.0)

    def forward(self, z: torch.Tensor):
        B = z.size(0)
        x: torch.Tensor = self.fc(z)
        x = x.view(B, self.base_len, self.feat_dim)
        x, _ = self.bilstm(x)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        # x = self.head(x)
        x = self.conv_out(x)
        x = self.tanh(x)
        return x


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
    if x.shape[1] == 640 and x.shape[2] == 3:
        x = x.permute(0, 2, 1)
    x = F.avg_pool1d(x, kernel_size=factor, stride=factor)
    return x.permute(0, 2, 1)


def train_wgan_gp(generator, critic, dataloader, num_epochs, latent_dim, n_critic, lambda_gp, lambda_dtw, g_optimizer, c_optimizer, device, image_path, model_path, lead_mins, lead_maxs):
    '''
    Training loop for the WGAN with gradient penalty model. Trains for the number of epochs
    specified using the optimizers provided for the generator and critic. Creates a noise
    vector with a latent space dimension specified. Trains critic for the number of times
    specified and adjusts the loss of the generator and critic using the scaling factors
    lambda_gp and lambda_dtw. Accepts training data from the dataloader. Saves images and
    the model.

    :param generator: Generator model
    :param critic: Critic model
    :param dataloader: Training set dataloader
    :param num_epochs: Number of epochs to train for
    :param latent_dim: Size of noise vector
    :param n_critic: Number of times to train critic
    :param lambda_gp: Gradient penalty regularization scaling value
    :param lambda_dtw: DTW regularization scaling value
    :param g_optimizer: Generator optimizer
    :param c_optimizer: Critic optimizer
    :param device: Device to send model and data to
    :param image_path: Path to store images
    :param model_path: Path to store model
    '''
    generator.train()  # Set generator to training mode
    critic.train()  # Set critic to training mode
    pynvml.nvmlInit()  # Initialise the NVIDIA management library
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get GPU:0
    # Create metrics history storage dictionary
    metrics_history = {
        'epoch': [],
        'gen_loss': [],
        'critic_loss': [],
        'mvdtw': [],
        'mmd': [],
        'wgap': [],
        'gp': [],
        'gpu_power_avg': []
    }
    dtw = SoftDTW(gamma=1.0, normalize=True)
    mmd = MMD(var=1.0, device=device)
    subset = min(32, BATCH_SIZE)
    for epoch in range(num_epochs):  # Train for number of epochs
        # Take start time for epoch start to track time per epoch
        start_time_epoch = time.time()
        # Set accumulation values to 0
        running_g_loss = 0.0
        running_c_loss = 0.0
        running_mmd = 0.0
        running_mvdtw = 0.0
        running_wgap = 0.0
        running_gp = 0.0
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
                fake_ecg = fake_ecg.permute(0, 2, 1)
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
            fake_ecg = fake_ecg.permute(0, 2, 1)
            critic_fake = critic(fake_ecg)  # Use critic to train generator
            loss_generator = -critic_fake.mean()  # Generator Wasserstein loss
            # Compute the mvDTW loss figure
            # idx = torch.randperm(batch_size, device=device)[:subset]
            fake_sub = downsample_for_dtw(fake_ecg, factor=4)
            real_sub = downsample_for_dtw(real_ecg, factor=4)
            mvDTW_value = dtw(fake_sub, real_sub).mean()
            mmd.reset()
            with torch.no_grad():
                fake_flat = flatten_ecg(fake_ecg)
                real_flat = flatten_ecg(real_ecg)
                fake_flat = fake_flat.to(device, dtype=torch.float32)
                real_flat = real_flat.to(device, dtype=torch.float32)
                mmd.update((fake_flat, real_flat))
            # mvdTW_value = compute_mvdTW(
            #     real_ecg, fake_ecg, metric_lib=metric_lib)  # Calcualte mvdtw
            # Calculate full generator loss
            loss_generator = loss_generator + \
                (lambda_dtw * mvDTW_value)
            loss_generator.backward()  # Calculate gradients for generator
            g_optimizer.step()  # Update parameters for generator
            # Compute the maximum mean discrepancy metric
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
            power_usage = pynvml.nvmlDeviceGetPowerUsage(
                handle) / 1000  # Gets power usage in watts
            power_readings.append(power_usage)
            end_time_step = time.time()  # Track end time for time per step
            print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {i+1}/{len(dataloader)} |"
                  f" Critic Loss: {loss_critic.item():.4f} | Generator Loss: {loss_generator.item():.4f} |"
                  f" MMD: {mmd_step:.4f} | mvdTW: {mvDTW_value:.4f} | D_Fake: {critic_fake.mean():.4f} | D_Real: {critic_real.mean():.4f} | W-Gap: {wgap:.4f} | GP: {gp:.4f} | Time: {end_time_step-start_time_step} |"
                  f" GPU Power: {power_usage:.2f}W")
        end_time_epoch = time.time()  # Track end time for time per epoch
        # Average power usage per step per epoch
        avg_gpu_power = sum(power_readings)/len(power_readings)
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
        # Add metrics to dictionary for saving
        metrics_history['epoch'].append(epoch+1)
        metrics_history['gen_loss'].append(gen_loss_epoch)
        metrics_history['critic_loss'].append(critic_loss_epoch)
        metrics_history['mvdtw'].append(mvdTW_epoch)
        metrics_history['mmd'].append(mmd_epoch)
        metrics_history['wgap'].append(wgap_epoch)
        metrics_history['gp'].append(gp_epoch)
        metrics_history['gpu_power_avg'].append(avg_gpu_power)
        print(
            f"Epoch time elapsed: {end_time_epoch-start_time_epoch}s | MMD: {mmd_epoch:.4f} | Avg GPU Power: {avg_gpu_power:.2f}W")
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
    pynvml.nvmlShutdown()  # Shutdown Nvidia management library


def main():
    if os.path.exists("biased_ptbxl_ecgs.npy"):  # Check for the saved numpy file
        # Load the saved numpy file
        data = np.load("biased_ptbxl_ecgs.npy", allow_pickle=True)
        normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(data)
    # Create numpy array of each normalized ecg
    normalized_data = np.array(normalized_data)
    # Convert the numpy array to a torch tensor
    dataset_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(dataset_tensor),
                            # Create a dataset loader for training the model, shuffles on each epoch
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50  # Number of epochs
    n_critic = 5  # Number of times critic is trained (default=5)
    lambda_gp = 10.0  # Gradient penalty modifier hyperparameter (default=10.0)
    # Dynamic time warping modifier hyperparameter (default=0.1)
    lambda_dtw = 1.0
    GAN_model_num = 0
    generator = Generator(latent_dim=latent_dim).to(device)
    critic = Critic(ecg_length=ecg_length, n_leads=n_leads).to(
        device)  # Create critic model and send to GPU
    g_optimizer = optim.Adam(generator.parameters(),
                             lr=2e-4, betas=[0.0, 0.9])
    c_optimizer = optim.Adam(critic.parameters(), lr=1e-4, betas=[0.0, 0.9])
    while os.path.exists(f"Final_models/WGAN/images/BiLSTM_CNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"):
        GAN_model_num += 1
    image_path = f"Final_models/WGAN/images/BiLSTM_CNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"
    os.makedirs(image_path)  # Create new folder for the images to be saved to
    GAN_model_num = 0  # Reset folder index number for model saving
    # Check for folder number availability
    while os.path.exists(f"Final_models/WGAN/models/BiLSTM_CNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"):
        GAN_model_num += 1  # Increment model number for folder naming
    # Assign path for model to be saved to
    model_path = f"Final_models/WGAN/models/BiLSTM_CNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"
    os.makedirs(model_path)  # Create new folder for the models to be saved to
    train_wgan_gp(generator, critic, dataloader, num_epochs, latent_dim,
                  # Begin training loop
                  n_critic, lambda_gp, lambda_dtw, g_optimizer, c_optimizer, device, image_path, model_path, lead_mins, lead_maxs)
    print(f"Model saved to: {model_path}/Model.pth")


if __name__ == "__main__":
    main()
