'''

'''
import os
import time
import numpy as np
import ctypes
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from preprocessing import bandpass_filter, per_lead_inverse_scaling, per_lead_minmax_scaling

# Calls setup for metrics functions from C
# Load the C Library for the metrics functions
metric_lib = ctypes.CDLL("gan_scripts/c_funcs/dtw.so")
# Set the arguments types for the dtw_distance function
metric_lib.dtw_distance.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
# Set return type for the dtw_distance function
metric_lib.dtw_distance.restype = ctypes.c_double
# Set the arguments types for the compute_mmd C function
metric_lib.compute_mmd.argtypes = [ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_double]
# Set return type for the compute_mmd function
metric_lib.compute_mmd.restype = ctypes.c_double


latent_dim = 50  # Latent space/noise dimension
num_seconds = 5  # Number of seconds as input
ecg_length = 128 * num_seconds  # Length of input ECG signals
n_leads = 3  # Number of leads as input and to generate
BATCH_SIZE = 64  # Batch size for dataset


if os.path.exists("biased_ptbxl_ecgs2.npy"):  # Check for the saved numpy file
    # Load the saved numpy file
    data = np.load("biased_ptbxl_ecgs2.npy", allow_pickle=True)
    normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(data)


# Create numpy array of each normalized ecg
normalized_data = np.array(normalized_data)

# Convert the numpy array to a torch tensor
dataset_tensor = torch.tensor(normalized_data, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(dataset_tensor),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  # Create a dataset loader for training the model, shuffles on each epoch


class Generator(nn.Module):
    '''
    Generator model for the WGAN-GP-DTW model. \\
    Consists of a twin stack of bidirectional LSTMs with 75 hidden units each \\
    and 5 
    '''

    def __init__(self, ecg_length=640, n_leads=3, latent_dim=50):
        super(Generator, self).__init__()
        self.ecg_length = ecg_length
        self.latent_dim = latent_dim
        self.n_leads = n_leads
        self.fc = nn.Linear(latent_dim, ecg_length * 32)
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
        self.tanh = nn.Tanh()  # Create tanh activation

    def forward(self, noise):
        x = self.fc(noise)
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


class MiniBatchDiscrimination(nn.Module):
    def __init__(self, input_dim, num_kernel, dim_kernel):
        super(MiniBatchDiscrimination, self).__init__()
        self.num_kernel = num_kernel
        self.dim_kernel = dim_kernel
        self.weight = nn.Parameter(torch.empty(
            input_dim, num_kernel * dim_kernel))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        activation = torch.matmul(x, self.weight)
        activation = activation.view(-1, self.num_kernel, self.dim_kernel)
        a = activation.unsqueeze(3)
        b = activation.permute(1, 2, 0).unsqueeze(0)
        diff = torch.abs(a - b)
        l1 = torch.sum(diff, dim=2)
        features = torch.sum(torch.exp(-l1), dim=2)
        out = torch.cat([x, features], dim=1)
        return out


class Critic(nn.Module):
    '''
    Critic class for the WGAN-GP-DTW model.\\
    Consists of 3 convolution1D-max pooling pair layers and a custom \\
    Minibatch discrimination layer
    '''

    def __init__(self, ecg_length=640, n_leads=3):
        super(Critic, self).__init__()
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        self.conv1 = spectral_norm(
            nn.Conv1d(n_leads, 64, kernel_size=9, stride=2, padding=4))
        self.conv2 = spectral_norm(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3))
        self.conv3 = spectral_norm(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2))
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.flatten_dim = self._get_flatten_dim()
        self.mb_discrim = MiniBatchDiscrimination(
            input_dim=self.flatten_dim, num_kernel=100, dim_kernel=5)
        self.fc = nn.Linear(self.flatten_dim + 100, 1)

    def _get_flatten_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_leads, self.ecg_length)
            x = self.conv1(dummy)
            x = self.leaky_relu(x)
            x = self.conv2(x)
            x = self.leaky_relu(x)
            x = self.conv3(x)
            x = self.leaky_relu(x)
            flat_dim = x.view(1, -1).size(1)
        return flat_dim

    def forward(self, ecg):
        x = ecg.transpose(1, 2)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = x.view(x.size(0), -1)
        x = self.mb_discrim(x)
        x = self.fc(x)
        return x


def compute_mmd(real_ecg, fake_ecg, sigma=1.0):
    real_np = real_ecg.detach().cpu().numpy().reshape(
        real_ecg.size(0), -1).astype(np.float64)
    fake_np = fake_ecg.detach().cpu().numpy().reshape(
        fake_ecg.size(0), -1).astype(np.float64)
    batch_real, features = real_np.shape
    batch_fake, _ = fake_np.shape
    result = metric_lib.compute_mmd(
        real_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fake_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        batch_real, batch_fake, features, sigma
    )
    return result


def compute_mvdTW(real_ecg, fake_ecg):
    if isinstance(real_ecg, torch.Tensor):
        real_np = real_ecg.detach().cpu().numpy()
    else:
        real_np = real_ecg
    if isinstance(fake_ecg, torch.Tensor):
        fake_np = fake_ecg.detach().cpu().numpy()
    else:
        fake_np = fake_ecg
    batch_size = real_np.shape[0]
    ecg_length = real_np.shape[1]
    n_leads = real_np.shape[2]
    distances = []
    for i in range(batch_size):
        seq1 = real_np[i].flatten().astype(np.float64)
        seq2 = fake_np[i].flatten().astype(np.float64)
        dtw_distance_val = metric_lib.dtw_distance(
            seq1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            seq2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ecg_length,
            ecg_length,
            n_leads
        )
        distances.append(dtw_distance_val)
    return np.mean(distances).astype(np.float32)


def gradient_penalty(critic, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolates.requires_grad_(True)
    critic_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(critic_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


def save_generated_ecg(generator, epoch, device, latent_dim=50, save_path="images"):
    os.makedirs(save_path, exist_ok=True)
    generator.eval()  # Set generator to evaluation mode i.e. Prevent training
    with torch.no_grad():  # Disable gradient calculation
        # Create noise in shape (1, latent_dim) latent_dim=50
        noise = torch.randn(1, latent_dim, device=device)
        gen_ecg = generator(noise).cpu().numpy()
    # Reverse the normalization done to ECG signals before training
    gen_ecg = per_lead_inverse_scaling(gen_ecg, lead_mins, lead_maxs)
    gen_ecg = gen_ecg.squeeze(0)
    # Post-process bandpass filter
    gen_ecg = bandpass_filter(gen_ecg, 0.5, 40, 128)
    num_leads = gen_ecg.shape[1]  # Get the number of leads to display
    Leads = ['III', 'V3', 'V5']  # Names of the leads used
    for lead in range(num_leads):  # Plot for each lead
        plt.figure(figsize=(8, 4))
        plt.plot(gen_ecg[:, lead], linewidth=1.5, color='black')
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude (mV)")
        plt.title(f"Generated ECG - Lead {Leads[lead]} - Epoch {epoch+1}")
        plt.grid(True)
        plt.savefig(os.path.join(
            save_path, f"ecg_epoch_{epoch+1}_lead_{lead+1}.png"), bbox_inches='tight', dpi=300)
        plt.close()
    generator.train()  # Set the generator back to training mode


def train_wgan_gp(generator, critic, dataloader, num_epochs, latent_dim, n_critic, lambda_gp, lambda_dtw, g_optimizer, c_optimizer, device, image_path, model_path):
    generator.train()  # Set generator to training mode
    critic.train()  # Set critic to training mode
    for epoch in range(num_epochs):  # Train for number of epochs
        # Take start time for epoch start to track time per epoch
        start_time_epoch = time.time()
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
                gp = gradient_penalty(critic, real_ecg, fake_ecg, device)
                # Modify the critic loss based on the gradient penalty
                loss_critic = loss_critic + lambda_gp * gp
                loss_critic.backward()  # Calculate gradients for critic
                c_optimizer.step()  # Update parameters for critic
            # Create new set of noises of shape (batch_size, latent_dim) latent_dim=50
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_ecg = generator(noise)
            g_optimizer.zero_grad()
            critic_fake = critic(fake_ecg)
            loss_generator = -critic_fake.mean()  # Generator Wasserstein loss
            # Compute the mvDTW loss figure
            dtw_loss = compute_mvdTW(real_ecg, fake_ecg)
            loss_generator = loss_generator + lambda_dtw * \
                torch.tensor(dtw_loss, device=device, dtype=torch.float32)
            loss_generator.backward()  # Calculate gradients for generator
            g_optimizer.step()  # Update parameters for generator
            # Compute the maximum mean discrepancy metric
            mmd_value = compute_mmd(real_ecg, fake_ecg)
            # Compute the Multivariate dynamic time warping metric
            mvdTW_value = compute_mvdTW(real_ecg, fake_ecg)
            end_time_step = time.time()  # Track end time for time per step
            print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {i+1}/{len(dataloader)} |"
                  f"Critic Loss: {loss_critic.item():.4f} | Generator Loss: {loss_generator.item():.4f} |"
                  f"MMD: {mmd_value:.4f} | mvdTW: {mvdTW_value:.4f} | Time: {end_time_step-start_time_step}")
        end_time_epoch = time.time()  # Track end time for time per epoch
        print(f"Epoch time elapsed: {end_time_epoch-start_time_epoch}")
        save_generated_ecg(generator, epoch,
                           device, latent_dim=latent_dim, save_path=image_path)  # Save images of each generated lead
        if epoch+1 == num_epochs:
            metrics = {
                "gen_loss": loss_generator.item(),
                "critic_loss": loss_critic.item(),
                "mmd": mmd_value,
                "mvdtw": mvdTW_value
            }
            model_state = {
                "generator_state_dict": generator.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "metrics": metrics
            }
            # Save generator model for later use
            torch.save(model_state, f"{model_path}/CWGAN.pth")


def main():
    # Set GPU device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim_local = latent_dim
    num_epochs = 50  # Number of epochs
    n_critic = 8  # Number of time (default=5)
    lambda_gp = 15.0  # Gradient penalty modifier hyperparameter (default=10.0)
    # Dynamic time warping modifier hyperparameter (default=0.1)
    lambda_dtw = 1.0
    GAN_model_num = 0
    generator = Generator(ecg_length=ecg_length, n_leads=n_leads,
                          latent_dim=latent_dim_local).to(device)  # Create Generator model and send to GPU
    critic = Critic(ecg_length=ecg_length, n_leads=n_leads).to(
        device)  # Create critic model and send to GPU
    g_optimizer = optim.RMSprop(generator.parameters(), lr=2e-4)
    c_optimizer = optim.RMSprop(critic.parameters(), lr=2e-4)
    while os.path.exists(f"images/WGAN_images/generated_images_wgan{GAN_model_num}"):
        GAN_model_num += 1
    image_path = f"images/WGAN_images/generated_images_wgan{GAN_model_num}"
    os.makedirs(image_path)  # Create new folder for the images to be saved to
    GAN_model_num = 0  # Reset folder index number for model saving
    # Check for folder number availability
    while os.path.exists(f"gan_scripts/gan/WGAN_models/wgan_{GAN_model_num}"):
        GAN_model_num += 1  # Increment model number for folder naming
    # Assign path for model to be saved to
    model_path = f"gan_scripts/gan/WGAN_models/wgan_{GAN_model_num}"
    os.makedirs(model_path)  # Create new folder for the models to be saved to
    train_wgan_gp(generator, critic, dataloader, num_epochs, latent_dim_local,
                  n_critic, lambda_gp, lambda_dtw, g_optimizer, c_optimizer, device, image_path, model_path)  # Begin training loop


if __name__ == "__main__":
    main()
