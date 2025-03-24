'''
:File: CWGAN_torch.py
:Author: Luke Bogdanovic
:Date: 12/03/2025
:Purpose: Uses the trained classifier specified to predict the classes of a given
    set of data.
'''
import os
import time
import torch
import ctypes
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import spectral_norm
import torch.optim as optim
import pynvml
from gan_scripts.preprocessing_utils import per_lead_minmax_scaling, save_generated_ecg, compute_mmd, compute_mvdTW, gradient_penalty

# Calls setup for metrics functions from C
# Load the C Library for the metrics functions
metric_lib = ctypes.CDLL("c_funcs/dtw.so")
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


class Generator(nn.Module):
    '''
    Generator model for the WGAN-GP-DTW model.\\
    Consists of an initial CNN block for the extraction of local features,\\
    a twin stack of bidirectional LSTMs with 50 hidden units each to model temporal features,\\
    and a final CNN block for signal refinement before being output through a tanh activation function.
    '''

    def __init__(self, ecg_length=640, n_leads=3, latent_dim=50, condition_dim=1):
        '''
        Initialisation function for the Generator model. Creates the layers used\\
        in the generator model.

        :param ecg_length: Length of ECG to generate
        :param n_leads: Number of leads to generate
        :param latent_dim: Size of latent noise vector
        :param condition_dim: Size of the conditioning variables being used.
        '''
        super(Generator, self).__init__()
        self.ecg_length = ecg_length  # Length of the ECG signal
        self.n_leads = n_leads  # Number of leads to generate
        # Layer to scale the latent noise vector to (batch, 32, 640)
        self.fc = nn.Linear(latent_dim + condition_dim, ecg_length * 32)
        # CNN block 1 definition with 2 convolutions, for local features
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=7, padding=3),
            nn.ReLU()
        )
        # LSTM block definition, consisting of bidirectional 50 hidden unit LSTMs
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=50,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50,
                             num_layers=1, batch_first=True, bidirectional=True)
        # Layer normalization for LSTM layers
        self.layer_norm = nn.LayerNorm(100)
        # CNN block 2 definition for signal refinement
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=192,
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
            nn.Tanh()  # Ensure outputs are in range [-1,1]
        )
        # Embedding layer for the number of classes in dataset
        self.condition_embedding = nn.Embedding(4, condition_dim)

    def forward(self, noise, condition):
        '''
        Forward pass function for the generator model.

        :param noise: Latent noise vector for shaping with generator
        :param condition: Condition to add class to generated signals

        Returns:
            x: Generated signals
        '''
        cond_emb = self.condition_embedding(
            condition.squeeze(1))  # Create embedding of the condition
        # Concatenate condition to noise vector
        combined = torch.cat((noise, cond_emb), dim=1)
        x = self.fc(combined)  # Scale the latent noise vector
        # Reshape the latent noise vector into (batch, 32, 640) (batch, leads, timestep)
        x = x.view(-1, 32, self.ecg_length)
        x = self.cnn1(x)  # Model local features using CNN
        # Permute features into (batch, timestep, leads)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)  # Model temporal features with LSTM
        x = self.layer_norm(x)  # Stabilize generator training using layer norm
        x, _ = self.lstm2(x)  # Model temportal features with LSTM
        x = self.layer_norm(x)  # Stabilize generator training using layer norm
        # Permute features into (batch, leads, timestep)
        x = x.permute(0, 2, 1)
        x = self.cnn2(x)  # Refine generated signal features
        # Permute features into (batch, timestep, leads)
        x = x.permute(0, 2, 1)
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


class Critic(nn.Module):
    '''
    Critic class for the WGAN-GP-DTW model.\\
    Consists of 3 convolution1D with spectral normalization for stability.\\
    Uses minibatch discrimination layer for prevention of mode collapse.
    '''

    def __init__(self, ecg_length=640, n_leads=3, condition_dim=1):
        '''
        Initialisation function for the Critic model.

        :param ecg_length: Length of ECG in samples
        :param n_leads: Number of leads in ECG
        :param condition_dim: Size of the conditioning variables being used
        '''
        super(Critic, self).__init__()
        self.ecg_length = ecg_length  # Length of the ECG in samples
        self.n_leads = n_leads  # Number of leads in ECG
        self.condition_dim = condition_dim  # Size of condition dim
        # Create strided convolution layers stabilized using spectral normalization
        self.conv1 = spectral_norm(
            nn.Conv1d(n_leads, 64, kernel_size=5, stride=2, padding=4))
        self.conv2 = spectral_norm(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3))
        self.conv3 = spectral_norm(
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=2))
        # Create LeakyReLU activation function
        self.leaky_relu = nn.LeakyReLU(0.2)
        # Embedding layer for the number of classes in dataset
        self.condition_embedding = nn.Embedding(4, condition_dim)
        # Create conditional branch fully connected layer
        self.cond_fc = nn.Linear(condition_dim, 50)
        self.flatten_dim = self._get_flatten_dim()  # Get flattened dimension
        self.mb_discrim = MiniBatchDiscrimination(
            input_dim=self.flatten_dim, num_kernel=100, dim_kernel=5)  # Create minibatch discrimination layer
        # Create fully connected output layer
        self.fc = nn.Linear(self.flatten_dim + 100 + 50, 1)

    def _get_flatten_dim(self):
        '''
        Gets the dimension of the flattened output of the 3rd convolutional layer.

        Returns:
            flat_dim: Flattened dimension
        '''
        with torch.no_grad():
            # Create a dummy input for getting the flattened dimension
            dummy = torch.zeros(1, self.n_leads, self.ecg_length)
            x = self.conv1(dummy)
            x = self.leaky_relu(x)
            x = self.conv2(x)
            x = self.leaky_relu(x)
            x = self.conv3(x)
            x = self.leaky_relu(x)
            # Reshape features into a flattened vector and get size
            flat_dim = x.view(1, -1).size(1)
        return flat_dim

    def forward(self, ecg, condition):
        '''
        Forward pass function for the critic model.

        :param ecg: generated/real ECG for the critic to compare against other samples

        Returns:
            x: Output prediction of the critic
        '''
        cond_emb = self.condition_embedding(condition.squeeze(1))
        cond_emb = self.cond_fc(cond_emb)
        # Transpose leads and timestep for input to CNN
        x = ecg.transpose(1, 2)
        # CNN block
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        # Reshape feature vector into flattened vector
        x = x.view(x.size(0), -1)
        x = self.mb_discrim(x)  # Minibatch discrimination
        # Concatenate extracted features with condition
        x = torch.cat([x, cond_emb], dim=1)
        x = self.fc(x)  # Final fully connected layer
        return x


def train_wgan_gp(generator, critic, dataloader, num_epochs, latent_dim, n_critic, lambda_gp, lambda_dtw, g_optimizer, c_optimizer, device, image_path, model_path):
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
        'gpu_power_avg': []
    }
    for epoch in range(num_epochs):  # Train for number of epochs
        # Take start time for epoch start to track time per epoch
        start_time_epoch = time.time()
        # Set accumulation values to 0
        running_g_loss = 0.0
        running_c_loss = 0.0
        running_mmd = 0.0
        running_mvdtw = 0.0
        power_readings = []
        # Loop for steps per epoch and grab data from the dataloader
        for i, (real_ecg, labels) in enumerate(dataloader):
            # Take start time for step start to track time per step
            start_time_step = time.time()
            real_ecg = real_ecg.to(device)  # Send real sample to GPU
            labels = labels.to(device)
            # Get batch size value from the real sample shape
            batch_size = real_ecg.size(0)
            for _ in range(n_critic):  # Train critic n times
                # Create noise of shape (batch_size, latent_dim) latent_dim=50
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake_ecg = generator(noise, labels)  # Generate fake samples
                c_optimizer.zero_grad()  # Set critic optimizer gradients to zero
                critic_real = critic(real_ecg, labels)
                critic_fake = critic(fake_ecg.detach(), labels)
                # Critic Wasserstein loss calculation
                loss_critic = critic_fake.mean() - critic_real.mean()
                # Calculate the gradient penalty for real and fake ecg samples
                gp = gradient_penalty(
                    critic, real_ecg, fake_ecg, device, labels)
                # Modify the critic loss based on the gradient penalty
                loss_critic = loss_critic + lambda_gp * gp
                loss_critic.backward()  # Calculate gradients for critic
                c_optimizer.step()  # Update parameters for critic
            # Create new set of noises of shape (batch_size, latent_dim) latent_dim=50
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_ecg = generator(noise, labels)
            g_optimizer.zero_grad()
            critic_fake = critic(fake_ecg, labels)
            loss_generator = -critic_fake.mean()  # Generator Wasserstein loss
            # Compute the mvDTW loss figure
            # Compute the Multivariate dynamic time warping metric
            mvdTW_value = compute_mvdTW(
                real_ecg, fake_ecg, metric_lib=metric_lib)
            loss_generator = loss_generator + lambda_dtw * \
                torch.tensor(mvdTW_value, device=device, dtype=torch.float32)
            loss_generator.backward()  # Calculate gradients for generator
            g_optimizer.step()  # Update parameters for generator
            # Compute the maximum mean discrepancy metric
            mmd_value = compute_mmd(real_ecg, fake_ecg, metric_lib=metric_lib)
            running_c_loss += loss_critic.item()
            running_g_loss += loss_generator.item()
            running_mmd += mmd_value
            running_mvdtw += mvdTW_value
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            power_readings.append(power_usage)
            end_time_step = time.time()  # Track end time for time per step
            print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {i+1}/{len(dataloader)} |"
                  f" Critic Loss: {loss_critic.item():.4f} | Generator Loss: {loss_generator.item():.4f} |"
                  f" MMD: {mmd_value:.4f} | mvdTW: {mvdTW_value:.4f} | Time: {end_time_step-start_time_step} |"
                  f" GPU Power: {power_usage:.2f}W")
        end_time_epoch = time.time()  # Track end time for time per epoch
        avg_gpu_power = sum(power_readings)/len(power_readings)
        print(
            f"Epoch time elapsed: {end_time_epoch-start_time_epoch}s | Avg GPU Power: {avg_gpu_power:.2f}W")
        save_generated_ecg(generator, epoch,
                           device, lead_maxs=lead_maxs, lead_mins=lead_mins, latent_dim=latent_dim, save_path=image_path)  # Save images of each generated lead
        # Calculate average metrics for epoch
        gen_loss_epoch = running_g_loss / len(dataloader)
        critic_loss_epoch = running_c_loss / len(dataloader)
        mmd_epoch = running_mmd / len(dataloader)
        mvdTW_epoch = running_mvdtw / len(dataloader)
        # Add metrics to dictionary for saving
        metrics_history['epoch'].append(epoch+1)
        metrics_history['gen_loss'].append(gen_loss_epoch)
        metrics_history['critic_loss'].append(critic_loss_epoch)
        metrics_history['mvdtw'].append(mvdTW_epoch)
        metrics_history['mmd'].append(mmd_epoch)
        metrics_history['gpu_power_avg'].append(avg_gpu_power)
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
    pynvml.nvmlShutdown()  # Shutdown Nvidia management library


def main(dataloader, num_epochs, latent_dim, n_critic, lambda_gp, lambda_dtw, ecg_length, n_leads, pre_train=False):
    # Set GPU device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(ecg_length=ecg_length, n_leads=n_leads,
                          latent_dim=latent_dim).to(device)  # Create Generator model and send to GPU
    critic = Critic(ecg_length=ecg_length, n_leads=n_leads).to(
        device)  # Create critic model and send to GPU
    if pre_train:  # Check if the model is to use pretrained weights
        state_dict = torch.load(
            "gan_scripts/gan/WGAN_models/pretrain/CWGAN.pth", map_location=device, weights_only=False)
        # Load generator stated dictionary
        state_dict_gen = state_dict['gen_state_dict']
        pretrained_fc_weight = state_dict_gen["fc.weight"]
        extra_column = torch.zeros(pretrained_fc_weight.size(
            0), 1, device=pretrained_fc_weight.device)
        new_fc_weight = torch.cat(
            (pretrained_fc_weight, extra_column), dim=1)
        state_dict_gen["fc.weight"] = new_fc_weight
        missing_keys_gen, unexpected_keys_gen = generator.load_state_dict(
            state_dict_gen, strict=False)  # Load state dict weights into model
        print("Generator missing keys:", missing_keys_gen)
        print("Generator unexpected keys:", unexpected_keys_gen)
        state_dict_crit = state_dict['critic_state_dict']
        pretrained_fc_weight = state_dict_crit["fc.weight"]
        extra_columns = torch.zeros(pretrained_fc_weight.size(
            0), 50, device=pretrained_fc_weight.device)
        new_fc_weight = torch.cat((pretrained_fc_weight, extra_columns), dim=1)
        state_dict_crit["fc.weight"] = new_fc_weight
        missing_keys_crit, unexpected_keys_crit = critic.load_state_dict(
            state_dict_crit, strict=False)  # Load state dict weights into model
        print("Critic missing keys:", missing_keys_crit)
        print("Critic unexpected keys:", unexpected_keys_crit)
        for name, param in generator.named_parameters():
            if any(key in name for key in ["cnn1", "lstm1", "layernorm"]):
                param.requires_grad = False  # Set these layers to not train
            else:
                param.requires_grad = True  # Set layers to train
        for name, param in critic.named_parameters():
            # Freeze the convolutional and minibatch discrimination layers
            if any(key in name for key in ["conv1", "conv2"]):
                param.requires_grad = False  # Set layers to not train
            else:
                param.requires_grad = True  # Set layers to train
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=[0.0, 0.9])
    c_optimizer = optim.Adam(critic.parameters(), lr=1e-4, betas=[0.0, 0.9])
    GAN_model_num = 0
    while os.path.exists(f"images/CWGAN_images/generated_images_cwgan{GAN_model_num}"):
        GAN_model_num += 1
    image_path = f"images/CWGAN_images/generated_images_cwgan{GAN_model_num}"
    os.makedirs(image_path)  # Create new folder for the images to be saved to
    GAN_model_num = 0  # Reset folder index number for model saving
    # Check for folder number availability
    while os.path.exists(f"gan_scripts/gan/CWGAN_models/cwgan_{GAN_model_num}"):
        GAN_model_num += 1  # Increment model number for folder naming
    # Assign path for model to be saved to
    model_path = f"gan_scripts/gan/CWGAN_models/cwgan_{GAN_model_num}"
    os.makedirs(model_path)  # Create new folder for the models to be saved to
    train_wgan_gp(generator, critic, dataloader, num_epochs, latent_dim,
                  n_critic, lambda_gp, lambda_dtw, g_optimizer, c_optimizer, device, image_path, model_path)  # Begin training loop


if __name__ == "__main__":
    num_epochs = 50  # Number of epochs
    # Number of critic trainings per generator training (default=5)
    n_critic = 1
    lambda_gp = 10.0  # Gradient penalty modifier hyperparameter (default=10.0)
    # Dynamic time warping modifier hyperparameter (default=0.1)
    lambda_dtw = 0.5
    latent_dim = 50  # Latent space/noise dimension
    num_seconds = 5  # Number of seconds as input
    ecg_length = 128 * num_seconds  # Length of input ECG signals
    n_leads = 3  # Number of leads as input and to generate
    BATCH_SIZE = 128  # Batch size for dataset
    if os.path.exists("fine_tune_data.npy"):  # Check for the saved numpy file
        data = np.load("fine_tune_data.npy", allow_pickle=True)
        # Grab all ECG segments from the numpy file
        segments = [item[0] for item in data]
        # Stack all ECG segments in a numpy array
        ecg_dataset = np.stack(segments)
        normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(
            ecg_dataset=ecg_dataset)  # Perform minmax scaling per lead
        # Grab all labels from the data in numpy file
        labels = [item[1] for item in data]
        labels = np.array(labels)  # Create numpy array for the labels
    # Convert the numpy array to a torch tensor
    ecg_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long).unsqueeze(1)
    dataloader = DataLoader(TensorDataset(ecg_tensor, labels_tensor),
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=False)  # Create a dataset loader for training the model, shuffles on each epoch
    main(dataloader=dataloader, num_epochs=num_epochs, latent_dim=latent_dim, n_critic=n_critic,
         lambda_gp=lambda_gp, lambda_dtw=lambda_dtw, ecg_length=ecg_length, n_leads=n_leads, pre_train=True)
