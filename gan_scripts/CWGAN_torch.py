import os
import time
import torch
import ctypes
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import spectral_norm
from preprocessing import bandpass_filter, per_lead_minmax_scaling, per_lead_inverse_scaling
import torch.optim as optim
import random


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

    def __init__(self, ecg_length=640, n_leads=3, condition_dim=1):
        super(Critic, self).__init__()
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        self.condition_dim = condition_dim
        self.conv1 = spectral_norm(
            nn.Conv1d(n_leads, 64, kernel_size=9, stride=2, padding=4))
        self.conv2 = spectral_norm(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3))
        self.conv3 = spectral_norm(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.flatten_dim = self._get_flatten_dim()
        self.mb_discrim = MiniBatchDiscrimination(
            input_dim=self.flatten_dim, num_kernel=100, dim_kernel=5)
        self.cond_fc = nn.Linear(condition_dim, 50)
        self.fc = nn.Linear(self.flatten_dim + 100 + 50, 1)
        self.condition_embedding = nn.Embedding(4, self.condition_dim)

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

    def forward(self, ecg, condition):
        cond_emb = self.condition_embedding(condition.squeeze(1))
        x = ecg.transpose(1, 2)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = x.view(x.size(0), -1)
        x = self.mb_discrim(x)
        cond_emb = self.cond_fc(cond_emb)
        x = torch.cat([x, cond_emb], dim=1)
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


def gradient_penalty(critic, real_samples, fake_samples, labels, device):
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolates.requires_grad_(True)
    critic_interpolates = critic(interpolates, labels)
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


def save_generated_ecg(generator, epoch, device, latent_dim=50, save_path="images", num_classes=4):
    os.makedirs(save_path, exist_ok=True)
    generator.eval()  # Set generator to evaluation mode i.e. Prevent training
    with torch.no_grad():  # Disable gradient calculation
        # Create noise in shape (1, latent_dim) latent_dim=50
        noise = torch.randn(1, latent_dim, device=device)
        random_label = random.randint(0, num_classes - 1)
        # Create a condition tensor with shape (1, 1)
        condition = torch.tensor(
            [random_label], device=device, dtype=torch.long).unsqueeze(1)
        # Generate ECG conditioned on the randomly chosen label
        gen_ecg = generator(noise, condition).cpu().numpy()
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
                    critic, real_ecg, fake_ecg, labels, device)
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
                  f" Critic Loss: {loss_critic.item():.4f} | Generator Loss: {loss_generator.item():.4f} |"
                  f" MMD: {mmd_value:.4f} | mvdTW: {mvdTW_value:.4f} | Time: {end_time_step-start_time_step}")
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


def main(dataloader, num_epochs, latent_dim, n_critic, lambda_gp, lambda_dtw, ecg_length, n_leads):
    # Set GPU device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(ecg_length=ecg_length, n_leads=n_leads,
                          latent_dim=latent_dim).to(device)  # Create Generator model and send to GPU
    critic = Critic(ecg_length=ecg_length, n_leads=n_leads).to(
        device)  # Create critic model and send to GPU
    # state_dict_gen = torch.load(
    #     "gan_scripts/gan/WGAN_models/wgan_38/generator.pth", map_location=device)
    # pretrained_fc_weight = state_dict_gen["fc.weight"]
    # extra_column = torch.zeros(pretrained_fc_weight.size(
    #     0), 1, device=pretrained_fc_weight.device)
    # new_fc_weight = torch.cat(
    #     (pretrained_fc_weight, extra_column), dim=1)
    # state_dict_gen["fc.weight"] = new_fc_weight
    # missing_keys_gen, unexpected_keys_gen = generator.load_state_dict(
    #     state_dict_gen, strict=False)
    # print("Generator missing keys:", missing_keys_gen)
    # print("Generator unexpected keys:", unexpected_keys_gen)
    # state_dict_crit = torch.load(
    #     "gan_scripts/gan/WGAN_models/wgan_38/critic.pth", map_location=device)
    # pretrained_fc_weight = state_dict_crit["fc.weight"]
    # extra_columns = torch.zeros(pretrained_fc_weight.size(
    #     0), 50, device=pretrained_fc_weight.device)
    # new_fc_weight = torch.cat((pretrained_fc_weight, extra_columns), dim=1)
    # state_dict_crit["fc.weight"] = new_fc_weight
    # missing_keys_crit, unexpected_keys_crit = critic.load_state_dict(
    #     state_dict_crit, strict=False)
    # print("Critic missing keys:", missing_keys_crit)
    # print("Critic unexpected keys:", unexpected_keys_crit)
    # for name, param in generator.named_parameters():
    #     if any(key in name for key in ["cnn1", "lstm1", "layernorm"]):
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    # for name, param in critic.named_parameters():
    #     # Freeze the convolutional and minibatch discrimination layers
    #     if any(key in name for key in ["conv1", "conv2", "mb_discrim"]):
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    g_optimizer = optim.RMSprop(generator.parameters(), lr=2e-4)
    c_optimizer = optim.RMSprop(critic.parameters(), lr=2e-4)
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
    n_critic = 8
    lambda_gp = 20.0  # Gradient penalty modifier hyperparameter (default=10.0)
    # Dynamic time warping modifier hyperparameter (default=0.1)
    lambda_dtw = 1.0
    latent_dim = 50  # Latent space/noise dimension
    num_seconds = 5  # Number of seconds as input
    ecg_length = 128 * num_seconds  # Length of input ECG signals
    n_leads = 3  # Number of leads as input and to generate
    BATCH_SIZE = 64  # Batch size for dataset
    if os.path.exists("fine_tune_data.npy"):  # Check for the saved numpy file
        data = np.load("fine_tune_data.npy", allow_pickle=True)
        segments = [item[0] for item in data]
        ecg_dataset = np.stack(segments)
        normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(
            ecg_dataset=ecg_dataset)
        labels = [item[1] for item in data]
        labels = np.array(labels)
    # Convert the numpy array to a torch tensor
    ecg_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long).unsqueeze(1)
    dataloader = DataLoader(TensorDataset(ecg_tensor, labels_tensor),
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  # Create a dataset loader for training the model, shuffles on each epoch
    main(dataloader=dataloader, num_epochs=num_epochs, latent_dim=latent_dim, n_critic=n_critic,
         lambda_gp=lambda_gp, lambda_dtw=lambda_dtw, ecg_length=ecg_length, n_leads=n_leads)
