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
from Final_models.WGAN.ConvTranspose1D import Generator as WGAN_Gen
from Final_models.WGAN.ConvTranspose1D import Critic as WGAN_Critic

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
        self.conv = nn.Conv1d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1
        )
        self.norm = nn.InstanceNorm1d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
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
        latent_dim: int = 100,
        L0: int = 5,
        ch0: int = 256,
        condition_dim=32
    ):
        super().__init__()

        assert L0 * \
            (2 **
             7) == ecg_length, f"L0 * 2^7 must equal ECG_length {ecg_length}"
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        self.latent_dim = latent_dim
        self.L0 = L0

        self.fc = nn.Linear(latent_dim+condition_dim, ch0*L0)
        chs = [ch0, 192, 128, 96, 64, 48, 32, 16]
        blocks = []
        for cin, cout in zip(chs[:-1], chs[1:]):
            blocks.append(DeconvBlock1D(cin, cout))
        self.deconv = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv1d(chs[-1], 32, kernel_size=15, padding=7, bias=True),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, n_leads, kernel_size=7, padding=3, bias=True),
            nn.Tanh()
        )

        self.cond_block = nn.Sequential(
            nn.Embedding(4, condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor, condition) -> torch.Tensor:
        cond_emb = self.cond_block(condition.squeeze(1))
        combined = torch.cat((z, cond_emb), dim=1)
        x = self.fc(combined)
        x = x.view(z.size(0), -1, self.L0)
        x = self.deconv(x)
        x = self.head(x)
        return x.permute(0, 2, 1)


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
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)
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


class FiLM1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 2*channels)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gb = self.net(c)
        g, b = gb.chunk(2, dim=1)
        return x * (1.0 + g.unsqueeze(-1)) + b.unsqueeze(-1)


class Critic(nn.Module):
    '''
    Critic class for the WGAN-GP-DTW model.\\
    Consists of 3 convolution1D with spectral normalization for stability.\\
    Uses minibatch discrimination layer for prevention of mode collapse.
    '''

    def __init__(self, ecg_length=640, n_leads=3, num_classes=4, num_kernel=50, dim_kernel=10, condition_dim=32, concat_condition=True):
        super().__init__()
        self.ecg_length = ecg_length
        self.n_leads = n_leads
        self.condition_dim = condition_dim
        self.concat_condition = concat_condition
        self.cond_block = nn.Sequential(
            nn.Embedding(num_classes, condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block1 = CriticConvBlock(n_leads, 32, 64)
        self.block2 = CriticConvBlock(64, 128, 256)
        self.film1 = FiLM1D(channels=64, cond_dim=condition_dim, hidden=64)
        self.film2 = FiLM1D(channels=256, cond_dim=condition_dim, hidden=64)
        with torch.no_grad():
            dummy = torch.zeros(1, n_leads, ecg_length)
            c_dummy = torch.zeros(1, dtype=torch.long)
            c_vec = self.cond_block(c_dummy)
            h = self.block1(dummy)
            h = self.film1(h, c_vec)
            h = self.block2(h)
            h = self.film2(h, c_vec)
            flatten_dim = h.view(1, -1).size(1)
        self.minibatch = MiniBatchDiscrimination(
            input_dim=flatten_dim,
            num_kernel=num_kernel,
            dim_kernel=dim_kernel
        )
        fc_in = flatten_dim+num_kernel
        if self.concat_condition:
            fc_in += condition_dim
        self.fc = nn.Linear(fc_in, 1)

    def forward(self, x, condition):
        if condition.dim() == 2 and condition.size(1) == 1:
            condition = condition.squeeze(1)
        c_vec = self.cond_block(condition)
        if x.dim() == 3 and x.shape[1] == self.ecg_length and x.shape[2] == self.n_leads:
            x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = self.film1(x, c_vec)
        x = self.block2(x)
        x = self.film2(x, c_vec)
        x = x.view(x.size(0), -1)
        x = self.minibatch(x)
        if self.concat_condition:
            x = torch.cat([x, c_vec], dim=1)
        return self.fc(x)


def load_wgan_to_cwgan_generator(wgan_gen: WGAN_Gen, cwgan_gen: Generator, latent_dim: int):
    wgan_sd = wgan_gen.state_dict()
    cwgan_sd = cwgan_gen.state_dict()
    compatible = {}
    for k, v in wgan_sd.items():
        if k in cwgan_sd and cwgan_sd[k].shape == v.shape:
            compatible[k] = v
    cwgan_gen.load_state_dict(compatible, strict=False)
    with torch.no_grad():
        cwgan_gen.fc.weight[:, :latent_dim].copy_(wgan_sd["fc.weight"])
        cwgan_gen.fc.bias.copy_(wgan_sd["fc.bias"])
    return cwgan_gen


def load_wgan_to_cwgan_critic(wgan_critic: WGAN_Critic, cwgan_critic: Critic):
    wgan_sd = wgan_critic.state_dict()
    wgan_sd_filtered = {k: v for k, v in wgan_sd.items() if k not in [
        "fc.weight"]}
    cwgan_critic.load_state_dict(wgan_sd_filtered, strict=False)
    with torch.no_grad():
        w_fc_w = wgan_sd["fc.weight"]
        w_fc_b = wgan_sd["fc.bias"]
        in_w = w_fc_w.shape[1]
        cwgan_critic.fc.weight[:, :in_w].copy_(w_fc_w)
        cwgan_critic.fc.bias.copy_(w_fc_b)
    return cwgan_critic


def flatten_ecg(x):
    return x.reshape(x.size(0), -1)


def downsample_for_dtw(x, factor=4):
    if x.shape[1] == 640 and x.shape[2] == 3:
        x = x.permute(0, 2, 1)
    x = F.avg_pool1d(x, kernel_size=factor, stride=factor)
    return x.permute(0, 2, 1)


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
    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        running_g_loss = 0.0
        running_c_loss = 0.0
        running_mmd = 0.0
        running_mvdtw = 0.0
        running_wgap = 0.0
        running_gp = 0.0
        mmd.reset()
        for i, (real_ecg, labels) in enumerate(dataloader):
            start_time_step = time.time()
            real_ecg: torch.Tensor = real_ecg.to(device)
            labels: torch.Tensor = labels.to(device)
            batch_size = real_ecg.size(0)
            for _ in range(n_critic):
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake_ecg: torch.Tensor = generator(noise, labels)
                c_optimizer.zero_grad()
                critic_real = critic(real_ecg, labels)
                critic_fake = critic(fake_ecg.detach(), labels)
                loss_critic = critic_fake.mean() - critic_real.mean()
                gp = gradient_penalty(
                    critic, real_ecg, fake_ecg, device=device, labels=labels)
                loss_critic = loss_critic + (lambda_gp*gp)
                loss_critic.backward()
                c_optimizer.step()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_ecg = generator(noise, labels)
            g_optimizer.zero_grad()
            critic_fake = critic(fake_ecg, labels)
            loss_generator = -critic_fake.mean()
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
                           device, latent_dim=latent_dim, save_path=image_path, lead_maxs=lead_maxs, lead_mins=lead_mins, num_classes=4)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wgan_gen = WGAN_Gen().to(device)
    wgan_critic = WGAN_Critic().to(device)
    ckpt = torch.load(
        "WGAN/models/DCNN_WGAN/Model_0_GP_10.0_DTW_0.0/Model.pth", weights_only=False)
    wgan_gen.load_state_dict(ckpt['gen_state_dict'])
    wgan_critic.load_state_dict(ckpt['critic_state_dict'])
    if os.path.exists("../fine_tune_data.npy"):
        data = np.load("../fine_tune_data.npy", allow_pickle=True)
        segments = [item[0] for item in data]
        ecg_dataset = np.stack(segments)
        normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(
            ecg_dataset=ecg_dataset)
        labels = [item[1] for item in data]
    labels = np.array(labels)
    normalized_data = np.array(normalized_data)
    dataset_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long).unsqueeze(1)
    dataloader = DataLoader(TensorDataset(dataset_tensor, labels_tensor),
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    num_epochs = 50
    n_critic = 3
    lambda_gp = 10.0
    lambda_dtw = 0.0
    GAN_model_num = 0
    generator = Generator(ecg_length=ecg_length,
                          n_leads=n_leads, latent_dim=latent_dim).to(device)
    critic = Critic(ecg_length=ecg_length, n_leads=n_leads).to(device)
    generator = load_wgan_to_cwgan_generator(
        wgan_gen=wgan_gen, cwgan_gen=generator, latent_dim=latent_dim)
    critic = load_wgan_to_cwgan_critic(
        wgan_critic=wgan_critic, cwgan_critic=critic)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=[0.0, 0.9])
    c_optimier = optim.Adam(critic.parameters(), lr=1e-4, betas=[0.0, 0.9])
    while os.path.exists(f"CWGAN/images/DCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"):
        GAN_model_num += 1
    image_path = f"CWGAN/images/DCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"
    os.makedirs(image_path)
    GAN_model_num = 0
    while os.path.exists(f"CWGAN/models/DCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"):
        GAN_model_num += 1
    model_path = f"CWGAN/models/DCNN_WGAN/Model_{GAN_model_num}_GP_{lambda_gp}_DTW_{lambda_dtw}"
    os.makedirs(model_path)
    train(generator, critic, dataloader, num_epochs, latent_dim, n_critic,
          lambda_gp, lambda_dtw, g_optimizer, c_optimier, device, image_path, model_path, lead_maxs, lead_mins)
    print(f"Model saved to: {model_path}/Model.pth")


if __name__ == "__main__":
    main()
