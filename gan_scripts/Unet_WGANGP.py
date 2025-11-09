import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

"""
Minimal, single-file WGAN-GP for 1D signals (ECG-ready).
- Tiny 1D U-Net generator (2 scales, simple skips)
- Simple 1D discriminator (no spectral norm by default)
- WGAN-GP loss + example train step

Target shape example: 3 leads, 5 s @ 500 Hz => (B, 3, 7500)
"""

# -----------------------------
# Tiny 1D U‑Net Generator
# -----------------------------


class GeneratorUNetTiny(nn.Module):
    def __init__(self, out_ch=3, z_ch=32, base=64):
        super().__init__()
        self.z_ch = z_ch
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(z_ch, base, 7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base, base, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Down1
        self.down1 = nn.Sequential(
            nn.Conv1d(base, base*2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base*2, base*2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Down2 (bottleneck in/out)
        self.down2 = nn.Sequential(
            nn.Conv1d(base*2, base*4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base*4, base*4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Up2
        self.up2 = nn.ConvTranspose1d(base*4, base*2, 4, stride=2, padding=1)
        self.post_up2 = nn.Sequential(
            nn.Conv1d(base*4, base*2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Up1
        self.up1 = nn.ConvTranspose1d(base*2, base, 4, stride=2, padding=1)
        self.post_up1 = nn.Sequential(
            nn.Conv1d(base*2, base, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Head
        self.head = nn.Sequential(
            nn.Conv1d(base, base//2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base//2, out_ch, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, z):
        # z: (B, z_ch, T)
        x0 = self.stem(z)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        u2 = self.up2(x2)
        # match lengths if off by 1
        if u2.size(-1) != x1.size(-1):
            u2 = F.pad(u2, (0, x1.size(-1) - u2.size(-1)))
        u2 = self.post_up2(torch.cat([u2, x1], dim=1))
        u1 = self.up1(u2)
        if u1.size(-1) != x0.size(-1):
            u1 = F.pad(u1, (0, x0.size(-1) - u1.size(-1)))
        u1 = self.post_up1(torch.cat([u1, x0], dim=1))
        out = self.head(u1)
        return out


# -----------------------------
# Simple 1D Discriminator (Critic)
# -----------------------------
class DiscriminatorTiny(nn.Module):
    def __init__(self, in_ch=3, base=64, num_layers=5):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, base, 7, padding=3), nn.LeakyReLU(
                0.2, inplace=True)
        ]
        ch = base
        for _ in range(num_layers-1):
            layers += [
                nn.Conv1d(ch, ch*2, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            ch *= 2
        self.features = nn.Sequential(*layers)
        self.final = nn.Linear(ch, 1)

    def forward(self, x):
        f = self.features(x)        # (B, C, T')
        f = f.mean(dim=-1)          # global mean pool → (B, C)
        out = self.final(f)         # (B, 1)
        return out.squeeze(-1)


# -----------------------------
# WGAN‑GP utilities
# -----------------------------

def gradient_penalty(D, real, fake, lambda_gp=10.0):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, device=real.device)
    x_hat = eps*real + (1-eps)*fake
    x_hat.requires_grad_(True)
    d_hat = D(x_hat)
    grads = autograd.grad(
        outputs=d_hat.sum(), inputs=x_hat,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1.0)**2).mean() * lambda_gp
    return gp


# -----------------------------
# One training step each
# -----------------------------
@torch.no_grad()
def sample_noise(batch, length, z_ch=32, device="cuda"):
    return torch.randn(batch, z_ch, length, device=device)


def d_step(D, G, real, opt_d, lambda_gp=10.0):
    D.train()
    G.train()
    B, C, T = real.shape
    z = sample_noise(B, T, z_ch=G.z_ch, device=real.device)
    fake = G(z).detach()

    d_real = D(real).mean()
    d_fake = D(fake).mean()
    gp = gradient_penalty(D, real, fake, lambda_gp=lambda_gp)

    loss_d = -(d_real - d_fake) + gp
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()

    return {
        'loss_d': float(loss_d.detach().cpu()),
        'd_real': float(d_real.detach().cpu()),
        'd_fake': float(d_fake.detach().cpu()),
        'gp': float(gp.detach().cpu()),
    }


def g_step(D, G, batch_size, length, opt_g, device):
    D.train()
    G.train()
    z = sample_noise(batch_size, length, z_ch=G.z_ch, device=device)
    fake = G(z)
    d_fake = D(fake).mean()
    loss_g = -d_fake
    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_g.step()
    return {'loss_g': float(loss_g.detach().cpu())}


# -----------------------------
# Quick start / smoke test
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH, LEADS, T = 4, 3, 7500

    G = GeneratorUNetTiny(out_ch=LEADS, z_ch=32, base=64).to(device)
    D = DiscriminatorTiny(in_ch=LEADS, base=64, num_layers=5).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Dummy real batch (replace with your normalized ECG windows)
    real = torch.randn(BATCH, LEADS, T, device=device)

    # Typical WGAN‑GP schedule: several D steps per G step
    for i in range(3):
        log_d = d_step(D, G, real, opt_d, lambda_gp=10.0)
    log_g = g_step(D, G, BATCH, T, opt_g, device)

    print({**log_d, **log_g})

    with torch.no_grad():
        z = sample_noise(8, T, z_ch=G.z_ch, device=device)
        fake = G(z).cpu()
        print("Fake shape:", fake.shape)  # (8, 3, 7500)
