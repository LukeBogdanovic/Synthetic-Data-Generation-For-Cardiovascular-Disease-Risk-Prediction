# GAN Models In Folder
All Generative Adversarial network models trialled in the generation of ECG waveforms for Bogdanovic et al.

## Wasserstein GAN (WGAN)
Clamping of values in the critic are used here to keep the Lipschitz continuity to 1-Lipschitz

### Loss functions of WGAN
Generator loss function uses the mean value of all possible samples $z$ from the latent noise probability distribution $\mathbb{P}_{z}$ after they have been given a score by the critic $D$, with the generated sample being scored labelled as $G(z)$.
$$L_{G} = - \mathbb{E}_{z \sim \mathbb{P}_{z}}[D(G(z))]$$

Critic loss function uses the mean value of all possible samples $z$ from the probability distribution $\mathbb{P}_{z}$ after they have been given a score by the critic $D$, with generated samples being scored labelled as $G(z)$. The mean value of all possible samples $x$ from the real dataset probability distribution $\mathbb{P}_{data}$ after they have been given a score by the critic $D$, is also used, with $x$ denoting a real sample. These two means are then subtracted from one another to get a final value for the critic loss value. 
$$L_{C} = \mathbb{E}_{z \sim \mathbb{P}_{z}}[D(G(z))] - \mathbb{E}_{x \sim \mathbb{P}_{data}}[D(x)]$$

## Wasserstein GAN with Gradient Penalty (WGAN-GP)

### Loss functions of WGAN-GP
$$L_{G} = - \mathbb{E}_{z \sim \mathbb{P}_{z}}[D(G(z))]$$

$$L_{C} = \mathbb{E}_{z \sim \mathbb{P}_{z}}[D(G(z))] - \mathbb{E}_{x \sim \mathbb{P}_{data}}[D(x)] + \lambda_{gp} * [(||\nabla_{\hat x}D(\hat x)||_{2} - 1)^{2}]$$

## Spectral normalization GAN (SW-GAN)