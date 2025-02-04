import tensorflow as tf
import numpy as np
import os
from keras import Input, Model
from keras.api.losses import BinaryCrossentropy
from keras.api.optimizers import Adam
from keras.api.layers import Bidirectional, TimeDistributed, LSTM, Dense, Flatten, Conv1D, Reshape, Dropout, MaxPool1D, BatchNormalization, LeakyReLU
import matplotlib.pyplot as plt
from gan_pretrain_preprocessing import reverse_ecg_normalization, normalize_ecg
from sklearn.preprocessing import MinMaxScaler

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if os.path.exists("normalized_ecg.npy"):
    data = np.load("normalized_ecg.npy", allow_pickle=True)
    n_records = len(data)
    subset_size = 10000
    indices_all = np.arange(n_records)
    indices = np.random.choice(indices_all, subset_size, replace=False)
    subset_ecg_dataset = [data[i] for i in indices]
    m_scaler = MinMaxScaler(feature_range=(-1, 1)
                            ).fit(np.vstack(subset_ecg_dataset))
    normalized_data = []
    for ecg in subset_ecg_dataset:
        ecg_normalized = normalize_ecg(ecg, m_scaler)
        normalized_data.append(ecg_normalized)
else:
    from gan_pretrain_preprocessing import normalized_data


def build_generator_unconditional(ecg_length=128, n_leads=3, latent_dim=100) -> Model:
    noise_input = Input(shape=(latent_dim,), name='Noise_input')
    x = Dense(ecg_length * 32, activation='relu')(noise_input)
    x = Reshape((ecg_length, 32), name='reshape_for_lstm')(x)
    x = Bidirectional(
        LSTM(units=100, return_sequences=True),
        name='BiLSTM1'
    )(x)
    x = Bidirectional(
        LSTM(units=100, return_sequences=True),
        name='BiLSTM2'
    )(x)
    x = Dropout(rate=0.5, name='Dropout')(x)
    gen_ecg = TimeDistributed(Dense(n_leads, activation='tanh'),
                              name='gen_ecg')(x)
    generator = Model(
        inputs=noise_input,
        outputs=gen_ecg,
        name='Generator'
    )
    return generator


def build_discriminator(ecg_length=128, n_leads=3) -> Model:
    ecg_input = Input(shape=(ecg_length, n_leads), name='ecg_input')
    x_ecg = Conv1D(filters=16, kernel_size=5, strides=2,
                   padding='same', activation='relu')(ecg_input)
    x_ecg = MaxPool1D(pool_size=2)(x_ecg)
    x_ecg = Conv1D(filters=32, kernel_size=5, strides=2, padding='same')(x_ecg)
    x_ecg = MaxPool1D(pool_size=2)(x_ecg)
    x_ecg = Flatten()(x_ecg)
    x = Dense(64, activation='relu')(x_ecg)
    x = Dropout(0.3)(x)
    validity = Dense(1, activation='sigmoid', name='validity')(x)
    discriminator = Model(
        inputs=ecg_input,
        outputs=validity,
        name='Discriminator'
    )
    return discriminator


def train_gan(generator, discriminator, dataset, g_optimizer: Adam, d_optimizer: Adam, epochs=10, latent_dim=100, checkpoint_manager=None) -> None:
    bce_loss = BinaryCrossentropy()

    @tf.function
    def train_step(real_ecg):
        batch_size = tf.shape(real_ecg)[0]
        noise = tf.random.normal((batch_size, latent_dim))
        fake_ecg = generator(noise, training=True)
        with tf.GradientTape(persistent=True) as d_tape:
            pred_real = discriminator(real_ecg, training=True)
            pred_fake = discriminator(fake_ecg, training=True)
            d_loss_real = bce_loss(tf.ones_like(pred_real)*0.9, pred_real)
            d_loss_fake = bce_loss(tf.zeros_like(pred_fake)*0.1, pred_fake)
            d_loss = d_loss_real + d_loss_fake
        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(
            zip(d_grads, discriminator.trainable_variables))
        noise2 = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape(persistent=True) as g_tape:
            fake_ecg2 = generator(noise2, training=True)
            pred_fake2 = discriminator(fake_ecg2, training=True)
            g_loss = bce_loss(tf.ones_like(pred_fake2), pred_fake2)
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(
            zip(g_grads, generator.trainable_variables))
        return d_loss, g_loss

    Leads = ['III', 'V3', 'V5']
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        for step, real_ecg in enumerate(dataset):
            d_loss, g_loss = train_step(real_ecg)
            if step % 10 == 0:
                print(
                    f"  Step {step}: d_loss={d_loss.numpy():.4f}, g_loss={g_loss.numpy():.4f}")
            if step % 50 == 0:
                sample_noise = tf.random.normal((1, latent_dim))
                fake_ecg = generator(sample_noise, training=False)
                ecg_fake_np = fake_ecg[0].numpy()
                ecg_fake_np = reverse_ecg_normalization(ecg_fake_np, m_scaler)
                if plt.fignum_exists(0):
                    plt.close(0)
                plt.figure(0, figsize=(12, 6))
                for lead_idx in range(ecg_fake_np.shape[1]):
                    plt.subplot(ecg_fake_np.shape[1], 1, lead_idx + 1)
                    plt.plot(ecg_fake_np[:, lead_idx])
                    plt.title(
                        f'Fake ECG - Lead {Leads[lead_idx]} (Epoch={epoch}, Step={step})')
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.001)
        if checkpoint_manager is not None:
            ckpt_save_path = checkpoint_manager.save(checkpoint_number=epoch)
            print(f"Checkpoint saved at epoch {epoch}: {ckpt_save_path}")
    Model.save(generator, f"gan_scripts/gan/generator3.keras")
    Model.save(discriminator, f"gan_scripts/gan/discriminator3.keras")


BATCH_SIZE = 24
dataset = tf.data.Dataset.from_tensor_slices((normalized_data))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


latent_dim = 100
generator = build_generator_unconditional(
    ecg_length=128*5, n_leads=3, latent_dim=latent_dim)
discriminator = build_discriminator(ecg_length=128*5, n_leads=3)
gen_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
disc_optimizer = Adam(learning_rate=5e-5, beta_1=0.5)


checkpoint_dir = "./checkpoints"
checkpoint = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer
)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint,
    checkpoint_dir,
    max_to_keep=5
)
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print(f"Checkpoint restored from: {checkpoint_manager.latest_checkpoint}")


train_gan(
    generator=generator,
    discriminator=discriminator,
    dataset=dataset,
    g_optimizer=gen_optimizer,
    d_optimizer=disc_optimizer,
    epochs=50,
    latent_dim=latent_dim,
    checkpoint_manager=checkpoint_manager
)
