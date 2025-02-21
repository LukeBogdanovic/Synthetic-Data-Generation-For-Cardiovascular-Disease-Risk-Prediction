import tensorflow as tf
import numpy as np
import os
from keras import Input, Model
from keras.api.losses import BinaryCrossentropy
from keras.api.optimizers import Adam
from keras.api.layers import Bidirectional, TimeDistributed, LSTM, Dense, Flatten, Conv1D, Reshape, Dropout, LeakyReLU
import matplotlib.pyplot as plt
from gan_pretrain_preprocessing import reverse_ecg_normalization, normalize_ecg
from sklearn.preprocessing import MinMaxScaler
import time

plt.ion()

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
    subset_size = 20000
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
    x = Reshape((ecg_length, 32))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    out = TimeDistributed(Dense(n_leads, activation='tanh'))(x)
    generator = Model(
        inputs=noise_input,
        outputs=out,
        name='Generator'
    )
    return generator


def build_discriminator(ecg_length=128, n_leads=3) -> Model:
    ecg_input = Input(shape=(ecg_length, n_leads), name='ecg_input')
    x = Conv1D(64, kernel_size=3, strides=2, padding='same')(ecg_input)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(256, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    out = Dense(1)(x)
    discriminator = Model(
        inputs=ecg_input,
        outputs=out,
        name='Discriminator'
    )
    return discriminator


def train_gan(generator, discriminator, dataset, g_optimizer: Adam, d_optimizer: Adam, epochs=10, latent_dim=100, checkpoint_manager=None) -> None:

    @tf.function
    def train_step(real_ecg):
        batch_size = tf.shape(real_ecg)[0]
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            generated_ecg = generator(noise, training=True)
            real_output = discriminator(real_ecg, training=True)
            fake_output = discriminator(generated_ecg, training=True)
            g_loss = generator_loss(fake_output)
            d_loss = discriminator_loss(real_output, fake_output)
        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(
            zip(d_grads, discriminator.trainable_variables))
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(
            zip(g_grads, generator.trainable_variables))
        pred_real = tf.cast(tf.sigmoid(real_output) > 0.5, tf.int32)
        pred_fake = tf.cast(tf.sigmoid(fake_output) > 0.5, tf.int32)
        # True labels: real samples are 1, fake samples are 0
        true_real = tf.ones_like(pred_real)
        true_fake = tf.zeros_like(pred_fake)
        predictions = tf.concat([pred_real, pred_fake], axis=0)
        labels = tf.concat([true_real, true_fake], axis=0)
        return d_loss, g_loss, predictions, labels

    precision_metric = tf.keras.metrics.Precision(name='precision')
    recall_metric = tf.keras.metrics.Recall(name='recall')
    Leads = ['III', 'V3', 'V5']
    fig = plt.figure(figsize=(12, 6))
    for epoch in range(epochs):
        start = time.time()
        print(f"Epoch {epoch+1}/{epochs}")
        precision_metric.reset_state()
        recall_metric.reset_state()
        for step, real_ecg in enumerate(dataset):
            d_loss, g_loss, predictions, labels = train_step(real_ecg)
            precision_metric.update_state(labels, predictions)
            recall_metric.update_state(labels, predictions)
            if step % 10 == 0:
                precision_value = precision_metric.result().numpy()
                recall_value = recall_metric.result().numpy()
                f1_score = 2 * (precision_value * recall_value) / \
                    (precision_value + recall_value + 1e-7)
                print(f"Step {step}  |  Gen Loss: {g_loss.numpy():.4f}  |  Disc Loss: {d_loss.numpy():.4f}  |  Precision: {precision_value:.4f}  |  Recall: {recall_value:.4f}  |  F1 Score: {f1_score:.4f}  |  Time: {time.time()-start:.2f}s")
        sample_noise = tf.random.normal((1, latent_dim))
        fake_ecg = generator(sample_noise, training=False)
        ecg_fake_np = fake_ecg[0].numpy()
        ecg_fake_np = reverse_ecg_normalization(ecg_fake_np, m_scaler)
        plt.clf()
        for lead_idx in range(ecg_fake_np.shape[1]):
            ax = plt.subplot(ecg_fake_np.shape[1], 1, lead_idx + 1)
            ax.plot(ecg_fake_np[:, lead_idx])
            ax.set_title(
                f'Fake ECG - Lead {Leads[lead_idx]} (Epoch={epoch+1}, Step={step})')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.05)
        if checkpoint_manager is not None:
            ckpt_save_path = checkpoint_manager.save(checkpoint_number=epoch+1)
            print(f"Checkpoint saved at epoch {epoch+1}: {ckpt_save_path}")
    Model.save(generator, f"gan_scripts/gan/generator5.keras")
    Model.save(discriminator, f"gan_scripts/gan/discriminator5.keras")


BATCH_SIZE = 12
dataset = tf.data.Dataset.from_tensor_slices((normalized_data))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


latent_dim = 100
num_seconds = 10
generator = build_generator_unconditional(
    ecg_length=128*num_seconds, n_leads=3, latent_dim=latent_dim)
discriminator = build_discriminator(ecg_length=128*num_seconds, n_leads=3)
cross_entropy = BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


gen_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5)
disc_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5)


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
    epochs=500,
    latent_dim=latent_dim,
    checkpoint_manager=checkpoint_manager
)
