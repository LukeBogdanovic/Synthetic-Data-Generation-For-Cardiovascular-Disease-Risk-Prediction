import os
import wfdb
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from wfdb import processing
from tqdm import tqdm
from scipy import signal
from sklearn.impute import SimpleImputer
from keras import Input, Model
from keras.api.losses import BinaryCrossentropy
from keras.api.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.api.layers import Bidirectional, TimeDistributed, LSTM, Dense, Flatten, Conv1D, Reshape, Dropout, MaxPool1D
from keras.api.metrics import Precision, Recall, Accuracy


ECG_names = sorted(os.listdir("dataset"))
ECG_names = [name for name in ECG_names if not any(
    exclude in name for exclude in ['02076', '02089', '02148', '02152'])]
scaler = StandardScaler()
all_ecgs = []


def downsample_ecg(ecg, samples=128):
    time_len, n_leads = ecg.shape
    new_ecg = np.zeros((samples, n_leads))
    for lead_idx in range(n_leads):
        lead_data = ecg[:, lead_idx]
        new_ecg[:, lead_idx] = signal.resample(lead_data, samples)
    return new_ecg


def load_data(segment_length):
    combined_data = []
    CRFs = load_and_process_crf_data()
    idx = 0
    for ecgfilename in tqdm(ECG_names):
        if ecgfilename.endswith(".dat"):
            ecgfilename = ecgfilename.strip(".dat")
            centered_segments = __load_ecg_data(f"dataset/{ecgfilename}")
            for segment in centered_segments:
                ecg_downsampled = downsample_ecg(
                    segment, samples=128*segment_length)
                all_ecgs.append(ecg_downsampled)
                combined_data.append((ecg_downsampled, CRFs.iloc[idx].values))
            idx += 1
    m_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.vstack(all_ecgs))
    normalized_data = []
    for ecg, crf in combined_data:
        ecg_normalized = normalize_ecg(ecg, m_scaler)
        normalized_data.append((ecg_normalized, crf))
    return normalized_data, m_scaler


def __load_ecg_data(filename, target_fs=100, segment_length=5):
    x = wfdb.rdrecord(filename, sampfrom=20000,
                      sampto=20000+(128*60), channels=[0, 1, 2])
    data = np.asarray(x.p_signal, dtype=np.float64)
    r_peaks = processing.xqrs_detect(sig=data[:, 0], fs=x.fs)
    centered_segments = []
    segment_samples = segment_length * x.fs
    for r_peak in r_peaks:
        start = max(0, r_peak - segment_samples // 2)
        end = start + segment_samples
        if end <= len(data):
            centered_segments.append(data[start:end])
    return centered_segments


def load_and_process_crf_data():
    CRFs = pd.read_csv(f"CRFs.csv")
    CRFs = CRFs[~CRFs['Record'].isin(['02076', '02089', '02148', '02152'])]
    CRFs = CRFs.drop(columns=['Record', 'IMT MAX', 'LVMi', 'EF'])
    CRFs['Gender'] = CRFs['Gender'].str.upper().map({'M': 0, 'F': 1})
    CRFs['Smoker'] = CRFs['Smoker'].str.upper().map({'NO': 0, 'YES': 1})
    CRFs['Vascular event'] = CRFs['Vascular event'].str.lower().map(
        {'none': 0, 'myocardial infarction': 1, 'stroke': 2, 'syncope': 3})
    num_imputer = SimpleImputer(strategy='mean')
    CRFs[['SBP', 'DBP']] = num_imputer.fit_transform(CRFs[['SBP', 'DBP']])
    num_cols = ['Age', 'Weight', 'Height', 'SBP', 'DBP', 'BSA', 'BMI']
    CRFs[num_cols] = scaler.fit_transform(CRFs[num_cols])
    return CRFs


def normalize_ecg(ecg, s_scaler: MinMaxScaler):
    return s_scaler.transform(ecg)


combined_data, m_scaler = load_data(segment_length=1)
# Placeholder before trying to train for both ecg and crf
norm_data = [ecg for ecg, _ in combined_data]


def reverse_crf_normalization(crf, scaler: StandardScaler, col_names):
    original_crf = scaler.inverse_transform(crf)
    return pd.DataFrame(original_crf, columns=col_names)


def reverse_ecg_normalization(normalized_ecg, scaler: MinMaxScaler):
    return scaler.inverse_transform(normalized_ecg)


def build_generator(ecg_length=128, n_leads=3, latent_dim=100, crf_dim=10):
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

    x = TimeDistributed(Dense(n_leads, activation='tanh'),
                        name='Output_Layer')(x)

    generator = Model(inputs=noise_input, outputs=x, name='Generator')

    return generator


def build_discriminator(ecg_length=128, n_leads=3, crf_dim=10):
    ecg_input = Input(shape=(ecg_length, n_leads), name='ecg_input')

    x_ecg = Conv1D(filters=32, kernel_size=5, strides=2,
                   padding='same', activation='relu')(ecg_input)
    x_ecg = MaxPool1D(pool_size=2)(x_ecg)
    x_ecg = Conv1D(filters=64, kernel_size=5, strides=2, padding='same')(x_ecg)
    x_ecg = MaxPool1D(pool_size=2)(x_ecg)
    x_ecg = Flatten()(x_ecg)

    x = Dense(64, activation='relu')(x_ecg)
    x = Dropout(0.3)(x)

    validity = Dense(1, activation='sigmoid', name='validity')(x)

    discriminator = Model(ecg_input, validity, name='Discriminator')
    return discriminator


def train_gan(generator, discriminator, dataset, g_optimizer, d_optimizer, epochs=10, latent_dim=100, checkpoint_manager=None):
    g_optimizer = g_optimizer
    d_optimizer = d_optimizer
    bce_loss = BinaryCrossentropy()

    @tf.function
    def train_step(real_ecg):
        batch_size = tf.shape(real_ecg)[0]
        noise = tf.random.normal((batch_size, latent_dim))
        fake_ecg = generator(noise, training=True)
        with tf.GradientTape() as d_tape:
            pred_real = discriminator(real_ecg, training=True)
            pred_fake = discriminator(fake_ecg, training=True)
            d_loss_real = bce_loss(tf.ones_like(pred_real), pred_real)
            d_loss_fake = bce_loss(tf.zeros_like(pred_fake), pred_fake)
            d_loss = d_loss_real + d_loss_fake
        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(
            zip(d_grads, discriminator.trainable_variables))
        noise2 = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as g_tape:
            fake_ecg2 = generator(noise2, training=True)
            pred_fake2 = discriminator(fake_ecg2, training=True)
            g_loss = bce_loss(tf.ones_like(pred_fake2), pred_fake2)
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(
            zip(g_grads, generator.trainable_variables))
        return d_loss, g_loss

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, batch_ecg in enumerate(dataset):
            d_loss, g_loss = train_step(batch_ecg)
            if step % 10 == 0:
                print(
                    f"  Step {step}: d_loss={d_loss.numpy():.4f}, g_loss={g_loss.numpy():.4f}")
            if step % 20 == 0:
                sample_noise = tf.random.normal((1, latent_dim))
                ecg_fake = generator(sample_noise, training=False)
                ecg_fake_np = ecg_fake[0].numpy()
                ecg_fake_np = reverse_ecg_normalization(ecg_fake_np, m_scaler)
                if plt.fignum_exists(0):
                    plt.close(0)
                plt.figure(0, figsize=(12, 6))
                for lead_idx in range(ecg_fake_np.shape[1]):
                    plt.subplot(ecg_fake_np.shape[1], 1, lead_idx + 1)
                    plt.plot(ecg_fake_np[:, lead_idx],
                             label=f'Lead {lead_idx+1}')
                    plt.title(
                        f'Fake ECG - Lead {lead_idx+1} (Epoch={epoch+1}, Step={step})')
                    plt.legend(loc='upper right')
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.001)
        if checkpoint_manager is not None:
            ckpt_save_path = checkpoint_manager.save(checkpoint_number=epoch)
            print(f"Checkpoint saved at epoch {epoch}: {ckpt_save_path}")
    Model.save(generator, f"gan/generator.keras")
    Model.save(discriminator, f"gan/discriminator.keras")


BATCH_SIZE = 32
dataset = tf.data.Dataset.from_tensor_slices(norm_data)
dataset = dataset.shuffle(270).batch(BATCH_SIZE, drop_remainder=True)


latent_dim = 100
generator = build_generator(ecg_length=128, n_leads=3, latent_dim=latent_dim)
discriminator = build_discriminator(ecg_length=128, n_leads=3)
gen_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
disc_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)


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
