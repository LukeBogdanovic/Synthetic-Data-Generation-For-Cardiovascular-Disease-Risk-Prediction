import os
import tensorflow as tf
from keras import Model
from keras.api.optimizers import Adam
from keras.api.losses import BinaryCrossentropy
from keras.api.layers import Input, Dense, Reshape, Bidirectional, LSTM, Dropout, TimeDistributed, Conv1D, MaxPool1D, Flatten, Concatenate
from gan_preprocessing import load_data


# def build_generator(ecg_length=128, n_leads=3, latent_dim=100, crf_dim=10) -> Model:
#     '''
#     Defines the generator network for the GAN.
#     '''
#     noise_input = Input(shape=(latent_dim,), name='Noise_input')

#     x = Dense(ecg_length * 32, activation='relu')(noise_input)
#     x = Reshape((ecg_length, 32), name='reshape_for_lstm')(x)

#     x = Bidirectional(
#         LSTM(units=100, return_sequences=True),
#         name='BiLSTM1'
#     )(x)

#     x = Bidirectional(
#         LSTM(units=100, return_sequences=True),
#         name='BiLSTM2'
#     )(x)

#     x = Dropout(rate=0.5, name='Dropout')(x)

#     gen_ecg = TimeDistributed(Dense(n_leads, activation='tanh'),
#                               name='gen_ecg')(x)

#     x_crf = Dense(64, activation='relu')(noise_input)
#     gen_crf = Dense(crf_dim, activation='tanh', name='gen_crf')(x_crf)

#     generator = Model(inputs=noise_input, outputs=[
#                       gen_ecg, gen_crf], name='Generator')

#     return generator


# def build_discriminator(ecg_length=128, n_leads=3, crf_dim=10) -> Model:
#     '''

#     '''
#     ecg_input = Input(shape=(ecg_length, n_leads), name='ecg_input')
#     crf_input = Input(shape=(crf_dim,), name='crf_input')

#     x_ecg = Conv1D(filters=32, kernel_size=5, strides=2,
#                    padding='same', activation='relu')(ecg_input)
#     x_ecg = MaxPool1D(pool_size=2)(x_ecg)
#     x_ecg = Conv1D(filters=64, kernel_size=5, strides=2, padding='same')(x_ecg)
#     x_ecg = MaxPool1D(pool_size=2)(x_ecg)
#     x_ecg = Flatten()(x_ecg)

#     x_crf = Dense(16, activation='relu')(crf_input)

#     x = Concatenate()([x_ecg, x_crf])

#     x = Dense(64, activation='relu')(x_ecg)
#     x = Dropout(0.3)(x)

#     validity = Dense(1, activation='sigmoid', name='validity')(x)

#     discriminator = Model(
#         inputs=[ecg_input, crf_input], outputs=validity, name='Discriminator')
#     return discriminator


# generator = build_generator()
# discriminator = build_discriminator()
# gen_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
# disc_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)

# checkpoint = tf.train.Checkpoint(generator=generator, gen_optimizer=gen_optimizer,
#                                  discriminator=discriminator, disc_optimizer=disc_optimizer)

# checkpoint_dir = "./checkpoints"
# ckpt_manager = tf.train.CheckpointManager(
#     checkpoint, checkpoint_dir, max_to_keep=5)

# if ckpt_manager.latest_checkpoint:
#     checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
#     print(f"Restored From: {ckpt_manager.latest_checkpoint}")
# else:
#     print("No checkpoint found. Nothing to restore")

# generator.save("gan_scripts/gan/generator2.keras")

import numpy as np
from gan_preprocessing import split_crf

combined_data, m_scaler = load_data(segment_length=2)

ecg_list = []
crf_list = []

for (ecg, crf) in combined_data:
    ecg_list.append(ecg)  # each ecg should be shape (128, 3)
    crf_list.append(crf)  # each crf should be shape (10,)

# Convert lists to NumPy arrays
ecg_array = np.stack(ecg_list, axis=0)  # shape => (N, 128, 3)
crf_array = np.stack(crf_list, axis=0)  # shape => (N, 10)

crf5d, crf3d = split_crf(crf_array)
num_crf_array = np.stack(crf5d, axis=0)

gen_list = []
smoker_list = []
vasc_list = []

for (gen, smoker, vasc) in crf3d:
    gen_list.append(gen)
    smoker_list.append(smoker)
    vasc_list.append(vasc)

gen_crf_array = np.stack(gen_list, axis=0)
smoker_crf_array = np.stack(smoker_list, axis=0)
vasc_crf_array = np.stack(vasc_list, axis=0)

print("ecg_array shape:", ecg_array.shape)
print("num_crf_array shape:", num_crf_array.shape)
print("gen_crf_array shape:", gen_crf_array.shape)
print("smoker_crf_array shape:", smoker_crf_array.shape)
print("vasc_crf_array shape:", vasc_crf_array.shape)


print()
