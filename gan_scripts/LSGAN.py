import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.api.losses import BinaryCrossentropy
from keras.api.optimizers import Adam
from keras.api.layers import Bidirectional, TimeDistributed, LSTM, Dense, Flatten, Conv1D, Reshape, Dropout, MaxPool1D, Concatenate, Embedding
from gan_preprocessing import load_data, reverse_ecg_normalization, split_crf

segment_length = 2
combined_data, m_scaler, c_bin_minmax_scaler = load_data(
    segment_length=segment_length)
# Placeholder before trying to train for both ecg and crf
norm_data = [ecg for ecg, _ in combined_data]

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


def build_generator(ecg_length=128, n_leads=3, latent_dim=100, crf_n_dim=5, num_classes_gender=1, num_classes_smoker=1, num_classes_vasc=1) -> Model:
    '''
    Defines the generator network for the GAN.
    '''
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

    # CRF
    x_crf = Dense(64, activation='relu')(noise_input)
    gen_numeric_crf = Dense(
        crf_n_dim, activation='tanh', name='gen_crf')(x_crf)

    gen_gender_crf = Dense(
        num_classes_gender, activation='sigmoid', name='gen_gender')(x_crf)

    gen_smoker_crf = Dense(
        num_classes_smoker, activation='sigmoid', name='gen_smoker')(x_crf)

    gen_vasc_crf = Dense(
        num_classes_vasc, activation='linear', name='gen_vasc')(x_crf)

    generator = Model(
        inputs=noise_input,
        outputs=[gen_ecg, gen_numeric_crf,
                 gen_gender_crf, gen_smoker_crf, gen_vasc_crf],
        name='Generator'
    )

    return generator


def build_discriminator(ecg_length=128, n_leads=3, crf_n_dim=5, crf_gender_dim=1, crf_smoker_dim=1, crf_vasc_dim=1) -> Model:
    '''

    '''
    ecg_input = Input(shape=(ecg_length, n_leads), name='ecg_input')
    crf_n_input = Input(shape=(crf_n_dim,), name='numeric_input')
    crf_g_input = Input(shape=(crf_gender_dim,), name='gender_input')
    crf_s_input = Input(shape=(crf_smoker_dim,), name='smoker_input')
    crf_v_input = Input(shape=(crf_vasc_dim,),   name='vasc_input')

    x_ecg = Conv1D(filters=32, kernel_size=5, strides=2,
                   padding='same', activation='relu')(ecg_input)
    x_ecg = MaxPool1D(pool_size=2)(x_ecg)
    x_ecg = Conv1D(filters=64, kernel_size=5, strides=2, padding='same')(x_ecg)
    x_ecg = MaxPool1D(pool_size=2)(x_ecg)
    x_ecg = Flatten()(x_ecg)
    # Numeric CRF
    x_crf_n = Dense(32, activation='relu')(crf_n_input)
    # vasc crf
    vasc_embedding = Embedding(
        input_dim=4, output_dim=8, name='vasc_embedding')(crf_v_input)
    vasc_flat = Flatten()(vasc_embedding)

    # Categorical CRF
    x_crf_c_cat = Concatenate()([crf_g_input, crf_s_input, vasc_flat])
    x_crf_c_cat = Dense(16, activation='relu')(x_crf_c_cat)
    # Combined
    x_combined = Concatenate()([x_ecg, x_crf_n, x_crf_c_cat])
    x = Dense(64, activation='relu')(x_combined)
    x = Dropout(0.3)(x)

    validity = Dense(1, activation='sigmoid', name='validity')(x)

    discriminator = Model(
        inputs=[ecg_input, crf_n_input, crf_g_input, crf_s_input, crf_v_input],
        outputs=validity,
        name='Discriminator'
    )
    return discriminator


def train_gan(generator, discriminator, dataset, g_optimizer: Adam, d_optimizer: Adam, epochs=10, latent_dim=100, checkpoint_manager=None) -> None:
    '''

    '''
    g_optimizer = g_optimizer
    d_optimizer = d_optimizer
    bce_loss = BinaryCrossentropy()

    @tf.function
    def train_step(real_ecg, real_crf_n, real_crf_g, real_crf_s, real_crf_v):
        '''
        Defines the operation that occurs during each step of the training process of the GAN
        Args:

        '''
        batch_size = tf.shape(real_ecg)[0]
        noise = tf.random.normal((batch_size, latent_dim))
        fake_ecg, fake_crf_n, fake_crf_g, fake_crf_s, fake_crf_v = generator(
            noise, training=True)
        with tf.GradientTape() as d_tape:
            pred_real = discriminator(
                [real_ecg, real_crf_n, real_crf_g, real_crf_s, real_crf_v], training=True)
            pred_fake = discriminator(
                [fake_ecg, fake_crf_n, fake_crf_g, fake_crf_s, fake_crf_v], training=True)
            d_loss_real = bce_loss(tf.ones_like(pred_real), pred_real)
            d_loss_fake = bce_loss(tf.zeros_like(pred_fake), pred_fake)
            d_loss = d_loss_real + d_loss_fake
        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(
            zip(d_grads, discriminator.trainable_variables))
        noise2 = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as g_tape:
            fake_ecg2, fake_crf_n2, fake_crf_g2, fake_crf_s2, fake_crf_v2 = generator(
                noise2, training=True)
            pred_fake2 = discriminator(
                [fake_ecg2, fake_crf_n2, fake_crf_g2, fake_crf_s2, fake_crf_v2], training=True)
            g_loss = bce_loss(tf.ones_like(pred_fake2), pred_fake2)
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(
            zip(g_grads, generator.trainable_variables))
        return d_loss, g_loss

    Leads = ['III', 'V3', 'V5']
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(dataset):
            real_ecg, real_num_crf, real_gen_crf, real_smoker_crf, real_vasc_crf = batch
            d_loss, g_loss = train_step(
                real_ecg, real_num_crf, real_gen_crf, real_smoker_crf, real_vasc_crf)
            if step % 10 == 0:
                print(
                    f"  Step {step}: d_loss={d_loss.numpy():.4f}, g_loss={g_loss.numpy():.4f}")
            if step % 20 == 0:
                sample_noise = tf.random.normal((1, latent_dim))
                fake_ecg, fake_crf_n, fake_crf_g, fake_crf_s, fake_crf_v = generator(
                    sample_noise, training=False)
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


BATCH_SIZE = 32
dataset = tf.data.Dataset.from_tensor_slices(
    (ecg_array, num_crf_array, gen_crf_array, smoker_crf_array, vasc_crf_array))
dataset = dataset.shuffle(len(ecg_array)*5).batch(
    BATCH_SIZE, drop_remainder=True)


latent_dim = 100
generator = build_generator(
    ecg_length=128*segment_length, n_leads=3, latent_dim=latent_dim)
discriminator = build_discriminator(ecg_length=128*segment_length, n_leads=3)
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
    epochs=1500,
    latent_dim=latent_dim,
    checkpoint_manager=checkpoint_manager
)
