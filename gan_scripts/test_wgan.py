import os
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.api.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from gan_pretrain_preprocessing import reverse_ecg_normalization, normalize_ecg
from keras.api.layers import Bidirectional, TimeDistributed, LSTM, Dense, Flatten, Conv1D, Reshape, Dropout, LeakyReLU, Layer
from keras.api.metrics import Mean
from keras.api.saving import register_keras_serializable
import matplotlib.pyplot as plt
from keras.api.callbacks import Callback
import ctypes

dtw_lib = ctypes.CDLL("gan_scripts/c_funcs/dtw.so")
dtw_lib.dtw_distance.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
dtw_lib.dtw_distance.restype = ctypes.c_double
dtw_lib.compute_mmd.argtypes = [ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_double
                                ]
dtw_lib.compute_mmd.restype = ctypes.c_double


tf.keras.mixed_precision.set_global_policy("mixed_float16")
# tf.config.optimizer.set_jit(True)  # Enable XLA

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
    x = Reshape((ecg_length, 32))(x)
    x = Conv1D(64, kernel_size=5, strides=1,
               padding='same', activation='relu')(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    out = TimeDistributed(Dense(n_leads, activation='tanh'))(x)
    generator = Model(
        inputs=noise_input,
        outputs=out,
        name='Generator'
    )
    return generator


def build_critic(ecg_length=128, n_leads=3) -> Model:
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
    x = MiniBatchDiscrimination(num_kernel=100, dim_kernel=5)(x)
    out = Dense(1)(x)
    critic = Model(
        inputs=ecg_input,
        outputs=out,
        name='Discriminator'
    )
    return critic


def compute_mmd(real_ecg, fake_ecg, sigma=1.0):
    if isinstance(real_ecg, tf.Tensor):
        real_ecg = tf.convert_to_tensor(real_ecg)
        real_ecg = real_ecg.numpy()
    if isinstance(fake_ecg, tf.Tensor):
        fake_ecg = tf.convert_to_tensor(fake_ecg)
        fake_ecg = fake_ecg.numpy()
    real_np = real_ecg.reshape(real_ecg.shape[0], -1).astype(np.float64)
    fake_np = fake_ecg.reshape(fake_ecg.shape[0], -1).astype(np.float64)
    batch_real, features = real_np.shape
    batch_fake, _ = fake_np.shape
    result = dtw_lib.compute_mmd(
        real_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fake_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        batch_real, batch_fake, features, sigma
    )
    return result


def compute_mvdTW(real_ecg, fake_ecg):
    real_np = real_ecg.numpy() if isinstance(real_ecg, tf.Tensor) else real_ecg
    fake_np = fake_ecg.numpy() if isinstance(fake_ecg, tf.Tensor) else fake_ecg
    batch_size = real_np.shape[0]
    ecg_length = real_np.shape[1]
    n_leads = real_np.shape[2]
    distances = []
    for i in range(batch_size):
        seq1 = real_np[i].flatten().astype(np.float64)
        seq2 = fake_np[i].flatten().astype(np.float64)
        dtw_distance_val = dtw_lib.dtw_distance(
            seq1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            seq2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ecg_length,
            ecg_length,
            n_leads
        )
        distances.append(dtw_distance_val)
    return np.mean(distances).astype(np.float32)


def mvDTW_loss(real_ecgs, fake_ecgs):
    return tf.numpy_function(compute_mvdTW, [real_ecgs, fake_ecgs], tf.float32)


@register_keras_serializable(package='Custom')
class WGANGP(Model):
    def __init__(self, generator: Model, critic: Model, latent_dim=100, n_critic=5, lambda_gp=10.0, lambda_dtw=1.0, **kwargs):
        super(WGANGP, self).__init__(**kwargs)
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lambda_dtw = lambda_dtw
        self.mmd_metric = Mean(name="mmd")
        # Multivariate dynamic time warping
        self.mvdTW_metric = Mean(name="mvdTW")

    def generate(self, noise):
        '''
        Function for large quantity generation of Multivariate ECG signals.\\
        Quantity should be greater than the batch size set for the during training model.

        :param:
        noise

        :return:

        '''
        return self.generator.predict(noise)

    def discriminate(self, ecg):
        '''
        Function for checking validity of validity of large quantity of Multivariate ECG signals.\\
        Quantity should be greater than the batch size set for the model during training.
        '''
        return self.critic.predict(ecg)

    def call_generate(self, noise, training=False):
        '''
        Function for creating single sample of Multivariate ECG signals.
        '''
        return self.generator(noise, training=training)

    def call_discriminate(self, ecg, training=False):
        '''
        Function for checking validity of single sample of Multivariate ECG signals.
        '''
        return self.critic(ecg, training=training)

    @property
    def metrics(self):
        return [self.mmd_metric, self.mvdTW_metric]

    def compile(self, g_optimizer: Adam, c_optimizer: Adam, **kwargs):
        super(WGANGP, self).compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer

    def get_config(self):
        config = super(WGANGP, self).get_config()
        config.update({
            "generator": self.generator,
            "critic": self.critic,
            "latent_dim": self.latent_dim,
            "n_critic": self.n_critic,
            "lambda_gp": self.lambda_gp,
            "lambda_dtw": self.lambda_dtw
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Loads the model from the saved configuration.
        """
        generator = config.pop("generator")
        critic = config.pop("critic")
        return cls(generator=generator, critic=critic, **config)

    def gradient_penalty(self, real_samples, fake_samples):
        batch_size = tf.shape(real_samples)[0]
        epsilon = tf.random.uniform(
            [batch_size, 1, 1], 0.0, 1.0, dtype=tf.float16)
        interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            interpolated_score = self.critic(interpolated, training=True)
        grads = tape.gradient(interpolated_score, [interpolated])[0]
        grads = tf.reshape(grads, [batch_size, -1])
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
        return penalty

    def train_step(self, real_ecg):
        batch_size = tf.shape(real_ecg)[0]
        for _ in range(self.n_critic):
            noise = tf.random.normal([batch_size, self.latent_dim])
            with tf.GradientTape() as tape:
                fake_ecg = self.generator(noise, training=True)
                total_critic_loss = self.critic_loss(real_ecg, fake_ecg)
            critic_gradients = tape.gradient(
                total_critic_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(critic_gradients, self.critic.trainable_variables))
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            fake_ecg = self.generator(noise, training=True)
            g_loss = self.generator_loss(real_ecg, fake_ecg)
        generator_gradients = tape.gradient(
            g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        mmd_value = tf.py_function(func=compute_mmd, inp=[
                                   real_ecg, fake_ecg], Tout=tf.float32)
        mvdTW_value = tf.py_function(func=compute_mvdTW, inp=[
                                     real_ecg, fake_ecg], Tout=tf.float32)
        mmd_value.set_shape([])
        mvdTW_value.set_shape([])
        self.mmd_metric.update_state(mmd_value)
        self.mvdTW_metric.update_state(mvdTW_value)
        return {"critic_loss": total_critic_loss, "generator_loss": g_loss, "mmd": self.mmd_metric.result(), "mvDTW": self.mvdTW_metric.result()}

    def generator_loss(self, real_ecgs, fake_ecgs):
        critic_fake = self.critic(fake_ecgs, training=True)
        g_loss = -tf.reduce_mean(critic_fake)
        dtw_loss = mvDTW_loss(real_ecgs, fake_ecgs)
        return tf.cast(g_loss, tf.float32) + self.lambda_dtw * dtw_loss

    def critic_loss(self, real_ecg, fake_ecg):
        critic_real = self.critic(real_ecg, training=True)
        critic_fake = self.critic(fake_ecg, training=True)
        critic_loss = tf.reduce_mean(
            critic_fake) - tf.reduce_mean(critic_real)
        gp = self.gradient_penalty(real_ecg, fake_ecg)
        return critic_loss + self.lambda_gp * gp


@register_keras_serializable(package='Custom')
class MiniBatchDiscrimination(Layer):
    def __init__(self, num_kernel, dim_kernel, kernel_initalizer='glorot_uniform', **kwargs):
        self.num_kernel = num_kernel
        self.dim_kernel = dim_kernel
        self.kernel_initializer = kernel_initalizer
        super(MiniBatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], self.num_kernel * self.dim_kernel),
            initializer=self.kernel_initializer,
            trainable=True
        )
        super(MiniBatchDiscrimination, self).build(input_shape)

    def call(self, x):
        activation = tf.matmul(x, self.kernel)
        activation = tf.reshape(
            activation, shape=(-1, self.num_kernel, self.dim_kernel))
        tmp1 = tf.expand_dims(activation, 3)
        tmp2 = tf.transpose(activation, perm=[1, 2, 0])
        tmp2 = tf.expand_dims(tmp2, 0)
        diff = tmp1 - tmp2
        l1 = tf.reduce_sum(tf.math.abs(diff), axis=2)
        features = tf.reduce_sum(tf.math.exp(-l1), axis=2)
        return tf.concat([x, features], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.num_kernel)


class SaveGeneratedECG(Callback):
    def __init__(self, generator, save_path="images/generated_images"):
        super(SaveGeneratedECG, self).__init__()
        self.generator = generator
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        noise = np.random.normal(0, 1, (1, 50))
        gen_ecg = self.generator.predict(
            noise, verbose=0).squeeze()
        gen_ecg = reverse_ecg_normalization(gen_ecg, m_scaler)
        if gen_ecg.ndim == 1:
            gen_ecg = gen_ecg[:, np.newaxis]
        num_leads = gen_ecg.shape[1]
        for lead in range(num_leads):
            plt.figure(figsize=(8, 4))
            plt.plot(gen_ecg[:, lead], linewidth=1.5, color='black')
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.title(f"Generated ECG - Lead {lead+1} - Epoch {epoch + 1}")
            plt.grid(True)
            plt.savefig(
                f"{self.save_path}/ecg_epoch_{epoch + 1}_lead_{lead + 1}.png", bbox_inches='tight', dpi=300)
            plt.close()


BATCH_SIZE = 80
AUTOTUNE = tf.data.AUTOTUNE
dataset = tf.data.Dataset.from_tensor_slices(normalized_data)
dataset = dataset.cache()  # Cache dataset in memory (if it fits)
dataset = dataset.map(lambda x: tf.cast(x, tf.float16),
                      num_parallel_calls=AUTOTUNE)
dataset = dataset.shuffle(buffer_size=min(10000, len(
    normalized_data)), reshuffle_each_iteration=True)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True,
                        num_parallel_calls=AUTOTUNE)
dataset = dataset.prefetch(AUTOTUNE)

latent_dim = 50
num_seconds = 5
n_critic = 3
lambda_gp = 10.0  # Gradient penalty
ecg_length = 128 * num_seconds
n_leads = 3

generator = build_generator_unconditional(
    ecg_length=ecg_length, n_leads=n_leads, latent_dim=latent_dim)
critic = build_critic(ecg_length=ecg_length, n_leads=n_leads)
g_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5)
c_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5)

wgangp = WGANGP(generator, critic, latent_dim=latent_dim,
                n_critic=n_critic, lambda_gp=lambda_gp)
wgangp.compile(g_optimizer=g_optimizer,
               c_optimizer=c_optimizer)

sample_data = next(iter(dataset))

save_ecg_callback = SaveGeneratedECG(wgangp.generator)

wgangp.build(input_shape=(None, ecg_length, n_leads))

wgangp.fit(dataset, epochs=100, callbacks=[save_ecg_callback])

wgangp.save("gan_scripts/gan/wgan.keras")
