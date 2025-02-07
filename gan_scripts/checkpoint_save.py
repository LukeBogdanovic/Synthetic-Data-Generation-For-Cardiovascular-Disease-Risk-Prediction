import tensorflow as tf
from keras.api.layers import Input, Dense, Reshape, Bidirectional, LSTM, Dropout, TimeDistributed, LeakyReLU, BatchNormalization, Conv1D, Flatten
from keras.api import Model
from keras.api.optimizers import Adam


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
    x = Dropout(rate=0.4, name='Dropout')(x)
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
    x = Conv1D(filters=16, kernel_size=5, strides=2,
               padding='same', activation=LeakyReLU(negative_slope=0.2))(ecg_input)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=5, strides=2,
               padding='same', activation=LeakyReLU(negative_slope=0.2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(64, activation=LeakyReLU(negative_slope=0.2))(x)
    x = Dropout(0.4)(x)
    validity = Dense(1, activation='sigmoid', name='validity')(x)
    discriminator = Model(
        inputs=ecg_input,
        outputs=validity,
        name='Discriminator'
    )
    return discriminator


# Step 1: Recreate your models with the same architecture
latent_dim = 100
generator = build_generator_unconditional(
    ecg_length=128*5, n_leads=3, latent_dim=latent_dim)
discriminator = build_discriminator(ecg_length=128*5, n_leads=3)

# Create optimizers (if you are tracking them as part of the checkpoint)
gen_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
disc_optimizer = Adam(learning_rate=5e-5, beta_1=0.5)

# Step 2: Create a checkpoint object and manager.
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

# Restore the latest checkpoint if it exists.
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore("checkpoints/ckpt-1")

# Step 3: Save the complete model.
# Here we are saving the generator as a complete Keras model.
Model.save(generator, "generator.keras")
model_loaded: Model = tf.keras.models.load_model("generator.keras")
model_loaded.summary()
