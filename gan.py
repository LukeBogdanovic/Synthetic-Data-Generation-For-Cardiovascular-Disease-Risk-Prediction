import os
import wfdb
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal
from sklearn.impute import SimpleImputer
from keras import Input, Model
from keras.api.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.api.layers import Dense, Flatten, Conv1D, Concatenate, Reshape
from keras.api.metrics import Precision, Recall, Accuracy


# 
ECG_names = sorted(os.listdir("dataset"))


def downsample_ecg(ecg,samples=1536,leads=3):
    '''
    
    '''
    new_ecg = np.zeros((leads, samples))
    for i, j in enumerate(ecg):
        new_ecg[i,:] = signal.resample(j, samples)
    return new_ecg


def reconfig_data(data):
    '''
    Transposes the leads in the dataset from being rows to columns
    '''
    leads = data.shape[1]
    reconfigured_data = []
    for lead in range(leads):
        lead_data = data[:, lead]
        reconfigured_data.append(lead_data)
    reconfigured_data = np.array(reconfigured_data)
    return reconfigured_data


def load_data():
    '''
    
    '''
    combined_data = []
    CRFs = load_and_process_crf_data()
    idx = 0
    for ecgfilename in tqdm(ECG_names):
        if ecgfilename.endswith(".dat"):
            ecgfilename = ecgfilename.strip(".dat")
            data = __load_ecg_data(f"dataset/{ecgfilename}")[0]
            data = reconfig_data(data)
            ecg_downsampled = downsample_ecg(data)
            combined_data.append((ecg_downsampled, CRFs.iloc[idx].values))
            idx += 1
    return combined_data


def load_and_process_crf_data():
    '''
    
    '''
    CRFs = pd.read_csv(f"CRFs.csv")
    CRFs.drop(columns=['IMT MAX', 'LVMi', 'EF'])
    CRFs['Gender'] = CRFs['Gender'].str.upper().map({'M':0, 'F':1})
    CRFs['Smoker'] = CRFs['Smoker'].str.upper().map({'NO':0, 'YES': 1})
    CRFs['Vascular event'] = CRFs['Vascular event'].str.lower().map({'none': 0, 'myocardial infarction': 1, 'stroke': 2, 'syncope': 3})
    num_imputer = SimpleImputer(strategy='mean')
    CRFs[['SBP', 'DBP']] = num_imputer.fit_transform(CRFs[['SBP', 'DBP']])
    scaler = StandardScaler()
    num_cols = ['Age','Weight','Height','SBP','DBP','BSA','BMI']
    CRFs[num_cols] = scaler.fit_transform(CRFs[num_cols])
    CRFs.drop(columns=['Record'])
    return CRFs


def __load_ecg_data(filename):
    '''
    
    '''
    x = wfdb.rdrecord(filename, sampfrom=20000, sampto=20000+(128*60),channels=[0,1,2])
    data = np.asarray(x.p_signal, dtype=np.float64)
    new_file = f"{filename}.hea"
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


def build_generator(latent_dim=100,ecg_shape=(3, 1536), crf_dim=14):
    noise_input = Input(shape=(latent_dim,))
    input_condition = Input(shape=(crf_dim,))
    # Combined noise and condition to generate
    combined_input = Concatenate()([noise_input, input_condition])
    # ECG leads generation
    x_noise = Dense(256,activation='relu')(combined_input)
    x_noise = Dense(ecg_shape[0] * ecg_shape[1], activation='relu')(x_noise)
    x_noise = Reshape((ecg_shape[0], ecg_shape[1]))(x_noise)
    x_noise = Conv1D(128,kernel_size=15, activation='relu', padding='same')(x_noise)
    x_noise = Conv1D(64, kernel_size=15, activation='relu', padding='same')(x_noise)
    gen_ecg = Conv1D(1536, kernel_size=15, activation='tanh', padding='same')(x_noise)
    # Clinical risk factors generation
    x_crf = Dense(128, activation='relu')(combined_input)
    gen_crf = Dense(crf_dim, activation='tanh')(x_crf)
    # Return model with inputs and outputs along with name
    return Model([noise_input, input_condition], [gen_ecg, gen_crf], name="Generator")


def build_discriminator(ecg_shape=(3,1536), crf_dim=14):
    ecg_input = Input(shape=ecg_shape)
    crf_input = Input(shape=(crf_dim,))
    input_condition = Input(shape=(crf_dim,))

    combined_input = Concatenate()([crf_input, input_condition])

    x_ecg = Conv1D(64,kernel_size=15, activation='relu', strides=2, padding='same')(ecg_input)
    x_ecg = Conv1D(128, kernel_size=15, activation='relu', strides=2, padding='same')(x_ecg)
    x_ecg = Flatten()(x_ecg)

    x_crf = Dense(128, activation='relu')(combined_input)

    combined = Concatenate()([x_ecg, x_crf])
    combined = Dense(128, activation='relu')(combined)
    x_combined = Dense(1, activation='sigmoid')(combined)
    
    return Model([ecg_input, crf_input, input_condition], x_combined, name="Discriminator")


def train_gan(dataset, epochs=1000, batch_size=16, latent_dim=100):
    '''
    GAN training function. Builds the discriminator and generator networks.
    Compiles both networks, combined gen & discriminator network and discriminator network,
    sets discriminator to not trainable.

    Trains on the provided pre-processed dataset for the given number of epochs
    '''
    generator = build_generator()
    discriminator = build_discriminator()

    noise = Input(shape=(latent_dim,))
    real_crf = Input(shape=(14,))
    gen_ecg, gen_crf = generator([noise, real_crf])
    validity = discriminator([gen_ecg, gen_crf, real_crf])
    combined = Model([noise, real_crf], validity)
    combined.compile(optimizer=Adam(1e-4, beta_1=0.5), loss='binary_crossentropy', metrics=[Accuracy(), Precision(), Recall()])
    discriminator.compile(optimizer=Adam(1e-4, beta_1=0.5), loss='binary_crossentropy', metrics=[Accuracy(), Precision(), Recall()])
    discriminator.trainable = False
    # Training loop
    for epoch in range(epochs):
        idxs = np.random.randint(0, len(dataset), batch_size)
        real_ecgs = []
        real_crfs = []
        for i in idxs:
            ecg_data, crf_data = dataset[i]
            real_crfs.append(crf_data)
            real_ecgs.append(ecg_data)
        real_ecgs = np.asarray(real_ecgs).astype("float32")
        real_crfs = np.asarray(real_crfs).astype("float32")

        noise = np.random.normal(0,1,(batch_size,latent_dim))
        gen_ecgs, gen_crfs = generator.predict([noise, real_crfs]) # Generates sample for discriminator
        d_loss_real = discriminator.train_on_batch([real_ecgs,real_crfs, real_crfs], np.ones((batch_size,1)))
        d_loss_fake = discriminator.train_on_batch([gen_ecgs,gen_crfs, real_crfs], np.zeros((batch_size, 1)))
        noise = np.random.normal(0,1,(batch_size, latent_dim))
        g_loss = combined.train_on_batch([noise, real_crfs], np.ones((batch_size,1)))

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | D Loss Real: {d_loss_real} | D Loss Fake: {d_loss_fake} | G Loss: {g_loss}")
    
    combined.save(f"gan/Model.keras")


def save_model():
    '''
    Saves the generator model to allow for generation of new data
    '''
    pass


def main():
    combined_data = load_data()
    train_gan(dataset=combined_data)


if __name__ == "__main__":
    main()