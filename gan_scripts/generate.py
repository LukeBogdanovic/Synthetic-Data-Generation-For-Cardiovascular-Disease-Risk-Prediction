import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wfdb import processing
from tqdm import tqdm
from scipy import signal
from sklearn.impute import SimpleImputer
from keras import Model
from keras.api.saving import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler


ECG_names = sorted(os.listdir("../data/dataset"))
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
            centered_segments = __load_ecg_data(
                f"../data/dataset/{ecgfilename}")
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
        if end <= len(data):  # Ensure bounds
            centered_segments.append(data[start:end])
    return centered_segments


def load_and_process_crf_data():
    CRFs = pd.read_csv(f"../data/CRFs.csv")
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

norm_data = [ecg for ecg, _ in combined_data]


def reverse_crf_normalization(crf, scaler: StandardScaler, col_names):
    original_crf = scaler.inverse_transform(crf)
    return pd.DataFrame(original_crf, columns=col_names)


def reverse_ecg_normalization(normalized_ecg, scaler: MinMaxScaler):
    return scaler.inverse_transform(normalized_ecg)


model: Model = load_model(f"gan/generator.keras")
seconds_to_generate = 60
noise = np.random.normal(0, 1, (seconds_to_generate, 100))
gen_ecgs = model.predict(noise)

# Concatenate the generated segments
gen_ecgs_full = np.concatenate(gen_ecgs, axis=0)
gen_ecgs_full = reverse_ecg_normalization(gen_ecgs_full, m_scaler)
plt.figure(0, figsize=(12, 6))
for lead_idx in range(gen_ecgs_full.shape[1]):
    plt.subplot(3, 1, lead_idx + 1)
    plt.plot(gen_ecgs_full[:, lead_idx], label=f'Lead {lead_idx+1}')
    plt.title(f'Fake ECG - Lead {lead_idx+1}')
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
