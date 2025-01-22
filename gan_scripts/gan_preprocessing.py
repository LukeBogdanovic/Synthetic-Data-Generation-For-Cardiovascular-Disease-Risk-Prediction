import os
import wfdb
import numpy as np
import pandas as pd
from scipy import signal
from wfdb import processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './'
CRFs_path = f'{path}'
ECG_path = f'{path}'
Model_path = f'{path}'

for root, dirs, files in os.walk(path):
    path = './'
    if 'data' in dirs:
        path = f"{path}data"
        for root2, dirs2, files2 in os.walk(path):
            if 'CRFs.csv' in files2:
                CRFs_path = f'{path}/CRFs.csv'
            if 'dataset' in dirs2:
                ECG_path = f'{path}/dataset'
            break
    path = './'
    if 'gan_scripts' in dirs:
        path = f'{path}gan_scripts'
        for root2, dirs2, files2 in os.walk(path):
            if 'gan' in dirs2:
                Model_path = f'{path}/gan'
                break
    break

ECG_names = sorted(os.listdir(f"{ECG_path}"))
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
    CRFs, c_bin_minmax_scaler = load_and_process_crf_data()
    idx = 0
    for ecgfilename in ECG_names:
        if ecgfilename.endswith(".dat"):
            ecgfilename = ecgfilename.strip(".dat")
            segment = __load_ecg_data(
                f"{ECG_path}/{ecgfilename}", segment_length=segment_length)
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
    return normalized_data, m_scaler, c_bin_minmax_scaler


def __load_ecg_data(filename, segment_length=5, include_all=True):
    x = wfdb.rdrecord(filename, sampfrom=20000,
                      sampto=20000+(128*60), channels=[0, 1, 2])
    data = np.asarray(x.p_signal, dtype=np.float64)
    fs = x.fs
    r_peaks = processing.xqrs_detect(sig=data[:, 0], fs=fs)

    seg_samples = int(segment_length * fs)
    half = seg_samples // 2

    centered_segments = []

    for r_peak in r_peaks:
        start = r_peak - half
        end = r_peak + half

        if start < 0 or end > len(data):
            if include_all:
                if start < 0:
                    start = 0
                    end = seg_samples
                elif end > len(data):
                    end = len(data)
                    start = end-seg_samples
                centered_segments.append(data[start:end])
            else:
                continue
        else:
            centered_segments.append(data[start:end])

    return np.concatenate(centered_segments)


def load_and_process_crf_data():
    CRFs = pd.read_csv(f"{CRFs_path}")
    CRFs = CRFs[~CRFs['Record'].isin(['02076', '02089', '02148', '02152'])]
    CRFs = CRFs.drop(columns=['Record', 'BSA', 'BMI', 'IMT MAX', 'LVMi', 'EF'])
    CRFs['Gender'] = CRFs['Gender'].str.upper().map({'M': 0, 'F': 1})
    CRFs['Smoker'] = CRFs['Smoker'].str.upper().map({'NO': 0, 'YES': 1})
    CRFs['Vascular event'] = CRFs['Vascular event'].str.lower().map(
        {'none': 0, 'myocardial infarction': 1, 'stroke': 2, 'syncope': 3})
    num_imputer = SimpleImputer(strategy='mean')
    CRFs[['SBP', 'DBP']] = num_imputer.fit_transform(CRFs[['SBP', 'DBP']])
    num_cols = ['Age', 'Weight', 'Height', 'SBP', 'DBP']
    c_bin_cols = ['Gender', 'Smoker']
    CRFs[num_cols] = scaler.fit_transform(CRFs[num_cols])
    c_bin_minmax_scaler = MinMaxScaler(feature_range=(
        0, 1)).fit(np.hstack([CRFs[c_bin_cols]]))
    CRFs[c_bin_cols] = c_bin_minmax_scaler.transform(CRFs[c_bin_cols])
    return CRFs, c_bin_minmax_scaler


def normalize_ecg(ecg, s_scaler: MinMaxScaler):
    return s_scaler.transform(ecg)


def reverse_crf_normalization(crf_8d, scaler: StandardScaler, c_bin_minmax_scaler: MinMaxScaler):
    """
    crf_8d: shape (N, 8) => [Gender, Age, Weight, Height, BSA, BMI, Smoker, SBP, DBP, VascularEvent]

    We'll do:
      - pick out the 5 numeric columns in the same order as 'num_cols'
      - inverse_transform them
      - recombine with the 3 columns that were never scaled
    """
    # Suppose your final order is exactly:
    # index 0 -> Gender        (not scaled)
    # index 1 -> Age           (scaled)
    # index 2 -> Weight        (scaled)
    # index 3 -> Height        (scaled)
    # index 4 -> Smoker        (not scaled)
    # index 5 -> SBP           (scaled)
    # index 6 -> DBP           (scaled)
    # index 7 -> VascularEvent (not scaled)

    # 1) Extract unscaled columns
    gender_col = crf_8d[0]  # shape (N,1)
    smoker_col = crf_8d[4]  # shape (N,1)
    vascular_event_col = np.round(crf_8d[7])  # shape (N,1)

    # 2) Extract numeric scaled columns in the order that matches 'num_cols'
    # num_cols = ['Age','Weight','Height','SBP','DBP','BSA','BMI']
    # We must map them to the correct indices in crf_10d
    # Age -> index=1
    # Weight -> index=2
    # Height -> index=3
    # SBP -> index=7
    # DBP -> index=8
    # BSA -> index=4
    # BMI -> index=5

    crf_scaled = [crf_8d[1], crf_8d[2], crf_8d[3],
                  crf_8d[5], crf_8d[6]]  # shape => (N, 5)

    # 3) Inverse transform the 7 numeric columns
    crf_unscaled_5 = scaler.inverse_transform([crf_scaled])
    # shape => (N,5)

    # 4) Re-insert them in correct order back to a final shape (N,8)
    # We'll build them as a list of columns in the final order
    # final order => [Gender, Age, Weight, Height, Smoker, SBP, DBP, VascularEvent]
    # We already have unscaled_7 in the order [Age,Weight,Height,SBP,DBP]

    # Let's break down crf_unscaled_5 into separate columns for readability
    age_col = np.round(crf_unscaled_5[0][0])
    weight_col = np.round(crf_unscaled_5[0][1])
    height_col = np.round(crf_unscaled_5[0][2])
    sbp_col = np.round(crf_unscaled_5[0][3])
    dbp_col = np.round(crf_unscaled_5[0][4])

    gen_smoke_vals = c_bin_minmax_scaler.inverse_transform(
        [[gender_col, smoker_col]])
    gender_col = np.round(gen_smoke_vals[0][0])
    smoker_col = np.round(gen_smoke_vals[0][1])
    # 5) Concatenate everything in final shape (N,8)
    # Make sure the dimension is correct for each
    final_crf = np.concatenate([
        [[gender_col]],   # (N,1)
        [[age_col]],      # (N,1)
        [[weight_col]],
        [[height_col]],
        [[smoker_col]],   # (N,1)
        [[sbp_col]],
        [[dbp_col]],
        [[vascular_event_col]]
    ], axis=0)
    final_crf = final_crf.reshape((1, 8))

    return final_crf


def reverse_crf_to_df(crf_8d, scaler, c_bin_minmax_scaler, col_names):
    arr = reverse_crf_normalization(crf_8d, scaler, c_bin_minmax_scaler)
    return pd.DataFrame(arr, columns=col_names)


def reverse_ecg_normalization(normalized_ecg, scaler: MinMaxScaler):
    return scaler.inverse_transform(normalized_ecg)


def split_crf(crf_8d):
    crf_5d_arr = []
    crf_3d_arr = []
    for arr in crf_8d:
        crf_5d = [arr[1], arr[2], arr[3], arr[5], arr[6]]
        crf_5d_arr.append(crf_5d)
        crf_3d = [arr[0], arr[4], arr[7]]
        crf_3d_arr.append(crf_3d)
    return crf_5d_arr, crf_3d_arr
