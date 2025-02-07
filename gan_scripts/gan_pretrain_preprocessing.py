import wfdb
import os
import numpy as np
import pandas as pd
import ast
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler

fs_target = 128

path = './'
CRFs_path = f'{path}'
ECG_path = f'{path}'
Model_path = f'{path}'
pretrain_path = f'{path}'

for root, dirs, files in os.walk(path):
    path = './'
    if 'data' in dirs:
        path = f"{path}data"
        for root2, dirs2, files2 in os.walk(path):
            if 'CRFs.csv' in files2:
                CRFs_path = f'{path}/CRFs.csv'
            if 'dataset' in dirs2:
                ECG_path = f'{path}/dataset'
            if 'pretrain' in dirs2:
                pretrain_path = f'{path}/pretrain'
            break
    path = './'
    if 'gan_scripts' in dirs:
        path = f'{path}gan_scripts'
        for root2, dirs2, files2 in os.walk(path):
            if 'gan' in dirs2:
                Model_path = f'{path}/gan'
                break
    break


def read_record(df, sampling_rate, path: str):
    if sampling_rate == 100:
        record = [wfdb.rdrecord(f'{path}/{f}', channel_names=['III', 'V3', 'V5'], physical=False)
                  for f in df.filename_lr]
    else:
        record = [wfdb.rdrecord(f'{path}/{f}', channel_names=['III', 'V3', 'V5'], physical=False)
                  for f in df.filename_hr]
    data = np.array([signal for signal in record])
    return data


def process_record(record, fs_target=128):
    signal = record.d_signal
    s_min = np.min(signal)
    s_max = np.max(signal)
    if np.abs(s_max - s_min) < 1e-8:
        s_max = s_min + 1e-8
    signal_scaled = (signal - s_min) / (s_max - s_min) * 255.0
    signal_uint8 = np.round(signal_scaled).astype(np.uint8)
    signal_int8 = signal_uint8.astype(np.int16) - 128
    signal_int8 = np.clip(signal_int8, -128, 127).astype(np.int8)
    signal_resampled = resample(signal_int8, fs_target*10)
    return signal_resampled


def normalize_ecg(ecg, s_scaler: MinMaxScaler):
    return s_scaler.transform(ecg)


def reverse_ecg_normalization(normalized_ecg, scaler: MinMaxScaler):
    return scaler.inverse_transform(normalized_ecg)


def get_normalized_values():
    Y = pd.read_csv(pretrain_path+'/ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    X = read_record(Y, 100, pretrain_path)
    print(f"Number of records loaded: {len(X)}")
    processed_signals = []
    for idx, rec in enumerate(X):
        processed = process_record(rec, fs_target)
        processed_signals.append(processed)

    processed_signals_array = np.array(processed_signals)

    np.save("normalized_ecg.npy", processed_signals_array)


# get_normalized_values()
