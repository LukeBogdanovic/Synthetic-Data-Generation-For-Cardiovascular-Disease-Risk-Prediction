import wfdb
import os
import numpy as np
import pandas as pd
import ast
from scipy.signal import resample
import wfdb.processing
from preprocessing import bandpass_filter, extract_centered_segment_ptb

fs_target = 128  # Target sampling frequency for resampling

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


def read_record(df: pd.DataFrame, sampling_rate: int, path: str, physical: bool):
    '''
    Reads a WFDB record from the given path and returns the 3 leads required for training.
    Can return either a physical or digital signal in the form of numpy array.

    :param DataFrame df:
    :param int sampling_rate:
    :param str path:
    :param bool physical:

    :return NDArray[Any] data:
    '''
    if sampling_rate == 100:
        record = [wfdb.rdrecord(f'{path}/{f}', channel_names=['III', 'V3', 'V5'], physical=physical)
                  for f in df.filename_lr]
    else:
        record = [wfdb.rdrecord(f'{path}/{f}', channel_names=['III', 'V3', 'V5'], physical=physical)
                  for f in df.filename_hr]
    data = np.array([signal for signal in record])
    return data


def process_record(record: wfdb.Record, fs_target: int = 128, num_seconds: int = 5, lowcut: float = 0.5, highcut: int = 40, physical: bool = True, align_r_peak: bool = False, segment_length: int = 128):
    '''
    Pre-processes the provided ECG record. \
    Performs resampling of the signal and bandpass filtering.\
    If the signal is a digital signal, the signal is quantized to\
    8-bit resolution in line with chosen dataset.

    :param Record record:
    :param int fs_target:
    :param int num_seconds:
    :param float lowcut:
    :param int highcut:
    :param bool physical:
    :param bool align_r_peak:
    :param int segment_length:

    :return NDArray[Any]:
    '''
    if physical:
        signal = record.p_signal.astype(np.float32)
    else:
        signal = record.d_signal.astype(np.float32)
    if not physical:
        s_min = np.min(signal)
        s_max = np.max(signal)
        if np.abs(s_max - s_min) < 1e-8:
            s_max = s_min + 1e-8
        signal_scaled = (signal - s_min) / (s_max - s_min) * 255.0
        signal_uint8 = np.round(signal_scaled).astype(np.uint8)
        signal_int8 = signal_uint8.astype(np.int16) - 128
        signal_int8 = np.clip(signal_int8, -128, 127).astype(np.int8)
        signal_resampled = resample(signal_int8, fs_target*num_seconds)
    else:
        signal_resampled = resample(signal, fs_target*num_seconds)
    filtered_signal = bandpass_filter(
        signal_resampled, lowcut, highcut, record.fs, order=3)
    if align_r_peak:
        aligned_segments, validity = extract_centered_segment(
            filtered_signal, fs_target, segment_length)
        if not validity:
            return False
        return aligned_segments
    else:
        return filtered_signal


def get_normalized_values(num_seconds, physical=True):
    Y = pd.read_csv(pretrain_path+'/ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    X = read_record(Y, 100, pretrain_path, physical)
    print(f"Number of records loaded: {len(X)}")
    processed_signals = []
    for _, rec in enumerate(X):
        processed = process_record(
            rec,  fs_target, num_seconds, physical=physical)
        if type(processed) is bool:
            continue
        processed_signals.append(processed)
    processed_signals_array = np.array(processed_signals)
    np.save("normalized_ecg_phys.npy", processed_signals_array)


# get_normalized_values(num_seconds=10)
