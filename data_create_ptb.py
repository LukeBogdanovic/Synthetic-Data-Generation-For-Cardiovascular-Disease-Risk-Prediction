'''
:File: data_create_ptb.py
:Author: Luke Bogdanovic
:Date Updated: 12/03/2025
:Purpose: Creates dataset for training the WGAN using the PTB-XL dataset
'''
import pandas as pd
import wfdb
import numpy as np
from scipy.signal import resample
from gan_scripts.preprocessing_utils import bandpass_filter, extract_centered_segment_ptb
import ast
import os

fs_target = 128  # Target sampling rate

path = './'
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
                  for f in df.filename_lr]  # Read lower 100Hz sampling rate file from dataframe
    else:
        record = [wfdb.rdrecord(f'{path}/{f}', channel_names=['III', 'V3', 'V5'], physical=physical)
                  for f in df.filename_hr]  # Read higher 500Hz sampling rate file from dataframe
    # Add signals to numpy array
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
    :type bool:
    :param bool align_r_peak:
    :param int segment_length:

    :return NDArray[Any]:
    '''
    if physical:
        signal = record.p_signal.astype(np.float32)
    else:
        signal = record.d_signal.astype(np.float32)
    if not physical:  # Not used - Used if digital signals from recorder ADC is chosen
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
        # Resample to the target 5s @ 128Hz = 640 samples
        signal_resampled = resample(signal, fs_target*num_seconds)
    filtered_signal = bandpass_filter(
        signal_resampled, lowcut, highcut, record.fs, order=3)  # Bandpass filter
    if align_r_peak:  # Check for if r-peak alignment is wanted for centred r-peaks
        aligned_segments, validity = extract_centered_segment_ptb(
            filtered_signal, fs_target, segment_length)  # Extract the centred segments
        if not validity:
            return False
        return aligned_segments
    else:
        return filtered_signal


def get_normalized_values(num_seconds, physical=True):
    '''
    Gets the normalized per-lead ECG signals using the PTB-XL CSV file and 500 Hz signals

    :param num_seconds: Number of seconds to pull from record
    :param physical: Chooses phyical or digital signals
    '''
    Y = pd.read_csv(pretrain_path+'/ptbxl_database.csv',
                    index_col='ecg_id')  # Read CSV data
    # Convert strings into python dictionaries
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    X = read_record(Y, 500, pretrain_path, physical)  # Read the 500 Hz signals
    print(f"Number of records loaded: {len(X)}")
    processed_signals = []
    for _, rec in enumerate(X):
        processed = process_record(
            rec,  fs_target, num_seconds, physical=physical)  # Process record
        if type(processed) is bool:  # Check if boolean has been returned to indicate a failure
            continue
        processed_signals.append(processed)  # Add processed signal to list
    # Create numpy array using processed signals
    processed_signals_array = np.array(processed_signals)
    # Save as numpy file
    np.save("normalized_ecg_phys.npy", processed_signals_array)


get_normalized_values(5)
