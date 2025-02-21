import wfdb
import os
import numpy as np
import pandas as pd
import ast
from scipy.signal import resample, filtfilt, butter
from sklearn.preprocessing import MinMaxScaler
import wfdb.processing

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


def normalize_ecg(ecg, s_scaler: MinMaxScaler):
    '''
    Normalizes the provided ecg record using the provided SKlearn MinMaxScaler.

    :param
    '''
    return s_scaler.transform(ecg)


def reverse_ecg_normalization(normalized_ecg, scaler: MinMaxScaler):
    return scaler.inverse_transform(normalized_ecg)


def get_normalized_values(num_seconds, physical=True):
    Y = pd.read_csv(pretrain_path+'/ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    X = read_record(Y, 100, pretrain_path, physical)
    print(f"Number of records loaded: {len(X)}")
    processed_signals = []
    for _, rec in enumerate(X):
        processed = process_record(
            rec,  fs_target, num_seconds, physical=physical)
        processed_signals.append(processed)
    processed_signals_array = np.array(processed_signals)
    np.save("normalized_ecg_phys.npy", processed_signals_array)


def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    '''
    Bandpass filters the provided signal between the provided lowcut point\
    and highcut point using a butterworth IIR filter.\
    Returns the filtered signal.

    :param:
    :param float lowcut:
    :param int highcut:
    :param int fs:
    :param int order:

    :return filtered_signal: 
    '''
    nyquist = 0.5 * fs  # Get Nyquist frequency for the signal
    low = lowcut / nyquist  # Normalize the lowcut value using the nyquist
    high = highcut / nyquist  # Normalize the highcut value using the nyquist
    # Get b and a coefficients of the filter
    b, a = butter(order, [low, high], btype='band')
    # Filter the signal using the b and a coefficients
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal


def extract_centered_segment(ecg_signal, fs=128, segment_length=640):
    """
    Extracts a 5-second segment (640 samples at 128 Hz) from a 10-second ECG signal (1280 samples)
    such that an R-peak is centered in the middle of the segment.

    :param ecg_signal: Numpy array of shape (1280,) or (1280, n_channels) representing the ECG signal.
    :param fs: Sampling frequency, default is 128 Hz.
    :param segment_length: Length of the segment in samples, default is 640 (5 seconds at 128 Hz).
    :return: Tuple (segment, selected_peak) where segment is the 640-sample signal 
             with the R-peak centered and selected_peak is the index of the R-peak.
    """
    # Use the first channel if signal has multiple channels.
    if ecg_signal.ndim > 1:
        channel = ecg_signal[:, 0]
    else:
        channel = ecg_signal

    # Detect R-peaks using wfdb's gqrs detector.
    r_peaks = wfdb.processing.gqrs_detect(sig=channel, fs=fs)

    # 320 samples; R-peak should be at index 320 in the segment.
    half_seg = segment_length // 2

    # Only consider R-peaks that allow a full segment extraction:
    valid_r_peaks = [r for r in r_peaks if r >=
                     half_seg and r <= len(channel) - half_seg]

    if not valid_r_peaks:
        return ecg_signal, False

    # The ideal segment center (in the full 10-second signal) is at sample 640.
    center_of_signal = len(channel) // 2  # 1280/2 = 640

    # Select the valid R-peak closest to the center of the 10-second signal.
    selected_peak = min(valid_r_peaks, key=lambda r: abs(r - center_of_signal))

    # Extract the segment with the selected R-peak in the center.
    start = selected_peak - half_seg
    end = selected_peak + half_seg
    if ecg_signal.ndim > 1:
        segment = ecg_signal[start:end, :]
    else:
        segment = ecg_signal[start:end]

    return segment, True

# get_normalized_values(num_seconds=10)
