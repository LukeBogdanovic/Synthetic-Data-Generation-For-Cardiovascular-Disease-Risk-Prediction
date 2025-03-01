from scipy.signal import butter, filtfilt
import wfdb
import numpy as np


def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    '''
    Bandpass filters the provided signal between the provided lowcut point\
    and highcut point using a butterworth IIR filter.\
    Returns the filtered signal.

    :param ndarray signal:
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


def extract_centered_segment_ptb(ecg_signal, fs=128, segment_length=640):
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


def per_lead_minmax_scaling(ecg_dataset, feature_range=(-1, 1)):
    """
    :param ecg_dataset: np.ndarray of shape (N, L, n_leads)
    :param feature_range: tuple (min_val, max_val) for scaling, e.g., (-1, 1) or (0, 1)

    Returns:
        scaled_ecg: scaled dataset of shape (N, L, n_leads)
        lead_mins: list of shape (n_leads,) storing each lead's min
        lead_maxs: list of shape (n_leads,) storing each lead's max
    """
    min_val, max_val = feature_range
    n_leads = ecg_dataset.shape[2]
    # Initialize arrays to store the global min/max for each lead
    lead_mins = np.zeros(n_leads)
    lead_maxs = np.zeros(n_leads)
    # Compute global min and max for each lead across the entire dataset
    for lead_idx in range(n_leads):
        # Flatten the lead across N * L
        lead_data = ecg_dataset[:, :, lead_idx].reshape(-1)
        lead_mins[lead_idx] = lead_data.min()
        lead_maxs[lead_idx] = lead_data.max()
    # Scale each lead independently to the desired feature_range
    scaled_ecg = np.copy(ecg_dataset)
    for lead_idx in range(n_leads):
        # Avoid division by zero in case of a constant lead
        denom = (lead_maxs[lead_idx] - lead_mins[lead_idx])
        if denom == 0:
            denom = 1e-12
        # Scale to [0, 1] first
        scaled_ecg[:, :, lead_idx] = (
            (scaled_ecg[:, :, lead_idx] - lead_mins[lead_idx]) / denom
        )
        # Then scale to [min_val, max_val]
        scaled_ecg[:, :, lead_idx] = (
            scaled_ecg[:, :, lead_idx] * (max_val - min_val) + min_val
        )
    return scaled_ecg, lead_mins, lead_maxs


def per_lead_inverse_scaling(ecg_dataset_scaled, lead_mins, lead_maxs, feature_range=(-1, 1)):
    """
    ecg_dataset_scaled: np.ndarray of shape (N, L, n_leads) previously scaled
    lead_mins, lead_maxs: arrays from per_lead_minmax_scaling
    feature_range: tuple (min_val, max_val) used during scaling

    Returns:
        unscaled_ecg: original scale dataset
    """
    min_val, max_val = feature_range
    n_leads = ecg_dataset_scaled.shape[2]
    unscaled_ecg = np.copy(ecg_dataset_scaled)
    for lead_idx in range(n_leads):
        denom = (lead_maxs[lead_idx] - lead_mins[lead_idx])
        if denom == 0:
            denom = 1e-12
        # Inverse scale from [min_val, max_val] back to [0, 1]
        unscaled_ecg[:, :, lead_idx] = (
            (unscaled_ecg[:, :, lead_idx] - min_val) / (max_val - min_val)
        )
        # Then revert to original amplitude range
        unscaled_ecg[:, :, lead_idx] = (
            unscaled_ecg[:, :, lead_idx] * denom + lead_mins[lead_idx]
        )
    return unscaled_ecg
