import wfdb.processing
from gan_scripts.preprocessing import bandpass_filter
import wfdb
import numpy as np
from pathlib import Path
import pandas as pd


# def remove_trailing_nans(ecg_data):
#     """
#     Remove trailing rows that are entirely NaN from a multi-lead ECG array.

#     Parameters:
#     -----------
#     ecg_data : np.ndarray
#         2D array of shape (num_samples, num_leads).

#     Returns:
#     --------
#     ecg_data_trimmed : np.ndarray
#         ECG array with trailing NaN rows removed.
#     last_valid_idx : int
#         Index of the last row that contains at least one non-NaN value.
#     """
#     # Check if every value in a row is NaN
#     all_nan_per_row = np.any(np.isnan(ecg_data), axis=1)

#     # Indices of rows that have at least one valid (non-NaN) entry
#     valid_indices = np.where(~all_nan_per_row)[0]

#     if valid_indices.size == 0:
#         # If the entire array is NaN, return the original or an empty array
#         # depending on what you prefer. Here, we return the original.
#         return ecg_data, -1

#     # Last valid row index
#     last_valid_idx = valid_indices[-1]

#     # Slice the array up to (and including) the last valid row
#     ecg_data_trimmed = ecg_data[:last_valid_idx+1, :]

#     return ecg_data_trimmed, last_valid_idx


# def extract_segments_from_record(filtered_signal, r_peaks, segment_length=640):
#     """
#     Extract segments centered on valid R-peaks.
#     """
#     half_seg = segment_length // 2
#     segments = []
#     for r in r_peaks:
#         # Only use peaks where a full segment can be extracted
#         if r >= half_seg and r <= len(filtered_signal) - half_seg:
#             segment = filtered_signal[r - half_seg: r + half_seg, :]
#             segments.append(segment)
#     return np.array(segments)


def get_longest_valid_interval(ecg_record):
    """
    Find the longest contiguous interval in the record that contains no NaN values.

    Parameters:
      ecg_record (np.ndarray): 2D array (samples x leads)

    Returns:
      (start_idx, end_idx): tuple of indices (inclusive) of the longest valid segment.
                            Returns (None, None) if no valid segment is found.
    """
    # Create a boolean mask: True if the sample (row) is valid across all leads.
    valid = ~np.isnan(ecg_record).any(axis=1)

    best_start, best_end = None, None
    current_start = None
    best_length = 0

    for i, is_valid in enumerate(valid):
        if is_valid:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                current_length = i - current_start
                if current_length > best_length:
                    best_length = current_length
                    best_start, best_end = current_start, i - 1
                current_start = None
    # Handle the tail end of the record
    if current_start is not None:
        current_length = len(valid) - current_start
        if current_length > best_length:
            best_start, best_end = current_start, len(valid) - 1

    return best_start, best_end


def extract_segments_centered_on_rpeaks(ecg_record, r_peaks, segment_length):
    """
    Extract segments from an ECG record centered on valid R-peaks.

    Parameters:
      ecg_record (np.ndarray): 2D array (samples x leads)
      r_peaks (array-like): Array or list of R-peak indices.
      segment_length (int): Desired segment length in samples.

    Returns:
      segments (list of np.ndarray): List of segments, each of shape (segment_length, num_leads)
    """
    half_seg = segment_length // 2

    # Get the longest valid interval in the record (NaN-free region).
    # This function should return the start and end indices of that interval.
    valid_start, valid_end = get_longest_valid_interval(ecg_record)

    segments = []
    for r in r_peaks:
        # Only consider R-peaks that are far enough from the edges
        # so that a full segment can be extracted.
        if r - half_seg >= valid_start and r + half_seg <= valid_end:
            segment = ecg_record[r - half_seg: r + half_seg, :]
            segments.append(segment)

    return segments


def read_record(ecg_signals, segment_length=640, vascular_event=0):
    if np.isnan(ecg_signals).any() or np.isinf(ecg_signals).any():
        print("Warning: Input signal contains NaN or inf values!")
        # ecg_signals, _ = remove_trailing_nans(ecg_signals)
    channel = ecg_signals[:, 0]
    r_peaks = wfdb.processing.gqrs_detect(sig=channel, fs=128)
    candidate_segments = extract_segments_centered_on_rpeaks(
        ecg_signals, r_peaks, segment_length)
    segments_array = np.array(candidate_segments)
    filtered_signals = bandpass_filter(
        segments_array, 0.5, 40, 128, order=3)

    # Decide how many segments to keep from this record based on the class.
    # These numbers are chosen such that total segments per class become 4026.
    # For example, if vascular_event==0 ('none') and there are 122 records, then 4026/122 = 33 segments per record.
    max_segments_per_record = [33, 366, 1342, 1342]
    desired_segments = max_segments_per_record[vascular_event]

    num_candidates = filtered_signals.shape[0]
    if num_candidates == 0:
        # If no valid segment is found, return an empty array.
        return  # np.empty((0, segment_length, filtered_signal.shape[1]))

    if num_candidates >= desired_segments:
        # Sample without replacement if you have more candidates than needed.
        indices = np.random.choice(
            num_candidates, desired_segments, replace=False)
    else:
        # If fewer, oversample (with replacement) to reach the desired count.
        indices = np.random.choice(
            num_candidates, desired_segments, replace=True)

    segments = filtered_signals[indices]
    return segments


def read_csv():
    data = pd.read_csv("./data/CRFs.csv")
    data['Vascular event'] = data['Vascular event'].str.lower().map(
        {'none': 0, 'myocardial infarction': 1, 'stroke': 2, 'syncope': 3})
    data_vasc_conds = dict(
        zip(data['Record'], data['Vascular event']))
    return data_vasc_conds


def get_unique_filenames_pathlib(folder_path):
    folder = Path(folder_path)
    files = {file.stem for file in folder.glob(
        "*") if file.is_file()}  # Using set comprehension
    return sorted(files)  # Sort the filenames


def read_data():
    files = get_unique_filenames_pathlib("./data/dataset/")
    data_vasc_conds = read_csv()
    data = []
    ecg_signals = []
    for file in files:
        file_key = int(file.lstrip('0'))
        vasc_condition = data_vasc_conds[file_key]
        record = wfdb.rdrecord(
            f"./data/dataset/{file}", sampfrom=768000, sampto=4000000)
        ecg_signal = record.p_signal.astype(np.float32)
        ecg_signals.append((ecg_signal, vasc_condition))
    for ecg in ecg_signals:
        ecg_record = ecg[0]
        vasc_condition = ecg[1]
        record = read_record(ecg_record, 640, vasc_condition)
        for val in record:
            # Create an object array explicitly
            val_cond = np.array((val, vasc_condition), dtype=object)
            data.append(val_cond)
    return data


def main():
    data = read_data()
    dataset = np.array(data, dtype=object)
    np.save("fine_tune_data2.npy", dataset, True)
    return


if __name__ == "__main__":
    main()
