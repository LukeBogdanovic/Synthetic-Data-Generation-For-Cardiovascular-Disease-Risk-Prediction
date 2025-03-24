'''
:File: data_create.py
:Author: Luke Bogdanovic
:Date Updated: 12/03/2025
:Purpose: Creates dataset for training the CWGAN using the SHAREEDB dataset
'''
import wfdb.processing
from gan_scripts.preprocessing_utils import bandpass_filter
import wfdb
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def get_longest_valid_interval(ecg_record):
    """
    Find the longest contiguous interval in the record that contains no NaN values.

    Parameters:
      ecg_record (np.ndarray): 2D array (samples x leads)

    Returns:
      (start_idx, end_idx): tuple of indices (inclusive) of the longest valid segment.
                            Returns (None, None) if no valid segment is found.
    """
    valid = ~np.isnan(ecg_record).any(
        axis=1)  # Create boolean mask indicating which rows have no NaN values
    # Initialise variables to store the longest valid segment
    best_start, best_end = None, None
    current_start = None
    best_length = 0
    # Iterate through the mask to identify valid segments
    for i, is_valid in enumerate(valid):
        if is_valid:
            if current_start is None:  # Start of new valid segment
                current_start = i
        else:
            # End of a valid segment
            if current_start is not None:
                current_length = i - current_start
                if current_length > best_length:
                    best_length = current_length
                    best_start, best_end = current_start, i - 1
                current_start = None  # Reset segment tracker
    if current_start is not None:  # Handle last segment continuing to end of the array.
        current_length = len(valid) - current_start
        if current_length > best_length:
            best_start, best_end = current_start, len(valid) - 1
    # return start and end best indices for longest valid segment.
    return best_start, best_end


def extract_segments_centered_on_rpeaks(ecg_record, r_peaks, segment_length):
    """
    Extract segments from an ECG record centered on valid R-peaks.


    :param ecg_record (np.ndarray): 2D array (samples x leads)
    :param r_peaks (array-like): Array or list of R-peak indices.
    :param segment_length (int): Desired segment length in samples.

    Returns:
      segments (list of np.ndarray): List of segments, each of shape (segment_length, num_leads)
    """
    half_seg = segment_length // 2
    # Get the longest valid interval in the record
    valid_start, valid_end = get_longest_valid_interval(ecg_record)
    segments = []
    for r in r_peaks:
        # Only consider R-peaks that are far enough from the edges so that a full segment can be extracted
        if r - half_seg >= valid_start and r + half_seg <= valid_end:
            segment = ecg_record[r - half_seg: r + half_seg, :]
            segments.append(segment)
            break
    return segments


def read_record(ecg_signals, segment_length=640, vascular_event=0):
    '''
    :param ecg_signals: The ecg_signals to read from and process
    :param segment_length: Length of the required segment
    :param vascular_event: The class of vascular event
    '''
    if np.isnan(ecg_signals).any() or np.isinf(ecg_signals).any():  # Check for infinite/NaN values in signals
        print("Warning: Input signal contains NaN or inf values!")
    channel = ecg_signals[:, 0]  # Use first channel/lead
    r_peaks = wfdb.processing.gqrs_detect(
        sig=channel, fs=128)  # Detect r-peaks
    candidate_segments = extract_segments_centered_on_rpeaks(
        ecg_signals, r_peaks, segment_length)  # Extract all usable segments centred on an r-peak
    segments_array = np.array(candidate_segments).squeeze(
        0)  # Squeeze the first dimension to remove it
    filtered_signals = bandpass_filter(
        segments_array, 0.5, 40, 128, order=3)  # Bandpass filter the ECG signals in the segments array
    # Number of segments allowed per record LCM(121,11,3)
    max_segments_per_record = [33, 366, 1342, 1342]
    # Get number of samples to get from record.
    desired_segments = max_segments_per_record[vascular_event]
    filtered_signals = np.expand_dims(filtered_signals, axis=0)
    num_candidates = filtered_signals.shape[0]  # Get the number of candidates
    if num_candidates == 0:  # Check if number of candidates is 0
        return
    # If there are more or equal candidates use random choice and remove previously used sample
    if num_candidates >= desired_segments:
        indices = np.random.choice(
            num_candidates, desired_segments, replace=False)
    else:
        indices = np.random.choice(
            num_candidates, desired_segments, replace=True)  # If there are less candidates use random choice and replace the previously used sample.
    segments = filtered_signals[indices]
    return segments


def read_csv():
    '''
    Reads data from the CSV file and preprocesses it before being used in real dataset.

    Returns:
        crf_dict: Dictionary of the CRF values
    '''
    # Read the CRF CSV file
    data = pd.read_csv("./data/CRFs.csv")
    # Make sure Gender is uppercase for consistency
    data['Gender'] = data['Gender'].str.upper()
    # Define the continuous and categorical columns you want to transform
    continuous_cols = ['Age', 'Weight', 'Height', 'SBP', 'DBP']
    categorical_cols = ['Smoker', 'Gender']
    # Convert Record column to integer
    data['Record'] = data['Record'].astype(int)
    # Initialize the scalers/encoders
    scaler = StandardScaler()
    encoder = OrdinalEncoder()
    # Scale the continuous columns
    data_cont_scaled = scaler.fit_transform(data[continuous_cols])
    # Convert back to DataFrame with original column names
    data_cont_scaled = pd.DataFrame(
        data_cont_scaled, columns=continuous_cols, index=data.index)
    # Encode the categorical columns
    data_cat_encoded = encoder.fit_transform(data[categorical_cols])
    data_cat_encoded = pd.DataFrame(
        data_cat_encoded, columns=categorical_cols, index=data.index)
    # Update the original DataFrame with the transformed columns
    data[continuous_cols] = data[continuous_cols].astype(float)
    data.update(data_cont_scaled)  # Inplace update of the continuous values
    data.update(data_cat_encoded)  # Inplace update of the categorical values
    # Drop unused features
    data = data.drop(columns=['BSA', 'BMI', 'IMT MAX', 'EF', 'LVMi'])
    data['Vascular event'] = data['Vascular event'].str.lower().map(
        {'none': 0, 'myocardial infarction': 1, 'stroke': 2, 'syncope': 3})  # Map vascular event to class number
    data = data.dropna()  # Drop rows with NaN
    # Set the Record column as the index so that you can retrieve CRF features based on Record
    data.set_index("Record", inplace=True)
    # Convert the DataFrame to a dictionary with Record numbers as keys
    crf_dict = data.to_dict(orient="index")
    return crf_dict


def get_unique_filenames_pathlib(folder_path):
    '''
    Gets all sorted unique filenames in a directory

    :param folder_path:

    Returns:
    '''
    folder = Path(folder_path)  # Create a path for the specified folder path
    files = {file.stem for file in folder.glob(
        "*") if file.is_file()}  # Get all unique filenames in the specified folder path.
    return sorted(files)  # Sort the files for return


def read_data():
    '''
    Reads and process the data for both ECGs and CRFs. Returns them in separated lists.

    Returns:
        (ecg_segments, crf_data): NDArray of ECG segments and NDArray of CRFs in a tuple
    '''
    files = get_unique_filenames_pathlib(
        "./data/dataset/")  # Get ECG files for the dataset
    crf_dict = read_csv()  # Read the csv file
    crf_data = []
    ecg_segments = []
    for file in files:
        # Remove the leading zero on the filename
        file_key = int(file.lstrip('0'))
        if file_key not in crf_dict:  # Check if
            print(f"Warning: Record {file_key} not found in CRFs data.")
            continue
        crf_info = crf_dict[file_key]
        record = wfdb.rdrecord(
            f"./data/dataset/{file}", sampfrom=899968, sampto=899968+(128*60))  # Read 60 seconds from the sample
        ecg_signal = record.p_signal.astype(
            np.float32)  # Get physical signal as float32
        # Read and process the ECG record
        segments = read_record(ecg_signal, 640, crf_info)
        # Check if the signal produced any segments
        if segments is None or len(segments) == 0:
            continue
        for segment in segments:  # Split ecg segments and crf info separate lists
            ecg_segments.append(segment)
            crf_data.append(crf_info)
    return np.array(ecg_segments), np.array(crf_data)


def main():
    ecg, crf = read_data()  # Collect ECGs and CRFs
    np.save("real_ecg.npy", ecg, True)  # Save the ECGs in a numpy file
    np.save("real_crf.npy", crf, True)  # Save the CRFs in a numpy file
    return


if __name__ == "__main__":
    main()
