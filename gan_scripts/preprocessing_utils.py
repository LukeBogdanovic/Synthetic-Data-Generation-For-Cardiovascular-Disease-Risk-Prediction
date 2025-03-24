'''
:File: preprocessing_utils.py
:Author: Luke Bogdanovic
:Date: 12/3/2025
:Purpose: Holds functions used for pre-processing, post-processing and utility across
          model files created.
'''
from scipy.signal import butter, filtfilt
import numpy as np
from wfdb.processing import gqrs_detect
import torch
import torch.nn.functional as F
from torchmetrics import Recall, Precision, F1Score, Accuracy, ConfusionMatrix
import ctypes
import random
import matplotlib.pyplot as plt
import os


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
    r_peaks = gqrs_detect(sig=channel, fs=fs)
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
    if ecg_signal.ndim > 1:  # Check if the signal is single or multi lead
        segment = ecg_signal[start:end, :]
    else:
        segment = ecg_signal[start:end]
    return segment, True  # Return true if valid r_peaks have been found


def per_lead_minmax_scaling(ecg_dataset, feature_range=(-1, 1)):
    """
    Performs minmax scaling on the provided lead of ecg data within
    a set feature range. Range should generally correspond to the
    output values possible by the output layer activation function.
    Default value is for (-1, 1), or a tanh activation function.

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
            denom = 1e-12  # Prevent division by 0
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
    Performs inverse minmax scaling on the ECG lead provided within the set feature range.
    The calculated lead minimums and maximums calculated when performing per lead
    minmax scaling are required to reverse the scaling.

    :param ecg_dataset_scaled: np.ndarray of shape (N, L, n_leads) previously scaled
    :param lead_mins: array from per_lead_minmax_scaling
    :param lead_maxs: array from per_lead_minmax_scaling
    :param feature_range: tuple (min_val, max_val) used during scaling

    Returns:
        unscaled_ecg: original scale dataset
    """
    min_val, max_val = feature_range  # Get min and max value of feature range
    n_leads = ecg_dataset_scaled.shape[2]  # Get the number of leads in ECG
    # Copy the array to new variable for alteration
    unscaled_ecg = np.copy(ecg_dataset_scaled)
    # Perform inverse scaling for each lead
    for lead_idx in range(n_leads):
        # Find range of values for the lead
        denom = (lead_maxs[lead_idx] - lead_mins[lead_idx])
        if denom == 0:
            denom = 1e-12  # Prevent multiplication by 0
        # Inverse scale from [min_val, max_val] back to [0, 1]
        unscaled_ecg[:, :, lead_idx] = (
            (unscaled_ecg[:, :, lead_idx] - min_val) / (max_val - min_val)
        )
        # Then revert to original amplitude range
        unscaled_ecg[:, :, lead_idx] = (
            unscaled_ecg[:, :, lead_idx] * denom + lead_mins[lead_idx]
        )
    return unscaled_ecg


def reorder_features(row):
    '''
    Reorders the features in the real tabular dataset to have all inputs
    across both synthetic and real data match.

    :param row: NDarray of 7 features from tabular dataset

    Returns:
        new_row: NDarray of reordered tabular features
    '''
    # Make sure the row has at least 5 elements
    if len(row) < 5:
        raise ValueError("Each row must have at least 5 elements.")
    # Remove the first (index 0) and the fifth (index 4) element from the row
    remaining = np.delete(row, [0, 4])
    # Append the original 5th element and then the original 1st element
    new_row = np.concatenate([remaining, row[4:5], row[0:1]])
    return new_row


def evaluate_on_test(classifier, testloader, device, num_classes=4):
    '''
    Evaluates the performance of the classifier on the data in the
    testing set provided by the testloader. Calculates and returns
    metrics for the testing set using the testing set dataloader on
    a chosen device.

    :param classifier: Classifier used for predictions
    :param testloader: The data loader for the testing set
    :param device: Device to send data/model to
    :param num_classes: Number of classes possible to be predicted

    Returns:
        Metrics_dict: Dictionary with calculated metrics

    '''
    classifier.eval()  # Set model to evaluation mode
    # Initialise arrays for holding predictions, labels, probabilities
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []
    with torch.no_grad():  # Disable gradient calculations
        for ecg, risk, labels in testloader:  # Loop through the test set data
            ecg = ecg.to(device)  # Send ECG to chosen device
            risk = risk.to(device)  # Send risk factors to chosen device
            labels = labels.to(device)  # Send labels to chosen device
            logits = classifier(ecg, risk)  # Use model for inference
            # Find class with highest logit
            preds = torch.argmax(logits, dim=1)
            # Convert logits to class probabilities
            probs = F.softmax(logits, dim=1)
            # Store all predictions, labels, probabilities in lists
            all_test_preds.append(preds)
            all_test_labels.append(labels)
            all_test_probs.append(probs)
    # Concatenate all values in each list to each other
    # Create single torch tensor
    all_test_preds = torch.cat(all_test_preds)
    all_test_labels = torch.cat(all_test_labels)
    all_test_probs = torch.cat(all_test_probs)
    # Setup metrics for classifier
    test_recall = Recall(
        task="multiclass", num_classes=num_classes, average="none").to(device)
    test_precision = Precision(
        task="multiclass", num_classes=num_classes, average="none").to(device)
    test_f1 = F1Score(task="multiclass", num_classes=num_classes,
                      average="none").to(device)
    test_accuracy = Accuracy(
        task="multiclass", num_classes=num_classes, average="macro").to(device)
    test_conf_mat = ConfusionMatrix(
        task="multiclass", num_classes=num_classes).to(device)
    # Calculate metrics for test set
    test_rec_val = test_recall(all_test_preds, all_test_labels)
    test_prec_val = test_precision(all_test_preds, all_test_labels)
    test_f1_val = test_f1(all_test_preds, all_test_labels)
    test_acc_val = test_accuracy(all_test_preds, all_test_labels)
    test_conf_mat_val = test_conf_mat(all_test_preds, all_test_labels)
    # Print testing set metrics
    print("Test Metrics:")
    print(f"Accuracy: {test_acc_val:.4f}")
    print(f"Recall: {test_rec_val}")  # Set to per class recall
    print(f"Precision: {test_prec_val}")  # Set to per class precision
    print(f"F1 Score: {test_f1_val}")  # Set to per class F1 score
    print("Confusion Matrix:")
    print(test_conf_mat_val.cpu().numpy())
    # Return testing set metrics for later storage in model
    return {
        'accuracy': test_acc_val.item(),
        'recall': test_rec_val,
        'precision': test_prec_val,
        'f1': test_f1_val,
        'confusion_matrix': test_conf_mat_val.cpu().numpy()
    }


def compute_mmd(real_ecg, fake_ecg, metric_lib, sigma=1.0):
    '''
    Calculates the maximum mean discrepancy (mmd) score for the
    provided signals and returns the calculated value.

    :param real_ecg: ECG from the real dataset
    :param fake_ecg: ECG generated by the generator model
    :param metric_lib: Metrics C library file
    :param sigma: Bandwidth parameter for the MMD kernel

    Returns:
        result: MMD value
    '''
    # Convert real and fake ECGs to numpy format from pytorch tensors
    real_np = real_ecg.detach().cpu().numpy().reshape(
        real_ecg.size(0), -1).astype(np.float64)  # As double
    fake_np = fake_ecg.detach().cpu().numpy().reshape(
        fake_ecg.size(0), -1).astype(np.float64)  # As double
    # Get number of batches and features in real and fake ECGs
    batch_real, features = real_np.shape
    batch_fake, _ = fake_np.shape
    # Call C function to calculate MMD
    result = metric_lib.compute_mmd(
        real_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fake_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        batch_real, batch_fake, features, sigma
    )
    return result


def compute_mvdTW(real_ecg, fake_ecg, metric_lib):
    '''
    Calculates the multivariate dynamic time warping (mvDTW) value for
    the provide signals and returns the calculated value.
    Measures similarity between real and fake ECG samples using
    mvDTW. Helps to evaluate quality of the generated signals.
    Useful for evaluation as able to handle time shifts and variations 
    in heart rate.

    :param real_ecg: ECG from the real dataset
    :param fake_ecg: ECG generated by the generator model
    :param metric_lib: Metrics C library file

    Returns:
        mvDTW: Average distance across the batches of real and fake ECGs
    '''
    # Convert pytorch tensors to numpy arrays
    if isinstance(real_ecg, torch.Tensor):
        real_np = real_ecg.detach().cpu().numpy()
    else:
        real_np = real_ecg
    if isinstance(fake_ecg, torch.Tensor):
        fake_np = fake_ecg.detach().cpu().numpy()
    else:
        fake_np = fake_ecg
    # Extract dimensions for batch size, ecg length, and leads
    batch_size, ecg_length, n_leads = real_np.shape
    distances = []
    # Iterate over each sample in batch
    for i in range(batch_size):
        # Flatten ECG sequences from multivariate to 1D array
        seq1 = real_np[i].flatten().astype(np.float64)
        seq2 = fake_np[i].flatten().astype(np.float64)
        # Call C function to calculate dtw
        dtw_distance_val = metric_lib.dtw_distance(
            seq1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            seq2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ecg_length,
            ecg_length,
            n_leads
        )
        # Store dtw distances for each sample
        distances.append(dtw_distance_val)
    # Return the average distance across batch
    return np.mean(distances).astype(np.float32)


def gradient_penalty(critic, real_samples, fake_samples, device, labels=None):
    '''
    Calculates the gradient penalty for a set of real and fake samples. Used to
    enforce the Lipschitz constraint by keeping the gradient values close to 1.
    Provides for smooth training process of a GAN and provides stability to the
    training process.

    :param critic: Critic model to judge samples
    :param real_samples: Samples from the real dataset
    :param fake_samples: Samples generated by the GAN generator
    :param labels: Labels for if the model being trained is CWGAN
    :param device: Device to send model and data to

    Returns:
        Penalty: Gradient penalty value to keep gradients close to 1 and
        to enforce the 1-Lipschitz constraint.
    '''
    batch_size = real_samples.size(0)  # Get batch size
    # Generate random interpolation coefficient to ensure per-sample weighting
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    # Computes interpolated samples: Create new samples between real and fake distributions
    interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples
    # Compute gradient on interpolates for later gradient calculations
    interpolates.requires_grad_(True)
    # Check if CWGAN or WGAN
    if labels is not None:
        # Critic scoring for interpolated samples : CWGAN
        critic_interpolates = critic(interpolates, labels)
    else:
        # Critic scoring for interpolated samples : WGAN
        critic_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(critic_interpolates, device=device)
    gradients = torch.autograd.grad(  # Compute gradients of the critic
        outputs=critic_interpolates,  # Critic scores
        inputs=interpolates,  # Interpolated samples
        grad_outputs=grad_outputs,  # Gradient of output
        create_graph=True,
        retain_graph=True,
        only_inputs=True  # compute only for the inputs
    )[0]
    # Flatten gradients per sample for computation of norms
    gradients = gradients.reshape(batch_size, -1)
    # Compute L2 norm of gradients for each sample in the batch
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()  # Calculate the penalty
    return penalty


def save_generated_ecg(generator, epoch, device, lead_mins, lead_maxs, latent_dim=50, save_path="images", num_classes=4):
    '''
    Saves images of generated ECG leads to the provided image path. Allows the evaluation of training
    progress through qualitative methods.

    :param generator: Generator model
    :param epoch: Current epoch
    :param device: Device to send model and data to
    :param lead_mins: Lead minimum values list
    :param lead_maxs: Lead maximum values list
    :param latent_dim: Size of the noise vector
    :param save_path: Path to save images to
    :param num_classes: Number of classes
    '''
    os.makedirs(save_path, exist_ok=True)
    generator.eval()  # Set generator to evaluation mode i.e. Prevent training
    with torch.no_grad():  # Disable gradient calculation
        # Create noise in shape (1, latent_dim) latent_dim=50
        noise = torch.randn(1, latent_dim, device=device)
        if num_classes != 0:  # For CWGAN
            random_label = random.randint(0, num_classes - 1)
            # Create a condition tensor with shape (1, 1)
            condition = torch.tensor(
                [random_label], device=device, dtype=torch.long).unsqueeze(1)
            # Generate ECG conditioned on the randomly chosen label
            gen_ecg = generator(noise, condition).cpu().numpy()
        else:  # For WGAN
            gen_ecg = generator(noise).cpu().numpy()
    # Reverse the normalization done to ECG signals before training
    gen_ecg = per_lead_inverse_scaling(gen_ecg, lead_mins, lead_maxs)
    gen_ecg = gen_ecg.squeeze(0)
    # Post-process bandpass filter
    gen_ecg = bandpass_filter(gen_ecg, 0.5, 40, 128)
    num_leads = gen_ecg.shape[1]  # Get the number of leads to display
    Leads = ['III', 'V3', 'V5']  # Names of the leads used
    for lead in range(num_leads):  # Plot for each lead
        plt.figure(figsize=(8, 4))
        plt.plot(gen_ecg[:, lead], linewidth=1.5, color='black')
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude (mV)")
        plt.title(f"Generated ECG - Lead {Leads[lead]} - Epoch {epoch+1}")
        plt.grid(True)
        plt.savefig(os.path.join(
            save_path, f"ecg_epoch_{epoch+1}_lead_{lead+1}.png"), bbox_inches='tight', dpi=300)
        plt.close()
    generator.train()  # Set the generator back to training mode


def plot_generated_sample(generated_signal, lead_mins, lead_maxs):
    '''
    Plots the signals generated by the GAN.

    :param generated_signal: The signal generated by the GAN generator
    :param lead_mins: List of minimums for each lead
    :param lead_maxs: List of maximums for each lead
    '''
    generated_sample = generated_signal.cpu().numpy()  # Convert tensor to numpy array
    generated_sample = per_lead_inverse_scaling(
        generated_sample, lead_mins, lead_maxs)  # Reverse scaling for the leads generated
    generated_sample = generated_sample.squeeze(
        0)  # Remove batch size dimension
    # Post-process filter the signals
    generated_sample = bandpass_filter(generated_sample, 0.5, 40, 128)
    # Subplot the leads
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    lead_labels = ['Lead III', 'V3', 'V5']
    for i, ax in enumerate(axs):
        ax.plot(generated_sample[:320, i])  # Take half of the signal
        ax.set_title(lead_labels[i])
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True)
    plt.tight_layout()
    plt.show()
