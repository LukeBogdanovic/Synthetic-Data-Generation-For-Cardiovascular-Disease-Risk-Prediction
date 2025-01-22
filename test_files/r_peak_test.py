import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import shift

# Simulated ECG Signal (Replace with real data)
fs = 500  # Sampling frequency (Hz)
t = np.linspace(0, 10, fs * 10)  # 10 seconds of data
ecg_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * \
    np.sin(2 * np.pi * 50 * t)  # Fake signal
ecg_signal += 1.5 * np.random.randn(len(t))  # Adding noise

# Step 1: Detect R-Peaks
# Use scipy's `find_peaks` to detect R-peaks (for real ECG, use an advanced method)
peaks, _ = find_peaks(ecg_signal, height=np.mean(
    ecg_signal) + 0.5, distance=fs // 2)

# Step 2: Define a Fixed Window Around R-Peaks
window_size = int(1.0 * fs)  # 1-second window
half_window = window_size // 2

# Step 3: Extract and Center Beats
aligned_beats = []

for peak in peaks:
    # Ensure the window does not go out of bounds
    start_idx = max(0, peak - half_window)
    end_idx = min(len(ecg_signal), peak + half_window)
    beat = ecg_signal[start_idx:end_idx]

    # Pad if the extracted beat is not full-length
    if len(beat) < window_size:
        padding = np.zeros(window_size - len(beat))
        beat = np.concatenate([beat, padding])

    aligned_beats.append(beat)

# Convert to array for easier processing
aligned_beats = np.array(aligned_beats)

# Step 4: Visualize Results
plt.figure(figsize=(15, 10))
plt.title("Aligned ECG Beats with Centered R-Peaks")
for i, beat in enumerate(aligned_beats[:5]):  # Plot first 5 beats
    plt.subplot(5, 1, i + 1)
    plt.plot(beat, label=f'Beat {i + 1}')
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
plt.legend()
plt.show()
