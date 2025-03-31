import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN message
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mfpt_raw_data
from preprocessing import resample_signal, preprocess_signal

# Load raw data
base_dir = os.getcwd()
raw_data, labels = load_mfpt_raw_data(base_dir)

# Select a sample (first Normal signal)
sample_signal, sample_fs, sample_condition, sample_file = raw_data[0]
print(f"Selected signal: {sample_condition} from {sample_file}")

# Resample
resampled_signal = resample_signal(sample_signal, sample_fs)
time = np.arange(10000) / 10000

# Convert to spectrogram
spectrogram = preprocess_signal(sample_signal, sample_fs, snr=None)
spectrogram_img = spectrogram[:, :, 0]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(time, resampled_signal, color='blue')
ax1.set_title(f"Raw Signal: {sample_condition}")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude (g)")
ax1.grid(True)
im = ax2.imshow(spectrogram_img, aspect='auto', cmap='viridis', origin='lower')
ax2.set_title("Spectrogram")
ax2.set_xlabel("Time (224 bins)")
ax2.set_ylabel("Frequency (224 bins)")
plt.colorbar(im, ax=ax2, label="Log Power Spectral Density")
plt.tight_layout()
plt.show()