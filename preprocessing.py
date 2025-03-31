# preprocessing.py
import numpy as np
import scipy.signal
import tensorflow as tf
import os
import pickle

def resample_signal(signal, original_fs, target_fs=10000, target_length=10000):
    num_samples = int(len(signal) * target_fs / original_fs)
    resampled = scipy.signal.resample(signal, num_samples)
    if len(resampled) >= target_length:
        return resampled[:target_length]
    return np.pad(resampled, (0, target_length - len(resampled)), 'constant')

def add_awgn(signal, snr):
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def preprocess_signal(signal, fs=10000, snr=None, target_size=128):  # Reduced target size
    signal_resampled = resample_signal(signal, fs)
    if snr is not None:
        signal_resampled = add_awgn(signal_resampled, snr)
    f, t, Sxx = scipy.signal.stft(signal_resampled, fs=fs, nperseg=256)
    Sxx = np.abs(Sxx)
    Sxx = np.log1p(Sxx)
    Sxx_resized = tf.image.resize(Sxx[np.newaxis, :, :, np.newaxis], (target_size, target_size)).numpy()[0, :, :, 0]
    Sxx_resized = (Sxx_resized - np.min(Sxx_resized)) / (np.max(Sxx_resized) - np.min(Sxx_resized))
    return np.repeat(Sxx_resized[..., np.newaxis], 3, axis=-1)

def preprocess_data(raw_data, labels, snr_levels=[0, 10, 20], cache_dir='cache'):
    # Check if cached data exists
    cache_file = os.path.join(cache_dir, 'preprocessed_data.pkl')
    if os.path.exists(cache_file):
        print("Loading preprocessed data from cache...")
        with open(cache_file, 'rb') as f:
            X, y = pickle.load(f)
        return X, y
    
    # Preprocess if no cache
    os.makedirs(cache_dir, exist_ok=True)
    X = []
    y = []
    for (signal, fs, condition, file), label in zip(raw_data, labels):
        for snr in snr_levels:
            spectrogram = preprocess_signal(signal, fs, snr, target_size=128)
            X.append(spectrogram)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump((X, y), f)
    print(f"Saved preprocessed data to {cache_file}")
    
    return X, y