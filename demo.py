# demo.py
import numpy as np
from preprocessing import resample_signal, preprocess_signal
from model import build_hybrid_model, real_time_inference
from data_loader import load_mfpt_raw_data

# Load a sample signal
base_dir = os.getcwd()
raw_data, labels = load_mfpt_raw_data(base_dir)
sample_signal, sample_fs, _, _ = raw_data[0]

# Load trained model
model = build_hybrid_model(num_classes=4, input_size=128)  # Updated input size
model.load_weights('saved_model/hybrid_model.weights.h5')

# Simulate real-time inference
signal = resample_signal(sample_signal, sample_fs)
real_time_inference(model, signal)