# main.py
import os
import numpy as np
from data_loader import load_mfpt_raw_data, class_map
from preprocessing import preprocess_data, resample_signal
from model import build_hybrid_model, train_and_evaluate, real_time_inference

def main():
    np.random.seed(42)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Load and preprocess data
    base_dir = os.getcwd()
    print("Loading MFPT data...")
    raw_data, labels = load_mfpt_raw_data(base_dir)
    print(f"Loaded {len(raw_data)} signals")
    print(f"Label distribution: {np.bincount(labels)}")
    
    snr_levels = [0, 10, 20]
    print("Preprocessing data...")
    X, y = preprocess_data(raw_data, labels, snr_levels)
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Label distribution after SNR: {np.bincount(y)}")
    
    # Build and train model
    print("Building and training hybrid model...")
    model = build_hybrid_model(num_classes=len(class_map))
    model.summary()
    trained_model, history, X_test, y_test = train_and_evaluate(model, X, y)
    
    # Test real-time inference
    print("Testing real-time inference...")
    test_signal = resample_signal(X_test[0].flatten(), 10000)
    real_time_inference(trained_model, test_signal)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")