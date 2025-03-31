# data_loader.py
import os
import scipy.io
import numpy as np

class_map = {'Normal': 0, 'Bearing Fault': 1, 'Misalignment': 2, 'Rotor Imbalance': 3}

def simulate_misalignment(signal, fs=10000):
    t = np.arange(len(signal)) / fs
    return signal + 0.1 * np.sin(2 * np.pi * 1 * t)

def simulate_rotor_imbalance(signal, fs=10000):
    t = np.arange(len(signal)) / fs
    return signal + 0.1 * np.sin(2 * np.pi * 25 * t)

def simulate_bearing_fault(signal, fs=10000):
    t = np.arange(len(signal)) / fs
    return signal + 0.2 * np.sin(2 * np.pi * 100 * t)  # Higher freq for BF

def load_mfpt_raw_data(base_dir):
    raw_data = []
    labels = []
    subfolders = {
        'Three Baseline Conditions': 'Normal',
        'Three Outer Race Fault Conditions': 'Bearing Fault',
        'Seven More Outer Race Fault Conditions': 'Bearing Fault',
        'Seven Inner Race Fault Conditions': 'Bearing Fault'
    }
    
    mfpt_dir = os.path.join(base_dir, 'MFPT_Dataset')
    if not os.path.exists(mfpt_dir):
        raise FileNotFoundError(f"Directory '{mfpt_dir}' not found.")
    
    for subfolder, condition in subfolders.items():
        subfolder_path = os.path.join(mfpt_dir, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"Warning: Subfolder '{subfolder_path}' not found. Skipping.")
            continue
        
        for file in os.listdir(subfolder_path):
            if file.endswith('.mat'):
                file_path = os.path.join(subfolder_path, file)
                mat = scipy.io.loadmat(file_path)
                signal = mat['bearing']['gs'][0][0].flatten()
                fs = mat['bearing']['sr'][0][0][0][0]
                raw_data.append((signal, fs, condition, file))
                labels.append(class_map[condition])
    
    # Simulate additional signals to reach ~100 per class
    original_data = raw_data.copy()
    for _ in range(5):  # Repeat 5x to increase dataset
        for signal, fs, condition, file in original_data:
            if condition == 'Normal':
                raw_data.append((simulate_misalignment(signal), fs, 'Misalignment', f"{file}_MA_{_}"))
                labels.append(class_map['Misalignment'])
                raw_data.append((simulate_rotor_imbalance(signal), fs, 'Rotor Imbalance', f"{file}_RI_{_}"))
                labels.append(class_map['Rotor Imbalance'])
                raw_data.append((simulate_bearing_fault(signal), fs, 'Bearing Fault', f"{file}_BF_{_}"))
                labels.append(class_map['Bearing Fault'])
            raw_data.append((signal, fs, condition, f"{file}_{_}"))
            labels.append(class_map[condition])
    
    return raw_data, labels