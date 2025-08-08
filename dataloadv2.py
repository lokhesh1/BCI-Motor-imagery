# dataloadv2.py
import mne
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.fft import rfft
from tensorflow.keras.utils import to_categorical

# Constants
SAMPLE_RATE = 250
FMIN, FMAX = 4, 38
EXPECTED_EPOCH_SAMPLES = 251

# Event labels: 0 = Not Motor, 1 = Motor
EVENT_LABELS = {
    '768': 0,  # Rest/Baseline
    '769': 1,  # Left hand
    '770': 1,  # Right hand
    '771': 1,  # Foot
    '772': 1   # Tongue
}

GLOBAL_EEG_CHANNEL_NAMES = None


def preprocess_single_gdf(file_path):
    """Load and preprocess a single GDF file."""
    global GLOBAL_EEG_CHANNEL_NAMES
    mne.set_log_level('ERROR')

    raw = mne.io.read_raw_gdf(file_path, preload=True)
    raw.filter(FMIN, FMAX, fir_design='firwin')

    # Select EEG channels once globally
    if GLOBAL_EEG_CHANNEL_NAMES is None:
        picks = mne.pick_types(raw.info, eeg=True)
        GLOBAL_EEG_CHANNEL_NAMES = [raw.info['ch_names'][i] for i in picks]
        print(f"Channels: {GLOBAL_EEG_CHANNEL_NAMES}")

    picks_idx = mne.pick_channels(raw.info["ch_names"], include=GLOBAL_EEG_CHANNEL_NAMES)

    epochs, labels = [], []
    for annot in raw.annotations:
        desc = annot['description']
        if desc in EVENT_LABELS:
            start = int(annot['onset'] * raw.info['sfreq'])
            stop = start + EXPECTED_EPOCH_SAMPLES
            if stop > raw.n_times:
                continue

            data = raw.get_data(picks=picks_idx, start=start, stop=stop)
            if data.shape[1] == EXPECTED_EPOCH_SAMPLES:
                epochs.append(data)
                labels.append(EVENT_LABELS[desc])

    return np.array(epochs), np.array(labels)


def load_and_preprocess_data(folder_path):
    """Load all GDF files and return processed data for model training."""
    files = sorted(Path(folder_path).glob("*.gdf"))
    if not files:
        print("No GDF files found.")
        return np.array([]), np.array([]), 0, (0, 0)

    all_X, all_y = [], []
    for file in files:
        try:
            Xf, yf = preprocess_single_gdf(str(file))
            if Xf.size > 0:
                all_X.append(Xf)
                all_y.append(yf)
        except Exception as e:
            print(f"Error {file.name}: {e}")

    if not all_X:
        return np.array([]), np.array([]), 0, (0, 0)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    num_classes = len(np.unique(y))
    if num_classes < 2:
        return np.array([]), np.array([]), 0, (0, 0)

    n_samples, n_channels, n_time = X.shape

    # Normalize per channel
    X_norm = X.reshape(-1, n_time)
    X_norm = StandardScaler().fit_transform(X_norm).reshape(n_samples, n_channels, n_time)

    # Frequency features (rFFT)
    n_freq_bins = n_time // 2 + 1
    X_freq = np.abs(rfft(X_norm, axis=2))

    # Sliding windows
    window_size, stride = 50, 10
    X_win, y_win = [], []
    for i in range(n_samples):
        for start in range(0, n_time - window_size + 1, stride):
            time_win = X_norm[i, :, start:start+window_size]
            freq_win = X_freq[i, :, :window_size]
            X_win.append(np.stack([time_win, freq_win], axis=-1))
            y_win.append(y[i])

    X_win = np.array(X_win, dtype=np.float32)
    y_win = np.array(y_win)
    X_win = X_win.transpose(0, 2, 1, 3).reshape(X_win.shape[0], window_size, -1)

    y_cat = to_categorical(y_win, num_classes=num_classes)
    return X_win, y_cat, num_classes, (X_win.shape[1], X_win.shape[2])


if __name__ == "__main__":
    Xp, yp, nc, shape = load_and_preprocess_data("data/train")
    print(f"Data: {Xp.shape}, Labels: {yp.shape}, Classes: {nc}, Input shape: {shape}")
