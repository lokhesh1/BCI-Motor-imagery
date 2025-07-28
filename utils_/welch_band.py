from scipy.signal import welch
import mne
import numpy as np

def compute_band_powers(eeg_data, sfreq, bands):
    n_channels, n_times = eeg_data.shape
    band_powers = {band: [] for band in bands}

    for ch in range(n_channels):
        f, psd = welch(eeg_data[ch], fs=sfreq, nperseg=sfreq*2)
        for band in bands:
            fmin, fmax = bands[band]
            idx_band = np.logical_and(f >= fmin, f <= fmax)
            power = np.trapz(psd[idx_band], f[idx_band])  # Area under PSD
            band_powers[band].append(power)

    # Return as array: shape = (n_channels, n_bands)
    return np.array([band_powers[band] for band in bands]).T
freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100),
}

# data = epochs.get_data()  # Assuming epochs is an MNE Epochs object
# sfreq = epochs.info['sfreq']  # Sampling frequency in Hz
# band_power_features = compute_band_powers(data, sfreq, freq_bands)
# print("Shape:", band_power_features.shape)  # (n_channels, n_bands)
