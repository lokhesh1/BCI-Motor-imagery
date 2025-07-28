import mne
import numpy as np
from scipy.signal import firwin, filtfilt
import logging
import warnings

import random
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
# Completely silence MNE-Python output
mne.set_log_level('WARNING')  # or 'ERROR' for even less output
logging.getLogger('mne').setLevel(logging.WARNING)
def load_eeg(path):
    raw = mne.io.read_raw_gdf(path, preload=True)

    # Step 2: Get the sampling frequency
    sfreq = raw.info['sfreq']  # Hz

    # Step 3: Define FIR filter parameters
    low_cutoff = 1.0   # Hz
    high_cutoff = 30.0 # Hz
    filter_order = 177 # Must be odd for linear-phase FIR

    nyquist = 0.5 * sfreq


    fir_coeffs = firwin(
        numtaps=filter_order,
        cutoff=[low_cutoff / nyquist, high_cutoff / nyquist],
        pass_zero=False,
        window='blackman'
    )
    eeg_data = raw.get_data()
    filtered_data = filtfilt(fir_coeffs, 1.0, eeg_data, axis=1)

    new_raw = mne.io.RawArray(filtered_data, raw.info.copy())
    annotations = raw.annotations 
    new_raw.set_annotations(annotations)

    return new_raw

def preprocess(path, test=False):
    raw = load_eeg(path)
    eeg_channels = raw.ch_names[:22]
    raw.pick(eeg_channels)
    
    if not test:
        events, events_id = mne.events_from_annotations(raw, event_id= {'769': 0,'770': 1,'771': 2,'772': 3})
    else:
        events, events_id = mne.events_from_annotations(raw, event_id= {'768':6})
        #print(events_id)
    #print(events_id)
    epochs = mne.Epochs(
        raw,
        events=events,  
        tmin=0,     
        tmax=4.0,
        event_id=events_id,
        baseline=None,
        preload=True
    )

    labels = epochs.events[:, 2]
    #print(labels)
    #data = epochs.get_data()
    
    #return epochs
    return {
        'epochs': epochs,   
        'labels': labels
    }

#a = preprocess("E:/LOKI/BCI-IV/A01T.gdf")
# print(a['epochs'].shape)
import numpy as np
from scipy.signal import stft

def anchored_stft_epochs(epochs, nperseg=None, noverlap=None, window='hann'):

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    n_epochs, n_channels, n_times = data.shape

    # sensible defaults
    if nperseg is None:
        nperseg = n_times // 2
    if noverlap is None:
        noverlap = nperseg // 2

    # Pre-allocate output
    # Run a dummy STFT on one channel to get f and t axes
    f, t, Z = stft(data[0, 0], fs=sfreq, window=window,
                   nperseg=nperseg, noverlap=noverlap, boundary=None,   # avoid padding beyond epoch
                    padded=False)
    n_freqs, n_times_stft = Z.shape

    Sxx = np.zeros((n_epochs, n_channels, n_freqs, n_times_stft), dtype=complex)

    # Compute STFT for each epoch & channel
    for ei in range(n_epochs):
        for ch in range(n_channels):
            _, _, Zxx = stft(
                data[ei, ch],
                fs=sfreq,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                boundary=None,   # avoid padding beyond epoch
                padded=False
            )
            Sxx[ei, ch] = Zxx

    return f, t, Sxx

# epochs = preprocess("/teamspace/studios/this_studio/EEG_REC/A01T.gdf")
# # Suppose you already have an `mne.Epochs` object `epochs`
# f, t, Sxx = anchored_stft_epochs(
#     epochs,
#     nperseg=int(0.25 * epochs.info['sfreq']),  # 0.5 s windows
#     noverlap=int(0.20 * epochs.info['sfreq']) # 50% overlap
# )

# print("STFT output shape:", Sxx.shape)
# # → (n_epochs, n_channels, n_freqs, n_time_bins)

# # To get power (magnitude squared):
# power = np.abs(Sxx) ** 2
import os
import numpy as np

def preprocess_and_save_epochs(data_dir, output_dir, preprocess):
    """
    Load all EEG files in data_dir, preprocess them via preprocess_fn, 
    and save each epoch separately as a .npy file.

    - Each raw file yields some number of epochs (e.g., 288).
    - Each epoch is saved as `epoch_{idx}_data.npy` and `epoch_{idx}_label.npy`.

    Args:
        data_dir (str): Directory containing raw EEG files (e.g., .gdf).
        output_dir (str): Directory where epoch .npy files will be saved.
        preprocess_fn (callable): Function taking (filepath, test=False) 
                                  and returning dict with 'epochs' and 'labels'.
    """
    os.makedirs(output_dir, exist_ok=True)
    epoch_idx = 0

    # List all raw EEG files
    allowed_subjects = {'1', '2', '3', '5', '6', '7', '8', '9'}

    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('T.gdf') and f[1:3].lstrip('0') in allowed_subjects
    ])


    for fname in files:
        filepath = os.path.join(data_dir, fname)
        out = preprocess(filepath, test=False)
        epochs = out['epochs']
        f, t, Sxx = anchored_stft_epochs(
            epochs,
            nperseg=int(0.24 * epochs.info['sfreq']),  # 0.5 s windows
            noverlap=int(0.20 * epochs.info['sfreq']) # 50% overlap
        )
        data = Sxx   # shape: (n_epochs, n_ch, n_freqs, n_times)
        labels = out['labels'] # shape: (n_epochs,)

        n_epochs = data.shape[0]
        for i in range(n_epochs):
            epoch_data = data[i]      # single epoch
            epoch_label = labels[i]   # single label

            # Save epoch data
            np.save(os.path.join(output_dir, f'epoch_{epoch_idx}.npy'), {"data": epoch_data, "label": epoch_label})

            epoch_idx += 1

    print(f"Saved {epoch_idx} individual epochs to {output_dir}")

# Example usage:
preprocess_and_save_epochs(
    '/teamspace/studios/this_studio/EEG_REC',
    '/teamspace/studios/this_studio/epochs',
    preprocess
)
