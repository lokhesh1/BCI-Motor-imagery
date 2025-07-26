import os
import numpy as np

def preprocess_and_save_batches(data_dir, output_dir, batch_size, preprocess_fn):
    """
    Load all EEG files in data_dir, preprocess them via preprocess_fn, and save epochs in batches.

    - Each raw file yields some number of epochs (typically 288).
    - Batches of size `batch_size` are saved as separate .npy files named batch_{idx}_data.npy and batch_{idx}_labels.npy.
    - If a file's epochs don't perfectly fill the batch, leftover epochs are carried over and combined with the next file's epochs.
    - After the final file, any remaining epochs in the buffer are saved as the last batch (may be smaller than batch_size).

    Args:
        data_dir (str): Directory containing raw EEG files (e.g., .gdf).
        output_dir (str): Directory where batch .npy files will be saved.
        batch_size (int): Number of epochs per saved file.
        preprocess_fn (callable): Function taking (filepath, test=False) and returning dict with 'epochs' and 'labels'.
    """
    os.makedirs(output_dir, exist_ok=True)
    buffer_data = []
    buffer_labels = []
    batch_idx = 0

    # List all raw files (e.g., .gdf)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('T.gdf')])

    for fname in files:
        filepath = os.path.join(data_dir, fname)
        out = preprocess_fn(filepath, test=False)
        data = out['epochs']   # shape: (n_epochs, n_ch, n_freqs, n_times)
        labels = out['labels'] # shape: (n_epochs,)

        # Append to buffer
        buffer_data.append(data)
        buffer_labels.append(labels)

        # Concatenate buffer
        buf_data = np.concatenate(buffer_data, axis=0)
        buf_labels = np.concatenate(buffer_labels, axis=0)

        # Split into full batches
        n_total = buf_data.shape[0]
        n_full = n_total // batch_size
        if n_full > 0:
            # Only full batches
            full_data = buf_data[:n_full * batch_size]
            full_labels = buf_labels[:n_full * batch_size]

            # Save each batch
            for i in range(n_full):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch_data = full_data[start:end]
                batch_labels = full_labels[start:end]

                np.save(os.path.join(output_dir, f'batch_{batch_idx}_data.npy'), batch_data)
                np.save(os.path.join(output_dir, f'batch_{batch_idx}_labels.npy'), batch_labels)
                batch_idx += 1

        # Keep remainder in buffer
        rem = n_total % batch_size
        if rem > 0:
            buffer_data = [buf_data[-rem:]]
            buffer_labels = [buf_labels[-rem:]]
        else:
            buffer_data = []
            buffer_labels = []

    # After processing all files, save any leftover epochs
    if buffer_data:
        final_data = np.concatenate(buffer_data, axis=0)
        final_labels = np.concatenate(buffer_labels, axis=0)
        np.save(os.path.join(output_dir, f'batch_{batch_idx}_data.npy'), final_data)
        np.save(os.path.join(output_dir, f'batch_{batch_idx}_labels.npy'), final_labels)

    print(f"Saved {batch_idx + (1 if buffer_data else 0)} batches to {output_dir}")
    
preprocess_and_save_batches('/teamspace/studios/this_studio/EEG_REC', '/teamspace/studios/this_studio/numpy', 32, preprocess)