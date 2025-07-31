# scripts/prepare_data_for_transfer.py

import numpy as np
import mne
import os
import glob
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = 'data_preprocessed/acc_1_dataset/'
DATASET_NAME = 'NUMBERS_COG_ACC1'
OUTPUT_DIR = f'third_party/DeepTransferEEG/data/{DATASET_NAME}'

def prepare_data_for_transfer():
    """
    Reads MNE .fif files and converts them into X.npy, labels.npy, AND S.npy
    (subject indices) required for robust splitting in DeepTransferEEG.
    """
    print(f"Reading MNE epochs from: {INPUT_DIR}")
    fif_files = sorted(glob.glob(os.path.join(INPUT_DIR, 'sub-*.fif')))

    if not fif_files:
        print(f"Error: No .fif files found in {INPUT_DIR}. Please run your preprocessing first.")
        return

    all_data = []
    all_labels = []
    all_subject_indices = []

    print(f"Found {len(fif_files)} subject files. Processing...")
    # Enumerate to get a consistent subject index (0, 1, 2, ...)
    for subject_idx, f_path in enumerate(tqdm(fif_files, desc="Processing subjects")):
        epochs = mne.read_epochs(f_path, preload=True, verbose=False)

        # Extract labels using the modulo operator for the numeric 'Condition' column
        labels = (epochs.metadata['Condition'] % 10) - 1
        all_labels.append(labels.values)

        # Get the EEG data
        data = epochs.get_data(copy=False)
        all_data.append(data)

        # --- THIS IS THE NEW PART ---
        # Create an array of the current subject_idx, repeated for each trial
        subject_ids_for_file = np.full(len(epochs), subject_idx)
        all_subject_indices.append(subject_ids_for_file)

    # Concatenate all data into single arrays
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    S = np.concatenate(all_subject_indices, axis=0) # The new subject index array

    print("\nFinal data shapes:")
    print(f"  X (data): {X.shape}")
    print(f"  y (labels): {y.shape}")
    print(f"  S (subject IDs): {S.shape}") # You should see this new line

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving converted data to: {OUTPUT_DIR}")

    # Save all three files
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, 'S.npy'), S) # Save the new file

    print("\nData preparation complete!")
    print("Please use these parameters for the next step (in dnn.py and ttime.py):")
    print(f"  Dataset Name: '{DATASET_NAME}'")
    print(f"  Number of subjects (N): {len(fif_files)}")
    print(f"  Number of channels (chn): {X.shape[1]}")
    print(f"  Number of time samples (time_sample_num): {X.shape[2]}")
    print(f"  Number of classes (class_num): {len(np.unique(y))}")

if __name__ == '__main__':
    prepare_data_for_transfer()