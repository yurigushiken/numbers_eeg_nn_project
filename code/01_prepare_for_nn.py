# D:\numbers_eeg_nn_project\code\01_prepare_for_nn.py

import os
import re
import pandas as pd
import mne
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION ---
# Keep only accurate trials if the environment variable PROCESS_ACC_ONLY is set to "true" (case-insensitive).
# Defaults to True to preserve existing behaviour.
PROCESS_ACC_ONLY = os.getenv("PROCESS_ACC_ONLY", "True").lower() == "true"

# --- PATHS ---
BASE_PROJECT_DIR = r"D:\numbers_eeg_nn_project"
BASE_INPUT_DIR = os.path.join(BASE_PROJECT_DIR, "data_input")
BASE_OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "data_preprocessed")
BEHAVIORAL_DATA_DIR = os.path.join(BASE_INPUT_DIR, "Final Behavioral Data Files", "data_UFT8_lite")
HAPPE_SET_DIR = os.path.join(BASE_INPUT_DIR, "5 - processed")
HAPPE_USABLE_TRIALS_FILE = os.path.join(BASE_INPUT_DIR, "HAPPE_Usable_Trials.csv")

# --- DYNAMIC OUTPUT PATH ---
if PROCESS_ACC_ONLY:
    output_dir = os.path.join(BASE_OUTPUT_DIR, "acc_1_dataset")
else:
    output_dir = os.path.join(BASE_OUTPUT_DIR, "all_trials_dataset")
os.makedirs(output_dir, exist_ok=True)
print(f"--- Outputting to: {output_dir} ---")

# --- 2. LOAD MASTER USABLE TRIAL LIST ---
usable_trials_df = pd.read_csv(HAPPE_USABLE_TRIALS_FILE)
usable_trials_df.rename(columns={usable_trials_df.columns[0]: 'SessionInfo'}, inplace=True)

# --- 3. PRE-SCAN FOR ALL LABELS TO CREATE A GLOBAL ENCODER ---
print("--- Pass 1: Scanning for all unique labels ---")
all_labels = set()
subject_ids_in_folder = sorted([re.search(r'(?:Subject|Subj)(\d+)', f).group(1).zfill(2) for f in os.listdir(BEHAVIORAL_DATA_DIR) if re.search(r'(?:Subject|Subj)(\d+)', f)])

for subject_id in subject_ids_in_folder:
    try:
        behavioral_file = os.path.join(BEHAVIORAL_DATA_DIR, f"Subject{subject_id}_lite.csv")
        behavioral_df = pd.read_csv(behavioral_file, on_bad_lines='warn', low_memory=False)
        all_labels.update(behavioral_df['Condition'].astype(str).dropna().unique())
    except Exception as e:
        print(f"Could not process {behavioral_file} for label scanning: {e}")

global_le = LabelEncoder()
global_le.fit(sorted(list(all_labels)))
print(f"Found {len(global_le.classes_)} unique labels across all subjects.")
print("-" * 50)


# --- 4. MAIN PROCESSING LOOP ---
print("\n--- Pass 2: Processing and saving individual subjects ---")
for subject_id in subject_ids_in_folder:
    try:
        print(f"Processing Subject {subject_id}...")

        set_file_path = os.path.join(HAPPE_SET_DIR, f"Subject{subject_id}.set")
        epochs = mne.io.read_epochs_eeglab(set_file_path, verbose=False)

        pattern = re.compile(f'(?:Subject|Subj){int(subject_id)}')
        subject_row = usable_trials_df[usable_trials_df['SessionInfo'].str.contains(pattern, na=False)].iloc[0]
        kept_indices_1based = [int(i.strip()) for i in subject_row['Kept_Segs_Indxs'].split(',')]
        
        behavioral_file = os.path.join(BEHAVIORAL_DATA_DIR, f"Subject{subject_id}_lite.csv")
        behavioral_df = pd.read_csv(behavioral_file, on_bad_lines='warn', low_memory=False)
        
        non_practice_mask = behavioral_df['Procedure'] != "Practiceproc"
        behavioral_df.loc[non_practice_mask, 'Trial_Continuous'] = np.arange(1, non_practice_mask.sum() + 1)
        
        
        behavioral_df_filtered = behavioral_df[behavioral_df['Trial_Continuous'].isin(kept_indices_1based)].copy()
        
        if len(epochs) != len(behavioral_df_filtered):
            raise ValueError(f"FATAL MISMATCH for Subject {subject_id}: Num epochs ({len(epochs)}) != Num behavioral trials ({len(behavioral_df_filtered)}).")
        
        encoded_labels = global_le.transform(behavioral_df_filtered['Condition'].astype(str))
        events_from_metadata = np.array([
            np.arange(len(behavioral_df_filtered)),
            np.zeros(len(behavioral_df_filtered), int),
            encoded_labels
        ]).T

        event_id = {label: i for i, label in enumerate(global_le.classes_)}

        epochs_with_metadata = mne.EpochsArray(
            epochs.get_data(), info=epochs.info, events=events_from_metadata,
            tmin=epochs.tmin, event_id=event_id, metadata=behavioral_df_filtered
        )

        if PROCESS_ACC_ONLY:
            final_epochs = epochs_with_metadata[epochs_with_metadata.metadata['Target.ACC'] == 1]
            print(f"  Keeping {len(final_epochs)} accurate trials.")
        else:
            final_epochs = epochs_with_metadata
            print(f"  Keeping all {len(final_epochs)} trials.")

        # --- DROP UNWANTED TRANSITION LABELS (e.g., "99") ---
        final_epochs = final_epochs[final_epochs.metadata['Condition'] != 99]
        print(f"  After dropping label-99 trials: {len(final_epochs)} epochs remain.")

        output_filename = f"sub-{subject_id}_preprocessed-epo.fif"
        output_path = os.path.join(output_dir, output_filename)
        final_epochs.save(output_path, overwrite=True, verbose=False)

        print(f"  -> Saved to {output_path}")

    except Exception as e:
        print(f"!!! FAILED for subject {subject_id}: {e}")
        continue

print("\n--- DATA PREPARATION COMPLETE ---") 