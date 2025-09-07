import os
import mne
import pandas as pd
import re

# --- CONFIGURATION ---
BASE_PROJECT_DIR = r"D:\numbers_eeg_nn_project"
BASE_PREPROCESSED_DIR = os.path.join(BASE_PROJECT_DIR, "data_preprocessed")
DIR_V1 = os.path.join(BASE_PREPROCESSED_DIR, "all_trials_dataset (30hz)")
DIR_V2 = os.path.join(BASE_PREPROCESSED_DIR, "all_trials_dataset (30hz) V2")
HAPPE_FILE = os.path.join(BASE_PROJECT_DIR, "data_input", "HAPPE_Usable_Trials.csv")

print("--- Verifying Trial Counts Against HAPPE Ground Truth ---")
print("-" * 60)

# --- Load Ground Truth Data ---
try:
    happ_df = pd.read_csv(HAPPE_FILE)
    # The first column name is messy, rename it for easier access
    happ_df.rename(columns={happ_df.columns[0]: 'SessionInfo'}, inplace=True)
    print(f"Successfully loaded ground truth from {os.path.basename(HAPPE_FILE)}")
except Exception as e:
    print(f"FATAL: Could not load HAPPE file: {e}")
    exit()

# Get a list of subjects from the V2 directory to iterate through
try:
    v2_files = [f for f in os.listdir(DIR_V2) if f.endswith('.fif')]
    subject_ids = sorted([re.search(r'sub-(\d+)_', f).group(1) for f in v2_files])
    if not subject_ids:
        raise FileNotFoundError("No subject files found in V2 directory.")
except Exception as e:
    print(f"Error: Could not list subjects from V2 directory. {e}")
    exit()

# --- Comparison Loop ---
for subject_id in subject_ids:
    print(f"\n--- Subject {subject_id} ---")
    
    # 1. Get Ground Truth Count from HAPPE file
    ground_truth_count = -1
    try:
        # Robustly find the subject row, matching "Subj2", "Subject02", etc.
        pattern = re.compile(f'(?:Subject|Subj){int(subject_id)}', re.IGNORECASE)
        subject_row = happ_df[happ_df['SessionInfo'].str.contains(pattern, na=False)]
        
        if not subject_row.empty:
            kept_indices_str = subject_row.iloc[0]['Kept_Segs_Indxs']
            kept_indices = [int(i.strip()) for i in kept_indices_str.split(',')]
            ground_truth_count = len(kept_indices)
            print(f"Ground Truth (HAPPE): {ground_truth_count} trials")
        else:
            print(f"Ground Truth (HAPPE): NOT FOUND for Subject {subject_id}")

    except Exception as e:
        print(f"Ground Truth (HAPPE): ERROR - {e}")

    # 2. Get Trial Count from V1
    try:
        file_v1 = os.path.join(DIR_V1, f"sub-{subject_id}_preprocessed-epo.fif")
        if os.path.exists(file_v1):
            epochs_v1 = mne.read_epochs(file_v1, preload=False, verbose='ERROR')
            trials_v1 = len(epochs_v1)
            match_str = "MATCH" if trials_v1 == ground_truth_count else "MISMATCH"
            print(f"V1 Dataset Trials:    {trials_v1} trials -> {match_str}")
        else:
            print("V1 Dataset Trials:    NOT FOUND")
    except Exception as e:
        print(f"V1 Dataset Trials:    ERROR - {e}")
        
    # 3. Get Trial Count from V2
    try:
        file_v2 = os.path.join(DIR_V2, f"sub-{subject_id}_preprocessed-epo.fif")
        if os.path.exists(file_v2):
            epochs_v2 = mne.read_epochs(file_v2, preload=False, verbose='ERROR')
            trials_v2 = len(epochs_v2)
            match_str = "MATCH" if trials_v2 == ground_truth_count else "MISMATCH"
            print(f"V2 Dataset Trials:    {trials_v2} trials -> {match_str}")
        else:
            print("V2 Dataset Trials:    NOT FOUND")
    except Exception as e:
        print(f"V2 Dataset Trials:    ERROR - {e}")

print("-" * 60)
print("Verification complete.")
