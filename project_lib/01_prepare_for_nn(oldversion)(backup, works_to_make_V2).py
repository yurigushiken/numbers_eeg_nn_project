# D:\numbers_eeg_nn_project\code\01_prepare_for_nn.py

import os
import re
import pandas as pd
import mne
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION ---
PROCESS_ACC_ONLY = True  # <-- Set to True for acc=1 data

# --- PATHS ---
BASE_PROJECT_DIR = r"D:\numbers_eeg_nn_project"
BASE_INPUT_DIR = os.path.join(BASE_PROJECT_DIR, "data_input")
BASE_OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "data_preprocessed")
BEHAVIORAL_DATA_DIR = os.path.join(BASE_INPUT_DIR, "Final Behavioral Data Files", "data_UTF8")   
HAPPE_SET_DIR = os.path.join(BASE_INPUT_DIR, "5 - processed (30hz)")
HAPPE_USABLE_TRIALS_FILE = os.path.join(BASE_INPUT_DIR, "HAPPE_Usable_Trials.csv")

# --- DYNAMIC OUTPUT PATH ---
if PROCESS_ACC_ONLY:
    output_dir = os.path.join(BASE_OUTPUT_DIR, "acc_1_dataset (30hz) V2") # Outputting to V2
else:
    output_dir = os.path.join(BASE_OUTPUT_DIR, "all_trials_dataset (30hz) V2") # Outputting to V2
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
        behavioral_file = os.path.join(BEHAVIORAL_DATA_DIR, f"Subject{subject_id}.csv")
        behavioral_df = pd.read_csv(behavioral_file, on_bad_lines='warn', low_memory=False)      
        behavioral_df['transition_label'] = behavioral_df['CellNumber'].astype(str)
        all_labels.update(behavioral_df['transition_label'].dropna().unique())
    except Exception as e:
        print(f"Could not process {behavioral_file} for label scanning: {e}")

global_le = LabelEncoder()
global_le.fit(sorted(list(all_labels)))
print(f"Found {len(global_le.classes_)} unique labels across all subjects.")
print("-" * 50)


# --- NEW: LABELING HELPERS (ported from 01_prepare_for_nn.py) ---
SMALL_SET = {1, 2, 3}
LARGE_SET = {4, 5, 6}

def direction_label(cond):
    try:
        s = str(int(cond)).zfill(2)
        prime, target = s[0], s[1]
    except (ValueError, TypeError):
        return pd.NA
    if prime == target:
        return "NC"
    return "I" if prime < target else "D"

def transition_category(cond):
    try:
        s = str(int(cond)).zfill(2)
        a, b = int(s[0]), int(s[1])
    except (ValueError, TypeError):
        return pd.NA
    if a == b:
        return "NC"
    if a in SMALL_SET and b in SMALL_SET:
        return "iSS" if a < b else "dSS"
    if a in LARGE_SET and b in LARGE_SET:
        return "iLL" if a < b else "dLL"
    if a in SMALL_SET and b in LARGE_SET:
        return "iSL"
    if a in LARGE_SET and b in SMALL_SET:
        return "dLS"
    return pd.NA

def size_category(cond):
    try:
        s = str(int(cond)).zfill(2)
        a, b = int(s[0]), int(s[1])
    except (ValueError, TypeError):
        return pd.NA
    if a == b:
        return "NC"
    a_small, b_small = a in SMALL_SET, b in SMALL_SET
    if a_small and b_small:
        return "SS"
    if not a_small and not b_small:
        return "LL"
    return "cross"
# --- END NEW HELPERS ---


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

        behavioral_file = os.path.join(BEHAVIORAL_DATA_DIR, f"Subject{subject_id}.csv")
        behavioral_df = pd.read_csv(behavioral_file, on_bad_lines='warn', low_memory=False)      

        non_practice_mask = behavioral_df['Procedure[Block]'] != "Practiceproc"
        behavioral_df.loc[non_practice_mask, 'Trial_Continuous'] = np.arange(1, non_practice_mask.sum() + 1)

        behavioral_df['transition_label'] = behavioral_df['CellNumber'].astype(str)

        behavioral_df_filtered = behavioral_df[behavioral_df['Trial_Continuous'].isin(kept_indices_1based)].copy()

        # --- FINAL FIX: Remove trials with Condition == 99 ---
        # A boolean mask is created from the 'CellNumber' column (which serves as 'Condition').
        # This MUST be done after the HAPPE alignment but before the main length check.
        valid_condition_mask = behavioral_df_filtered['CellNumber'].astype(str) != '99'
        
        # The mask is applied to BOTH the epochs object and the behavioral dataframe.
        # This keeps them perfectly synchronized.
        epochs = epochs[valid_condition_mask]
        behavioral_df_filtered = behavioral_df_filtered[valid_condition_mask]
        # --- END FINAL FIX ---

        # --- FIX: Create a unified accuracy column from all Target*.ACC columns ---
        acc_cols = [col for col in behavioral_df_filtered.columns if 'ACC' in col and 'Target' in col and 'OverallAcc' not in col]
        if acc_cols:
            # Convert all found acc columns to numeric, coercing errors
            for col in acc_cols:
                behavioral_df_filtered[col] = pd.to_numeric(behavioral_df_filtered[col], errors='coerce')
            
            # Create the unified column by filling NaNs from one column with values from the next
            unified_acc = behavioral_df_filtered[acc_cols[0]]
            for i in range(1, len(acc_cols)):
                unified_acc = unified_acc.fillna(behavioral_df_filtered[acc_cols[i]])
            behavioral_df_filtered['unified_ACC'] = unified_acc
        else:
            # Fallback if no Target*.ACC columns are found, use the original column
            behavioral_df_filtered['unified_ACC'] = behavioral_df_filtered['Target.ACC']
        # --- END FIX ---

        if len(epochs) != len(behavioral_df_filtered):
            raise ValueError(f"FATAL MISMATCH for Subject {subject_id}: Num epochs ({len(epochs)}) != Num behavioral trials ({len(behavioral_df_filtered)}).")

        # --- FIX: Create simplified metadata to match trainer expectations ---
        # 1. Create the descriptive columns that were in V1
        behavioral_df_filtered['direction'] = behavioral_df_filtered['CellNumber'].apply(direction_label)
        behavioral_df_filtered['change_group'] = behavioral_df_filtered['CellNumber'].apply(transition_category)
        behavioral_df_filtered['size'] = behavioral_df_filtered['CellNumber'].apply(size_category)
        
        # 2. Build the final metadata dataframe with specific columns and names
        final_metadata = pd.DataFrame({
            'SubjectID': subject_id,
            'Block': behavioral_df_filtered['Block'],
            'Trial': behavioral_df_filtered['Trial'],
            'Procedure': behavioral_df_filtered['Procedure[Block]'],
            'Condition': behavioral_df_filtered['CellNumber'],
            'Target.ACC': behavioral_df_filtered['unified_ACC'],
            'Target.RT': behavioral_df_filtered['Target.RT'],
            'Trial_Continuous': behavioral_df_filtered['Trial_Continuous'],
            'direction': behavioral_df_filtered['direction'],
            'change_group': behavioral_df_filtered['change_group'],
            'size': behavioral_df_filtered['size']
        })
        # --- END METADATA FIX ---

        encoded_labels = global_le.transform(behavioral_df_filtered['transition_label'])
        events_from_metadata = np.array([
            np.arange(len(behavioral_df_filtered)),
            np.zeros(len(behavioral_df_filtered), int),
            encoded_labels
        ]).T

        event_id = {label: i for i, label in enumerate(global_le.classes_)}

        epochs_with_metadata = mne.EpochsArray(
            epochs.get_data(), info=epochs.info, events=events_from_metadata,
            tmin=epochs.tmin, event_id=None, metadata=final_metadata
        )

        if PROCESS_ACC_ONLY:
            # Filter using the new final_metadata table
            final_epochs = epochs_with_metadata[epochs_with_metadata.metadata['Target.ACC'] == 1]
            print(f"  Keeping {len(final_epochs)} accurate trials.")
        else:
            final_epochs = epochs_with_metadata
            print(f"  Keeping all {len(final_epochs)} trials.")

        output_filename = f"sub-{subject_id}_preprocessed-epo.fif"
        output_path = os.path.join(output_dir, output_filename)
        final_epochs.save(output_path, overwrite=True, verbose=False)

        print(f"  -> Saved to {output_path}")

    except Exception as e:
        import traceback
        print(f"!!! FAILED for subject {subject_id}: {e}")
        traceback.print_exc()
        continue

print("\n--- DATA PREPARATION COMPLETE ---")