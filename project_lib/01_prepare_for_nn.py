# D:\numbers_eeg_nn_project\code\01_prepare_for_nn.py

import os
import re
import argparse
import pandas as pd
import mne
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION & CLI ---
# You can override defaults via CLI:
#   --input-set-dir  D:\numbers_eeg_nn_project\data_input\5 - processed (45hz)
#   --subset         {acc1|acc0|all}
# If --subset is omitted, we fall back to PROCESS_ACC_ONLY env var for backwards compatibility

# --- PATHS ---
BASE_PROJECT_DIR = r"D:\\numbers_eeg_nn_project"
BASE_INPUT_DIR = os.path.join(BASE_PROJECT_DIR, "data_input")
BASE_OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "data_preprocessed")
# REVERTED: Point back to the original, stable behavioral data directory
BEHAVIORAL_DATA_DIR = os.path.join(BASE_INPUT_DIR, "Final Behavioral Data Files", "data_UFT8_lite")
HAPPE_QC_FILE = os.path.join(BASE_INPUT_DIR, "6 - quality_assessment_outputs (45hz)", "HAPPE_dataQC_20-08-2025.csv")

# --- CLI ARGUMENTS ---
parser = argparse.ArgumentParser(description="Convert EEGLAB .set epochs to MNE .fif with behavioral metadata.")
parser.add_argument(
    "--input-set-dir",
    type=str,
    default=os.path.join(BASE_INPUT_DIR, "5 - processed"),
    help="Directory containing cleaned EEGLAB .set files (e.g., '5 - processed (45hz)')",
)
parser.add_argument(
    "--subset",
    type=str,
    choices=["acc1", "acc0", "all"],
    default=None,
    help="Which trials to export: acc1 (correct only), acc0 (incorrect only), or all",
)
parser.add_argument(
    "--name-suffix",
    type=str,
    default=None,
    help="Optional suffix to append to dataset dir names, e.g. '45hz' to create 'acc_1_dataset (45hz)'",
)
args = parser.parse_args()

HAPPE_SET_DIR = args.input_set_dir

# Backwards-compatibility with PROCESS_ACC_ONLY env var
env_acc_only = os.getenv("PROCESS_ACC_ONLY")
if args.subset is None:
    if env_acc_only is None:
        chosen_subset = "acc1"  # historical default
    else:
        chosen_subset = "acc1" if env_acc_only.lower() == "true" else "all"
else:
    chosen_subset = args.subset

# --- DATASET NAME SUFFIX ---
if args.name_suffix is not None and len(args.name_suffix.strip()) > 0:
    suffix_str = args.name_suffix.strip()
    if not suffix_str.startswith("("):
        suffix_str = f" ({suffix_str})"
else:
    lower_dir = HAPPE_SET_DIR.lower()
    if "45hz" in lower_dir:
        suffix_str = " (45hz)"
    elif "30hz" in lower_dir:
        suffix_str = " (30hz)"
    else:
        suffix_str = ""

# --- DYNAMIC OUTPUT PATH ---
if chosen_subset == "acc1":
    base_name = "acc_1_dataset"
elif chosen_subset == "acc0":
    base_name = "acc_0_dataset"
else:
    base_name = "all_trials_dataset"

output_dir = os.path.join(BASE_OUTPUT_DIR, base_name + suffix_str)
# Ensure fresh outputs: remove previous .fif files so we don't mix old and new data
if os.path.exists(output_dir):
    for _f in os.listdir(output_dir):
        if _f.endswith('.fif'):
            try:
                os.remove(os.path.join(output_dir, _f))
            except Exception as _e:
                print(f"Warning: could not delete {_f}: {_e}")

os.makedirs(output_dir, exist_ok=True)
print(f"--- Input .set dir: {HAPPE_SET_DIR}")
print(f"--- Subset: {chosen_subset}")
print(f"--- Outputting to: {output_dir} ---")

# ---------------------------------------------------------------------------
# NEW LABELING HELPERS
# ---------------------------------------------------------------------------

SMALL_SET = {1, 2, 3}
LARGE_SET = {4, 5, 6}


def direction_label(cond):
    """
    Return 'I' for ascending, 'D' for descending, 'NC' for no-change.
    Return pd.NA if *cond* is missing or non-numeric.
    """
    try:
        s = str(int(cond)).zfill(2)
        prime, target = s[0], s[1]
    except (ValueError, TypeError):
        return pd.NA

    if prime == target:
        return "NC"
    return "I" if prime < target else "D"


def transition_category(cond):
    """
    Map condition code to one of iSS, dSS, iLL, dLL, iSL, dLS, or 'NC'.
    """
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
    """
    Return 'SS', 'LL', 'cross', or 'NC'.
    """
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


# --- 2. LOAD MASTER USABLE TRIAL LIST ---
usable_trials_df = pd.read_csv(HAPPE_QC_FILE)
usable_trials_df.rename(columns={usable_trials_df.columns[0]: 'SessionInfo'}, inplace=True)

# --- 3. PRE-SCAN FOR ALL LABELS TO CREATE A GLOBAL ENCODER ---
print("--- Pass 1: Scanning for all unique labels ---")
all_labels = set()

# CORRECTED: Use the BEHAVIORAL_DATA_DIR variable defined in the PATHS section
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

        # Support both legacy and new 45 Hz filenames
        candidate_paths = [
            os.path.join(HAPPE_SET_DIR, f"Subject{subject_id}.set"),
            os.path.join(HAPPE_SET_DIR, f"subject{subject_id}_processed.set"),
            os.path.join(HAPPE_SET_DIR, f"subject{int(subject_id)}_processed.set"),
        ]
        set_file_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if set_file_path is None:
            raise FileNotFoundError(
                f"Could not find .set for Subject {subject_id} in {HAPPE_SET_DIR}. "
                f"Tried: {', '.join(os.path.basename(p) for p in candidate_paths)}"
            )
        epochs = mne.io.read_epochs_eeglab(set_file_path, verbose=False)

        # Resolve subject usable-trials row robustly (zero-padded or not)
        
        # New robust matching for formats like "subject02.mff"
        subject_id_int = int(subject_id)
        # Match against 'subject02.mff', 'subject2.mff' etc.
        # CORRECTED: The f-string requires only a single backslash for the regex escape.
        pattern = re.compile(f"subject0*{subject_id_int}\\.mff", re.IGNORECASE)
        subject_rows = usable_trials_df[usable_trials_df['SessionInfo'].str.match(pattern, na=False)]

        if subject_rows.empty:
            raise ValueError(f"No usable-trials row found for subject {subject_id} in {HAPPE_QC_FILE}")
        subject_row = subject_rows.iloc[0]
        kept_indices_1based = [int(i.strip()) for i in str(subject_row['Kept_Segs_Indxs']).split(',') if str(i).strip() != '']
        
        behavioral_file = os.path.join(BEHAVIORAL_DATA_DIR, f"Subject{subject_id}_lite.csv")
        behavioral_df = pd.read_csv(behavioral_file, on_bad_lines='warn', low_memory=False)
        
        # --- FINAL, SCIENTIFICALLY CORRECT ALIGNMENT LOGIC (Mk V) ---
        
        # 1. Pre-filter the behavioral data to get the subset of trials we are interested in.
        #    This is the scientifically correct order of operations.
        if chosen_subset == "acc1":
            behavioral_subset = behavioral_df[behavioral_df['Target.ACC'] == 1].copy()
        elif chosen_subset == "acc0":
            behavioral_subset = behavioral_df[behavioral_df['Target.ACC'] == 0].copy()
        else: # 'all'
            behavioral_subset = behavioral_df.copy()
        
        # Also remove the unwanted '99' conditions from our target set.
        behavioral_subset = behavioral_subset[behavioral_subset['Condition'] != 99]

        # 2. Get the indices of our desired behavioral subset (0-based)
        behavioral_indices_0based = set(behavioral_subset['CumulativeTrial'] - 1)
        
        # 3. Get the indices from MNE and HAPPE
        mne_kept_0based = set(epochs.selection)
        happe_kept_0based = {idx - 1 for idx in kept_indices_1based}
        
        # 4. The definitive set of trials is the three-way intersection
        final_kept_indices_0based = sorted(list(
            mne_kept_0based.intersection(happe_kept_0based).intersection(behavioral_indices_0based)
        ))
        
        if not final_kept_indices_0based:
            raise ValueError(f"No overlap found for Subject {subject_id} after all filters.")
        
        # 5. Filter the MNE epochs object using the definitive list
        mne_pos_map = {orig_idx: pos for pos, orig_idx in enumerate(epochs.selection)}
        mne_positions_to_keep = [mne_pos_map[idx] for idx in final_kept_indices_0based]
        epochs = epochs[mne_positions_to_keep]
        
        # 6. Filter the original behavioral dataframe using the definitive list for metadata
        final_kept_indices_1based = [i + 1 for i in final_kept_indices_0based]
        behavioral_df_filtered = behavioral_df[behavioral_df['CumulativeTrial'].isin(final_kept_indices_1based)].copy()
        behavioral_df_filtered.sort_values(by='CumulativeTrial', inplace=True)
        
        if len(epochs) != len(behavioral_df_filtered):
            raise ValueError(f"STRICT ALIGNMENT FAIL for Subject {subject_id}")
            
        # Add descriptive columns for number-pair properties
        behavioral_df_filtered["direction"] = behavioral_df_filtered["Condition"].apply(direction_label)
        behavioral_df_filtered["change_group"] = behavioral_df_filtered["Condition"].apply(transition_category)
        behavioral_df_filtered["size"] = behavioral_df_filtered["Condition"].apply(size_category)

        if len(epochs) != len(behavioral_df_filtered):
            raise ValueError(
                f"STRICT ALIGNMENT FAIL for Subject {subject_id}: epochs={len(epochs)} vs behavioral={len(behavioral_df_filtered)}."
            )
        
        encoded_labels = global_le.transform(behavioral_df_filtered['Condition'].astype(str))
        events_from_metadata = np.array([
            np.arange(len(behavioral_df_filtered)),
            np.zeros(len(behavioral_df_filtered), int),
            encoded_labels
        ]).T

        # Let MNE infer event_id from the numeric event codes we provide
        epochs_with_metadata = mne.EpochsArray(
            epochs.get_data(), info=epochs.info, events=events_from_metadata,
            tmin=epochs.tmin, event_id=None, metadata=behavioral_df_filtered
        )

        # No further filtering is needed here as it was all handled pre-alignment.
        final_epochs = epochs_with_metadata
        print(f"  Final trials after all filtering and alignment: {len(final_epochs)}.")

        output_filename = f"sub-{subject_id}_preprocessed-epo.fif"
        output_path = os.path.join(output_dir, output_filename)
        final_epochs.save(output_path, overwrite=True, verbose=False)

        print(f"  -> Saved to {output_path}")

    except Exception as e:
        print(f"!!! FAILED for subject {subject_id}: {e}")
        continue

print("\n--- DATA PREPARATION COMPLETE ---") 