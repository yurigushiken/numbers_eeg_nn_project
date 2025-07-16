# this script is archived because it is not working.
# D:\numbers_eeg_nn_project\code\02_train_decoder.py
"""Train an EEGNetv4 decoder on the numerical-change EEG study using the
current (0.8+) Braindecode API.  This script strictly follows the official
examples: https://braindecode.org/stable/auto_examples/

Key differences from the previous attempt:
1. We directly convert the already pre-epoched .fif files with
   `create_from_mne_epochs` – no manual RawArray/annotation fiddling.
2. Balancing (SMOTE) is omitted for the first successful run; we can add it
   later once the pipeline is stable.
3. Skorch's `predefined_split` is used for a proper validation split.
"""

import os
import re
import mne
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from braindecode.datasets import BaseDataset, BaseConcatDataset, WindowsDataset
from braindecode.models import EEGNetv4
from braindecode.training import CroppedLoss
from braindecode.classifier import EEGClassifier
from braindecode.util import set_random_seeds
from skorch.helper import predefined_split
from braindecode.datasets import BaseDataset

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
DATASET_DIR = r"D:\numbers_eeg_nn_project\data_preprocessed\acc_1_dataset"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Reproducibility -------------------------------------------------------------
set_random_seeds(seed=SEED, cuda=torch.cuda.is_available())

# -----------------------------------------------------------------------------
# 2. LOAD PRE-EPOCHED DATA AND BUILD WindowDataset
# -----------------------------------------------------------------------------
print("Loading .fif epochs …")
base_datasets = []
for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.endswith(".fif"):
        continue
    path = os.path.join(DATASET_DIR, fname)
    epochs = mne.read_epochs(path, preload=True, verbose=False)

    # Attach label column once, guaranteed to persist
    epochs.metadata = pd.DataFrame({
        "target": epochs.events[:, 2]      # already 0-indexed integers
    })

    sid = int(re.search(r"\d+", fname).group())
    base_datasets.append(BaseDataset(epochs, description={"subject": sid}))

# bundle all subjects
concat_ds = BaseConcatDataset(base_datasets)

# Manually create windows and metadata
X, y, metadata = [], [], []
sfreq = concat_ds.datasets[0].raw.info["sfreq"]
win_size = int(0.8 * sfreq)
stride = int(0.1 * sfreq)

for ds in concat_ds.datasets:
    n_trials, n_chans, n_times = ds.raw.get_data().shape
    for i in range(n_trials):
        for start in range(0, n_times - win_size + 1, stride):
            X.append(ds.raw.get_data()[i, :, start : start + win_size])
            y.append(ds.raw.metadata.iloc[i]["target"])
            metadata.append({"subject": ds.description["subject"], "target": ds.raw.metadata.iloc[i]["target"]})

X = np.array(X)
y = np.array(y)
metadata = pd.DataFrame(metadata)

windows_dataset = WindowsDataset(X, y=y, description=metadata)

print(f"Windows created: {len(windows_dataset)}")

# -----------------------------------------------------------------------------
# 3. SUBJECT-WISE SPLIT: train / val / test
# -----------------------------------------------------------------------------
splits = windows_dataset.split("subject")      # 80/10/10 by default
train_set = splits["train"]
valid_set = splits["valid"]
test_set  = splits["test"]
print(f"Train/Val/Test sets created: {len(train_set)} / {len(valid_set)} / {len(test_set)}")


# -----------------------------------------------------------------------------
# 4. MODEL & CLASSIFIER
# -----------------------------------------------------------------------------
# Basic geometry
# -----------------------------------------------------------------------------
# Get input dimensions
N_CHANS, INPUT_WINDOW_SAMPLES = windows_dataset[0][0].shape
N_CLASSES = len(np.unique(windows_dataset.y))

clf = EEGClassifier(
    EEGNetv4(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=INPUT_WINDOW_SAMPLES,
    ),
    cropped=True,
    criterion=CroppedLoss,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    batch_size=64,
    max_epochs=50,          # keep short for first run; adjust later
    device=DEVICE,
    train_split=predefined_split(valid_set),
    verbose=1,
)

# -----------------------------------------------------------------------------
# 5. TRAIN
# -----------------------------------------------------------------------------
print("\nStarting training …")
clf.fit(train_set, y=None, epochs=25)

# -----------------------------------------------------------------------------
# 6. EVALUATE
# -----------------------------------------------------------------------------
test_accuracy = clf.score(test_set, y=None)
print(f"\nTest accuracy: {test_accuracy:.4f}") 