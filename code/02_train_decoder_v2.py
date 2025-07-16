# D:\numbers_eeg_nn_project\code\02_train_decoder_v2.py
"""Subject-aware EEG decoder training

Implements GroupKFold (subject-independent) cross-validation and class-weighted
loss to establish a scientifically sound baseline, following Consultant tier-1
recommendations.
"""

import os
import re
import glob
from pathlib import Path

import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, Subset

from torcheeg.models import EEGNet

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
DATASET_DIR = Path(r"D:\numbers_eeg_nn_project\data_preprocessed\acc_1_dataset")
LABEL_COLUMN = "transition_label"

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
N_SPLITS = 5  # subject-aware folds
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda.is_available(): {torch.cuda.is_available()} | device: {DEVICE}")

# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------

def load_concat_epochs(root: Path) -> mne.Epochs:
    """Load all *_preprocessed-epo.fif files, append subject id to metadata, concatenate."""
    fif_files = sorted(root.glob("sub-*_preprocessed-epo.fif"))
    if not fif_files:
        raise FileNotFoundError(f"No .fif files found in {root}")

    epochs_list = []
    subj_pattern = re.compile(r"sub-(\d+)_preprocessed")
    for fpath in fif_files:
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
        subj_match = subj_pattern.search(fpath.name)
        if subj_match is None:
            raise ValueError(f"Cannot extract subject id from filename {fpath.name}")
        subject_id = int(subj_match.group(1))
        # attach subject id into metadata
        epochs.metadata = epochs.metadata.copy() if epochs.metadata is not None else mne.create_info([], 0)
        if 'subject' not in epochs.metadata.columns:
            epochs.metadata['subject'] = subject_id
        else:
            epochs.metadata['subject'] = subject_id  # overwrite to be safe
        epochs_list.append(epochs)

    mega_epochs = mne.concatenate_epochs(epochs_list)
    # ensure LABEL_COLUMN categorical
    mega_epochs.metadata[LABEL_COLUMN] = mega_epochs.metadata[LABEL_COLUMN].astype('category')
    print(f"Loaded {len(fif_files)} subjects -> total epochs: {len(mega_epochs)}")
    return mega_epochs

# ---------------------------------------------------------------------------
# 3. TRAIN / VALID FUNCTIONS
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def eval_epoch(model, loader, loss_fn):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_total += loss_fn(out, y).item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_total / len(loader), 100 * correct / total

# ---------------------------------------------------------------------------
# 4. MAIN WORKFLOW
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # speed-up

    epochs_all = load_concat_epochs(DATASET_DIR)

    # tensors
    data_np = epochs_all.get_data(copy=False) * 1e6  # convert to microvolts
    data_tensor = torch.from_numpy(data_np).float().unsqueeze(1)  # (E,1,C,T)

    # labels
    le = LabelEncoder()
    labels_np = le.fit_transform(epochs_all.metadata[LABEL_COLUMN])
    num_classes = len(le.classes_)
    labels_tensor = torch.from_numpy(labels_np).long()

    # subject groups for CV
    groups_np = epochs_all.metadata['subject'].values

    full_ds = TensorDataset(data_tensor, labels_tensor)

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(data_tensor, labels_tensor, groups_np)):
        print(f"\n========== Fold {fold+1}/{N_SPLITS} ==========")

        # class weights based on training fold only
        cls_weights = compute_class_weight('balanced', classes=np.unique(labels_np), y=labels_np[train_idx])
        cls_weights_t = torch.tensor(cls_weights, dtype=torch.float32).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=cls_weights_t)

        # model init per fold
        _, _, num_electrodes, chunk_size = data_tensor.shape  # (E,1,C,T)
        model = EEGNet(num_classes=num_classes, num_electrodes=num_electrodes, chunk_size=chunk_size).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # data loaders
        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        for epoch in range(1, NUM_EPOCHS + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, loss_fn)
            val_loss, val_acc = eval_epoch(model, val_loader, loss_fn)
            print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Train {tr_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.2f}%")

        # final fold accuracy
        _, final_acc = eval_epoch(model, val_loader, loss_fn)
        fold_accs.append(final_acc)
        print(f"Fold {fold+1} final accuracy: {final_acc:.2f}%")

    print("\n================ Cross-Validation Summary ================")
    print(f"Mean accuracy: {np.mean(fold_accs):.2f}% (+/- {np.std(fold_accs):.2f})")
    print(f"Chance level ({num_classes} classes): {100/num_classes:.2f}%") 