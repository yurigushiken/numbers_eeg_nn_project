
# D:\numbers_eeg_nn_project\code\03_train_torcheeg.py
import os
import re
import mne
import torch
import numpy as np
import pandas as pd
from torcheeg.datasets import MNEEpochsDataset
from torcheeg.transforms import ToTensor
from torch.utils.data import DataLoader
from torcheeg.models import EEGNet

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
DATASET_DIR = r"D:\numbers_eeg_nn_project\data_preprocessed\acc_1_dataset"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Reproducibility -------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# 2. LOAD PRE-EPOCHED DATA
# -----------------------------------------------------------------------------
print("Loading .fif epochs …")
all_epochs = []
for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.endswith(".fif"):
        continue
    epochs_path = os.path.join(DATASET_DIR, fname)
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    all_epochs.append(epochs)

epochs = mne.concatenate_epochs(all_epochs)
print(f"Loaded {len(all_epochs)} subject files → {len(epochs)} trials")

# -----------------------------------------------------------------------------
# 3. CREATE TorchEEG DATASET
# -----------------------------------------------------------------------------
dataset = MNEEpochsDataset(epochs,
                           transform=ToTensor(),
                           target_transform=lambda x: torch.tensor(x, dtype=torch.long))

# -----------------------------------------------------------------------------
# 4. CREATE DATALOADERS
# -----------------------------------------------------------------------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------------------------------------------------------
# 5. BUILD AND TRAIN MODEL
# -----------------------------------------------------------------------------
model = EEGNet(chunk_size=epochs.get_data().shape[2],
               num_electrodes=epochs.info['nchan'],
               num_classes=len(np.unique(epochs.events[:, 2]))).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nStarting training …")
for epoch in range(25):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/25, Loss: {loss.item():.4f}")

# -----------------------------------------------------------------------------
# 6. EVALUATE
# -----------------------------------------------------------------------------
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print(f"\nTest accuracy: {100. * correct / len(test_loader.dataset):.2f}%") 