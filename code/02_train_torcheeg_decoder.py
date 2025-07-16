import os

# --- GPU SELECTION ---
# No explicit CUDA_VISIBLE_DEVICES is set here. By default, the primary CUDA-capable GPU (index 0) will be used.

import glob
import re
import mne
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import StratifiedKFold
from torcheeg import transforms
from torcheeg.models import EEGNet

# --- 1. CONFIGURATION ---
# Define the dataset we want to analyze
DATASET_DIR = r"D:\numbers_eeg_nn_project\data_preprocessed\acc_1_dataset"
# The column in our MNE metadata that holds the string labels we want to predict
LABEL_COLUMN = 'transition_label'

# --- HYPERPARAMETERS ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
N_SPLITS = 5
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quick diagnostic to always show CUDA visibility
print(f"cuda.is_available(): {torch.cuda.is_available()} | device_count: {torch.cuda.device_count()}")

# --- Initial Script Printout ---
print("--- Configuration & Verification ---")
if DEVICE.type == 'cuda':
    print(f"Targeting CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"Targeting CPU")
print(f"Using dataset: {DATASET_DIR}")
print("-" * 34)


# --- 2. CUSTOM DATASET CLASS ---
class MNEEpochsDataset(Dataset):
    """Custom PyTorch Dataset for MNE Epochs"""
    def __init__(self, epochs: mne.Epochs):
        self.epochs = epochs
        self.labels = self.epochs.metadata[LABEL_COLUMN].cat.codes.values
        self.classes = self.epochs.metadata[LABEL_COLUMN].unique()
        print("\nDataset Details:")
        print(f"  Total trials: {len(self.epochs)}")
        print(f"  Number of classes: {len(self.classes)}")

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        # Get data and label for the given index
        eeg_data = self.epochs[idx].get_data(copy=False)[0]  # Get the first (and only) epoch in the selection
        label = self.labels[idx]
        
        # Convert to tensor and add a channel dimension
        eeg_tensor = torch.from_numpy(eeg_data).float().unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return eeg_tensor, label_tensor


# --- 3. DATA LOADING ---
def build_dataset(data_dir):
    """Loads all subject .fif files, concatenates them, and creates the Dataset."""
    fif_files = sorted(glob.glob(os.path.join(data_dir, "sub-*.fif")))
    if not fif_files:
        raise FileNotFoundError(f"No .fif files found in {data_dir}")

    all_epochs = [mne.read_epochs(f, preload=True, verbose=False) for f in fif_files]
    epochs = mne.concatenate_epochs(all_epochs)
    
    # Convert the label column to a categorical type
    epochs.metadata[LABEL_COLUMN] = epochs.metadata[LABEL_COLUMN].astype('category')
    
    print(f"Loaded and concatenated {len(fif_files)} subject files into {len(epochs)} total epochs.")
    return MNEEpochsDataset(epochs)


# --- 4. MODEL INITIALIZATION ---
def initialize_model(dataset):
    """Initializes the EEGNet model."""
    num_classes = len(dataset.classes)
    
    # Infer channel and sample count from the dataset
    num_channels, num_timepoints = dataset[0][0].shape[1:]

    model = EEGNet(
        num_classes=num_classes, 
        chunk_size=num_timepoints,
        num_electrodes=num_channels
    ).to(DEVICE)

    print("\n--- Model Architecture ---")
    print(model)
    print("-" * 25)
    
    return model


# --- 5. TRAINING & VALIDATION WORKFLOW ---
def train(model, train_loader, optimizer, loss_fn):
    """Main training loop for one epoch."""
    model.train()
    total_loss = 0
    for eeg_tensor, label_tensor in train_loader:
        eeg_tensor, label_tensor = eeg_tensor.to(DEVICE), label_tensor.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(eeg_tensor)
        loss = loss_fn(outputs, label_tensor)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model, val_loader, loss_fn):
    """Validation loop."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for eeg_tensor, label_tensor in val_loader:
            eeg_tensor, label_tensor = eeg_tensor.to(DEVICE), label_tensor.to(DEVICE)
            
            outputs = model(eeg_tensor)
            loss = loss_fn(outputs, label_tensor)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += label_tensor.size(0)
            correct += (predicted == label_tensor).sum().item()
            
    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy


# --- 6. MAIN EXECUTION ---
if __name__ == '__main__':
    # Build the dataset
    dataset = build_dataset(DATASET_DIR)
    
    # Get labels for stratified splitting
    labels = dataset.labels

    # Use Stratified K-Fold to ensure each fold has a similar distribution of classes.
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n========== FOLD {fold+1}/{N_SPLITS} ==========")
        
        # Initialize model for each fold
        model = initialize_model(dataset)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()

        # Create data subsets and DataLoaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # NOTE: num_workers > 0 can cause issues on Windows. Setting to 0.
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # Training loop for the current fold
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, train_loader, optimizer, loss_fn)
            val_loss, val_acc = validate(model, val_loader, loss_fn)
            
            print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%") 