"""
Dataset classes for the Dual-Stream CNN.

This module contains two dataset classes:
1. DualStreamDataset: The original class that processes raw .fif files on the fly.
   It is kept for backward compatibility and for comparison.

2. DualStreamPreprocessedDataset: A much more efficient class that reads data
   from a preprocessed format (.npy and .feather files). This is the
   recommended dataset for the dual_stream engine as it significantly
   reduces startup time and data loading overhead.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torchaudio.transforms as T
import pandas as pd
import numpy as np

from code.datasets import RawEEGDataset, BaseDataset


# --- Spectrogram Generation (used by both datasets) ---
def generate_spectrogram(ts_tensor: torch.Tensor, n_fft: int = 128, hop_length: int = 16) -> torch.Tensor:
    """Generates a log-spectrogram from a multi-channel time-series tensor."""
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2,
    )
    spec = spectrogram_transform(ts_tensor)
    log_spec = torch.log(spec + 1e-6)
    return log_spec


# --- ORIGINAL DATASET (kept for compatibility) ---
class DualStreamDataset(RawEEGDataset):
    """
    Original dataset that returns both the raw time-series and a spectrogram
    for each EEG trial, generated on-the-fly from .fif files.
    """
    def __init__(self, cfg, label_fn):
        super().__init__(cfg, label_fn)
        self.spectrogram_n_fft = cfg.get("spectrogram_n_fft", 128)
        self.spectrogram_hop_length = cfg.get("spectrogram_hop_length", 16)

    def __getitem__(self, index: int) -> tuple:
        """
        Returns a tuple of ((timeseries, spectrogram), label) for the given index.
        """
        ts_tensor, label = super().__getitem__(index)
        ts_tensor = ts_tensor.squeeze(0)
        spec_tensor = generate_spectrogram(
            ts_tensor,
            n_fft=self.spectrogram_n_fft,
            hop_length=self.spectrogram_hop_length
        )
        ts_tensor = ts_tensor.unsqueeze(0)
        return (ts_tensor, spec_tensor), label


# --- EFFICIENT PREPROCESSED DATASET ---
class DualStreamPreprocessedDataset(BaseDataset):
    """
    Efficient dataset that reads preprocessed time-series and spectrogram
    data from .npy files, using a .feather metadata file.
    """
    def __init__(self, cfg, label_fn):
        super().__init__(cfg, label_fn)
        
        # Dynamically construct the dataset directory path
        dataset_name = self.cfg.get("dataset_name", "acc_1_dataset") 
        spec_config_name = self.cfg.get("spec_config_name", "128-16")
        full_dataset_name = f"{dataset_name}_{spec_config_name}"
        self.root = Path(self.cfg["dataset_dir_base"]) / full_dataset_name
        
        print(f"Loading preprocessed dataset from: {self.root}")

        self.meta = pd.read_feather(self.root / "metadata.feather")
        
        self.ts_dir = self.root / "ts_data"
        self.spec_dir = self.root / "spec_data"

        # Apply the task-specific label function to the metadata
        self.meta["__y_task"] = label_fn(self.meta)
        
        # Filter out trials that are not relevant to the current task
        valid_mask = ~self.meta["__y_task"].isna()
        if not valid_mask.any():
            raise ValueError(f"No valid samples found for task '{label_fn.__name__}' in {self.root}")
            
        self.meta = self.meta[valid_mask].reset_index(drop=True)

        # Encode the labels
        labels = self.meta["__y_task"]
        if pd.api.types.is_categorical_dtype(labels) and labels.cat.ordered:
            y_np, class_names_raw = pd.factorize(labels, sort=False)
            self.class_names = list(class_names_raw)
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_np = le.fit_transform(labels)
            self.class_names = list(le.classes_)
            
        self.y = torch.from_numpy(y_np).long()
        self.groups = self.meta["subject"].values

        self.in_memory = self.cfg.get("in_memory", True)
        self.ts_data = None
        self.spec_data = None
        if self.in_memory:
            print(f"Loading {len(self.meta)} trials into memory...", flush=True)
            from tqdm import tqdm
            
            indices = self.meta['trial_idx'].values
            self.ts_data = [np.load(self.ts_dir / f"{idx}.npy") for idx in tqdm(indices, desc="Loading Time-Series")]
            self.spec_data = [np.load(self.spec_dir / f"{idx}.npy") for idx in tqdm(indices, desc="Loading Spectrograms")]
            print("...loading complete.", flush=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        if self.in_memory:
            ts_data = self.ts_data[index]
            spec_data = self.spec_data[index]
        else:
            trial_info = self.meta.iloc[index]
            original_idx = trial_info['trial_idx']
            ts_data = np.load(self.ts_dir / f"{original_idx}.npy")
            spec_data = np.load(self.spec_dir / f"{original_idx}.npy")

        # Convert to tensors
        # Time-series needs a "channel" dim for the 1D CNN stream
        ts_tensor = torch.from_numpy(ts_data).float().unsqueeze(0)
        spec_tensor = torch.from_numpy(spec_data).float()
        
        label = self.y[index]

        # Note: Augmentations would be applied here if we add them later
        if self.transform:
            # This is a placeholder; dual-stream augmentations are complex
            # and would need to handle the tuple of tensors.
            pass

        return (ts_tensor, spec_tensor), label

    def get_all_labels(self) -> np.ndarray:
        return self.y.numpy()

    @property
    def num_channels(self):
        # From the time-series data shape
        return self.__getitem__(0)[0][0].shape[1]

    @property
    def time_points(self):
        # From the time-series data shape
        return self.__getitem__(0)[0][0].shape[2]
