"""
Dataset classes for the unified training framework.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Any, List, Dict, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, TensorDataset
try:
    from torcheeg import transforms
    _HAS_TORCHEEG = True
except Exception:
    _HAS_TORCHEEG = False

_CACHE: Dict[str, Any] = {}

class BaseDataset(Dataset):
    """Abstract base class for datasets in this project."""
    def __init__(self, cfg: Dict[str, Any], label_fn: Callable):
        self.cfg = cfg
        self.label_fn = label_fn
        self.transform = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def get_all_labels(self) -> np.ndarray:
        raise NotImplementedError

    def set_transform(self, transform: Callable | None):
        self.transform = transform

class RawEEGDataset(BaseDataset):
    """Dataset for loading and preprocessing raw MNE epochs."""
    def __init__(self, cfg: Dict[str, Any], label_fn: Callable):
        super().__init__(cfg, label_fn)
        self.root = Path(self.cfg["dataset_dir"])
        self.X, self.y, self.groups, self.class_names = self._load_data()
        
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[str]]:
        """Loads and caches data from .fif files."""
        cache_key = f"{self.root.resolve()}::{self.label_fn.__name__}"
        if cache_key in _CACHE:
            return _CACHE[cache_key]

        files = sorted(self.root.glob("sub-*preprocessed-epo.fif"))
        if not files:
            raise FileNotFoundError(f"No .fif files found in {self.root}")

        sid_re = re.compile(r"sub-(\d+)_preprocessed")
        epochs_list = []
        for fp in files:
            ep = mne.read_epochs(fp, preload=True, verbose=False)
            sid_match = sid_re.search(fp.name)
            if sid_match:
                sid = int(sid_match.group(1))
                ep.metadata["subject"] = sid
            epochs_list.append(ep)
        
        all_ep = mne.concatenate_epochs(epochs_list)
        all_ep.metadata["__y"] = self.label_fn(all_ep.metadata)
        all_ep = all_ep[~all_ep.metadata["__y"].isna()]

        # --- NEW: Opt-in channel exclusion ---
        list_name_to_use = self.cfg.get("use_channel_list")
        if list_name_to_use:
            print(f"INFO: Attempting to exclude channel list: '{list_name_to_use}'")
            all_channel_lists = self.cfg.get("channel_lists", {})
            channels_to_exclude = all_channel_lists.get(list_name_to_use)
            
            if channels_to_exclude:
                ch_to_drop = [ch for ch in channels_to_exclude if ch in all_ep.ch_names]
                if ch_to_drop:
                    print(f"INFO: Excluding {len(ch_to_drop)} channels...")
                    all_ep.drop_channels(ch_to_drop)
                    print(f"INFO: Remaining channels: {len(all_ep.ch_names)}")
            else:
                print(f"WARNING: Channel list '{list_name_to_use}' not found in config. Using all channels.")

        # --- Preprocessing ---
        X = all_ep.get_data(copy=False).astype(np.float32) * 1e6 # V to uV
        if X.shape[1] > 128: # Truncate channels if necessary
            X = X[:, :128, :]
        
        if _HAS_TORCHEEG:
            X = transforms.MeanStdNormalize(axis=-1)(eeg=X)['eeg']
        else:
            # Fallback normalization along time axis
            mu = X.mean(axis=-1, keepdims=True)
            sd = X.std(axis=-1, keepdims=True) + 1e-6
            X = (X - mu) / sd
        X_t = torch.from_numpy(X).float().unsqueeze(1)
        
        # --- Label Encoding ---
        # Check if the label_fn returned an ordered categorical series
        labels = all_ep.metadata["__y"]
        if pd.api.types.is_categorical_dtype(labels) and labels.cat.ordered:
            y_np, class_names_raw = pd.factorize(labels, sort=False)
            class_names = list(class_names_raw)
        else:
            le = LabelEncoder()
            y_np = le.fit_transform(labels)
            class_names = list(le.classes_)
        
        y_t = torch.from_numpy(y_np).long()

        groups = all_ep.metadata["subject"].values
        
        result = (X_t, y_t, groups, class_names)
        _CACHE[cache_key] = result
        return result

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x_item = self.X[index]
        y_item = self.y[index]
        if self.transform:
            x_item = self.transform(x_item)
        return x_item, y_item

    def get_all_labels(self) -> np.ndarray:
        return self.y.numpy()

    @property
    def num_channels(self):
        return self.X.shape[2]

    @property
    def time_points(self):
        return self.X.shape[3]

    @property
    def channel_names(self) -> List[str]:
        # After dropping channels, the definitive source is the MNE object's ch_names
        # We need to re-create it to be sure, as the instance isn't stored.
        # This is a bit redundant but guarantees correctness.
        cache_key = f"{self.root.resolve()}::{self.label_fn.__name__}::__ch_names__"
        if cache_key in _CACHE:
            return _CACHE[cache_key]

        files = sorted(self.root.glob("sub-*preprocessed-epo.fif"))
        all_ep = mne.concatenate_epochs([mne.read_epochs(fp, preload=False, verbose=False) for fp in files])
        
        list_name_to_use = self.cfg.get("use_channel_list")
        if list_name_to_use:
            all_channel_lists = self.cfg.get("channel_lists", {})
            channels_to_exclude = all_channel_lists.get(list_name_to_use)
            if channels_to_exclude:
                ch_to_drop = [ch for ch in channels_to_exclude if ch in all_ep.ch_names]
                if ch_to_drop:
                    all_ep.drop_channels(ch_to_drop)
        
        result = all_ep.ch_names
        _CACHE[cache_key] = result
        return result

    @property
    def sfreq(self) -> float:
        cache_key = f"{self.root.resolve()}::__sfreq__"
        if cache_key in _CACHE:
            return _CACHE[cache_key]
        
        fp = next(self.root.glob("sub-*preprocessed-epo.fif"))
        info = mne.io.read_info(fp)
        result = info['sfreq']
        _CACHE[cache_key] = result
        return result

class SpectrogramDataset(BaseDataset):
    """Dataset for loading spectrograms and metadata."""
    def __init__(self, cfg: Dict[str, Any], label_fn: Callable):
        super().__init__(cfg, label_fn)
        self.root = Path(self.cfg["dataset_dir"])
        self.in_memory = self.cfg.get("in_memory", False)
        
        self.meta = pd.read_csv(self.root / "metadata.csv")
        self.files = self.meta["file_path"].apply(lambda p: self.root / p).tolist()
        
        # Apply label function
        self.meta["__y_task"] = label_fn(self.meta)
        valid_mask = ~self.meta["__y_task"].isna()
        
        if not valid_mask.any():
            raise ValueError("No valid samples after applying label_fn.")
            
        self.meta = self.meta[valid_mask].reset_index(drop=True)
        self.files = [self.files[i] for i in valid_mask[valid_mask].index]

        # --- Label Encoding ---
        labels = self.meta["__y_task"]
        if pd.api.types.is_categorical_dtype(labels) and labels.cat.ordered:
            y_np, class_names_raw = pd.factorize(labels, sort=False)
            self.class_names = list(class_names_raw)
        else:
            le = LabelEncoder()
            y_np = le.fit_transform(labels)
            self.class_names = list(le.classes_)
            
        self.y = y_np
        
        if "subject" not in self.meta.columns:
            raise ValueError("metadata.csv must contain a 'subject' column for LOSO.")
        self.groups = self.meta["subject"].values

        self.data = None
        if self.in_memory:
            self.data = [np.load(f) for f in self.files]
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.in_memory:
            x_item = self.data[index]
        else:
            x_item = np.load(self.files[index])

        # Add channel dimension, convert to float32 tensor
        x_item = torch.from_numpy(x_item).float().unsqueeze(0)
        
        y_item = self.y[index]

        if self.transform:
            x_item = self.transform(x_item)
            
        return x_item, y_item

    def get_all_labels(self) -> np.ndarray:
        return self.y
        
    @property
    def num_channels(self):
        # Spectrograms are single-channel images
        return 1

    @property
    def img_size(self):
        return self.__getitem__(0)[0].shape[-1]
