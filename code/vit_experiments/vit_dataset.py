import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

__all__ = [
    'SpectrogramDataset',
    'SpecAugment',
]


def _time_mask(x: torch.Tensor, p: float, mask_frac: float):
    """Randomly mask a vertical band (time) of width = mask_frac*W."""
    if p == 0 or torch.rand(1).item() > p:
        return x
    _, _, H, W = x.shape
    mask_w = max(1, int(mask_frac * W))
    start = torch.randint(0, W - mask_w + 1, (1,)).item()
    x = x.clone()
    x[..., start:start + mask_w] = 0.0
    return x


def _freq_mask(x: torch.Tensor, p: float, mask_frac: float):
    """Randomly mask a horizontal band (frequency)."""
    if p == 0 or torch.rand(1).item() > p:
        return x
    _, _, H, W = x.shape
    mask_h = max(1, int(mask_frac * H))
    start = torch.randint(0, H - mask_h + 1, (1,)).item()
    x = x.clone()
    x[..., start:start + mask_h, :] = 0.0
    return x


class SpecAugment:
    """Apply SpecAugment time & frequency masking on-the-fly (GPU-friendly)"""

    def __init__(self, time_mask_p: float = 0.0, time_mask_frac: float = 0.15,
                 freq_mask_p: float = 0.0, freq_mask_frac: float = 0.15):
        self.time_mask_p = time_mask_p
        self.time_frac = time_mask_frac
        self.freq_mask_p = freq_mask_p
        self.freq_frac = freq_mask_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a 4-D batch or 3-D single image tensor."""
        added_dim = False
        if x.dim() == 3:  # (C,H,W) → add batch dim
            x = x.unsqueeze(0)
            added_dim = True
        # Now x is (B,C,H,W)
        x = _time_mask(x, self.time_mask_p, self.time_frac)
        x = _freq_mask(x, self.freq_mask_p, self.freq_frac)
        if added_dim:
            x = x.squeeze(0)
        return x


class SpectrogramDataset(Dataset):
    """Loads pre-generated 128-channel spectrogram .npy files.

    Each sample returns:
        tensor: (C=128, H, W)  float32 0-1
        label : int
    """

    def __init__(self, root: str | Path, label_col: str = 'landing_digit',
                 transform=None):
        root = Path(root)
        meta_path = root / 'metadata.csv'
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        self.meta = pd.read_csv(meta_path)
        self.root = root
        self.label_col = label_col

        # Optional global Z-score statistics (computed at preprocessing time)
        stats_fp = root / 'stats.json'
        if stats_fp.exists():
            import json
            stats = json.loads(stats_fp.read_text())
            self._mu = float(stats.get('mean', 0.0))
            self._sigma = float(stats.get('std', 1.0)) or 1.0
        else:
            self._mu = None
            self._sigma = None
        self.transform = transform

        # ------------------------------------------------------------------
        # Encode labels to contiguous 0-based indices **in all cases**.  This
        # ensures compatibility with PyTorch CrossEntropyLoss, which expects
        # targets in the range [0, num_classes-1].  We keep a mapping so we
        # can recover the original class names (e.g. digits 1-6).
        # ------------------------------------------------------------------

        classes = sorted(self.meta[label_col].unique())  # e.g. [1,2,3,4,5,6]
        self._cls2idx = {c: i for i, c in enumerate(classes)}
        self._idx2cls = {i: c for c, i in self._cls2idx.items()}

        # Map every row to its integer index               → stored in '_y'
        self.meta['_y'] = self.meta[label_col].map(self._cls2idx).astype(int)

        # Public attributes
        self.classes_ = classes            # original label names, ordered
        self.class_to_idx = self._cls2idx  # convenience for callers

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        fp = self.root / row['filepath']
        arr = np.load(fp)  # (128,H,W)
        if self._mu is not None:
            arr = (arr - self._mu) / self._sigma
        x = torch.from_numpy(arr)
        if self.transform:
            x = self.transform(x)
        y = int(row['_y'])
        return x, y 