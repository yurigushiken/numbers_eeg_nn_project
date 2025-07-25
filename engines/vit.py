"""Vision-Transformer training engine (spectrogram input).

Ported from the old `code/vit_experiments/02_train_vit_…` wrapper but rewritten
in a reusable form so *any* task can call it via `train.py --engine vit`.

Currently supports the landing-digit dataset or any other task whose
`metadata.csv` contains the columns required by the `label_fn`.
"""

from __future__ import annotations

from typing import Dict, Callable, List, Sequence
from pathlib import Path
import datetime, sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score

from utils.plots import plot_confusion, plot_curves
from vit_experiments.vit_dataset import SpectrogramDataset, SpecAugment
from vit_experiments.models.eeg_vit import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

ENGINE_DEFAULTS: Dict = {
    "model_name": "vit_relpos_medium_patch16_224",
    "img_size": 128,
    "batch_size": 32,
    "lr": 3e-4,
    "epochs": 50,
    "early_stop": 10,
    # SpecAugment defaults (can reuse time_mask_p keys from canonical defaults)
    "freq_mask_p": 0.0,
    "freq_mask_frac": 0.15,
    # Domain-adapter on by default
    "domain_adapter": True,
}


def _numeric_cast(cfg: Dict):
    """Ensure numeric cfg values are correct dtype (mirrors eeg_train)."""
    int_keys = ["batch_size", "epochs", "early_stop"]
    float_keys = ["lr", "freq_mask_p", "freq_mask_frac", "time_mask_p", "time_mask_frac"]
    for k in int_keys:
        if k in cfg and cfg[k] is not None and not isinstance(cfg[k], int):
            cfg[k] = int(cfg[k])
    for k in float_keys:
        if k in cfg and cfg[k] is not None and not isinstance(cfg[k], float):
            cfg[k] = float(cfg[k])


def _make_transforms(cfg: Dict):
    return SpecAugment(
        time_mask_p=cfg.get("time_mask_p", 0.0),
        time_mask_frac=cfg.get("time_mask_frac", 0.15),
        freq_mask_p=cfg.get("freq_mask_p", 0.0),
        freq_mask_frac=cfg.get("freq_mask_frac", 0.15),
    )


def run(cfg: Dict, label_fn: Callable):
    """Train ViT on spectrograms. Returns summary dict with mean/std accuracy."""

    # Merge defaults without overwriting explicit cfg keys
    for k, v in ENGINE_DEFAULTS.items():
        cfg.setdefault(k, v)

    _numeric_cast(cfg)

    run_dir = Path(cfg["run_dir"])

    # ---------------- Dataset loading ----------------
    root = Path(cfg["dataset_dir"])
    if not root.exists():
        sys.exit(f"Spectrogram dataset dir not found: {root}")

    # Load once; we will create Subset views per fold
    base_ds = SpectrogramDataset(root, transform=None, in_memory=False)

    # Override labels via task-specific label_fn → ensures generic task support
    base_ds.meta["__y_task"] = label_fn(base_ds.meta)
    valid_mask = ~base_ds.meta["__y_task"].isna()
    if valid_mask.sum() == 0:
        sys.exit("No valid samples after applying label_fn – check task/dataset")

    # Encode to contiguous integers
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    base_ds.meta.loc[valid_mask, "_y"] = le.fit_transform(base_ds.meta.loc[valid_mask, "__y_task"])
    class_names: List[str] = list(le.classes_)
    num_cls = len(class_names)

    # Utility to fetch y given idx list (for class-weight computation)
    _y_all = base_ds.meta.loc[valid_mask, "_y"].astype(int).values

    # subject groups for LOSO
    if "subject" not in base_ds.meta.columns:
        sys.exit("metadata.csv must contain a 'subject' column for LOSO split")
    groups_all: np.ndarray = base_ds.meta.loc[valid_mask, "subject"].values

    # indices of valid samples
    valid_indices: np.ndarray = np.where(valid_mask.values)[0]

    # Pre-compute class weights (full dataset)
    cls_w = compute_class_weight("balanced", classes=np.arange(num_cls), y=_y_all)
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)

    log_accs: List[float] = []
    overall_true: List[int] = []
    overall_pred: List[int] = []

    scaler = GradScaler(enabled=USE_AMP)

    # LOSO splitter
    logo = LeaveOneGroupOut().split(np.zeros_like(_y_all), _y_all, groups_all)

    for fold, (tr_raw_idx, va_raw_idx) in enumerate(logo):
        # Optional cap on number of folds (smoke-tests)
        if cfg.get("max_folds") is not None and fold >= int(cfg["max_folds"]):
            break

        # Map raw subset indices to original dataset indices
        tr_idx = valid_indices[tr_raw_idx]
        va_idx = valid_indices[va_raw_idx]

        # Data augmentation only on train set
        aug = _make_transforms(cfg)
        base_ds.transform = None  # reset
        tr_ds = Subset(base_ds, tr_idx)
        tr_ds.dataset.transform = aug  # type: ignore
        va_ds = Subset(base_ds, va_idx)
        va_ds.dataset.transform = None  # type: ignore

        tr_ld = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
        va_ld = DataLoader(va_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

        # ---------------- Model ----------------
        model = build_model(
            cfg["model_name"],
            num_classes=num_cls,
            img_size=cfg.get("img_size", 128),
            in_chans=128,
            domain_adapter=cfg.get("domain_adapter", True),
        ).to(DEVICE)

        opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=float(cfg.get("weight_decay", 0.0) or 0.0))
        sched = CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-5)
        loss_fn = nn.CrossEntropyLoss(cls_w_t)

        best_val = float("inf")
        best_acc = 0.0
        patience = 0
        tr_hist: List[float] = []
        va_hist: List[float] = []
        acc_hist: List[float] = []

        for epoch in range(1, cfg["epochs"] + 1):
            # ---- train ----
            model.train()
            tot_loss = 0.0
            for xb, yb in tr_ld:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                opt.zero_grad()
                with autocast(enabled=USE_AMP):
                    out = model(xb)
                    loss = loss_fn(out.float(), yb)
                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward(); opt.step()
                tot_loss += loss.item()
            tr_loss = tot_loss / len(tr_ld)

            # ---- val ----
            model.eval(); val_loss = 0.0; corr = 0; n = 0
            y_true_fold: List[int] = []
            y_pred_fold: List[int] = []
            with torch.no_grad():
                for xb, yb in va_ld:
                    xb = xb.to(DEVICE)
                    with autocast(enabled=USE_AMP):
                        out = model(xb)
                        loss = loss_fn(out.float(), yb.to(DEVICE))
                    val_loss += loss.item()
                    pred = out.argmax(1).cpu()
                    corr += (pred == yb).sum().item()
                    n += yb.size(0)
                    y_true_fold.extend(yb.tolist())
                    y_pred_fold.extend(pred.tolist())
            val_loss /= len(va_ld)
            val_acc = 100 * corr / n
            sched.step()

            tr_hist.append(tr_loss); va_hist.append(val_loss); acc_hist.append(val_acc)

            if val_loss < best_val:
                best_val = val_loss
                best_acc = val_acc
                patience = 0
            else:
                patience += 1
            if patience >= cfg["early_stop"]:
                break

            print(f"Fold {fold+1} | Ep {epoch:03d} | Tr {tr_loss:.3f} | Val {val_loss:.3f} | Acc {val_acc:.2f}%", flush=True)

        log_accs.append(best_acc)
        overall_true.extend(y_true_fold)
        overall_pred.extend(y_pred_fold)

        # ---------- plots ----------
        fold_title = f"Fold {fold+1} · Acc {best_acc:.1f}%"
        plot_confusion(
            y_true_fold,
            y_pred_fold,
            class_names,
            run_dir / f"fold{fold+1}_confusion.png",
            title=fold_title,
        )
        plot_curves(
            tr_hist,
            va_hist,
            acc_hist,
            run_dir / f"fold{fold+1}_curves.png",
            title=fold_title,
        )

    # Overall confusion
    macro_f1 = f1_score(overall_true, overall_pred, average="macro") * 100
    plot_confusion(
        overall_true,
        overall_pred,
        class_names,
        run_dir / "overall_confusion.png",
        title=f"Overall · Mean {np.mean(log_accs):.1f}% · Macro-F1 {macro_f1:.1f}%",
    )

    return {
        "mean_acc": float(np.mean(log_accs)),
        "std_acc": float(np.std(log_accs)),
    } 