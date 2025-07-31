#!/usr/bin/env python
"""
Final Experiment: Applies T-TIME adaptation to the gold-standard CNN checkpoints.
This script is self-contained and uses the project's native data loaders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import deque
from datetime import datetime

# --- Pathing Setup ---
# This ensures we can import from 'code' and 'utils' correctly.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Also expose the 'code' directory itself so that `import run_index` works.
CODE_DIR = ROOT_DIR / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import importlib, sys
# ------------------------------------------------------------------
# Remove stdlib 'code' module if already imported so that our 'code' package
# can be imported without shadowing.
# ------------------------------------------------------------------
if 'code' in sys.modules and getattr(sys.modules['code'], '__path__', None) is None:
    del sys.modules['code']

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torcheeg.models import EEGNet
from scipy.linalg import fractional_matrix_power

# --- Import OUR project's trusted tools ---
# Import dataset loader and the label function directly
from code.eeg_train import load_dataset
from tasks.landing_digit import label_fn
from utils.plots import plot_confusion
from code.eeg_train import CANONICAL_DEFAULTS

# Number of output classes for landing-digit task (digits 1-6 encoded as 0-5)
N_CLASSES = 6

# ============================================================================
# ALGORITHMIC CORE - Surgically extracted from DeepTransferEEG
# We paste the essential functions directly here to avoid all import issues.
# ============================================================================

def EA_online(sample: np.ndarray, R_ref: np.ndarray, n_seen: int) -> np.ndarray:
    """Calculates the incremental Euclidean Alignment reference matrix."""
    cov = np.cov(sample)
    # The first sample initializes the reference matrix.
    if n_seen == 0:
        return cov
    else:
        return (R_ref * n_seen + cov) / (n_seen + 1)

# Helper to get logits regardless of model return type
def _logits(model: nn.Module, x):
    out = model(x)
    if isinstance(out, tuple):
        # DeepTransferEEG backbones return (feat, logits)
        return out[-1]
    return out


def TTIME_core(loader: DataLoader, model: nn.Module, args) -> tuple[np.ndarray, np.ndarray]:
    """
    The core T-TIME adaptation algorithm.
    This function is a simplified, direct copy of the essential logic.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    y_true, y_pred_probs = [], []
    R_ref, num_seen = 0, 0
    window = deque(maxlen=args.test_batch)
    softmax = nn.Softmax(dim=1)
    device = torch.device("cuda" if args.data_env == "gpu" else "cpu")

    for i, (x, y) in enumerate(loader):
        x_cpu = x.float().cpu() # (1, 1, C, T)

        # Phase 1: Prediction
        model.eval()
        with torch.no_grad():
            if args.align:
                sample = x_cpu.numpy().squeeze()
                R_ref = EA_online(sample, R_ref, num_seen)
                try:
                    sqrtRef = fractional_matrix_power(R_ref, -0.5)
                    sample_aligned = np.dot(sqrtRef, sample).reshape(1, 1, args.chn, args.time_sample_num)
                    x_for_pred = torch.from_numpy(sample_aligned).float().to(device)
                except np.linalg.LinAlgError:
                    print("Warning: Singular matrix in EA. Skipping alignment for this sample.")
                    x_for_pred = x.to(device) # Use original sample if alignment fails
            else:
                x_for_pred = x.to(device)
            
            out = _logits(model, x_for_pred)
            probs = softmax(out)
            y_pred_probs.append(probs.cpu().numpy())
            y_true.append(y.item())

        # Phase 2: Adaptation
        model.train()
        window.append(x_for_pred) # Use the aligned sample for adaptation
        if len(window) == args.test_batch and (i + 1) % args.stride == 0:
            batch = torch.cat(list(window), 0).to(device)
            for _ in range(args.steps):
                out_b = _logits(model, batch)
                softmax_out = softmax(out_b / args.t)
                
                # Entropy Minimization Loss
                entropy_loss = -torch.mean(torch.sum(softmax_out * torch.log(softmax_out + 1e-5), dim=1))
                
                # Marginal Distribution Regularization
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
                loss = entropy_loss + gentropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        num_seen += 1

    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.vstack(y_pred_probs)
    y_pred = y_pred_probs.argmax(axis=1)
    
    return y_true, y_pred

# ============================================================================
# Main Experiment Runner
# ============================================================================

def load_source_model(ckpt_path: Path, C: int, T: int, device: torch.device) -> torch.nn.Module:
    """Instantiates and loads your proven CNN model."""
    model = EEGNet(num_classes=N_CLASSES, num_electrodes=C, chunk_size=T, dropout=0.5)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run T-TIME adaptation on CNN checkpoints")
    parser.add_argument("--run-dir", required=True, type=Path, help="Folder with fold_##_best.ckpt")
    parser.add_argument("--out-dir", type=Path, default=None, help="Destination folder; default is auto timestamp under results/runs/")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    cli_args = parser.parse_args()

    device = torch.device("cuda" if cli_args.cuda and torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------------
    # Determine output directory
    # ------------------------------------------------------------------
    if cli_args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        default_name = f"{ts}_landing_digit_tta"
        out_dir = ROOT_DIR / "results" / "runs" / default_name
    else:
        out_dir = cli_args.out_dir

    # If the requested out-dir already exists and is non-empty, append a
    # numeric suffix (_1, _2, …) to create a fresh directory instead of
    # overwriting previous results.
    if out_dir.exists() and any(out_dir.iterdir()):
        idx = 1
        while True:
            candidate = Path(f"{out_dir}_{idx}")
            if not candidate.exists():
                out_dir = candidate; break
            idx += 1
        print(f"Output folder {cli_args.out_dir} exists – writing to {out_dir} instead.")
        out_dir.mkdir(parents=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    cli_args.out_dir = out_dir

    # Load data using YOUR project's trusted loader
    dataset_dir = Path(CANONICAL_DEFAULTS['dataset_dir'])
    ds, groups, classes = load_dataset(dataset_dir, label_fn)
    C = ds.tensors[0].shape[2]
    T = ds.tensors[0].shape[3]
    global N_CLASSES
    N_CLASSES = len(classes)
    print(f"Loaded dataset with {C} channels, {T} time samples, {N_CLASSES} classes.")

    summary, all_y_true, all_y_pred = [], [], []
    ckpts = sorted(cli_args.run_dir.glob("fold_*_best.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {cli_args.run_dir}")

    print(f"Found {len(ckpts)} checkpoints – starting adaptation")
    subject_ids = np.unique(groups)

    for fold_idx, ckpt in enumerate(ckpts):
        target_subject_id = subject_ids[fold_idx]
        print(f"--- Processing Subject {target_subject_id} (Fold {fold_idx}) ---")

        # Create a DataLoader for the target subject
        mask = (groups == target_subject_id)
        target_ds = TensorDataset(ds.tensors[0][mask], ds.tensors[1][mask])
        loader = DataLoader(target_ds, batch_size=1, shuffle=False)

        # Load the corresponding source model checkpoint
        model = load_source_model(ckpt, C, T, device)

# Allow full-network adaptation (previous freeze caused zero gradients)
        for param in model.parameters():
            param.requires_grad = True

        # Lightweight args namespace for the T-TIME algorithm
        class _Args: pass
        tta_args = _Args()
        tta_args.lr = 5e-4  # lower LR to avoid over-shooting
        tta_args.align = False  # baseline: no Euclidean Alignment
        tta_args.chn = C
        tta_args.time_sample_num = T
        tta_args.test_batch = 16
        tta_args.stride = 8   # update every 8th sample
        tta_args.steps = 3   # 3 gradient steps per update
        tta_args.t = 1.0
        tta_args.data_env = "gpu" if device.type == "cuda" else "local"

        # Run the adaptation
        y_true, y_pred = TTIME_core(loader, model, tta_args)
        acc = (y_true == y_pred).mean() * 100.0
        print(f"  → Final Accuracy: {acc:.2f}% ({len(y_true)} trials)")

        summary.append({"subject": int(target_subject_id), "accuracy": acc})
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    mean_acc = np.mean([d["accuracy"] for d in summary])
    std_acc = np.std([d["accuracy"] for d in summary])
    
    # Save final results
    out_json = cli_args.out_dir / "summary_tta_final.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": summary, "mean": mean_acc, "std": std_acc}, f, indent=2)

    cm_path = cli_args.out_dir / "tta_overall_confusion.png"
    class_names = [str(i) for i in range(1, N_CLASSES + 1)]
    plot_confusion(all_y_true, all_y_pred, class_names, cm_path, title=f"T-TIME on CNN Checkpoints\nMean Acc: {mean_acc:.2f}%")

    print(f"\nFinished – Mean Accuracy: {mean_acc:.2f} ± {std_acc:.2f} %. Results saved to {cli_args.out_dir}.")

if __name__ == "__main__":
    main()

# ============================================================================
# Optuna helper (imported by scripts/optuna_tune.py)
# ============================================================================

def run_tta(hp: dict) -> float:
    """Run one full LOSO T-TIME pass with the given hyper-parameters.

    hp must contain at least: lr, steps, test_batch, t, align (bool).
    Returns the mean accuracy.
    The run folder is auto-timestamped under results/runs/.
    """
    lr          = float(hp.get('lr', 1e-3))
    steps       = int(hp.get('steps', 3))
    test_batch  = int(hp.get('test_batch', 16))
    stride      = int(hp.get('stride', max(1, test_batch//2)))
    temp_t      = float(hp.get('t', 1.0))
    align_flag  = bool(hp.get('align', False))

    # ------------------------------------------------------------------
    # Re-use the internal functions defined above, mimicking the CLI path
    # ------------------------------------------------------------------
    dataset_dir = Path(CANONICAL_DEFAULTS['dataset_dir'])
    ds, groups, classes = load_dataset(dataset_dir, label_fn)
    C = ds.tensors[0].shape[2]
    T = ds.tensors[0].shape[3]
    global N_CLASSES
    N_CLASSES = len(classes)

    ckpts_root = Path(hp.get('run_dir', 'results/runs/20250724_1424_landing_digit_cnn'))

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path("results/runs") / f"{ts}_landing_digit_tta_optuna"
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = np.unique(groups)
    ckpts = sorted(ckpts_root.glob("fold_*_best.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in " + str(ckpts_root))

    summary = []
    all_y_true, all_y_pred = [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold_idx, ckpt in enumerate(ckpts):
        sid = subject_ids[fold_idx]
        mask = (groups == sid)
        loader = DataLoader(TensorDataset(ds.tensors[0][mask], ds.tensors[1][mask]), batch_size=1, shuffle=False)

        model = load_source_model(ckpt, C, T, device)

        class _Args: pass
        args = _Args()
        args.lr = lr; args.align = align_flag; args.chn = C; args.time_sample_num = T
        args.test_batch = test_batch; args.stride = stride; args.steps = steps; args.t = temp_t
        args.data_env = 'gpu' if device.type == 'cuda' else 'local'

        y_true, y_pred = TTIME_core(loader, model, args)
        acc = (y_true == y_pred).mean()*100.0
        summary.append(acc)

    mean_acc = float(np.mean(summary))
    # write minimal json for quick reference
    (out_dir / "summary.json").write_text(json.dumps({"hp": hp, "mean_acc": mean_acc}))

    # ---------------- trial-level confusion matrix ----------------
    try:
        from utils.plots import plot_confusion
        cm_path = out_dir / "tta_overall_confusion.png"
        class_names = [str(i + 1) for i in range(N_CLASSES)]
        y_true_int = np.asarray(all_y_true, dtype=int)
        y_pred_int = np.asarray(all_y_pred, dtype=int)
        # If labels are 1-6 convert to 0-5
        if y_true_int.min() == 1 and y_true_int.max() == 6:
            y_true_int -= 1
        if y_pred_int.min() == 1 and y_pred_int.max() == 6:
            y_pred_int -= 1
        plot_confusion(
            y_true_int,
            y_pred_int,
            class_names,
            cm_path,
            title=f"T-TIME on CNN checkpoints\nMean Acc: {mean_acc:.2f}%",
        )
    except Exception as e:
        print(f"Warning: could not plot confusion – {e}")

    return mean_acc