#!/usr/usr/bin/env python
"""
Apply T-TIME adaptation to the LOSO checkpoints produced by `train.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# --- Pathing Setup ---
ROOT_DIR = Path(__file__).resolve().parent.parent
# Ensure project root precedes any site-packages so that our local 'utils'
# package is found before any 3rd-party module named 'utils'.
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR))

# --- Load project utils temporarily so that code.eeg_train imports succeed ---
import importlib.util
_utils_init = ROOT_DIR / 'utils' / '__init__.py'
_plots_path = ROOT_DIR / 'utils' / 'plots.py'

# Create fake 'utils' package for code.eeg_train
spec_utils_pkg = importlib.util.spec_from_file_location('utils', _utils_init)
utils_pkg = importlib.util.module_from_spec(spec_utils_pkg)
spec_utils_pkg.loader.exec_module(utils_pkg)  # type: ignore
sys.modules['utils'] = utils_pkg

# Attach plots submodule
spec_p = importlib.util.spec_from_file_location('utils.plots', _plots_path)
plots_mod = importlib.util.module_from_spec(spec_p)
spec_p.loader.exec_module(plots_mod)  # type: ignore
sys.modules['utils.plots'] = plots_mod
plot_confusion = plots_mod.plot_confusion

# Ensure top-level import 'run_index' resolves to code/run_index.py
run_index_path = ROOT_DIR / 'code' / 'run_index.py'
if run_index_path.exists():
    spec_ri = importlib.util.spec_from_file_location('run_index', run_index_path)
    ri_mod = importlib.util.module_from_spec(spec_ri)
    spec_ri.loader.exec_module(ri_mod)  # type: ignore
    sys.modules['run_index'] = ri_mod

TL_DIR = ROOT_DIR / "third_party" / "DeepTransferEEG" / "tl"
if TL_DIR.exists() and str(TL_DIR) not in sys.path:
    sys.path.insert(0, str(TL_DIR))

# ------------------------------------------------------------------
# Provide a minimal 'utils' package that points at DeepTransferEEG/tl/utils
# so that imports like `from utils.network import backbone_net` resolve.
# Do this *before* importing tta_utils.
# ------------------------------------------------------------------
import types
if 'utils' not in sys.modules:
    utils_stub = types.ModuleType('utils')
    utils_stub.__path__ = [str(TL_DIR / 'utils')]
    sys.modules['utils'] = utils_stub

# Preload utils.network so tta_utils resolves quickly
net_path = TL_DIR / 'utils' / 'network.py'
import importlib.util as _iu
if net_path.exists():
    spec_net = _iu.spec_from_file_location('utils.network', net_path)
    net_mod = _iu.module_from_spec(spec_net)
    spec_net.loader.exec_module(net_mod)  # type: ignore
    sys.modules['utils.network'] = net_mod

# Preload utils.LogRecord for ttime
lr_path = TL_DIR / 'utils' / 'LogRecord.py'
if lr_path.exists():
    spec_lr = _iu.spec_from_file_location('utils.LogRecord', lr_path)
    lr_mod = _iu.module_from_spec(spec_lr)
    spec_lr.loader.exec_module(lr_mod)  # type: ignore
    sys.modules['utils.LogRecord'] = lr_mod

# Preload utils.utils
uu_path = TL_DIR / 'utils' / 'utils.py'
if uu_path.exists():
    spec_uu = _iu.spec_from_file_location('utils.utils', uu_path)
    uu_mod = _iu.module_from_spec(spec_uu)
    spec_uu.loader.exec_module(uu_mod)  # type: ignore
    sys.modules['utils.utils'] = uu_mod

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torcheeg.models import EEGNet

# --- Import OUR project's trusted tools ---
# Now import code.eeg_train which expects utils.plots
from code.eeg_train import load_dataset
import tasks.landing_digit as task

# --- Import the T-TIME algorithm ---
from third_party.DeepTransferEEG.tl.tta_utils import ttime_adapt

# --- Get constants from your project's training script ---
from code.eeg_train import CANONICAL_DEFAULTS

# These are now defined by your project's data, not hardcoded
N_CLASSES = 6

def load_model(ckpt_path: Path, C: int, T: int, device: torch.device) -> torch.nn.Module:
    """Instantiate EEGNet with the same hyper-params used in cnn engine."""
    model = EEGNet(
        num_classes=N_CLASSES,
        num_electrodes=C,
        chunk_size=T,
        dropout=0.5,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def main() -> None:
    parser = argparse.ArgumentParser(description="Run T-TIME adaptation on CNN checkpoints")
    parser.add_argument("--run-dir", required=True, type=Path, help="Folder with fold_##_best.ckpt")
    parser.add_argument("--out-dir", required=True, type=Path, help="Where results (plots & csv) are saved")
    parser.add_argument("--cuda", action="store_true", help="Force CUDA if available")
    cli_args = parser.parse_args()

    device = torch.device("cuda" if cli_args.cuda and torch.cuda.is_available() else "cpu")
    cli_args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- THIS IS THE CRITICAL CHANGE: Load data THE RIGHT WAY ---
    # We use the exact same data loader as your successful cnn engine.
    # This guarantees consistency in channels, labels, and preprocessing.
    dataset_dir = Path(CANONICAL_DEFAULTS['dataset_dir'])
    ds, groups, (C, T) = load_dataset(dataset_dir, task.label_fn)
    # ds is a TensorDataset with (X, y)
    # groups is a numpy array of subject IDs
    # C and T are the correct number of channels and time samples

    print(f"Loaded dataset with {C} channels and {T} time samples.")
    # --- END CRITICAL CHANGE ---

    summary = []
    all_y_true, all_y_pred = [], []
    
    ckpts = sorted(cli_args.run_dir.glob("fold_*_best.ckpt"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found in {cli_args.run_dir}")

    print(f"Found {len(ckpts)} checkpoints – starting adaptation")

    # The mapping from fold index to subject ID is based on the sorted unique subject IDs
    subject_ids = np.unique(groups)

    for fold_idx, ckpt in enumerate(ckpts):
        target_subject_id = subject_ids[fold_idx]
        print(f"Subject {target_subject_id}: loading model & data …")

        # Create a DataLoader for the current target subject
        mask = (groups == target_subject_id)
        target_ds = TensorDataset(ds.tensors[0][mask], ds.tensors[1][mask])
        loader = DataLoader(target_ds, batch_size=1, shuffle=False)

        # Load the corresponding checkpoint
        model = load_model(ckpt, C, T, device)

        # Build lightweight args namespace for T-TIME
        class _Args: pass
        tta_args = _Args()
        tta_args.lr = 1e-5 # Start with the very low LR that showed promise
        tta_args.align = True
        tta_args.class_num = N_CLASSES
        tta_args.chn = C
        tta_args.time_sample_num = T
        tta_args.test_batch = 16
        tta_args.stride = 1
        tta_args.steps = 1
        tta_args.t = 1.0
        tta_args.epsilon = 1e-5
        tta_args.data_env = "gpu" if device.type == "cuda" else "local"

        y_true, y_pred = ttime_adapt(loader, model, tta_args)
        acc = (y_true == y_pred).mean() * 100.0
        print(f"  → Acc after T-TIME: {acc:.2f}% ({len(y_true)} trials)")

        summary.append({"subject": target_subject_id, "accuracy": acc})
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    mean_acc = np.mean([d["accuracy"] for d in summary])
    std_acc = np.std([d["accuracy"] for d in summary])
    
    # Save summary and confusion matrix
    out_json = cli_args.out_dir / "summary_ttime.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": summary, "mean": mean_acc, "std": std_acc}, f, indent=2)

    cm_path = cli_args.out_dir / "tta_overall_confusion.png"
    class_names = [str(i) for i in range(1, N_CLASSES + 1)] # Labels are 1-6
    plot_confusion(all_y_true, all_y_pred, class_names, cm_path, title=f"T-TIME on CNN checkpoints\nMean Acc: {mean_acc:.2f}%")

    print(f"\nFinished – mean acc {mean_acc:.2f} ± {std_acc:.2f} %. Results saved to {cli_args.out_dir}.")

if __name__ == "__main__":
    main()
    