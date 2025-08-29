#!/usr/bin/env python
"""Optuna tuner for T-TIME hyper-parameters.

Usage:
    python scripts/tune_tta_experiment.py \
        --run-dir results/runs/20250724_1424_landing_digit_cnn \
        --space   configs/landing_digit/optuna_space_tta.yaml \
        --db      sqlite:///optuna_studies/landing_digit_tta.db \
        --trials  50
"""
from __future__ import annotations

import argparse, yaml, sys, json
from pathlib import Path
import optuna

import torch

# -------------------------------------------------------------
# Ensure GPU presence – abort if CUDA is not available
# -------------------------------------------------------------
if not torch.cuda.is_available():
    sys.exit("CUDA device not available – aborting Optuna TTA tuning. Activate a GPU-enabled environment or run on a CUDA-capable machine.")

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_cnn_tta_experiment import run_tta  # type: ignore

p = argparse.ArgumentParser(description="Optuna tuner for TTA")
p.add_argument("--run-dir", required=True, type=Path, help="Folder with fold_##_best.ckpt")
p.add_argument("--space", required=True, type=Path, help="YAML search space")
p.add_argument("--db", required=True, help="SQLite URI e.g. sqlite:///optuna_studies/tta.db")
p.add_argument("--trials", type=int, default=20)
args = p.parse_args()

space = yaml.safe_load(args.space.read_text()) or {}

# Ensure parent directory for SQLite storage exists
if args.db.startswith("sqlite:///"):
    db_path = Path(args.db.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)

def objective(trial: optuna.Trial):
    hp = {"run_dir": str(args.run_dir)}
    for name, spec in space.items():
        m = spec["method"]
        low = float(spec.get("low", 0)) if "low" in spec else None
        high = float(spec.get("high", 0)) if "high" in spec else None
        if m == "uniform":
            hp[name] = trial.suggest_float(name, low, high)
        elif m == "log_uniform":
            hp[name] = trial.suggest_float(name, low, high, log=True)
        elif m == "categorical":
            hp[name] = trial.suggest_categorical(name, spec["choices"])
    return run_tta(hp)

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler, storage=args.db, study_name="tta_tuning", load_if_exists=True)
print(f"Starting study with {args.trials} trials …")
study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
print("Best", study.best_value, study.best_params)

# ------------------------------------------------------------------
# Export study plots as PNGs into timestamped directory
# ------------------------------------------------------------------
from datetime import datetime
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
import matplotlib.pyplot as _plt

ts = datetime.now().strftime("%Y%m%d_%H%M")
plots_root = ROOT / "results" / "optuna_plots" / f"{ts}_tta_tuning"
plots_root.mkdir(parents=True, exist_ok=True)

fig1 = plot_optimization_history(study)
fig1.savefig(plots_root / "history.png", dpi=300, bbox_inches="tight")
_plt.close(fig1)

fig2 = plot_parallel_coordinate(study)
fig2.savefig(plots_root / "parallel.png", dpi=300, bbox_inches="tight")
_plt.close(fig2)

fig3 = plot_param_importances(study)
fig3.savefig(plots_root / "importances.png", dpi=300, bbox_inches="tight")
_plt.close(fig3)

print(f"Saved PNG plots to {plots_root}") 