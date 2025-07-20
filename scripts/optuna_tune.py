#!/usr/bin/env python
"""Universal Optuna wrapper around the existing *02_train_decoder_*.py* scripts.

Usage (PowerShell):
  conda activate torcheeg-env
  python scripts/optuna_tune.py \
      --decoder code/02_train_decoder_direction_binary.py \
      --base    configs/direction_binary/base.yaml \
      --study   sqlite:///direction_optuna.db \
      --trials  40

The script
  • defines a fixed search-space of nine hyper-parameters (see PARAM_RANGES).
  • for every Optuna trial it spawns the decoder as a subprocess passing the
    sampled parameters via --set k=v …
  • reads *summary_*.json* from the newly created *results/runs/<RUN_ID>/*
    directory and returns `mean_acc` as the objective value to maximise.
  • is fully resumable thanks to the SQLite storage backend.
  • writes interactive Plotly HTML plots to *results/optuna_plots/* when done.

Nothing inside the decoders is modified – this is Phase-1 “wrapper mode”.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
import textwrap
import time
import yaml
import optuna
import optuna.visualization
import plotly
from pathlib import Path
import sys as _sys
from pathlib import Path as _P
# ensure 'code' dir is on sys.path so we can import eeg_train from anywhere
proj_root = _P(__file__).resolve().parent.parent
code_dir = proj_root / 'code'
if str(code_dir) not in _sys.path:
    _sys.path.insert(0, str(code_dir))

import eeg_train as et  # shared engine, now resolvable
import importlib
from typing import Dict, Any # Added missing import

# --- Constants ---
# All hyper-parameter ranges must be defined here for Optuna to sample from.
# The names must match the keys in eeg_train.CANONICAL_DEFAULTS or the decoder's base YAML.
PARAM_RANGES = {
    # name        : (type, low, high, [scale]) or (type, [choices])
    "lr"            : ("float", 1e-5, 1e-3, "log"),
    "batch_size"    : ("categorical", [32, 64, 128]),
    "mixup_alpha"   : ("float", 0.0, 0.5),
    "time_mask_p"   : ("float", 0.0, 0.8),
    "time_mask_frac": ("float", 0.05, 0.30),
    "chan_mask_p"   : ("float", 0.0, 0.8),
    "chan_mask_ratio": ("float", 0.05, 0.30),
    "noise_std"     : ("float", 0.0, 0.05),
    "channel_dropout_p": ("float", 0.0, 0.5),
    "shift_p"       : ("float", 0.0, 1.0),
    "shift_min_frac": ("float", 0.001, 0.01),
    "shift_max_frac": ("float", 0.02, 0.08),
    "scale_p"       : ("float", 0.0, 1.0),
    "scale_min"     : ("float", 0.7, 0.95),
    "scale_max"     : ("float", 1.05, 1.3),
    "early_stop"    : ("categorical", [10, 15, 20]), # make explicit if it's sweepable
    "epochs"        : ("categorical", [50, 80, 100]), # make explicit if it's sweepable
}

# --- CLI arguments ---
ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                             description=textwrap.dedent("""Universal Optuna wrapper around the existing *02_train_decoder_*.py* scripts."""))
ap.add_argument("--decoder", type=Path, required=True, help="Path to the decoder script (e.g., code/02_train_decoder_direction_binary.py)")
ap.add_argument("--base", type=Path, required=True, help="Path to the base YAML config file")
ap.add_argument("--study", type=str, required=True, help="Optuna study name or path to SQLite DB (e.g., sqlite:///db.sqlite)")
ap.add_argument("--trials", type=int, default=20, help="Number of Optuna trials (default 20)")
ap.add_argument("--space", type=Path, help="Optional YAML file describing the search space (overrides built-in PARAM_RANGES)")
ap.add_argument("--sampler-seed", type=int, help="Seed for TPESampler (optional)")
ap.add_argument("--set", nargs="*", metavar="KEY=VAL", help="Static overrides passed to every trial, e.g. --set max_folds=3 epochs=80")

args = ap.parse_args()

# -------------------------------------------------------------
# Optional: override PARAM_RANGES via YAML search-space file
# -------------------------------------------------------------

def _load_yaml_space(fp: Path):
    import yaml, math
    txt = fp.read_text()
    raw = yaml.safe_load(txt) or {}
    space: dict[str, tuple] = {}
    for k, v in raw.items():
        method = v.get('method')
        if method in ('uniform', 'log_uniform'):
            low, high = float(v['low']), float(v['high'])
            scale = 'log' if method == 'log_uniform' else None
            space[k] = ('float', low, high, scale) if scale else ('float', low, high)
        elif method == 'categorical':
            space[k] = ('categorical', v['choices'])
        else:
            raise ValueError(f"Unknown method '{method}' for param {k}")
    return space

if args.space:
    PARAM_RANGES = _load_yaml_space(args.space)

# parse fixed overrides (args.set)
fixed_overrides = {}
if args.set:
    for kv in args.set:
        if "=" not in kv:
            sys.exit("--set expects KEY=VAL pairs, got {kv}")
        k, v = kv.split('=', 1)
        fixed_overrides[k] = yaml.safe_load(v)

# -------------------------------------------------------------
# Optuna objective
# -------------------------------------------------------------

# The objective receives a `trial` object and returns the score to optimize.
# This is what Optuna calls for each parameter set.

def objective_factory(task_module_name: str, decoder_path: Path, base_yaml: Path, fixed: Dict[str, Any]):
    """Return a callable `objective(trial)` for Optuna."""
    # ---------------------------------------------------------------------------
    # HACK: dynamic import of task module to get label_fn and BASE_YAML
    # This is a temporary hack for Phase 1. In Phase 2, the `run_loso` call
    # will be direct within this script, and BASE_YAML will be passed from CLI.
    # ---------------------------------------------------------------------------
    task_module = importlib.import_module(task_module_name)

    def objective(trial: optuna.Trial):
        params = {}
        # Sample hyper-parameters from predefined ranges
        for name, spec in PARAM_RANGES.items():
            if name in fixed: # Fixed overrides take precedence
                params[name] = fixed[name]
                continue
            
            p_type = spec[0]
            if p_type == "float":
                low, high = spec[1], spec[2]
                log_scale = spec[3] == "log" if len(spec) > 3 else False
                params[name] = trial.suggest_float(name, low, high, log=log_scale)
            elif p_type == "int": # for integer values, e.g., for model architectures
                low, high = spec[1], spec[2]
                params[name] = trial.suggest_int(name, low, high)
            elif p_type == "categorical":
                choices = spec[1]
                params[name] = trial.suggest_categorical(name, choices)
            else:
                raise ValueError(f"Unknown parameter type: {p_type}")

        # -----------------------------------------------------
        # Path A: decoder exposes label_fn → use eeg_train
        # Path B: no label_fn (e.g. ViT script) → spawn subprocess
        # -----------------------------------------------------

        if hasattr(task_module, 'label_fn'):
            # Handle special param conversion
            if 'betas' in params and not isinstance(params['betas'], (list, tuple)):
                beta2 = float(params.pop('betas'))
                params['betas'] = [0.9, beta2]

            task_defaults = getattr(task_module, 'TASK_DEFAULTS', None)
            cfg = et.resolve_cfg(base_yaml, task_defaults=task_defaults)
            cfg.update(params)
            cfg["optuna_trial_id"] = trial.number
            cfg["optuna_params"] = params.copy()
            try:
                mean_acc = et.run_loso(cfg, task_module.label_fn, trial=trial)
                return mean_acc
            except Exception as e:
                print(f"Trial {trial.number} failed with exception: {e}")
                return float('nan')

        # ---------- subprocess fallback ----------
        import subprocess, json, glob, os, time
        start_time = time.time()
        # build --set string list
        if 'betas' in params and not isinstance(params['betas'], (list, tuple)):
            beta2 = float(params.pop('betas'))
            params['betas'] = [0.9, beta2]
        set_list = [f"{k}={v}" for k, v in params.items()]
        # build command
        cmd = [sys.executable, str(decoder_path), '--cfg', str(base_yaml)]
        if set_list:
            cmd += ['--set'] + set_list
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"Trial {trial.number} subprocess failed (exit {proc.returncode})")
            print(proc.stdout)
            print(proc.stderr)
            return float('nan')

        # Find newest summary_*.json written after start_time
        summary_files = sorted(Path('results').glob('runs/*/summary_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        for fp in summary_files:
            if fp.stat().st_mtime >= start_time:
                try:
                    data = json.loads(fp.read_text())
                    return float(data.get('mean_acc', 'nan'))
                except Exception as e:
                    print(f"Trial {trial.number}: failed reading summary {fp}: {e}")
                    break
        print(f"Trial {trial.number}: no summary found")
        return float('nan')

    return objective

# --- Main execution ---
if __name__ == '__main__':
    # Resolve paths for the specific task
    # This needs to be done dynamically based on what the user passes to --decoder and --base
    # For now, let's assume the user specifies a 'task' argument or we infer it
    # We need a way to map from a task name to its decoder script and base YAML.
    # For this example, let's hardcode one task for simplicity to test the new flow.
    # A more robust solution would involve a lookup table or argparse arguments.

    # --- TEMPORARY: Hardcode one task for testing Phase 2 integration ---
    # In a real scenario, these would come from CLI args or a more complex task selection mechanism.
    # For now, we'll pick the direction_binary as our test case.
    
    # This part replaces --decoder and --base flags
    decoder_script_path = args.decoder # Use args.decoder
    base_yaml_path = args.base # Use args.base

    # Dynamic import for label_fn and BASE_YAML path
    sys.path.append(str(decoder_script_path.parent))
    task_module_name = decoder_script_path.stem
    task_module = importlib.import_module(task_module_name)

    # The objective factory now takes the task module name and base YAML path
    objective = objective_factory(task_module_name, decoder_script_path, base_yaml_path, fixed_overrides)

    # Optuna study creation and optimization (remains largely same)
    study = optuna.create_study(direction="maximize",
                                study_name=str(args.study),
                                sampler=optuna.samplers.TPESampler(seed=args.sampler_seed) if args.sampler_seed else None,
                                storage=args.study,
                                load_if_exists=True)

    print(f"\nStarting Optuna study '{study.study_name}' – adding {args.trials} trials…")
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    print("\nStudy finished. Best trial:")
    print(f"  Value: {study.best_value:.2f}%")
    print(f"  Params: {study.best_params}")

    # Plotting
    study_dir = Path("results/optuna_plots") / study.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.io as pio
        fig_history = optuna.visualization.plot_optimization_history(study)
        plotly.offline.plot(fig_history, filename=str(study_dir / "optimization_history.html"), auto_open=False)
        pio.write_image(fig_history, str(study_dir / "optimization_history.png"), format="png")
        print(f"Optimization history plots saved to {study_dir}")

        fig_param_importances = optuna.visualization.plot_param_importances(study)
        plotly.offline.plot(fig_param_importances, filename=str(study_dir / "param_importances.html"), auto_open=False)
        pio.write_image(fig_param_importances, str(study_dir / "param_importances.png"), format="png")
        print(f"Parameter importances plot saved to {study_dir}")

        fig_slice = optuna.visualization.plot_slice(study)
        plotly.offline.plot(fig_slice, filename=str(study_dir / "slice.html"), auto_open=False)
        pio.write_image(fig_slice, str(study_dir / "slice.png"), format="png")
        print(f"Slice plot saved to {study_dir}")

    except Exception as e:
        print(f"Error generating plots: {e}") 