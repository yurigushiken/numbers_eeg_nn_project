#!/usr/bin/env python
"""Unified training entry-point.

Example:
    python train.py --task landing_digit --engine cnn \
                    --cfg configs/landing_digit/base.yaml \
                    --set epochs=50 lr=5e-4
"""

from __future__ import annotations

import argparse
import datetime
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Ensure 'code/' directory is importable
proj_root = Path(__file__).resolve().parent
code_dir = proj_root / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

# Ensure immediate console feedback
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import tasks as task_registry
import engines as engine_registry
from utils.summary import write_summary
from utils.seeding import seed_everything


def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train an EEG decoder on a given task/engine pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", required=True, help="Task name (see tasks registry)")
    p.add_argument("--engine", required=True, choices=list(engine_registry.ENGINES.keys()))
    p.add_argument("--cfg", help="Base YAML config file (defaults to configs/<task>/base.yaml)")
    p.add_argument("--set", nargs="*", metavar="KEY=VAL", help="Override any hyper-parameter.")
    p.add_argument(
        '--run-xai',
        action='store_true',
        help='If set, run XAI analysis automatically after training completes.'
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Builds the configuration dictionary from YAML and CLI overrides."""
    # --- 1. Load common config first ---
    common_yaml = Path("configs") / "common.yaml"
    if common_yaml.exists():
        cfg = yaml.safe_load(common_yaml.read_text()) or {}
    else:
        cfg = {}

    # --- 2. Load task-specific config and merge ---
    if args.cfg:
        base_yaml = Path(args.cfg)
    else:
        base_yaml = Path("configs") / args.task / "base.yaml"

    if not base_yaml.exists():
        print(f"Warning: Task-specific YAML not found at {base_yaml}. Using common config only.")
        task_cfg = {}
    else:
        task_cfg = yaml.safe_load(base_yaml.read_text()) or {}
    
    cfg.update(task_cfg) # Task config overrides common config

    # --- 3. Apply --set overrides (highest priority) ---
    if args.set:
        for kv in args.set:
            if "=" not in kv:
                sys.exit(f"--set expects KEY=VAL, not {kv}")
            k, v = kv.split("=", 1)
            try:
                # Infer type from value
                cfg[k] = yaml.safe_load(v)
            except yaml.YAMLError:
                cfg[k] = v # Treat as string if parsing fails
    
    return cfg


def main():
    """Main execution function."""
    args = parse_args()
    
    # --- Configuration ---
    cfg = build_config(args)
    cfg["task"] = args.task
    
    # --- SEED EVERYTHING for Reproducibility ---
    # Gets seed from config. If 'seed' is not in the config or is null,
    # seed_everything handles it gracefully.
    seed = cfg.get("seed")
    seed_everything(seed)
    
    # --- Registries ---
    label_fn = task_registry.get(args.task)
    engine_run = engine_registry.get(args.engine)
    
    # --- Run Directory Setup ---
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    ds_tag = Path(cfg.get("dataset_dir", "unknown")).name
    # Optional crop_ms suffix in run directory name, e.g., (crop_ms 50-200)
    crop = cfg.get("crop_ms")
    crop_suffix = ""
    if isinstance(crop, (list, tuple)) and len(crop) == 2:
        try:
            crop_suffix = f" (crop_ms {int(crop[0])}-{int(crop[1])})"
        except Exception:
            crop_suffix = ""
    run_dir = Path("results") / "runs" / f"{run_id}_{args.task}_{args.engine}_{ds_tag}{crop_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["run_dir"] = str(run_dir)
    
    # --- Launch Engine ---
    print(f"--- Starting run: {run_dir.name} ---")
    summary_raw = engine_run(cfg, label_fn)
    
    # --- Summarize and Save ---
    summary = {
        "run_id": run_id,
        "dataset_dir": cfg.get("dataset_dir"),
        **summary_raw,
        "study": cfg.get("study"),
        "trial_id": cfg.get("trial_id"),
        "hyper": {k: v for k, v in cfg.items() if k not in {"dataset_dir", "run_dir"}},
    }
    
    write_summary(run_dir, summary, args.task, args.engine)
    
    print(f"--- Run finished. Mean accuracy: {summary.get('mean_acc', 0.0):.2f}% ---")

    if args.run_xai:
        print("\n--- Training complete. Starting XAI analysis... ---")
        
        # Prerequisite check: XAI requires saved model checkpoints.
        if not cfg.get("save_ckpt", False):
            print("WARNING: Cannot run XAI. 'save_ckpt' was not set to true in the config.")
            print("--- XAI analysis skipped. ---")
        else:
            import subprocess
            
            # Use sys.executable to ensure we use the same Python environment
            command = [
                sys.executable, 
                "scripts/run_xai_analysis.py",
                "--run-dir",
                str(run_dir)
            ]
            
            try:
                subprocess.run(command, check=True)
                print("--- XAI analysis complete. ---")
            except subprocess.CalledProcessError as e:
                print(f"--- XAI analysis failed with error: {e} ---")


if __name__ == "__main__":
    main()
