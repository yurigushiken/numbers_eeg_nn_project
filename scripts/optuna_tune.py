#!/usr/bin/env python
"""Unified Optuna tuner for any task / engine pair.

Example:
    python scripts/optuna_tune.py \
        --task landing_digit \
        --engine vit \
        --base  configs/landing_digit/base.yaml \
        --space configs/landing_digit/optuna_space.yaml \
        --db    sqlite:///optuna_studies/landing_digit_vit.db \
        --trials 50
"""

from __future__ import annotations

import argparse, yaml, json, sys, time, datetime, os
from pathlib import Path
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Ensure project root and code/ are on sys.path so that `import tasks` and
# `import engines` resolve correctly when this script is executed via an
# absolute path (Python adds only the *script's* directory by default).
# -----------------------------------------------------------------------------
proj_root = Path(__file__).resolve().parent.parent
code_dir = proj_root / "code"
for p in (str(proj_root), str(code_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import optuna

# ------------------------------------------------------------------
# Ensure that any pre-imported 'utils' from external packages does not
# shadow the project-local `utils` package (which contains plots.py etc.).
# Some third-party libraries imported by Optuna may have already imported
# an unrelated package named "utils".  If that happens, our subsequent
# `import utils.plots` will fail.  We proactively remove such a module
# unless it comes from this project root.
# ------------------------------------------------------------------
if 'utils' in sys.modules:
    _u_mod = sys.modules['utils']
    u_path = getattr(_u_mod, '__file__', '') or ''
    if proj_root.as_posix() not in u_path.replace('\\', '/'):
        # Remove the foreign module so Python can import our local one
        del sys.modules['utils']

import tasks as task_reg
import engines as engine_reg
from utils.summary import write_summary


# -----------------------------------------------------------------------------
# Global setup (argument parsing, etc.) needs to be at the top level
# so the `objective` function can access `args` and other globals.
# -----------------------------------------------------------------------------

p = argparse.ArgumentParser(description="Optuna tuner (unified)")
p.add_argument("--task", required=True, help="Task name registered in tasks/")
p.add_argument("--engine", required=True, choices=list(engine_reg.ENGINES))
p.add_argument("--base", type=Path, required=True, help="Base YAML path")
p.add_argument("--space", type=Path, required=True, help="Optuna search-space YAML")
p.add_argument("--db", required=True, help="Optuna SQLite URI, e.g. sqlite:///my.db")
p.add_argument("--trials", type=int, default=20)
args = p.parse_args()

space = yaml.safe_load(args.space.read_text()) or {}
STUDY_TAG = Path(args.db).stem
OPTUNA_RUN_ROOT = Path("results") / "optuna" / STUDY_TAG
OPTUNA_RUN_ROOT.mkdir(parents=True, exist_ok=True)

label_fn = task_reg.get(args.task)
engine_run = engine_reg.get(args.engine)


# -----------------------------------------------------------------------------
# Objective function for Optuna
# -----------------------------------------------------------------------------

def build_cfg(base_yaml_path: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a config dict from a base YAML and Optuna overrides."""
    cfg = yaml.safe_load(base_yaml_path.read_text()) or {}
    cfg.update(overrides)
    return cfg

def objective(trial: optuna.Trial):
    sampled: Dict[str, Any] = {}
    for name, spec in space.items():
        m = spec["method"]
        if m in ("uniform", "float"):
            sampled[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif m == "log_uniform":
            sampled[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif m == "int":
             sampled[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif m == "categorical":
            sampled[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported method {m} for param {name}")

    cfg = build_cfg(args.base, sampled)
    cfg["task"] = args.task
    ds_tag = Path(cfg.get("dataset_dir", "unknown")).name
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["run_dir"] = str(
        OPTUNA_RUN_ROOT / f"{ts}_{args.task}_{args.engine}_{ds_tag}_t{trial.number:03d}"
    )
    cfg["study"] = STUDY_TAG
    cfg["trial_id"] = trial.number

    # Build prettier hyper-parameter lines for plots
    plot_lines = []
    for name, spec in space.items():
        val = sampled.get(name, cfg.get(name))
        if val is None: continue
        if spec["method"] in {"uniform", "log_uniform"}:
            lo, hi = spec["low"], spec["high"]
            try:
                p = (val - lo) / (hi - lo)
                bar_pos = int(p * 10)
            except Exception:
                bar_pos = 0
            bar = "|" + "-" * bar_pos + "●" + "-" * (10 - bar_pos) + "|"
            plot_lines.append(f"{name}: {lo:.1e} {bar} {hi:.1e}\nval: {val:.1e}")
        elif spec["method"] == "categorical":
            choices = spec["choices"]
            choices_str = [str(c) if c != val else f"[{c}]" for c in choices]
            plot_lines.append(f"{name}: " + " | ".join(choices_str))
    cfg["plot_hyper_lines"] = plot_lines

    try:
        res = engine_run(cfg, label_fn)
        summary = {
            "run_id": ts,
            **res,
            "study": cfg.get("study"),
            "trial_id": cfg.get("trial_id"),
            "hyper": {
                k: v for k, v in cfg.items()
                if k not in {"dataset_dir", "run_dir", "plot_hyper_lines"}
            },
        }
        write_summary(cfg["run_dir"], summary, args.task, args.engine)
        return res["mean_acc"]
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return float('nan')

# -----------------------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------------------

def main():
    """Sets up and runs the Optuna study."""
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name=STUDY_TAG, storage=args.db,
                                load_if_exists=True)

    print(f"Starting study {study.study_name} with {args.trials} trials…")
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    print("Best value", study.best_value, "Params", study.best_params)

    # --- save Optuna summary plots ---
    import plotly.io as pio, optuna.visualization as vis
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = Path("results") / "optuna_plots"
    plot_ts_dir = plot_dir / f"{ts}_{args.task}_{args.engine}"
    plot_ts_dir.mkdir(parents=True, exist_ok=True)

    plot_funcs = {
        "history": vis.plot_optimization_history,
        "slice": vis.plot_slice,
        "contour": vis.plot_contour,
        "parallel": vis.plot_parallel_coordinate,
    }

    for name, fn in plot_funcs.items():
        try:
            fig = fn(study)
            pio.write_image(fig, plot_ts_dir / f"{name}.png", scale=2)
        except Exception as e:
            print(f"Plot {name} failed: {e}")


if __name__ == '__main__':
    main()
