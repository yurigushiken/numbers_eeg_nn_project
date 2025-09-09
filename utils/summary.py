"""Utility to write summary JSON, TXT report, and update runs_index.csv."""

from __future__ import annotations

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import sys
from collections import defaultdict

# switched to new unified index writer
from utils.index_writer import append_index
from .reporting import create_consolidated_reports

__all__ = ["write_summary"]

def write_summary(run_dir: Path | str, summary: Dict, task: str, engine: str) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- JSON ----------------
    json_path = run_dir / f"summary_{task}_{engine}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # --- Part 2: Human-readable TXT report (Revised for clarity) ---
    txt_report_path = run_dir / f"report_{task}_{engine}.txt"
    report_parts = []
    
    report_parts.append(f"--- Run Summary ---")
    report_parts.append(f"Task: {task}   |   Engine: {engine}   |   Run ID: {summary.get('run_id', 'N/A')}")
    report_parts.append("-" * 60)

    # --- Section 1: Overall Performance ---
    report_parts.append("\n--- Overall Performance ---")
    report_parts.append(f"  Mean Accuracy     : {summary.get('mean_acc', 0.0):.2f}% (std: {summary.get('std_acc', 0.0):.2f}%)  (The overall average correctness across all subjects)")
    macro_f1 = summary.get('macro_f1', 0.0)
    if macro_f1: # Only show if calculated
        report_parts.append(f"  Macro F1-Score    : {macro_f1:.2f}%  (A class-balanced score, treating each digit's performance equally)")
        report_parts.append(f"  Weighted F1-Score : {summary.get('weighted_f1', 0.0):.2f}%  (Score weighted by the frequency of each digit in the test set)")

    fold_accs = summary.get('fold_accuracies', [])
    if fold_accs:
        report_parts.append(f"  Fold Accuracy Range : [{min(fold_accs):.2f}% - {max(fold_accs):.2f}%]  (The performance on the best- vs. worst-performing subjects)")

    # --- Section 2: Class Performance ---
    class_perf = summary.get("per_class_f1_mean", [])
    if class_perf:
        report_parts.append("\n--- Cross-Fold Class Performance (Mean F1-Score) ---")
        sorted_classes = sorted(class_perf, key=lambda x: x["f1"], reverse=True)
        
        # Display all classes, as top/worst can be repetitive for few classes
        for item in sorted_classes:
             report_parts.append(f"  - Class {item['class_name']:<10}: {item['f1']:.2f}")

    # --- Section 3: Key Hyper-parameters ---
    report_parts.append("\n--- Key Hyper-parameters ---")
    hyper = summary.get("hyper", {})
    key_params = [
        "model_name", "lr", "batch_size", "epochs", "early_stop", 
        "weight_decay", "scheduler"
    ]
    for key in key_params:
        if key in hyper:
            report_parts.append(f"  {key:<20}: {hyper[key]}")

    # NEW: Channel selection details for transparency/reproducibility
    include_list = hyper.get("include_channels")
    if isinstance(include_list, (list, tuple)) and len(include_list) > 0:
        report_parts.append(f"  include_channels    : {', '.join(map(str, include_list))}")
    use_list = hyper.get("use_channel_list")
    if isinstance(use_list, str) and use_list:
        report_parts.append(f"  use_channel_list    : {use_list}")

    # --- Footer ---
    report_parts.append(f"\n" + "-" * 60)
    report_parts.append(f"Note: Full machine-readable details in summary_{task}_{engine}.json")
    
    txt_report_path.write_text("\n".join(report_parts))

    # --- Part 3: Append to global index ---
    csv_path = Path("results") / "runs_index.csv"
    try:
        append_index(summary, f"{task}_{engine}")
    except Exception as e:
        print(f"[WARN] Could not append runs index: {e}")

    # -------------------------------------------------------------------
    # If this run belongs to an Optuna study (results/optuna/…), trigger a
    # rebuild of the dedicated index CSV that lives in that folder.
    # -------------------------------------------------------------------
    try:
        if summary.get("study"):
            run_path = run_dir if isinstance(run_dir, Path) else Path(run_dir)
            if "optuna" in run_path.parts:
                # locate results/optuna directory
                idx = run_path.parts.index("optuna")
                optuna_root = Path(*run_path.parts[: idx + 1])
                if str(optuna_root) not in sys.path:
                    sys.path.insert(0, str(optuna_root))
                try:
                    import importlib
                    mod = importlib.import_module("optuna_index_builder")
                    if hasattr(mod, "rebuild_index"):
                        mod.rebuild_index()
                except Exception as e:
                    print(f"[WARN] Could not rebuild Optuna index: {e}")
    except Exception as _e:
        print(f"[WARN] Optuna index hook error: {_e}") 

    # --- Part 4: Channel Gate Aggregates & Plots (robust alignment) ---
    try:
        gate_files = sorted(run_dir.glob("fold*_gate_values.json"))
        if gate_files:
            import numpy as np, csv
            # Load all folds
            folds = [json.loads(fp.read_text()) for fp in gate_files]
            # Union of all channels to handle any mismatch
            channel_sets = [set(f.get("channels", [])) for f in folds if f.get("channels")]
            all_channels = sorted(set().union(*channel_sets)) if channel_sets else []
            if all_channels:
                name_to_idx = {ch: i for i, ch in enumerate(all_channels)}
                mat = np.full((len(folds), len(all_channels)), np.nan, dtype=float)
                for r, fdat in enumerate(folds):
                    chs = fdat.get("channels", []) or []
                    vals = np.array(fdat.get("gates", []), dtype=float)
                    for ch, v in zip(chs, vals):
                        c = name_to_idx.get(ch)
                        if c is not None:
                            mat[r, c] = v
                gates_mean = np.nanmean(mat, axis=0)
                # Append to TXT report (top-k)
                top_k = int(summary.get('hyper', {}).get('xai_top_k_channels', 10) or 10)
                order = np.argsort(gates_mean)[::-1][:top_k]
                top_lines = [f"{all_channels[i]}:{gates_mean[i]:.3f}" for i in order if np.isfinite(gates_mean[i])]
                with txt_report_path.open("a") as f:
                    f.write("\n\n--- Channel Gate (Aggregated) ---\n")
                    if np.isnan(mat).any():
                        f.write("  Note: aligned across folds using channel union with NaN padding.\n")
                    f.write("  Top channels: " + ", ".join(top_lines) + "\n")
                # Save CSV
                csv_path = run_dir / "gate_values_mean.csv"
                with csv_path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["channel", "gate_mean"])
                    for ch, val in zip(all_channels, gates_mean):
                        w.writerow([ch, float(val) if np.isfinite(val) else "nan"])
    except Exception as e:
        print(f"[WARN] Gate aggregation failed: {e}")

    # --- Temporal gate aggregation & export (robust alignment) ---
    try:
        tg_files = sorted(run_dir.glob("fold*_time_gate_values.json"))
        if tg_files:
            import numpy as np, csv
            folds = [json.loads(fp.read_text()) for fp in tg_files]
            # Normalize times to int(ms) to prevent float mismatch
            fold_times = [list(map(lambda t: int(round(t)), f.get("times_ms", []))) for f in folds]
            all_times = sorted(set().union(*map(set, fold_times))) if fold_times else []
            if all_times:
                t_to_idx = {t: i for i, t in enumerate(all_times)}
                mat = np.full((len(folds), len(all_times)), np.nan, dtype=float)
                for r, (fdat, tlist) in enumerate(zip(folds, fold_times)):
                    vals = np.array(fdat.get("gates", []), dtype=float)
                    for t, v in zip(tlist, vals):
                        c = t_to_idx.get(t)
                        if c is not None:
                            mat[r, c] = v
                g_mean = np.nanmean(mat, axis=0)
                csv_path = run_dir / "time_gate_values_mean.csv"
                with csv_path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["time_ms", "gate_mean"])
                    for t, v in zip(all_times, g_mean):
                        w.writerow([float(t), float(v) if np.isfinite(v) else "nan"])
                with txt_report_path.open("a") as f:
                    f.write("\n--- Temporal Gate (Aggregated) ---\n")
                    if np.isnan(mat).any():
                        f.write("  Note: aligned across folds using time union (int ms) with NaN padding.\n")
                    if len(g_mean) > 0 and np.any(np.isfinite(g_mean)):
                        peak_idx = int(np.nanargmax(g_mean))
                        f.write(f"  Peak gate at ~{float(all_times[peak_idx]):.0f} ms\n")
    except Exception as e:
        print(f"[WARN] Time-gate aggregation failed: {e}")

    # --- Part 5: NEW - Generate consolidated HTML and PDF reports ---
    print("\n--- Generating Consolidated Reports ---")
    try:
        create_consolidated_reports(run_dir, summary, task, engine)
    except Exception as e:
        print(f" !! ERROR generating consolidated reports: {e}") 