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
    report_parts.append(f"  Mean Accuracy     : {summary.get('mean_acc', 0.0):.2f}% (std: {summary.get('std_acc', 0.0):.2f}%)")
    macro_f1 = summary.get('macro_f1', 0.0)
    if macro_f1: # Only show if calculated
        report_parts.append(f"  Macro F1-Score    : {macro_f1:.2f}%")
        report_parts.append(f"  Weighted F1-Score : {summary.get('weighted_f1', 0.0):.2f}%")

    fold_accs = summary.get('fold_accuracies', [])
    if fold_accs:
        report_parts.append(f"  Fold Accuracy Range : [{min(fold_accs):.2f}% - {max(fold_accs):.2f}%]")

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

    # --- Part 4: NEW - Generate consolidated HTML and PDF reports ---
    print("\n--- Generating Consolidated Reports ---")
    try:
        create_consolidated_reports(run_dir, summary, task, engine)
    except Exception as e:
        print(f" !! ERROR generating consolidated reports: {e}") 