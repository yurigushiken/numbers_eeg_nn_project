"""Utility to write summary JSON, TXT report, and update runs_index.csv."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import sys

# switched to new unified index writer
from utils.index_writer import append_index

__all__ = ["write_summary"]

def write_summary(run_dir: Path | str, summary: Dict, task: str, engine: str) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- JSON ----------------
    json_path = run_dir / f"summary_{task}_{engine}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # ---------------- TXT (human-readable) ----------------
    lines = [
        f"Task: {task}   ·   Engine: {engine}",
        f"Run ID: {summary['run_id']}",
        "",
        "Performance:",
        f"  Mean acc: {summary['mean_acc']:.2f}%",
    ]
    if "std_acc" in summary:
        lines.append(f"  Std acc : {summary['std_acc']:.2f}%")

    lines.extend(["", "Hyper-parameters (non-path):"])
    for k, v in sorted(summary.get("hyper", {}).items()):
        if k in {"dataset_dir", "run_dir"}:  # paths already implicit
            continue
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("Artifacts:")
    for p in sorted(run_dir.glob("*.png")):
        lines.append(f"  {p.name}")

    lines.append("")
    lines.append(f"Note: full machine-readable details in {json_path.name}")

    (run_dir / f"report_{task}_{engine}.txt").write_text("\n".join(lines))

    # ---------------- global CSV index (separate optuna / non-optuna) ----------------
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