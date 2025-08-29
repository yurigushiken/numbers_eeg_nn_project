"""Unified CSV index writer.

Creates/updates two separate global indices:

1. results/optuna/runs_index_optuna.csv      – one line per Optuna trial
2. results/runs/runs_index_non-optuna.csv    – one line per stand-alone training run

The caller passes the raw summary dict (see utils.summary.write_summary).
The function decides which index to touch based on the presence of the
"trial_id" key (Optuna) and writes the row, auto-expanding the header when
new hyper-parameter columns appear.
"""
from __future__ import annotations

from pathlib import Path
import csv
from typing import Dict, Any


def _ensure_parent(fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)


def append_index(summary: Dict[str, Any], script_tag: str) -> None:
    """Append `summary` to the appropriate global CSV index.

    Parameters
    ----------
    summary : dict
        Parsed summary produced by train.py or optuna_tune.
    script_tag : str
        Something like "landing_digit_cnn"; recorded verbatim in the CSV.
    """
    # classify: Optuna summaries always set a *numeric* trial_id; standalone
    # train.py runs either omit the key or leave it as None.  We consider a
    # run Optuna-generated only when trial_id is not None.
    is_optuna = summary.get("trial_id") is not None

    if is_optuna:
        csv_path = Path("results") / "optuna" / "runs_index_optuna.csv"
    else:
        csv_path = Path("results") / "runs" / "runs_index_non-optuna.csv"

    _ensure_parent(csv_path)

    # -------------------------------- Row build ---------------------------------
    row: Dict[str, Any] = {
        "run_id": summary["run_id"],
        "script": script_tag,
        "mean_acc": f"{summary.get('mean_acc', float('nan')):.4f}",
        "std_acc": f"{summary.get('std_acc', float('nan')):.4f}",
    }

    # Preserve Optuna-specific identifiers
    if is_optuna:
        row["study"] = summary.get("study", "")
        row["trial_id"] = summary.get("trial_id", "")

    # Flatten hyper-parameters
    for k, v in summary.get("hyper", {}).items():
        # skip very large helper fields if any
        if k == "plot_hyper_lines":
            continue
        row[k] = v

    # ---------------------- Load existing header/rows ---------------------------
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        header = list(row.keys())
        rows = []
    else:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
            header = rdr.fieldnames or []

    # auto-expand header
    for c in row:
        if c not in header:
            header.append(c)
            for r in rows:
                r[c] = ""

    # ensure every row has all columns
    for c in header:
        row.setdefault(c, "")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
        w.writerow(row) 