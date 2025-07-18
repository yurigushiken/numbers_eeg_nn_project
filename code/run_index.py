from pathlib import Path
import csv

# Resolve project root as parent of this file (which lives in code/)
_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = _ROOT / "results" / "runs_index.csv"


def _ensure_parent():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_run_index(summary: dict, script_name: str):
    """Append summary and auto-expand header if new hyper-params appear."""
    _ensure_parent()

    row = {
        "run_id": summary["run_id"],
        "script": script_name,
        "mean_acc": f"{summary['mean_acc']:.4f}",
        "std_acc": f"{summary['std_acc']:.4f}",
    }
    for k, v in summary["hyper"].items():
        row[k] = v

    # new file – write header directly
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        with CSV_PATH.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader(); w.writerow(row); return

    # load existing
    with CSV_PATH.open(newline="") as f:
        rdr = csv.DictReader(f); rows = list(rdr); header = rdr.fieldnames or []

    new_cols = [c for c in row if c not in header]
    if new_cols:
        header += new_cols
        for r in rows:
            for c in new_cols:
                r[c] = ""

    for c in header:
        row.setdefault(c, "")

    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader(); w.writerows(rows); w.writerow(row) 