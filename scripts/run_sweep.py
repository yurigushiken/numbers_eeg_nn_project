import argparse
import itertools
import subprocess
import sys
import csv
import json
from pathlib import Path
import yaml

"""Generic sequential sweep controller.

Usage example (PowerShell):
  python scripts/run_sweep.py \
      --script code/02_train_decoder_landing_digit_enhanced-data-augmentation.py \
      --base   configs/landing_digit/base.yaml \
      --sweep  configs/landing_digit/full_sweep.yaml \
      --env    torcheeg-env

The controller will read `base.yaml` for defaults, `full_sweep.yaml` for the grid
of hyper-parameters, build every combination, and launch the decoder script
sequentially via subprocess.  It skips combinations that already appear in
results/runs_index.csv (resumable).
"""

CSV_PATH = Path("results") / "runs_index.csv"


def load_yaml(path: str) -> dict:
    fp = Path(path)
    if not fp.exists():
        sys.exit(f"Config file {path} not found.")
    try:
        return yaml.safe_load(fp.read_text()) or {}
    except Exception as e:
        sys.exit(f"Failed to parse YAML {path}: {e}")


def existing_runs() -> list[dict]:
    """Return rows from runs_index.csv or []."""
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(newline="") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def combo_done(rows: list[dict], script_stem: str, combo: dict) -> bool:
    """Return True if a combination appears in the runs_index CSV."""
    for r in rows:
        if r.get("script") != script_stem:
            continue
        match = True
        for k, v in combo.items():
            if k not in r:
                match = False
                break
            # CSV stores everything as strings
            if str(r[k]) != str(v):
                match = False
                break
        if match:
            return True
    return False


def build_combinations(sweep_cfg: dict) -> list[dict]:
    """Cartesian product over list-valued keys; scalar values stay fixed."""
    keys = list(sweep_cfg.keys())
    vals_lists = [v if isinstance(v, list) else [v] for v in sweep_cfg.values()]
    combos = []
    for prod in itertools.product(*vals_lists):
        combos.append(dict(zip(keys, prod)))
    return combos


def main():
    ap = argparse.ArgumentParser(description="Sequential hyper-parameter sweep controller")
    ap.add_argument("--script", required=True, help="Path to decoder script to run")
    ap.add_argument("--base", required=True, help="Path to base YAML config for that script")
    ap.add_argument("--sweep", required=True, help="Path to YAML defining sweep parameters")
    ap.add_argument("--env", help="Conda environment name to activate for each run")
    ap.add_argument("--dry", action="store_true", help="Just print commands without executing")
    args = ap.parse_args()

    script = Path(args.script)
    if not script.exists():
        sys.exit(f"Decoder script {args.script} not found.")

    sweep_cfg = load_yaml(args.sweep)
    if not sweep_cfg:
        sys.exit("Sweep YAML is empty – nothing to do.")

    combos = build_combinations(sweep_cfg)

    rows = existing_runs()
    script_stem = script.stem

    pending = [c for c in combos if not combo_done(rows, script_stem, c)]
    total = len(combos)
    print(f"Total combinations: {total} | Pending: {len(pending)}")

    for idx, combo in enumerate(pending, 1):
        # Build command string
        set_args = " ".join(f"{k}={v}" for k, v in combo.items())
        env_cmd = f"conda activate {args.env}; " if args.env else ""
        cmd = (
            f"{env_cmd}python {script} --cfg {args.base} --set {set_args}"
        )
        print(f"\n[{idx}/{len(pending)}] Running: {cmd}\n")
        if args.dry:
            continue
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed (returncode {e.returncode}). Stopping sweep.")
            sys.exit(e.returncode)

    print("\nSweep complete.")


if __name__ == "__main__":
    main() 