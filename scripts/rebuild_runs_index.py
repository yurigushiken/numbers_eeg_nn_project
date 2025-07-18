import json, csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "runs"
INDEX_PATH = ROOT / "results" / "runs_index.csv"

rows = []
all_hyper_keys = set()

for summ_fp in RUNS_DIR.rglob("summary_*.json"):
    try:
        data = json.loads(summ_fp.read_text())
    except Exception as e:
        print(f"Skip {summ_fp}: {e}")
        continue
    row = {
        "run_id": data.get("run_id"),
        "script": data.get("run_id").split("_", 2)[2] if data.get("run_id") else "",
        "mean_acc": f"{data.get('mean_acc', '')}",
        "std_acc": f"{data.get('std_acc', '')}",
    }
    hyper = data.get("hyper", {})
    for k, v in hyper.items():
        row[k] = v
        all_hyper_keys.add(k)
    rows.append(row)

if not rows:
    sys.exit("No summary_*.json files found under results/runs.")

header = ["run_id","script","mean_acc","std_acc"] + sorted(all_hyper_keys)

INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
with INDEX_PATH.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    for r in rows:
        for k in header:
            r.setdefault(k, "")
        w.writerow({k: r[k] for k in header})

print(f"Rebuilt {INDEX_PATH} with {len(rows)} runs and {len(header)} columns.") 