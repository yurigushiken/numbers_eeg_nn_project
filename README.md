# EEG Numerical-Cognition Decoding Project

Deep-learning pipelines for decoding numerical cognition from EEG.

The repository now offers:

* 7 production-ready decoder scripts (`code/02_train_decoder_*.py`)
* YAML-driven configuration for every hyper-parameter
* Sequential, resumable hyper-parameter sweep controller
* Unified JSON + TXT run reports
* Central `results/runs_index.csv` catalogue that auto-expands when new hyper-parameters appear

---

## 1 · Project Setup

```powershell
# one-time environment
conda create -n torcheeg-env python=3.11 -y
conda activate torcheeg-env

# core dependencies (CUDA 11.8 wheel shown; adjust as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torcheeg mne scikit-learn matplotlib pandas seaborn pyyaml
```

### One-off data prep (documented here for completeness)

```powershell
python code/01_prepare_for_nn.py  # already executed on the repo’s data dump
```

---

## 2 · Running a Single Training Job

```powershell
python code/02_train_decoder_landing_digit_enhanced-data-augmentation.py \
       --cfg configs/landing_digit/base.yaml
```

Outputs:

```
results/
  runs/
    20250719_1042_02_train_decoder_landing_digit_enhanced-data-augmentation/
        summary_02_train_decoder_landing_digit_enhanced-data-augmentation.json
        report_02_train_decoder_landing_digit_enhanced-data-augmentation.txt
        fold1_confusion_….png
        …
results/runs_index.csv   ← row appended automatically
```

---

## 3 · YAML Configuration Layers

1. **`DEFAULTS` dict** inside each script – canonical list of hyper-parameters.
2. **`base.yaml`** – task-specific defaults.
3. **Sweep YAMLs** (`full_sweep.yaml`, `lr_sweep.yaml`, …) – grid definitions.
4. **CLI flags** – explicit overrides (`--lr 5e-4`, `--epochs 75`, …).
5. **`--set key=val`** – free-form highest-priority overrides.

Example:

```powershell
python … --cfg configs/landing_digit/base.yaml \
         --lr 5e-4 --set noise_std=0.05 shift_p=0.7
```

Creating a new task:

```
configs/
  new_task/
    base.yaml          # full hyper-param list for that script
    full_sweep.yaml    # optional grid
```

---

## 4 · Hyper-parameter Sweeps

`scripts/run_sweep.py` orchestrates sequential grid searches (single-GPU friendly).
It is **resumable**: before each run it checks `results/runs_index.csv` and skips combos already logged.

```powershell
# inside torcheeg-env
python scripts/run_sweep.py \
  --script code/02_train_decoder_landing_digit_enhanced-data-augmentation.py \
  --base   configs/landing_digit/base.yaml \
  --sweep  configs/landing_digit/full_sweep.yaml
```

If you prefer the controller to activate the environment for every subprocess use `--env torcheeg-env` (PowerShell users: note `conda activate` cannot be chained with `python` on the same line, so activate the env once or run under Bash).

Interrupt at any time – re-run the same command and it continues where it left off.

---

## 5 · Result Catalogue (`runs_index.csv`)

* Auto-appended by every decoder via `code/run_index.py`.
* Header expands on-the-fly when new hyper-parameters appear.
* Locked by Excel? Close the file and rerun, or regenerate:

```powershell
python scripts/rebuild_runs_index.py  # scans every summary_*.json and rewrites index
```

Quick analysis example:

```python
import pandas as pd, seaborn as sns

df = pd.read_csv('results/runs_index.csv')
best = df.sort_values('mean_acc', ascending=False).head(5)
print(best[['run_id','mean_acc','shift_p','noise_std','scale_p']])
```

---

## 6 · Decoder Script Template

All `02_train_decoder_*.py` share the following skeleton:

1. **Imports + `DEFAULTS`**
2. **Helper functions** `load_cfg()` & `parse_args()`
3. **Config resolution** (DEFAULTS → base YAML → CLI flags → `--set`) → `cfg`
4. **Typed reassignment** of module-level constants for downstream code.
5. **Training loop** (LOSO or k-fold).
6. **Confusion matrices** per fold & overall.
7. **`summary.json`** – single source of truth.
8. **Human-readable `report.txt`** – generated from the JSON.
9. **`append_run_index(...)`** – one-line catalogue update.

To build a new decoder:

* Copy an existing script.
* Adapt the data-loading & label logic.
* Ensure every new hyper-parameter is represented in `DEFAULTS` and, if needed, in a new `configs/<task>/base.yaml`.

---

## 7 · Directory Layout

```
code/                      # decoder & utility scripts
configs/
  landing_digit/
    base.yaml
    full_sweep.yaml
  … other tasks …
results/
  runs/                    # one directory per run
  runs_index.csv           # global catalogue (git-ignored)
scripts/
  run_sweep.py             # sweep controller
  rebuild_runs_index.py    # rebuild catalogue from JSON summaries
```

`results/` is intentionally **git-ignored**; only code & configs are version-controlled.

---

## 8 · Common Issues & Fixes

| Symptom | Cause & Remedy |
|---------|----------------|
| `ModuleNotFoundError: torcheeg` | Forgot to `conda activate torcheeg-env` **or** ran sweep without env. Activate first or use `--env`. |
| `PermissionError` on `runs_index.csv` | File open in Excel – close it, rerun, or regenerate with `rebuild_runs_index.py`. |
| Controller reruns completed combos | Rows were missing; regenerate the index then relaunch. |

---

## 9 · Contributing / Extending

* **Add a new decoder**: create `code/02_train_decoder_<task>.py`, provide matching `configs/<task>/base.yaml`, run a smoke test, then sweep.
* **Parallel sweeps**: `run_sweep.py` is sequential by design; add a `--workers N` flag and use `concurrent.futures` for multi-GPU setups.
* **New metrics**: extend the `summary` dict – TXT reports remain in sync because they are always generated from JSON.

PRs welcome – happy decoding!
