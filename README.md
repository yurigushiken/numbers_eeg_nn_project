# EEG Numerical-Cognition Decoding Project

Deep-learning pipelines for decoding numerical cognition from EEG.

The repository offers two main deep-learning pipelines for decoding numerical cognition from EEG:

1.  **CNN-based Decoders** (Legacy / Time-Series Focus): These models directly process EEG time-series data.
2.  **Vision Transformer (ViT) Decoders** (New / Time-Frequency Focus): These models convert EEG into spectrogram images and apply state-of-the-art image classification techniques.

Both pipelines share the following robust features:

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

#### What the data-prep script produces

Running `01_prepare_for_nn.py` writes **per-subject** MNE Epoch files that live in
`data_preprocessed/`:

| Folder | When it is created | What it contains |
|--------|-------------------|------------------|
| `acc_1_dataset/` | Default run (or `PROCESS_ACC_ONLY=True`) | One `sub-XX_preprocessed-epo.fif` per subject containing **only trials with `Target.ACC == 1`** |
| `all_trials_dataset/` | Set environment variable `PROCESS_ACC_ONLY=False` before running | Same file layout but **all usable trials**, regardless of accuracy |

During every execution existing `.fif` files in the chosen folder are
deleted first, ensuring you always work with a fresh, coherent dataset.

Each `.fif` file is an MNE `Epochs` object with EEG data in
micro-volts and a rich `epochs.metadata` **Pandas DataFrame**. These are the
columns you will find (all strings unless noted):

| Column | Description |
|--------|-------------|
| `SubjectID` *(int)* | Numeric participant identifier |
| `Block` *(int)* | Experimental block (1–5) |
| `Trial` *(int)* | Trial index **within the block** (1–60) |
| `Procedure` | ‘Mainproc’, ‘Practiceproc’, … |
| `Condition` | Two-digit code describing the prime→target transition (e.g. `12`, `43`, `99`) |
| `Target.ACC` *(int)* | 1 = correct, 0 = incorrect |
| `Target.RT` *(float)* | Reaction time in ms |
| `Trial_Continuous` *(int)* | Trial index across the whole session (1–300) |
| `direction` | `"I"` increasing, `"D"` decreasing, `"NC"` no-change |
| `change_group` | One of `iSS`, `dSS`, `iLL`, `dLL`, `iSL`, `dLS`, `NC` *(see dissertation)* |
| `size` | `"SS"` small→small, `"LL"` large→large, `"cross"` small↔large, `"NC"` no-change |

These descriptive columns allow every decoder script in `code/02_train_decoder_*.py`
to filter and relabel trials **without having to touch the raw behavioural CSV files**.

Example quick peek:

```python
import mne, pandas as pd

ep = mne.read_epochs('data_preprocessed/acc_1_dataset/sub-02_preprocessed-epo.fif', preload=False)
print(ep.get_data().shape)           # (n_epochs, n_channels, n_times)
print(ep.metadata[['direction','change_group','size']].value_counts())
```

---

## 2 · Running a Single Training Job


### 2.1 · Running a CNN-based Training Job

CNN-based decoders process preprocessed MNE Epoch files directly from `data_preprocessed/acc_1_dataset`.
Example:

```powershell
python code/cnn_experiments/02_train_decoder_landing_digit_enhanced-data-augmentation.py \
       --cfg configs/cnn/landing_digit/base.yaml
```

### 2.2 · Running a Vision Transformer (ViT) Training Job

ViT decoders require an additional data preprocessing step: converting MNE Epochs into spectrogram images.

**Stage 1: Preprocess EEG to Spectrogram Images (Run Once)**

This script reads the raw `.fif` epochs from `data_preprocessed/acc_1_dataset/` and converts each 128-channel EEG trial into a fixed-size (e.g., 128×128) **time-frequency image**.

Key options (defaults shown):

* `--method cwt|stft`   – Continuous Wavelet Transform (**default**) or classic STFT.
* `--norm  minmax|zscore` – Per-image 0-1 scaling (**minmax**) **or** dataset-wide Z-scoring (writes a `stats.json`).

The output directory holds one tensor per trial plus:

```
metadata.csv   # tabular labels for each trial
stats.json     # μ / σ written when --norm zscore (used automatically by the dataset loader)
```

Example CWT + global Z-score run:

```powershell
# Activate your ViT-specific environment (recommended: `conda activate eeg_vit`)
python code/vit_experiments/01_preprocess_eeg_to_spectrograms.py \
       --input_dir data_preprocessed/acc_1_dataset \
       --output_root data_spectrograms/landing_digit_cwt_128x128 \
       --img_size 128 --method cwt --norm zscore
```

Outputs:

```
data_spectrograms/
  landing_digit_cwt_128x128/     # Pre-computed spectrogram tensors
    trial_000001.npy
    trial_000002.npy
    metadata.csv
    stats.json
```

**Stage 2: Train the ViT Model**

This script loads the pre-computed spectrograms from `data_spectrograms/` and trains a Vision Transformer model using Leave-One-Subject-Out (LOSO) cross-validation. By default it uses the **medium** backbone (`vit_relpos_medium_patch16_224`) but any ViT name known to `timm` can be passed via `--set model_name=…`. The script supports YAML/CLI configuration and integrates with Optuna for advanced sweeping.

⚠️  Ensure you have a recent `timm` build (≥ v1.0.17 or current `main` branch) so that `vit_relpos_medium_patch16_224` is recognised:

```powershell
pip install -U "git+https://github.com/rwightman/pytorch-image-models.git"
```

Example:

```powershell
# Ensure `timm` and other ViT dependencies are installed in your environment
python code/vit_experiments/02_train_vit_landing_digit.py \
       --cfg configs/vit/base_vit.yaml
```

Both CNN and ViT training jobs produce the same output structure:

```
results/
  runs/
     20250719_1042_02_train_decoder_landing_digit-OR-02_train_vit_landing_digit/
         summary_*.json
         report_*.txt
         fold1_confusion_….png
        fold1_curves.png
         …
        overall_confusion.png
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

This structure now supports separate configuration directories for CNN and ViT models:

```
configs/
  cnn/                 # For CNN-based decoder configurations
    landing_digit/
      base.yaml
      full_sweep.yaml
  vit/                 # For Vision Transformer configurations
    base_vit.yaml
    optuna_space_vit.yaml # Specific search space for ViT Optuna tuning
```

---

## 4 · Hyper-parameter Sweeps


The project supports two main hyper-parameter tuning strategies:

### 4.1 · Sequential Grid Sweeps (`scripts/run_sweep.py`)

Orchestrates sequential grid searches (single-GPU friendly) and is **resumable**. It checks `results/runs_index.csv` and skips already logged combinations.

Example for CNN-based decoder:

```powershell
# inside torcheeg-env
python scripts/run_sweep.py \
  --script code/cnn_experiments/02_train_decoder_landing_digit_enhanced-data-augmentation.py \
  --base   configs/cnn/landing_digit/base.yaml \
  --sweep  configs/cnn/landing_digit/full_sweep.yaml
```

### 4.2 · Optuna Hyper-parameter Tuning (`scripts/optuna_tune.py`)

This script uses [Optuna](https://optuna.org/) to perform efficient hyper-parameter optimization (e.g., Tree-structured Parzen Estimator). It is **fully resumable** thanks to SQLite storage, allowing for flexible and adaptive search strategies.

Example for ViT-based decoder:

```powershell
# Activate your ViT-specific environment
conda activate eeg_vit
# Run Optuna study
python scripts/optuna_tune.py \
  --script code/vit_experiments/02_train_vit_landing_digit.py \
  --base   configs/vit/base_vit.yaml \
  --space  configs/vit/optuna_space_vit.yaml \
  --db     optuna_studies/vit_optuna.db \
  --trials 50 # Number of trials to run
```

Outputs:

*   Interactive Plotly HTML plots are written to `results/optuna_plots/`.
*   Each trial also appends a row to `results/runs_index.csv`.

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


The project now uses two main decoder script templates:

### 6.1 · CNN Decoder Skeleton (`code/cnn_experiments/02_train_decoder_*.py`)

These scripts process MNE Epoch objects. They follow this skeleton:

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


### 6.2 · Vision Transformer (ViT) Decoder Skeleton (`code/vit_experiments/02_train_vit_*.py`)

These scripts process pre-generated spectrogram images. They share this expanded skeleton:

1.  **Imports + `DEFAULTS`**: Defines canonical hyper-parameters for ViT models and spectrogram processing.
2.  **Configuration Resolution**: Merges defaults with YAML (`configs/vit/base_vit.yaml`) and CLI overrides.
3.  **Spectrogram Dataset (`vit_experiments/vit_dataset.py`)**: Loads pre-computed `.npy` spectrogram files and their `metadata.csv`.
4.  **Data Augmentation (`vit_experiments/vit_dataset.py#SpecAugment`)**: Applies on-the-fly transformations like time and frequency masking.
5.  **ViT Model Definition (`vit_experiments/models/eeg_vit.py`)**: Utilizes `timm` Vision-Transformer backbones, adapted for 128-channel spectrogram “images” and **6 output classes** (landing digits 1-6). An optional lightweight domain-adapter MLP improves cross-subject generalisation.
6.  **Training Loop**: Implements Leave-One-Subject-Out (LOSO) cross-validation with AdamW optimizer, CosineAnnealingLR scheduler, and **always class-balanced Cross-Entropy Loss** – a full 6-length weight vector is built each fold, so the loss remains balanced even if a class happens to be absent in that fold.
7.  **Comprehensive Reporting**: Generates detailed `summary.json` and `report.txt`, plus 300-dpi confusion-matrix heat-maps styled like the CNN pipeline (row-wise percentages, black diagonal, red per-row worst error) and training-curve plots.
8.  **Result Indexing**: Appends run details to `results/runs_index.csv`.

To build a new ViT decoder task, copy `02_train_vit_landing_digit.py` and adapt the `SpectrogramDataset` loading and label interpretation logic.

---

## 7 · Directory Layout

```
code/                      # decoder & utility scripts
configs/
  cnn/                     # Configurations for CNN-based decoders
    landing_digit/
      base.yaml
      full_sweep.yaml
  vit/                     # Configurations for Vision Transformer decoders
    base_vit.yaml
    optuna_space_vit.yaml  # Optuna search space for ViT
results/
  runs/                    # one directory per run
  runs_index.csv           # global catalogue (git-ignored)
scripts/
  run_sweep.py             # sweep controller
  rebuild_runs_index.py    # rebuild catalogue from JSON summaries
 data_spectrograms/         # NEW: Pre-computed spectrogram images for ViT inputs
   landing_digit_cwt_128x128/     # Example dataset: CWT spectrograms, 128×128 resolution
     trial_000001.npy
     metadata.csv
 optuna_studies/            # NEW: Optuna database files for different task types
   vit_optuna.db            # Example: Optuna study for ViT models
```

`results/` is intentionally **git-ignored**; only code & configs are version-controlled.

---

## 8 · Common Issues & Fixes

| Symptom | Cause & Remedy |
|---------|----------------|
| `ModuleNotFoundError: torcheeg` | Forgot to `conda activate torcheeg-env` **or** ran sweep without env. Activate first or use `--env`. |
| `PermissionError` on `runs_index.csv` | File open in Excel – close it, rerun, or regenerate with `rebuild_runs_index.py`. |
| Controller reruns completed combos | Rows were missing; regenerate the index then relaunch. |

### 8.1 · ViT-Specific Issues

| Symptom | Cause & Remedy |
|---------|----------------|
| `RuntimeError: Unknown model (vit_tiny_patch16_128)` | Using an invalid `model_name` for `timm.create_model`. Use canonical names like `vit_tiny_patch16_224` and specify `img_size` separately. |
| `TypeError: can't convert cuda:0 device type tensor to numpy` | Attempting `.numpy()` on a GPU tensor. Use `.cpu().numpy()` to move to host memory first. |
| `RuntimeError: Expected target size [N, C], got [N]` | **Hard labels (1-D integer indices) passed to a loss function expecting soft labels (N x C float matrix).** Ensure `yb` is `torch.long` and `num_classes` is consistent (10 for landing digit). If using MixUp, ensure `CrossEntropyLoss` is configured for soft labels and targets are one-hot encoded and mixed. |

---

## 9 · Contributing / Extending

* **Add a new decoder**: create `code/02_train_decoder_<task>.py`, provide matching `configs/<task>/base.yaml`, run a smoke test, then sweep.
* **Parallel sweeps**: `run_sweep.py` is sequential by design; add a `--workers N` flag and use `concurrent.futures` for multi-GPU setups.
* **New metrics**: extend the `summary` dict – TXT reports remain in sync because they are always generated from JSON.


conda activate torcheeg-env