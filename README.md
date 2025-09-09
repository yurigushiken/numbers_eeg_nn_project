# EEG Numerical-Cognition Decoding Project (Unified 2025-07)

**⚠️ Quick environment setup — READ ME FIRST**

*All commands in this repo assume you have activated the `eegnex-env` conda environment.*

```powershell
conda activate eegnex-env
```

Once activated, you can run all subsequent python commands directly.

For Windows consoles, you may need to run `chcp 65001` once per shell to avoid Unicode errors.

You are now set – **and when running via the Cursor / Chat agent always keep the terminal in the conversation** so logs appear inline for easier debugging.

---

This repository contains a **single, configurable training pipeline** for
classifying cognitive variables from EEG.  You pick a *task* (what to
predict) and an *engine* (how to model the data); everything else—training,
plots, reports, catalogue—is shared.

The project is developed openly on GitHub at
[https://github.com/yurigushiken/numbers_eeg_nn_project](https://github.com/yurigushiken/numbers_eeg_nn_project).  

(First-time users: follow the full installation steps in the *Environment Setup* section at the end of this document.)

Engines currently available

| Engine   | Data source                                   | Notes |
|----------|-----------------------------------------------|-------|
| `cnn`    | Raw time-series Epochs                        | EEGNet family (`model_name`: `eegnet`, `eegnet_se`, `multi_scale_cnn`) |
| `hybrid` | Raw time-series Epochs                        | CwA-Transformer (channel-wise CNN + Transformer) |
| `dual_stream` | Pre-processed time-series **+** spectrogram (`.npy`) | Late-fusion CNN: 1-D Multi-Scale CNN + 2-D spectrogram CNN.
  • **V2 (2025-07-31)** now supports an *architecture switch* (`spec_arch`): choose between the original *custom* 2-D CNN **or** an ImageNet-pre-trained **ResNet-18** backend via a single YAML flag.
  • Data representation is also tunable: four offline-generated spectrogram variants (`128-16`, `128-32`, `64-8`, `256-32`).
| `cnn_spectrogram` | 128 × 128 CWT spectrograms                 | Image-classification CNN baseline |

Note: For the NUMBERS‑COG study, we focus on the `cnn` engine with `model_name: eegnex`.

All three engines are launched through `train.py`.  Internally they all call the
shared `code/training_runner.py`, so every experiment (raw EEG or spectrogram)
follows the exact same LOSO, early-stopping and reporting logic.  Each run
produces the following artefacts:

```
results/runs/<timestamp>_<task>_<engine>/
    summary_*.json      # single source of truth (includes per-fold & per-class metrics)
    report_*.txt        # human-readable
    fold1_confusion.png fold1_curves.png …
    overall_confusion.png
    fold*_gate_values.json  # if Channel Gate is enabled: per-fold channel gates/L1/sparsity
    gate_values_mean.csv    # aggregated channel gates across folds
    fold*_time_gate_values.json  # if Temporal Gate is enabled: per-fold time gates/L1/TV
    time_gate_values_mean.csv    # aggregated temporal gates across folds
results/runs_index.csv  # global catalogue auto-expanded
```

The consolidated reports now include:
- Macro F1-Score: class-balanced performance across classes
- Weighted F1-Score: weighted by class frequencies
- Conditions subtitle: when the task exposes a `CONDITIONS` list (e.g., `[44, 55, 66]`)
- Included channels banner: when `include_channels` is set in the YAML
- Channel/Temporal Gate aggregates (when gates are enabled), with robust cross-fold alignment

---

## 1 · Quick Start (Smoke Tests)

```powershell
# CNN – two epochs, one fold
python train.py --task landing_digit --engine cnn --set epochs=2 max_folds=1

# Hybrid – two epochs, one fold
python train.py --task landing_digit --engine hybrid --set epochs=2 max_folds=1

# EEGNet with SE blocks – two epochs, one fold

# Dual-Stream CNN – two epochs, one fold
python train.py --task landing_digit --engine dual_stream --set epochs=2 max_folds=1
python train.py --task landing_digit --engine cnn `
                --cfg configs/landing_digit/eegnet_se_base.yaml `
                --set epochs=2 max_folds=1
```

---

### **1.1 · Understanding the Data & Experimental Design**

To effectively create new tasks and interpret results, it's essential to understand the structure of the experimental data.

#### **The "Condition" Code: The Key to Everything**

The most important piece of metadata for any trial is the `Condition` column. This is a two-digit integer that encodes both the **prime** number (what the subject was repeatedly shown) and the **stimulus** number (the final number they saw).

*   **Format:** `PrimeDigit` `StimulusDigit`
*   **Example:** A `Condition` of `32` means the subject was primed with the number `3` and then shown the number `2` as the final stimulus.

#### **`ACC=1` vs. `ALL` Datasets: A Critical Distinction**

You will find two primary types of datasets in the `data_preprocessed/` directory:

*   **`acc_1_dataset`**: This dataset contains only trials where the participant **correctly detected a change** in the number of dots and pressed the space key. Because no key press was required for "no-change" trials, this dataset **contains only non-cardinality trials** (e.g., `12`, `34`, `52`).

*   **`all_trials_dataset`**: This is the complete dataset. It includes all trials, regardless of whether the participant's response was correct, incorrect, or absent. Crucially, this is the **only dataset that contains "cardinality" trials**.

#### **What is "Cardinality"?**

In the context of this study, "cardinality" refers to the **"no-change" condition**, where the prime and stimulus numbers are identical.

*   **Examples:** `11` (prime 1, stimulus 1), `22`, `33`, `44`, `55`, `66`.

Any task designed to analyze these specific trials (like our `cardinality` task) **must** use one of the `all_trials_dataset` variants as its data source.

---

### **IMPORTANT**: Foundational Data Pre-processing (from raw `.set` files)

All `... V2` datasets in the `data_preprocessed/` directory (e.g., `all_trials_dataset (30hz) V2`) were generated by the script:
`project_lib/01_prepare_for_nn(oldversionworkingagainforV2).py`.

This is the legacy, verified, and **now preferred method** for transforming the original raw `.set` EEG files and behavioral `.csv` files into curated `.fif` epochs ready for training. It handles several critical data cleaning and alignment steps that are essential for reproducible results. Subsequent preprocessing steps, like the one for the Dual-Stream engine below, should use these `V2` datasets as their input.

The training pipeline uses Leave-One-Subject-Out (LOSO) cross-validation by default, where each "fold" corresponds to holding out one subject for testing. For faster runs, you can specify a smaller number of folds (e.g., `n_folds: 4`) in the YAML config.

### Pre-processing for the Dual-Stream engine

Before any Dual-Stream training run you must convert the curated `.fif` files
(one-time only). These commands should be run on the `V2` datasets.  **V2 tip:** you can generate *multiple* spectrogram datasets in one go by
varying `--n-fft` and `--hop-length`:

```powershell
# Baseline (balanced) - Example using the 30Hz V2 dataset
python scripts/preprocess_for_dual_stream.py --input-dir "data_preprocessed/acc_1_dataset (30hz) V2" `
       --output-dir data_dual_stream/acc_1_dataset_128-16_30hz --n-fft 128 --hop-length 16
# Higher temporal resolution
python scripts/preprocess_for_dual_stream.py --input-dir "data_preprocessed/acc_1_dataset (30hz) V2" `
       --output-dir data_dual_stream/acc_1_dataset_64-8_30hz  --n-fft 64  --hop-length 8
# Lower temporal, higher freq resolution
python scripts/preprocess_for_dual_stream.py --input-dir "data_preprocessed/acc_1_dataset (30hz) V2" `
       --output-dir data_dual_stream/acc_1_dataset_128-32_30hz --n-fft 128 --hop-length 32
# Finest frequency bins
python scripts/preprocess_for_dual_stream.py --input-dir "data_preprocessed/acc_1_dataset (30hz) V2" `
       --output-dir data_dual_stream/acc_1_dataset_256-32_30hz --n-fft 256 --hop-length 32
```


Repeat for `all_trials_dataset` if required.  The Dual-Stream configs then use paths pointing to these newly generated directories:

```yaml
dataset_dir: data_dual_stream/acc_1_dataset_128-16_30hz # Example path
in_memory:   true            # load all trials into RAM once at startup
```

The other engines (`cnn`, `hybrid`, `cnn_spectrogram`) still read directly from
`data_preprocessed/*` and do **not** require this step.

---

## 2 · Repository Layout (after 2025-07 overhaul)

```
configs/
  landing_digit/
    base.yaml           # defaults for CNN/Hybrid
    eegnet_se_base.yaml # SE-enhanced EEGNet
    multi_scale_cnn_base.yaml # Multi-Scale CNN
    optuna_space_cnn.yaml   # Optuna search space for raw-EEG models
  direction_binary/
    base.yaml
  …                     # one folder per cognitive task

engines/                # cnn.py, hybrid.py, cnn_spectrogram.py
tasks/                  # landing_digit.py, direction_binary.py …
code/
  training_runner.py     # shared LOSO runner used by all engines
  models/cwa_transformer.py
results/                # git-ignored run outputs
scripts/
  optuna_tune.py        # unified tuner (calls train.py in-process)
archive/legacy_configs/ # old cnn/ hybrid/ YAML trees (read-only)
```

---

## 3 · Running a Training Job

### 3.1 Command-line anatomy

```
python train.py --task <task> --engine <cnn|hybrid|cnn_spectrogram> \
                [--cfg configs/<task>/base.yaml]         \
                --set key=value key=value …
```

Order of precedence for hyper-parameters

1. YAML given via `--cfg` (default: `configs/<task>/base.yaml`)
2. Explicit CLI flags (`--engine`, `--task`, etc.)
3. `--set key=val` overrides (highest priority)

### 3.2 Creating a new task

1. Add `tasks/<new_task>.py` with a single `label_fn(meta)`.
2. Create `configs/<new_task>/base.yaml` (plus optional spectrogram-specific YAML if using `cnn_spectrogram`).
3. **Register your new task in `tasks/__init__.py`** by importing it and adding it to the `TASKS` dictionary.
4. Run a smoke test:
   ```powershell
   python train.py --task <new_task> --engine cnn --set epochs=2 max_folds=1
   ```

### 3.3 Leakage-safe training behavior

For every outer fold, we create an inner, subject-aware split from the training subjects:
- Early stopping and the LR scheduler are driven by the inner validation loss only.
- The outer test fold is evaluated once at the end using the best inner‑val checkpoint.
- Class weights are computed from inner‑train labels only.

Optional knob (defaults to 0.2):
```yaml
inner_val_frac: 0.2
```

---

## 4 · Configuration Files

Each task folder can hold multiple presets:

* `base.yaml`       – generic defaults (raw-EEG engines)
* `eegnet_se_base.yaml` – SE-enhanced EEGNet defaults
* `multi_scale_cnn_base.yaml` – Multi-Scale CNN defaults
* `optuna_space.yaml`

Example `configs/landing_digit/multi_scale_cnn_base.yaml` excerpt:

```yaml
model_name: eegnex
dataset_dir: "data_preprocessed/all_trials_dataset (1-45hz) V2"

# --- Data & Preprocessing ---
use_channel_list: non_scalp
include_channels: [E76, E59, E83]
crop_ms: [50, 200]

# --- XAI ---
xai_top_k_channels: 20

# --- Training ---
epochs: 100
early_stop: 20
batch_size: 16
lr: 0.00078
```

### **4.1 · Available Tasks**

| Task Name | Description | # Classes |
| :--- | :--- | :--- |
| `landing_digit` | Predict the final stimulus digit (1–6). | 6 |
| `no_cross_landing_digit` | Predict the final stimulus digit (1–6), but only for trials where prime and stimulus are both small {1,2,3} or both large {4,5,6}. | 6 |
| `cardinality` | Binary: Was it a "no-change" trial (e.g., 11, 22)? | 2 |
| `cardinality_1_3` | 3‑class: No‑change trials 11/22/33 (subset 1–3). | 3 |
| `cardinality_4_6` | 3‑class: No‑change trials 44/55/66 (subset 4–6). | 3 |
| `change_no_change` | Binary: Was there any change between prime and stimulus? | 2 |
| `direction_binary` | Binary: Was the change direction positive (e.g., 24) or negative (e.g., 42)? Ignores no-change trials. | 2 |
| `land1_binary` | Binary: Did the trial land on the digit '1'? | 2 |
| `land1_binary_explicit` | Same as `land1_binary`, but with all conditions explicitly listed in the task file for clarity. | 2 |
| `landing_digit_1_3_from_any` | Predict the final stimulus digit (1-3), allowing any valid prime digit. | 3 |
| `landing_digit_4_6_from_any` | Predict the final stimulus digit (4-6), allowing any valid prime digit. | 3 |
| `landing_digit_1_3_within_small_and_cardinality` | Predict the final stimulus digit (1-3), but only for trials where prime and stimulus are both in the small group {1,2,3}. | 3 |
| `landing_digit_4_6_within_large` | Predict the final stimulus digit (4-6) for non-cardinality trials where prime and stimulus are both in the large group {4,5,6}. | 3 |
| `landing_digit_1_3_within_small` | Predict the final stimulus digit (1-3) for non-cardinality trials where prime and stimulus are both in the small group {1,2,3}. | 3 |
| `landing_digit_4_6_within_large` | Predict the final stimulus digit (4-6) for non-cardinality trials where prime and stimulus are both in the large group {4,5,6}. | 3 |
| `all_pairs_XvY` | Binary: Distinguish between landing digits X and Y. A set of 15 tasks for all unique pairs from 1-6. | 2 |
| `numbers_pairs_12_21` | Binary: Was the condition `12` or `21`? | 2 |
| *... (and all other `numbers_pairs_X_Y` variants)* | *... (Binary classification for other specific pairs)* | 2 |

### **4.2 · Reproducibility & Seeding**

For Optuna studies, we fix randomness to ensure identical data splits and initial weights across trials:
- Add to the base YAML:
```yaml
seed: 42
random_state: 42
inner_val_frac: 0.2
```
- The tuner calls `seed_everything(cfg.seed)` per trial; outer and inner subject-aware splits use `random_state`.

For final reported results, you can either:
- Run full LOSO with the tuned hyper‑parameters (seeded), or
- Report a mean ± std over multiple seeds.

### **4.3 · Data Preprocessing Options**

You can control several data preprocessing steps directly from the `.yaml` configuration files.

#### **Opt-in Channel Exclusion**

For certain analyses, it may be desirable to exclude non-scalp (e.g., ocular) channels from the model training process. The project supports an explicit, opt-in mechanism for this.

The `configs/common.yaml` file contains a named list of channels to exclude, currently defined as `non_scalp`. To activate this for a specific training run, add the following key to that run's `.yaml` file:

```yaml
# In your task's .yaml file:
use_channel_list: non_scalp
```

When the training starts, the system prints a confirmation message indicating exactly which channels have been removed for that run.

#### **Explicit Channel Inclusion (Keep-Only)**

To restrict training to a specific subset of channels (e.g., those identified by XAI), provide `include_channels`. All other channels are discarded, and the order you list is preserved.

```yaml
include_channels:
  - E55
  - E39
  - E40
  - E31
  - E87
  - Cz
  - E115
  - E53
  - E78
  - E100
```

Notes:
- Exclusion via `use_channel_list` is applied first, then inclusion.
- If `include_channels` is empty or omitted, all remaining channels are used.

#### **Configurable Time-Window Cropping**

To focus the model on a specific time window of the EEG epoch (e.g., to isolate an ERP), add the `crop_ms` key to your configuration. This truncates each trial's data before it is passed to the model.

The feature is controlled by a list of two integers representing the start and end times in milliseconds:

```yaml
# In your task's .yaml file, this truncates the data to a 50–200 ms window:
crop_ms: [50, 200]
```

When present, the loader prints a confirmation message, for example:

```
INFO: Applying time cropping from 50ms to 200ms (0.050s to 0.200s)
```

Example (post‑stimulus only; 45 Hz dataset):
```yaml
crop_ms: [0, 596]
```
The loader prints:
```
INFO: Applying time cropping from 0ms to 596ms (0.000s to 0.596s)
```

Notes:

- Cropping uses `include_tmax=True`, so the end time bound is included.
- Times are in milliseconds relative to stimulus onset; negative values indicate the pre-stimulus baseline.
- All plots and XAI outputs use the dataset's canonical time axis (`times_ms`), which automatically reflects `crop_ms` when set.

### **4.4 · Augmentations (train‑only)**

Raw‑EEG augmentations are applied to the training loader only; validation and test are never augmented. The following knobs are available in YAML (typical ranges shown):
```yaml
mixup_alpha: 0.0–0.5
shift_p: 0.0–0.5
shift_max_frac: 0.01–0.08
scale_p: 0.0–0.3
scale_min: 0.9
scale_max: 1.1–1.2
noise_p: 0.0–0.5
noise_std: 0.01–0.05
time_mask_p: 0.2–0.5
time_mask_frac: 0.05–0.3
chan_mask_p: 0.2–0.7
chan_mask_ratio: 0.05
```

### **4.5 · Channel Gate regularization (optional)**

Enable a learnable, non‑negative per‑channel gate vector with L1 regularization for parsimony. This lets the model learn which electrodes matter without manual pre‑selection.

```yaml
channel_gate: true       # enable/disable
gate_init: 1.0           # exact via inverse‑softplus init
gate_l1_lambda: 0.0002   # tune 1e‑5–5e‑4
```

If enabled, each run exports:
- `foldNN_gate_values.json` per fold (channels, gates, L1 sum, sparsity fraction)
- `gate_values_mean.csv` aggregated across folds
The TXT report includes a “Channel Gate (Aggregated)” Top‑K section.

### **4.6 · Temporal Gate regularization (optional)**

Enable a learnable, non‑negative per‑timepoint gate vector applied before the backbone. This lets the model learn which time regions matter.

```yaml
temporal_gate: true        # enable/disable
time_gate_init: 1.0        # exact via inverse‑softplus init
time_gate_l1_lambda: 0.0002  # sparsity; tune ~1e‑5–5e‑4
time_gate_tv_lambda: 0.0001  # smoothness; tune ~1e‑6–1e‑3
```

If enabled, each run exports:
- `foldNN_time_gate_values.json` per fold (times_ms, gates, L1 sum, TV sum)
- `time_gate_values_mean.csv` aggregated across folds
The TXT report includes a “Temporal Gate (Aggregated)” section with a peak‑time note.

---

## 5 · Hyper-parameter Optimisation (Optuna)

Grid-sweep controller was deprecated.  Use Optuna instead:

```powershell
python scripts/optuna_tune.py `
       --task  landing_digit `
       --engine dual_stream `
       --base  configs/landing_digit/dual_stream_base_nf4_acc1.yaml `
       --space configs/landing_digit/optuna_space_dual_stream_v2.yaml `
       --db    "sqlite:///optuna_studies/landing_digit_dual_stream-acc1-nf4-02.db" `
       --trials 36
```

The tuner samples parameters, calls the chosen engine **in-process**, and logs key metrics per trial:
- `inner_mean_macro_f1`: mean of best inner‑validation macro‑F1 across folds (used for model selection)
- `inner_mean_acc`: mean of best inner‑validation accuracies (logged)
- `mean_acc`: outer‑test mean accuracy (held‑out estimate; not used for selection)
Each trial is appended to `results/runs_index.csv`.

Notes:
- The tuner now merges `configs/common.yaml` + the base YAML + trial overrides (same precedence as `train.py`). This allows `use_channel_list: non_scalp` to work during studies.
- You can also include a Channel Gate term in your search space, e.g.:
```yaml
gate_l1_lambda:
  method: log_uniform
  low: 1.0e-5
  high: 5.0e-4
```
- Temporal Gate terms for search spaces, e.g.:
```yaml
time_gate_l1_lambda:
  method: log_uniform
  low: 1.0e-5
  high: 5.0e-4
time_gate_tv_lambda:
  method: log_uniform
  low: 1.0e-4
  high: 1.0e-2
```
- The study optimizes `inner_mean_macro_f1` (leakage‑safe). Use the best trial’s hyper‑parameters to run a final full LOSO evaluation and report outer‑test `mean_acc`.
- Advanced: dataset caching respects `crop_ms` / `use_channel_list` / `include_channels` per trial. To fully isolate caches across trials, set:
```yaml
cache_isolate_trials: true
```

 

## 6 · Common Issues & Fixes

| Symptom | Remedy |
|---------|--------|
| `ModuleNotFoundError: models.cwa_transformer` | Ensure `code/` is on `PYTHONPATH` (train.py inserts it automatically) or run Python directly after `conda activate torcheeg-env`. |
| Confusion matrix only shows first row | Your spectrogram metadata already contains the `landing_digit` column; the task’s `label_fn` now checks for it. |
| `landing_digit` shows a 7th "nan" class | Fixed: the task now preserves NaNs (not the string "nan"); pull latest code. |

---

## 7 · Extending the Project

* **New engine** – create `engines/<name>.py` that picks a **Dataset**, a **model builder**, and an **augmentation builder**, then calls `TrainingRunner`.  (See `engines/cnn.py` for a minimal template.)  Register the new engine in `engines/__init__.py`.  No training loop code should be written inside the engine anymore.
* **New task** – add `tasks/<task>.py` and config folder as described above.
* **New metric / plot** – extend `utils/plots.py` or augment the summary dict; the TXT report is always generated from the JSON.

---

## 8 · Environment Setup (one-time)

```powershell
conda create -n eegnex-env python=3.11 -y
conda activate eegnex-env

# CUDA 11.8 build of PyTorch 2.1 (adjust for your GPU / CUDA version)
pip install torch==2.1.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core scientific stack
pip install torcheeg mne scikit-learn matplotlib pandas seaborn pyyaml pyarrow

# EEGNeX (Braindecode) for the `eegnex` model
pip install "braindecode>=1.1.0,<2.0.0"

# Extra dependencies for Optuna tuning
pip install timm optuna plotly

# Optional: HTML/PDF consolidated reports (first run may need a browser install)
pip install playwright
python -m playwright install chromium

# add project code to PYTHONPATH for interactive work (optional)
set PYTHONPATH=%PYTHONPATH%;D:\numbers_eeg_nn_project\code
```

## 9 · Customisations for NUMBERS-COG Study (2025-07-24)

The upstream DeepTransferEEG codebase was adapted to work with this project’s
pre-processed EEG datasets and training pipeline.  All functional changes are
confined to `third_party/DeepTransferEEG` and one helper script in `scripts/`.

| File | What changed & why |
|------|--------------------|
| `scripts/prepare_data_for_transfer.py` | *New logic*: 1) Recursively finds every `.fif` file under `data_preprocessed/`. 2) Groups by the top-level dataset folder (`acc_0_dataset`, `acc_1_dataset`, `all_trials_dataset`). 3) Saves one `X.npy` and `labels.npy` pair per group into the mirror path inside `third_party/DeepTransferEEG/data/…`.  Label extraction now reads the **second digit of the `Condition` column** and converts to 0-based class IDs. |
| `tl/dnn.py` | • Removed stray markdown back-ticks at EOF (syntax error).  • Added a dedicated configuration block for the **NUMBERS_COG_ACC1** dataset (24 subjects, 129 channels, 175 timepoints, 6 classes).  • `feature_deep_dim` is now computed as `(time_sample_num // 4) - 2` (→ 40).  • Pass a CUDA device index on the command line (`python dnn.py 0`) to enable GPU; falls back to CPU if none is given. |
| `tl/utils/utils.py` | No functional edits, only inspected. |
| `tl/utils/data_utils.py` | Already contained a robust `traintest_split_cross_subject` that supports unequal trial counts; kept unchanged. |
| `tl/utils/dataloader.py` | Unmodified; dataset path expectations documented below. |
| `tl/ttime.py` | No changes made; retains original behaviour. |

### Running the adapted pipeline

1. Convert EEG `.fif` files once:

```powershell
python scripts/prepare_data_for_transfer.py  # produces data/acc_*_dataset/X.npy …
```

2. Train baseline EEGNet with offline EA on the RTX 4090 GPU:

```powershell
# NOTE: This uses a separate environment
conda activate transfer-eeg-env
python -X utf8 -u third_party/DeepTransferEEG/tl/dnn.py 0
```

(The positional `0` selects CUDA device 0; omit it to run on CPU.)

All trained checkpoints are stored in `third_party/DeepTransferEEG/runs/NUMBERS_COG_ACC1/` and logs in `third_party/DeepTransferEEG/logs/`.

---

## 10 · Test-Time Adaptation with T-TIME (2025-07-25)

Goal: reuse the high-accuracy CNN baseline produced by `train.py` and adapt it 
on each unseen subject with the **T-TIME** algorithm from *DeepTransferEEG*.
The original third-party script was surgically extracted to avoid Python-path
conflicts and to guarantee the exact same preprocessing (128 channels, 0-5
labels) used during CNN training.

### 10.1 New files added today

| Path | Purpose |
|------|---------|
| `scripts/run_cnn_tta_experiment.py` | Stand-alone runner that loads **one full CNN run directory** (all 24 LOSO checkpoints), streams the target subject trials, and applies T-TIME online.  Produces `summary.json`, confusion-matrix PNGs and per-trial CSVs in a fresh `results/runs/<timestamp>_landing_digit_tta/` folder. |
| `scripts/tune_tta_experiment.py` | Optuna driver that wraps the runner above, samples hyper-parameters from a YAML search-space, and logs each trial to an SQLite DB. |
| `configs/landing_digit/optuna_space_tta.yaml` | Search space: `lr`, `steps`, `test_batch`, `stride`, `t`, `align`.  Defaults chosen to keep runs affordable on GPU. |
| `code/__init__.py` | Empty marker so `import code.eeg_train` resolves from the local package, not the std-lib `code` module. |
| `third_party/__init__.py` + `third_party/DeepTransferEEG/__init__.py` | Same rationale: make the vendored repo importable without adding it to `sys.path`. |

### 10.2 How to run a single T-TIME pass

```powershell
# Adapt all 24 CNN checkpoints (≈5 min on RTX 4090)
python scripts/run_cnn_tta_experiment.py \
       --run-dir results/runs/20250724_1424_landing_digit_cnn \
       --out-dir results/runs/$(Get-Date -UFormat %Y%m%d_%H%M)_landing_digit_tta \
       --gpu 0                  # optional; defaults to CPU if omitted
```

### 10.3 Hyper-parameter search example

```powershell
python scripts/tune_tta_experiment.py \
       --run-dir results/runs/20250724_1424_landing_digit_cnn \
       --space   configs/landing_digit/optuna_space_tta.yaml \
       --db      sqlite:///optuna_studies/landing_digit_tta.db \
       --trials  50 \
       --gpu     0
```

Each Optuna trial creates a **new** timestamped results folder (and a
`summary.json` similar to the CNN runs) so you can inspect individual
adaptation curves.  The study database can be visualised with:

```powershell
python - <<'PY'
import optuna, plotly.io as pio
study = optuna.load_study("sqlite:///optuna_studies/landing_digit_tta.db", "tta_tuning")
pio.write_image(optuna.visualization.plot_optimization_history(study), "results/optuna_plots/tta_history.png")
PY
```

### 10.4 Performance & GPU flag

If you forget the `--gpu` (or pass an index that is not visible) the script
falls back to `args.data_env = 'local'` and **all forward/backward passes run
on the CPU** – roughly 20× slower.  Always check the banner printed at the
start of `run_cnn_tta_experiment.py`; it will say `device=cuda:0` when the
GPU is active.

---

## 11 · Explainable AI (XAI) Analysis

After a model has been trained, you can run an Explainable AI (XAI) analysis to understand which EEG channels and time-points were most important for its predictions. This process uses the saved model checkpoints from a training run to generate attribution maps via the Integrated Gradients method.

### 11.1 Prerequisites

Before running the XAI analysis, ensure that the training run was completed with the `save_ckpt: true` flag set in its YAML configuration. This guarantees that the necessary model checkpoint files (`.ckpt`) were saved for each fold.

### 11.2 How to Run XAI Analysis

The analysis is performed by the `run_xai_analysis.py` script, which requires the path to a completed run directory.

```powershell
# Activate the environment
conda activate eegnex-env

# Run the XAI analysis on a specific run directory
python scripts/run_xai_analysis.py --run-dir "results/runs/<your_run_directory_name>"
```

For example, for a recent run:
```powershell
python scripts/run_xai_analysis.py --run-dir "results/runs/20250906_1955_numbers_pairs_12_21_cnn_all_trials_dataset (45hz) V2"
```

### 11.3 Understanding the Output

The script will create a new subfolder named `xai_analysis/` inside your run directory. The key output is a consolidated report that summarizes the findings across all cross-validation folds:

*   **`consolidated_xai_report.html` / `.pdf`**: A detailed report showing:
    *   A summary of the top 10 most important channels.
    *   The peak time window of importance.
    *   A grand average **Channel Importance Topoplot** with the top 10 channels labeled.
    *   Grand average and per-fold attribution heatmaps.

This analysis is invaluable for interpreting the model's behavior and linking its predictions back to neurophysiological patterns.

### 11.4 Customizing the XAI Report

#### Top-K Channels

Control the number of “Top” channels shown in the report with `xai_top_k_channels`:

```yaml
xai_top_k_channels: 30
```

This updates the summary text, the channels highlighted on the topoplot, and per-peak analyses.

---

## 12 · Example Commands

### Optuna Hyper-parameter Search

```powershell
# Assumes you have already run: conda activate eegnex-env
python -X utf8 -u scripts/optuna_tune.py `
  --task  landing_digit `
  --engine cnn `
  --base  configs/landing_digit/eegnex_base_45hz.yaml `
  --space configs/landing_digit/optuna_space_eegnex.yaml `
  --db    "sqlite:///optuna_studies/landing_digit_eegnex-acc1-45hz-01.db" `
  --trials 48
```

### Environment Activation

This project uses the `eegnex-env` conda environment. You can also use `mamba` for faster activation.

```powershell
conda activate eegnex-env

also we use conda activate torcheeg-env (secondary)
# or
mamba activate eegnex-env
```


# 5.1 · Recommended staged tuning

- Stage 1 (training & gating): tune `lr`, `batch_size`, `scheduler_patience`, `drop_prob`, `filter_1`, `kernel_block_1_2`, `gate_l1_lambda`, `time_gate_l1_lambda`, `time_gate_tv_lambda` (fix `auto_lr: false`)
- Stage 2 (architecture): fix Stage 1 winners; tune `filter_2`, `kernel_block_4/5`, `avg_pool_block4/5`, `dilation_block_4/5`, `depth_multiplier`, `activation`, and (optionally) `max_norm_*`
- Stage 3 (augmentations): fix architecture; tune `mixup_alpha`, `shift_*`, `scale_*`, `noise_*`, `time_mask_*`, `chan_mask_*`

conda activate convert_for_cartool