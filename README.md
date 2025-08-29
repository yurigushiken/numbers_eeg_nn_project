# EEG Numerical-Cognition Decoding Project (Unified 2025-07)

**⚠️ Quick environment setup — READ ME FIRST**

*All commands in this repo assume the `torcheeg-env` conda environment.*  Pick **one** of the following patterns before running anything:
Cursor / VS Code agent shells often launch without `conda` on `PATH`.  The fool-proof way to run any command in this project is to call the interpreter that lives *inside* the environment:

```powershell
# Windows – adapt the path if your env lives elsewhere
& "$Env:USERPROFILE\.conda\envs\torcheeg-env\python.exe" -X utf8 -u train.py --task landing_digit --engine hybrid
```

This implicitly activates the environment, so **no prior `conda activate` is required**.  `-X utf8` (or setting `PYTHONIOENCODING=utf-8`) avoids Unicode errors in Windows consoles; alternatively run `chcp 65001` once per shell.

(If you already have `conda` on PATH you can still `conda activate torcheeg-env` and run `python …`, but the interpreter-path method above is the reference approach used throughout this README.)

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

All three engines are launched through `train.py`.  Internally they all call the
shared `code/training_runner.py`, so every experiment (raw EEG or spectrogram)
follows the exact same LOSO, early-stopping and reporting logic.  Each run
produces the following artefacts:

```
results/runs/<timestamp>_<task>_<engine>/
    summary_*.json      # single source of truth
    report_*.txt        # human-readable
    fold1_confusion.png fold1_curves.png …
    overall_confusion.png
results/runs_index.csv  # global catalogue auto-expanded
```

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

---

## 4 · Configuration Files

Each task folder can hold multiple presets:

* `base.yaml`       – generic defaults (raw-EEG engines)
* `eegnet_se_base.yaml` – SE-enhanced EEGNet defaults
* `multi_scale_cnn_base.yaml` – Multi-Scale CNN defaults
* `optuna_space.yaml`

Example `configs/landing_digit/multi_scale_cnn_base.yaml` excerpt:

```yaml
dataset_dir: data_preprocessed/acc_1_dataset (30hz) V2 # Or (45hz) V2, etc.
model_name: multi_scale_cnn
kernel_sizes: [7, 11, 15, 19]
batch_size: 64
lr: 1e-4
epochs: 100
early_stop: 15
```

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

The tuner samples parameters, calls the chosen engine **in-process**, records
the `mean_acc`, and appends each trial to `results/runs_index.csv`.

---

## 6 · Common Issues & Fixes

| Symptom | Remedy |
|---------|--------|
| `ModuleNotFoundError: models.cwa_transformer` | Ensure `code/` is on `PYTHONPATH` (train.py inserts it automatically) or run Python directly after `conda activate torcheeg-env`. |
| Confusion matrix only shows first row | Your spectrogram metadata already contains the `landing_digit` column; the task’s `label_fn` now checks for it. |

---

## 7 · Extending the Project

* **New engine** – create `engines/<name>.py` that picks a **Dataset**, a **model builder**, and an **augmentation builder**, then calls `TrainingRunner`.  (See `engines/cnn.py` for a minimal template.)  Register the new engine in `engines/__init__.py`.  No training loop code should be written inside the engine anymore.
* **New task** – add `tasks/<task>.py` and config folder as described above.
* **New metric / plot** – extend `utils/plots.py` or augment the summary dict; the TXT report is always generated from the JSON.

---

## 8 · Environment Setup (one-time)

```powershell
conda create -n torcheeg-env python=3.11 -y
conda activate torcheeg-env

# CUDA 11.8 build of PyTorch 2.1 (adjust for your GPU / CUDA version)
pip install torch==2.1.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core scientific stack
pip install torcheeg mne scikit-learn matplotlib pandas seaborn pyyaml pyarrow

# Extra dependencies for Optuna tuning
pip install timm optuna plotly

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
& "$Env:USERPROFILE\.conda\envs\transfer-eeg-env\python.exe" -X utf8 -u third_party/DeepTransferEEG/tl/dnn.py 0
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

& "$Env:USERPROFILE\.conda\envs\torcheeg-env\python.exe" -X utf8 -u scripts/optuna_tune.py `
>>   --task  landing_digit `
>>   --engine cnn `
>>   --base  configs/landing_digit/eegnex_base_45hz.yaml `
>>   --space configs/landing_digit/optuna_space_eegnex.yaml `
>>   --db    "sqlite:///optuna_studies/landing_digit_eegnex-acc1-45hz-01.db" `
>>   --trials 48

We are now using mamba
mamba activate torcheeg-env
conda activate torcheeg-env

conda activate eegnex-env