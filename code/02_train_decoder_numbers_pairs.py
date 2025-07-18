# D:\numbers_eeg_nn_project\code\02_train_decoder_v3.py
"""Subject-aware EEG decoder ‑ Spectral feature version (Tier-2)

Pipeline:
1. Load all pre-processed epoch files, concatenate across subjects.
2. Feature engineering = (i) per-channel z-score, (ii) Band Differential Entropy (delta-gamma).
3. Leave-One-Subject-Out cross-validation with class-weighted loss.
4. Model = TorchEEG EEGNet; temporal axis equals number of frequency bands (5).
Outputs per run: logs.csv, summary.json, foldN_best_model.pt in results/runs/<timestamp>/
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os, re, glob, json, csv, datetime, subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import mne
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback to plain iterator if tqdm missing

from torcheeg import transforms
from torcheeg.models import EEGNet
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import argparse, yaml, textwrap, sys  # NEW
from run_index import append_run_index

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
LABEL_COLUMN = "Condition"
DATASET_DIR = Path("data_preprocessed/acc_1_dataset")

DEFAULTS = {
    "dataset_dir": str(DATASET_DIR),
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 100,
    "early_stop": 15,
    "noise_std": 0.05,
    "channel_dropout_p": 0.1,
    "max_folds": None,
}

# --------- YAML / CLI helpers ---------

def load_cfg(path: str | None):
    if path is None:
        return {}
    fp = Path(path)
    if not fp.exists():
        sys.exit(f"Config {path} not found")
    txt = fp.read_text()
    return yaml.safe_load(txt) if fp.suffix.lower() in {".yml", ".yaml"} else json.loads(txt)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Numbers-pairs EEG decoder (task-level YAML overrides supported)")
    p.add_argument("--cfg", type=str, help="Path to YAML/JSON overrides")
    p.add_argument("--dataset_dir")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--early_stop", type=int)
    p.add_argument("--noise_std", type=float)
    p.add_argument("--channel_dropout_p", type=float)
    p.add_argument("--max_folds", type=int)
    p.add_argument("--set", nargs="*", metavar="KEY=VAL")
    return p.parse_args()

args = parse_args()
cfg = DEFAULTS.copy()
cfg.update(load_cfg(args.cfg))
explicit = {k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None}
cfg.update(explicit)
if args.set:
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v) # Rely on yaml.safe_load for type inference

    # Re-assign config vars
    DATASET_DIR = Path(cfg["dataset_dir"])
    BATCH_SIZE = cfg["batch_size"]
    LEARNING_RATE = cfg["lr"]
    NUM_EPOCHS = cfg["epochs"]
    EARLY_STOPPING_PATIENCE = cfg["early_stop"]
    NOISE_STD = cfg["noise_std"]
    CHANNEL_DROPOUT_P = cfg["channel_dropout_p"]
    MAX_FOLDS = cfg["max_folds"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE} | CUDA available: {torch.cuda.is_available()}")

    SCRIPT_NAME = Path(__file__).stem # Get script name for output naming

    # Timestamped run directory (local 24-hour clock)
    RUN_ID = datetime.datetime.now().strftime(f"%Y%m%d_%H%M_{SCRIPT_NAME}")
    RUN_DIR = Path("results") / "runs" / RUN_ID
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Helper – current git hash
    def git_hash():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "nogit"

    # ---------------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # ---------------------------------------------------------------------------

    def build_feature_tensors(data_dir: Path):
        """Load epochs, extract spectral or raw-normalized features, return tensors and metadata."""
        fif_files = sorted(data_dir.glob("sub-*_preprocessed-epo.fif"))
        if not fif_files:
            raise FileNotFoundError(f"No .fif files found in {data_dir}")

        subj_pattern = re.compile(r"sub-(\d+)_preprocessed")
        epochs_list = []
        for fp in fif_files:
            ep = mne.read_epochs(fp, preload=True, verbose=False)
            sid = int(subj_pattern.search(fp.name).group(1))
            if ep.metadata is None:
                ep.metadata = mne.create_info([], 0)
            ep.metadata['subject'] = sid
            epochs_list.append(ep)

        all_epochs = mne.concatenate_epochs(epochs_list)
        all_epochs.metadata[LABEL_COLUMN] = all_epochs.metadata[LABEL_COLUMN].astype(str).astype('category')
        print(f"Loaded {len(fif_files)} subjects -> {len(all_epochs)} epochs")

        X_raw = all_epochs.get_data(copy=False) * 1e6  # shape (E, C, T) in µV
        sfreq = all_epochs.info['sfreq']

        # Always use raw normalized voltage features (frequency-band path removed)
        X_norm = transforms.MeanStdNormalize(axis=-1)(eeg=X_raw)['eeg']
        X_tensor = torch.from_numpy(X_norm).float().unsqueeze(1)  # (E,1,C,T)

        print(f"Feature tensor shape: {X_tensor.shape}")

        le = LabelEncoder()
        y_tensor = torch.from_numpy(le.fit_transform(all_epochs.metadata[LABEL_COLUMN].astype(str))).long()
        groups = all_epochs.metadata['subject'].values

        dataset = TensorDataset(X_tensor, y_tensor)
        num_classes = len(le.classes_)
        return dataset, groups, num_classes, le

    # ---------------------------------------------------------------------------
    # 3. TRAIN/VAL HELPERS
    # ---------------------------------------------------------------------------

    def train_epoch(model, loader, optimizer, loss_fn):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # ---------------- Data Augmentation -----------------
            if NOISE_STD > 0:
                x = x + NOISE_STD * torch.randn_like(x)
            if CHANNEL_DROPOUT_P > 0:
                # Create dropout mask broadcast across time axis
                mask = (torch.rand(x.size(0), 1, x.size(2), 1, device=x.device) > CHANNEL_DROPOUT_P).float()
                x = x * mask
            # ----------------------------------------------------

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        return loss_sum / len(loader)


    def eval_epoch(model, loader, loss_fn):
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss_sum += loss_fn(out, y).item()
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return loss_sum / len(loader), 100 * correct / total

    # ---------------------------------------------------------------------------
    # 4. MAIN EXECUTION
    # ---------------------------------------------------------------------------
    if __name__ == "__main__":
        torch.backends.cudnn.benchmark = True

        full_ds, groups_np, num_classes, label_enc = build_feature_tensors(DATASET_DIR)
        data_tensor, _ = full_ds.tensors  # tensors property of TensorDataset

        # Logging setup
        summary = {
            "run_id": RUN_ID,
            "git": git_hash(),
            "device": str(DEVICE),
            "task": "numbers_pairs",
            "classes": [str(i) for i in range(6)],
            "hyper": cfg.copy(),
            "fold_metrics": []
        }

        log_f = open(RUN_DIR / f"logs_{SCRIPT_NAME}.csv", "w", newline="")
        csv_writer = csv.writer(log_f)
        csv_writer.writerow(["fold", "epoch", "train_loss", "val_loss", "val_acc"])

        logo = LeaveOneGroupOut()
        fold_accs = []
        cm_total = np.zeros((num_classes, num_classes), dtype=int)

        for fold, (train_idx, val_idx) in enumerate(logo.split(np.zeros(len(full_ds)), np.zeros(len(full_ds)), groups_np)):
            if MAX_FOLDS is not None and fold >= MAX_FOLDS:
                print(f"Reached MAX_FOLDS ({MAX_FOLDS}); stopping early for quick iteration.")
                break

            print(f"\n======== Fold {fold+1}/{len(np.unique(groups_np))} ========")

            # DataLoaders
            train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

            # Class-weighted loss
            y_train = full_ds.tensors[1][train_idx].numpy()
            weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            loss_fn = nn.CrossEntropyLoss(torch.tensor(weights, dtype=torch.float32).to(DEVICE))

            # Model init – chunk_size is number of bands or T
            _, _, num_electrodes, chunk_size = data_tensor.shape
            model = EEGNet(num_classes=num_classes, num_electrodes=num_electrodes, chunk_size=chunk_size, dropout=0.5).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

            # Early stopping vars
            best_val, patience = float('inf'), 0
            best_acc = 0.0
            best_path = RUN_DIR / f"fold{fold+1}_best_model_{SCRIPT_NAME}.pt"

            for epoch in range(1, NUM_EPOCHS + 1):
                tr_loss = train_epoch(model, train_loader, optimizer, loss_fn)
                val_loss, val_acc = eval_epoch(model, val_loader, loss_fn)
                scheduler.step(val_loss)

                csv_writer.writerow([fold+1, epoch, f"{tr_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.2f}"])
                print(f"Fold {fold+1} | Epoch {epoch:03d} | Train {tr_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.2f}%")

                if val_loss < best_val:
                    torch.save(model.state_dict(), best_path)
                    best_val, best_acc, patience = val_loss, val_acc, 0
                else:
                    patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping…")
                    break

            # Confusion matrix on validation set using final model
            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    out = model(xb)
                    preds = out.argmax(1).cpu().numpy()
                    y_true.extend(yb.numpy())
                    y_pred.extend(preds)

            labels_range = list(range(num_classes))
            cm = confusion_matrix(y_true, y_pred, labels=labels_range)
            cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

            # Heatmap with counts annotation and outlined diagonal
            plt.figure(figsize=(12,10))
            row_tot = cm.sum(axis=1, keepdims=True)
            annot = np.round(cm_pct, 1)
            ax = sns.heatmap(cm_pct, annot=annot, fmt='.1f', cmap='Blues', xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
            plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Fold {fold+1} Confusion (%)')
            plt.xticks(rotation=0); plt.yticks(rotation=0)
            # outline diagonal
            for i in range(num_classes):
                ax.add_patch(Rectangle((i,i),1,1,fill=False,edgecolor='black',lw=1.5))
            # outline highest-percentage cell per row (thicker)
            for i in range(num_classes):
                j = int(np.argmax(cm_pct[i]))
                ax.add_patch(Rectangle((j,i),1,1,fill=False,edgecolor='red',lw=2.5))
            plt.tight_layout()
            plt.savefig(RUN_DIR / f'fold{fold+1}_confusion_{SCRIPT_NAME}.png')
            plt.close()

            # accumulate for overall
            cm_total += cm

            fold_accs.append(best_acc)
            summary["fold_metrics"].append({"fold": fold+1, "best_val_acc": best_acc, "confusion_matrix": cm.tolist()})
            print(f"Fold {fold+1} best acc: {best_acc:.2f}%")

        summary["mean_acc"] = float(np.mean(fold_accs))
        summary["std_acc"] = float(np.std(fold_accs))
        chance = 100/num_classes
        recalls = np.divide(cm_total.diagonal(), cm_total.sum(axis=1), out=np.zeros(num_classes), where=cm_total.sum(axis=1)!=0)
        best_idx = int(np.argmax(recalls)); worst_idx = int(np.argmin(recalls))
        summary.update({
            "chance_level_pct": chance,
            "confusion_total": cm_total.tolist(),
            "per_digit_recall": [float(r) for r in recalls],
            "best_digit": str(best_idx),
            "worst_digit": str(worst_idx)
        })

        # write summary json
        summ_path = RUN_DIR / f"summary_{SCRIPT_NAME}.json"
        with open(summ_path, "w") as fp:
            json.dump(summary, fp, indent=2)

        # generate txt report from json
        with open(summ_path) as fp:
            s = json.load(fp)
        report_lines = [
            f"{SCRIPT_NAME} report (raw z-score)",
            f"Run ID        : {s['run_id']}",
            "",
            "--- Hyper-parameters ---",
            f"  Batch size          : {s['hyper']['batch_size']}",
            f"  Learning rate       : {s['hyper']['lr']}",
            f"  Max epochs          : {s['hyper']['epochs']}",
            f"  Early-stop patience : {s['hyper']['early_stop']}",
            f"  Noise std           : {s['hyper']['noise_std']}",
            f"  Channel dropout P   : {s['hyper']['channel_dropout_p']}",
            "",
            "--- Results ---",
            f"Mean LOSO accuracy : {s['mean_acc']:.2f}% (±{s['std_acc']:.2f}%)",
            f"Chance level      : {s['chance_level_pct']:.2f}% (6-class)",
            "",
            "Per-digit recall (TPR):"
        ]
        for i,r in enumerate(s['per_digit_recall']):
            report_lines.append(f"  {i} : {r*100:.1f}%   | TP={s['confusion_total'][i][i]} / {np.sum(s['confusion_total'][i])}")
        report_lines.extend([
            "",
            f"Best recalled digit : {s['best_digit']}",
            f"Worst recalled digit: {s['worst_digit']}"
        ])
        with open(RUN_DIR / f"report_{SCRIPT_NAME}.txt", "w") as rp:
            rp.write("\n".join(report_lines))

        # Append to expandable index CSV
        append_run_index(s, SCRIPT_NAME)

        print(f"Report written to {RUN_DIR / f'report_{SCRIPT_NAME}.txt'}")

        # Overall confusion heatmap
        cm_total_pct = cm_total / cm_total.sum(axis=1, keepdims=True) * 100
        plt.figure(figsize=(12,10))
        row_tot_o = cm_total.sum(axis=1, keepdims=True)
        annot_o = np.round(cm_total_pct, 1)
        ax = sns.heatmap(cm_total_pct, annot=annot_o, fmt='.1f', cmap='Blues', xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Overall Confusion (%)')
        plt.xticks(rotation=0); plt.yticks(rotation=0)
        for i in range(num_classes): ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=1.5))
        for i in range(num_classes):
            j = int(np.argmax(cm_total_pct[i]))
            ax.add_patch(Rectangle((j,i),1,1,fill=False,edgecolor='red',lw=2.5))
        plt.tight_layout()
        plt.savefig(RUN_DIR / f'overall_confusion_{SCRIPT_NAME}.png')
        plt.close()

        log_f.close()
        print("\n==== SUMMARY ====")
        print(f"Mean accuracy: {summary['mean_acc']:.2f}% (+/- {summary['std_acc']:.2f})")
        print(f"Chance level ({num_classes} classes): {chance:.2f}%") 