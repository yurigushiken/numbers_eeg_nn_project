import os
import re
import glob
from pathlib import Path
import json, csv, datetime, subprocess
import argparse, yaml, textwrap, sys

import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from torcheeg.models import EEGNet
from torcheeg import transforms
from run_index import append_run_index

"""EEGNet binary decoder: landing-on-1 vs other.
Derived from 02_train_decoder_v3.py but maps the original Condition codes into a
binary target column `land1` (1 if the ones digit of the Condition == 1, else 0).
Only uses accuracy-filtered dataset (acc_1_dataset).

We keep the quick-iteration settings: MAX_FOLDS = 2, raw z-score features,
no data augmentation (can be enabled later).
"""

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
LABEL_COLUMN = "land1"  # binary target created on-the-fly
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

def load_cfg(p):
    if p is None: return {}
    fp=Path(p)
    if not fp.exists(): sys.exit(f"cfg {p} missing")
    txt=fp.read_text(); return yaml.safe_load(txt) if fp.suffix.lower() in {'.yml','.yaml'} else json.loads(txt)

def parse_args():
    pr=argparse.ArgumentParser(description="land1 binary decoder YAML")
    pr.add_argument('--cfg')
    for k in DEFAULTS: pr.add_argument(f"--{k}")
    pr.add_argument('--set', nargs='*')
    return pr.parse_args()

args=parse_args(); cfg=DEFAULTS.copy(); cfg.update(load_cfg(args.cfg)); cfg.update({k:getattr(args,k) for k in DEFAULTS if getattr(args,k) is not None})
if args.set:
    for kv in args.set:
        k,v=kv.split('=',1)
        cfg[k]=yaml.safe_load(v) # Rely on yaml.safe_load for type inference

DATASET_DIR=Path(cfg['dataset_dir']); BATCH_SIZE=cfg['batch_size']; LEARNING_RATE=cfg['lr']; NUM_EPOCHS=cfg['epochs']; EARLY_STOPPING_PATIENCE=cfg['early_stop']; NOISE_STD=cfg['noise_std']; CHANNEL_DROPOUT_P=cfg['channel_dropout_p']; MAX_FOLDS=cfg['max_folds']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda.available: {torch.cuda.is_available()} | device: {DEVICE}")

SCRIPT_NAME = Path(__file__).stem # Get script name for output naming
RUN_ID = datetime.datetime.now().strftime(f"%Y%m%d_%H%M_{SCRIPT_NAME}")
RUN_DIR = Path("results") / "runs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------

def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def build_feature_tensors(root: Path):
    fif_files = sorted(root.glob("sub-*preprocessed-epo.fif"))
    if not fif_files:
        raise FileNotFoundError(f"No .fif files found in {root}")

    subj_re = re.compile(r"sub-(\d+)_preprocessed")
    epochs_list = []
    for fp in fif_files:
        ep = mne.read_epochs(fp, preload=True, verbose=False)
        sid = int(subj_re.search(fp.name).group(1))
        if ep.metadata is None:
            ep.metadata = mne.create_info([], 0)
        ep.metadata['subject'] = sid
        epochs_list.append(ep)

    all_ep = mne.concatenate_epochs(epochs_list)

    # ---- Create binary label ----
    cond_int = all_ep.metadata['Condition'].astype(int)
    all_ep.metadata[LABEL_COLUMN] = (cond_int % 10 == 1).astype(int)

    # cast to category for consistency
    all_ep.metadata[LABEL_COLUMN] = all_ep.metadata[LABEL_COLUMN].astype('category')

    print(f"Loaded {len(fif_files)} subjects -> {len(all_ep)} epochs")

    # --- Feature extraction: per-channel z-score ---
    X = all_ep.get_data(copy=False) * 1e6  # µV
    X = transforms.MeanStdNormalize(axis=-1)(eeg=X)['eeg']
    X_t = torch.from_numpy(X).float().unsqueeze(1)  # (E,1,C,T)

    le = LabelEncoder()
    y_t = torch.from_numpy(le.fit_transform(all_ep.metadata[LABEL_COLUMN])).long()
    num_classes = len(le.classes_)

    groups = all_ep.metadata['subject'].values
    ds = TensorDataset(X_t, y_t)
    return ds, groups, num_classes

# ---------------------------------------------------------------------------
# 3. TRAIN / VAL
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optim_, loss_fn):
    model.train()
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if NOISE_STD > 0:
            xb = xb + NOISE_STD * torch.randn_like(xb)
        if CHANNEL_DROPOUT_P > 0:
            m = (torch.rand(xb.size(0), 1, xb.size(2), 1, device=xb.device) > CHANNEL_DROPOUT_P).float()
            xb = xb * m
        optim_.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optim_.step()
        loss_sum += loss.item()
    return loss_sum / len(loader)


def eval_epoch(model, loader, loss_fn):
    model.eval()
    l_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            l_sum += loss_fn(out, yb).item()
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return l_sum / len(loader), 100 * correct / total

# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    ds_full, groups_np, num_classes = build_feature_tensors(DATASET_DIR)
    data_tensor, _ = ds_full.tensors

    # logging setup
    summary = {
        "run_id": RUN_ID,
        "git": git_hash(),
        "device": str(DEVICE),
        "task": "land1_vs_other",
        "classes": [str(i) for i in range(2)],
        "hyper": cfg.copy(),
        "fold_metrics": []
    }

    log_f = open(RUN_DIR / f"logs_{SCRIPT_NAME}.csv", "w", newline="")
    csv_wr = csv.writer(log_f)
    csv_wr.writerow(["fold", "epoch", "train_loss", "val_loss", "val_acc"])

    logo = LeaveOneGroupOut()
    cm_total = np.zeros((2,2), dtype=int)

    for fold, (tr_idx, val_idx) in enumerate(logo.split(np.zeros(len(ds_full)), np.zeros(len(ds_full)), groups_np)):
        if MAX_FOLDS is not None and fold >= MAX_FOLDS:
            print(f"Reached MAX_FOLDS ({MAX_FOLDS}); stopping.")
            break

        print(f"\n===== Fold {fold+1} =====")
        train_ld = DataLoader(Subset(ds_full, tr_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_ld = DataLoader(Subset(ds_full, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # weighted loss
        y_train = ds_full.tensors[1][tr_idx].numpy()
        w = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        loss_fn = nn.CrossEntropyLoss(torch.tensor(w, dtype=torch.float32).to(DEVICE))

        _, _, C, T = data_tensor.shape
        model = EEGNet(num_classes=num_classes, num_electrodes=C, chunk_size=T, dropout=0.5).to(DEVICE)
        optim_ = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        sched = ReduceLROnPlateau(optim_, 'min', patience=5, factor=0.5)

        best_val, best_acc, patience = float('inf'), 0.0, 0
        best_path = RUN_DIR / f"fold{fold+1}_best.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            tr_loss = train_epoch(model, train_ld, optim_, loss_fn)
            val_loss, val_acc = eval_epoch(model, val_ld, loss_fn)
            sched.step(val_loss)

            csv_wr.writerow([fold+1, epoch, f"{tr_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.2f}"])
            print(f"Fold {fold+1} | Ep {epoch:03d} | Tr {tr_loss:.3f} | Val {val_loss:.3f} | Acc {val_acc:.2f}%")

            if val_loss < best_val:
                torch.save(model.state_dict(), best_path)
                best_val, best_acc, patience = val_loss, val_acc, 0
            else:
                patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early stopping.")
                break

        # After training, compute confusion matrix on validation set
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_ld:
                xb = xb.to(DEVICE)
                out = model(xb)
                preds = out.argmax(1).cpu().numpy()
                y_true.extend(yb.numpy())
                y_pred.extend(preds)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        print(f"Confusion matrix Fold {fold+1}:\n{cm}")

        # Save heatmap
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot_pct = np.round(cm_pct,1)
        plt.figure(figsize=(4,4))
        ax = sns.heatmap(cm_pct, annot=annot_pct, fmt='.1f', cmap='Blues', xticklabels=['other','land1'], yticklabels=['other','land1'])
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Fold {fold+1} Confusion')
        plt.xticks(rotation=0); plt.yticks(rotation=0)
        # outline diagonal
        for i in range(2):
            ax.add_patch(Rectangle((i,i),1,1,fill=False,edgecolor='black',lw=1.5))
        # outline highest percentage cell per row
        for i in range(2):
            j = int(np.argmax(cm_pct[i]))
            ax.add_patch(Rectangle((j,i),1,1,fill=False,edgecolor='red',lw=2.5))
        plt.tight_layout()
        plt.savefig(RUN_DIR / f'fold{fold+1}_confusion_{SCRIPT_NAME}.png')
        plt.close()

        cm_total += cm

        summary["fold_metrics"].append({"fold": fold+1, "best_val_acc": best_acc, "confusion_matrix": cm.tolist()})

    accs = [m["best_val_acc"] for m in summary["fold_metrics"]]
    summary["mean_acc"] = float(np.mean(accs))
    summary["std_acc"] = float(np.std(accs))
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

    # Persist the enriched summary
    summ_path = RUN_DIR / f"summary_{SCRIPT_NAME}.json"
    with open(summ_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    # Reload to guarantee that the txt report is generated strictly from persisted JSON data
    with open(summ_path) as fp:
        summary_loaded = json.load(fp)

    log_f.close()
    print("\n==== SUMMARY ====")
    print(f"Mean acc: {summary['mean_acc']:.2f}% (+/- {summary['std_acc']:.2f})")
    # Overall confusion heatmap
    cm_total_pct = cm_total/cm_total.sum(axis=1,keepdims=True)*100
    annot_tot = np.round(cm_total_pct,1)
    plt.figure(figsize=(4,4))
    ax = sns.heatmap(cm_total_pct, annot=annot_tot, fmt='.1f', cmap='Blues', xticklabels=summary_loaded['classes'], yticklabels=summary_loaded['classes'])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Overall Confusion (24 folds)')
    plt.xticks(rotation=0); plt.yticks(rotation=0)
    for i in range(num_classes):
        ax.add_patch(Rectangle((i,i),1,1,fill=False,edgecolor='black',lw=1.5))
        j=int(np.argmax(cm_total_pct[i]))
        ax.add_patch(Rectangle((j,i),1,1,fill=False,edgecolor='red',lw=2.5))
    plt.tight_layout(); plt.savefig(RUN_DIR / f'overall_confusion_{SCRIPT_NAME}.png'); plt.close()

    # Textual report from loaded summary (guarantees consistency)
    report_lines = [
        f"{SCRIPT_NAME} report (raw z-score)",
        f"Run ID        : {summary_loaded['run_id']}",
        "",
        "--- Hyper-parameters ---",
        f"  Batch size          : {summary_loaded['hyper']['batch_size']}",
        f"  Learning rate       : {summary_loaded['hyper']['lr']}",
        f"  Max epochs          : {summary_loaded['hyper']['epochs']}",
        f"  Early-stop patience : {summary_loaded['hyper']['early_stop']}",
        f"  Noise std           : {summary_loaded['hyper']['noise_std']}",
        f"  Channel dropout P   : {summary_loaded['hyper']['channel_dropout_p']}",
        "",
        "--- Results ---",
        f"Mean LOSO accuracy : {summary_loaded['mean_acc']:.2f}% (±{summary_loaded['std_acc']:.2f}%)",
        f"Chance level      : {summary_loaded['chance_level_pct']:.2f}% (binary)",
        "",
        "Per-class recall (TPR):"
    ]
    for i,r in enumerate(summary_loaded['per_digit_recall']):
        report_lines.append(f"  {summary_loaded['classes'][i]} : {r*100:.1f}%   | TP={summary_loaded['confusion_total'][i][i]} / {np.sum(summary_loaded['confusion_total'][i])}")
    report_lines.extend([
        "",
        f"Best recalled class : {summary_loaded['best_digit']}",
        f"Worst recalled class: {summary_loaded['worst_digit']}"
    ])
    with open(RUN_DIR / f"report_{SCRIPT_NAME}.txt", "w") as rp:
        rp.write("\n".join(report_lines))
    print(f"Report written to {RUN_DIR / f'report_{SCRIPT_NAME}.txt'}")

    # Append to expandable index CSV
    append_run_index(summary_loaded, SCRIPT_NAME) 