import os
import re
from pathlib import Path
import json, csv, datetime, subprocess
import argparse, textwrap, sys, yaml
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

"""EEGNet decoder: classify landing digit (ones-place of Condition) **with enhanced data augmentation**.

Augmentation applied only on the training batches (on-the-fly):
  • Random circular time-shift (±0.5–4 % of window length)
  • Random amplitude scaling (0.9–1.1 ×)
  • Random additive Gaussian noise (σ = 0.02)

No Band Differential Entropy features are used.
"""

# ---------------- CONFIG -----------------
DATASET_DIR = Path("data_preprocessed/acc_1_dataset")  # default; now overridable
LABEL_COLUMN = "landing_digit"

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100
EARLY_STOP = 15

# ----- augmentation hyper-params (fractions refer to window length T) -----
SHIFT_P = 0.5        # probability to apply time-shift
SHIFT_MIN_FRAC = 0.005  # min shift as fraction of T
SHIFT_MAX_FRAC = 0.04   # max shift as fraction of T
SCALE_P = 0.5         # probability to apply amplitude scaling
SCALE_MIN = 0.9
SCALE_MAX = 1.1
NOISE_P = 0.3         # probability to apply additive noise
NOISE_STD = 0.02

MAX_FOLDS = None  # run all subjects

# ---- NEW unified default map (will be merged with YAML / CLI overrides) ----
DEFAULTS = {
    "dataset_dir": str(DATASET_DIR),
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "epochs": EPOCHS,
    "early_stop": EARLY_STOP,
    "shift_p": SHIFT_P,
    "shift_min_frac": SHIFT_MIN_FRAC,
    "shift_max_frac": SHIFT_MAX_FRAC,
    "scale_p": SCALE_P,
    "scale_min": SCALE_MIN,
    "scale_max": SCALE_MAX,
    "noise_p": NOISE_P,
    "noise_std": NOISE_STD,
    "max_folds": MAX_FOLDS,
}

# ---------------- CLI / YAML helpers (NEW) -----------------

def load_cfg(path: str | None) -> dict:
    """Return {} if path is None; else load YAML or JSON file."""
    if path is None:
        return {}
    fp = Path(path)
    if not fp.exists():
        sys.exit(f"Config file {path} not found.")
    text = fp.read_text()
    try:
        return yaml.safe_load(text) if fp.suffix.lower() in {".yml", ".yaml"} else json.loads(text)
    except Exception as e:
        sys.exit(f"Failed to parse config {path}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        EEGNet landing-digit trainer with on-the-fly augmentation.
        You may override any hyper-parameter via CLI flags or a YAML/JSON file.
        Examples:
          python script.py --cfg configs/landing_digit/base.yaml
          python script.py --lr 5e-4 --batch_size 128 --dataset_dir data_preprocessed/acc_1_dataset
        """)
    )
    p.add_argument("--cfg", type=str, help="Path to YAML/JSON with parameter overrides")
    # expose common flags explicitly
    p.add_argument("--dataset_dir", type=str)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--early_stop", type=int)
    p.add_argument("--shift_p", type=float)
    p.add_argument("--shift_min_frac", type=float)
    p.add_argument("--shift_max_frac", type=float)
    p.add_argument("--scale_p", type=float)
    p.add_argument("--scale_min", type=float)
    p.add_argument("--scale_max", type=float)
    p.add_argument("--noise_p", type=float)
    p.add_argument("--noise_std", type=float)
    p.add_argument("--max_folds", type=int)
    # free-form overrides --set k=v
    p.add_argument("--set", nargs="*", metavar="KEY=VAL",
                   help="Arbitrary param overrides, e.g. --set lr=0.0005 scale_min=0.8")
    return p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA: {torch.cuda.is_available()} | device: {DEVICE}")

SCRIPT_NAME = Path(__file__).stem  # Get script name for output naming
RUN_ID = datetime.datetime.now().strftime(f"%Y%m%d_%H%M_{SCRIPT_NAME}")
RUN_DIR = Path("results") / "runs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

torch.backends.cudnn.benchmark = True
# Allow fast non-deterministic algorithms for the shift op.
torch.use_deterministic_algorithms(False)

# ------------ helpers --------------

def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"

def load_epochs_concat(root: Path):
    files = sorted(root.glob("sub-*preprocessed-epo.fif"))
    subj_re = re.compile(r"sub-(\d+)_preprocessed")
    eps = []
    for fp in files:
        ep = mne.read_epochs(fp, preload=True, verbose=False)
        sid = int(subj_re.search(fp.name).group(1))
        if ep.metadata is None:
            ep.metadata = mne.create_info([], 0)
        ep.metadata['subject'] = sid
        eps.append(ep)
    all_ep = mne.concatenate_epochs(eps)
    # derive landing digit label
    cond = all_ep.metadata['Condition'].astype(int)
    all_ep.metadata[LABEL_COLUMN] = (cond % 10).astype(str)
    print(f"Loaded {len(files)} subjects -> {len(all_ep)} epochs")
    return all_ep

def build_tensors(epochs: mne.Epochs):
    X = epochs.get_data(copy=False) * 1e6  # V → µV
    X = transforms.MeanStdNormalize(axis=-1)(eeg=X)['eeg']
    X_t = torch.from_numpy(X).float().unsqueeze(1)  # shape: N×1×C×T
    le = LabelEncoder()
    y_t = torch.from_numpy(le.fit_transform(epochs.metadata[LABEL_COLUMN])).long()
    groups = epochs.metadata['subject'].values
    return X_t, y_t, groups, le

# -------------- augmentation helpers --------------

class RandomAmpScale:
    """Randomly scale the entire epoch amplitude by U(scale_min, scale_max) with prob. *p*."""

    def __init__(self, p: float = SCALE_P, scale_min: float = SCALE_MIN, scale_max: float = SCALE_MAX):
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        # Apply amplitude scaling with probability p and return the tensor.
        if torch.rand(1, device=eeg.device).item() < self.p:
            s = torch.empty(1, device=eeg.device).uniform_(self.scale_min, self.scale_max)
            eeg = eeg * s
        return eeg


# --------- GPU-friendly augmentation transforms (PyTorch) ---------


class RandomTimeShiftTorch:
    """Roll the signal along `dim` by a random amount within [shift_min, shift_max]."""

    def __init__(self, p: float, shift_min: int, shift_max: int, dim: int = -1):
        self.p = p
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.dim = dim

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:  # eeg already on GPU
        if self.shift_max > 0 and torch.rand(1, device=eeg.device).item() < self.p:
            shift = torch.randint(self.shift_min, self.shift_max + 1, (1,), device=eeg.device).item()
            # 50 % chance to roll left or right
            if torch.rand(1, device=eeg.device).item() < 0.5:
                shift = -shift
            eeg = torch.roll(eeg, shifts=shift, dims=self.dim)
        return eeg


class RandomNoiseTorch:
    """Add Gaussian noise with given std with probability *p*."""

    def __init__(self, p: float, std: float):
        self.p = p
        self.std = std

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.std > 0 and torch.rand(1, device=eeg.device).item() < self.p:
            eeg = eeg + self.std * torch.randn_like(eeg)
        return eeg


class ComposeTorch:
    """Simple compose that runs each transform in sequence (PyTorch tensors)."""

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            eeg = t(eeg)
        return eeg

def make_train_aug(T: int):
    """Create GPU-friendly augmentation pipeline for window length *T* samples."""
    return ComposeTorch([
        RandomTimeShiftTorch(p=SHIFT_P,
                             shift_min=max(1, int(SHIFT_MIN_FRAC * T)),
                             shift_max=int(SHIFT_MAX_FRAC * T)),
        RandomAmpScale(p=SCALE_P, scale_min=SCALE_MIN, scale_max=SCALE_MAX),
        RandomNoiseTorch(p=NOISE_P, std=NOISE_STD)
    ])

# -------------- train helpers ----------------

def train_epoch(model, loader, opt, loss_fn, aug):
    model.train()
    tot = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        xb = aug(xb)  # GPU-based augmentation
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot / len(loader)


def eval_epoch(model, loader, loss_fn):
    model.eval()
    tot = 0
    corr = 0
    N = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = loss_fn(out, yb)
            tot += loss.item()
            pred = out.argmax(1)
            corr += (pred == yb).sum().item()
            N += yb.size(0)
    return tot / len(loader), 100 * corr / N

# --------------- main ----------------
if __name__ == "__main__":
    # ---- resolve configuration (NEW) ----
    args = parse_args()
    cfg = DEFAULTS.copy()
    cfg.update(load_cfg(args.cfg))
    explicit = {k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None}
    cfg.update(explicit)
    if args.set:
        for kv in args.set:
            if "=" not in kv:
                sys.exit("--set expects KEY=VAL pairs")
            k, v = kv.split("=", 1)
            if k not in DEFAULTS:
                sys.exit(f"Unknown hyper-parameter {k}")
            cfg[k] = yaml.safe_load(v) # Rely on yaml.safe_load for type inference

    # ---- re-assign module-level constants so downstream code uses overrides ----
    DATASET_DIR = Path(cfg["dataset_dir"])
    BATCH_SIZE  = int(cfg["batch_size"])
    LR          = float(cfg["lr"])
    EPOCHS      = int(cfg["epochs"])
    EARLY_STOP  = int(cfg["early_stop"])
    SHIFT_P     = float(cfg["shift_p"])
    SHIFT_MIN_FRAC = float(cfg["shift_min_frac"])
    SHIFT_MAX_FRAC = float(cfg["shift_max_frac"])
    SCALE_P     = float(cfg["scale_p"])
    SCALE_MIN   = float(cfg["scale_min"])
    SCALE_MAX   = float(cfg["scale_max"])
    NOISE_P     = float(cfg["noise_p"])
    NOISE_STD   = float(cfg["noise_std"])
    MAX_FOLDS   = cfg["max_folds"] # Can be None or int, no explicit cast needed

    epochs_all = load_epochs_concat(DATASET_DIR)
    X_t, y_t, groups, le = build_tensors(epochs_all)
    num_classes = len(le.classes_)
    full_ds = TensorDataset(X_t, y_t)

    summary = {"run_id": RUN_ID, "git": git_hash(), "device": str(DEVICE),
                "task": "landing_digit", "classes": list(le.classes_),
                "hyper": cfg.copy(), "fold_metrics": []}
    # Augment hyper-parameter section with all additional settings so both JSON and txt match
    # summary["hyper"].update({
    #     "batch_size": BATCH_SIZE,
    #     "early_stop": EARLY_STOP,
    #     "shift_p": SHIFT_P,
    #     "shift_min_frac": SHIFT_MIN_FRAC,
    #     "shift_max_frac": SHIFT_MAX_FRAC,
    #     "scale_p": SCALE_P,
    #     "scale_min": SCALE_MIN,
    #     "scale_max": SCALE_MAX,
    #     "noise_p": NOISE_P,
    #     "noise_std": NOISE_STD,
    # })

    log_path = RUN_DIR / f"logs_{SCRIPT_NAME}.csv"
    csv_f = open(log_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["fold", "epoch", "train_loss", "val_loss", "val_acc"])

    logo = LeaveOneGroupOut()
    cm_total = np.zeros((num_classes, num_classes), int)
    fold_accs = []

    for fold, (tr_idx, val_idx) in enumerate(logo.split(np.zeros(len(full_ds)), np.zeros(len(full_ds)), groups)):
        if MAX_FOLDS and fold >= MAX_FOLDS:
            break
        print(f"\n=== Fold {fold + 1} ===")
        train_ld = DataLoader(Subset(full_ds, tr_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_ld = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        cls_w = compute_class_weight('balanced', classes=np.unique(y_t.numpy()), y=y_t[tr_idx].numpy())
        loss_fn = nn.CrossEntropyLoss(torch.tensor(cls_w, dtype=torch.float32).to(DEVICE))

        _, _, C, T = X_t.shape
        model = EEGNet(num_classes=num_classes, num_electrodes=C, chunk_size=T, dropout=0.5).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=LR)
        sched = ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)

        train_aug = make_train_aug(T)

        best_v = float('inf')
        best_acc = 0
        patience = 0
        for epoch in range(1, EPOCHS + 1):
            tr_l = train_epoch(model, train_ld, opt, loss_fn, train_aug)
            val_l, val_a = eval_epoch(model, val_ld, loss_fn)
            sched.step(val_l)
            csv_w.writerow([fold + 1, epoch, f"{tr_l:.3f}", f"{val_l:.3f}", f"{val_a:.2f}"])
            if val_l < best_v:
                best_v, best_acc, patience = val_l, val_a, 0
            else:
                patience += 1
            if patience >= EARLY_STOP:
                break
        print(f"Fold {fold + 1} best acc {best_acc:.2f}%")
        fold_accs.append(best_acc)

        # confusion on val set
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_ld:
                pred = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
                y_pred.extend(pred)
                y_true.extend(yb.numpy())
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        cm_total += cm
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = [[f"{cm[i, j]}/{cm.sum(axis=1)[i]}" if cm[i, j] > 0 else "" for j in range(num_classes)] for i in range(num_classes)]
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.xlabel('Pred')
        plt.ylabel('True')
        plt.title(f'Fold {fold + 1} Confusion (%)')
        for i in range(num_classes):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=1.5))
        # outline highest-percentage cell per row (thicker)
        for i in range(num_classes):
            j = int(np.argmax(cm_pct[i]))
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2.5))
        plt.tight_layout()
        plt.savefig(RUN_DIR / f'fold{fold + 1}_confusion_{SCRIPT_NAME}.png')
        plt.close()

        summary["fold_metrics"].append({"fold": fold + 1, "best_acc": best_acc})

    summary["mean_acc"] = float(np.mean(fold_accs))
    summary["std_acc"] = float(np.std(fold_accs))
    # (JSON writing moved below after we gather additional statistics to keep one single source of truth)
    # with open(RUN_DIR / f"summary_{SCRIPT_NAME}.json", "w") as fp:
    #     json.dump(summary, fp, indent=2)

    # overall confusion
    cm_pct_tot = cm_total / cm_total.sum(axis=1, keepdims=True) * 100
    annot_tot = [[f"{cm_total[i, j]}/{cm_total.sum(axis=1)[i]}" if cm_total[i, j] > 0 else "" for j in range(num_classes)] for i in range(num_classes)]
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm_pct_tot, annot=annot_tot, fmt='', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('Overall Confusion (%)')
    for i in range(num_classes):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=1.5))
    # outline highest-percentage cell per row (thicker)
    for i in range(num_classes):
        j = int(np.argmax(cm_pct_tot[i]))
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2.5))
    plt.tight_layout()
    plt.savefig(RUN_DIR / f'overall_confusion_{SCRIPT_NAME}.png')
    plt.close()

    # ------------- textual report -------------
    chance = 100 / num_classes
    recalls = np.divide(cm_total.diagonal(), cm_total.sum(axis=1, keepdims=False),
                        out=np.zeros_like(cm_total.diagonal(), dtype=float), where=cm_total.sum(axis=1) != 0)
    best_idx = int(np.argmax(recalls))
    worst_idx = int(np.argmin(recalls))
    # ---- add the extra metrics collected above into the summary dict ----
    summary.update({
        "chance_level_pct": chance,
        "confusion_total": cm_total.tolist(),
        "per_digit_recall": [float(r) for r in recalls],
        "best_digit": le.classes_[best_idx],
        "worst_digit": le.classes_[worst_idx],
    })
    # Persist the enriched summary
    summ_path = RUN_DIR / f"summary_{SCRIPT_NAME}.json"
    with open(summ_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    # Reload to guarantee that the txt report is generated strictly from persisted JSON data
    with open(summ_path) as fp:
        summary_loaded = json.load(fp)

    report_lines = [
        f"{SCRIPT_NAME} report (enhanced augmentation)",
        f"Run ID        : {summary_loaded['run_id']}",
        "",
        "--- Hyper-parameters ---",
        f"  Batch size          : {summary_loaded['hyper']['batch_size']}",
        f"  Learning rate       : {summary_loaded['hyper']['lr']}",
        f"  Max epochs          : {summary_loaded['hyper']['epochs']}",
        f"  Early-stop patience : {summary_loaded['hyper']['early_stop']}",
        f"  Shift P / range     : {summary_loaded['hyper']['shift_p']} | {summary_loaded['hyper']['shift_min_frac']}–{summary_loaded['hyper']['shift_max_frac']} × T",
        f"  Scale P / range     : {summary_loaded['hyper']['scale_p']} | {summary_loaded['hyper']['scale_min']}-{summary_loaded['hyper']['scale_max']}",
        f"  Noise P / std       : {summary_loaded['hyper']['noise_p']} | {summary_loaded['hyper']['noise_std']}",
        "",
        "--- Results ---",
        f"Mean LOSO accuracy : {summary_loaded['mean_acc']:.2f}% (±{summary_loaded['std_acc']:.2f}%)",
        f"Chance level      : {summary_loaded['chance_level_pct']:.2f}% (6-class)",
        "",
        "Per-digit recall (TPR):"
    ]
    for i, cls in enumerate(le.classes_):
        report_lines.append(f"  {cls} : {recalls[i] * 100:.1f}%   | TP={cm_total[i, i]} / {cm_total.sum(axis=1)[i]}")
    report_lines.extend([
        "",
        f"Best recalled digit : {le.classes_[best_idx]} ({recalls[best_idx] * 100:.1f}%)",
        f"Worst recalled digit: {le.classes_[worst_idx]} ({recalls[worst_idx] * 100:.1f}%)",
        "",
        "Confusion matrix totals (rows = true label):",
        json.dumps(cm_total.tolist())
    ])
    report_path = RUN_DIR / f"report_{SCRIPT_NAME}.txt"
    with open(report_path, 'w') as rp:
        rp.write("\n".join(report_lines))
    print(f"Report written to {report_path}")

    # Append to expandable index CSV
    csv_f.close()
    append_run_index(summary_loaded, SCRIPT_NAME)