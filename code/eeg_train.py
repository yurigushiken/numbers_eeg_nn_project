"""Shared EEGNet training engine (Phase-2).

Usage from a task wrapper:

    import eeg_train as et
    from pathlib import Path

    def label_fn(meta):
        # example binary mapping
        return meta['direction'].map({'I': 0, 'D': 1})

    if __name__ == '__main__':
        cfg = et.resolve_cfg(Path(__file__).with_suffix('.yaml'))  # base YAML path
        et.run_loso(cfg, label_fn)

The engine handles:
    * config merging (DEFAULTS → base YAML → CLI flags → --set overrides)
    * data loading and tensor building (cached)
    * augmentation pipeline
    * LOSO cross-validation with early stopping
    * confusion-matrix plotting (with black diag + red max cell)
    * summary JSON, TXT report, runs_index.csv update

Only task-specific pieces are:
    * label_fn – derives target labels from epochs.metadata
    * OPTIONAL task-specific default overrides
"""
from __future__ import annotations

import argparse, json, sys, yaml, textwrap, datetime, csv, re
from pathlib import Path
from typing import Callable, Dict, Any, Tuple, Optional

import numpy as np
import mne
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# unified plotting helpers
from utils.plots import plot_confusion, plot_curves

# Plot aesthetics similar to ViT pipeline
sns.set(style="white", font="DejaVu Sans", font_scale=1.0)

from matplotlib.patches import Rectangle

# -----------------------------------------------------------------------------
# TorchEEG base models (EEGNet, etc.). TorchEEG 2.x may optionally depend on
# heavy GNN stacks that require extra libs (DGL).  To avoid hard crashes when
# these optional dependencies are missing, guard the import and fall back to
# a soft-fail (EEGNet=None).  Tasks that need EEGNet will raise later with a
# clear message; other tasks such as the pretrained TSCNN branch continue to
# run.
# -----------------------------------------------------------------------------
try:
    from torcheeg.models import EEGNet  # noqa: F401
except Exception as _eegnet_err:  # pragma: no cover – depends on user env
    print("[TORCHEEG_EEGNet_IMPORT_ERROR]", _eegnet_err)
    EEGNet = None

# Optional: CwA-Transformer (channel-wise CNN encoder + Transformer)
try:
    from models.cwa_transformer import CwaTransformer  # local minimal impl
except Exception:
    CwaTransformer = None

# NEW: TorchEEG pre-trained encoders (TSCNN, etc.)
try:
    from torcheeg.models import TSCNNPretrain
except Exception:
    TSCNNPretrain = None

# (Braindecode support removed as of 2025-07) – legacy imports deleted per user request.
bd_get_model = None
from torcheeg import transforms
from run_index import append_run_index

from torch.cuda.amp import GradScaler, autocast
USE_AMP = torch.cuda.is_available()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Canonical default hyper-parameters (single source of truth) ---
# Any parameter used by eeg_train should be here, to guarantee it has a default.
# These are the overall sane defaults for any EEGNet task.
CANONICAL_DEFAULTS = {
    "dataset_dir": str(Path("data_preprocessed/acc_1_dataset")),
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 100,
    "early_stop": 15,
    "max_folds": None,
    "shift_p": 0.5,
    "shift_min_frac": 0.005,
    "shift_max_frac": 0.04,
    "scale_p": 0.5,
    "scale_min": 0.9,
    "scale_max": 1.1,
    "noise_p": 0.3,
    "noise_std": 0.02,
    "mixup_alpha": 0.0,
    "time_mask_p": 0.0,
    "time_mask_frac": 0.15,
    "chan_mask_p": 0.0,
    "chan_mask_ratio": 0.10,
    "channel_dropout_p": 0.0, # for compatibility with optuna_tune
    # --- new defaults ---
    "auto_lr": False,        # enable linear LR scaling w.r.t batch size
    "enc_stride": 2          # default temporal down-sampling factor for CwA-T
}

def resolve_cfg(base_yaml: Path, task_defaults: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Resolve configuration from canonical defaults, base YAML, and CLI overrides."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""Generic EEGNet decoder wrapper produced by eeg_train.py""")
    )
    p.add_argument('--cfg', default=str(base_yaml), help='Path to base YAML file')
    p.add_argument('--set', nargs='*', metavar='KEY=VAL', help='Arbitrary param overrides, e.g. --set lr=0.0005 scale_min=0.8')
    args, unknown = p.parse_known_args()

    # Layer 1: Canonical defaults
    cfg = CANONICAL_DEFAULTS.copy()
    # Layer 2: Task-specific defaults (from wrapper, e.g., different max_folds, etc.)
    if task_defaults:
        cfg.update(task_defaults)
    # Layer 3: Base YAML (model/task-specific recommended defaults)
    cfg.update(load_yaml(args.cfg))
    # Layer 4: CLI --set overrides (highest priority)
    if args.set:
        for kv in args.set:
            if '=' not in kv:
                sys.exit(f"--set expects KEY=VAL pairs, got {kv}")
            k, v = kv.split('=', 1)
            cfg[k] = yaml.safe_load(v) # Safely load to infer type (int, float, bool, None)
    return cfg

# -------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------

def load_yaml(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    fp = Path(path)
    if not fp.exists():
        sys.exit(f'Config {path} missing')
    txt = fp.read_text()
    return yaml.safe_load(txt) or {}

_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}

def load_dataset(root: Path, label_fn: Callable[[Any], Any]) -> Tuple[TensorDataset, np.ndarray, list[str]]:
    """Load/ cache tensors."""
    key = f'{root.resolve()}::{label_fn.__name__}'
    if key in _CACHE:
        X_t, y_t, groups, classes = _CACHE[key]
        return TensorDataset(X_t, y_t), groups, classes

    files = sorted(root.glob('sub-*preprocessed-epo.fif'))
    sid_re = re.compile(r'sub-(\d+)_preprocessed')
    epochs_list = []
    for fp in files:
        ep = mne.read_epochs(fp, preload=True, verbose=False)
        sid = int(sid_re.search(fp.name).group(1))
        ep.metadata['subject'] = sid
        epochs_list.append(ep)
    all_ep = mne.concatenate_epochs(epochs_list)
    # derive label column
    all_ep.metadata['__y'] = label_fn(all_ep.metadata)
    # drop rows with NaN labels (tasks that discard certain trials)
    all_ep = all_ep[~all_ep.metadata['__y'].isna()]

    X = all_ep.get_data() * 1e6

    # ------------------------------------------------------------------
    # QUICK FIX: Some recordings include an additional reference channel
    # (e.g., Cz) leading to 129 channels on nominal 128-ch HydroCel caps.
    # Drop any surplus channels beyond the first 128 so that the model
    # dimension (C) is always even, preventing positional-encoding shape
    # errors in the Hybrid Transformer.
    # ------------------------------------------------------------------
    if X.shape[1] > 128:
        X = X[:, :128, :]

    X = transforms.MeanStdNormalize(axis=-1)(eeg=X)['eeg']
    X_t = torch.from_numpy(X).float().unsqueeze(1)
    le = LabelEncoder(); y_t = torch.from_numpy(le.fit_transform(all_ep.metadata['__y'])).long()
    groups = all_ep.metadata['subject'].values
    classes = list(le.classes_)
    _CACHE[key] = (X_t, y_t, groups, classes)
    return TensorDataset(X_t, y_t), groups, classes

# -------------------------------------------------------------
# Augmentation pipeline (GPU-friendly)
# -------------------------------------------------------------
class ComposeT:
    def __init__(self, ts): self.ts = ts
    def __call__(self, x):
        for t in self.ts: x = t(x)
        return x
class RTShift:
    def __init__(self, p, mn, mx): self.p, self.mn, self.mx = p, mn, mx
    def __call__(self,x):
        if self.mx and torch.rand(1,device=x.device) < self.p:
            sh = torch.randint(self.mn, self.mx+1,(1,),device=x.device).item()
            if torch.rand(1,device=x.device)<0.5: sh=-sh
            return torch.roll(x,shifts=sh,dims=-1)
        return x
class RScale:
    def __init__(self, p: float, lo: float, hi: float):
        self.p = p
        self.lo = lo
        self.hi = hi
    def __call__(self,x):
        if torch.rand(1,device=x.device) < self.p:
            return x*torch.empty(1,device=x.device).uniform_(self.lo,self.hi)
        return x
class RNoise:
    def __init__(self,p,std): self.p,self.std=p,std
    def __call__(self,x):
        if self.std>0 and torch.rand(1,device=x.device)<self.p:
            return x + self.std*torch.randn_like(x)
        return x
class RTMask:
    def __init__(self,p,frac): self.p,self.frac=p,frac
    def __call__(self,x):
        if self.p==0 or torch.rand(1,device=x.device)>=self.p: return x
        T=x.shape[-1]; m=max(1,int(self.frac*T)); st=torch.randint(0,T-m+1,(1,),device=x.device).item(); x=x.clone(); x[...,st:st+m]=0; return x
class RCMask:
    def __init__(self,p,ratio): self.p,self.r= p, ratio
    def __call__(self,x):
        if self.p==0 or torch.rand(1,device=x.device)>=self.p: return x
        C=x.shape[-2]; k=max(1,int(self.r*C)); idx=torch.randperm(C,device=x.device)[:k]; x=x.clone(); x[...,idx,:]=0; return x

def make_aug(cfg: Dict[str, Any], T: int):
    return ComposeT([
        RTShift(cfg['shift_p'], max(1,int(cfg['shift_min_frac']*T)), int(cfg['shift_max_frac']*T)),
        RScale(cfg['scale_p'], cfg['scale_min'], cfg['scale_max']),
        RNoise(cfg['noise_p'], cfg['noise_std']),
        RTMask(cfg['time_mask_p'], cfg['time_mask_frac']),
        RCMask(cfg['chan_mask_p'], cfg['chan_mask_ratio'])
    ])

def mixup(x, y, alpha):
    if alpha<=0: return x, y, y, 1.0
    lam=torch.distributions.Beta(alpha,alpha).sample().to(x.device).item(); idx=torch.randperm(x.size(0),device=x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

# -------------------------------------------------------------
# Core training routine
# -------------------------------------------------------------

def run_loso(cfg: Dict[str, Any], label_fn: Callable[[Any], Any], trial: Optional['optuna.Trial']=None) -> float:
    # Optional run directory where artefacts are written
    run_dir: Optional[Path] = None
    if 'run_dir' in cfg and cfg['run_dir']:
        run_dir = Path(cfg['run_dir'])
        run_dir.mkdir(parents=True, exist_ok=True)
    # --- ensure numeric types are correct (yaml or cli may leave str) ---
    numeric_int_keys = ['batch_size', 'epochs', 'early_stop', 'max_folds', 'enc_stride']
    numeric_float_keys = ['lr', 'noise_std', 'mixup_alpha', 'time_mask_p', 'chan_mask_p',
                          'shift_p', 'scale_p', 'channel_dropout_p', 'shift_min_frac', 'shift_max_frac',
                          'scale_min', 'scale_max', 'noise_p']
    for k in numeric_int_keys:
        if k in cfg and cfg[k] is not None and not isinstance(cfg[k], int):
            cfg[k] = int(cfg[k])
    for k in numeric_float_keys:
        if k in cfg and cfg[k] is not None and not isinstance(cfg[k], float):
            cfg[k] = float(cfg[k])

    # -----------------------------------------------------
    # Optional linear LR scaling when batch size changes
    # -----------------------------------------------------
    if cfg.get("auto_lr"):
        ref_bs = 64
        if cfg["batch_size"] != ref_bs:
            cfg["lr"] = cfg["lr"] * (cfg["batch_size"] / ref_bs)

    root = Path(cfg['dataset_dir'])

    scaler = GradScaler(enabled=USE_AMP)
    ds, groups, class_names = load_dataset(root, label_fn)
    num_cls = len(class_names)

    # -----------------------------------------------------
    # Compute balanced class weights ONCE over the full data
    # -----------------------------------------------------
    _full_cls_w = compute_class_weight('balanced', classes=np.arange(num_cls), y=ds.tensors[1].numpy())
    cls_w_t = torch.tensor(_full_cls_w, dtype=torch.float32).to(DEVICE)

    log_accs = []
    overall_y_true, overall_y_pred = [], []
    if cfg.get('n_folds'):
        k = int(cfg['n_folds'])
        gss = GroupShuffleSplit(n_splits=k, test_size=1.0/k,
                                random_state=(trial.number if trial is not None else None))
        fold_iter = gss.split(np.zeros(len(ds)), np.zeros(len(ds)), groups)
    else:
        fold_iter = LeaveOneGroupOut().split(np.zeros(len(ds)), np.zeros(len(ds)), groups)

    for fold, (tr_idx, va_idx) in enumerate(fold_iter):
        if cfg.get('max_folds') and fold >= cfg['max_folds']:
            break
        va_subject = groups[va_idx][0]
        tr_ld = DataLoader(Subset(ds, tr_idx), batch_size=cfg['batch_size'], shuffle=True, drop_last=False)
        va_ld = DataLoader(Subset(ds, va_idx), batch_size=cfg['batch_size'])

        # loss function (shared weights)
        loss_fn = nn.CrossEntropyLoss(cls_w_t)

        # ------------------------------------------------------------------
        # Model instantiation – EEGNet (default) OR the new CwA-Transformer
        # ------------------------------------------------------------------
        _, _, C, T = ds.tensors[0].shape

        # Existing selection for model
        model_name = cfg.get("model_name", "eegnet").lower()
        if model_name == "cwat":
            if CwaTransformer is None:
                raise ImportError(
                    "CwaTransformer model not found. Please install torcheeg>=1.6.0rc2 "
                    "or ensure it is importable."
                )

            model = CwaTransformer(
                in_channels=C,
                enc_kernel_size=cfg.get("enc_kernel", 7),
                enc_latent_dim=cfg.get("latent_dim", 64),
                num_heads=cfg.get("n_heads", 4),
                num_layers=cfg.get("depth", 4),
                dropout=cfg.get("dropout", 0.1),
                num_classes=num_cls,
                enc_stride=cfg.get("enc_stride", 2),   # <-- new arg
            ).to(DEVICE)
        # 'deep4_pretrained' branch removed (Braindecode deprecated)
        elif model_name == "tscnn_pretrained":
            if TSCNNPretrain is None:
                raise ImportError("TorchEEG>=2.2 is required for TSCNNPretrain. Activate torcheeg-env and run: pip install -U 'torcheeg>=2.2'")
            # ------------------------------------------------------------------
            # Load BYOL-E (or other) checkpoint from the HF hub.  The subfolder
            # can be overridden via cfg['pretrain_ckpt'] (default: tscnn/byol_e).
            # ------------------------------------------------------------------
            backbone = TSCNNPretrain.from_hf_hub(
                cfg.get("pretrain_repo", "torcheeg/pretrain-zoo"),
                subfolder=cfg.get("pretrain_ckpt", "tscnn/byol_e"),
                freeze_stem=cfg.get("freeze_stem", False)
            )
            # Determine feature dimension in a robust way                                       
            feat_dim = getattr(backbone, "out_features", None) or \
                       getattr(backbone, "feature_dim", None)
            if feat_dim is None:
                # Fallback: do a dummy forward pass
                with torch.no_grad():
                    dummy = torch.zeros(1, C, T)  # (B, C, T)
                    feat_dim = backbone(dummy).shape[-1]
            classifier = nn.Linear(feat_dim, num_cls)
            model = nn.Sequential(backbone, classifier).to(DEVICE)
            # Optionally freeze encoder parameters for first few epochs
            if cfg.get("freeze_backbone", False):
                for p in backbone.parameters():
                    p.requires_grad_(False)
        # 'bendr' branch removed – Bendr support deprecated.
        else:
            # Default back to EEGNet (legacy pipeline)
            if EEGNet is None:
                raise ImportError(
                    "EEGNet model not found. Please install torcheeg>=1.6.0rc2 "
                    "or ensure it is importable."
                )
            model = EEGNet(
                num_classes=num_cls,
                num_electrodes=C,
                chunk_size=T,
                dropout=0.5,
            ).to(DEVICE)
        wd = float(cfg.get('weight_decay', 0.0) or 0.0)
        opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=wd)
        sched = ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
        aug = make_aug(cfg, T)

        best_val = float('inf'); best_acc = 0; best_epoch=1; patience = 0
        tr_history, val_history, acc_history = [], [], []
        for epoch in range(1, cfg['epochs']+1):
            # train
            model.train(); tot=0
            for xb,yb in tr_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                xb = aug(xb)
                xb, y_a, y_b, lam = mixup(xb, yb, cfg['mixup_alpha'])

                # Models that operate on (B, C, T) tensors rather than (B, 1, C, T)
                if model_name in {"cwat", "tscnn_pretrained"}:
                    x_device = xb.to(DEVICE)
                    if model_name in {"cwat", "tscnn_pretrained"}:
                        x_in = x_device.squeeze(1)
                    else:
                        x_in = x_device
                else:
                    x_in = xb

                opt.zero_grad()
                with autocast(enabled=USE_AMP):
                    out = model(x_in)
                # compute loss in fp32 to avoid dtype mismatch with class weights
                loss = lam*loss_fn(out.float(),y_a)+(1-lam)*loss_fn(out.float(),y_b) if cfg['mixup_alpha']>0 else loss_fn(out.float(),yb)
                if USE_AMP:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                tot += loss.item()
            tr_loss = tot/len(tr_ld)
            # val
            model.eval(); val_loss=0; corr=0; N=0
            y_true_fold, y_pred_fold = [], []
            with torch.no_grad():
                for xb,yb in va_ld:
                    xb_device = xb.to(DEVICE)
                    if model_name in {"cwat", "tscnn_pretrained"}:
                        xv = xb_device.squeeze(1)
                    else:
                        xv = xb_device
                    with autocast(enabled=USE_AMP):
                        out = model(xv)
                    val_loss += loss_fn(out.float(), yb.to(DEVICE)).item()
                    pred = out.argmax(1).cpu()
                    corr += (pred == yb).sum().item(); N += yb.size(0)
                    y_true_fold.extend(yb.numpy()); y_pred_fold.extend(pred.numpy())
            val_loss/=len(va_ld); val_acc=100*corr/N; sched.step(val_loss)
            # accumulate predictions for overall confusion
            overall_y_true.extend(y_true_fold)
            overall_y_pred.extend(y_pred_fold)
            tr_history.append(tr_loss); val_history.append(val_loss); acc_history.append(val_acc)
            if val_loss < best_val:
                best_val, best_acc, patience = val_loss, val_acc, 0
                best_epoch = epoch
                # Optional checkpoint saving (disabled by default)
                if run_dir is not None and cfg.get("save_ckpt", False):
                    ckpt_path = run_dir / f"fold_{fold:02d}_best.ckpt"
                    torch.save(model.state_dict(), ckpt_path)
            else:
                patience += 1
            if patience>=cfg['early_stop']: break

            # --- progress print ---
            print(f"Fold {fold+1} | Ep {epoch:03d} | Tr {tr_loss:.3f} | Val {val_loss:.3f} | Acc {val_acc:.2f}%", flush=True)
        log_accs.append(best_acc)

        # --- concise fold summary to console ---
        epochs_run = len(tr_history)
        early_tag = " (early-stop)" if epochs_run < cfg['epochs'] else ""
        print(f"── Fold {fold+1:02d} finished | best {best_acc:5.2f}% | epochs {epochs_run:3d}{early_tag}")

        # -----------------------------------------------------------
        # Save per-fold confusion matrix if run_dir is provided
        # -----------------------------------------------------------
        if run_dir is not None:
            # -----------------------------------------
            # Prepare optional hyper-param annotations
            # -----------------------------------------
            hyper_lines = cfg.get("plot_hyper_lines")
            if hyper_lines is None and model_name == "cwat":
                hyper_lines = [
                    f"batch: {cfg.get('batch_size')}",
                    f"lr: {cfg.get('lr'):.1e}",
                    f"depth: {cfg.get('depth')}",
                    f"latent: {cfg.get('latent_dim')}",
                    f"drop: {cfg.get('dropout') if cfg.get('dropout') is not None else 'n/a'}",
                    f"ker: {cfg.get('enc_kernel')}",
                    f"stride: {cfg.get('enc_stride')}",
                    f"mixup: {cfg.get('mixup_alpha'):.3f}",
                    f"wd: {cfg.get('weight_decay'):.1e}" if cfg.get('weight_decay') is not None else "wd: n/a",
                ]

            # Confusion matrix plot
            plot_confusion(
                y_true_fold,
                y_pred_fold,
                class_names,
                run_dir / f"fold{fold+1}_confusion.png",
                title=f"Fold {fold+1}  ·  Acc {best_acc:.1f}%  (ep {best_epoch}/{len(tr_history)})\nLeft-out subject {va_subject}",
                hyper_lines=hyper_lines,
            )

            # Training curves plot
            plot_curves(
                tr_history,
                val_history,
                acc_history,
                run_dir / f"fold{fold+1}_curves.png",
                hyper_lines=hyper_lines,
                lock_acc_axis=(model_name == "cwat"),
                title=f"Fold {fold+1}  ·  Acc {best_acc:.1f}%  (ep {best_epoch}/{len(tr_history)})",
            )
        if trial is not None:
            trial.report(np.mean(log_accs), fold)
            if trial.should_prune():
                raise __import__('optuna').exceptions.TrialPruned()

    # ---------------- overall confusion -----------------
    if run_dir is not None and overall_y_true:
        macro_f1 = f1_score(overall_y_true, overall_y_pred, average='macro') * 100
        plot_confusion(
            overall_y_true,
            overall_y_pred,
            class_names,
            run_dir / "overall_confusion.png",
            title=f"Overall  ·  Mean {np.mean(log_accs):.1f}%  ·  Macro-F1 {macro_f1:.1f}%",
            hyper_lines=cfg.get("plot_hyper_lines"),
        )

    return float(np.mean(log_accs)) 