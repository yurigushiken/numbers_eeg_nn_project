import argparse, textwrap, sys, datetime, json, csv, yaml, math, os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.patches import Rectangle

# plotting libs
import seaborn as sns
import matplotlib.pyplot as plt

# project-relative imports

proj_root = Path(__file__).resolve().parents[2]
sys.path.append(str((proj_root / 'code').resolve()))

from vit_experiments.vit_dataset import SpectrogramDataset, SpecAugment  # type: ignore
from vit_experiments.models.eeg_vit import build_model  # type: ignore
from run_index import append_run_index  # existing util

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Optional speed-ups for NVIDIA RTX-class GPUs (safe defaults).
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True  # A&B table B
torch.backends.cudnn.allow_tf32 = True

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    'dataset_dir': 'data_spectrograms/landing_digit_cwt_128x128',
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 100,
    'early_stop': 15,
    'model_name': 'vit_tiny_patch16_128',
    'img_size': 128,
    'mixup_alpha': 0.0,
    'time_mask_p': 0.3,
    'time_mask_frac': 0.15,
    'freq_mask_p': 0.3,
    'freq_mask_frac': 0.15,
    'dropout': 0.1,
    'weight_decay': 1e-5,
    'betas': (0.9, 0.95),
    'max_folds': None,

    # -- regulariser knobs (timm) --
    'drop_path_rate': 0.1,
    'attn_drop_rate': 0.0,
    'token_drop_rate': 0.1,
    'qkv_bias': True,

    # -- loss & optimisation tweaks --
    'label_smoothing': 0.05,
    'grad_clip_norm': 1.0,
    'min_lr': 1e-6,
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        sys.exit(f'Config {p} not found')
    return yaml.safe_load(p.read_text()) or {}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""ViT landing-digit trainer (pre-generated spectrograms).
        Usage:
            python 02_train_vit_landing_digit.py --cfg configs/vit/base_vit.yaml
            python 02_train_vit_landing_digit.py --lr 5e-4 --batch_size 16
        """))
    p.add_argument('--cfg', type=str, help='YAML/JSON config path')
    p.add_argument('--set', nargs='*', metavar='KEY=VAL', help='Free-form overrides')
    return p.parse_args()


def resolve_cfg(base_yaml: str | None, overrides: list[str] | None) -> dict[str, Any]:
    cfg = DEFAULTS.copy()
    cfg.update(load_yaml(base_yaml))
    if overrides:
        for kv in overrides:
            if '=' not in kv:
                sys.exit(f'--set expects KEY=VAL pairs, got "{kv}"')
            k, v = kv.split('=', 1)
            cfg[k] = yaml.safe_load(v)
    return cfg

# ----------------------------------------------------------------------------
# Mixup helper
# ----------------------------------------------------------------------------

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device).item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# ----------------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------------

def train_epoch(model, loader, opt, loss_fn, aug: SpecAugment, mixup_alpha: float, num_classes: int, *, grad_clip_norm: float | None = None):
    model.train()
    tot_loss = 0.0
    first_batch = True
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        yb = yb.to(DEVICE, dtype=torch.long, non_blocking=True)
        xb = aug(xb)
        if mixup_alpha > 0:
            xb, y_a, y_b, lam = mixup(xb, yb, mixup_alpha)
        opt.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(xb)
            if mixup_alpha > 0:
                # one-hot encode then mix → soft label matrix [N,C]
                y_a_oh = F.one_hot(y_a, num_classes=num_classes).float()
                y_b_oh = F.one_hot(y_b, num_classes=num_classes).float()
                y_mix = lam * y_a_oh + (1 - lam) * y_b_oh
                loss = loss_fn(out, y_mix)
            else:
                if first_batch:
                    print('DEBUG', out.shape, yb.shape, yb.dtype, flush=True)
                    first_batch = False
                loss = loss_fn(out, yb)
        loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        opt.step()
        tot_loss += loss.item()
    return tot_loss / len(loader)


def eval_epoch(model, loader, loss_fn, return_preds: bool = False):
    model.eval()
    tot_loss = 0.0
    correct = 0
    N = 0
    y_true_fold = []
    y_pred_fold = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(xb)
                pred = out.argmax(1)
                pred_cpu = pred.cpu()
                tot_loss += loss_fn(out, yb).item()
                correct += (pred_cpu == yb.cpu()).sum().item()
                N += yb.size(0)
                y_true_fold.extend(yb.cpu().numpy())
                y_pred_fold.extend(pred_cpu.numpy())
    if return_preds:
        return tot_loss / max(1, len(loader)), 100.0 * correct / max(1, N), y_true_fold, y_pred_fold
    return tot_loss / max(1, len(loader)), 100.0 * correct / max(1, N)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = resolve_cfg(args.cfg, args.set)

    run_id = datetime.datetime.now().strftime(f"%Y%m%d_%H%M_02_train_vit_landing_digit")
    run_dir = Path('results') / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    ds = SpectrogramDataset(cfg['dataset_dir'])
    groups = ds.meta['subject'].values

    # ------------------------------------------------------------------
    # Derive the true number of classes from the dataset (should be 6 for
    # this study) instead of assuming 10.  The SpectrogramDataset now
    # guarantees that labels are contiguous 0-based indices.
    # ------------------------------------------------------------------

    n_classes = len(ds.classes_)

    logo = LeaveOneGroupOut()

    val_accs = []
    overall_y_true = []
    overall_y_pred = []
    for fold, (tr_idx, va_idx) in enumerate(logo.split(np.zeros(len(ds)), np.zeros(len(ds)), groups)):
        if cfg.get('max_folds') and fold >= cfg['max_folds']:
            break
        tr_ld = DataLoader(
            Subset(ds, tr_idx),
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=False,
        )
        va_ld = DataLoader(
            Subset(ds, va_idx),
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        # class weights from training subset
        y_train = ds.meta.iloc[tr_idx]['_y'].to_numpy()
        present = np.unique(y_train)

        # ------------------------------------------------------------------
        # ALWAYS apply class-balancing – even when some classes are absent in
        # the current training fold (which is frequent with LOSO + imbalanced
        # data).  For every missing class we assign weight = 0 so it does not
        # influence the loss, yet the tensor keeps a fixed length = n_classes.
        # ------------------------------------------------------------------

        cls_weights = compute_class_weight(
            class_weight = 'balanced',
            classes      = present,     # only classes that actually appear
            y            = y_train
        )

        # Build a full-length weight vector, fill absentees with zeros.
        weight_arr = np.zeros(n_classes, dtype=np.float32)
        for cls_idx, w in zip(present, cls_weights):
            weight_arr[int(cls_idx)] = w

        weight_tensor = torch.tensor(weight_arr, dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=float(cfg['label_smoothing']))

        model = build_model(
            cfg['model_name'],
            n_classes,
            img_size=cfg['img_size'],
            in_chans=128,
            domain_adapter=True,
            adapter_dim=384,
            drop_path_rate=cfg['drop_path_rate'],
            attn_drop_rate=cfg['attn_drop_rate'],
            drop_rate=cfg['token_drop_rate'],
            qkv_bias=cfg['qkv_bias'],
        ).to(DEVICE, memory_format=torch.channels_last)  # table C (channels-last)

        # Grad-checkpointing to save VRAM (table E)
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)

        # ---------------------------------------------------------
        # Safe torch.compile() – only enable when a working Triton
        # backend is present AND we are not on Windows. We attempt
        # a tiny CUDA TriL op; if that fails we stay in eager mode.
        # ---------------------------------------------------------

        def _torch_compile_safe(m: nn.Module):
            if not (hasattr(torch, 'compile') and torch.cuda.is_available() and os.name != 'nt'):
                return m
            try:
                # this op calls a Triton kernel under Inductor
                torch.empty(8, 8, device='cuda').tril()
                return torch.compile(m, mode='reduce-overhead', fullgraph=False)
            except Exception as e:
                print(f"[INFO] Skip torch.compile – {e}")
                return m

        model = _torch_compile_safe(model)

        if cfg.get('dropout'):
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = cfg['dropout']

        raw_betas = cfg.get('betas', (0.9, 0.999))
        if isinstance(raw_betas, (float, int)):
            betas = (0.9, float(raw_betas))
        else:
            betas = tuple(raw_betas)
        opt = optim.AdamW(
            model.parameters(),
            lr=float(cfg['lr']),
            weight_decay=float(cfg['weight_decay']),
            betas=betas,
        )
        sched = CosineAnnealingLR(
            opt,
            T_max=cfg['epochs'],
            eta_min=float(cfg.get('min_lr', 0.0)),
        )
        aug = SpecAugment(cfg['time_mask_p'], cfg['time_mask_frac'], cfg['freq_mask_p'], cfg['freq_mask_frac'])

        best_acc = 0.0
        patience = 0
        epoch_train_loss = []
        epoch_val_acc = []
        fold_y_true = []
        fold_y_pred = []
        for epoch in range(1, cfg['epochs'] + 1):
            tr_loss = train_epoch(model, tr_ld, opt, loss_fn, aug, cfg['mixup_alpha'], n_classes, grad_clip_norm=cfg.get('grad_clip_norm'))
            va_loss, va_acc, y_true_fold, y_pred_fold = eval_epoch(model, va_ld, loss_fn, return_preds=True)
            epoch_train_loss.append(tr_loss)
            epoch_val_acc.append(va_acc)
            sched.step()
            if va_acc > best_acc:
                best_acc = va_acc
                patience = 0
            else:
                patience += 1
            print(f"Fold {fold+1}  Ep{epoch:03d}  tr {tr_loss:.3f}  val {va_loss:.3f}  acc {va_acc:.2f}%")
            if patience >= cfg['early_stop']:
                break
        if not math.isfinite(best_acc):
            raise RuntimeError('Validation accuracy is NaN/inf – aborting trial.')
        val_accs.append(best_acc)

        # confusion matrix for this fold
        from sklearn.metrics import confusion_matrix
        model.eval()
        y_true_f, y_pred_f = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                out = model(xb.to(DEVICE))
                pred = out.argmax(1).cpu()
                y_true_f.extend(yb.cpu().numpy())
                y_pred_f.extend(pred.numpy())
        cm_counts = confusion_matrix(y_true_f, y_pred_f, labels=list(range(n_classes)))
        # row-wise percentages so every row sums to 100
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums!=0) * 100

        fig = plt.figure(figsize=(6,5))
        ax = plt.gca()
        digit_labels = [str(d) for d in ds.classes_]
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=100,
                    xticklabels=digit_labels, yticklabels=digit_labels, square=True, cbar_kws={'label':'%'} )
        # Highlight diag in black & largest off-diag in red
        for i in range(n_classes):
            y_idx = n_classes - 1 - i  # because seaborn.heatmap inverts y-axis
            ax.add_patch(Rectangle((i, y_idx), 1, 1, fill=False, edgecolor='black', lw=2))
            # red rectangle on row-wise worst error (highest off-diag)
            row_vals = cm[i].copy(); row_vals[i] = 0
            if row_vals.max() > 0:
                j = int(row_vals.argmax())
                ax.add_patch(Rectangle((j, y_idx), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.title(f'Fold {fold+1} Confusion')
        plt.ylabel('True'); plt.xlabel('Pred')
        fig_path = run_dir / f'fold{fold+1}_confusion.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close(fig)

        # accumulate for overall
        overall_y_true.extend(y_true_f)
        overall_y_pred.extend(y_pred_f)

        # loss/acc curve plot for this fold
        fig2 = plt.figure();
        plt.plot(epoch_train_loss, label='train_loss');
        plt.twinx(); plt.plot(epoch_val_acc, color='green', label='val_acc');
        plt.title(f'Fold {fold+1} Curves');
        fig2.savefig(run_dir / f'fold{fold+1}_curves.png'); plt.close(fig2)

        print(f"Fold {fold+1} best {best_acc:.2f}%")

    mean_acc = float(np.mean(val_accs))
    std_acc = float(np.std(val_accs))

    # overall confusion
    overall_counts = confusion_matrix(overall_y_true, overall_y_pred, labels=list(range(n_classes)))
    row_sums = overall_counts.sum(axis=1, keepdims=True)
    overall_cm = np.divide(overall_counts, row_sums, out=np.zeros_like(overall_counts, dtype=float), where=row_sums!=0) * 100

    fig = plt.figure(figsize=(6,5))
    ax = plt.gca()
    digit_labels = [str(d) for d in ds.classes_]
    sns.heatmap(overall_cm, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=100,
                xticklabels=digit_labels, yticklabels=digit_labels, square=True, cbar_kws={'label':'%'} )
    for i in range(n_classes):
        y_idx = n_classes - 1 - i
        ax.add_patch(Rectangle((i, y_idx), 1, 1, fill=False, edgecolor='black', lw=2))
        row_vals = overall_cm[i].copy(); row_vals[i] = 0
        if row_vals.max() > 0:
            j = int(row_vals.argmax())
            ax.add_patch(Rectangle((j, y_idx), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.title('Overall Confusion')
    plt.ylabel('True'); plt.xlabel('Pred')
    overall_cm_fp = run_dir / 'overall_confusion.png'
    fig.savefig(overall_cm_fp, dpi=300, bbox_inches='tight'); plt.close(fig)

    # Final summary for JSON & report
    summary = {
        'run_id': run_id,
        'script': Path(__file__).name,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'hyper': {k: v for k, v in cfg.items() if k not in ['optuna_trial_id', 'optuna_params']},
        'overall_confusion_matrix_data': overall_cm.tolist(),
        'artefacts': {
            'overall_confusion_matrix_png': str(overall_cm_fp.relative_to(Path('results'))),
            'fold_confusion_matrices_png': [],
            'fold_curve_plots_png': [],
        },
        'per_fold_details': []
    }

    # Update per-fold details and artifact paths in summary
    for fold, acc in enumerate(val_accs):
        summary['per_fold_details'].append({
            'fold': fold + 1,
            'best_val_acc': acc
        })
        summary['artefacts']['fold_confusion_matrices_png'].append(str((run_dir / f'fold{fold+1}_confusion.png').relative_to(Path('results'))))
        summary['artefacts']['fold_curve_plots_png'].append(str((run_dir / f'fold{fold+1}_curves.png').relative_to(Path('results'))))

    # write summary json
    (run_dir / 'summary_02_train_vit_landing_digit.json').write_text(json.dumps(summary, indent=2))

    # Append to global index CSV
    append_run_index(summary, Path(__file__).name)

    # ------------------------------------------------------------------
    # Human-readable TXT report – aimed at EEG researchers first, coders
    # second (they can consult the JSON for every detail)
    # ------------------------------------------------------------------
    cmd_line = 'python ' + ' '.join(sys.argv)

    # List only the most relevant hyper-parameters to keep the report short
    key_hypers = [
        'dataset_dir', 'batch_size', 'lr', 'epochs', 'model_name', 'img_size',
        'mixup_alpha', 'dropout', 'weight_decay', 'drop_path_rate'
    ]
    hyper_short = {k: v for k, v in cfg.items() if k in key_hypers}

    lines: list[str] = []
    lines.append(f"EEG ViT Landing-Digit Decoder  —  Run {run_id}\n")
    lines.append("Command:")
    lines.append(f"  {cmd_line}\n")

    lines.append("Performance (Leave-One-Subject-Out, % accuracy):")
    lines.append(f"  Mean = {mean_acc:.2f}   Std = {std_acc:.2f}")
    per_fold_txt = ', '.join([f"{d['fold']}:{d['best_val_acc']:.2f}" for d in summary['per_fold_details']])
    lines.append(f"  Per-fold best = {per_fold_txt}\n")

    lines.append("Key Hyper-parameters:")
    for k, v in hyper_short.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("Artifacts:")
    lines.append(f"  Overall confusion: {summary['artefacts']['overall_confusion_matrix_png']}")
    for p in summary['artefacts']['fold_confusion_matrices_png']:
        lines.append(f"  Fold confusion:   {p}")
    for p in summary['artefacts']['fold_curve_plots_png']:
        lines.append(f"  Training curves:  {p}")
    lines.append("")

    lines.append("Note: Full machine-readable details in summary_02_train_vit_landing_digit.json")

    report_path = run_dir / 'report_02_train_vit_landing_digit.txt'
    report_path.write_text('\n'.join(lines))

    print(f"Mean LOSO accuracy: {mean_acc:.2f}%")


if __name__ == '__main__':
    main() 