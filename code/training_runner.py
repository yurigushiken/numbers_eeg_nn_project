"""
Unified Training Runner for all engines.

This module provides a generic, reusable training framework that handles the
common logic for all experiments, including:
- Leave-One-Subject-Out (LOSO) cross-validation
- Per-epoch training and validation loops
- Early stopping
- Model checkpointing
- Results collection (metrics, predictions)
- Plotting and summary generation

Engines are responsible for providing the components specific to their task:
- The data loader
- The model architecture
- Data augmentations
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Any, Tuple, Optional, List
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    SDP_IS_AVAILABLE = True
except ImportError:
    SDP_IS_AVAILABLE = False
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from utils.plots import plot_confusion, plot_curves

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


class TrainingRunner:
    """
    A generic runner to handle the training and evaluation loop for any model/dataset.
    """

    def __init__(self, cfg: Dict[str, Any], label_fn: Callable):
        self.cfg = cfg
        self.label_fn = label_fn
        self.run_dir: Optional[Path] = None
        if "run_dir" in cfg and cfg["run_dir"]:
            self.run_dir = Path(cfg["run_dir"])
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self._numeric_cast()

    def _numeric_cast(self):
        """Ensure numeric cfg values have correct dtype."""
        # This can be expanded as needed
        int_keys = ["batch_size", "epochs", "early_stop", "max_folds", "enc_stride"]
        for k in int_keys:
            if k in self.cfg and self.cfg[k] is not None and not isinstance(self.cfg[k], int):
                self.cfg[k] = int(self.cfg[k])

        float_keys = ['lr', 'weight_decay', 'noise_std', 'mixup_alpha', 'time_mask_p',
                      'chan_mask_p', 'shift_p', 'scale_p', 'channel_dropout_p',
                      'shift_min_frac', 'shift_max_frac', 'scale_min', 'scale_max',
                      'noise_p', 'freq_mask_p', 'freq_mask_frac']
        for k in float_keys:
            if k in self.cfg and self.cfg[k] is not None and not isinstance(self.cfg[k], float):
                self.cfg[k] = float(self.cfg[k])

    def get_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Creates optimizer and scheduler from config."""
        lr = float(self.cfg.get("lr", 1e-4))
        wd = float(self.cfg.get("weight_decay", 0.0) or 0.0)
        
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        scheduler_name = self.cfg.get("scheduler", "plateau").lower()
        epochs = int(self.cfg.get("epochs", 100))
        
        if scheduler_name == "cosine":
            sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        else: # Default to ReduceLROnPlateau
            sched = ReduceLROnPlateau(opt, "min", patience=self.cfg.get("scheduler_patience", 5), factor=0.5)

        return opt, sched

    def run(self,
            dataset: Dataset,
            groups: np.ndarray,
            class_names: List[str],
            model_builder: Callable[[Dict, int], nn.Module],
            aug_builder: Callable[[Dict], nn.Module],
            input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        
        num_cls = len(class_names)
        scaler = GradScaler(enabled=USE_AMP)

        # Compute balanced class weights ONCE over the full data
        y_all = dataset.get_all_labels() # Assumes dataset has this helper method
        cls_w = compute_class_weight("balanced", classes=np.arange(num_cls), y=y_all)
        cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(cls_w_t)

        log_accs = []
        overall_y_true, overall_y_pred = [], []

        
        # --- Cross-Validation Strategy ---
        # If n_folds is specified, use GroupShuffleSplit. Otherwise, default to LOSO.
        # Also, if n_folds is set, it implies max_folds should match unless explicitly overridden.
        if self.cfg.get("n_folds") and self.cfg.get("max_folds") is None:
            self.cfg["max_folds"] = self.cfg["n_folds"]

        if self.cfg.get('n_folds'):
            k = int(self.cfg['n_folds'])
            gss = GroupShuffleSplit(n_splits=k, test_size=1.0/k, random_state=self.cfg.get("random_state"))
            fold_iter = gss.split(np.zeros(len(dataset)), y_all, groups)
        else:
            fold_iter = LeaveOneGroupOut().split(np.zeros(len(dataset)), y_all, groups)
        
        main_loop_context = sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]) if SDP_IS_AVAILABLE else nullcontext()

        with main_loop_context:
            for fold, (tr_idx, va_idx) in enumerate(fold_iter):
                if self.cfg.get("max_folds") is not None and fold >= self.cfg["max_folds"]:
                    break
                
                sys.stdout.write(f"\n--- Starting Fold {fold+1:02d} ---\n")
                sys.stdout.flush()
                sys.stdout.write("Preparing data loaders...\n")
                sys.stdout.flush()

                # Build data loaders for the fold
                num_workers = self.cfg.get("num_workers")
                if num_workers is None:
                    # Heuristic: num_workers=0 is often faster on Windows due to 'spawn' overhead,
                    # especially for complex models that can be bottlenecked by data loading.
                    model_name = self.cfg.get("model_name", "eegnet")
                    if model_name in ["cwat", "eegnet_se", "dual_stream"]:
                        print(f"NOTE: Model '{model_name}' detected. Setting num_workers=0 for optimal performance on Windows. This can be overridden with 'num_workers' in the config.", flush=True)
                        num_workers = 0
                    else:
                        num_workers = 2

                aug = aug_builder(self.cfg, dataset)
                dataset.set_transform(aug) # Assumes dataset has a method to set transform
                tr_ld = DataLoader(Subset(dataset, tr_idx), batch_size=self.cfg["batch_size"], shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)
                
                dataset.set_transform(None) # No augmentation for validation
                va_ld = DataLoader(Subset(dataset, va_idx), batch_size=self.cfg["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)

                model = model_builder(self.cfg, num_cls).to(DEVICE)
                opt, sched = self.get_optimizer(model)
                
                best_val_loss = float("inf")
                best_acc = 0.0
                patience = 0
                
                tr_history, val_history, acc_history = [], [], []
                best_y_true_fold, best_y_pred_fold = [], []

                for epoch in range(1, self.cfg["epochs"] + 1):
                    # --- Train Step ---
                    model.train()
                    train_loss_total = 0.0
                    for xb, yb in tr_ld:
                        if isinstance(xb, (list, tuple)):
                            xb = [x.to(DEVICE) for x in xb]
                        else:
                            xb = xb.to(DEVICE)
                        yb = yb.to(DEVICE)
                        
                        input_adapted = input_adapter(xb) if input_adapter else xb
                        
                        opt.zero_grad()
                        with autocast(enabled=USE_AMP):
                            if isinstance(input_adapted, (list, tuple)):
                                out = model(*input_adapted)
                            else:
                                out = model(input_adapted)
                            loss = loss_fn(out.float(), yb)
                        
                        if USE_AMP:
                            scaler.scale(loss).backward()
                            scaler.step(opt)
                            scaler.update()
                        else:
                            loss.backward()
                            opt.step()
                        
                        train_loss_total += loss.item()
                    
                    train_loss = train_loss_total / len(tr_ld)
                    tr_history.append(train_loss)

                    # --- Validation Step ---
                    model.eval()
                    val_loss_total, correct, total = 0.0, 0, 0
                    y_true_fold, y_pred_fold = [], []
                    with torch.no_grad():
                        for xb, yb in va_ld:
                            yb_cpu = yb
                            if isinstance(xb, (list, tuple)):
                                xb = [x.to(DEVICE) for x in xb]
                            else:
                                xb = xb.to(DEVICE)
                            yb_gpu = yb_cpu.to(DEVICE)
                            
                            input_adapted = input_adapter(xb) if input_adapter else xb

                            with autocast(enabled=USE_AMP):
                                if isinstance(input_adapted, (list, tuple)):
                                    out = model(*input_adapted)
                                else:
                                    out = model(input_adapted)
                            
                            loss = loss_fn(out.float(), yb_gpu)
                            val_loss_total += loss.item()
                            
                            preds = out.argmax(1).cpu()
                            correct += (preds == yb_cpu).sum().item()
                            total += yb_cpu.size(0)
                            
                            y_true_fold.extend(yb_cpu.tolist())
                            y_pred_fold.extend(preds.tolist())

                    val_loss = val_loss_total / len(va_ld)
                    val_acc = 100 * correct / total
                    val_history.append(val_loss)
                    acc_history.append(val_acc)

                    sched.step(val_loss)

                    print(f"Fold {fold+1:02d} | Ep {epoch:03d} | Tr Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Acc {val_acc:.2f}%", flush=True)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_acc = val_acc
                        patience = 0
                        best_y_true_fold = y_true_fold
                        best_y_pred_fold = y_pred_fold
                        if self.run_dir and self.cfg.get("save_ckpt", False):
                            torch.save(model.state_dict(), self.run_dir / f"fold_{fold+1:02d}_best.ckpt")
                    else:
                        patience += 1

                    if patience >= self.cfg["early_stop"]:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                log_accs.append(best_acc)
                overall_y_true.extend(best_y_true_fold)
                overall_y_pred.extend(best_y_pred_fold)

                sys.stdout.write(f"Fold {fold+1:02d} complete. Best accuracy: {best_acc:.2f}%.\n")
                sys.stdout.flush()
                # --- Plotting ---
                if self.run_dir:
                    sys.stdout.write(f"Saving plots for fold {fold+1:02d}...\n")
                    sys.stdout.flush()
                    fold_title = f"{self.cfg.get('task','').replace('_',' ')} · F{fold+1} · Acc {best_acc:.1f}%"
                    plot_confusion(
                        best_y_true_fold, best_y_pred_fold, class_names,
                        self.run_dir / f"fold{fold+1}_confusion.png", title=fold_title
                    )
                    plot_curves(
                        tr_history, val_history, acc_history,
                        self.run_dir / f"fold{fold+1}_curves.png", title=fold_title
                    )


        # --- Overall Results ---
        mean_acc = np.mean(log_accs)
        std_acc = np.std(log_accs)
        if self.run_dir and overall_y_true:
            macro_f1 = f1_score(overall_y_true, overall_y_pred, average="macro") * 100
            plot_confusion(
                overall_y_true,
                overall_y_pred,
                class_names,
                self.run_dir / "overall_confusion.png",
                title=f"Overall · Mean {mean_acc:.1f}% · Macro-F1 {macro_f1:.1f}%",
            )
        
        return {"mean_acc": float(mean_acc), "std_acc": float(std_acc)}
