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
import copy
from typing import Callable, Dict, Any, Tuple, Optional, List
from contextlib import nullcontext
import random
import json

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
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from utils.plots import plot_confusion, plot_curves

import optuna

def seed_worker(worker_id):
    """
    Seeds a DataLoader worker to ensure reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
        # Optional linear LR scaling with batch size
        if bool(self.cfg.get("auto_lr", False)):
            try:
                ref_bs = int(self.cfg.get("auto_lr_ref_bs", 64))
                bs = int(self.cfg.get("batch_size", ref_bs))
                if bs != ref_bs and ref_bs > 0:
                    lr = lr * (bs / ref_bs)
                    print(f"[auto_lr] Scaling LR to {lr:.3e} for batch_size={bs} (ref {ref_bs})", flush=True)
            except Exception:
                pass
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
            input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            optuna_trial: optuna.Trial | None = None):
        
        num_cls = len(class_names)
        scaler = GradScaler(enabled=USE_AMP)

        # Access all labels once (used for splitting only; loss weights are per-fold)
        y_all = dataset.get_all_labels() # Assumes dataset has this helper method

        log_accs = []
        overall_y_true, overall_y_pred = [], []
        per_fold_metrics = []
        fold_split_info = []
        inner_val_best_accs = []
        inner_val_best_macro_f1s = []

        global_step = 0  # Initialize global counter before the loop
        
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
                
                # --- CAPTURE the test subject IDs for this fold ---
                test_subjects = np.unique(groups[va_idx]).tolist()
                fold_split_info.append({
                    "fold": fold + 1,
                    "test_subjects": test_subjects
                })

                sys.stdout.write(f"\n--- Starting Fold {fold+1:02d} (Test Subjects: {test_subjects}) ---\n")
                sys.stdout.flush()
                sys.stdout.write("Preparing data loaders...\n")
                sys.stdout.flush()

                # Build data loaders for the fold
                num_workers = self.cfg.get("num_workers")
                if num_workers is None:
                    # Heuristic: num_workers=0 is often faster on Windows due to 'spawn' overhead,
                    # especially for complex models that can be bottlenecked by data loading.
                    model_name = self.cfg.get("model_name", "eegnet")
                    if model_name in ["cwat", "eegnet_se", "dual_stream", "eegnex"]:
                        print(f"NOTE: Model '{model_name}' detected. Setting num_workers=0 for optimal performance on Windows. This can be overridden with 'num_workers' in the config.", flush=True)
                        num_workers = 0
                    else:
                        num_workers = 2

                # --- CREATE a seeded generator for the DataLoader ---
                g = torch.Generator()
                seed = self.cfg.get("seed")
                if seed is not None:
                    g.manual_seed(seed)

                # --- Inner validation split (leakage fix) ---
                # Create a subject-aware split inside the outer training indices.
                inner_val_frac = float(self.cfg.get("inner_val_frac", 0.2))
                gss_inner = GroupShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=self.cfg.get("random_state"))
                inner_split = next(gss_inner.split(
                    np.zeros(len(tr_idx)),
                    y_all[tr_idx],
                    groups[tr_idx]
                ))
                inner_tr_rel, inner_va_rel = inner_split
                inner_tr_idx = tr_idx[inner_tr_rel]
                inner_va_idx = tr_idx[inner_va_rel]

                # Build two shallow copies so transforms don't leak across loaders
                dataset_tr = copy.copy(dataset)
                dataset_eval = copy.copy(dataset)

                aug = aug_builder(self.cfg, dataset)
                dataset_tr.set_transform(aug) # Train-time augmentation only
                tr_ld = DataLoader(
                    Subset(dataset_tr, inner_tr_idx), 
                    batch_size=self.cfg["batch_size"], 
                    shuffle=True, 
                    drop_last=False, 
                    num_workers=num_workers, 
                    pin_memory=True,
                    worker_init_fn=seed_worker if seed is not None else None,
                    generator=g if seed is not None else None
                )

                # Inner validation loader (no augmentation)
                dataset_eval.set_transform(None)
                va_ld = DataLoader(
                    Subset(dataset_eval, inner_va_idx),
                    batch_size=self.cfg["batch_size"],
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )

                # Outer test loader (never used for early stopping/scheduling)
                te_ld = DataLoader(
                    Subset(dataset_eval, va_idx),
                    batch_size=self.cfg["batch_size"],
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )

                model = model_builder(self.cfg, num_cls).to(DEVICE)
                opt, sched = self.get_optimizer(model)

                # --- Per-fold class weights computed from INNER TRAIN ONLY (leakage fix) ---
                y_inner_train = y_all[inner_tr_idx]
                cls_w = compute_class_weight("balanced", classes=np.arange(num_cls), y=y_inner_train)
                cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
                loss_fn = nn.CrossEntropyLoss(cls_w_t)

                best_val_loss = float("inf")
                best_inner_val_acc = 0.0
                best_inner_val_macro_f1 = 0.0
                best_acc = 0.0
                patience = 0
                
                tr_history, val_history, acc_history = [], [], []
                best_state_dict = None

                for epoch in range(1, self.cfg["epochs"] + 1):
                    # --- Train Step ---
                    model.train()
                    train_loss_total = 0.0
                    mixup_alpha = float(self.cfg.get("mixup_alpha", 0.0) or 0.0)
                    for xb, yb in tr_ld:
                        if isinstance(xb, (list, tuple)):
                            xb = [x.to(DEVICE) for x in xb]
                        else:
                            xb = xb.to(DEVICE)
                        yb = yb.to(DEVICE)
                        
                        input_adapted = input_adapter(xb) if input_adapter else xb
                        
                        opt.zero_grad()
                        with autocast(enabled=USE_AMP):
                            if mixup_alpha > 0.0:
                                # --- MixUp ---
                                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                                perm = torch.randperm(yb.size(0), device=yb.device)
                                yb_perm = yb[perm]
                                if isinstance(input_adapted, (list, tuple)):
                                    mixed_inputs = []
                                    for xi in input_adapted:
                                        mixed_inputs.append(lam * xi + (1.0 - lam) * xi[perm])
                                    out = model(*mixed_inputs)
                                else:
                                    mixed_input = lam * input_adapted + (1.0 - lam) * input_adapted[perm]
                                    out = model(mixed_input)
                                ce1 = loss_fn(out.float(), yb)
                                ce2 = loss_fn(out.float(), yb_perm)
                                loss = lam * ce1 + (1.0 - lam) * ce2
                            else:
                                if isinstance(input_adapted, (list, tuple)):
                                    out = model(*input_adapted)
                                else:
                                    out = model(input_adapted)
                                loss = loss_fn(out.float(), yb)
                        # Add L1 penalty on channel gates if enabled
                        # Enforce explicit naming for channel gate L1
                        if "gate_l1_lambda" in self.cfg:
                            raise ValueError("Config key 'gate_l1_lambda' has been removed. Use 'channel_gate_l1_lambda'.")
                        l1_lambda = float(self.cfg.get("channel_gate_l1_lambda", 0.0) or 0.0)
                        if l1_lambda > 0.0 and hasattr(model, "gate_l1_penalty"):
                            loss = loss + l1_lambda * model.gate_l1_penalty()
                        # Add temporal gate penalties if enabled
                        tg_l1 = float(self.cfg.get("time_gate_l1_lambda", 0.0) or 0.0)
                        tg_tv = float(self.cfg.get("time_gate_tv_lambda", 0.0) or 0.0)
                        if tg_l1 > 0.0 and hasattr(model, "time_gate_l1_penalty"):
                            loss = loss + tg_l1 * model.time_gate_l1_penalty()
                        if tg_tv > 0.0 and hasattr(model, "time_gate_tv_penalty"):
                            loss = loss + tg_tv * model.time_gate_tv_penalty()
                        
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

                    # --- Inner Validation Step (used for early stopping & scheduler) ---
                    model.eval()
                    val_loss_total, correct, total = 0.0, 0, 0
                    val_y_true_epoch: List[int] = []
                    val_y_pred_epoch: List[int] = []
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
                            # Collect for macro-F1 on inner validation
                            val_y_true_epoch.extend(yb_cpu.tolist())
                            val_y_pred_epoch.extend(preds.tolist())

                    val_loss = val_loss_total / len(va_ld)
                    val_acc = 100 * correct / total
                    # Compute macro-F1 for inner validation at this epoch
                    try:
                        val_macro_f1 = f1_score(val_y_true_epoch, val_y_pred_epoch, average="macro") * 100
                    except Exception:
                        val_macro_f1 = 0.0
                    val_history.append(val_loss)
                    acc_history.append(val_acc)

                    sched.step(val_loss)

                    print(f"Fold {fold+1:02d} | Ep {epoch:03d} | Tr Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Acc {val_acc:.2f}%", flush=True)

                    if optuna_trial:
                        global_step += 1  # Increment the global step
                        optuna_trial.report(val_acc, global_step) # Report using the global step
                        if optuna_trial.should_prune():
                            print(f"  Trial pruned at epoch {epoch} of fold {fold+1} due to low performance.")
                            raise optuna.exceptions.TrialPruned()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_inner_val_acc = val_acc
                        best_inner_val_macro_f1 = val_macro_f1
                        patience = 0
                        best_state_dict = copy.deepcopy(model.state_dict())
                        if self.run_dir and self.cfg.get("save_ckpt", False):
                            torch.save(model.state_dict(), self.run_dir / f"fold_{fold+1:02d}_best.ckpt")
                    else:
                        patience += 1

                    if patience >= self.cfg["early_stop"]:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                # --- Test Evaluation (outer fold) using best inner-val checkpoint ---
                if best_state_dict is not None:
                    model.load_state_dict(best_state_dict)
                model.eval()
                y_true_fold, y_pred_fold = [], []
                correct_test, total_test = 0, 0
                with torch.no_grad():
                    for xb, yb in te_ld:
                        yb_cpu = yb
                        if isinstance(xb, (list, tuple)):
                            xb = [x.to(DEVICE) for x in xb]
                        else:
                            xb = xb.to(DEVICE)
                        yb_gpu = yb_cpu.to(DEVICE)

                        # Apply the same input adapter used during training/validation
                        input_adapted = input_adapter(xb) if input_adapter else xb

                        with autocast(enabled=USE_AMP):
                            if isinstance(input_adapted, (list, tuple)):
                                out = model(*input_adapted)
                            else:
                                out = model(input_adapted)

                        preds = out.argmax(1).cpu()
                        correct_test += (preds == yb_cpu).sum().item()
                        total_test += yb_cpu.size(0)
                        y_true_fold.extend(yb_cpu.tolist())
                        y_pred_fold.extend(preds.tolist())

                best_acc = 100 * correct_test / total_test
                log_accs.append(best_acc)
                overall_y_true.extend(y_true_fold)
                overall_y_pred.extend(y_pred_fold)
                inner_val_best_accs.append(best_inner_val_acc)
                inner_val_best_macro_f1s.append(best_inner_val_macro_f1)

                # --- Print per-fold inner validation accuracy (at best val loss) ---
                try:
                    print(
                        f"Fold {fold+1:02d} | Best inner-val acc (at best val loss): {best_inner_val_acc:.2f}% | "
                        f"Best inner-val macro-F1: {best_inner_val_macro_f1:.2f}%",
                        flush=True
                    )
                except Exception:
                    pass

                # --- Save per-fold Channel Gate values (if present) ---
                try:
                    if hasattr(model, "get_gate_values") and hasattr(dataset, "channel_names"):
                        gates = model.get_gate_values().cpu().numpy().tolist()
                        ch_names = dataset.channel_names
                        # Compute simple sparsity metrics
                        gates_np = np.array(gates)
                        l1_val = float(np.sum(gates_np))
                        # Fraction below small threshold (e.g., <0.1)
                        sparsity_frac = float((gates_np < 0.1).mean())
                        payload = {
                            "channels": ch_names,
                            "gates": gates,
                            "l1_sum": l1_val,
                            "sparsity_frac_lt_0_1": sparsity_frac,
                        }
                        (self.run_dir / f"fold{fold+1}_gate_values.json").write_text(json.dumps(payload, indent=2))
                except Exception as _e:
                    print(f"[WARN] Could not save gate values for fold {fold+1}: {_e}")

                # --- Save per-fold Temporal Gate values (if present) ---
                try:
                    if hasattr(model, "get_time_gate_values"):
                        g_time = model.get_time_gate_values().cpu().numpy().tolist()
                        times_ms = dataset.times_ms.tolist() if hasattr(dataset, "times_ms") else list(range(len(g_time)))
                        g_np = np.array(g_time)
                        l1_sum = float(np.sum(g_np))
                        tv_sum = float(np.sum(np.abs(np.diff(g_np)))) if g_np.size > 1 else 0.0
                        payload = {
                            "times_ms": times_ms,
                            "gates": g_time,
                            "l1_sum": l1_sum,
                            "tv_sum": tv_sum,
                        }
                        (self.run_dir / f"fold{fold+1}_time_gate_values.json").write_text(json.dumps(payload, indent=2))
                except Exception as _e:
                    print(f"[WARN] Could not save temporal gate values for fold {fold+1}: {_e}")

                # --- New: Generate and store per-class metrics for this fold ---
                report = classification_report(
                    y_true_fold,
                    y_pred_fold,
                    labels=list(range(num_cls)),
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0
                )
                per_fold_metrics.append({"fold": fold + 1, "classification_report": report})

                sys.stdout.write(f"Fold {fold+1:02d} complete. Best accuracy: {best_acc:.2f}%.\n")
                sys.stdout.flush()
                # --- Plotting ---
                if self.run_dir:
                    sys.stdout.write(f"Saving plots for fold {fold+1:02d}...\n")
                    sys.stdout.flush()
                    fold_title = f"{self.cfg.get('task','').replace('_',' ')} · F{fold+1} · Acc {best_acc:.1f}%"
                    plot_confusion(
                        y_true_fold, y_pred_fold, class_names,
                        self.run_dir / f"fold{fold+1}_confusion.png", title=fold_title
                    )
                    plot_curves(
                        tr_history, val_history, acc_history,
                        self.run_dir / f"fold{fold+1}_curves.png", title=fold_title
                    )


        # --- Overall Results ---
        mean_acc = np.mean(log_accs)
        std_acc = np.std(log_accs)

        # NEW: Compute class-aggregated F1 metrics once (if we have predictions)
        macro_f1 = 0.0
        weighted_f1 = 0.0
        if overall_y_true:
            macro_f1 = f1_score(overall_y_true, overall_y_pred, average="macro") * 100
            weighted_f1 = f1_score(overall_y_true, overall_y_pred, average="weighted") * 100

        if self.run_dir and overall_y_true:
            plot_confusion(
                overall_y_true,
                overall_y_pred,
                class_names,
                self.run_dir / "overall_confusion.png",
                title=f"Overall · Mean {mean_acc:.1f}% · Macro-F1 {macro_f1:.1f}%",
            )
        
        inner_mean_acc = float(np.mean(inner_val_best_accs)) if inner_val_best_accs else 0.0
        inner_mean_macro_f1 = float(np.mean(inner_val_best_macro_f1s)) if inner_val_best_macro_f1s else 0.0
        try:
            print(
                f"--- Inner mean validation accuracy (best per fold): {inner_mean_acc:.2f}% | "
                f"Inner mean macro-F1: {inner_mean_macro_f1:.2f}% ---",
                flush=True
            )
        except Exception:
            pass
        return {
            "mean_acc": float(mean_acc), 
            "std_acc": float(std_acc), 
            "inner_mean_acc": inner_mean_acc,
            "inner_mean_macro_f1": inner_mean_macro_f1,
            "inner_accs": inner_val_best_accs,
            "inner_macro_f1s": inner_val_best_macro_f1s,
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "fold_accuracies": log_accs,
            "per_fold_class_metrics": per_fold_metrics,
            "fold_splits": fold_split_info
        }
