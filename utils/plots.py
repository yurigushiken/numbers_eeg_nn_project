"""Shared plotting utilities for confusion matrices and training curves.

All engines (CNN, Hybrid, ViT, …) should import these helpers so that every
run produces visually identical artefacts.

plot_confusion(...)
plot_curves(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix

# Global style – matches what the ViT / Hybrid scripts used
sns.set(style="white", font="DejaVu Sans", font_scale=1.0)

__all__ = ["plot_confusion", "plot_curves"]

def _ensure_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)


def plot_confusion(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
    outfile: Path | str,
    title: str | None = None,
    hyper_lines: Optional[List[str]] = None,
    vmax: int = 100,
) -> None:
    """Plot percentage confusion matrix with black diagonal & orange worst-error.

    Parameters
    ----------
    y_true / y_pred
        Ground-truth and predicted class indices.
    class_names
        Iterable of class labels to show on axes.
    outfile
        Image path where the PNG is saved (dpi=300).
    title
        Optional title above the plot.
    hyper_lines
        Optional list of strings ("key: value") written on the **left margin**
        – useful to display Optuna hyper-parameters for quick glance.
    vmax
        Upper colour-bar limit (default 100 for percentages).
    """
    outfile = _ensure_path(outfile)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_perc = cm.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_perc = cm_perc / cm.sum(axis=1, keepdims=True)
    cm_perc[np.isnan(cm_perc)] = 0.0  # rows with no samples

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_perc * 100,
        annot=False,  # we'll annotate manually to work around seaborn glitches
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        vmin=0,
        vmax=vmax,
    )

    # ------------------------------------------------------------------
    # Manual annotation layer – ensures cells with 0 % (or NaN → 0) still
    # display a value (or a dash when the row has no samples).
    # ------------------------------------------------------------------
    for r in range(cm_perc.shape[0]):
        row_total = cm[r].sum()
        for c in range(cm_perc.shape[1]):
            perc = cm_perc[r, c] * 100
            # Show “—” if that class never occurs in the ground-truth.
            label = "—" if row_total == 0 else f"{perc:.1f}"
            col = "white" if perc > vmax * 0.5 else "black"
            ax.text(c + 0.5, r + 0.5, label, ha="center", va="center", color=col, fontsize=9)

    # Highlight correct diag in black & worst error per row in orange.
    for r in range(len(class_names)):
        ax.add_patch(Rectangle((r, r), 1, 1, fill=False, edgecolor="black", lw=2))
        if cm_perc[r].sum() > 0:
            worst_c = cm_perc[r].argmax()
            ax.add_patch(Rectangle((worst_c, r), 1, 1, fill=False, edgecolor="orange", lw=2))

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_yticklabels(class_names, rotation=0)
    if title:
        ax.set_title(title)

    # Optional hyper-param annotation on the left margin
    if hyper_lines:
        max_len = max(len(s) for s in hyper_lines)
        left_margin = min(0.5, 0.15 + max_len * 0.006)
        fig.subplots_adjust(left=left_margin)
        fig.text(0.02, 0.5, "\n".join(hyper_lines), va="center", ha="left", fontsize=7, family="monospace")
    else:
        fig.subplots_adjust(left=0.40)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    val_accs: Sequence[float],
    outfile: Path | str,
    hyper_lines: Optional[List[str]] = None,
    lock_acc_axis: bool = False,
    title: str | None = None,
) -> None:
    """Plot training vs. validation loss curves + accuracy overlay.

    Mirrors the styling used by the Hybrid CwA-Transformer plots.
    """
    outfile = _ensure_path(outfile)
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accs, color="green", label="Val acc")
    ax2.set_ylabel("Val Acc %")
    if lock_acc_axis:
        ax2.set_ylim(0, 40)

    # Optional title (mirrors confusion matrix helper)
    if title:
        ax1.set_title(title)

    # Hyper-parameter text on the left
    if hyper_lines:
        max_len = max(len(s) for s in hyper_lines)
        left_margin = min(0.6, 0.25 + max_len * 0.007)
        fig.subplots_adjust(left=left_margin)
        fig.text(0.02, 0.5, "\n".join(hyper_lines), va="center", ha="left", fontsize=6, family="monospace")
    else:
        fig.subplots_adjust(left=0.20)

    # Combine legends
    lines, labels = [], []
    for ax in (ax1, ax2):
        lns, lbls = ax.get_legend_handles_labels()
        lines.extend(lns)
        labels.extend(lbls)
    fig.legend(lines, labels, loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig) 