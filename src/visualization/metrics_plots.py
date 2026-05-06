"""
Confusion matrix and ROC curve plots for the evaluation phase.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path | str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Save a seaborn confusion matrix heatmap.

    Args:
        cm:           integer confusion matrix (n_classes, n_classes).
        class_names:  ordered list of class label strings.
        output_path:  file path for the PNG.
        title:        plot title.
        normalize:    if True, normalize each row to [0,1] (recall per class).
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # avoid divide-by-zero for absent classes
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        cm_plot = cm.astype(float)
        fmt = ".0f"
        vmin, vmax = 0, None

    n = len(class_names)
    fig_size = max(8, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.3,
        linecolor="lightgrey",
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title + (" (row-normalized)" if normalize else ""), fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_path: Path | str,
    title: str = "ROC Curves (one-vs-rest)",
) -> dict[str, float]:
    """Save per-class ROC curves (one-vs-rest) and return AUC dict.

    Args:
        y_true:       integer ground-truth labels, shape (N,).
        y_prob:       softmax probabilities, shape (N, n_classes).
        class_names:  ordered list of class label strings.
        output_path:  file path for the PNG.
        title:        plot title.

    Returns:
        dict mapping class_name → AUC (NaN for absent classes).
    """
    n_classes = len(class_names)
    present   = np.unique(y_true)
    classes   = np.arange(n_classes)

    # Binarize ground truth (N, n_classes)
    y_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    # Compute per-class ROC and AUC
    fpr_dict:  dict[int, np.ndarray] = {}
    tpr_dict:  dict[int, np.ndarray] = {}
    auc_dict:  dict[str, float]      = {}

    for i, name in enumerate(class_names):
        if i not in present:
            auc_dict[name] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fpr_dict[i]  = fpr
        tpr_dict[i]  = tpr
        auc_dict[name] = float(auc(fpr, tpr))

    # Plot
    cmap   = plt.colormaps["tab20"]
    colors = [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

    for i, name in enumerate(class_names):
        if i not in fpr_dict:
            continue
        auc_val = auc_dict[name]
        ax.plot(
            fpr_dict[i], tpr_dict[i],
            color=colors[i], lw=1.5,
            label=f"{name}  (AUC={auc_val:.3f})",
        )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curves saved → {output_path}")

    return auc_dict
