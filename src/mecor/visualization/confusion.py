"""Confusion matrix visualization."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from mecor.visualization.style import save_fig, setup_style


def plot_confusion_matrices(
    cv_results,
    labels: List[str],
    method: str = "ensemble",
    output_path: Path = None,
    dpi: int = 300,
):
    """Plot normalized confusion matrices for a method across all drugs.

    Accumulates confusion matrices across all CV folds and normalizes.
    """
    setup_style()

    method_results = [r for r in cv_results.fold_results if r.method == method]
    n_labels = len(labels)

    fig, axes = plt.subplots(1, n_labels, figsize=(3.5 * n_labels, 3.5))
    if n_labels == 1:
        axes = [axes]

    cmap = "Greens" if method == "ensemble" else "Blues"

    for idx, label in enumerate(labels):
        ax = axes[idx]
        cm_sum = np.zeros((2, 2))

        for r in method_results:
            if label in r.y_true and label in r.y_pred:
                cm = confusion_matrix(r.y_true[label], r.y_pred[label], labels=[0, 1])
                cm_sum += cm

        cm_norm = cm_sum / cm_sum.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            ax=ax,
            cbar=False,
            vmin=0,
            vmax=1,
            xticklabels=["S", "R"],
            yticklabels=["S", "R"],
        )
        ax.set_title(f"{label}\n({method})", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle("Normalized Confusion Matrices", fontsize=10, y=1.02)
    plt.tight_layout()
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig


def plot_confusion_matrices_grid(
    cv_results,
    labels: List[str],
    output_path: Path = None,
    dpi: int = 300,
):
    """Plot confusion matrices for BR, Chain Ensemble side by side."""
    setup_style()

    methods = []
    if any(r.method == "binary_relevance" for r in cv_results.fold_results):
        methods.append(("binary_relevance", "BR", "Blues"))
    if any(r.method == "ensemble" for r in cv_results.fold_results):
        methods.append(("ensemble", "Chain Ensemble", "Greens"))

    n_methods = len(methods)
    n_labels = len(labels)

    fig, axes = plt.subplots(n_labels, n_methods, figsize=(3.5 * n_methods, 3.5 * n_labels))
    if n_labels == 1:
        axes = axes.reshape(1, -1)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, (method_key, method_name, cmap) in enumerate(methods):
        method_results = [r for r in cv_results.fold_results if r.method == method_key]

        for row_idx, label in enumerate(labels):
            ax = axes[row_idx, col_idx]
            cm_sum = np.zeros((2, 2))

            for r in method_results:
                if label in r.y_true and label in r.y_pred:
                    cm = confusion_matrix(r.y_true[label], r.y_pred[label], labels=[0, 1])
                    cm_sum += cm

            cm_norm = cm_sum / cm_sum.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

            sns.heatmap(
                cm_norm, annot=True, fmt=".3f", cmap=cmap, ax=ax,
                cbar=False, vmin=0, vmax=1,
                xticklabels=["S", "R"], yticklabels=["S", "R"],
            )
            ax.set_title(f"{label} - {method_name}", fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

    plt.tight_layout()
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig
