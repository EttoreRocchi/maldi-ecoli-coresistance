"""Feature importance and chain gain visualization."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from mecor.evaluation.importance import compute_chain_gain
from mecor.evaluation.metrics import ci_halfwidth, error_bar_halfwidth
from mecor.visualization.style import PALETTE, save_fig, setup_style


def plot_feature_importances(
    cv_results,
    labels: List[str],
    output_dir: Path,
    top_n: int = 20,
    confidence: float = 0.95,
    dpi: int = 300,
    error_mode: str = "sd",
):
    """Plot top feature importances aggregated across chain orderings.

    Highlights prediction_* features in salmon to show chain contribution.
    """
    setup_style()

    chain_results = [
        r for r in cv_results.fold_results
        if r.method.startswith("chain_") and r.method != "ensemble"
    ]

    for label in labels:
        all_data = []
        for r in chain_results:
            if label in r.importances:
                imp_array, feat_names = r.importances[label]
                fold_data = dict(zip(feat_names, imp_array))
                all_data.append(fold_data)

        if not all_data:
            continue

        all_feat_names = sorted(set(k for d in all_data for k in d.keys()))

        imp_matrix = np.full((len(all_data), len(all_feat_names)), np.nan)
        for i, fold_data in enumerate(all_data):
            for j, name in enumerate(all_feat_names):
                if name in fold_data:
                    imp_matrix[i, j] = fold_data[name]

        mean_imp = np.nanmean(imp_matrix, axis=0)
        std_imp = np.nanstd(imp_matrix, axis=0, ddof=1)
        n_obs = np.sum(~np.isnan(imp_matrix), axis=0)

        valid_mask = ~np.isnan(mean_imp)
        valid_idx = np.where(valid_mask)[0]
        sorted_idx = valid_idx[np.argsort(mean_imp[valid_idx])[-top_n:][::-1]]

        plot_names = [all_feat_names[i] for i in sorted_idx]
        plot_means = mean_imp[sorted_idx]
        k = cv_results.config.n_splits_outer if cv_results.config else None
        plot_cis = np.array([
            error_bar_halfwidth(std_imp[i], n_obs[i], mode=error_mode, confidence=confidence, k=k) for i in sorted_idx
        ])

        fig, ax = plt.subplots(figsize=(3.5, max(3, top_n * 0.2)))
        y_pos = np.arange(len(plot_names))

        colors = [
            PALETTE["chain_prediction"] if "prediction_" in n else PALETTE["original_feature"]
            for n in plot_names
        ]
        ax.barh(y_pos, plot_means, xerr=plot_cis, capsize=2, alpha=0.8, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_names, fontsize=6)
        eb_text = "Mean \u00b1 SD" if error_mode == "sd" else "Mean \u00b1 95% CI"
        ax.set_xlabel(f"Feature Importance ({eb_text})")
        ax.set_title(f"{label} - Top {top_n} Features\n(Aggregated across orderings)", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        if any("prediction_" in n for n in plot_names):
            legend = [
                Patch(facecolor=PALETTE["original_feature"], alpha=0.8, label="m/z features"),
                Patch(facecolor=PALETTE["chain_prediction"], alpha=0.8, label="Chain predictions"),
            ]
            ax.legend(handles=legend, loc="lower right", fontsize=6)

        plt.tight_layout()
        save_fig(fig, output_dir / f"feature_importance_{label}_aggregated.png", dpi)


def plot_chain_gain_summary(
    cv_results,
    labels: List[str],
    output_path: Path = None,
    dpi: int = 300,
):
    """Plot chain gain summary: relative importance of chain predictions per label."""
    setup_style()

    chain_results = [
        r for r in cv_results.fold_results
        if r.method.startswith("chain_") and r.method != "ensemble"
    ]

    importance_store = {}
    for r in chain_results:
        ordering_key = r.method.replace("chain_", "")
        if ordering_key not in importance_store:
            importance_store[ordering_key] = {l: [] for l in labels}
        for label in labels:
            if label in r.importances:
                importance_store[ordering_key][label].append(r.importances[label])

    gain = compute_chain_gain(importance_store, labels)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(labels))
    means = [gain[l]["mean_rel_importance"] * 100 for l in labels]
    stds = [gain[l]["std_rel_importance"] * 100 for l in labels]

    ax.bar(x, means, yerr=stds, capsize=5, color=PALETTE["chain_prediction"], alpha=0.8, width=0.5)
    ax.set_ylabel("Chain Gain (%)\n(Relative importance of predictions)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Chain Prediction Contribution", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig


def save_feature_importances_csv(
    cv_results,
    labels: List[str],
    output_dir: Path,
    confidence: float = 0.95,
):
    """Save aggregated feature importances to CSV files."""
    chain_results = [
        r for r in cv_results.fold_results
        if r.method.startswith("chain_") and r.method != "ensemble"
    ]

    for label in labels:
        all_data = []
        for r in chain_results:
            if label in r.importances:
                imp_array, feat_names = r.importances[label]
                all_data.append(dict(zip(feat_names, imp_array)))

        if not all_data:
            continue

        all_feat_names = sorted(set(k for d in all_data for k in d.keys()))
        imp_matrix = np.full((len(all_data), len(all_feat_names)), np.nan)
        for i, fold_data in enumerate(all_data):
            for j, name in enumerate(all_feat_names):
                if name in fold_data:
                    imp_matrix[i, j] = fold_data[name]

        mean_imp = np.nanmean(imp_matrix, axis=0)
        std_imp = np.nanstd(imp_matrix, axis=0, ddof=1)
        n_obs = np.sum(~np.isnan(imp_matrix), axis=0)
        k = cv_results.config.n_splits_outer if cv_results.config else None
        ci_imp = np.array([ci_halfwidth(std_imp[i], n_obs[i], confidence, k=k) for i in range(len(all_feat_names))])

        df = pd.DataFrame({
            "Feature": all_feat_names,
            "Mean_Importance": mean_imp,
            "Std": std_imp,
            "CI_95": ci_imp,
            "n_folds": n_obs,
        }).sort_values("Mean_Importance", ascending=False)

        df.to_csv(output_dir / f"feature_importance_{label}_aggregated.csv", index=False)
