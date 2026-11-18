"""Performance comparison plots: BR vs Chain Ensemble vs Best Chain, ordering comparison, cross-dataset."""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from mecor.evaluation.metrics import error_bar_halfwidth
from mecor.visualization import paper_style as ps
from mecor.visualization.style import PALETTE, get_display_name, save_fig, setup_style

# CTX (Cefotaxime, local) ≡ CRO (Ceftriaxone, DRIAMS-C), interchangeable ESBL markers
_LABEL_EQUIVALENTS = {"CTX": "CRO", "CRO": "CTX"}

# Display labels for plot titles (CRO and CTX denote the same drug across datasets).
_DRUG_DISPLAY = {"CRO": "CRO/CTX"}


def _resolve_label(label, available_labels):
    """Resolve a canonical label to the dataset's actual label (handles CTX↔CRO)."""
    if label in available_labels:
        return label
    if label == "CTX/CRO":
        for candidate in ("CTX", "CRO"):
            if candidate in available_labels:
                return candidate
    alt = _LABEL_EQUIVALENTS.get(label)
    if alt and alt in available_labels:
        return alt
    return label


def find_best_chain_per_label(fold_results, labels, metric_name="AUROC"):
    """Find the best-performing chain ordering for each label.

    Returns
    -------
    dict : label -> {"method": str, "display": str, "mean": float,
                      "std": float, "values": list}
    """
    chain_results = [r for r in fold_results if r.method.startswith("chain_")]
    orderings = sorted(set(r.method for r in chain_results))

    best = {}
    for label in labels:
        best_mean, best_ordering, best_vals = -1, None, []
        for ordering in orderings:
            vals = [r.metrics[label][metric_name]
                    for r in chain_results if r.method == ordering and label in r.metrics]
            m = np.mean(vals) if vals else -1
            if m > best_mean:
                best_mean, best_ordering, best_vals = m, ordering, vals
        if best_ordering:
            best[label] = {
                "method": best_ordering,
                "display": best_ordering.replace("chain_", "").replace("_", "\u2192"),
                "mean": best_mean,
                "std": np.std(best_vals, ddof=1),
                "values": best_vals,
            }
    return best


def _clean(values):
    return np.asarray([v for v in values if v is not None and not np.isnan(v)], float)


def summarize(values, interval_fn=None, error_mode="sd", k=None, confidence=0.95):
    """(mean, low, high) for a list of per-fold values.

    interval_fn(values) -> (mean, low, high) takes precedence (for example a BCa
    bootstrap); otherwise the interval is mean +/- error_bar_halfwidth(error_mode).
    """
    values = _clean(values)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if interval_fn is not None:
        m, lo, hi = interval_fn(values)
        return float(m), float(lo), float(hi)
    m = float(values.mean())
    h = (error_bar_halfwidth(np.std(values, ddof=1), len(values), mode=error_mode, confidence=confidence, k=k)
         if len(values) > 1 else 0.0)
    return m, m - float(h), m + float(h)


def method_values(cv_results, method, label, metric_name):
    """Per-fold metric values of one method for one drug, in fold order."""
    rs = sorted((r for r in cv_results.fold_results if r.method == method and label in r.metrics),
                key=lambda r: (r.repeat_idx, r.fold_idx))
    return [r.metrics[label][metric_name] for r in rs]


def paired_differences(cv_results, method, label, metric_name, reference="binary_relevance"):
    """Per-fold (method minus reference) differences paired on (repeat_idx, fold_idx)."""
    a = {(r.repeat_idx, r.fold_idx): r.metrics[label][metric_name]
         for r in cv_results.fold_results if r.method == method and label in r.metrics}
    b = {(r.repeat_idx, r.fold_idx): r.metrics[label][metric_name]
         for r in cv_results.fold_results if r.method == reference and label in r.metrics}
    keys = sorted(set(a) & set(b))
    return [a[k] - b[k] for k in keys]


def ordering_short(method):
    """'chain_CIP_CTX_CAZ' -> 'CIP→CTX→CAZ'."""
    return "→".join(method.replace("chain_", "").split("_"))


AUROC_LIM = (0.5, 1.0)   # common scale for every panel that shows AUROC levels

CEPHALOSPORINS = ("CTX", "CRO", "CAZ")
# upstream input of a drug in a chain ordering: marker color, marker, hollow, legend name
UPSTREAM_STYLE = {
    "first": (ps.INK, "o", True, "First (no upstream input)"),
    "cip": (ps.ROLE["cip_alone"], "D", False, "CIP alone upstream"),
    "ceph": (ps.MUTED, "o", False, "Cephalosporin upstream"),
}


def upstream_category(method, label):
    """'first', 'cip' (CIP is the only upstream drug) or 'ceph' (a cephalosporin is upstream, with or without CIP)."""
    order = method.replace("chain_", "").split("_"); up = order[:order.index(label)]
    if not up: return "first"
    return "ceph" if any(d in CEPHALOSPORINS for d in up) else "cip"


def _metric_ylim(ax, metric_name, lows, highs, pad=0.08, head=0.0):
    """Common AUROC scale (0.5 to 1.0); tight limits for other metrics."""
    if str(metric_name).upper() == "AUROC":
        ax.set_ylim(*AUROC_LIM)
    else:
        _tight_ylim(ax, lows, highs, pad=pad, head=head)


def _tight_ylim(ax, lows, highs, pad=0.08, head=0.0):
    vals = _clean(list(lows) + list(highs))
    if vals.size == 0:
        return
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-3)
    ax.set_ylim(lo - pad * span, hi + (pad + head) * span)


def _drug_text(ax, label):
    ax.text(0.02, 0.97, _DRUG_DISPLAY.get(label, label), transform=ax.transAxes, va="top", ha="left", fontsize=8)


def plot_br_vs_chain(
    cv_results,
    labels: List[str],
    metric_name: str = "AUROC",
    output_path: Path = None,
    dpi: int = 300,
    error_mode: str = "sd",
    interval_fn=None,
    best_metric: str = None,
):
    """Binary relevance, chain ensemble and best chain per drug as points with intervals.

    One panel: x = drugs; the three methods are dodged point series in the fixed
    role colors (binary relevance first, as the reference). best_metric selects the
    best ordering (default: metric_name).
    """
    setup_style()
    k = cv_results.config.n_splits_outer if cv_results.config else None
    best_chains = find_best_chain_per_label(cv_results.fold_results, labels, best_metric or metric_name)

    series = [("binary_relevance", ps.ROLE["br"], ps.ROLE_NAME["br"]),
              ("ensemble", ps.ROLE["ensemble"], ps.ROLE_NAME["ensemble"]),
              ("best", ps.ROLE["best"], ps.ROLE_NAME["best"])]
    fig, ax = ps.figure("single", height=2.5)
    x = np.arange(len(labels))
    offsets = np.linspace(-0.22, 0.22, len(series))
    lows, highs = [], []
    for (key, color, name), off in zip(series, offsets):
        stats = []
        for label in labels:
            method = best_chains.get(label, {}).get("method") if key == "best" else key
            vals = method_values(cv_results, method, label, metric_name) if method else []
            stats.append(summarize(vals, interval_fn, error_mode, k))
        m, lo, hi = (np.array([s[i] for s in stats]) for i in range(3))
        ps.ci_point(ax, x + off, m, lo, hi, color=color, label=name)
        lows += list(lo); highs += list(hi)
    _metric_ylim(ax, metric_name, lows, highs)
    ax.set_xticks(x)
    ax.set_xticklabels([_DRUG_DISPLAY.get(l, l) for l in labels])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(metric_name)
    ps.despine(ax)
    fig.tight_layout()
    ps.legend_above(fig, y=1.0)
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig


def plot_delta_performance(
    cv_results,
    labels: List[str],
    metric_name: str = "AUROC",
    output_path: Path = None,
    dpi: int = 300,
    error_mode: str = "sd",
    interval_fn=None,
    best_metric: str = None,
):
    """Gain over binary relevance (chain minus BR) for the ensemble and the best chain.

    One panel centred on zero: x = drugs, two dodged point series with intervals of
    the per-fold paired differences. best_metric selects the best ordering (default:
    metric_name).
    """
    setup_style()
    k = cv_results.config.n_splits_outer if cv_results.config else None
    best_chains = find_best_chain_per_label(cv_results.fold_results, labels, best_metric or metric_name)

    fig, ax = ps.figure("single", height=2.5)
    x = np.arange(len(labels))
    series = [("ensemble", ps.ROLE["ensemble"], ps.ROLE_NAME["ensemble"]),
              ("best", ps.ROLE["best"], ps.ROLE_NAME["best"])]
    offsets = (-0.16, 0.16)
    lows, highs = [], []
    for (key, color, name), off in zip(series, offsets):
        stats = []
        for label in labels:
            method = best_chains.get(label, {}).get("method") if key == "best" else key
            d = paired_differences(cv_results, method, label, metric_name) if method else []
            stats.append(summarize(d, interval_fn, error_mode, k))
        m, lo, hi = (np.array([s[i] for s in stats]) for i in range(3))
        ps.ci_point(ax, x + off, m, lo, hi, color=color, label=name)
        lows += list(lo); highs += list(hi)
    ps.hline(ax, 0.0)
    _tight_ylim(ax, lows + [0.0], highs + [0.0])
    ax.set_xticks(x)
    ax.set_xticklabels([_DRUG_DISPLAY.get(l, l) for l in labels])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(f"Gain over binary relevance ({metric_name})")
    ps.despine(ax)
    fig.tight_layout()
    ps.legend_above(fig, y=1.0)
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig


def plot_ordering_comparison(
    cv_results,
    labels: List[str],
    metric_name: str = "AUROC",
    output_path: Path = None,
    confidence: float = 0.95,
    dpi: int = 300,
    error_mode: str = "sd",
    interval_fn=None,
    fig=None,
    axes=None,
    letters: str = "abcdefgh",
    color_by_upstream: bool = False,
):
    """Metric of every chain ordering and of the ensemble, one panel per drug.

    x = the orderings (short labels) plus the ensemble; points with intervals;
    binary relevance as a horizontal reference line; drug named as in-panel text.
    color_by_upstream: color each ordering by what the panel's drug receives upstream (UPSTREAM_STYLE): nothing
    (first position, equivalent to binary relevance), CIP alone, or a cephalosporin (with or without CIP).
    """
    setup_style()
    k = cv_results.config.n_splits_outer if cv_results.config else None
    orderings = sorted({r.method for r in cv_results.fold_results if r.method.startswith("chain_")})
    names = [ordering_short(o) for o in orderings] + ["Ensemble"]

    n = len(labels)
    own = axes is None
    if own:
        fig, axes = ps.figure("double", height=2.7, ncols=n)
    axes = np.atleast_1d(axes)
    br_handle = None
    for i, (ax, label) in enumerate(zip(axes, labels)):
        stats = [summarize(method_values(cv_results, m, label, metric_name), interval_fn, error_mode, k, confidence)
                 for m in orderings + ["ensemble"]]
        m, lo, hi = (np.array([s[j] for s in stats]) for j in range(3))
        x = np.arange(len(names))
        n_ord = len(orderings)
        if color_by_upstream:
            cats = [upstream_category(o, label) for o in orderings]
            for cat, (color, marker, hollow, name) in UPSTREAM_STYLE.items():
                idx = [j for j, c in enumerate(cats) if c == cat]
                if idx:
                    ps.ci_point(ax, x[idx], m[idx], lo[idx], hi[idx], color=color, marker=marker, hollow=hollow, label=name)
        else:
            ps.ci_point(ax, x[:n_ord], m[:n_ord], lo[:n_ord], hi[:n_ord], color=ps.ROLE["best"], label="Chain ordering")
        ps.ci_point(ax, x[n_ord:], m[n_ord:], lo[n_ord:], hi[n_ord:], color=ps.ROLE["ensemble"], label=ps.ROLE_NAME["ensemble"])
        br = summarize(method_values(cv_results, "binary_relevance", label, metric_name), interval_fn, error_mode, k, confidence)
        if np.isfinite(br[0]):
            ps.hline(ax, br[0], color=ps.ROLE["br"], lw=0.9)
            if br_handle is None:
                br_handle = Line2D([0], [0], color=ps.ROLE["br"], lw=0.9, label=ps.ROLE_NAME["br"])
        _metric_ylim(ax, metric_name, list(lo) + [br[0]], list(hi) + [br[0]], head=0.22)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(names) - 0.4)
        if i == 0:
            ax.set_ylabel(metric_name)
        _drug_text(ax, label)
        ps.panel_label(ax, letters[i])
        ps.despine(ax)
    if own:
        fig.tight_layout(w_pad=1.5)
    seen = {}
    for ax in axes:   # union over panels (the CIP panel has no "CIP upstream" category), first occurrence wins
        for h, l in zip(*ax.get_legend_handles_labels()): seen.setdefault(l, h)
    order = [v[3] for v in UPSTREAM_STYLE.values()] + [ps.ROLE_NAME["ensemble"], "Chain ordering"]
    lab = sorted(seen, key=lambda l: order.index(l) if l in order else len(order)); handles = [seen[l] for l in lab]
    if br_handle is not None:
        handles.append(br_handle); lab.append(br_handle.get_label())
    ps.legend_above(fig, handles, lab, y=1.0)
    if own and output_path:
        save_fig(fig, output_path, dpi)
    return fig


def plot_cross_model_comparison(
    all_results: Dict[str, object],
    labels: List[str],
    metric_name: str = "AUROC",
    method: str = "ensemble",
    output_path: Path = None,
    dpi: int = 300,
    error_mode: str = "sd",
    interval_fn=None,
):
    """One method across learners: x = drugs, hue = learner (ps.LEARNER), points with intervals.

    Parameters
    ----------
    all_results : dict
        Maps model name -> CVResults object (e.g. {"logistic": cv, "catboost": cv})
    """
    setup_style()
    models = list(all_results.keys())
    fig, ax = ps.figure("single", height=2.5)
    x = np.arange(len(labels))
    offsets = np.linspace(-0.22, 0.22, len(models)) if len(models) > 1 else [0.0]
    lows, highs = [], []
    for model, off in zip(models, offsets):
        cv = all_results[model]
        k = cv.config.n_splits_outer if cv.config else None
        stats = [summarize(method_values(cv, method, _resolve_label(l, cv.labels), metric_name), interval_fn, error_mode, k)
                 for l in labels]
        m, lo, hi = (np.array([s[i] for s in stats]) for i in range(3))
        ps.ci_point(ax, x + off, m, lo, hi, color=ps.LEARNER.get(model, ps.INK), label=ps.LEARNER_NAME.get(model, model))
        lows += list(lo); highs += list(hi)
    _metric_ylim(ax, metric_name, lows, highs)
    ax.set_xticks(x)
    ax.set_xticklabels([_DRUG_DISPLAY.get(l, l) for l in labels])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    method_display = {"binary_relevance": ps.ROLE_NAME["br"], "ensemble": ps.ROLE_NAME["ensemble"]}.get(method, method.replace("_", " "))
    ax.set_ylabel(f"{metric_name} ({method_display.lower()})")
    ps.despine(ax)
    fig.tight_layout()
    ps.legend_above(fig, y=1.0)
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig


def plot_cross_dataset_comparison(
    results_dict: Dict[str, object],
    labels: List[str],
    metric_name: str = "AUROC",
    method: str = "ensemble",
    output_path: Path = None,
    dpi: int = 300,
    error_mode: str = "sd",
):
    """Plot performance comparison across multiple datasets.

    Parameters
    ----------
    results_dict : dict
        Maps dataset name -> CVResults object
    labels : list
        Canonical label names (may include "CTX/CRO"); resolved per dataset.
    """
    setup_style()

    datasets = list(results_dict.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(3.5 * len(labels), 3.5))
    if len(labels) == 1:
        axes = [axes]

    palette_keys = [f"dataset_{d}" for d in datasets]
    colors = [PALETTE.get(k, "#888888") for k in palette_keys]

    for idx, label in enumerate(labels):
        ax = axes[idx]
        x = np.arange(len(datasets))
        means = []
        cis_list = []

        for ds_name, cv_results in results_dict.items():
            k = cv_results.config.n_splits_outer if cv_results.config else None
            method_results = [r for r in cv_results.fold_results if r.method == method]
            ds_labels = {l for r in method_results for l in r.metrics}
            actual = _resolve_label(label, ds_labels)
            values = [r.metrics[actual][metric_name] for r in method_results if actual in r.metrics]
            values = np.array([v for v in values if not np.isnan(v)])
            means.append(np.mean(values) if len(values) > 0 else 0)
            cis_list.append(error_bar_halfwidth(np.std(values, ddof=1), len(values), mode=error_mode, k=k) if len(values) > 1 else 0)

        ax.bar(x, means, yerr=cis_list, capsize=5, color=colors, alpha=0.8, width=0.6)
        ax.set_ylabel(metric_name)
        ax.set_title(_DRUG_DISPLAY.get(label, label), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([get_display_name(d) for d in datasets], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"{metric_name}: Cross-Dataset Comparison ({method})", fontsize=10, y=1.02)
    eb_label = "Error bars: SD" if error_mode == "sd" else "Error bars: 95% CI (Nadeau & Bengio)"
    fig.text(0.99, 0.01, eb_label, ha="right", va="bottom", fontsize=6, fontstyle="italic", color="gray")
    plt.tight_layout()
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig


def plot_cross_dataset_heatmap(
    results_json_path,
    labels: List[str],
    method: str = "ensemble",
    output_path: Path = None,
    dpi: int = 300,
):
    """Plot 5×5 AUROC heatmap from cross-dataset JSON results.

    Parameters
    ----------
    results_json_path : str or Path
        Path to cross_dataset_results.json
    labels : list
        Drug labels (e.g. ["CIP", "CRO", "CAZ"])
    method : str
        "binary_relevance" or "ensemble"
    """
    setup_style()

    with open(results_json_path) as f:
        results = json.load(f)

    sites = sorted(set(r["train"] for r in results) | set(r["test"] for r in results))
    site_labels = [get_display_name(s) for s in sites]
    n_sites = len(sites)

    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 3.8))
    if len(labels) == 1:
        axes = [axes]

    for idx, label in enumerate(labels):
        ax = axes[idx]
        matrix = np.full((n_sites, n_sites), np.nan)

        for r in results:
            i = sites.index(r["train"])
            j = sites.index(r["test"])
            auc = r["methods"].get(method, {}).get(label, {}).get("AUROC")
            if auc is not None:
                matrix[i, j] = auc

        # Build annotation strings: "0.63" or "-" for diagonal
        annot = np.empty((n_sites, n_sites), dtype=object)
        for i in range(n_sites):
            for j in range(n_sites):
                if i == j:
                    annot[i, j] = "-"
                elif np.isnan(matrix[i, j]):
                    annot[i, j] = ""
                else:
                    annot[i, j] = f"{matrix[i, j]:.2f}"

        sns.heatmap(
            matrix, ax=ax, annot=annot, fmt="",
            xticklabels=site_labels, yticklabels=site_labels,
            cmap="RdYlGn", vmin=0.5, vmax=1.0,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "AUROC"},
        )
        ax.set_title(_DRUG_DISPLAY.get(label, label), fontweight="bold")
        ax.set_xlabel("Test site")
        ax.set_ylabel("Train site")
        ax.tick_params(axis="both", labelsize=7)

    method_display = method.replace("_", " ").title()
    plt.suptitle(f"Cross-Dataset AUROC ({method_display})", fontsize=10, y=1.02)
    plt.tight_layout()
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig


def plot_cross_dataset_delta_heatmap(
    results_json_path,
    labels: List[str],
    output_path: Path = None,
    dpi: int = 300,
):
    """Plot 5×5 delta AUROC heatmap (ensemble − BR) from cross-dataset JSON.

    Symmetric diverging colormap centered at 0: green = chain helps, red = chain hurts.
    """
    setup_style()

    with open(results_json_path) as f:
        results = json.load(f)

    sites = sorted(set(r["train"] for r in results) | set(r["test"] for r in results))
    site_labels = [get_display_name(s) for s in sites]
    n_sites = len(sites)

    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 3.8))
    if len(labels) == 1:
        axes = [axes]

    all_deltas = []
    matrices = {}
    for label in labels:
        matrix = np.full((n_sites, n_sites), np.nan)
        for r in results:
            i = sites.index(r["train"])
            j = sites.index(r["test"])
            br_auc = r["methods"].get("binary_relevance", {}).get(label, {}).get("AUROC")
            ens_auc = r["methods"].get("ensemble", {}).get(label, {}).get("AUROC")
            if br_auc is not None and ens_auc is not None:
                delta = ens_auc - br_auc
                matrix[i, j] = delta
                all_deltas.append(delta)
        matrices[label] = matrix

    vmax = max(abs(d) for d in all_deltas) if all_deltas else 0.05
    vmax = max(vmax, 0.01)

    for idx, label in enumerate(labels):
        ax = axes[idx]
        matrix = matrices[label]

        annot = np.empty((n_sites, n_sites), dtype=object)
        for i in range(n_sites):
            for j in range(n_sites):
                if i == j:
                    annot[i, j] = "-"
                elif np.isnan(matrix[i, j]):
                    annot[i, j] = ""
                else:
                    annot[i, j] = f"{matrix[i, j]:+.3f}"

        sns.heatmap(
            matrix, ax=ax, annot=annot, fmt="",
            xticklabels=site_labels, yticklabels=site_labels,
            cmap="RdYlGn", vmin=-vmax, vmax=vmax, center=0,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "\u0394 AUROC"},
        )
        ax.set_title(_DRUG_DISPLAY.get(label, label), fontweight="bold")
        ax.set_xlabel("Test site")
        ax.set_ylabel("Train site")
        ax.tick_params(axis="both", labelsize=7)

    plt.suptitle("Cross-Dataset \u0394 AUROC (Ensemble \u2212 BR)", fontsize=10, y=1.02)
    plt.tight_layout()
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig


def plot_cross_dataset_combined_figure(
    results_json_path,
    labels: List[str],
    output_path: Path = None,
    dpi: int = 300,
    stacking_json_path=None,
):
    """Figure 6: grid of 5 x 5 transfer heatmaps (rows = training site, columns = test site).

    Top row (a): chain-ensemble AUROC, SEQ_CMAP from 0.5 to 0.9. Row (b): gain over binary relevance
    (ensemble minus BR), DIV_CMAP centred at zero with symmetric limits. With stacking_json_path, row (c): gain of
    cross-fitted stacking over the same binary relevance, on the scale of (b). One colorbar per row, the drug written
    as small text above each panel, sites ordered by cohort size, diagonal (train = test) left white.
    """
    ps.apply()
    with open(results_json_path) as f:
        results = json.load(f)
    stack_results = json.load(open(stacking_json_path)) if stacking_json_path else None

    canon = {"local": "s_orsola"}   # old result files name the Sant'Orsola cohort "local"
    for r in results + (stack_results or []):
        r["train"], r["test"] = canon.get(r["train"], r["train"]), canon.get(r["test"], r["test"])
    present = set(r["train"] for r in results) | set(r["test"] for r in results)
    sites = [s for s in ps.SITE_ORDER if s in present] + sorted(s for s in present if s not in ps.SITE_ORDER)
    site_labels = [ps.SITE_NAME.get(s, s) for s in sites]
    n_sites, n_drugs = len(sites), len(labels)
    drug_display = {"CRO": "CTX/CRO", "CTX": "CTX/CRO"}

    def matrix(method_fn, res=None):
        m = np.full((n_sites, n_sites), np.nan)
        for r in (results if res is None else res):
            i, j = sites.index(r["train"]), sites.index(r["test"])
            v = method_fn(r)
            if v is not None:
                m[i, j] = v
        return m

    def _get(r, method, label):
        return r["methods"].get(method, {}).get(label, {}).get("AUROC")

    auroc = {l: matrix(lambda r, l=l: _get(r, "ensemble", l)) for l in labels}
    delta = {}
    for l in labels:
        def _d(r, l=l):
            a, b = _get(r, "ensemble", l), _get(r, "binary_relevance", l)
            return None if a is None or b is None else a - b
        delta[l] = matrix(_d)
    sdelta = {}
    if stack_results is not None:
        for l in labels:
            def _s(r, l=l):
                a, b = _get(r, "stacking", l), _get(r, "binary_relevance", l)
                return None if a is None or b is None else a - b
            sdelta[l] = matrix(_s, stack_results)
    dmax = np.nanmax([np.nanmax(np.abs(m)) for m in list(delta.values()) + list(sdelta.values())]) if delta else 0.05
    dmax = max(float(np.ceil(dmax * 100) / 100), 0.01)   # symmetric limits rounded up to 0.01

    rows = [("a", auroc, ps.SEQ_CMAP, 0.5, 0.9, "{:.2f}", "AUROC (chain ensemble)", [0.5, 0.6, 0.7, 0.8, 0.9]),
            ("b", delta, ps.DIV_CMAP, -dmax, dmax, "{:+.3f}", "Chain ensemble gain\nover BR (AUROC)", None)]
    if sdelta:
        rows.append(("c", sdelta, ps.DIV_CMAP, -dmax, dmax, "{:+.3f}", "Stacking gain\nover BR (AUROC)", None))
    fig = plt.figure(figsize=(ps.WIDTH["double"], 2.45 * len(rows)))
    gs = fig.add_gridspec(len(rows), n_drugs + 1, width_ratios=[1] * n_drugs + [0.045], wspace=0.12, hspace=0.22,
                          left=0.11, right=0.95, top=0.95, bottom=0.11 if len(rows) == 2 else 0.075)
    for row, (letter, mats, cmap, vmin, vmax, fmt, cb_label, ticks) in enumerate(rows):
        cmap = cmap.copy(); cmap.set_bad(ps.SURFACE)
        im = None
        for col, label in enumerate(labels):
            ax = fig.add_subplot(gs[row, col])
            m = mats[label].copy()
            np.fill_diagonal(m, np.nan)   # train = test carries no transfer value
            im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            for i in range(n_sites):
                for j in range(n_sites):
                    v = m[i, j]
                    if np.isnan(v):
                        continue
                    t = (v - vmin) / (vmax - vmin)
                    txt = "0.000" if (letter != "a" and abs(v) < 5e-4) else fmt.format(v).replace("-", "\u2212")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                            color=ps.text_on(cmap(t)))
            ax.set_xticks(range(n_sites)); ax.set_yticks(range(n_sites))
            ax.set_xticks(np.arange(0.5, n_sites - 1, 1), minor=True); ax.set_yticks(np.arange(0.5, n_sites - 1, 1), minor=True)   # internal separators only
            ax.grid(which="minor", color=ps.SURFACE, linewidth=1.2); ax.tick_params(which="both", length=0)
            for s in ax.spines.values(): s.set_visible(False)
            if row == len(rows) - 1:
                ax.set_xticklabels(site_labels, rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlabel("Test site")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_yticklabels(site_labels); ax.set_ylabel("Training site")
                ps.panel_label(ax, letter, dx=-0.02, dy=1.01)
            else:
                ax.set_yticklabels([])
            ax.text(0.5, 1.01, drug_display.get(label, label), transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        cax = fig.add_subplot(gs[row, n_drugs])
        cb = fig.colorbar(im, cax=cax, ticks=ticks)
        cb.set_label(cb_label); cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.6)
        if letter != "a":
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:+.2f}".replace("-", "\u2212")))

    if output_path:
        ps.save(fig, output_path)
    return fig
