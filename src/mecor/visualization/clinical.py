"""Clinical evaluation visualization: VME/ME threshold curves and method comparison (paper style)."""

from pathlib import Path
from typing import Dict, List

import numpy as np
from matplotlib.lines import Line2D

from mecor.visualization import paper_style as ps
from mecor.visualization.style import save_fig, setup_style

_METRICS = (("VME", "vme", "VME (missed resistance)"), ("ME", "me", "ME (false resistance)"))
_SERIES = (("br", ps.ROLE["br"], ps.ROLE_NAME["br"]),
           ("ensemble", ps.ROLE["ensemble"], ps.ROLE_NAME["ensemble"]),
           ("best", ps.ROLE["best"], ps.ROLE_NAME["best"]),
           ("stacking", ps.ROLE["stacking"], "Stacked model"))


def plot_vme_me_curves(
    clinical_results: Dict,
    labels: List[str],
    output_path: Path = None,
    dpi: int = 300,
):
    """VME and ME against the classification threshold, one panel.

    Hue = drug (ps.DRUG), line style = error type (VME solid, ME dashed), a thin
    vertical mark at the default 0.5 threshold, one legend above the panel.

    Parameters
    ----------
    clinical_results : dict
        Output of clinical_evaluation(), mapping label -> analysis dict
    """
    setup_style()
    fig, ax = ps.figure("single", height=2.6)
    drawn = []
    for label in labels:
        ta = clinical_results.get(label, {}).get("threshold_analysis")
        if ta is None:
            continue
        color = ps.DRUG.get(label, ps.INK)
        ax.plot(ta["thresholds"], ta["vme"], color=color, lw=1.2, ls="-")
        ax.plot(ta["thresholds"], ta["me"], color=color, lw=1.2, ls=(0, (3, 2)))
        drawn.append(label)
    ax.axvline(0.5, color=ps.LIGHT, lw=0.7, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Error rate")
    ps.despine(ax)
    handles = [Line2D([0], [0], color=ps.DRUG.get(l, ps.INK), lw=1.2, label=l) for l in drawn]
    handles += [Line2D([0], [0], color=ps.INK2, lw=1.2, ls="-", label="VME"),
                Line2D([0], [0], color=ps.INK2, lw=1.2, ls=(0, (3, 2)), label="ME")]
    fig.tight_layout()
    ps.legend_above(fig, handles, [h.get_label() for h in handles], y=1.0)
    if output_path:
        save_fig(fig, output_path, dpi)
    return fig


def _intervals_from_summaries(clinical_br, clinical_chain, clinical_best, labels):
    """Fallback intervals (mean +/- SD over folds) from clinical_evaluation() summaries."""
    out = {}
    for key, summ in (("br", clinical_br), ("ensemble", clinical_chain), ("best", clinical_best)):
        out[key] = {}
        for label in labels:
            s = summ.get(label, {}) if summ else {}
            out[key][label] = {}
            for metric, stem, _ in _METRICS:
                m = float(s.get(f"{stem}_mean", np.nan))
                sd = float(s.get(f"{stem}_std", 0.0) or 0.0)
                out[key][label][metric] = (m, m - sd, m + sd)
    return out


def plot_vme_me_comparison(
    clinical_br: Dict = None,
    clinical_chain: Dict = None,
    clinical_best: Dict = None,
    labels: List[str] = None,
    output_path: Path = None,
    dpi: int = 300,
    intervals: Dict = None,
    fig=None,
    axes=None,
    letters: str = "ab",
):
    """VME and ME of binary relevance, chain ensemble and best chain as points with intervals.

    Two panels (VME, ME), x = drugs, the three methods as dodged point series in the
    fixed role colors, one legend above the panels.

    intervals : {"br" | "ensemble" | "best": {label: {"VME" | "ME": (mean, low, high)}}}
        Precomputed point estimates and interval bounds (for example BCa bootstrap over
        the folds). When omitted, mean +/- SD over folds from the clinical summaries.
    """
    setup_style()
    labels = list(labels)
    if intervals is None:
        intervals = _intervals_from_summaries(clinical_br, clinical_chain, clinical_best, labels)

    own = axes is None
    if own:
        fig, axes = ps.figure("onehalf", height=2.5, ncols=2)
    x = np.arange(len(labels))
    series = [sr for sr in _SERIES if sr[0] in intervals] if intervals is not None else list(_SERIES)
    offsets = np.linspace(-0.27, 0.27, len(series)) if len(series) > 3 else np.linspace(-0.22, 0.22, len(series))
    all_vals = []
    for i, (ax, (metric, _, ylabel)) in enumerate(zip(axes, _METRICS)):
        lows, highs = [], []
        for (key, color, name), off in zip(series, offsets):
            if key not in intervals: continue   # a series without values (e.g. no stacked model) is skipped
            trip = [intervals.get(key, {}).get(l, {}).get(metric, (np.nan, np.nan, np.nan)) for l in labels]
            m, lo, hi = (np.array([t[j] for t in trip], float) for j in range(3))
            ps.ci_point(ax, x + off, m, lo, hi, color=color, label=name)
            lows += list(lo); highs += list(hi)
        all_vals.extend(v for v in lows + highs if np.isfinite(v))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylabel(ylabel)
        ps.panel_label(ax, letters[i])
        ps.despine(ax)
    top = 0.5 if max(all_vals, default=0) <= 0.5 else np.ceil(max(all_vals) * 10) / 10
    for ax in axes: ax.set_ylim(0.0, top)   # the two error-rate panels share one scale (0 to 0.5 unless a value exceeds it)
    if own:
        fig.tight_layout(w_pad=2.0)
    ps.legend_above(fig, y=1.0)
    if own and output_path:
        save_fig(fig, output_path, dpi)
    return fig
