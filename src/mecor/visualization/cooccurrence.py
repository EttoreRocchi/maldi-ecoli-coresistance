"""Co-occurrence visualization: phi matrices and UpSet plots (shared paper style)."""

from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mecor.visualization import paper_style as ps

# The third-generation cephalosporin is CTX at Sant'Orsola and CRO at the DRIAMS sites; one label for both.
CEPH = ("CTX", "CRO")
CEPH_LABEL = "CTX/CRO"
# neutral ramp for the conditional co-resistance column (a probability, kept apart from the phi scale)
COND_CMAP = ps.LinearSegmentedColormap.from_list("cond_greys", ["#f6f6f4", "#d9d8d3", "#a9a7a0", "#6e6c66", "#3b3a37"])
# shading of the UpSet intersection bars by the number of resistances in the profile (0 to 3)
N_RES_SHADE = ["#b5b3ad", "#8a8883", "#5f5e5a", ps.INK]


def _pair_label(a: str, b: str) -> str:
    """Two-line column label for a drug pair, with the cephalosporin written as CTX/CRO."""
    a = CEPH_LABEL if a in CEPH else a
    b = CEPH_LABEL if b in CEPH else b
    return f"{a} with\n{b}"


def _site_rows(names: List[str]) -> List[str]:
    """Sites ordered by cohort size (paper_style.SITE_ORDER), unknown names appended."""
    known = [s for s in ps.SITE_ORDER if s in names]
    return known + [s for s in names if s not in known]


def _to_frame(phi) -> pd.DataFrame:
    return phi if isinstance(phi, pd.DataFrame) else pd.DataFrame(phi)


def _white_frame(ax, n_rows, n_cols, lw=1.5):
    """Cover the outer edges of an imshow grid with unclipped surface lines (the raster can bleed one pixel past the
    clipped minor grid at the axes boundary)."""
    kw = dict(color=ps.SURFACE, lw=lw, clip_on=False, zorder=5, solid_capstyle="projecting")
    ax.plot([-0.5, -0.5], [-0.5, n_rows - 0.5], **kw); ax.plot([n_cols - 0.5, n_cols - 0.5], [-0.5, n_rows - 0.5], **kw)
    ax.plot([-0.5, n_cols - 0.5], [-0.5, -0.5], **kw); ax.plot([-0.5, n_cols - 0.5], [n_rows - 0.5, n_rows - 0.5], **kw)
    ax.set_xlim(-0.5, n_cols - 0.5); ax.set_ylim(n_rows - 0.5, -0.5)


def _phi_cell_text(ax, i, j, v, fmt="{:.3f}", fontsize=7):
    """Print a phi value in a cell; white or ink, whichever contrasts more with the SEQ_CMAP cell colour."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return
    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
            color=ps.text_on(ps.SEQ_CMAP(v)))


def plot_phi_heatmap(
    phi_matrix: pd.DataFrame,
    title: str = "",
    output_path: Path = None,
    dpi: int = 300,
    ax=None,
):
    """Lower-triangle phi matrix for one site (SEQ_CMAP from 0 to 1, values printed; title ignored)."""
    ps.apply()
    phi = _to_frame(phi_matrix).astype(float)
    k = len(phi)
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.55 * k + 1.0, 0.55 * k + 0.6))
    else:
        fig = ax.figure
    vals = phi.values.copy()
    vals[np.triu_indices(k)] = np.nan  # the diagonal and the upper triangle carry no information
    cmap = ps.SEQ_CMAP.copy(); cmap.set_bad(ps.SURFACE)
    ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=0, vmax=1, aspect="equal")
    for i in range(k):
        for j in range(i):
            _phi_cell_text(ax, i, j, vals[i, j])
    labels = [CEPH_LABEL if c in CEPH else c for c in phi.columns]
    ax.set_xticks(range(k)); ax.set_xticklabels(labels)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True); ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color=ps.SURFACE, linewidth=1.5); ax.tick_params(which="both", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    if output_path:
        ps.save(fig, output_path)
    return fig


def plot_phi_heatmaps_multi(
    analyses: Dict[str, Dict],
    output_path: Path = None,
    dpi: int = 300,
    ax=None,
    conditional: Optional[Dict[str, float]] = None,
):
    """Figure 1: one compact matrix of phi coefficients, rows = sites (ordered by cohort size), columns = the
    three drug pairs, SEQ_CMAP from 0 to 1 with the value printed in every cell.

    Parameters
    ----------
    analyses : dict
        Maps dataset name -> dict with at least "phi_matrix" (DataFrame or dict of dicts) and "n_samples".
    conditional : dict, optional
        Maps dataset name -> P(CIP-R | 3GC-R); drawn as a separate column to the right on its own neutral scale
        (values printed, no colorbar), because unlike phi it does not depend on the prevalences.
    """
    ps.apply()
    sites = _site_rows(list(analyses))
    pairs_display = None
    matrix = np.full((len(sites), 3), np.nan)
    for r, s in enumerate(sites):
        phi = _to_frame(analyses[s]["phi_matrix"]).astype(float)
        ceph = next(c for c in phi.columns if c in CEPH)
        pairs = [("CIP", ceph), ("CIP", "CAZ"), (ceph, "CAZ")]
        if pairs_display is None:
            pairs_display = [_pair_label(a, b) for a, b in pairs]
        for c, (a, b) in enumerate(pairs):
            matrix[r, c] = phi.loc[a, b]

    n_rows = len(sites)
    own = ax is None
    if own:
        fig, ax = ps.figure("single", height=0.42 * n_rows + 0.75)
    else:
        fig = ax.figure
    cmap = ps.SEQ_CMAP.copy(); cmap.set_bad(ps.SURFACE)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n_rows):
        for j in range(3):
            _phi_cell_text(ax, i, j, matrix[i, j])
    ax.set_xticks(range(3)); ax.set_xticklabels(pairs_display)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{ps.SITE_NAME.get(s, s)} (n = {int(analyses[s]['n_samples']):,})" for s in sites])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True); ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=ps.SURFACE, linewidth=1.5); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    _white_frame(ax, n_rows, 3)
    if conditional is not None:   # separate column, neutral ramp, header on top like the phi columns
        axc = ax.inset_axes([1.06, 0.0, 1.0 / 3.0, 1.0])
        col = np.array([[conditional.get(s, np.nan)] for s in sites])
        axc.imshow(col, cmap=COND_CMAP, vmin=0, vmax=1, aspect="auto")
        for i in range(n_rows):
            v = col[i, 0]
            if not np.isnan(v):
                axc.text(0, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=ps.text_on(COND_CMAP(v)))
        axc.set_xticks([0]); axc.set_xticklabels(["P(CIP-R |\n3GC-R)"]); axc.set_yticks([])
        axc.tick_params(axis="x", labeltop=True, labelbottom=False, length=0)
        axc.set_yticks(np.arange(-0.5, n_rows, 1), minor=True); axc.grid(which="minor", axis="y", color=ps.SURFACE, linewidth=1.5)
        axc.tick_params(which="minor", length=0)
        for sp in axc.spines.values(): sp.set_visible(False)
        _white_frame(axc, n_rows, 1)
        cax = ax.inset_axes([1.06 + 1.0 / 3.0 + 0.10, 0.12, 0.045, 0.76])
        cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    else:
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Phi coefficient"); cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.6)
    if own:
        fig.tight_layout()
        if output_path:
            ps.save(fig, output_path)
    return fig


def plot_resistance_profiles(
    profile_counts: pd.Series,
    title: str = "",
    output_path: Path = None,
    dpi: int = 300,
):
    """Resistance profile frequencies as a horizontal bar chart (counts, neutral bars, no title)."""
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.WIDTH["single"], max(1.6, len(profile_counts) * 0.22 + 0.5)))
    y_pos = np.arange(len(profile_counts))
    ax.barh(y_pos, profile_counts.values, height=0.6, color=ps.ROLE["br"])
    for y, v in zip(y_pos, profile_counts.values):
        ax.text(v, y, f" {int(v):,}", va="center", ha="left", fontsize=7, color=ps.INK)
    ax.set_yticks(y_pos); ax.set_yticklabels(profile_counts.index)
    ax.set_xlabel("Isolates"); ax.invert_yaxis()
    ax.set_xlim(0, profile_counts.values.max() * 1.15)
    ps.despine(ax, grid="x")
    fig.tight_layout()
    if output_path:
        ps.save(fig, output_path)
    return fig


def profile_counts_from_Y(Y: pd.DataFrame) -> Dict[str, int]:
    """Counts of the 2^k joint profiles, keyed by an R/S string in the column order of Y (e.g. 'RSS')."""
    Yb = Y.astype(int)
    counts = {}
    for combo in product([0, 1], repeat=Yb.shape[1]):
        key = "".join("R" if v else "S" for v in combo)
        counts[key] = int((Yb.values == np.array(combo)).all(axis=1).sum())
    return counts


def plot_upset_from_counts(
    counts: Dict[str, int],
    labels: List[str],
    output_path: Path = None,
    dpi: int = 300,
    fig=None,
):
    """UpSet plot drawn from the joint profile counts (no upsetplot dependency).

    Columns are the joint profiles sorted by count (all 2^k profiles, zero counts included so that profiles
    absent by construction are visible); the bars are neutral and thin, every bar carries its count (the figure
    is a count table), the matrix uses INK for resistant and LIGHT for susceptible, the set sizes are the
    resistant isolates per drug.
    """
    ps.apply()
    k = len(labels)
    profiles = sorted(counts, key=lambda p: (-counts[p], p))
    n_prof = len(profiles)
    set_sizes = [sum(v for p, v in counts.items() if p[i] == "R") for i in range(k)]
    disp = [CEPH_LABEL if l in CEPH else l for l in labels]

    own = fig is None
    if own:
        fig = plt.figure(figsize=(ps.WIDTH["single"], 2.0 + 0.22 * k))
    # columns: set-size bars | spacer holding the drug labels | intersection bars over the membership matrix
    label_w = 0.3 + 0.09 * max(len(str(d)) for d in disp)   # spacer wide enough for the longest drug label
    gs = fig.add_gridspec(2, 3, width_ratios=[0.8, label_w, 2.6], height_ratios=[2.6, 0.32 * k + 0.1],
                          wspace=0.0, hspace=0.06)
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_set = fig.add_subplot(gs[1, 0])
    ax_mat = fig.add_subplot(gs[1, 2], sharex=ax_bar)

    # intersection bars with counts above
    x = np.arange(n_prof)
    vals = np.array([counts[p] for p in profiles], float)
    n_res = [p.count("R") for p in profiles]; total = vals.sum()
    ax_bar.bar(x, vals, width=0.55, color=[N_RES_SHADE[min(r, 3)] for r in n_res], zorder=2)
    top = vals.max() * 1.30
    for xi, v in zip(x, vals):   # count, and its share of the cohort underneath
        ax_bar.text(xi, v + top * 0.015, f"{int(v):,}\n{100 * v / total:.0f}%", ha="center", va="bottom", fontsize=6.5,
                    color=ps.INK, linespacing=1.15)
    from matplotlib.patches import Patch
    ax_bar.legend(handles=[Patch(color=N_RES_SHADE[r], label=str(r)) for r in range(min(k, 3) + 1)], title="Resistances",
                  loc="upper right", ncol=4, fontsize=6.5, title_fontsize=6.5, handlelength=0.9, handletextpad=0.3, columnspacing=0.7,
                  borderaxespad=0.1)
    ax_bar.set_ylim(0, top); ax_bar.set_xlim(-0.6, n_prof - 0.4)
    ax_bar.set_ylabel("Isolates")
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.spines["bottom"].set_visible(False)
    ps.despine(ax_bar)
    ax_bar.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))

    # membership matrix: INK dots for resistant, LIGHT for susceptible, a line joins the members of a profile
    for j, p in enumerate(profiles):
        members = [i for i in range(k) if p[i] == "R"]
        absent = [i for i in range(k) if p[i] == "S"]
        if absent:
            ax_mat.plot([j] * len(absent), absent, "o", ms=5.5, color=ps.LIGHT, zorder=2)
        if len(members) > 1:
            ax_mat.plot([j, j], [min(members), max(members)], color=ps.INK, lw=1.2, zorder=3)
        if members:
            ax_mat.plot([j] * len(members), members, "o", ms=5.5, color=ps.INK, zorder=4)
    ax_mat.set_ylim(k - 0.5, -0.5)
    ax_mat.set_yticks(range(k)); ax_mat.set_yticklabels(disp)
    ax_mat.tick_params(axis="both", length=0, labelbottom=False)
    for s in ax_mat.spines.values(): s.set_visible(False)

    # set sizes (resistant isolates per drug), neutral bars pointing left, count at the bar end
    ax_set.barh(range(k), set_sizes, height=0.55, color=ps.MUTED, zorder=2)
    smax = max(set_sizes) if set_sizes else 1
    for i, v in enumerate(set_sizes):   # x axis is inverted: ha="right" puts the count beyond the bar end
        ax_set.text(v + smax * 0.04, i, f"{int(v):,}", ha="right", va="center", fontsize=7, color=ps.INK)
    ax_set.set_ylim(k - 0.5, -0.5); ax_set.set_xlim(smax * 1.75, 0)
    ax_set.set_yticks([]); ax_set.set_xlabel("Resistant", fontsize=7)
    ax_set.xaxis.set_major_locator(plt.MaxNLocator(2, integer=True))
    ax_set.tick_params(axis="x", labelsize=6.5, length=2)
    for s in ("top", "right", "left"): ax_set.spines[s].set_visible(False)

    if own and output_path:
        ps.save(fig, output_path)
    return fig


def plot_upset(
    Y: pd.DataFrame,
    output_path: Path = None,
    dpi: int = 300,
):
    """UpSet plot of the resistance set intersections from the label matrix Y (columns = drugs)."""
    return plot_upset_from_counts(profile_counts_from_Y(Y), list(Y.columns), output_path=output_path, dpi=dpi)
