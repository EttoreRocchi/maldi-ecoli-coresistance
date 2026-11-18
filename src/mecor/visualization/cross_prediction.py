"""Stacking gain and proxy margin along the learning curves (Fig. 4d) and the scatter of one against the other."""
from __future__ import annotations

import numpy as np

from mecor.visualization import paper_style as ps

SITE_NAME = {"driams_a": "DRIAMS-A", "s_orsola": "Sant'Orsola", "driams_a_grouped": "DRIAMS-A"}


def draw(ax, rows, rho, legend_size=None, compact=False):
    """Stacking gain against the proxy margin on ax, one trajectory per site and label along the training size.
    compact: legend in the empty lower-right quadrant, without the identity entry and the rho text (both in the caption)."""
    from matplotlib.lines import Line2D
    lim = (-0.10, 0.18)
    ax.plot(lim, lim, ls=(0, (1.5, 1.5)), lw=0.8, color=ps.MUTED, zorder=1)
    ps.hline(ax, 0.0); ax.axvline(0.0, color=ps.INK2, lw=0.7, zorder=1)
    style = {"s_orsola": ("--", "o"), "driams_a": ("-", "v"), "driams_a_grouped": ("-", "v")}
    handles = []
    sites = [s for s in ("driams_a", "driams_a_grouped", "s_orsola") if any(r["site"] == s for r in rows)]
    for site, name in ((s, SITE_NAME[s]) for s in sites):
        for lab in dict.fromkeys(r["label"] for r in rows if r["site"] == site):
            rr = [r for r in rows if r["site"] == site and r["label"] == lab]
            x = [r["margin"]["delta"] for r in rr]; y = [r["gain_stack"]["delta"] for r in rr]; c = ps.DRUG[lab]; ls, mk = style[site]
            ax.plot(x, y, color=c, ls=ls, lw=1.0, zorder=2)
            ax.plot(x[:-1], y[:-1], ls="none", marker=mk, ms=4.5, color=c, zorder=3)
            ax.plot(x[-1:], y[-1:], ls="none", marker=mk, ms=5.5, mfc=ps.SURFACE, mec=c, mew=1.1, zorder=3)   # largest size (full site at DRIAMS-A)
            handles.append(Line2D([], [], color=c, ls=ls, lw=1.0, marker=mk, ms=4.5, label=f"{name}, {lab}"))
    if not compact: handles.append(Line2D([], [], color=ps.MUTED, ls=(0, (1.5, 1.5)), lw=0.8, label="Stacking gain = margin"))
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xticks([-0.1, 0.0, 0.1]); ax.set_yticks([-0.1, 0.0, 0.1])
    ax.set_xlabel("Proxy margin (AUROC)"); ax.set_ylabel("Stacking gain (AUROC)" if compact else "Stacking gain over binary relevance (AUROC)")
    ps.despine(ax, grid=None)
    ax.legend(handles=handles, loc="lower right" if compact else "upper left", borderaxespad=0.3, fontsize=legend_size)
    if not compact:   # in the compact (main-figure) version rho is given in the caption
        ax.text(0.98, 0.03, f"Spearman $\\rho$ = {rho:+.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=legend_size or 7, color=ps.INK2)


def draw_size_view(ax_so, ax_a, rows, legend_size=None):
    """Stacking gain (solid, filled, BCa 95% band) and proxy margin (dashed, hollow) against the training size, one pair
    of lines per label: Sant'Orsola subsamples on ax_so (linear axis), DRIAMS-A subsamples and the full site on ax_a (log
    axis; the full-site gain has no interval). Drug colors only, so no role color shares the panel."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    full_n = {"driams_a": 4761, "s_orsola": 187}
    for ax, site, name in ((ax_so, "s_orsola", "Sant'Orsola"), (ax_a, "driams_a", "DRIAMS-A")):
        ps.hline(ax, 0.0)
        for lab in dict.fromkeys(r["label"] for r in rows if r["site"] == site):
            rr = sorted([r for r in rows if r["site"] == site and r["label"] == lab], key=lambda r: full_n[site] if r["n"] == "full" else r["n"])
            n = np.array([full_n[site] if r["n"] == "full" else r["n"] for r in rr], float); c = ps.DRUG[lab]
            g = np.array([r["gain_stack"]["delta"] for r in rr]); m = np.array([r["margin"]["delta"] for r in rr])
            lo = np.array([r["gain_stack"]["ci"][0] for r in rr], float); hi = np.array([r["gain_stack"]["ci"][1] for r in rr], float); ok = np.isfinite(lo)
            ax.fill_between(n[ok], lo[ok], hi[ok], color=c, alpha=0.12, lw=0, zorder=1)
            ax.plot(n, g, color=c, lw=1.2, marker="o", ms=3.4, zorder=3)
            ax.plot(n, m, color=c, lw=1.0, ls="--", marker="o", ms=3.4, mfc=ps.SURFACE, mec=c, mew=0.9, zorder=3)
        right = site == "driams_a"   # top-left at Sant'Orsola, top-right at DRIAMS-A (where the curves are low)
        ax.text(0.96 if right else 0.04, 0.97, name, transform=ax.transAxes, ha="right" if right else "left", va="top", fontsize=7, color=ps.INK)
        ps.despine(ax, grid="y")
    ax_so.set_xlim(52, 195); ax_so.set_xticks([60, 90, 120, 150, 187])
    ax_so.axvspan(52, 100, color=ps.GRID, alpha=0.7, lw=0, zorder=0)
    ax_so.text(76, 0.175, "fewer than\n100 isolates", ha="center", va="top", fontsize=5.8, color=ps.INK2, linespacing=1.1)
    ax_a.set_xscale("log"); ax_a.set_xlim(150, 6200); ax_a.minorticks_off()
    ax_a.set_xticks([187, 500, 1000, 2000, 4761]); ax_a.set_xticklabels(["187", "500", "1,000", "2,000", "4,761"], fontsize=6)
    ax_a.tick_params(labelleft=False)
    for ax in (ax_so, ax_a): ax.set_ylim(-0.09, 0.22)
    h_lab = [Line2D([], [], color=ps.DRUG[k], lw=1.2, label=l) for k, l in (("CTX", "CTX or CRO"), ("CAZ", "CAZ"))]
    h_sty = [Line2D([], [], color=ps.INK2, lw=1.2, marker="o", ms=3.4, label="Stacking gain"),
             Line2D([], [], color=ps.INK2, lw=1.0, ls="--", marker="o", ms=3.4, mfc=ps.SURFACE, mew=0.9, label="Proxy margin"),
             Patch(color=ps.INK2, alpha=0.15, lw=0, label="95% interval of the gain")]
    # one legend above the two sub-panels (columns: labels, line styles, band)
    ax_so.legend(handles=h_lab + h_sty, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=legend_size, borderaxespad=0.0,
                 handlelength=1.8, columnspacing=1.0)
