"""Supplementary figures: learning curves at both sites (hard, probability and, when available, stacking) and label
efficiency for all three drugs at both targets. Reads results/stats/learning_curve_*.json and
results/stats/domain_adaptation_*.json only. Outputs figures/supplementary/learning_curves.{png,pdf} and
label_efficiency.{png,pdf}."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.visualization import paper_style as ps  # noqa: E402

OUT = ROOT / "figures/supplementary"
CEPH = "CTX/CRO"   # the cephalosporin column: CTX at Sant'Orsola, CRO at the DRIAMS sites


def handle(color=ps.INK, ls="-", marker="o", hollow=False, label=None, ms=4.5):
    return Line2D([], [], color=color, ls=ls, lw=1.2, marker=marker, ms=ms, mfc=ps.SURFACE if hollow else color, mec=color, mew=1.0, label=label)


# learning curves (2 x 2: site x label)
VARIANTS = (("d_hard", "ensemble", "-", "o", False, "Chain, hard passing"), ("d_proba", "proba", "--", "o", True, "Chain, probability passing"),
            ("d_stack", "stacking", ":", "^", False, "Stacking (cross-fitted)"))
fig, axes = ps.figure("double", height=5.4, nrows=2, ncols=2, sharey=True, constrained_layout=True)
SITES = (("s_orsola", "Sant’Orsola", "CIP \u2192 CTX \u2192 CAZ"), ("driams_a", "DRIAMS-A", "CIP \u2192 CRO \u2192 CAZ"))
LC_FILE = {"driams_a": "learning_curve_driams_a_grouped_stack_inner.json"}   # DRIAMS-A: patient- and case-grouped run (scripts/05_regime/01_learning_curve.py --inner)
for i, (site, name, order) in enumerate(SITES):
    f = ROOT / "results/stats" / LC_FILE.get(site, f"learning_curve_{site}_stack.json")
    rows = json.load(open(f if f.exists() else ROOT / f"results/stats/learning_curve_{site}_stack.json"))
    drugs = sorted({r["drug"] for r in rows}, key=lambda d: 0 if d in ("CTX", "CRO") else 1)
    log = site == "driams_a"
    for j, d in enumerate(drugs):
        ax = axes[i, j]; rr = sorted([r for r in rows if r["drug"] == d], key=lambda r: r["n"]); n = np.array([r["n"] for r in rr], float)
        for (key, role, ls, mk, hollow, _), shift in zip(VARIANTS, ((0.965, 1.0, 1.035) if log else (-3.0, 0.0, 3.0))):
            if not all(key in r for r in rr): continue
            c = ps.ROLE[role]; x = n * shift if log else n + shift
            y = np.array([r[key] for r in rr]); lo = np.array([r[key + "_ci"][0] for r in rr]); hi = np.array([r[key + "_ci"][1] for r in rr])
            ax.plot(x, y, color=c, ls=ls, lw=1.2, zorder=2); ps.ci_point(ax, x, y, lo, hi, color=c, marker=mk, hollow=hollow)
        ps.hline(ax, 0); ps.despine(ax, grid=None)
        ax.set_xlabel("Training isolates")
        if j == 0: ax.set_ylabel("Gain over binary relevance (AUROC)")
        if log:
            ax.set_xscale("log"); ax.set_xticks([187, 500, 1000, 2000]); ax.set_xticklabels(["187", "500", "1,000", "2,000"]); ax.minorticks_off(); ax.set_xlim(160, 2400)
        else:
            ax.set_xticks([60, 90, 120, 150, 187]); ax.set_xlim(50, 197)
        ax.text(0.98, 0.97, f"{name}, {d}\n{order}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=ps.INK2, linespacing=1.3)
        ps.panel_label(ax, "abcd"[2 * i + j])
for ax in axes.flat: ax.set_ylim(-0.12, 0.26)   # all four panels on one scale
handles = [handle(ps.ROLE[role], ls, mk, hollow, lab) for _, role, ls, mk, hollow, lab in VARIANTS]
ps.legend_above(fig, handles, [h.get_label() for h in handles], y=1.02)   # above the panel letters
print("saved", ps.save(fig, OUT / "learning_curves"))

# label efficiency
METHODS = (("local_only", "-", False, -1.0, "Local isolates only"), ("source_plus_local", "--", True, 1.0, "Pooled with DRIAMS-A"))
DRUG_SHIFT = {"CIP": -3.0, "CRO": 0.0, "CAZ": 3.0}   # small x offsets so intervals at the same k do not overlap
fig, axes = ps.figure("double", height=3.1, nrows=1, ncols=2, constrained_layout=True)
for ax, target, letter, note in ((axes[0], "s_orsola", "a", "Target Sant’Orsola"), (axes[1], "driams_b", "b", "Target DRIAMS-B")):
    le = json.load(open(ROOT / f"results/stats/domain_adaptation_{target}.json"))["label_efficiency"]
    for d in ("CIP", "CRO", "CAZ"):
        c = ps.DRUG[d]
        for meth, ls, hollow, shift, _ in METHODS:
            rr = sorted([r for r in le if r["method"] == meth], key=lambda r: r["k"])
            k = np.array([r["k"] for r in rr], float) + shift + DRUG_SHIFT[d]; y = np.array([r[f"br_{d}"] for r in rr]); sd = np.array([r[f"br_{d}_sd"] for r in rr])
            ax.plot(k, y, color=c, ls=ls, lw=1.2, zorder=2); ps.ci_point(ax, k, y, y - sd, y + sd, color=c, marker="o", hollow=hollow)
    ax.set_xticks([25, 50, 100, 150]); ax.set_xlim(15, 160); ax.set_ylim(0.5, 1.0); ps.despine(ax, grid=None)
    ax.set_xlabel("Local labeled isolates"); ax.set_ylabel("AUROC (binary relevance)")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=ps.INK2)
    ps.panel_label(ax, letter)
handles = [handle(ps.DRUG[d], marker="none", label=lab) for d, lab in (("CIP", "CIP"), ("CTX", CEPH), ("CAZ", "CAZ"))] + [handle(ps.INK, ls, "o", hollow, lab) for _, ls, hollow, _, lab in METHODS]
ps.legend_above(fig, handles, [h.get_label() for h in handles], y=1.0)
print("saved", ps.save(fig, OUT / "label_efficiency"))
