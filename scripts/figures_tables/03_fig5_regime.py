"""Main-text Figure 5: the regime in which label-dependency models help (logistic regression).
(a) Headroom within DRIAMS-A, two strips over the training size: the labels' own binary-relevance AUROC (top) and the gain
    over binary relevance of the CIP-first chain (hard passing) and of stacking (bottom)
    (results/stats/learning_curve_driams_a_grouped_stack_inner.json; full-site points from the replication and the nested
    stacking run; all DRIAMS-A inputs from the patient- and case-grouped runs);
(b) regime map across sites: CIP binary-relevance AUROC (upstream accuracy) against the cephalosporin's own
    binary-relevance AUROC (headroom), each site labelled with its chain ensemble and stacking gains, co-resistance as the
    marker fill (results/stats/driams_replication_grouped.json, nested_stacking_*.json, coresistance_structure.json);
(c) leave-one-year-out at DRIAMS-A for CRO: gain over binary relevance per test year with isolate-bootstrap intervals
    (results/stats/temporal_driams_a.json);
(d) label efficiency (results/stats/domain_adaptation_{s_orsola,driams_b}.json).
Reads stored results only. Output: figures/main/fig5_regime.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.visualization import paper_style as ps  # noqa: E402

OUT = ROOT / "figures/main"
SITE_KEY = {"Sant'Orsola": "s_orsola", "DRIAMS-B": "driams_b", "DRIAMS-C": "driams_c", "DRIAMS-D": "driams_d", "DRIAMS-A": "driams_a"}
STACK_TAG = {"Sant'Orsola": "s_orsola_logistic_full", "DRIAMS-A": "driams_a_grouped_logistic", "DRIAMS-B": "driams_b_logistic",
             "DRIAMS-C": "driams_c_logistic", "DRIAMS-D": "driams_d_logistic"}
BREAK_EVEN = 0.70                                   # upstream AUROC at which the chain gain crosses zero (Fig. 3c, d)
# (b) site labels: (dx, dy points, ha, va)
SITE_LABEL = {"Sant'Orsola": (6, 7, "right", "bottom"), "DRIAMS-A": (-8, 0, "right", "center"), "DRIAMS-B": (-8, 0, "right", "center"),
              "DRIAMS-C": (8, 0, "left", "center"), "DRIAMS-D": (0, 8, "center", "bottom")}


def style_handle(color=ps.INK, ls="-", marker="o", hollow=False, lw=1.2, ms=4.5, **kw):
    return Line2D([], [], color=color, ls=ls, lw=lw, marker=marker, ms=ms, mfc=ps.SURFACE if hollow else color, mec=color, mew=1.0, **kw)


def fmt(v):
    return f"{v:+.3f}".replace("-", "−")


ps.apply()
fig, mosaic = plt.subplot_mosaic([["a_top", "b"], ["a", "b"], ["c", "d"]], figsize=(ps.WIDTH["double"], 6.15), layout="constrained",
                                 gridspec_kw=dict(height_ratios=[0.9, 1.6, 2.6]))
fig.get_layout_engine().set(hspace=0.03)
ax_top, ax_a, ax_b, ax_c, ax_d = (mosaic[k] for k in ("a_top", "a", "b", "c", "d"))
ax_a.sharex(ax_top)   # (a) is two strips over the same training-size axis
rp = json.load(open(ROOT / "results/stats/driams_replication_grouped.json"))   # DRIAMS-A from the grouped run, other sites unchanged
stack = {site: {r["drug"]: r for r in json.load(open(ROOT / f"results/stats/nested_stacking_{tag}.json"))["rows"]} for site, tag in STACK_TAG.items()}

# (a) headroom within DRIAMS-A: two strips over training size
lc = json.load(open(ROOT / "results/stats/learning_curve_driams_a_grouped_stack_inner.json"))
N_FULL = 4761
METHODS = (("hard", "-", "o", 0.965, "Chain (CIP first, hard passing)"), ("stack", ":", "^", 1.035, "Stacking"))
for drug in ("CRO", "CAZ"):
    c = ps.DRUG[drug]; rows = sorted([r for r in lc if r["drug"] == drug], key=lambda r: r["n"]); n = np.array([r["n"] for r in rows], float)
    rep_a = rp["sites"]["DRIAMS-A"]["drugs"][drug]["AUROC"]
    # top: the label's own binary-relevance AUROC (headroom shrinks as it rises); full site hollow
    own = [r["br_auroc"] for r in rows]
    ax_top.plot(list(n) + [N_FULL], own + [rep_a["BR"]], color=c, lw=1.2, zorder=2)
    ax_top.plot(n, own, ls="none", marker="o", ms=3.8, color=c, zorder=3)
    ax_top.plot([N_FULL], [rep_a["BR"]], ls="none", marker="o", ms=4.2, mfc=ps.SURFACE, mec=c, mew=1.0, zorder=3)
    # bottom: gain over binary relevance of the chain and of stacking, BCa 95% bands; full site hollow (no interval)
    full = {"hard": rep_a["dEns"], "stack": stack["DRIAMS-A"][drug]["stack_vs_br"]["delta"]}
    for m, ls, mk, sh, _ in METHODS:
        y = np.array([r[f"d_{m}"] for r in rows])
        ax_a.fill_between(n * sh, [r[f"d_{m}_ci"][0] for r in rows], [r[f"d_{m}_ci"][1] for r in rows], color=c, alpha=0.10, lw=0, zorder=1)
        ax_a.plot(list(n * sh) + [N_FULL * sh], list(y) + [full[m]], color=c, ls=ls, lw=1.1, zorder=2)
        ax_a.plot(n * sh, y, ls="none", marker=mk, ms=3.8, color=c, zorder=3)
        ax_a.plot([N_FULL * sh], [full[m]], ls="none", marker=mk, ms=4.2, mfc=ps.SURFACE, mec=c, mew=1.0, zorder=3)
ax_a.set_xscale("log"); ax_a.set_xticks([187, 500, 1000, 2000, 4761]); ax_a.set_xticklabels(["187", "500", "1,000", "2,000", "4,761"]); ax_a.minorticks_off()
ax_a.set_xlim(160, 5800)
ax_top.set_ylim(0.5, 1.0); ax_top.set_yticks([0.5, 0.75, 1.0]); ax_top.tick_params(labelbottom=False); ps.despine(ax_top)
ax_top.set_ylabel("Own BR\nAUROC")
ax_top.legend(handles=[style_handle(ps.DRUG["CRO"], marker="o", ms=3.8, label="CRO"), style_handle(ps.DRUG["CAZ"], marker="o", ms=3.8, label="CAZ")],
              loc="lower right", ncol=2, borderaxespad=0.3, handlelength=1.6).set_in_layout(False)   # inside legends stay out of the layout
ps.hline(ax_a, 0); ps.despine(ax_a, grid=None); ax_a.set_ylim(-0.09, 0.24)
ax_a.set_xlabel("Training isolates (DRIAMS-A)"); ax_a.set_ylabel("Gain over BR (AUROC)")
ax_a.legend(handles=[style_handle(ps.INK2, ls, mk, lw=1.1, ms=3.8, label=lab) for m, ls, mk, sh, lab in METHODS] +
            [style_handle(ps.INK2, "none", "o", hollow=True, ms=4.2, label="Full site (n = 4,761)")], loc="upper right", borderaxespad=0.3, handlelength=2.0).set_in_layout(False)
ps.panel_label(ax_top, "a")

# (b) regime map across sites (cephalosporin)
cs = json.load(open(ROOT / "results/stats/coresistance_structure.json"))["X10a"]["sites"]
ax_b.axvspan(0.60, BREAK_EVEN, color=ps.GRID, alpha=0.7, lw=0, zorder=0)
ax_b.axvline(BREAK_EVEN, color=ps.MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
ax_b.text(BREAK_EVEN - 0.004, 0.905, "CIP below\nbreak-even", ha="right", va="top", fontsize=6, color=ps.INK2, linespacing=1.1)
for r in [r for r in rp["site_level"] if r["drug"] in ("CTX", "CRO")]:
    site = r["site"]; key = SITE_KEY[site]; ceph = r["drug"]
    p = cs[key]["pairs"][f"CIP-{ceph}"][f"P(CIPR|{ceph}R)"]; weak = p < 0.65
    c = ps.SITE_ACCENT["s_orsola"] if key == "s_orsola" else ps.INK2
    ax_b.plot([r["cip_auroc"]], [r["drug_br_auroc"]], marker="o", ms=6, mfc=ps.SURFACE if weak else c, mec=c, mew=1.2, ls="none", zorder=3)
    dx, dy, ha, va = SITE_LABEL[site]
    ax_b.annotate(f"{site.replace(chr(39), chr(8217))}\n{fmt(r['dEns_auroc'])} | {fmt(stack[site][ceph]['stack_vs_br']['delta'])}", (r["cip_auroc"], r["drug_br_auroc"]),
                  xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=6.2, color=c, linespacing=1.15)
ax_b.set_xlim(0.60, 0.90); ax_b.set_ylim(0.62, 0.91)
ax_b.set_xlabel("CIP binary-relevance AUROC at the site"); ax_b.set_ylabel("Cephalosporin binary-relevance AUROC")
ax_b.annotate("", xy=(0.895, 0.90), xytext=(0.895, 0.83), arrowprops=dict(arrowstyle="->", lw=0.7, color=ps.INK2))
ax_b.text(0.889, 0.865, "less\nheadroom", ha="right", va="center", fontsize=6, color=ps.INK2, linespacing=1.1)
ax_b.legend(handles=[style_handle(ps.INK2, "none", "o", ms=5.5, label="P(CIP-R | 3GC-R) 0.73 to 0.76"), style_handle(ps.INK2, "none", "o", hollow=True, ms=5.5, label="0.53"),
                     Line2D([], [], ls="none", label="labels: chain | stacking gain")],
            loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0.0, handlelength=1.0, columnspacing=1.0)
ps.despine(ax_b, grid=None); ps.panel_label(ax_b, "b")

# (c) leave-one-year-out, CRO: gain over BR per year
tp = json.load(open(ROOT / "results/stats/temporal_driams_a.json"))["per_year"]
rows = [r for r in tp if r["drug"] == "CRO" and r["year"] != "mean"]; years = [str(r["year"]) for r in rows]; xs = np.arange(len(rows))
CONTRASTS = [("bag_boot-br", "bagged_boot", "v", False, "Bagged BR (bootstrap)"), ("bag_sub-br", "bagged_sub", "^", False, "Bagged BR (bins)"),
             ("permuted-br", "permuted", "X", False, "Permuted upstream"), ("hard-br", "ensemble", "s", False, "Chain ensemble"),
             ("proba-br", "proba", "s", True, "Chain, probability"), ("stack-br", "stacking", "^", False, "Stacking")]
offs = np.linspace(-0.30, 0.30, len(CONTRASTS))
for (key, role, mk, hollow, lab), off in zip(CONTRASTS, offs):
    c = ps.ROLE[role]; d = np.array([r["contrasts"][key]["delta"] for r in rows])
    ps.ci_point(ax_c, xs + off, d, [r["contrasts"][key]["ci"][0] for r in rows], [r["contrasts"][key]["ci"][1] for r in rows], color=c, marker=mk, hollow=hollow, ms=4.0, label=lab)
orc = np.array([r["contrasts"]["oracle-br"]["delta"] for r in rows])   # ceiling, light reference without interval
ax_c.hlines(orc, xs - 0.38, xs + 0.38, colors=ps.ROLE["oracle"], lw=1.0, alpha=0.55, zorder=1, label="Oracle (true upstream label, ceiling)")
ps.hline(ax_c, 0.0)
ax_c.set_xticks(xs); ax_c.set_xticklabels([f"{y}\n(n = {r['n_test']:,})" for y, r in zip(years, rows)]); ax_c.set_xlim(-0.5, len(rows) - 0.5)
ax_c.set_ylim(-0.06, 0.16); ax_c.set_yticks([-0.05, 0.0, 0.05, 0.10, 0.15]); ps.despine(ax_c)
ax_c.set_xlabel("Test year (DRIAMS-A, trained on the other three years)"); ax_c.set_ylabel("CRO gain over binary relevance (AUROC)")
leg_c = ps.legend_above(ax_c, ncol=2, y=1.02, columnspacing=0.8, handlelength=1.4, handletextpad=0.4)   # narrower than the column, or the layout collapses

# (d) label efficiency
SITE_TEXT = {"s_orsola": ("Sant’Orsola (CTX)", 75, 0.552), "driams_b": ("DRIAMS-B (CRO)", 62, 0.875)}   # direct labels (text, k, AUROC) in empty space
for target, drug, shift in (("s_orsola", "CRO", -1.5), ("driams_b", "CRO", 1.5)):
    le = json.load(open(ROOT / f"results/stats/domain_adaptation_{target}.json"))["label_efficiency"]
    c = ps.SITE_ACCENT[target]; mk = ps.SITE_MARKER[target]
    for meth, ls, hollow in (("local_only", "-", False), ("source_plus_local", "--", True)):
        rr = sorted([r for r in le if r["method"] == meth], key=lambda r: r["k"])
        k = np.array([r["k"] for r in rr], float) + shift; y = np.array([r[f"br_{drug}"] for r in rr]); sd = np.array([r[f"br_{drug}_sd"] for r in rr])
        ax_d.plot(k, y, color=c, ls=ls, lw=1.2, zorder=2)
        ps.ci_point(ax_d, k, y, y - sd, y + sd, color=c, marker=mk, hollow=hollow)
    txt, kx, yy = SITE_TEXT[target]
    ax_d.text(kx, yy, txt, ha="center", va="center", fontsize=6.5, color=c)
ax_d.set_xticks([25, 50, 100, 150]); ax_d.set_xlim(15, 160); ax_d.set_ylim(0.5, 1.0); ps.despine(ax_d, grid=None)
ax_d.set_xlabel("Local labeled isolates"); ax_d.set_ylabel("Cephalosporin AUROC (binary relevance)")
handles_d = [style_handle(ps.INK2, "-", "o", label="Local isolates only"), style_handle(ps.INK2, "--", "o", hollow=True, label="Pooled with DRIAMS-A")]
leg_d = ps.legend_above(ax_d, handles_d, [h.get_label() for h in handles_d], ncol=2, y=1.02, columnspacing=0.8, handlelength=1.8, handletextpad=0.4)
ps.panel_label(ax_c, "c"); ps.panel_label(ax_d, "d")

OUT.mkdir(parents=True, exist_ok=True)
print("saved", ps.save(fig, OUT / "fig5_regime"))
