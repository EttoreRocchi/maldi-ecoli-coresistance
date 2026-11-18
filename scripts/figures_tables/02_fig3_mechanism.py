"""Main-text Figure 3: what governs the chain benefit (plot-only; reads stored results, no computation).

(a) chain ensemble AUROC per drug under the four upstream-input variants (link ablation, script 20);
(b) CTX-downstream AUROC per ordering under the four variants (script 20);
(c) CTX gain over binary relevance versus the accuracy of the upstream CIP classifier (degradation experiment,
    script 23), hard label passing and probability passing, on axes fitted to the data;
(d) ceiling-to-gain curves: downstream gain of the saved links when the true upstream label is degraded by label noise
    (script 34; upstream AUROC = 1 - flip rate).
Inputs: results/stats/link_ablation.json, upstream_accuracy.json, ceiling_to_gain.json.
Output: figures/main/fig4_mechanism.png (+pdf).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.visualization import paper_style as ps  # noqa: E402

VARIANTS = ["permuted", "realized", "proba"]   # points, left to right: no information, realized, probability; the oracle is a ceiling line
VCOL = {"realized": ps.ROLE["ensemble"], "proba": ps.ROLE["proba"], "oracle": ps.ROLE["oracle"], "permuted": ps.ROLE["permuted"]}
VLAB = {"realized": "Realized prediction", "proba": "Probability passing", "oracle": "Oracle (true upstream label, ceiling)", "permuted": "Permuted label"}
OFF = {"permuted": -0.2, "realized": 0.0, "proba": 0.2}
CEIL_KW = dict(colors=ps.ROLE["oracle"], lw=1.0, alpha=0.55, zorder=1)   # oracle ceiling, drawn as in Fig. 5c
DRUGS = ["CIP", "CTX", "CAZ"]
UP_LS = {"CIP": "-", "CTX": "--", "CAZ": "-."}      # upstream drug as line style (panel d)
UP_MK = {"CIP": "o", "CTX": "s", "CAZ": "^"}        # upstream drug as marker (panel d)
REAL_LABEL_OFFSET = {"CIP->CTX": (8, 0, "left"), "CAZ->CTX": (7, -7, "left"), "CTX->CAZ": (6, -11, "left")}   # (dx, dy pt, ha)
GAIN_XLIM, GAIN_YLIM = (0.5, 1.0), (-0.22, 0.28)    # shared by the two gain-versus-upstream-accuracy panels (c, d)
UPSTREAM_ORDER = ["CIP", "CAZ", "CIP>CAZ", "CAZ>CIP"]
UPSTREAM_LABEL = {"CIP": "CIP", "CAZ": "CAZ", "CIP>CAZ": "CIP and CAZ\n(CIP first)", "CAZ>CIP": "CIP and CAZ\n(CAZ first)"}


def upstream_of_ctx(ordering):
    """Upstream drugs of CTX in an ordering such as 'CAZ->CIP->CTX', as 'CAZ>CIP' (in chain order)."""
    order = ordering.split("->"); return ">".join(order[:order.index("CTX")])


def variant_points(ax, x, rows, hollow_proba=True):
    """Dot-and-interval for the four variants at category x; rows: variant -> record with AUROC, dAUROC, dAUROC_ci.
    The interval is the BCa 95% interval of the paired difference versus binary relevance, shifted to the AUROC scale.
    A faint line joins the three variants; the oracle (true upstream label) is a light ceiling line over the group."""
    ax.plot([x + OFF[v] for v in VARIANTS], [rows[v]["AUROC"] for v in VARIANTS], color=ps.LIGHT, lw=0.8, zorder=1)
    ax.hlines(rows["oracle"]["AUROC"], x - 0.38, x + 0.38, **CEIL_KW)
    for v in VARIANTS:
        r = rows[v]
        lo = r["AUROC"] + (r["dAUROC_ci"][0] - r["dAUROC"])
        hi = r["AUROC"] + (r["dAUROC_ci"][1] - r["dAUROC"])
        ps.ci_point(ax, x + OFF[v], r["AUROC"], lo, hi, color=VCOL[v], hollow=(v == "proba" and hollow_proba))


def variant_legend_handles():
    hs = [Line2D([], [], ls="none", marker="o", ms=4.5, color=VCOL[v], mfc=ps.SURFACE if v == "proba" else VCOL[v], mec=VCOL[v], mew=1.0, label=VLAB[v])
          for v in VARIANTS]
    hs.append(Line2D([], [], ls="-", lw=1.0, color=ps.ROLE["oracle"], alpha=0.55, label=VLAB["oracle"]))
    hs.append(Line2D([], [], ls="-", lw=0.9, color=ps.ROLE["br"], label="Binary relevance"))
    return hs, [h.get_label() for h in hs]


def break_even(xs, ys):
    """First x at which the gain crosses zero from below (linear interpolation); None if it never does."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for i in range(1, len(xs)):
        if ys[i - 1] < 0 <= ys[i]:
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (-ys[i - 1]) / (ys[i] - ys[i - 1]))
    return None


def gain_curve(ax, rows, ls, hollow, label, alpha=0.13):
    """Gain against upstream accuracy: line with small markers and the BCa 95% interval as a light band."""
    xs = [r["cip_auroc"] for r in rows]; ys = [r["downstream_dAUROC"] for r in rows]; c = ps.ROLE["ensemble"]
    ax.fill_between(xs, [r["dAUROC_ci"][0] for r in rows], [r["dAUROC_ci"][1] for r in rows], color=c, alpha=alpha, lw=0, zorder=1)
    ax.plot(xs, ys, ls=ls, lw=1.2, color=c, marker="o", ms=3.2, mfc=ps.SURFACE if hollow else c, mec=c, mew=0.9, zorder=3, label=label)
    return break_even(xs, ys)


def axis_tick(ax, x, ls="-", color=ps.INK2, frac=0.045):
    """Short vertical mark rising from the x-axis at x (break-even marker)."""
    y0, y1 = ax.get_ylim()
    ax.plot([x, x], [y0, y0 + frac * (y1 - y0)], ls=ls, lw=1.0, color=color, zorder=4, clip_on=False, solid_capstyle="butt")


def main():
    la = json.load(open(ROOT / "results/stats/link_ablation.json"))
    ua = json.load(open(ROOT / "results/stats/upstream_accuracy.json"))
    cg = json.load(open(ROOT / "results/stats/ceiling_to_gain.json"))["Sant'Orsola"]

    fig, axes = ps.figure("double", height=5.8, nrows=2, ncols=2, constrained_layout=True)
    fig.get_layout_engine().set(hspace=0.07)   # room between the rotated ordering labels of (b) and the letter of (d)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) ensemble per drug under the four variants, BR as a short reference line per drug
    for j, d in enumerate(DRUGS):
        rows = {v: next(r for r in la["ensemble"] if r["variant"] == v and r["drug"] == d) for v in VARIANTS + ["oracle"]}
        variant_points(ax_a, j, rows)
        ax_a.hlines(la["br_reference"][d]["AUROC"], j - 0.42, j + 0.42, colors=ps.ROLE["br"], lw=0.9, zorder=1)
    ax_a.set_xticks(range(3)); ax_a.set_xticklabels(DRUGS); ax_a.set_xlim(-0.6, 2.6)
    ax_a.set_ylabel("Chain ensemble AUROC"); ax_a.set_ylim(0.4, 1.0); ps.despine(ax_a); ps.panel_label(ax_a, "a")

    # (b) CTX downstream per ordering
    rows_b = [r for r in la["per_ordering"] if r["downstream"] == "CTX"]
    oks = sorted(dict.fromkeys(r["ordering"] for r in rows_b), key=lambda o: UPSTREAM_ORDER.index(upstream_of_ctx(o)))
    for j, o in enumerate(oks):
        rows = {v: next(r for r in rows_b if r["ordering"] == o and r["variant"] == v) for v in VARIANTS + ["oracle"]}
        variant_points(ax_b, j, rows)
    ps.hline(ax_b, la["br_reference"]["CTX"]["AUROC"], color=ps.ROLE["br"], lw=0.9)
    ax_b.set_xticks(range(len(oks))); ax_b.set_xlim(-0.6, len(oks) - 0.4)
    ax_b.set_xticklabels([UPSTREAM_LABEL[upstream_of_ctx(o)] for o in oks])
    for j, o in enumerate(oks):   # the ordering itself, muted, under the upstream input
        ax_b.annotate(o.replace("->", "\u2192"), (j, 0), xycoords=("data", "axes fraction"), xytext=(0, -21), textcoords="offset points",
                      ha="center", va="top", fontsize=6, color=ps.MUTED)
    ax_b.set_xlabel("Upstream input of CTX", labelpad=14)
    ax_b.set_ylabel("Downstream CTX AUROC"); ax_b.set_ylim(0.4, 1.0); ps.despine(ax_b); ps.panel_label(ax_b, "b")

    # (c) CTX gain versus upstream CIP accuracy, own axes fitted to the data: hard passing (the scheme of the main analysis) with its
    #     band, probability passing as a thin dashed line; region below the hard-passing break-even shaded as in Fig. 5b
    sel = lambda passing: sorted([x for x in ua["levels"] if x["ordering"] == "CIP>CTX>CAZ" and x["passing"] == passing], key=lambda x: x["cip_auroc"])
    hard, prob = sel("hard"), sel("proba"); c = ps.ROLE["ensemble"]
    be_hard = gain_curve(ax_c, hard, "-", False, "Hard label passing")
    xp = [r["cip_auroc"] for r in prob]; yp = [r["downstream_dAUROC"] for r in prob]
    ax_c.plot(xp, yp, ls="--", lw=1.0, color=c, marker="o", ms=3.0, mfc=ps.SURFACE, mec=c, mew=0.9, zorder=3)
    C_XLIM, C_YLIM = (0.60, 0.95), (-0.085, 0.105)
    ax_c.axvspan(C_XLIM[0], be_hard, color=ps.GRID, alpha=0.7, lw=0, zorder=0)
    ax_c.axvline(be_hard, color=ps.MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax_c.text(be_hard - 0.004, C_YLIM[1] - 0.004, f"below break-even\n({be_hard:.2f})", ha="right", va="top", fontsize=6, color=ps.INK2, linespacing=1.1)
    ps.hline(ax_c, 0.0)
    for r, y_up in zip(hard, yp):   # share of m/z bins used to train the CIP classifier, above the upper point of each level
        if r["fraction"] < 1.0:
            dx, ha = (-3, "right") if r["fraction"] >= 0.25 else (0, "center")   # the last levels are close together
            ax_c.annotate(f"{r['fraction']:.0%}", (r["cip_auroc"], max(y_up, r["downstream_dAUROC"])), xytext=(dx, 6), textcoords="offset points",
                          ha=ha, va="bottom", fontsize=6, color=ps.INK2)
    full = hard[-1]
    ax_c.annotate("100%: real CIP\nclassifier", (full["cip_auroc"], full["downstream_dAUROC"]), xytext=(8, 0), textcoords="offset points",
                  ha="left", va="center", fontsize=6, color=ps.INK2, linespacing=1.1)
    ax_c.set_xlim(*C_XLIM); ax_c.set_ylim(*C_YLIM)
    ax_c.set_xlabel("Upstream CIP AUROC (share of m/z bins used)"); ax_c.set_ylabel("CTX gain over binary relevance (AUROC)")
    ps.despine(ax_c); ps.panel_label(ax_c, "c")
    hc = [Line2D([], [], ls="-", lw=1.2, marker="o", ms=3.2, color=c, label="Hard label passing"),
          Line2D([], [], ls="--", lw=1.0, marker="o", ms=3.0, color=c, mfc=ps.SURFACE, mew=0.9, label="Probability passing")]
    ax_c.legend(handles=hc, loc="lower right", handlelength=2.4)

    # (d) ceiling-to-gain: gain of the saved links as the true upstream label is degraded; filled symbol, realized chain
    ps.hline(ax_d, 0.0)
    ax_d.set_xlim(*GAIN_XLIM); ax_d.set_ylim(*GAIN_YLIM)
    hd = []
    for link in ("CIP->CTX", "CAZ->CTX", "CTX->CAZ"):
        e = cg[link]; up, down = link.split("->"); c = ps.DRUG[down]
        q = np.array(sorted(float(k) for k in e["curve"])); g = np.array([e["curve"][k] for k in sorted(e["curve"], key=float)])
        ax_d.plot(1 - q, g, color=c, ls=UP_LS[up], lw=1.2, zorder=2)
        ax_d.plot([e["upstream_auroc"]], [e["realized_hard"]], marker=UP_MK[up], color=c, mec=c, ms=5.5, ls="none", zorder=4)
        axis_tick(ax_d, e["breakeven_upstream_auroc"], color=c)
        dx, dy, ha = REAL_LABEL_OFFSET[link]   # where the link sits: AUROC of the real upstream classifier
        ax_d.annotate(f"real {up} {e['upstream_auroc']:.2f}", (e["upstream_auroc"], e["realized_hard"]), xytext=(dx, dy), textcoords="offset points",
                      ha=ha, va="center", fontsize=6.5, color=c)
        hd.append(Line2D([], [], color=c, ls=UP_LS[up], lw=1.2, marker=UP_MK[up], ms=5, label=f"{up} \u2192 {down}"))
    hd.append(Line2D([], [], color=ps.INK, ls="none", marker="|", ms=7, mew=1.2, label="Break-even upstream AUROC"))
    hd.insert(3, Line2D([], [], color=ps.INK2, ls="none", marker="o", ms=5, label="Realized chain (real upstream classifier)"))
    ax_d.set_xlabel("Upstream AUROC (true label degraded by noise)"); ax_d.set_ylabel("Gain over binary relevance (AUROC)")
    ps.despine(ax_d); ps.panel_label(ax_d, "d")
    ax_d.legend(handles=hd, loc="upper left", handlelength=2.4)

    handles, labels = variant_legend_handles()
    ps.legend_above(fig, handles, labels, ncol=5, y=1.035)   # above the panel letters of the top row
    out = ps.save(fig, ROOT / "figures/main/fig3_mechanism")
    print("saved", out)


if __name__ == "__main__":
    main()
