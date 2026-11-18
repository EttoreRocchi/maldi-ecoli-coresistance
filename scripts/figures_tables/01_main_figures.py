"""Main-text macro figures of the restructured manuscript (five figures).

Fig. 1 fig1_cooccurrence: (a) phi matrix sites x drug pairs, (b) UpSet of the Sant'Orsola profiles.
Fig. 2 fig2_prespecified: (a to c) AUROC per ordering and ensemble per drug, (d, e) VME and ME at 0.5.
Fig. 3 fig3_mechanism: copy of the mechanism figure (scripts/figures_tables/02_fig3_mechanism.py).
Fig. 4 fig4_stacking: (a) AUROC of BR, chain ensemble and stacked model per drug (full grid, BCa over 15 folds);
       (b) proxy margin against the gain over BR for every site, learner and learning-curve subsample (scripts/04_stacking/02_cross_prediction.py, scripts/05_regime/02_cross_prediction_lc.py);
       (c) AUROC of the own and the CIP classifier within the ciprofloxacin strata per site (scripts/04_stacking/05_stratified_analysis.py); (d) net benefit for CTX.
       DRIAMS-A in (b) and (c): the patient- and case-grouped runs (scripts/04_stacking/02_cross_prediction.py, scripts/05_regime/02_cross_prediction_lc.py, scripts/04_stacking/05_stratified_analysis.py --grouped).
Fig. 5 fig5_regime: copy of the regime figure (scripts/figures_tables/03_fig5_regime.py).
All panels are drawn from stored results; nothing is recomputed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci, load_cv_results  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.visualization import paper_style as ps  # noqa: E402
from mecor.visualization.clinical import plot_vme_me_comparison  # noqa: E402
from mecor.visualization.cooccurrence import plot_phi_heatmaps_multi, plot_upset_from_counts, profile_counts_from_Y  # noqa: E402
from mecor.visualization.performance import plot_ordering_comparison  # noqa: E402

OUT = ROOT / "figures/main"; DRUGS = ["CIP", "CTX", "CAZ"]; SITES = ["s_orsola", "driams_a", "driams_b", "driams_c", "driams_d"]
from mecor import shared as pf


def fig1():
    analyses = {}
    for s in SITES:
        f = ROOT / f"results/cooccurrence/{s}/cooccurrence_results.json"
        if f.exists(): analyses[s] = json.load(open(f))
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml")); X, Y = load_dataset(cfg)
    counts = profile_counts_from_Y(Y); labels = list(Y.columns)
    cs = json.load(open(ROOT / "results/stats/coresistance_structure.json"))["X10a"]["sites"]   # P(CIP-R | 3GC-R) per site
    cond = {}
    for s, v in cs.items():
        ceph = "CTX" if "CIP-CTX" in v["pairs"] else "CRO"; cond[s] = v["pairs"][f"CIP-{ceph}"][f"P(CIPR|{ceph}R)"]
    ps.apply(); fig = plt.figure(figsize=(7.2, 3.3)); subs = fig.subfigures(1, 2, width_ratios=[1.2, 0.95], wspace=0.04)
    ax = subs[0].add_axes([0.335, 0.08, 0.385, 0.78]); plot_phi_heatmaps_multi(analyses, ax=ax, conditional=cond)
    ax.set_xticklabels(["CIP and\nCTX/CRO", "CIP and\nCAZ", "CTX/CRO\nand CAZ"], fontsize=6.8)
    subs[0].text(0.02, 0.96, "A", fontsize=8, fontweight="bold", va="top")
    plot_upset_from_counts(counts, labels, fig=subs[1]); subs[1].text(0.02, 0.96, "B", fontsize=8, fontweight="bold", va="top")
    ps.save(fig, OUT / "fig1_cooccurrence")


def fig2():
    cv = load_cv_results(ROOT / "results/s_orsola/logistic/cv_results.pkl")
    ps.apply(); fig = plt.figure(figsize=(7.2, 5.9)); subs = fig.subfigures(2, 1, height_ratios=[1.0, 0.95], hspace=0.08)
    axes = subs[0].subplots(1, 3); subs[0].subplots_adjust(left=0.08, right=0.99, bottom=0.30, top=0.86, wspace=0.32)
    plot_ordering_comparison(cv, DRUGS, metric_name="AUROC", interval_fn=pf._bca, fig=subs[0], axes=axes, letters="abc", color_by_upstream=True)
    axes2 = subs[1].subplots(1, 2); subs[1].subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.86, wspace=0.32)
    iv = pf.clinical_intervals(cv); iv["stacking"] = _stacking_error_intervals()
    plot_vme_me_comparison(labels=DRUGS, intervals=iv, fig=subs[1], axes=axes2, letters="de")
    ps.save(fig, OUT / "fig2_prespecified")


def _stacking_error_intervals(threshold=0.5):
    """VME and ME of the stacked model (full grid) at the 0.5 threshold per outer fold, BCa 95% over the 15 folds."""
    out = {d: {"VME": [], "ME": []} for d in DRUGS}
    for f in sorted((ROOT / "results/nested_stacking/s_orsola_logistic_full").glob("r*_f?.npz")):
        z = np.load(f)
        for j, d in enumerate(DRUGS):
            y = z["y"][:, j]; yhat = z["stack"][:, j] >= threshold
            out[d]["VME"].append(np.mean(~yhat[y == 1])); out[d]["ME"].append(np.mean(yhat[y == 0]))
    assert all(len(v["VME"]) == 15 for v in out.values())
    return {d: {m: pf._bca(v[m]) for m in ("VME", "ME")} for d, v in out.items()}


def _pooled_predictions():
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml")); X, Y = load_dataset(cfg); labels = list(Y.columns)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    folds = {(g // 5, g % 5): te for g, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, strat))}
    cv = load_cv_results(ROOT / "results/s_orsola/logistic/cv_results.pkl"); ens = {(fr.repeat_idx, fr.fold_idx): fr for fr in cv.fold_results if fr.method == "ensemble"}
    pooled = {m: {d: [] for d in labels} for m in ("br", "ens", "stack")}; y = {d: [] for d in labels}; per_fold = {}
    for (r, f), te in sorted(folds.items()):
        z = np.load(ROOT / f"results/nested_stacking/s_orsola_logistic_full/r{r}_f{f}.npz"); fr = ens[(r, f)]
        per_fold[(r, f)] = json.load(open(ROOT / f"results/nested_stacking/s_orsola_logistic_full/r{r}_f{f}.json"))
        for j, d in enumerate(labels):
            y[d].append(z["y"][:, j]); pooled["br"][d].append(z["br"][:, j]); pooled["stack"][d].append(z["stack"][:, j]); pooled["ens"][d].append(np.asarray(fr.y_prob[d]))
    return labels, {d: np.concatenate(v) for d, v in y.items()}, {m: {d: np.concatenate(v) for d, v in dd.items()} for m, dd in pooled.items()}, per_fold


def _cip_proxy_per_fold(labels):
    """AUROC of the CIP binary-relevance classifier scored against each label, per outer fold (full grid)."""
    out = {d: [] for d in labels}
    for f in sorted((ROOT / "results/nested_stacking/s_orsola_logistic_full").glob("r*_f?.npz"), key=lambda q: tuple(int(t[1:]) for t in q.stem.split("_"))):
        z = np.load(f)
        for j, d in enumerate(labels): out[d].append(roc_auc_score(z["y"][:, j], z["br"][:, labels.index("CIP")]))
    return {d: np.array(v) for d, v in out.items()}


def _net_benefit(y, p, t):
    yhat = p >= t; n = len(y); tp = np.sum(yhat & (y == 1)); fp = np.sum(yhat & (y == 0)); return tp / n - fp / n * t / (1 - t)


def fig4():
    labels, y, P, per_fold = _pooled_predictions(); keys = sorted(per_fold)
    series = [("br", ps.ROLE["br"], "Binary relevance", "o"), ("ens", ps.ROLE["ensemble"], "Chain ensemble", "s"), ("stack", ps.ROLE["stacking"], "Stacked model", "^")]
    ps.apply(); fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3), gridspec_kw=dict(wspace=0.30, hspace=0.50, left=0.08, right=0.99, top=0.90, bottom=0.10))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    # (a) AUROC per drug, BCa over the 15 folds; the CIP classifier scored against each other label (proxy, scripts/04_stacking/02_cross_prediction.py)
    x = np.arange(3); offs = np.linspace(-0.27, 0.27, 4)
    for (m, col, name, mk), off in zip(series, offs[:3]):
        vals = [np.array([per_fold[k][f"{m}_{d}"]["AUROC"] for k in keys]) for d in labels]
        mean = [v.mean() for v in vals]; ci = [bca_mean_ci(v, seed=42) for v in vals]
        ps.ci_point(ax_a, x + off, mean, [c[1] for c in ci], [c[2] for c in ci], color=col, marker=mk, label=name)
    proxy = _cip_proxy_per_fold(labels)
    xp = json.load(open(ROOT / "results/stats/cross_prediction_grouped.json"))   # DRIAMS-A from the patient- and case-grouped run
    for j, d in enumerate(labels[1:], start=1):
        v = proxy[d]; ref = next(r for r in xp if r["tag"] == "s_orsola_logistic_full" and r["label"] == d)["auroc_by_classifier"]["CIP"]
        assert abs(v.mean() - ref) < 1e-9, (d, v.mean(), ref)
        _, lo, hi = bca_mean_ci(v, seed=42)
        ps.ci_point(ax_a, j + offs[3], v.mean(), lo, hi, color=ps.INK2, marker="D", hollow=True, ms=4.2)
    ax_a.set_xticks(x); ax_a.set_xticklabels(labels); ax_a.set_xlim(-0.6, 2.6); ax_a.set_ylim(0.5, 1.0); ax_a.set_ylabel("AUROC"); ps.despine(ax_a); ps.panel_label(ax_a, "a")
    # (b) proxy margin against the gain over BR for every point: full cohorts (sites and learners; stacking, scripts/04_stacking/02_cross_prediction.py) and
    #     learning-curve subsamples (stacking filled, hard-label chain hollow; scripts/05_regime/02_cross_prediction_lc.py)
    lc = [r for r in json.load(open(ROOT / "results/stats/cross_prediction_lc_grouped.json"))["rows"] if isinstance(r.get("gain_hard"), dict)]
    key = lambda lab: "CTX" if lab in ("CTX", "CRO") else lab
    xl_, yl_ = (-0.11, 0.17), (-0.07, 0.17)
    ax_b.axvspan(0, xl_[1], color=ps.GRID, alpha=0.6, lw=0, zorder=0)
    ax_b.plot(xl_, xl_, color=ps.LIGHT, lw=0.8, ls="--", zorder=1); ps.hline(ax_b, 0.0); ax_b.axvline(0, color=ps.INK2, lw=0.7, zorder=1)
    for r in lc:
        c = ps.DRUG[key(r["label"])]
        ax_b.plot(r["margin"]["delta"], r["gain_hard"]["delta"], ls="none", marker="^", ms=3.6, mfc=ps.SURFACE, mec=c, mew=0.9, zorder=3)
        ax_b.plot(r["margin"]["delta"], r["gain_stack"]["delta"], ls="none", marker="^", ms=3.6, color=c, zorder=4)
    for r in xp:
        ax_b.plot(r["proxy_minus_own"]["delta"], r["stacked"] - r["auroc_by_classifier"][r["label"]], ls="none", marker="o", ms=4.8, color=ps.DRUG[key(r["label"])], mec=ps.SURFACE, mew=0.6, zorder=5)
    so = next(r for r in xp if r["tag"] == "s_orsola_logistic_full" and r["label"] == "CTX")
    ax_b.annotate("Sant'Orsola\nCTX", (so["proxy_minus_own"]["delta"], so["stacked"] - so["auroc_by_classifier"]["CTX"]), xytext=(0.124, 0.074), textcoords="data", ha="left", va="center", fontsize=6, color=ps.INK2, linespacing=1.1,
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=ps.MUTED, shrinkA=1, shrinkB=3))
    ax_b.text(xl_[1] - 0.007, -0.062, "another classifier\nbeats the own one", ha="right", va="bottom", fontsize=6, color=ps.INK2, linespacing=1.1)
    ax_b.set_xlim(xl_); ax_b.set_ylim(yl_); ax_b.set_xlabel("Proxy margin (AUROC)"); ax_b.set_ylabel("Gain over binary relevance (AUROC)"); ps.despine(ax_b, grid=None); ps.panel_label(ax_b, "b")
    hb = [Line2D([], [], ls="none", marker="o", ms=4.8, color=ps.INK2, mec=ps.SURFACE, label="Full cohort, stacking"),
          Line2D([], [], ls="none", marker="^", ms=3.6, color=ps.INK2, label="Subsample, stacking"),
          Line2D([], [], ls="none", marker="^", ms=3.6, mfc=ps.SURFACE, mec=ps.INK2, mew=0.9, label="Subsample, chain")]
    hb += [Line2D([], [], ls="none", marker="s", ms=4, color=ps.DRUG[k], label=l) for k, l in (("CIP", "CIP"), ("CTX", "CTX or CRO"), ("CAZ", "CAZ"))]
    ax_b.legend(handles=hb, loc="upper left", frameon=False, fontsize=6, handletextpad=0.3, labelspacing=0.35, borderaxespad=0.2)
    # (c) AUROC on the cefotaxime (Sant'Orsola) or ceftriaxone (DRIAMS) label within the ciprofloxacin strata: own and CIP classifier (scripts/04_stacking/05_stratified_analysis.py)
    st = [r for r in json.load(open(ROOT / "results/stats/stratified_analysis_grouped.json")) if r["label"] in ("CTX", "CRO") and "logistic" in r["learner"]]
    sites_c = ["Sant'Orsola", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D", "DRIAMS-A"]; ticks_c = [t.replace("'", "\u2019") for t in sites_c]
    ax_c.axhline(0.5, color=ps.MUTED, lw=0.8, ls=":", zorder=1)
    for i, site in enumerate(sites_c):
        for stratum, hollow, base in (("CIP-S", False, -0.2), ("CIP-R", True, 0.2)):
            r = next(q for q in st if q["site"] == site and q["stratum"] == stratum)
            for clf, col, dx in (("own", ps.DRUG["CTX"], -0.07), ("CIP", ps.DRUG["CIP"], 0.07)):
                e = r["auroc"][clf]; ps.ci_point(ax_c, i + base + dx, e["est"], e["ci"][0], e["ci"][1], color=col, hollow=hollow, ms=3.8)
    ax_c.set_xticks(range(len(sites_c))); ax_c.set_xticklabels(ticks_c, fontsize=6.5); ax_c.set_xlim(-0.55, len(sites_c) - 0.45); ax_c.set_ylim(0.3, 1.0)
    ax_c.set_ylabel("AUROC within the CIP stratum"); ps.despine(ax_c); ps.panel_label(ax_c, "c")
    hc_ = [Line2D([], [], ls="none", marker="o", ms=4, color=ps.DRUG["CTX"], label="Own classifier (CTX or CRO)"), Line2D([], [], ls="none", marker="o", ms=4, color=ps.DRUG["CIP"], label="CIP classifier"),
           Line2D([], [], ls="none", marker="o", ms=4, color=ps.INK2, label="CIP-susceptible"), Line2D([], [], ls="none", marker="o", ms=4, mfc=ps.SURFACE, mec=ps.INK2, mew=1.0, label="CIP-resistant")]
    ax_c.legend(handles=hc_, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=6, columnspacing=1.0, handletextpad=0.3)
    # (d) net benefit for CTX
    d = "CTX"; yy = y[d]; prev = yy.mean(); ts = np.linspace(0.05, 0.60, 111)
    ax_d.axvspan(0.30, 0.60, color=ps.GRID, alpha=0.6, lw=0, zorder=0)   # window where the chain and the stacked model add net benefit over BR
    ax_d.text(0.535, 0.26, "chain and stacked\nmodel above BR", ha="center", va="center", fontsize=6, color=ps.INK2, linespacing=1.1)
    ax_d.plot(ts, [prev - (1 - prev) * t / (1 - t) for t in ts], color=ps.MUTED, ls="--", lw=1.0, label="Treat all")
    ax_d.axhline(0, color=ps.MUTED, ls=":", lw=0.9, label="Treat none")
    for m, col, name, mk in series: ax_d.plot(ts, [_net_benefit(yy, P[m][d], t) for t in ts], color=col, lw=1.3, label=name)
    ax_d.axvline(prev, color=ps.LIGHT, lw=0.8); ax_d.text(prev - 0.008, 0.40, "cohort\nprevalence", fontsize=6, color=ps.INK2, va="top", ha="right")
    ax_d.set_xlabel("Threshold probability"); ax_d.set_ylabel("Net benefit per isolate (CTX)"); ax_d.set_ylim(-0.05, 0.42); ps.despine(ax_d); ps.panel_label(ax_d, "d")
    hd, ld = ax_d.get_legend_handles_labels(); ax_d.legend(hd[:2], ld[:2], loc="upper right", frameon=False, fontsize=6.5)   # references only; models in the figure legend
    model_handles = [Line2D([], [], color=col, marker=mk, ms=4, lw=1.3, label=name) for m, col, name, mk in series]
    model_handles.append(Line2D([], [], color=ps.INK2, marker="D", ms=4, mfc=ps.SURFACE, mew=1.0, ls="none", label="CIP classifier (proxy)"))
    ps.legend_above(fig, model_handles, [h.get_label() for h in model_handles], ncol=4, y=0.955)
    ps.save(fig, OUT / "fig4_stacking")


def supp_calibration():
    """Supplementary figure: reliability curves (10 equal-count bins) of binary relevance, the chain ensemble and the stacked
    model for each drug on the pooled out-of-fold predictions; Brier skill and calibration slope from scripts/04_stacking/03_stacking_clinical.py."""
    labels, y, P, _ = _pooled_predictions()
    cal = json.load(open(ROOT / "results/stats/stacking_clinical_s_orsola_logistic_full.json"))["point"]
    series = [("br", ps.ROLE["br"], "Binary relevance", "o"), ("ens", ps.ROLE["ensemble"], "Chain ensemble", "s"), ("stack", ps.ROLE["stacking"], "Stacked model", "^")]
    fig, axes = ps.figure("double", height=2.9, nrows=1, ncols=3, sharey=True, constrained_layout=True)
    for ax, d, letter in zip(axes, labels, "abc"):
        ax.plot([0, 1], [0, 1], color=ps.LIGHT, lw=0.8, ls="--", zorder=1)
        for m, col, name, mk in series:
            p = P[m][d]; bins = np.array_split(np.argsort(p), 10)
            ax.plot([p[b].mean() for b in bins], [y[d][b].mean() for b in bins], color=col, marker=mk, ms=3.5, lw=1.1, zorder=3 if m == "stack" else 2)
        short = {"br": "BR", "ens": "Chain", "stack": "Stacked"}
        txt = "\n".join(f"{short[m]}: skill {cal[d][m]['Brier_skill']:+.2f}, slope {cal[d][m]['cal_slope']:.2f}" for m, col, name, mk in series).replace("-", "\u2212")
        ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=ps.INK2, linespacing=1.4)
        ax.text(0.03, 0.97, d, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold", color=ps.INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("Mean predicted probability"); ps.despine(ax, grid=None); ps.panel_label(ax, letter)
    axes[0].set_ylabel("Observed resistance frequency")
    hs = [Line2D([], [], color=col, marker=mk, ms=3.5, lw=1.1, label=name) for m, col, name, mk in series] + [Line2D([], [], color=ps.LIGHT, lw=0.8, ls="--", label="Perfect calibration")]
    ps.legend_above(fig, hs, [h.get_label() for h in hs], y=1.0)
    ps.save(fig, ROOT / "figures/supplementary/calibration_stacking")


def supp_cross_dataset():
    """Supplementary figure: cross-dataset transfer maps (logistic regression), drawn from the stored cross-site results."""
    from mecor.visualization.performance import plot_cross_dataset_combined_figure
    plot_cross_dataset_combined_figure(ROOT / "results/cross_dataset/logistic/cross_dataset_results.json", ["CIP", "CRO", "CAZ"],
                                       output_path=ROOT / "figures/supplementary/cross_dataset_auroc.png", dpi=300,
                                       stacking_json_path=ROOT / "results/cross_dataset/logistic/cross_dataset_stacking.json")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "4", "supp"]
    if "1" in which: fig1()
    if "2" in which: fig2()
    if "4" in which: fig4()
    if "supp" in which: supp_calibration(); supp_cross_dataset()
    print("done", which)
