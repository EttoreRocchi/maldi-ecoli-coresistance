"""Directed-link ablation of the classifier chains on the Sant'Orsola cohort
(logistic regression, saved fold models, no refitting).

Question: is the chain benefit carried by the information in the upstream label
(co-resistance), by the accuracy of the upstream prediction, or by neither?

For every saved chain model (15 outer folds x 6 orderings) and every downstream
position, the downstream drug is re-predicted on the test fold with its upstream
feature column(s) set to:
  realized : hard 0/1 predictions of the upstream links (the paper's scheme; reproduces the stored probabilities)
  oracle   : the TRUE upstream labels of the test isolates (ceiling of the co-resistance information)
  permuted : the true upstream labels permuted across the test isolates (same marginal, no information; mean of 20 permutations)
  proba    : the upstream links' predicted probabilities instead of hard labels (probability passing at test time)
plus, for positions with two upstream drugs, single-link permutation (permute one upstream column, keep the other realized),
which gives a test-time permutation importance per directed link.
Metrics per fold: AUROC, MCC, VME, ME of the downstream drug; chain ensembles over the six orderings are rebuilt per variant.
Standardized logistic coefficients of the upstream columns and their rank among all features are also reported.

Outputs: results/stats/link_ablation.json, tables/link_ablation.md, figures/supplementary/link_ablation.png (+pdf).
Run with --plot-only to redraw the figure from results/stats/link_ablation.json without recomputing.
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)

from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci, load_cv_results, metric_by_fold  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.models.persistence import load_model  # noqa: E402
from mecor.shared import metrics, predict_variant, sign_flip_p  # noqa: E402

SEED = 42
N_PERM = 20
MODEL = "logistic"
MODEL_DIR = ROOT / "results/s_orsola/models"
CV_PKL = ROOT / "results/s_orsola/logistic/cv_results.pkl"
VARIANTS = ["realized", "oracle", "permuted", "proba"]


def main():
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml"))
    X, Y = load_dataset(cfg)
    labels = list(Y.columns)
    cv = load_cv_results(CV_PKL)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    folds = {}
    for g, (tr, te) in enumerate(outer.split(X, strat)):
        folds[(g // 5, g % 5)] = te
    # verify fold reconstruction against stored y_true
    for fr in cv.fold_results:
        if fr.method == "binary_relevance":
            te = folds[(fr.repeat_idx, fr.fold_idx)]
            for d in labels:
                assert np.array_equal(Y.iloc[te][d].values, np.asarray(fr.y_true[d])), "fold mismatch"
    print("fold reconstruction verified")

    orderings = [list(o) for o in permutations(labels)]
    rng = np.random.default_rng(SEED)
    # per (ordering, downstream drug, variant) -> {fold: metrics}
    rec = {}
    coefs = []
    stored_check = []
    ens_probs = {v: {d: {} for d in labels} for v in VARIANTS}  # variant -> drug -> fold -> list of probs (over orderings)
    single_perm = {}  # (ordering, downstream, permuted upstream) -> fold -> AUROC
    for (r, f), te in sorted(folds.items()):
        Xt, Yt = X.iloc[te], Y.iloc[te]
        for order in orderings:
            m = load_model(MODEL_DIR, "chain", r, f, MODEL, order)
            assert m is not None, f"missing model r{r} f{f} {order}"
            ests = m.chain.estimators_
            okey = "->".join(order)
            for k, d in enumerate(order):
                if k == 0:
                    p0 = ests[0].predict_proba(np.asarray(Xt, float))[:, 1]
                    for v in VARIANTS:
                        ens_probs[v][d].setdefault((r, f), []).append(p0)
                    continue
                for v in VARIANTS:
                    if v == "permuted":
                        ps = [predict_variant(ests, order, k, Xt, Yt, v, rng=rng) for _ in range(N_PERM)]
                        ms = [metrics(Yt[d].values, p) for p in ps]
                        mm = {key: float(np.nanmean([x[key] for x in ms])) for key in ms[0]}
                        p_mean = np.mean(ps, axis=0)
                    else:
                        p_mean = predict_variant(ests, order, k, Xt, Yt, v)
                        mm = metrics(Yt[d].values, p_mean)
                    rec.setdefault((okey, d, v), {})[(r, f)] = mm
                    ens_probs[v][d].setdefault((r, f), []).append(p_mean)
                    if v == "realized":
                        stored = [fr for fr in cv.fold_results if fr.method == "chain_" + "_".join(order)
                                  and fr.repeat_idx == r and fr.fold_idx == f][0]
                        stored_check.append(float(np.max(np.abs(np.asarray(stored.y_prob[d]) - p_mean))))
                if k == 2:
                    for u in order[:2]:
                        aucs = []
                        for _ in range(N_PERM):
                            p = predict_variant(ests, order, k, Xt, Yt, "single_perm", rng=rng, permute_only=u)
                            aucs.append(metrics(Yt[d].values, p)["AUROC"])
                        single_perm.setdefault((okey, d, u), {})[(r, f)] = float(np.nanmean(aucs))
                # coefficients of upstream columns (standardized features; scaler is inside the pipeline)
                est = ests[k]
                pipe = est.best_estimator_ if hasattr(est, "best_estimator_") else est
                clf = pipe.named_steps["clf"]
                coef = np.asarray(clf.coef_).ravel()
                n_feat = coef.size - k
                ranks = (-np.abs(coef)).argsort().argsort() + 1
                for j in range(k):
                    coefs.append({"ordering": okey, "downstream": d, "upstream": order[j], "repeat": r, "fold": f,
                                  "coef": float(coef[n_feat + j]), "rank": int(ranks[n_feat + j]),
                                  "n_features": int(coef.size)})
    print(f"max |stored - realized| over all chain probabilities: {max(stored_check):.2e}")

    br = {d: {m: metric_by_fold(cv, "binary_relevance", d, m) for m in ("AUROC", "MCC")} for d in labels}
    brc = {d: {m: metric_by_fold(cv, "binary_relevance", d, m, clinical=True) for m in ("VME", "ME")} for d in labels}

    out = {"per_ordering": [], "ensemble": [], "single_link_permutation": [], "coefficients": []}
    rows = []
    for (okey, d, v), fm in sorted(rec.items()):
        keys = sorted(fm)
        row = {"ordering": okey, "downstream": d, "variant": v}
        for met in ("AUROC", "MCC", "VME", "ME"):
            vals = np.array([fm[kk][met] for kk in keys])
            ref = (br if met in ("AUROC", "MCC") else brc)[d][met]
            refv = np.array([ref[kk] for kk in keys])
            dlt = vals - refv
            row[met] = float(np.nanmean(vals))
            row[f"d{met}"] = float(np.nanmean(dlt))
            if met in ("AUROC", "MCC"):
                _, lo, hi = bca_mean_ci(dlt)
                row[f"d{met}_ci"] = [lo, hi]
                row[f"d{met}_p"] = sign_flip_p(dlt)
        out["per_ordering"].append(row)
        rows.append(row)

    for v in VARIANTS:
        for d in labels:
            fm = {}
            for kk, plist in ens_probs[v][d].items():
                assert len(plist) == 6
                te = folds[kk]
                fm[kk] = metrics(Y.iloc[te][d].values, np.mean(plist, axis=0))
            keys = sorted(fm)
            row = {"variant": v, "drug": d}
            for met in ("AUROC", "MCC", "VME", "ME"):
                vals = np.array([fm[kk][met] for kk in keys])
                ref = (br if met in ("AUROC", "MCC") else brc)[d][met]
                dlt = vals - np.array([ref[kk] for kk in keys])
                row[met] = float(np.nanmean(vals))
                row[f"d{met}"] = float(np.nanmean(dlt))
                if met in ("AUROC", "MCC"):
                    _, lo, hi = bca_mean_ci(dlt)
                    row[f"d{met}_ci"] = [lo, hi]
                    row[f"d{met}_p"] = sign_flip_p(dlt)
            out["ensemble"].append(row)

    for (okey, d, u), fm in sorted(single_perm.items()):
        keys = sorted(fm)
        real = rec[(okey, d, "realized")]
        drop = np.array([real[kk]["AUROC"] - fm[kk] for kk in keys])
        out["single_link_permutation"].append({"ordering": okey, "downstream": d, "permuted_upstream": u,
                                               "AUROC_realized": float(np.mean([real[kk]["AUROC"] for kk in keys])),
                                               "AUROC_permuted": float(np.mean([fm[kk] for kk in keys])),
                                               "AUROC_drop": float(drop.mean()), "drop_ci": list(bca_mean_ci(drop)[1:]),
                                               "p": sign_flip_p(drop)})
    cdf = pd.DataFrame(coefs)
    for (okey, d, u), g in cdf.groupby(["ordering", "downstream", "upstream"]):
        out["coefficients"].append({"ordering": okey, "downstream": d, "upstream": u,
                                    "coef_mean": float(g.coef.mean()), "coef_sd": float(g.coef.std(ddof=1)),
                                    "rank_median": float(g["rank"].median()), "rank_min": int(g["rank"].min()),
                                    "rank_max": int(g["rank"].max()), "n_features": int(g.n_features.iloc[0])})

    out["br_reference"] = {d: {m: float(np.mean(list(br[d][m].values()))) for m in ("AUROC", "MCC")} for d in labels}
    out["notes"] = {"n_perm": N_PERM, "max_abs_diff_stored_vs_realized": float(max(stored_check)),
                    "variants": VARIANTS}
    (ROOT / "results/stats").mkdir(exist_ok=True, parents=True)
    (ROOT / "results/stats/link_ablation.json").write_text(json.dumps(out, indent=1))

    df = pd.DataFrame(rows)
    lines = ["# Directed-link ablation (logistic regression, Sant'Orsola, 15 outer folds)", "",
             "Reference BR AUROC: " + ", ".join(f"{d} {out['br_reference'][d]['AUROC']:.3f}" for d in labels), "",
             "## Per ordering and downstream drug (AUROC; delta versus BR with sign-flip p)", "",
             "| Ordering | Downstream | Upstream input | realized | oracle | permuted | proba |", "|---|---|---|---|---|---|---|"]
    for (okey, d), g in df.groupby(["ordering", "downstream"], sort=False):
        order = okey.split("->")
        ups = ", ".join(order[:order.index(d)])
        cells = []
        for v in VARIANTS:
            rr = g[g.variant == v].iloc[0]
            cells.append(f"{rr.AUROC:.3f} ({rr.dAUROC:+.3f}, p={rr.dAUROC_p:.3f})")
        lines.append(f"| {okey} | {d} | {ups} | " + " | ".join(cells) + " |")
    lines += ["", "## Chain ensemble rebuilt per variant (delta versus BR)", "",
              "| Variant | Drug | AUROC | dAUROC [CI] p | MCC | dMCC [CI] p | VME | ME |", "|---|---|---|---|---|---|---|---|"]
    for row in out["ensemble"]:
        lines.append(f"| {row['variant']} | {row['drug']} | {row['AUROC']:.3f} | {row['dAUROC']:+.3f} [{row['dAUROC_ci'][0]:+.3f}, {row['dAUROC_ci'][1]:+.3f}] p={row['dAUROC_p']:.3f} | {row['MCC']:.3f} | {row['dMCC']:+.3f} [{row['dMCC_ci'][0]:+.3f}, {row['dMCC_ci'][1]:+.3f}] p={row['dMCC_p']:.3f} | {row['VME']:.3f} | {row['ME']:.3f} |")
    lines += ["", "## Single-link test-time permutation importance (position 3 only; AUROC drop when one upstream column is permuted)", "",
              "| Ordering | Downstream | Permuted upstream | AUROC realized | AUROC permuted | Drop [CI] | p |", "|---|---|---|---|---|---|---|"]
    for row in out["single_link_permutation"]:
        lines.append(f"| {row['ordering']} | {row['downstream']} | {row['permuted_upstream']} | {row['AUROC_realized']:.3f} | {row['AUROC_permuted']:.3f} | {row['AUROC_drop']:+.3f} [{row['drop_ci'][0]:+.3f}, {row['drop_ci'][1]:+.3f}] | {row['p']:.3f} |")
    lines += ["", "## Standardized coefficient of each upstream column (mean over 15 folds) and its rank among all features", "",
              "| Ordering | Downstream | Upstream | coef (sd) | rank median [min, max] | n features |", "|---|---|---|---|---|---|"]
    for row in out["coefficients"]:
        lines.append(f"| {row['ordering']} | {row['downstream']} | {row['upstream']} | {row['coef_mean']:+.3f} ({row['coef_sd']:.3f}) | {row['rank_median']:.0f} [{row['rank_min']}, {row['rank_max']}] | {row['n_features']} |")
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables/link_ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_from_results()
    print("saved figure and JSON")


def plot_from_results(stats_path=ROOT / "results/stats/link_ablation.json", out_stem=ROOT / "figures/supplementary/link_ablation"):
    """Supplementary figure from the stored JSON only: downstream AUROC per ordering under the four upstream variants
    (dot-and-interval; the interval is the BCa 95% interval of the paired difference versus BR shifted to the AUROC scale),
    one panel per downstream drug, binary relevance as a horizontal reference line, one legend above."""
    from matplotlib.lines import Line2D

    from mecor.visualization import paper_style as ps
    out = json.load(open(stats_path))
    labels = list(out["br_reference"])
    col = {"realized": ps.ROLE["ensemble"], "proba": ps.ROLE["proba"], "oracle": ps.ROLE["oracle"], "permuted": ps.ROLE["permuted"]}
    name = {"realized": "Realized prediction", "proba": "Probability passing", "oracle": "True label (oracle)", "permuted": "Permuted label"}
    off = {"realized": -0.27, "proba": -0.09, "oracle": 0.09, "permuted": 0.27}
    fig, axes = ps.figure("double", height=2.9, ncols=3, constrained_layout=True)
    for ax, d, letter in zip(axes, labels, "abc"):
        rows = [r for r in out["per_ordering"] if r["downstream"] == d]
        oks = list(dict.fromkeys(r["ordering"] for r in rows))
        for j, o in enumerate(oks):
            for v in off:
                r = next(x for x in rows if x["ordering"] == o and x["variant"] == v)
                lo = r["AUROC"] + (r["dAUROC_ci"][0] - r["dAUROC"]); hi = r["AUROC"] + (r["dAUROC_ci"][1] - r["dAUROC"])
                ps.ci_point(ax, j + off[v], r["AUROC"], lo, hi, color=col[v], hollow=(v == "proba"))
        ps.hline(ax, out["br_reference"][d]["AUROC"], color=ps.ROLE["br"], lw=0.9)
        ax.set_xticks(range(len(oks))); ax.set_xticklabels([o.replace("->", ">") for o in oks], rotation=35, ha="right"); ax.set_xlim(-0.6, len(oks) - 0.4)
        ax.set_ylabel(f"Downstream {d} AUROC"); ax.set_ylim(0.4, 1.0); ps.despine(ax); ps.panel_label(ax, letter)
    handles = [Line2D([], [], ls="none", marker="o", ms=4.5, color=col[v], mfc=ps.SURFACE if v == "proba" else col[v], mec=col[v], mew=1.0, label=name[v]) for v in off]
    handles.append(Line2D([], [], ls="-", lw=0.9, color=ps.ROLE["br"], label="Binary relevance"))
    ps.legend_above(fig, handles, [h.get_label() for h in handles], ncol=5, y=1.0)
    return ps.save(fig, out_stem)


if __name__ == "__main__":
    if "--plot-only" in sys.argv[1:]:
        plot_from_results()
    else:
        main()
