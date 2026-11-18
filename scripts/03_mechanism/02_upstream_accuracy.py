"""Upstream accuracy at constant co-resistance (Sant'Orsola, logistic regression, saved downstream models).

Question: at fixed co-resistance (the labels and phi are untouched), does the downstream cephalosporin gain
depend on how accurately ciprofloxacin is predicted?

For the orderings CIP -> CTX -> CAZ and CIP -> CAZ -> CTX the saved downstream models are kept and the
upstream CIP prediction is replaced by a deliberately degraded CIP classifier trained on the same training fold
with a random subset of the m/z bins (fractions 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0; five random subsets each),
using the hyperparameters selected by that fold's saved BR GridSearchCV for CIP. The downstream drug at position 2
receives the degraded CIP prediction as a hard label (the paper's scheme) or as a probability (probability passing).
For each degradation level: CIP AUROC and the downstream AUROC and MCC, averaged over the 15 folds; the relation
between upstream accuracy and downstream gain is summarized by Spearman correlation across levels.

Outputs: results/stats/upstream_accuracy.json, tables/upstream_accuracy.md,
         figures/supplementary/upstream_accuracy.png (+pdf)
Run with --plot-only to redraw the figure from results/stats/upstream_accuracy.json without recomputing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor import shared as la
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci, load_cv_results, metric_by_fold  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.models.persistence import load_model  # noqa: E402

SEED = 42
N_JOBS = 16
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
N_SUB = 5
MODEL_DIR = ROOT / "results/s_orsola/models"
CV_PKL = ROOT / "results/s_orsola/logistic/cv_results.pkl"
ORDERS = [["CIP", "CTX", "CAZ"], ["CIP", "CAZ", "CTX"]]


def degraded_cip(params, Xtr, ytr, Xte, frac, seed):
    rng = np.random.default_rng(seed)
    n = max(1, int(round(frac * Xtr.shape[1])))
    cols = np.sort(rng.choice(Xtr.shape[1], n, replace=False))
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED, n_jobs=1,
                             C=params["clf__C"], penalty=params["clf__penalty"], solver=params["clf__solver"])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(Xtr[:, cols], ytr)
    return pipe.predict_proba(Xte[:, cols])[:, 1]


def main():
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml"))
    X, Y = load_dataset(cfg)
    labels = list(Y.columns)
    Xa = np.asarray(X, float)
    cv = load_cv_results(CV_PKL)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    folds = {}
    for g, (tr, te) in enumerate(outer.split(X, strat)):
        folds[(g // 5, g % 5)] = (tr, te)
    br_auc = {d: metric_by_fold(cv, "binary_relevance", d, "AUROC") for d in labels}
    br_mcc = {d: metric_by_fold(cv, "binary_relevance", d, "MCC") for d in labels}

    rec = {}  # (order, downstream, passing, frac) -> {fold: (cip_auc, down_auc, down_mcc)}
    for (r, f), (tr, te) in sorted(folds.items()):
        Xtr, Xte, Ytr, Yte = Xa[tr], Xa[te], Y.iloc[tr], Y.iloc[te]
        br = load_model(MODEL_DIR, "binary_relevance", r, f, "logistic")
        params = br.estimators_["CIP"].best_params_
        jobs = [(frac, s) for frac in FRACTIONS for s in range(N_SUB)]
        preds = Parallel(n_jobs=N_JOBS)(delayed(degraded_cip)(params, Xtr, Ytr["CIP"].values, Xte, frac, SEED + 1000 * r + 100 * f + 7 * s)
                                        for frac, s in jobs)
        chains = {tuple(o): load_model(MODEL_DIR, "chain", r, f, "logistic", o) for o in ORDERS}
        for (frac, s), p_cip in zip(jobs, preds):
            cip_auc = la.metrics(Yte["CIP"].values, p_cip)["AUROC"]
            for o in ORDERS:
                ests = chains[tuple(o)].chain.estimators_
                d = o[1]
                for passing in ("hard", "proba"):
                    col = (p_cip >= 0.5).astype(float) if passing == "hard" else p_cip
                    p_d = ests[1].predict_proba(np.column_stack([Xte, col]))[:, 1]
                    m = la.metrics(Yte[d].values, p_d)
                    rec.setdefault((">".join(o), d, passing, frac), {}).setdefault((r, f), []).append((cip_auc, m["AUROC"], m["MCC"]))
        print(f"fold r{r} f{f} done", flush=True)

    out = {"levels": [], "spearman": {}}
    for (okey, d, passing, frac), fm in sorted(rec.items()):
        keys = sorted(fm)
        cip = np.array([np.mean([x[0] for x in fm[k]]) for k in keys])
        dauc = np.array([np.mean([x[1] for x in fm[k]]) - br_auc[d][k] for k in keys])
        dmcc = np.array([np.mean([x[2] for x in fm[k]]) - br_mcc[d][k] for k in keys])
        _, lo, hi = bca_mean_ci(dauc)
        out["levels"].append({"ordering": okey, "downstream": d, "passing": passing, "fraction": frac,
                              "cip_auroc": float(cip.mean()), "downstream_dAUROC": float(dauc.mean()), "dAUROC_ci": [lo, hi],
                              "dAUROC_p": la.sign_flip_p(dauc), "downstream_dMCC": float(dmcc.mean())})
    for okey in [">".join(o) for o in ORDERS]:
        for passing in ("hard", "proba"):
            rows = [x for x in out["levels"] if x["ordering"] == okey and x["passing"] == passing]
            rho, p = spearmanr([x["cip_auroc"] for x in rows], [x["downstream_dAUROC"] for x in rows])
            out["spearman"][f"{okey}|{passing}"] = {"rho": float(rho), "p": float(p), "n_levels": len(rows)}
    (ROOT / "results/stats/upstream_accuracy.json").write_text(json.dumps(out, indent=1))

    lines = ["# Upstream accuracy at constant co-resistance (degraded CIP classifier feeding the saved downstream models)", "",
             "| Ordering | Downstream | Passing | Bin fraction | CIP AUROC | downstream dAUROC [CI] p | downstream dMCC |", "|---|---|---|---|---|---|---|"]
    for x in out["levels"]:
        lines.append(f"| {x['ordering']} | {x['downstream']} | {x['passing']} | {x['fraction']} | {x['cip_auroc']:.3f} | {x['downstream_dAUROC']:+.3f} [{x['dAUROC_ci'][0]:+.3f}, {x['dAUROC_ci'][1]:+.3f}] p={x['dAUROC_p']:.3f} | {x['downstream_dMCC']:+.3f} |")
    lines += ["", "Spearman correlation between CIP AUROC and downstream gain across degradation levels: " +
              "; ".join(f"{k}: rho={v['rho']:.2f} (p={v['p']:.3f})" for k, v in out["spearman"].items())]
    (ROOT / "tables/upstream_accuracy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_from_results()


def break_even(xs, ys):
    """First x at which the gain crosses zero from below (linear interpolation); None if it never does."""
    for i in range(1, len(xs)):
        if ys[i - 1] < 0 <= ys[i]:
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (-ys[i - 1]) / (ys[i] - ys[i - 1]))
    return None


def plot_from_results(stats_path=ROOT / "results/stats/upstream_accuracy.json", out_stem=ROOT / "figures/supplementary/upstream_accuracy"):
    """Supplementary figure from the stored JSON only: downstream gain over BR against the upstream CIP AUROC, hard label
    passing (solid, filled) and probability passing (dashed, hollow) in the chain hue, BCa 95% intervals, zero line and
    break-even upstream AUROC as ticks on the horizontal axis; one panel per ordering, one legend above."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.lines import Line2D

    from mecor.visualization import paper_style as ps
    out = json.load(open(stats_path))
    orders = list(dict.fromkeys(x["ordering"] for x in out["levels"]))
    orders = sorted(orders, key=lambda o: [">".join(oo) for oo in ORDERS].index(o) if o in [">".join(oo) for oo in ORDERS] else 99)
    blue = ps.ROLE["ensemble"]
    fig, axes = ps.figure("double", height=2.8, ncols=len(orders), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    series = lambda okey, passing: sorted([x for x in out["levels"] if x["ordering"] == okey and x["passing"] == passing], key=lambda x: x["cip_auroc"])
    be = {}
    for ax, okey, letter in zip(axes, orders, "abcdef"):
        d = okey.split(">")[1]
        ps.hline(ax, 0.0)
        for passing, ls, hollow in (("hard", "-", False), ("proba", "--", True)):
            rows = series(okey, passing)
            xs = [x["cip_auroc"] for x in rows]; ys = [x["downstream_dAUROC"] for x in rows]
            ax.plot(xs, ys, ls=ls, lw=1.2, color=blue, zorder=2)
            ps.ci_point(ax, xs, ys, [x["dAUROC_ci"][0] for x in rows], [x["dAUROC_ci"][1] for x in rows], color=blue, hollow=hollow)
            be[(okey, passing)] = break_even(xs, ys)
        ax.set_xlabel("Upstream CIP AUROC (degraded classifier)")
        ax.text(0.03, 0.97, f"{d} downstream ({okey})", transform=ax.transAxes, va="top", ha="left", color=ps.INK2)
        ps.despine(ax); ps.panel_label(ax, letter)
    axes[0].set_ylabel("Downstream gain over binary relevance (AUROC)")
    for ax in axes: ax.margins(y=0.12)
    y0, y1 = axes[0].get_ylim(); axes[0].set_ylim(y0, y1)
    for ax, okey in zip(axes, orders):   # break-even ticks on the horizontal axis (solid: hard passing, dotted: probability passing)
        for passing, ls in (("hard", "-"), ("proba", (0, (1.5, 1.5)))):
            if be[(okey, passing)] is not None:
                ax.plot([be[(okey, passing)]] * 2, [y0, y0 + 0.045 * (y1 - y0)], ls=ls, lw=1.0, color=ps.INK2, zorder=4, clip_on=False, solid_capstyle="butt")
        bes = [b for b in (be[(okey, "hard")], be[(okey, "proba")]) if b is not None]
        if bes:
            txt = f"break-even {min(bes):.2f} to {max(bes):.2f}" if len(bes) == 2 and abs(bes[0] - bes[1]) > 0.005 else f"break-even {bes[0]:.2f}"
            ax.text(float(np.mean(bes)), y0 + 0.06 * (y1 - y0), txt, ha="center", va="bottom", fontsize=6.5, color=ps.INK2)
    handles = [Line2D([], [], ls="-", lw=1.2, marker="o", ms=4.5, color=blue, label="Hard label passing"),
               Line2D([], [], ls="--", lw=1.2, marker="o", ms=4.5, color=blue, mfc=ps.SURFACE, mew=1.0, label="Probability passing")]
    ps.legend_above(fig, handles, [h.get_label() for h in handles], ncol=2, y=1.0, handlelength=2.4)
    return ps.save(fig, out_stem)


if __name__ == "__main__":
    if "--plot-only" in sys.argv[1:]:
        plot_from_results()
    else:
        main()
