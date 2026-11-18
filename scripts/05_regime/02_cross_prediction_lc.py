"""Cross-prediction along the learning curves: does proxy superiority track the label-dependency gain?

Refits the three binary-relevance classifiers of scripts/05_regime/01_learning_curve.py (--stacking runs) on exactly the same subsamples and folds,
scores every classifier against every label on the test fold, and pairs the proxy margin (best other-label classifier
minus the label's own classifier) with the stored chain and stacking gains of the same folds. The refitted own-label
AUROC is checked against the stored binary-relevance AUROC fold by fold. The expected pattern was recorded before the margins were computed.
Checkpoints: results/cross_prediction_lc/<site>/n<n>_draw<k>.json
Outputs: results/stats/cross_prediction_lc.json, tables/cross_prediction_lc.md,
figures/supplementary/cross_prediction_lc.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.evaluation.checkpoint import atomic_json_dump, ensure_signature, load_json_or_none  # noqa: E402
from mecor.evaluation.cv import _build_search_estimator  # noqa: E402
from mecor.evaluation.grouping import check_no_patient_overlap, inner_splits, load_patient_groups, outer_splits, subsample  # noqa: E402
from mecor.shared import LEARNING_CURVE_PLAN as PLAN  # noqa: E402
from mecor.visualization import paper_style as ps  # noqa: E402
from mecor.visualization.cross_prediction import draw  # noqa: E402

FULL_SITE = {"driams_a": "driams_a_logistic", "driams_a_grouped": "driams_a_grouped_logistic"}   # full-site cross-prediction from scripts/04_stacking/02_cross_prediction.py (same reduced grid)
SITE_NAME = {"driams_a": "DRIAMS-A", "s_orsola": "Sant'Orsola", "driams_a_grouped": "DRIAMS-A"}


def run_draw(X, Y, cfg, drugs, n_repeats, seed, groups=None):
    """Per fold: 3 x 3 AUROC matrix (row = label, column = classifier)."""
    out = []
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    if groups is None:
        splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed).split(X, strat)
    else:   # the same patient-grouped folds as scripts/05_regime/01_learning_curve.py
        splits = outer_splits(strat.values, groups, 5, n_repeats, seed); check_no_patient_overlap(splits, groups)
    for tr, te in splits:
        P = []
        inner = None if groups is None else inner_splits(strat.values[tr], groups[tr], cfg.n_splits_inner, cfg.seed)
        for d in drugs:
            est = _build_search_estimator("logistic", cfg, inner); est.fit(X[tr], Y[d].values[tr]); P.append(est.predict_proba(X[te])[:, 1])
        out.append([[float(roc_auc_score(Y[a].values[te], P[j])) for j in range(3)] for a in drugs])
    return out


def contrast(d):
    m, lo, hi = bca_mean_ci(np.asarray(d, float)); return {"delta": m, "ci": [lo, hi]}


def plot(rows, rho, suffix=""):
    fig, ax = ps.figure("onehalf", height=4.1)
    draw(ax, rows, rho)
    print("saved", ps.save(fig, ROOT / f"figures/supplementary/cross_prediction_lc{suffix}"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sites", default="driams_a,s_orsola"); ap.add_argument("--n-jobs", type=int, default=32)
    args = ap.parse_args()
    sites = args.sites.split(",")
    suffix = "_grouped" if any(s.endswith("_grouped") for s in sites) else ""   # the patient-grouped run keeps its own file names
    rows = []
    for site in sites:
        plan = PLAN[site]; drugs = (plan["up"], plan["mid"], plan["last"])
        cfg = Config.from_yaml(str(ROOT / f"configs/{site}.yaml")); cfg.model = "logistic"; cfg.param_search_mode = "liblinear"; cfg.n_jobs = args.n_jobs
        X, Y = load_dataset(cfg); Xa = np.asarray(X, float)
        strat_all = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
        groups_all = load_patient_groups(cfg, X.index)   # None unless the site config groups by patient
        ck = ROOT / "results/cross_prediction_lc" / site; ck.mkdir(parents=True, exist_ok=True)
        ensure_signature(ck, {"site": site, "grid": "liblinear", "group_by_patient": groups_all is not None, "n_isolates": int(len(Y)),
                              "draws": plan["draws"], "repeats": plan["repeats"], "drugs": list(drugs)})
        for n in plan["sizes"]:
            draws = 1 if n >= len(Y) else plan["draws"]
            lc_dir = ROOT / f"results/learning_curve/{site}_stack"
            if not lc_dir.exists(): lc_dir = ROOT / f"results/learning_curve/{site}_stack_inner"   # runs saved with scripts/05_regime/01_learning_curve.py --inner
            folds, stored = [], []
            for k in range(draws):
                f = ck / f"n{n}_draw{k}.json"
                st = json.load(open(lc_dir / f"n{n}_draw{k}.json"))
                mats = load_json_or_none(f)
                if mats is None and all("test_auc" in s for s in st):   # --inner runs stored the test matrix of the same fitted classifiers
                    assert all(s["drugs"] == list(drugs) for s in st); mats = [s["test_auc"] for s in st]; atomic_json_dump(mats, f)
                if mats is None:
                    if n >= len(Y): idx = np.arange(len(Y))
                    elif groups_all is None: idx = train_test_split(np.arange(len(Y)), train_size=n, stratify=strat_all, random_state=100 + k)[0]
                    else: idx = subsample(n, strat_all.values, groups_all, 100 + k)   # the same patient subsample as scripts/05_regime/01_learning_curve.py
                    mats = run_draw(Xa[idx], Y.iloc[idx].reset_index(drop=True), cfg, drugs, plan["repeats"], 42 + k,
                                    groups=None if groups_all is None else groups_all[idx])
                    atomic_json_dump(mats, f); print(f"{site} n={n} draw {k} done", flush=True)
                folds += mats; stored += st
            A_all = np.array(folds, float)   # fold x label x classifier
            for i, d in ((1, plan["mid"]), (2, plan["last"])):
                br_all = np.array([s[f"br_{d}"]["AUROC"] for s in stored], float)
                gain_all = {g: np.array([s[f"{key}_{d}"]["AUROC"] for s in stored], float) - br_all for g, key in (("stack", "stack"), ("hard", "chain_hard"), ("proba", "chain_proba"))}
                ok = np.isfinite(A_all[:, i, :]).all(axis=1) & np.all([np.isfinite(v) for v in gain_all.values()], axis=0)   # test folds without both classes have no AUROC
                A, br, gains = A_all[ok], br_all[ok], {g: v[ok] for g, v in gain_all.items()}
                own = A[:, i, i]; others = [j for j in range(3) if j != i]; j = max(others, key=lambda c: A[:, i, c].mean())
                rows.append({"site": site, "n": n, "label": d, "folds": len(own), "own": float(own.mean()), "proxy": drugs[j], "proxy_auroc": float(A[:, i, j].mean()),
                             "margin": contrast(A[:, i, j] - own), "check_max_abs_diff": float(np.max(np.abs(own - br))),
                             **{f"gain_{g}": contrast(v) for g, v in gains.items()},
                             "fold_rho_margin_stack": float(spearmanr(A[:, i, j] - own, gains["stack"]).statistic)})
        cp_file = ROOT / f"results/stats/cross_prediction{'_grouped' if site.endswith('_grouped') else ''}.json"
        if site in FULL_SITE and cp_file.exists():   # full-site point from the nested stacking run (scripts/04_stacking/02_cross_prediction.py)
            for r in json.load(open(cp_file)):
                if r["tag"] == FULL_SITE[site] and r["label"] in (plan["mid"], plan["last"]):
                    rows.append({"site": site, "n": "full", "label": r["label"], "own": r["auroc_by_classifier"][r["label"]], "proxy": r["proxy"], "proxy_auroc": r["auroc_by_classifier"][r["proxy"]],
                                 "margin": r["proxy_minus_own"], "gain_stack": {"delta": r["stacked"] - r["auroc_by_classifier"][r["label"]], "ci": [np.nan, np.nan]}})
    pts = [r for r in rows if r["n"] != "full"]
    rho = spearmanr([r["margin"]["delta"] for r in pts], [r["gain_stack"]["delta"] for r in pts])
    rho_hard = spearmanr([r["margin"]["delta"] for r in pts], [r["gain_hard"]["delta"] for r in pts])
    summary = {"rows": rows, "spearman_margin_vs_stack": [float(rho.statistic), float(rho.pvalue)], "spearman_margin_vs_hard": [float(rho_hard.statistic), float(rho_hard.pvalue)], "n_points": len(pts)}
    atomic_json_dump(summary, ROOT / f"results/stats/cross_prediction_lc{suffix}.json", indent=1)
    f = lambda c: f"{c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
    lines = ["# Proxy superiority along the learning curves (logistic regression, reduced grid; same subsamples and folds as scripts/05_regime/01_learning_curve.py)", "",
             "Margin = AUROC of the best other-label classifier minus the label's own classifier (paired per fold, BCa 95% CI). Gains are the stored gains of the same folds.", "",
             "| Site | n | Label | Own BR AUROC | Proxy (AUROC) | Proxy margin [CI] | Stacking gain [CI] | Chain gain hard [CI] | Chain gain proba [CI] | Refit check (max abs diff) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["n"] == "full":
            lines.append(f"| {SITE_NAME[r['site']]} | full | {r['label']} | {r['own']:.3f} | {r['proxy']} ({r['proxy_auroc']:.3f}) | {f(r['margin'])} | {r['gain_stack']['delta']:+.3f} | n/a | n/a | n/a |")
        else:
            lines.append(f"| {SITE_NAME[r['site']]} | {r['n']} | {r['label']} | {r['own']:.3f} | {r['proxy']} ({r['proxy_auroc']:.3f}) | {f(r['margin'])} | {f(r['gain_stack'])} | {f(r['gain_hard'])} | {f(r['gain_proba'])} | {r['check_max_abs_diff']:.3f} |")
    lines += ["", f"Spearman (margin vs stacking gain, {len(pts)} size-by-label points): rho = {rho.statistic:+.2f}, p = {rho.pvalue:.3f}; margin vs hard chain gain: rho = {rho_hard.statistic:+.2f}, p = {rho_hard.pvalue:.3f}"]
    (ROOT / f"tables/cross_prediction_lc{suffix}.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    plot(rows, float(rho.statistic), suffix)


if __name__ == "__main__":
    main()
