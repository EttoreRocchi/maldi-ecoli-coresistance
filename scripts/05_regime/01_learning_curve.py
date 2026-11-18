"""Learning curve of the chain benefit versus training-set size (logistic regression).

Sant'Orsola: stratified subsamples of n = 60, 90, 120, 150 (five draws each) and the full 187; DRIAMS-A: subsamples of
n = 187, 500, 1000, 2000 (three draws each). For each subsample, 5-fold cross-validation (two repeats at Sant'Orsola,
one at DRIAMS-A) with inner 3-fold tuning on the reduced L2 grid, comparing binary relevance with the a priori chain
CIP -> CTX/CRO -> CAZ (hard and probability passing at prediction). Reports the per-size BR AUROC of the cephalosporins
(headroom) and the chain gain with BCa intervals.
Outputs: results/stats/learning_curve.json, tables/learning_curve.md, figures/supplementary/learning_curve.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, train_test_split

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.evaluation.checkpoint import atomic_json_dump, ensure_signature, load_json_or_none  # noqa: E402
from mecor.evaluation.cv import _build_search_estimator  # noqa: E402
from mecor.evaluation.grouping import check_no_patient_overlap, inner_splits, load_patient_groups, outer_splits, subsample  # noqa: E402
from mecor.shared import LEARNING_CURVE_PLAN as PLAN  # noqa: E402

PLAN["driams_a_grouped"] = dict(PLAN["driams_a"])   # same plan; subsamples are whole patients and folds are patient-grouped


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan, "MCC": float(matthews_corrcoef(y, yhat))}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def run_one(X, Y, cfg, up, mid, last, n_repeats, seed, stacking=False, groups=None, inner_auc=False):
    out = []
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    if groups is None:
        splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed).split(X, strat)
    else:
        splits = outer_splits(strat.values, groups, 5, n_repeats, seed); check_no_patient_overlap(splits, groups)
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]; rec = {}
        g_tr = None if groups is None else groups[tr]; s_tr = strat.values[tr]
        inner = None if g_tr is None else inner_splits(s_tr, g_tr, cfg.n_splits_inner, cfg.seed)
        br = {}
        for d in (up, mid, last):
            est = _build_search_estimator("logistic", cfg, inner); est.fit(Xtr, Y[d].values[tr]); br[d] = est
        p_up = br[up].predict_proba(Xte)[:, 1]
        link_mid = _build_search_estimator("logistic", cfg, inner); link_mid.fit(np.column_stack([Xtr, Y[up].values[tr]]), Y[mid].values[tr])
        link_last = _build_search_estimator("logistic", cfg, inner); link_last.fit(np.column_stack([Xtr, Y[up].values[tr], Y[mid].values[tr]]), Y[last].values[tr])
        for d in (mid, last):
            rec[f"br_{d}"] = metrics(Y[d].values[te], br[d].predict_proba(Xte)[:, 1])
        for mode in ("hard", "proba"):
            u_in = (p_up >= 0.5).astype(float) if mode == "hard" else p_up
            p_mid = link_mid.predict_proba(np.column_stack([Xte, u_in]))[:, 1]
            m_in = (p_mid >= 0.5).astype(float) if mode == "hard" else p_mid
            p_last = link_last.predict_proba(np.column_stack([Xte, u_in, m_in]))[:, 1]
            rec[f"chain_{mode}_{mid}"] = metrics(Y[mid].values[te], p_mid); rec[f"chain_{mode}_{last}"] = metrics(Y[last].values[te], p_last)
        if stacking:  # cross-fitted stacking of the three BR probabilities (5-fold inside the training fold), meta-logistic per drug
            drugs = (up, mid, last); Ptr = np.zeros((len(tr), 3)); Pte = np.column_stack([br[d].predict_proba(Xte)[:, 1] for d in drugs])
            for j, d in enumerate(drugs):
                if g_tr is None:
                    Ptr[:, j] = cross_val_predict(_build_search_estimator("logistic", cfg), Xtr, Y[d].values[tr], cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
                else:   # patient-grouped cross-fitting, each fit with patient-grouped inner folds
                    ytr = Y[d].values[tr]
                    for a, b in inner_splits(s_tr, g_tr, 5, 42):
                        est_k = _build_search_estimator("logistic", cfg, inner_splits(s_tr[a], g_tr[a], cfg.n_splits_inner, cfg.seed))
                        est_k.fit(Xtr[a], ytr[a]); Ptr[b, j] = est_k.predict_proba(Xtr[b])[:, 1]
            if inner_auc:   # cross-prediction AUROC matrices [label i scored by classifier j], training-fold (cross-fitted) and test
                auc = lambda y, p: float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
                rec["drugs"] = list(drugs)
                rec["inner_auc"] = [[auc(Y[a].values[tr], Ptr[:, j]) for j in range(3)] for a in drugs]
                rec["test_auc"] = [[auc(Y[a].values[te], Pte[:, j]) for j in range(3)] for a in drugs]
            for j, d in enumerate((mid, last)):
                meta_cv = StratifiedKFold(3, shuffle=True, random_state=42) if g_tr is None else inner_splits(s_tr, g_tr, 3, 42)
                meta = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=meta_cv, scoring="roc_auc")
                meta.fit(_logit(Ptr), Y[d].values[tr]); rec[f"stack_{d}"] = metrics(Y[d].values[te], meta.predict_proba(_logit(Pte))[:, 1])
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--site", required=True); ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--grid", default="liblinear", choices=["reduced", "liblinear", "full", "quick"]); ap.add_argument("--sizes", default=None, help="comma-separated subset of sizes"); ap.add_argument("--stacking", action="store_true", help="also fit cross-fitted stacking (separate output tag _stack)")
    ap.add_argument("--inner", action="store_true", help="with --stacking: also store the training-fold and test cross-prediction AUROC matrices (tag _stack_inner)")
    args = ap.parse_args()
    site = args.site; plan = dict(PLAN[site])
    if args.sizes: plan["sizes"] = [int(s) for s in args.sizes.split(",")]
    cfg = Config.from_yaml(str(ROOT / f"configs/{site}.yaml")); cfg.model = "logistic"; cfg.param_search_mode = args.grid; cfg.n_jobs = args.n_jobs
    X, Y = load_dataset(cfg); Xa = np.asarray(X, float)
    strat_all = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    groups_all = load_patient_groups(cfg, X.index)   # None unless the site config groups by patient (driams_a_grouped)
    tag = site if args.grid == "liblinear" else f"{site}_{args.grid}"
    if args.stacking: tag += "_stack"
    if args.inner: assert args.stacking, "--inner needs --stacking"; tag += "_inner"
    out_dir = ROOT / "results/learning_curve" / tag; out_dir.mkdir(parents=True, exist_ok=True)
    # checkpoints are valid only for these settings: refuse to resume with anything else
    ensure_signature(out_dir, {"site": site, "grid": args.grid, "stacking": args.stacking, "group_by_patient": groups_all is not None,
                               "n_isolates": int(len(Y)), "draws": plan["draws"], "repeats": plan["repeats"],
                               "chain": [plan["up"], plan["mid"], plan["last"]]} | ({"inner": True} if args.inner else {})
                   | ({"subsample": "patients stratified by profile"} if groups_all is not None else {}))
    results = {}
    for n in plan["sizes"]:
        draws = 1 if n >= len(Y) else plan["draws"]
        for k in range(draws):
            f = out_dir / f"n{n}_draw{k}.json"
            done = load_json_or_none(f)
            if done is not None: results[(n, k)] = done; continue
            if n >= len(Y): idx = np.arange(len(Y))
            elif groups_all is None:
                idx, _ = train_test_split(np.arange(len(Y)), train_size=n, stratify=strat_all, random_state=100 + k)
            else:   # whole patients until about n isolates
                idx = subsample(n, strat_all.values, groups_all, 100 + k)
            g_sub = None if groups_all is None else groups_all[idx]
            folds = run_one(Xa[idx], Y.iloc[idx].reset_index(drop=True), cfg, plan["up"], plan["mid"], plan["last"], plan["repeats"], 42 + k, stacking=args.stacking, groups=g_sub, inner_auc=args.inner)
            if g_sub is not None:   # realized size of the patient subsample
                atomic_json_dump({"n_target": n, "n_isolates": int(len(idx)), "n_patients": int(len(set(g_sub))),
                                  "resistant": {d: int(Y.iloc[idx][d].sum()) for d in Y.columns}}, out_dir / f"n{n}_draw{k}_size.json", indent=1)
            atomic_json_dump(folds, f); results[(n, k)] = folds
            print(f"{site} n={n} draw {k}: BR {plan['mid']} {np.nanmean([x['br_'+plan['mid']]['AUROC'] for x in folds]):.3f}, chain hard {np.nanmean([x['chain_hard_'+plan['mid']]['AUROC'] for x in folds]):.3f}", flush=True)
    rows = []
    for n in plan["sizes"]:
        for d in (plan["mid"], plan["last"]):
            folds = [x for (nn, k), fl in results.items() if nn == n for x in fl]
            br = np.array([x[f"br_{d}"]["AUROC"] for x in folds]); hard = np.array([x[f"chain_hard_{d}"]["AUROC"] for x in folds]); proba = np.array([x[f"chain_proba_{d}"]["AUROC"] for x in folds])
            mcc_br = np.array([x[f"br_{d}"]["MCC"] for x in folds]); mcc_h = np.array([x[f"chain_hard_{d}"]["MCC"] for x in folds])
            ok = lambda v: v[~np.isnan(v)]
            _, lo, hi = bca_mean_ci(ok(hard - br)); _, lo2, hi2 = bca_mean_ci(ok(proba - br))
            row = {"n": n, "drug": d, "folds": len(folds), "br_auroc": float(np.nanmean(br)), "d_hard": float(np.nanmean(hard - br)), "d_hard_ci": [lo, hi],
                   "d_proba": float(np.nanmean(proba - br)), "d_proba_ci": [lo2, hi2], "d_mcc_hard": float(np.nanmean(mcc_h - mcc_br))}
            if args.stacking:
                st = np.array([x[f"stack_{d}"]["AUROC"] for x in folds]); _, lo3, hi3 = bca_mean_ci(ok(st - br)); row.update({"d_stack": float(np.nanmean(st - br)), "d_stack_ci": [lo3, hi3], "d_mcc_stack": float(np.nanmean(np.array([x[f"stack_{d}"]["MCC"] for x in folds]) - mcc_br))})
            rows.append(row)
    atomic_json_dump(rows, ROOT / f"results/stats/learning_curve_{tag}.json", indent=1)
    lines = [f"# Learning curve at {site} (logistic regression, {args.grid} grid, chain CIP -> {plan['mid']} -> {plan['last']})", "",
             "| n | Drug | folds | BR AUROC | chain gain hard [CI] | chain gain proba [CI] | MCC gain hard |" + (" stacking gain [CI] | MCC gain stacking |" if args.stacking else ""), "|---|---|---|---|---|---|---|" + ("---|---|" if args.stacking else "")]
    for r in rows:
        lines.append(f"| {r['n']} | {r['drug']} | {r['folds']} | {r['br_auroc']:.3f} | {r['d_hard']:+.3f} [{r['d_hard_ci'][0]:+.3f}, {r['d_hard_ci'][1]:+.3f}] | {r['d_proba']:+.3f} [{r['d_proba_ci'][0]:+.3f}, {r['d_proba_ci'][1]:+.3f}] | {r['d_mcc_hard']:+.3f} |" + (f" {r['d_stack']:+.3f} [{r['d_stack_ci'][0]:+.3f}, {r['d_stack_ci'][1]:+.3f}] | {r['d_mcc_stack']:+.3f} |" if args.stacking else ""))
    (ROOT / f"tables/learning_curve_{tag}.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
