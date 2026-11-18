"""Flat multi-label comparators on the same outer folds (Sant'Orsola, logistic-regression era baselines).

Question: is the Introduction's claim that flat multi-label models need to observe each co-resistance combination
(and so do worse in a small cohort) true here, and is binary relevance the right reference?

Comparators, same 15 outer folds (5 x 3 repeated stratified on the joint profile, seed 42), inner 3-fold tuning:
  LP   : Label Powerset, multinomial logistic regression over the eight joint profiles (StandardScaler, class_weight
         balanced, C tuned over the paper's six values, lbfgs); per-drug probability = sum of the class probabilities
         of profiles in which the drug is resistant.
  MLP  : multi-output multilayer perceptron (one hidden layer of 128 units, shared representation, sigmoid outputs),
         alpha tuned over {1e-4, 1e-3, 1e-2}, early stopping.
  RF   : multi-output random forest (500 trees, class_weight balanced_subsample), max_features tuned over {sqrt, 0.1}.
  RF-BR: the same forest fitted per drug (binary relevance), the same-family control that isolates the multi-output effect.
Reference: stored single BR and chain ensemble (logistic) on the same folds.

Outputs: results/stats/flat_multilabel.json, tables/flat_multilabel.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor import shared as la
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci, load_cv_results  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402

SEED = 42
N_JOBS = 16
CV_PKL = ROOT / "results/s_orsola/logistic/cv_results.pkl"


def multilabel_auc(est, X, Y):
    P = est.predict_proba(X)
    if isinstance(P, list):  # RF returns a list per output
        P = np.column_stack([p[:, 1] for p in P])
    aucs = []
    for j in range(Y.shape[1]):
        if len(np.unique(Y[:, j])) > 1:
            aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs))


def make_lp_scorer(profiles, n_labels):
    """Inner-CV score for Label Powerset: mean per-drug AUROC of the marginal probabilities."""
    def score(est, X, y_lp):
        P = est.predict_proba(X)
        p_marg = np.zeros((len(y_lp), n_labels))
        for ci, c in enumerate(est.classes_):
            prof = profiles[c]
            for j in range(n_labels):
                if prof[j] == "1":
                    p_marg[:, j] += P[:, ci]
        Ymat = np.array([[int(profiles[c][j]) for j in range(n_labels)] for c in y_lp])
        aucs = [roc_auc_score(Ymat[:, j], p_marg[:, j]) for j in range(n_labels) if len(np.unique(Ymat[:, j])) > 1]
        return float(np.mean(aucs)) if aucs else 0.5
    return score


def inner_cv(y_strat, n_splits=3):
    counts = np.bincount(y_strat)
    if counts[counts > 0].min() >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return KFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def main():
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml"))
    X, Y = load_dataset(cfg)
    labels = list(Y.columns)
    Xa = np.asarray(X, float)
    Ya = Y.values.astype(int)
    cv = load_cv_results(CV_PKL)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    folds = {}
    for g, (tr, te) in enumerate(outer.split(X, strat)):
        folds[(g // 5, g % 5)] = (tr, te)
    profiles = sorted(set(strat))
    prof_index = {p: i for i, p in enumerate(profiles)}
    y_lp_all = np.array([prof_index[p] for p in strat])

    per_fold = {}
    for (r, f), (tr, te) in sorted(folds.items()):
        Xtr, Xte, Ytr, Yte = Xa[tr], Xa[te], Ya[tr], Ya[te]
        ytr_lp = y_lp_all[tr]
        lp = GridSearchCV(
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED, solver="lbfgs"))]),
            {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, scoring=make_lp_scorer(profiles, len(labels)),
            cv=inner_cv(ytr_lp), n_jobs=N_JOBS, refit=True)
        lp.fit(Xtr, ytr_lp)
        P = lp.predict_proba(Xte)
        p_lp = np.zeros((len(te), len(labels)))
        for ci, c in enumerate(lp.classes_):
            prof = profiles[c]
            for j in range(len(labels)):
                if prof[j] == "1":
                    p_lp[:, j] += P[:, ci]
        mlp = GridSearchCV(
            Pipeline([("scaler", StandardScaler()),
                      ("clf", MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, max_iter=500, random_state=SEED))]),
            {"clf__alpha": [1e-4, 1e-3, 1e-2]}, scoring=multilabel_auc, cv=inner_cv(ytr_lp), n_jobs=N_JOBS, refit=True)
        mlp.fit(Xtr, Ytr)
        p_mlp = mlp.predict_proba(Xte)
        rf = GridSearchCV(
            RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=SEED, n_jobs=N_JOBS),
            {"max_features": ["sqrt", 0.1]}, scoring=multilabel_auc, cv=inner_cv(ytr_lp), n_jobs=1, refit=True)
        rf.fit(Xtr, Ytr)
        p_rf = np.column_stack([p[:, 1] for p in rf.predict_proba(Xte)])
        # per-drug random forest (binary relevance with the same forest), the same-family control
        p_rfbr = np.zeros((len(te), len(labels)))
        for j in range(len(labels)):
            rfb = GridSearchCV(
                RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=SEED, n_jobs=N_JOBS),
                {"max_features": ["sqrt", 0.1]}, scoring="roc_auc", cv=inner_cv(Ytr[:, j]), n_jobs=1, refit=True)
            rfb.fit(Xtr, Ytr[:, j])
            p_rfbr[:, j] = rfb.predict_proba(Xte)[:, 1]
        fr_br = [x for x in cv.fold_results if x.method == "binary_relevance" and x.repeat_idx == r and x.fold_idx == f][0]
        fr_en = [x for x in cv.fold_results if x.method == "ensemble" and x.repeat_idx == r and x.fold_idx == f][0]
        for j, d in enumerate(labels):
            assert np.array_equal(Yte[:, j], np.asarray(fr_br.y_true[d]))
            per_fold.setdefault(("label_powerset_lr", d), {})[(r, f)] = la.metrics(Yte[:, j], p_lp[:, j])
            per_fold.setdefault(("multioutput_mlp", d), {})[(r, f)] = la.metrics(Yte[:, j], p_mlp[:, j])
            per_fold.setdefault(("multioutput_rf", d), {})[(r, f)] = la.metrics(Yte[:, j], p_rf[:, j])
            per_fold.setdefault(("br_rf", d), {})[(r, f)] = la.metrics(Yte[:, j], p_rfbr[:, j])
            per_fold.setdefault(("br_single", d), {})[(r, f)] = la.metrics(np.asarray(fr_br.y_true[d]), np.asarray(fr_br.y_prob[d]))
            per_fold.setdefault(("chain_ensemble", d), {})[(r, f)] = la.metrics(np.asarray(fr_en.y_true[d]), np.asarray(fr_en.y_prob[d]))
        print(f"fold r{r} f{f} done: LP C={lp.best_params_['clf__C']}, MLP alpha={mlp.best_params_['clf__alpha']}, RF mf={rf.best_params_['max_features']}", flush=True)

    methods = ["br_single", "chain_ensemble", "label_powerset_lr", "multioutput_mlp", "multioutput_rf", "br_rf"]
    out = {"means": {}, "contrasts": [], "profiles": profiles,
           "profile_counts": {p: int((strat == p).sum()) for p in profiles}}
    for (mth, d), fm in per_fold.items():
        out["means"].setdefault(mth, {})[d] = {k: float(np.nanmean([v[k] for v in fm.values()])) for k in ("AUROC", "MCC", "VME", "ME")}
    for a in ("label_powerset_lr", "multioutput_mlp", "multioutput_rf", "chain_ensemble", "br_rf"):
        for b in ("br_single", "chain_ensemble", "br_rf"):
            if a == b or (a == "br_rf" and b == "chain_ensemble") or (b == "br_rf" and a != "multioutput_rf"):
                continue
            for d in labels:
                fa, fb = per_fold[(a, d)], per_fold[(b, d)]
                keys = sorted(fa)
                row = {"method": a, "reference": b, "drug": d}
                for met in ("AUROC", "MCC", "VME", "ME"):
                    dlt = np.array([fa[k][met] - fb[k][met] for k in keys])
                    _, lo, hi = bca_mean_ci(dlt)
                    row[met] = {"delta": float(np.nanmean(dlt)), "ci95": [lo, hi], "p": la.sign_flip_p(dlt)}
                out["contrasts"].append(row)
    (ROOT / "results/stats/flat_multilabel.json").write_text(json.dumps(out, indent=1))
    lines = ["# Flat multi-label comparators (Sant'Orsola, same 15 outer folds)", "",
             "Joint profiles (CIP CTX CAZ) and counts: " + ", ".join(f"{p}: {out['profile_counts'][p]}" for p in profiles), "",
             "| Method | Drug | AUROC | MCC | VME | ME |", "|---|---|---|---|---|---|"]
    for mth in methods:
        for d in labels:
            m = out["means"][mth][d]
            lines.append(f"| {mth} | {d} | {m['AUROC']:.3f} | {m['MCC']:.3f} | {m['VME']:.3f} | {m['ME']:.3f} |")
    lines += ["", "## Paired contrasts (delta, BCa 95 percent CI, sign-flip p)", "", "| Method | Reference | Drug | dAUROC | dMCC | dVME | dME |", "|---|---|---|---|---|---|---|"]
    for row in out["contrasts"]:
        cells = [f"{row[m]['delta']:+.3f} [{row[m]['ci95'][0]:+.3f}, {row[m]['ci95'][1]:+.3f}] p={row[m]['p']:.3f}" for m in ("AUROC", "MCC", "VME", "ME")]
        lines.append(f"| {row['method']} | {row['reference']} | {row['drug']} | " + " | ".join(cells) + " |")
    (ROOT / "tables/flat_multilabel.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
