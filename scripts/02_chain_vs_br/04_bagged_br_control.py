"""Size-matched bagged binary-relevance ensemble as the control for the chain-ensemble endpoint
(logistic regression, Sant'Orsola, same 15 outer folds).

Question: how much of the chain-ensemble gain over a single binary-relevance (BR) model is explained by
averaging six models rather than by the inter-label pathway?

For each outer fold and each drug, six BR members are fitted on the training fold with the hyperparameters
selected by that fold's saved BR GridSearchCV (same pipeline: StandardScaler + LogisticRegression,
class_weight balanced), either on bootstrap resamples of the training rows ("bootstrap") or on random 50 percent
subsets of the m/z bins ("subspace"). Member probabilities are averaged. The chain ensemble (stored, hard-label
passing) and the chain ensemble with probability passing (rebuilt from the saved chain models) are compared with
the single BR model and with the two bagged-BR ensembles on the same folds (paired sign-flip tests, BCa intervals).

Outputs: results/stats/bagged_br_control.json, tables/bagged_br_control.md
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)

from mecor import shared as la
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci, load_cv_results  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.models.persistence import load_model  # noqa: E402

SEED = 42
N_MEMBERS = 6
MODEL_DIR = ROOT / "results/s_orsola/models"
CV_PKL = ROOT / "results/s_orsola/logistic/cv_results.pkl"
N_JOBS = 24


def fit_member(params, Xtr, ytr, Xte, seed, mode, rng_seed):
    rng = np.random.default_rng(rng_seed)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed, n_jobs=1,
                             C=params["clf__C"], penalty=params["clf__penalty"], solver=params["clf__solver"])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if mode == "bootstrap":
        idx = rng.integers(0, len(ytr), len(ytr))
        # guard against a resample without both classes
        while len(np.unique(ytr[idx])) < 2:
            idx = rng.integers(0, len(ytr), len(ytr))
        pipe.fit(Xtr[idx], ytr[idx])
        return pipe.predict_proba(Xte)[:, 1]
    cols = rng.choice(Xtr.shape[1], Xtr.shape[1] // 2, replace=False)
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
    for fr in cv.fold_results:
        if fr.method == "binary_relevance":
            tr, te = folds[(fr.repeat_idx, fr.fold_idx)]
            for d in labels:
                assert np.array_equal(Y.iloc[te][d].values, np.asarray(fr.y_true[d]))
    print("fold reconstruction verified")
    orderings = [list(o) for o in permutations(labels)]

    per_fold = {}  # (method, drug) -> {fold: metrics}
    for (r, f), (tr, te) in sorted(folds.items()):
        Xtr, Xte, Ytr, Yte = Xa[tr], Xa[te], Y.iloc[tr], Y.iloc[te]
        br = load_model(MODEL_DIR, "binary_relevance", r, f, "logistic")
        for d in labels:
            est = br.estimators_[d]
            params = est.best_params_
            ytr = Ytr[d].values
            for mode in ("bootstrap", "subspace"):
                probs = Parallel(n_jobs=N_JOBS)(
                    delayed(fit_member)(params, Xtr, ytr, Xte, SEED, mode, SEED + 1000 * r + 100 * f + 10 * labels.index(d) + i)
                    for i in range(N_MEMBERS))
                p = np.mean(probs, axis=0)
                per_fold.setdefault((f"bagged_br_{mode}", d), {})[(r, f)] = la.metrics(Yte[d].values, p)
            # single BR and stored chain ensemble from cv results
            fr_br = [x for x in cv.fold_results if x.method == "binary_relevance" and x.repeat_idx == r and x.fold_idx == f][0]
            fr_en = [x for x in cv.fold_results if x.method == "ensemble" and x.repeat_idx == r and x.fold_idx == f][0]
            per_fold.setdefault(("br_single", d), {})[(r, f)] = la.metrics(np.asarray(fr_br.y_true[d]), np.asarray(fr_br.y_prob[d]))
            per_fold.setdefault(("chain_ensemble", d), {})[(r, f)] = la.metrics(np.asarray(fr_en.y_true[d]), np.asarray(fr_en.y_prob[d]))
        # chain ensemble with probability passing, rebuilt from saved chain models
        plist = {d: [] for d in labels}
        for order in orderings:
            m = load_model(MODEL_DIR, "chain", r, f, "logistic", order)
            ests = m.chain.estimators_
            for k, d in enumerate(order):
                if k == 0:
                    plist[d].append(ests[0].predict_proba(Xte)[:, 1])
                else:
                    plist[d].append(la.predict_variant(ests, order, k, Xte, Yte, "proba"))
        for d in labels:
            per_fold.setdefault(("chain_ensemble_proba", d), {})[(r, f)] = la.metrics(Yte[d].values, np.mean(plist[d], axis=0))
        print(f"fold r{r} f{f} done", flush=True)

    methods = ["br_single", "bagged_br_bootstrap", "bagged_br_subspace", "chain_ensemble", "chain_ensemble_proba"]
    out = {"means": {}, "contrasts": []}
    for (mth, d), fm in per_fold.items():
        out["means"].setdefault(mth, {})[d] = {k: float(np.nanmean([v[k] for v in fm.values()])) for k in ("AUROC", "MCC", "VME", "ME")}
    contrasts = [("bagged_br_bootstrap", "br_single"), ("bagged_br_subspace", "br_single"),
                 ("chain_ensemble", "br_single"), ("chain_ensemble", "bagged_br_bootstrap"), ("chain_ensemble", "bagged_br_subspace"),
                 ("chain_ensemble_proba", "br_single"), ("chain_ensemble_proba", "bagged_br_bootstrap"), ("chain_ensemble_proba", "bagged_br_subspace")]
    for a, b in contrasts:
        for d in labels:
            fa, fb = per_fold[(a, d)], per_fold[(b, d)]
            keys = sorted(fa)
            row = {"method": a, "reference": b, "drug": d}
            for met in ("AUROC", "MCC", "VME", "ME"):
                dlt = np.array([fa[k][met] - fb[k][met] for k in keys])
                _, lo, hi = bca_mean_ci(dlt)
                row[met] = {"delta": float(np.nanmean(dlt)), "ci95": [lo, hi], "p": la.sign_flip_p(dlt)}
            out["contrasts"].append(row)
    out["notes"] = {"n_members": N_MEMBERS, "members_use_fold_tuned_hyperparameters": True,
                    "subspace_fraction": 0.5, "n_folds": len(folds)}
    (ROOT / "results/stats/bagged_br_control.json").write_text(json.dumps(out, indent=1))

    lines = ["# Bagged binary-relevance control (logistic regression, Sant'Orsola, 15 folds)", "",
             "## Mean per-fold metrics", "", "| Method | Drug | AUROC | MCC | VME | ME |", "|---|---|---|---|---|---|"]
    for mth in methods:
        for d in labels:
            m = out["means"][mth][d]
            lines.append(f"| {mth} | {d} | {m['AUROC']:.3f} | {m['MCC']:.3f} | {m['VME']:.3f} | {m['ME']:.3f} |")
    lines += ["", "## Paired contrasts (delta, BCa 95 percent CI, exact sign-flip p over 15 folds)", "",
              "| Method | Reference | Drug | dAUROC | dMCC | dVME | dME |", "|---|---|---|---|---|---|---|"]
    for row in out["contrasts"]:
        cells = [f"{row[m]['delta']:+.3f} [{row[m]['ci95'][0]:+.3f}, {row[m]['ci95'][1]:+.3f}] p={row[m]['p']:.3f}" for m in ("AUROC", "MCC", "VME", "ME")]
        lines.append(f"| {row['method']} | {row['reference']} | {row['drug']} | " + " | ".join(cells) + " |")
    (ROOT / "tables/bagged_br_control.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
