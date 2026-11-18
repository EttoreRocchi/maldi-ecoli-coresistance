#!/usr/bin/env python3
"""Cross-site transfer of cross-fitted stacking (logistic regression).

For each training site, the three binary-relevance probabilities are cross-fitted (5-fold) on the training site with the
hyperparameters selected for the cached cross-dataset binary-relevance models (scripts/07_transfer/01_cross_dataset.py, results/cross_dataset/logistic/
models), a logistic meta-model per drug (C in {0.01, 0.1, 1, 10}, as in the nested stacking) is fitted on their logits,
and it is applied to the binary-relevance probabilities of every other site. The test site never enters fitting or
tuning. The binary-relevance AUROC of every pair is checked against cross_dataset_results.json. Paired isolate-level
bootstrap (2,000 resamples) of stacking minus binary relevance per pair and drug.
Outputs: results/cross_dataset/logistic/cross_dataset_stacking.json (same layout as cross_dataset_results.json),
         results/stats/cross_dataset_stacking.json, tables/cross_dataset_stacking.md
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402

SITES = [("s_orsola", "s_orsola"), ("driams_a", "driams_a"), ("driams_b", "driams_b"), ("driams_c", "driams_c"), ("driams_d", "driams_d")]
LABELS = ["CIP", "CRO", "CAZ"]
LABEL_MAP = {"CTX": "CRO"}
N_BOOT = 2000
OUT = ROOT / "results/cross_dataset/logistic"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def oof_column(est, X, y):
    """5-fold cross-fitted probabilities of one drug on the training site."""
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        p[te] = clone(est).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def boot_delta(y, a, b, seed):
    rng = np.random.default_rng(seed); n = len(y); out = []
    while len(out) < N_BOOT:
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2: continue
        out.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    global SITES
    ref = {(r["train"], r["test"]): r for r in json.load(open(OUT / "cross_dataset_results.json"))}
    if any(k[0] == "local" for k in ref):   # older runs of scripts/07_transfer/01_cross_dataset.py name the Sant'Orsola cohort "local"
        SITES = [("local", "s_orsola")] + SITES[1:]
    data, cfgs = {}, {}
    for name, cfg_name in SITES:
        cfg = Config.from_yaml(str(ROOT / f"configs/{cfg_name}.yaml")); cfg.model = "logistic"; cfgs[name] = cfg
        X, Y = load_dataset(cfg); Y = Y.rename(columns=LABEL_MAP)
        data[name] = (np.asarray(X, float), Y[LABELS].values.astype(int))
        print(f"loaded {name}: {X.shape}", flush=True)

    br = {name: joblib.load(OUT / f"models/cross_{name}_logistic_br.joblib") for name, _ in SITES}

    # cross-fitted training probabilities, one job per site and drug (checkpointed)
    ck = OUT / "stacking_oof"; ck.mkdir(exist_ok=True)
    todo = [(name, j) for name, _ in SITES for j in range(3) if not (ck / f"{name}_{LABELS[j]}.npy").exists()]
    def job(name, j):
        X, Y = data[name]; est = br[name].estimators_[LABELS[j]].best_estimator_
        np.save(ck / f"{name}_{LABELS[j]}.npy", oof_column(est, X, Y[:, j])); return name, LABELS[j]
    for name, d in Parallel(n_jobs=len(todo) or 1, verbose=5)(delayed(job)(n, j) for n, j in todo):
        print(f"cross-fitted {name} {d}", flush=True)

    results, stats = [], []
    for train, _ in SITES:
        Xtr, Ytr = data[train]; Ptr = np.column_stack([np.load(ck / f"{train}_{d}.npy") for d in LABELS])
        metas = []
        for j in range(3):
            meta = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring="roc_auc")
            metas.append(meta.fit(logit(Ptr), Ytr[:, j]))
        for test, _ in SITES:
            if test == train: continue
            Xte, Yte = data[test]; pb = br[train].predict_proba(Xte); Pte = np.column_stack([np.asarray(pb[d]) for d in LABELS])
            entry = {"train": train, "test": test, "model": "logistic", "n_train": len(Ytr), "n_test": len(Yte), "methods": {"binary_relevance": {}, "stacking": {}}}
            for j, d in enumerate(LABELS):
                ps = metas[j].predict_proba(logit(Pte))[:, 1]; y = Yte[:, j]
                a_br, a_st = roc_auc_score(y, Pte[:, j]), roc_auc_score(y, ps)
                a_ref = ref[(train, test)]["methods"]["binary_relevance"][d]["AUROC"]
                assert abs(a_br - a_ref) < 1e-9, f"BR mismatch {train}->{test} {d}: {a_br} vs {a_ref}"
                a_ens = ref[(train, test)]["methods"]["ensemble"][d]["AUROC"]
                entry["methods"]["binary_relevance"][d] = {"AUROC": a_br}; entry["methods"]["stacking"][d] = {"AUROC": a_st}
                stats.append({"train": train, "test": test, "drug": d, "br_auroc": a_br, "stack_auroc": a_st, "ens_auroc": a_ens,
                              "stack_minus_br": a_st - a_br, "stack_minus_br_ci": boot_delta(y, ps, Pte[:, j], 42 + j), "ens_minus_br": a_ens - a_br,
                              "meta_C": float(metas[j].best_params_["C"]), "meta_coef": metas[j].best_estimator_.coef_[0].tolist()})
            results.append(entry)
            print(f"{train} -> {test}: " + ", ".join(f"{s['drug']} BR {s['br_auroc']:.3f} stack {s['stack_auroc']:.3f} ({s['stack_minus_br']:+.3f})" for s in stats[-3:]), flush=True)

    json.dump(results, open(OUT / "cross_dataset_stacking.json", "w"), indent=1)
    summary = {}
    for d in LABELS:
        g = np.array([s["stack_minus_br"] for s in stats if s["drug"] == d]); e = np.array([s["ens_minus_br"] for s in stats if s["drug"] == d])
        summary[d] = {"mean_stack_minus_br": float(g.mean()), "median_stack_minus_br": float(np.median(g)), "n_positive": int((g > 0).sum()),
                      "n_ci_above_zero": int(sum(s["stack_minus_br_ci"][0] > 0 for s in stats if s["drug"] == d)),
                      "n_ci_below_zero": int(sum(s["stack_minus_br_ci"][1] < 0 for s in stats if s["drug"] == d)),
                      "mean_ens_minus_br": float(e.mean()), "n_pairs": len(g)}
    json.dump({"pairs": stats, "summary": summary, "n_boot": N_BOOT,
               "note": "cross-fitting with the hyperparameters of the cached binary-relevance models; meta-model C grid {0.01, 0.1, 1, 10}"},
              open(ROOT / "results/stats/cross_dataset_stacking.json", "w"), indent=1)
    lines = ["# Cross-site transfer of cross-fitted stacking (logistic regression)", "",
             "| Train | Test | Drug | BR AUROC | Stacking AUROC | Stacking - BR [95% CI] | Chain ensemble - BR |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {s['train']} | {s['test']} | {s['drug']} | {s['br_auroc']:.3f} | {s['stack_auroc']:.3f} | {s['stack_minus_br']:+.3f} [{s['stack_minus_br_ci'][0]:+.3f}, {s['stack_minus_br_ci'][1]:+.3f}] | {s['ens_minus_br']:+.3f} |" for s in stats]
    lines += ["", "| Drug | mean stacking - BR | median | pairs > 0 | CI above 0 | CI below 0 | mean chain ensemble - BR |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {d} | {v['mean_stack_minus_br']:+.3f} | {v['median_stack_minus_br']:+.3f} | {v['n_positive']}/{v['n_pairs']} | {v['n_ci_above_zero']} | {v['n_ci_below_zero']} | {v['mean_ens_minus_br']:+.3f} |" for d, v in summary.items()]
    (ROOT / "tables/cross_dataset_stacking.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines[-5:]))


if __name__ == "__main__":
    main()
