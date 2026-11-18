"""Directed-link ablation for the non-linear learners (CatBoost, TabPFN) on the saved Sant'Orsola chain models:
realized, oracle, permuted and probability upstream inputs per ordering and rebuilt ensembles, plus single-link
permutation importance at position 3. Must run in the environment that can unpickle the models.
Usage: <env python> scripts/03_mechanism/05_link_ablation_nonlinear.py --model catboost|tabpfn
Outputs: results/stats/link_ablation_<model>.json, tables/link_ablation_<model>.md
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.models.persistence import load_model  # noqa: E402


def metrics(y, p):
    yhat = (p >= 0.5).astype(int); return {"AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan, "MCC": float(matthews_corrcoef(y, yhat))}


def predict_variant(ests, order, k, X, Yt, variant, rng=None, permute_only=None):
    Xa = np.asarray(X, float); cols = []
    for j in range(k):
        pj = ests[j].predict_proba(np.column_stack([Xa] + cols) if cols else Xa)[:, 1]
        cols.append(pj.reshape(-1, 1) if variant == "proba" else (pj >= 0.5).astype(float).reshape(-1, 1))
    if variant == "oracle": cols = [Yt[order[j]].values.astype(float).reshape(-1, 1) for j in range(k)]
    elif variant == "permuted": cols = [Yt[order[j]].values.astype(float)[rng.permutation(len(Yt))].reshape(-1, 1) for j in range(k)]
    elif variant == "single_perm":
        cols = [c.copy() for c in cols]; j = order.index(permute_only); cols[j] = cols[j][rng.permutation(len(Yt))]
    return ests[k].predict_proba(np.column_stack([Xa] + cols))[:, 1]


def sign_flip_p(d):
    from itertools import product
    d = np.asarray(d, float); n = len(d); obs = abs(d.mean()); cnt = 0
    for signs in product([1.0, -1.0], repeat=n):
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12: cnt += 1
    return cnt / 2 ** n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--n-perm", type=int, default=10); args = ap.parse_args()
    model = args.model; N_PERM = args.n_perm
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml")); X, Y = load_dataset(cfg); labels = list(Y.columns)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    folds = {(g // 5, g % 5): te for g, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, strat))}
    # BR reference recomputed from the saved BR fold models (the cv_results pickle of TabPFN needs numpy 2, which the
    # TabPFN environment does not have); identical to the stored values for the logistic and CatBoost runs.
    br = {d: {} for d in labels}
    for (r, f), te in sorted(folds.items()):
        bm = load_model(ROOT / "results/s_orsola/models", "binary_relevance", r, f, model)
        for d in labels:
            br[d][(r, f)] = metrics(Y.iloc[te][d].values, bm.estimators_[d].predict_proba(np.asarray(X.iloc[te], float))[:, 1])["AUROC"]
    orderings = [list(o) for o in permutations(labels)]; rng = np.random.default_rng(42)
    VARIANTS = ["realized", "oracle", "permuted", "proba"]; rec = {}; ens = {v: {d: {} for d in labels} for v in VARIANTS}; single = {}
    out_dir = ROOT / f"results/link_ablation_{model}"; out_dir.mkdir(exist_ok=True)
    for (r, f), te in sorted(folds.items()):
        fp = out_dir / f"r{r}_f{f}.json"
        if fp.exists():
            saved = json.load(open(fp))
            for key, val in saved["rec"].items(): rec.setdefault(tuple(key.split("|")), {})[(r, f)] = val
            for v in VARIANTS:
                for d in labels: ens[v][d][(r, f)] = saved["ens"][v][d]
            for key, val in saved["single"].items(): single.setdefault(tuple(key.split("|")), {})[(r, f)] = val
            continue
        Xt, Yt = X.iloc[te], Y.iloc[te]; fold_rec, fold_ens, fold_single = {}, {v: {d: [] for d in labels} for v in VARIANTS}, {}
        for order in orderings:
            m = load_model(ROOT / "results/s_orsola/models", "chain", r, f, model, order); ests = m.chain.estimators_; okey = "->".join(order)
            for k, d in enumerate(order):
                if k == 0:
                    p0 = ests[0].predict_proba(np.asarray(Xt, float))[:, 1]
                    for v in VARIANTS: fold_ens[v][d].append(p0)
                    continue
                for v in VARIANTS:
                    if v == "permuted":
                        ps = [predict_variant(ests, order, k, Xt, Yt, v, rng=rng) for _ in range(N_PERM)]; ms = [metrics(Yt[d].values, p)[ "AUROC"] for p in ps]
                        fold_rec[f"{okey}|{d}|{v}"] = float(np.nanmean(ms)); fold_ens[v][d].append(np.mean(ps, axis=0))
                    else:
                        p = predict_variant(ests, order, k, Xt, Yt, v); fold_rec[f"{okey}|{d}|{v}"] = metrics(Yt[d].values, p)["AUROC"]; fold_ens[v][d].append(p)
                if k == 2:
                    for u in order[:2]:
                        fold_single[f"{okey}|{d}|{u}"] = float(np.nanmean([metrics(Yt[d].values, predict_variant(ests, order, k, Xt, Yt, "single_perm", rng=rng, permute_only=u))["AUROC"] for _ in range(N_PERM)]))
        fold_ens_m = {v: {d: metrics(Yt[d].values, np.mean(fold_ens[v][d], axis=0))["AUROC"] for d in labels} for v in VARIANTS}
        json.dump({"rec": fold_rec, "ens": fold_ens_m, "single": fold_single}, open(fp, "w"))
        for key, val in fold_rec.items(): rec.setdefault(tuple(key.split("|")), {})[(r, f)] = val
        for v in VARIANTS:
            for d in labels: ens[v][d][(r, f)] = fold_ens_m[v][d]
        for key, val in fold_single.items(): single.setdefault(tuple(key.split("|")), {})[(r, f)] = val
        print(f"{model} fold r{r} f{f} done", flush=True)
    out = {"per_ordering": [], "ensemble": [], "single_link_permutation": []}
    lines = [f"# Directed-link ablation, {model} (Sant'Orsola, 15 folds, AUROC)", "", "## Chain ensemble rebuilt per variant (delta versus BR, sign-flip p)", "", "| Variant | Drug | AUROC | dAUROC [CI] p |", "|---|---|---|---|"]
    for v in VARIANTS:
        for d in labels:
            keys = sorted(ens[v][d]); vals = np.array([ens[v][d][k] for k in keys]); dlt = vals - np.array([br[d][k] for k in keys]); _, lo, hi = bca_mean_ci(dlt)
            out["ensemble"].append({"variant": v, "drug": d, "AUROC": float(vals.mean()), "dAUROC": float(dlt.mean()), "ci": [lo, hi], "p": sign_flip_p(dlt)})
            lines.append(f"| {v} | {d} | {vals.mean():.3f} | {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {sign_flip_p(dlt):.3f} |")
    lines += ["", "## Per ordering and downstream drug (AUROC; delta versus BR)", "", "| Ordering | Downstream | realized | oracle | permuted | proba |", "|---|---|---|---|---|---|"]
    seen = {}
    for (okey, d, v), fm in sorted(rec.items()):
        keys = sorted(fm); vals = np.array([fm[k] for k in keys]); dlt = vals - np.array([br[d][k] for k in keys])
        out["per_ordering"].append({"ordering": okey, "downstream": d, "variant": v, "AUROC": float(vals.mean()), "dAUROC": float(dlt.mean()), "p": sign_flip_p(dlt)}); seen.setdefault((okey, d), {})[v] = (vals.mean(), dlt.mean())
    for (okey, d), vv in seen.items(): lines.append(f"| {okey} | {d} | " + " | ".join(f"{vv[v][0]:.3f} ({vv[v][1]:+.3f})" for v in VARIANTS) + " |")
    lines += ["", "## Single-link permutation importance at position 3 (AUROC drop)", "", "| Ordering | Downstream | Permuted upstream | drop |", "|---|---|---|---|"]
    for (okey, d, u), fm in sorted(single.items()):
        keys = sorted(fm); drop = np.array([rec[(okey, d, "realized")][k] - fm[k] for k in keys]); out["single_link_permutation"].append({"ordering": okey, "downstream": d, "permuted_upstream": u, "drop": float(drop.mean()), "p": sign_flip_p(drop)})
        lines.append(f"| {okey} | {d} | {u} | {drop.mean():+.3f} (p {sign_flip_p(drop):.3f}) |")
    json.dump(out, open(ROOT / f"results/stats/link_ablation_{model}.json", "w"), indent=1); (ROOT / f"tables/link_ablation_{model}.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
