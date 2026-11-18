"""Does heterogeneity of the cefotaxime label explain the cross-prediction margin? Refit without the CTX-R, CAZ-S isolates.

Sant'Orsola, logistic regression with the main-analysis (full) grid, the pipeline's 15 outer folds (5 x 3 repeated stratified
folds on the joint profile, seed 42). In each outer fold the CTX binary-relevance classifier is refitted after removing the
cefotaxime-resistant, ceftazidime-susceptible isolates from the training fold, and scored on the test fold with and without
those isolates. It is compared on the same test isolates with the stored CTX and CIP binary-relevance classifiers
(results/nested_stacking/s_orsola_logistic_full/r*_f*.npz). Contrasts are paired over the folds (BCa 95% interval, exact
sign-flip p). The AUROC of the three classifiers within the CIP strata is computed on the pooled out-of-fold rows.
Outputs: results/ctx_without_discordant/r<r>_f<f>.npz (checkpoints), results/stats/ctx_without_discordant.json,
         tables/ctx_without_discordant.md
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.evaluation.cv import _build_search_estimator  # noqa: E402


def sign_flip_p(d):
    d = np.asarray(d, float); obs = abs(d.mean()); signs = np.array(list(product([-1, 1], repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def contrast(d):
    m, lo, hi = bca_mean_ci(d); return {"delta": m, "ci": [lo, hi], "p": sign_flip_p(d)}


def main():
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml")); cfg.model = "logistic"; cfg.param_search_mode = "full"; cfg.n_jobs = 8
    X, Y = load_dataset(cfg); Xa = np.asarray(X, float)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    folds = {(g // 5, g % 5): (tr, te) for g, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, strat))}
    disc = ((Y["CTX"] == 1) & (Y["CAZ"] == 0)).values
    out_dir = ROOT / "results/ctx_without_discordant"; out_dir.mkdir(parents=True, exist_ok=True)
    auc = {k: [] for k in ("refit_all", "refit_conc", "stored_ctx_all", "stored_ctx_conc", "cip_all", "cip_conc")}
    for (r, f), (tr, te) in sorted(folds.items()):
        ck = out_dir / f"r{r}_f{f}.npz"
        if ck.exists(): p = np.load(ck)["p"]
        else:
            keep = tr[~disc[tr]]; est = _build_search_estimator("logistic", cfg); est.fit(Xa[keep], Y["CTX"].values[keep])
            p = est.predict_proba(Xa[te])[:, 1]; np.savez_compressed(ck, p=p)
            print(f"fold r{r} f{f}: removed {int(disc[tr].sum())} training isolates", flush=True)
        z = np.load(ROOT / f"results/nested_stacking/s_orsola_logistic_full/r{r}_f{f}.npz")
        assert np.array_equal(z["y"], Y.iloc[te].values.astype(int)), "fold reconstruction mismatch"
        y = Y["CTX"].values[te]; conc = ~disc[te]
        for name, s in (("refit", p), ("stored_ctx", z["br"][:, 1]), ("cip", z["br"][:, 0])):
            auc[f"{name}_all"].append(roc_auc_score(y, s)); auc[f"{name}_conc"].append(roc_auc_score(y[conc], s[conc]))
    A = {k: np.array(v) for k, v in auc.items()}
    pooled = {"y": [], "cip_lab": [], "refit": [], "stored_ctx": [], "cip": []}   # within-stratum AUROC on the pooled out-of-fold rows
    for (r, f), (tr, te) in sorted(folds.items()):
        z = np.load(ROOT / f"results/nested_stacking/s_orsola_logistic_full/r{r}_f{f}.npz")
        pooled["y"].append(Y["CTX"].values[te]); pooled["cip_lab"].append(Y["CIP"].values[te]); pooled["refit"].append(np.load(out_dir / f"r{r}_f{f}.npz")["p"])
        pooled["stored_ctx"].append(z["br"][:, 1]); pooled["cip"].append(z["br"][:, 0])
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    strata = {s: {k: float(roc_auc_score(P["y"][P["cip_lab"] == v], P[k][P["cip_lab"] == v])) for k in ("stored_ctx", "refit", "cip")} for s, v in (("CIP-S", 0), ("CIP-R", 1))}
    res = {"n_discordant": int(disc.sum()), "auroc": {k: float(v.mean()) for k, v in A.items()}, "auroc_within_cip_strata": strata,
           "margin_stored_all": contrast(A["cip_all"] - A["stored_ctx_all"]), "margin_refit_all": contrast(A["cip_all"] - A["refit_all"]),
           "margin_stored_conc": contrast(A["cip_conc"] - A["stored_ctx_conc"]), "margin_refit_conc": contrast(A["cip_conc"] - A["refit_conc"]),
           "refit_minus_stored_all": contrast(A["refit_all"] - A["stored_ctx_all"]), "refit_minus_stored_conc": contrast(A["refit_conc"] - A["stored_ctx_conc"])}
    fmt = lambda c: f"{c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] {c['p']:.3f}"
    lines = [f"# CTX classifier refitted without the {res['n_discordant']} CTX-R, CAZ-S isolates (Sant'Orsola, logistic regression, full grid, 15 outer folds)", "",
             "Test sets: all test isolates, or without the CTX-R, CAZ-S isolates (concordant). Margin = CIP classifier minus CTX classifier, scored on the CTX label. Contrasts paired over folds (BCa 95% CI, exact sign-flip p).", "",
             "| Test set | CTX classifier, stored | CTX classifier, refitted | CIP classifier | Margin with stored CTX | Margin with refitted CTX | Refitted minus stored |", "|---|---|---|---|---|---|---|"]
    for ts, lab in (("all", "All isolates"), ("conc", "Without CTX-R, CAZ-S")):
        a = res["auroc"]
        lines.append(f"| {lab} | {a['stored_ctx_' + ts]:.3f} | {a['refit_' + ts]:.3f} | {a['cip_' + ts]:.3f} | {fmt(res['margin_stored_' + ts])} | {fmt(res['margin_refit_' + ts])} | {fmt(res['refit_minus_stored_' + ts])} |")
    for s, v in strata.items():
        lines.append(f"| Within {s} (pooled out-of-fold) | {v['stored_ctx']:.3f} | {v['refit']:.3f} | {v['cip']:.3f} | | | |")
    (ROOT / "results/stats/ctx_without_discordant.json").write_text(json.dumps(res, indent=1))
    (ROOT / "tables/ctx_without_discordant.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
