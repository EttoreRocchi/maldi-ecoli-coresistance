"""Chain ensemble versus binary relevance: the 18 comparisons in one table.

For each learner (logistic regression, CatBoost, TabPFN; results/s_orsola/<model>/cv_results.pkl), drug and metric (AUROC,
MCC at 0.5), the mean over the 15 outer folds of binary relevance and of the six-ordering chain ensemble with BCa 95%
intervals, the paired difference with its BCa interval and exact sign-flip p (2^15 assignments), and the Holm and
Benjamini-Hochberg adjustments across the 18 comparisons.
Outputs: results/stats/chain_vs_br_18.json, tables/chain_vs_br_18.md
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.cvio import aligned, bca_mean_ci, load_cv_results, metric_by_fold  # noqa: E402

LEARNERS = [("logistic", "Logistic regression"), ("catboost", "CatBoost"), ("tabpfn", "TabPFN")]
DRUGS, METRICS = ["CIP", "CTX", "CAZ"], ["AUROC", "MCC"]


def sign_flip_p(d):
    d = np.asarray(d, float); obs = abs(d.mean()); signs = np.array(list(product([-1, 1], repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def main():
    rows = []
    for model, name in LEARNERS:
        if not (ROOT / f"results/s_orsola/{model}/cv_results.pkl").exists(): print("missing", model); continue
        cv = load_cv_results(ROOT / f"results/s_orsola/{model}/cv_results.pkl")
        for d in DRUGS:
            for m in METRICS:
                br, ens = aligned(metric_by_fold(cv, "binary_relevance", d, m), metric_by_fold(cv, "ensemble", d, m))
                rows.append({"learner": name, "drug": d, "metric": m, "n_folds": len(br),
                             "br": bca_mean_ci(br), "ens": bca_mean_ci(ens), "delta": bca_mean_ci(ens - br), "p": sign_flip_p(ens - br)})
    p = [r["p"] for r in rows]
    for r, h, b in zip(rows, multipletests(p, method="holm")[1], multipletests(p, method="fdr_bh")[1]): r["holm"], r["bh"] = float(h), float(b)
    f = lambda c: f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"
    fd = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"
    fp = lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"
    lines = [f"# Chain ensemble versus binary relevance, {len(rows)} comparisons (Sant'Orsola, 15 outer folds)", "",
             f"Means over the folds with BCa 95% intervals; difference = chain ensemble minus BR, paired over folds; p, exact sign-flip; Holm and BH across the {len(rows)} comparisons.", "",
             "| Learner | Drug | Metric | BR | Chain ensemble | Difference | p | Holm | BH |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['learner']} | {r['drug']} | {r['metric']} | {f(r['br'])} | {f(r['ens'])} | {fd(r['delta'])} | {fp(r['p'])} | {fp(r['holm'])} | {fp(r['bh'])} |" for r in rows]
    (ROOT / "results/stats/chain_vs_br_18.json").write_text(json.dumps(rows, indent=1))
    (ROOT / "tables/chain_vs_br_18.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
