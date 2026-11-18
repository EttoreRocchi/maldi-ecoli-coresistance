"""Sensitivity of the chain benefit to the chain training scheme, per model family.

Compares, on the same 15 outer folds, the canonical runs (downstream links trained on the TRUE upstream labels,
results/s_orsola/<model>) with the out-of-fold-training runs (links trained on 5-fold out-of-fold upstream
predictions, results/cvsens/<model>) for every model family whose cvsens run exists.
Outputs: results/stats/training_scheme_sensitivity.json, tables/training_scheme_sensitivity.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor import shared as la
from mecor.cvio import aligned, bca_mean_ci, chain_methods, load_cv_results, metric_by_fold  # noqa: E402


def best_by_mcc(cv, drug):
    return max(chain_methods(cv), key=lambda m: np.mean(list(metric_by_fold(cv, m, drug, "MCC").values())))


def main():
    out = {}
    lines = ["# Sensitivity to the chain training scheme (same 15 outer folds; chain ensemble minus BR; exact sign-flip p)", ""]
    for model in ("logistic", "catboost", "tabpfn"):
        pk_c = ROOT / f"results/s_orsola/{model}/cv_results.pkl"; pk_s = ROOT / f"results/cvsens/{model}/cv_results.pkl"
        if not pk_s.exists():
            lines.append(f"{model}: out-of-fold run not available"); continue
        cvs = {"true-label training": load_cv_results(pk_c), "out-of-fold training": load_cv_results(pk_s)}
        pk_soft = ROOT / f"results/cvsens_soft/{model}/cv_results.pkl"
        if pk_soft.exists():
            cvs["soft chain (out-of-fold probabilities, probability passing)"] = load_cv_results(pk_soft)
        if len(cvs["out-of-fold training"].fold_results) < 120:
            lines.append(f"{model}: out-of-fold run incomplete ({len(cvs['out-of-fold training'].fold_results)} fold results)"); continue
        out[model] = {}
        lines += [f"## {model}", "", "| Scheme | Drug | BR AUROC | Ens AUROC | dAUROC [CI] p | BR MCC | Ens MCC | dMCC [CI] p | Best ordering | dMCC best (naive p) | VME BR>Ens | ME BR>Ens |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for scheme, cv in cvs.items():
            for d in cv.labels:
                rec = {}
                for met in ("AUROC", "MCC"):
                    a, b = aligned(metric_by_fold(cv, "binary_relevance", d, met), metric_by_fold(cv, "ensemble", d, met))
                    _, lo, hi = bca_mean_ci(b - a)
                    rec[met] = {"BR": float(a.mean()), "Ens": float(b.mean()), "d": float((b - a).mean()), "ci": [lo, hi], "p": la.sign_flip_p(b - a)}
                bc = best_by_mcc(cv, d); a2, b2 = aligned(metric_by_fold(cv, "binary_relevance", d, "MCC"), metric_by_fold(cv, bc, d, "MCC"))
                rec["best"] = {"ordering": bc.replace("chain_", "").replace("_", ">"), "dMCC": float((b2 - a2).mean()), "p_naive": la.sign_flip_p(b2 - a2)}
                for met in ("VME", "ME"):
                    a, b = aligned(metric_by_fold(cv, "binary_relevance", d, met, clinical=True), metric_by_fold(cv, "ensemble", d, met, clinical=True))
                    rec[met] = {"BR": float(a.mean()), "Ens": float(b.mean())}
                out[model].setdefault(scheme, {})[d] = rec
                A, M = rec["AUROC"], rec["MCC"]
                lines.append(f"| {scheme} | {d} | {A['BR']:.3f} | {A['Ens']:.3f} | {A['d']:+.3f} [{A['ci'][0]:+.3f}, {A['ci'][1]:+.3f}] {A['p']:.3f} | {M['BR']:.3f} | {M['Ens']:.3f} | {M['d']:+.3f} [{M['ci'][0]:+.3f}, {M['ci'][1]:+.3f}] {M['p']:.3f} | {rec['best']['ordering']} | {rec['best']['dMCC']:+.3f} ({rec['best']['p_naive']:.3f}) | {rec['VME']['BR']:.3f}>{rec['VME']['Ens']:.3f} | {rec['ME']['BR']:.3f}>{rec['ME']['Ens']:.3f} |")
        lines.append("")
    (ROOT / "results/stats/training_scheme_sensitivity.json").write_text(json.dumps(out, indent=1))
    (ROOT / "tables/training_scheme_sensitivity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
