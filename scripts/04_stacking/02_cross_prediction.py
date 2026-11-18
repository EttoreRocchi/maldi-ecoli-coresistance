"""Cross-prediction matrix: each binary-relevance classifier scored against each label (no refitting).

Reads the per-fold out-of-fold predictions saved by scripts/04_stacking/01_nested_stacking.py (results/nested_stacking/<tag>/r*_f*.npz: y, br, stack)
and, for every run, reports the AUROC of each BR probability against each label, the stacked model, and two paired
contrasts per label over the 15 outer folds: best other-label classifier minus own classifier, and stacked model minus
that proxy (BCa 95% interval, exact sign-flip p).
Outputs: results/stats/cross_prediction.json, tables/cross_prediction.md
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.cvio import bca_mean_ci  # noqa: E402

RUNS = [("s_orsola_logistic_full", "Sant'Orsola", "logistic regression (full grid)", ["CIP", "CTX", "CAZ"]),
        ("s_orsola_catboost", "Sant'Orsola", "CatBoost", ["CIP", "CTX", "CAZ"]),
        ("s_orsola_tabpfn", "Sant'Orsola", "TabPFN", ["CIP", "CTX", "CAZ"]),
        ("driams_b_logistic", "DRIAMS-B", "logistic regression (liblinear grid)", ["CIP", "CRO", "CAZ"]),
        ("driams_c_logistic", "DRIAMS-C", "logistic regression (liblinear grid)", ["CIP", "CRO", "CAZ"]),
        ("driams_d_logistic", "DRIAMS-D", "logistic regression (liblinear grid)", ["CIP", "CRO", "CAZ"]),
        ("driams_a_logistic", "DRIAMS-A", "logistic regression (liblinear grid)", ["CIP", "CRO", "CAZ"])]
# --grouped: DRIAMS-A from the patient-grouped stacking run; outputs get the suffix _grouped (the DRIAMS-A analyses of the paper)
GROUPED = "--grouped" in sys.argv
SFX = "_grouped" if GROUPED else ""
if GROUPED:
    RUNS = [(t.replace("driams_a_logistic", "driams_a_grouped_logistic"), *rest) for t, *rest in RUNS]


def sign_flip_p(d):
    d = np.asarray(d, float); n = len(d); obs = abs(d.mean())
    signs = np.array(list(product([-1, 1], repeat=n)))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def contrast(d):
    m, lo, hi = bca_mean_ci(d)
    return {"delta": m, "ci": [lo, hi], "p": sign_flip_p(d)}


def main():
    out, lines = [], ["# Cross-prediction matrix: AUROC of each binary-relevance (BR) classifier scored against each label (15 outer folds; mean over folds)", "",
                      "Proxy = the other-label BR classifier with the highest mean AUROC for the label. Contrasts are paired over the 15 folds (BCa 95% CI, exact sign-flip p).", "",
                      "| Site | Learner | Label | BR CIP | BR CTX/CRO | BR CAZ | Stacked | Best proxy | Proxy minus own [CI] p | Stacked minus proxy [CI] p |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, site, learner, names in RUNS:
        files = sorted((ROOT / f"results/nested_stacking/{tag}").glob("r*_f?.npz"))
        if not files: print("missing", tag); continue
        folds = [np.load(f) for f in files]
        auc = np.array([[[roc_auc_score(d["y"][:, i], d["br"][:, j]) for j in range(3)] for i in range(3)] for d in folds])   # fold x label x classifier
        stk = np.array([[roc_auc_score(d["y"][:, i], d["stack"][:, i]) for i in range(3)] for d in folds])
        for i, lab in enumerate(names):
            others = [j for j in range(3) if j != i]
            j = max(others, key=lambda k: auc[:, i, k].mean())
            row = {"site": site, "learner": learner, "tag": tag, "label": lab, "n_folds": len(folds),
                   "auroc_by_classifier": {names[k]: float(auc[:, i, k].mean()) for k in range(3)}, "stacked": float(stk[:, i].mean()),
                   "proxy": names[j], "proxy_minus_own": contrast(auc[:, i, j] - auc[:, i, i]), "stacked_minus_proxy": contrast(stk[:, i] - auc[:, i, j])}
            out.append(row)
            f = lambda c: f"{c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] {c['p']:.3f}"
            a = row["auroc_by_classifier"]
            lines.append(f"| {site} | {learner} | {lab} | {a[names[0]]:.3f} | {a[names[1]]:.3f} | {a[names[2]]:.3f} | {row['stacked']:.3f} | {names[j]} | {f(row['proxy_minus_own'])} | {f(row['stacked_minus_proxy'])} |")
    (ROOT / f"results/stats/cross_prediction{SFX}.json").write_text(json.dumps(out, indent=1))
    (ROOT / f"tables/cross_prediction{SFX}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
