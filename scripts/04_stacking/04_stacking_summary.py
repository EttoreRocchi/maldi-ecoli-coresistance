"""Merge the nested-stacking results across sites and learners (scripts/04_stacking/01_nested_stacking.py) and the clinical profile of the stacked
model (scripts/04_stacking/03_stacking_clinical.py) into two Markdown tables for the supplement.
Outputs: tables/nested_stacking_all.md, tables/stacking_clinical_summary.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
SITE = {"s_orsola": "Sant'Orsola", "driams_b": "DRIAMS-B", "driams_c": "DRIAMS-C", "driams_d": "DRIAMS-D", "driams_a": "DRIAMS-A"}
LEARNER = {"logistic": "logistic regression", "catboost": "CatBoost", "tabpfn": "TabPFN"}
RUNS = [("s_orsola_logistic_full", "Sant'Orsola", "logistic regression (full grid)"), ("s_orsola_catboost", "Sant'Orsola", "CatBoost"), ("s_orsola_tabpfn", "Sant'Orsola", "TabPFN"),
        ("driams_b_logistic", "DRIAMS-B", "logistic regression (liblinear grid)"), ("driams_c_logistic", "DRIAMS-C", "logistic regression (liblinear grid)"),
        ("driams_d_logistic", "DRIAMS-D", "logistic regression (liblinear grid)"), ("driams_a_grouped_logistic", "DRIAMS-A", "logistic regression (liblinear grid)")]


def fmt(c):
    return f"{c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] {c['p']:.3f}"


def main():
    lines = ["# Nested stacking across sites and learners (15 outer folds each; meta-logistic on the three cross-fitted BR probabilities)", "",
             "| Site | Learner | Drug | BR AUROC | Stacked AUROC | Stacked vs BR [CI] p | dMCC (p) | Stacked vs permuted others [CI] p | Stacked vs chain ensemble [CI] p |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, site, learner in RUNS:
        f = ROOT / f"results/stats/nested_stacking_{tag}.json"
        if not f.exists(): print("missing", tag); continue
        d = json.load(open(f))
        for r in d["rows"]:
            ens = fmt(r["stack_vs_ens"]) if "stack_vs_ens" in r else "n/a"
            lines.append(f"| {site} | {learner} | {r['drug']} | {r['br_auroc']:.3f} | {r['stack_auroc']:.3f} | {fmt(r['stack_vs_br'])} | {r['stack_vs_br']['dMCC']:+.3f} ({r['stack_vs_br']['p_mcc']:.3f}) | {fmt(r['stack_vs_perm'])} | {ens} |")
    (ROOT / "tables/nested_stacking_all.md").write_text("\n".join(lines) + "\n")
    c = json.load(open(ROOT / "results/stats/stacking_clinical_s_orsola_logistic_full.json"))
    names = {"AUROC": "AUROC", "MCC": "MCC (0.5)", "VME": "VME (0.5)", "ME": "ME (0.5)", "Brier_skill": "Brier skill score", "cal_slope": "Calibration slope", "cal_intercept": "Calibration intercept",
             "NB_0.30": "Net benefit at 0.30", "NB_0.40": "Net benefit at 0.40", "NB_0.50": "Net benefit at 0.50", "NB_0.60": "Net benefit at 0.60", "NB_prev": "Net benefit at the prevalence"}
    lines = ["# Clinical profile of the stacked model (Sant'Orsola, logistic regression, full grid; pooled out-of-fold predictions; isolate bootstrap 95% CI)", "",
             "| Drug | Metric | BR | Chain ensemble | Stacked | Stacked minus BR [CI] | Stacked minus chain ensemble [CI] |", "|---|---|---|---|---|---|---|"]
    for d in ("CIP", "CTX", "CAZ"):
        for k, nm in names.items():
            pt = c["point"][d]; c1 = c["contrasts"][d]["stack-br"][k]; c2 = c["contrasts"][d]["stack-ens"][k]
            lines.append(f"| {d} | {nm} | {pt['br'][k]:.3f} | {pt['ens'][k]:.3f} | {pt['stack'][k]:.3f} | {c1['delta']:+.3f} [{c1['ci'][0]:+.3f}, {c1['ci'][1]:+.3f}] | {c2['delta']:+.3f} [{c2['ci'][0]:+.3f}, {c2['ci'][1]:+.3f}] |")
    (ROOT / "tables/stacking_clinical_summary.md").write_text("\n".join(lines) + "\n"); print("tables written")


if __name__ == "__main__":
    main()
