# Cross-prediction margin, cross-fitted stacking and the ciprofloxacin strata

Run from the repository root, in this order (see the Makefile for the exact arguments).

| Script | Purpose |
|---|---|
| `01_nested_stacking.py` | Nested stacking across sites and learners, with a permuted-probability control and saved out-of-fold predictions. |
| `02_cross_prediction.py` | Cross-prediction matrix: each binary-relevance classifier scored against each label (no refitting). |
| `03_stacking_clinical.py` | Clinical and calibration profile of the stacked model at Sant'Orsola (logistic regression), against binary relevance and the chain ensemble on the pooled out-of-fold predictions of the same 15 folds. |
| `04_stacking_summary.py` | Merge the nested-stacking results across sites and learners (scripts/04_stacking/01_nested_stacking.py) and the clinical profile of the stacked model (scripts/04_stacking/03_stacking_clinical.py) into two Markdown tables for the supplement. |
| `05_stratified_analysis.py` | Stratified analysis: discrimination and errors on the cephalosporin labels within the ciprofloxacin strata. |
| `06_ctx_without_discordant.py` | Does heterogeneity of the cefotaxime label explain the cross-prediction margin? Refit without the CTX-R, CAZ-S isolates. |
