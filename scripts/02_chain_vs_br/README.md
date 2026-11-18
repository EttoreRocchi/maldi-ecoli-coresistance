# Chain ensemble versus binary relevance

Run from the repository root, in this order (see the Makefile for the exact arguments).

| Script | Purpose |
|---|---|
| `01_nested_cv.py` | Step 3: Within-site nested cross-validation (Binary Relevance + Classifier Chain). |
| `02_significance.py` | Step 7: Within-site significance and confidence intervals (Sant'Orsola). |
| `03_inference_checks.py` | Inference checks for the chain-versus-binary-relevance (BR) comparison on the Sant'Orsola cohort and for the cross-site transfer experiment (no refitting). |
| `04_bagged_br_control.py` | Size-matched bagged binary-relevance ensemble as the control for the chain-ensemble endpoint (logistic regression, Sant'Orsola, same 15 outer folds). |
| `05_flat_multilabel.py` | Flat multi-label comparators on the same outer folds (Sant'Orsola, logistic-regression era baselines). |
| `06_chain_vs_br_table.py` | Chain ensemble versus binary relevance: the 18 comparisons in one table. |
| `07_permuted_label_chains.py` | Diversity-matched null for the chain ensemble: the six chain orderings trained on PERMUTED upstream labels. |
