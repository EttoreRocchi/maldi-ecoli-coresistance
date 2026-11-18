# Regime: learning curves, replication, temporal validation

Run from the repository root, in this order (see the Makefile for the exact arguments).

| Script | Purpose |
|---|---|
| `01_learning_curve.py` | Learning curve of the chain benefit versus training-set size (logistic regression). |
| `02_cross_prediction_lc.py` | Cross-prediction along the learning curves: does proxy superiority track the label-dependency gain? |
| `03_driams_replication.py` | Within-site replication of the chain effect across sites (logistic regression nested CV). |
| `04_temporal_driams_a.py` | Temporal validation within DRIAMS-A (2015 to 2018): leave-one-year-out, binary relevance versus classifier chains (six orderings and their ensemble), logistic regression (liblinear L1/L2 grid by default). |
| `05_margin_gain_correlations.py` | Margin-gain correlations along the learning curves: pooled, within each cohort and partial. |
| `06_exante_margin.py` | Training-only (ex-ante) cross-prediction margin and the stacking gain on the outer folds. |
