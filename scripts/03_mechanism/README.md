# Mechanism

Run from the repository root, in this order (see the Makefile for the exact arguments).

| Script | Purpose |
|---|---|
| `01_link_ablation.py` | Directed-link ablation of the classifier chains on the Sant'Orsola cohort (logistic regression, saved fold models, no refitting). |
| `02_upstream_accuracy.py` | Upstream accuracy at constant co-resistance (Sant'Orsola, logistic regression, saved downstream models). |
| `03_ceiling_to_gain.py` | Ceiling-to-gain curves: realized downstream gain as a function of upstream accuracy, by label noise on the true upstream label fed to the saved downstream link (no spectral degradation, no refitting). |
| `04_training_scheme.py` | Sensitivity of the chain benefit to the chain training scheme, per model family. |
| `05_link_ablation_nonlinear.py` | Directed-link ablation for the non-linear learners (CatBoost, TabPFN) on the saved Sant'Orsola chain models: realized, oracle, permuted and probability upstream inputs per ordering and rebuilt ensembles, plus single-link permutation importance at position 3. |
