# Transfer between sites

Run from the repository root, in this order (see the Makefile for the exact arguments).

| Script | Purpose |
|---|---|
| `01_cross_dataset.py` | Step 4: Cross-dataset validation (train on one site, test on another). |
| `02_cross_site_mixedeffects.py` | Does chain benefit scale with co-resistance strength? A dependence-aware test. |
| `03_domain_adaptation.py` | Cross-site transfer with domain adaptation (logistic regression, reduced grid). |
| `04_cross_dataset_stacking.py` | Cross-site transfer of cross-fitted stacking (logistic regression). |
