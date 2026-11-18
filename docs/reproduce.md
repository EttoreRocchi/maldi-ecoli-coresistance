# Reproducing the article

`make help` lists every target. Each step writes its statistics to `results/stats/*.json`, a readable Markdown version to `tables/` and figures to `figures/main` and `figures/supplementary`. Only the statistics are versioned; everything else in `results/`, `tables/` and `figures/` is generated locally. Long scripts write a checkpoint per fold or subsample and resume where they stopped.

Approximate run times on a 128-core machine: `make all` a few hours (the full-grid stacking with the spectrum variant dominates); `make driams` about a day (the DRIAMS-A analyses dominate: patient-grouped nested cross-validation and stacking, learning curve, temporal validation); `make catboost` and `make tabpfn` tens of hours each (TabPFN on GPU).

## Patient grouping at DRIAMS-A

DRIAMS-A is the only public site with patient identifiers (`patient_no` and `case_no` in `id/<year>/<year>_strat.csv`; they are specific to a year). Every within-site analysis of DRIAMS-A in the article (nested cross-validation, stacking, cross-prediction, ciprofloxacin strata, learning curves, ceiling-to-gain, training-only margin) uses `configs/driams_a_grouped.yaml`: isolates linked by a shared patient or hospital case stay on the same side of every outer and inner split, and the learning-curve subsamples are whole patients, stratified by the patient's resistance profile. The scripts select this run with `--site driams_a_grouped` or `--grouped` (outputs carry `driams_a_grouped` or the suffix `_grouped`). `configs/driams_a.yaml` is used where whole years or the whole site are the unit: co-resistance, leave-one-year-out validation and transfer between sites. DRIAMS-B to D have no patient identifier.

## Where each figure and table comes from

| Item | Script | Make target |
|---|---|---|
| Fig. 1 | `figures_tables/01_main_figures.py 1` (from `01_cooccurrence/01` and `02`) | `cooccurrence`, `driams`, `figures` |
| Table 1, Fig. 2 | `02_chain_vs_br/01_nested_cv.py`, `02_chain_vs_br/02_significance.py`, `04_stacking/01` (stacked model in d, e); `figures_tables/01_main_figures.py 2` | `chains`, `stacking`, `figures` |
| Fig. 3 | `03_mechanism/01` to `03`; `figures_tables/02_fig3_mechanism.py` | `mechanism`, `figures` |
| Fig. 4 | `04_stacking/01`, `02`, `05`, `05_regime/02`; `figures_tables/01_main_figures.py 4` | `stacking`, `regime`, `driams`, `catboost`, `tabpfn`, `summaries`, `figures` |
| Table 2 | `05_regime/03_driams_replication.py --grouped` | `driams` |
| Fig. 5 | `05_regime/01`, `03`, `04`, `04_stacking/01`, `01_cooccurrence/02`, `07_transfer/03`; `figures_tables/03_fig5_regime.py` | `stacking`, `driams`, `figures` |
| Tables S1, S2, S8 | `01_cooccurrence/02_coresistance_structure.py` | `driams` |
| Table S3 | `02_chain_vs_br/06_chain_vs_br_table.py` | `summaries` |
| Table S4 | `02_chain_vs_br/03_inference_checks.py` | `chains`, `summaries` |
| Table S5 | `02_chain_vs_br/04_bagged_br_control.py`, `02_chain_vs_br/07_permuted_label_chains.py` | `chains` |
| Fig. S1 | `figures_tables/01_main_figures.py supp` | `stacking`, `figures` |
| Tables S6, S7 | `03_mechanism/01_link_ablation.py`, `03_mechanism/03_ceiling_to_gain.py` | `mechanism`, `driams` |
| Tables S9, S10, S11 | `03_mechanism/04_training_scheme.py`, `05_regime/01_learning_curve.py --grid reduced`, `03_mechanism/05_link_ablation_nonlinear.py` | `mechanism`, `catboost`, `tabpfn` |
| Tables S12, S13 | `04_stacking/01`, `02`, `04` | `stacking`, `driams`, `catboost`, `tabpfn`, `summaries` |
| Tables S14, S16 | `04_stacking/05_stratified_analysis.py` | `stacking`, `summaries` |
| Table S15 | `04_stacking/06_ctx_without_discordant.py` | `stacking` |
| Fig. S2, Tables S17, S18 | `05_regime/01_learning_curve.py`, `05_regime/02_cross_prediction_lc.py`, `05_regime/05_margin_gain_correlations.py`; `figures_tables/04_supp_learning_curves.py` | `regime`, `driams`, `figures` |
| Table S19 | `05_regime/06_exante_margin.py` (stacking runs saved with `--save-inner`, learning curves with `--inner`) | `summaries` |
| Tables S20, S21 | `05_regime/04_temporal_driams_a.py` | `driams` |
| Fig. S3, Table S22 | `07_transfer/01_cross_dataset.py`, `07_transfer/04_cross_dataset_stacking.py`, `07_transfer/02_cross_site_mixedeffects.py`; `figures_tables/01_main_figures.py supp` | `driams`, `figures` |
| Tables S23, S24 | `07_transfer/03_domain_adaptation.py` | `driams` |
| Fig. S4, Tables S25, S26 | `06_clinical/01_clinical_decision.py` (stacked curves from `04_stacking/01`) | `clinical`, `stacking`, `figures` |
| Table S27 | `02_chain_vs_br/05_flat_multilabel.py` | `chains` |
| Table S28 | `configs/` and `src/mecor/models/factory.py` (settings and grids) | |

## Software versions

Python 3.10 or later. The exact versions are pinned in `requirements/`: `lock-core.txt` covers every logistic-regression analysis, the preprocessing and the figures; `pip install -e ".[catboost]"` adds CatBoost. The main nested cross-validation was trained with scikit-learn 1.6.1 and the follow-up analyses with scikit-learn 1.7.2; rerunning the main nested cross-validation from the raw spectra in the pinned core environment reproduces every out-of-fold prediction of the article exactly. Spectra were preprocessed with [MaldiAMRKit](https://github.com/EttoreRocchi/MaldiAMRKit) 0.7.0 (the spectral processing library of [MaldiSuite](https://github.com/EttoreRocchi/MaldiSuite)), which reproduces the stored feature matrix exactly. CatBoost 1.2.8 was used for the main CatBoost runs.

TabPFN 6.3.0 needs PyTorch and an older numpy; it was run in a separate environment (`requirements/lock-tabpfn.txt`), with the TabPFN v2.5 default classifier checkpoint passed through `TABPFN_MODEL_PATH`. A GPU is strongly recommended.

## Fonts

The figures are set in TeX Gyre Pagella (Palatino), the face of the article. `mecor.visualization.paper_style` reads the OpenType files from the TeX Live path or from `PAPER_PAGELLA_OTF` (a path pattern with `{}` for `regular`, `bold`, `italic` and `bolditalic`) and converts them once to TrueType in `~/.cache/maldi-ecoli-fonts`; without them another serif face is used and text widths differ slightly from the article.

## Code style

`ruff check src scripts` (configuration in `pyproject.toml`); `pre-commit install` runs it on every commit.
