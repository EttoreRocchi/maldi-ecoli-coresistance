# Reproduce the analyses of the paper, one target per Results section.
#   make all       Sant'Orsola, logistic regression (the default; a few hours on a many-core machine)
#   make driams    the DRIAMS analyses (download DRIAMS first, see data/README.md)
#   make catboost  CatBoost runs (pip install -e ".[catboost]")
#   make tabpfn    TabPFN runs (separate environment, see README; GPU recommended)
#   make figures   all main and supplementary figures (needs all of the above)
#   make full      everything
# Outputs: results/ (statistics in results/stats/*.json), tables/ (Markdown), figures/main and figures/supplementary.

PY ?= python
S := scripts

.PHONY: help all full data cooccurrence chains mechanism stacking regime clinical driams catboost tabpfn summaries figures lint

help:  ## list the targets
	@grep -E '^[a-z_-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t 16

all: data cooccurrence chains mechanism stacking regime clinical  ## Sant'Orsola with logistic regression

full: all driams catboost tabpfn summaries figures  ## every analysis of the paper, then all figures

data:  ## preprocess the Sant'Orsola spectra (MaldiAMRKit 0.7.0) and validate the cohort
	$(PY) $(S)/00_data/01_prepare_data.py --config configs/s_orsola.yaml --preprocess

cooccurrence:  ## co-resistance profiles at Sant'Orsola (Fig. 1b)
	$(PY) $(S)/01_cooccurrence/01_cooccurrence.py --configs configs/s_orsola.yaml

chains:  ## nested CV, BR versus chains, inference, bagged and permuted-label controls, flat multi-label comparators (Table 1, Fig. 2, Tables S4, S5, S27)
	$(PY) $(S)/02_chain_vs_br/01_nested_cv.py --config configs/s_orsola.yaml --model logistic
	$(PY) $(S)/02_chain_vs_br/02_significance.py
	$(PY) $(S)/02_chain_vs_br/03_inference_checks.py
	$(PY) $(S)/02_chain_vs_br/04_bagged_br_control.py
	$(PY) $(S)/02_chain_vs_br/05_flat_multilabel.py
	$(PY) $(S)/02_chain_vs_br/07_permuted_label_chains.py --site s_orsola

mechanism:  ## link ablation, upstream accuracy, ceiling-to-gain, training scheme (Fig. 3, Tables S6, S7, S9, S10)
	$(PY) $(S)/02_chain_vs_br/01_nested_cv.py --config configs/cvsens_local.yaml --model logistic
	$(PY) $(S)/02_chain_vs_br/01_nested_cv.py --config configs/cvsens_soft_local.yaml --model logistic
	$(PY) $(S)/03_mechanism/01_link_ablation.py
	$(PY) $(S)/03_mechanism/02_upstream_accuracy.py
	$(PY) $(S)/03_mechanism/03_ceiling_to_gain.py
	$(PY) $(S)/03_mechanism/04_training_scheme.py
	$(PY) $(S)/05_regime/01_learning_curve.py --site s_orsola --grid reduced

stacking:  ## cross-fitted stacking, cross-prediction matrix, ciprofloxacin strata, CTX label heterogeneity (Fig. 4, Fig. S1, Tables S12 to S16)
	$(PY) $(S)/04_stacking/01_nested_stacking.py --site s_orsola --model logistic --grid full --with-spectrum --save-inner
	$(PY) $(S)/04_stacking/02_cross_prediction.py --grouped
	$(PY) $(S)/04_stacking/03_stacking_clinical.py --tag s_orsola_logistic_full
	$(PY) $(S)/04_stacking/05_stratified_analysis.py --grouped
	$(PY) $(S)/04_stacking/06_ctx_without_discordant.py

regime:  ## learning curves and the cross-prediction margin at Sant'Orsola (Fig. S2, Table S17)
	$(PY) $(S)/05_regime/01_learning_curve.py --site s_orsola --stacking
	$(PY) $(S)/05_regime/01_learning_curve.py --site s_orsola --stacking --inner
	$(PY) $(S)/05_regime/01_learning_curve.py --site s_orsola --grid full --sizes 120,187
	$(PY) $(S)/05_regime/02_cross_prediction_lc.py --sites s_orsola

clinical:  ## decision curves, operating points, calibration, profile errors (Fig. S4, Tables S25, S26)
	$(PY) $(S)/06_clinical/01_clinical_decision.py

driams:  ## DRIAMS (DRIAMS-A within-site runs patient-grouped): co-resistance, replication, stacking, learning curve, temporal, transfer (Table 2, Fig. 5, Tables S1, S2, S7, S12 to S18, S20 to S24)
	$(PY) $(S)/01_cooccurrence/01_cooccurrence.py
	for s in driams_a_grouped driams_b driams_c driams_d; do $(PY) $(S)/02_chain_vs_br/01_nested_cv.py --config configs/$$s.yaml --model logistic || exit 1; done
	$(PY) $(S)/01_cooccurrence/02_coresistance_structure.py
	$(PY) $(S)/03_mechanism/03_ceiling_to_gain.py
	$(PY) $(S)/05_regime/03_driams_replication.py --grouped
	for s in driams_a_grouped driams_b driams_c driams_d; do $(PY) $(S)/04_stacking/01_nested_stacking.py --site $$s --model logistic --save-inner || exit 1; done
	$(PY) $(S)/04_stacking/02_cross_prediction.py --grouped
	$(PY) $(S)/04_stacking/05_stratified_analysis.py --grouped
	$(PY) $(S)/05_regime/01_learning_curve.py --site driams_a_grouped --stacking --inner
	$(PY) $(S)/05_regime/02_cross_prediction_lc.py --sites driams_a_grouped,s_orsola
	$(PY) $(S)/05_regime/05_margin_gain_correlations.py --grouped
	$(PY) $(S)/05_regime/04_temporal_driams_a.py
	$(PY) $(S)/07_transfer/01_cross_dataset.py --model logistic
	$(PY) $(S)/07_transfer/04_cross_dataset_stacking.py
	$(PY) $(S)/07_transfer/02_cross_site_mixedeffects.py
	$(PY) $(S)/07_transfer/03_domain_adaptation.py --target s_orsola
	$(PY) $(S)/07_transfer/03_domain_adaptation.py --target driams_b

catboost:  ## CatBoost nested CV, out-of-fold training and stacking (tens of hours)
	$(PY) $(S)/02_chain_vs_br/01_nested_cv.py --config configs/s_orsola.yaml --model catboost
	$(PY) $(S)/02_chain_vs_br/01_nested_cv.py --config configs/cvsens_local.yaml --model catboost
	$(PY) $(S)/04_stacking/01_nested_stacking.py --site s_orsola --model catboost

tabpfn:  ## TabPFN nested CV, training schemes, link ablation and stacking (set TABPFN_MODEL_PATH; GPU recommended)
	$(PY) $(S)/02_chain_vs_br/01_nested_cv.py --config configs/s_orsola.yaml --model tabpfn
	$(PY) $(S)/02_chain_vs_br/01_nested_cv.py --config configs/cvsens_local.yaml --model tabpfn
	$(PY) $(S)/02_chain_vs_br/01_nested_cv.py --config configs/cvsens_soft_local.yaml --model tabpfn
	$(PY) $(S)/03_mechanism/05_link_ablation_nonlinear.py --model tabpfn --n-perm 5
	$(PY) $(S)/04_stacking/01_nested_stacking.py --site s_orsola --model tabpfn

summaries:  ## recompute the statistics that pool the three learners and all sites (Tables S3, S12 to S14, S16, S19)
	$(PY) $(S)/02_chain_vs_br/02_significance.py
	$(PY) $(S)/02_chain_vs_br/03_inference_checks.py
	$(PY) $(S)/03_mechanism/04_training_scheme.py
	$(PY) $(S)/04_stacking/02_cross_prediction.py --grouped
	$(PY) $(S)/04_stacking/04_stacking_summary.py
	$(PY) $(S)/04_stacking/05_stratified_analysis.py --grouped
	$(PY) $(S)/05_regime/06_exante_margin.py
	$(PY) $(S)/02_chain_vs_br/06_chain_vs_br_table.py
	$(PY) $(S)/06_clinical/01_clinical_decision.py

figures:  ## main figures 1 to 5 and supplementary figures S1 to S4 (needs every analysis)
	$(PY) $(S)/figures_tables/01_main_figures.py 1 2 4 supp
	$(PY) $(S)/figures_tables/02_fig3_mechanism.py
	$(PY) $(S)/figures_tables/03_fig5_regime.py
	$(PY) $(S)/figures_tables/04_supp_learning_curves.py
	$(PY) $(S)/06_clinical/01_clinical_decision.py --plot-only --model logistic

lint:  ## ruff
	ruff check src scripts
