"""Nested cross-validation orchestration for AMR prediction."""

import logging
import os
import traceback
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from mecor.evaluation.checkpoint import digest, ensure_signature
from mecor.evaluation.grouping import check_no_patient_overlap, inner_splits, outer_splits
from mecor.evaluation.importance import extract_feature_importances
from mecor.evaluation.metrics import calc_binary_metrics, calc_clinical_metrics
from mecor.models.binary_relevance import BinaryRelevanceWrapper
from mecor.models.chain import ClassifierChainWrapper
from mecor.models.factory import get_base_estimator, get_param_grid, needs_scaling
from mecor.models.persistence import load_model, save_model

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single CV fold for a single method."""
    repeat_idx: int
    fold_idx: int
    method: str        # "binary_relevance", "chain_CTX_CAZ_CIP", "ensemble"
    model_type: str
    y_true: Dict[str, np.ndarray] = field(default_factory=dict)
    y_pred: Dict[str, np.ndarray] = field(default_factory=dict)
    y_prob: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    clinical_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    importances: Dict[str, Tuple[np.ndarray, List[str]]] = field(default_factory=dict)


@dataclass
class CVResults:
    """Aggregated results from nested cross-validation."""
    fold_results: List[FoldResult] = field(default_factory=list)
    config: object = None
    labels: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    n_samples: int = 0


def _build_search_estimator(estimator_type: str, config, inner_cv=None) -> object:
    """Build a hyperparameter-tuned pipeline.

    inner_cv: precomputed inner splits (patient-grouped folds); None keeps the default StratifiedKFold.
    """
    steps = []
    if needs_scaling(estimator_type):
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", get_base_estimator(estimator_type, config.seed)))
    pipe = Pipeline(steps)

    param_grid = get_param_grid(estimator_type, config.param_search_mode)

    # No hyperparameters to tune, return pipeline directly (e.g. TabPFN)
    if not param_grid:
        return pipe

    if config.param_search_mode == "random":
        return RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=config.random_search_iterations,
            scoring="roc_auc",
            cv=inner_cv if inner_cv is not None else StratifiedKFold(n_splits=config.n_splits_inner, shuffle=True, random_state=config.seed),
            n_jobs=config.n_jobs,
            random_state=config.seed,
            refit=True,
            verbose=0,
        )
    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv if inner_cv is not None else StratifiedKFold(n_splits=config.n_splits_inner, shuffle=True, random_state=config.seed),
        n_jobs=config.n_jobs,
        refit=True,
        verbose=0,
    )


def _evaluate_method(
    model,
    X_test,
    Y_test,
    method_name: str,
    model_type: str,
    feature_names: List[str],
    repeat_idx: int,
    fold_idx: int,
    labels: List[str],
) -> FoldResult:
    """Evaluate a trained model on test data and return FoldResult."""
    result = FoldResult(
        repeat_idx=repeat_idx,
        fold_idx=fold_idx,
        method=method_name,
        model_type=model_type,
    )

    y_probs = model.predict_proba(X_test)
    y_preds = model.predict(X_test)

    for label in labels:
        y_true = Y_test[label].values
        y_prob = y_probs[label]
        y_pred = y_preds[label]

        result.y_true[label] = y_true
        result.y_pred[label] = y_pred
        result.y_prob[label] = y_prob
        result.metrics[label] = calc_binary_metrics(y_true, y_pred, y_prob)
        result.clinical_metrics[label] = calc_clinical_metrics(y_true, y_pred)

    result.importances = extract_feature_importances(model, feature_names, model_type)

    return result


def run_nested_cv(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    config,
    groups: Optional[np.ndarray] = None,
) -> CVResults:
    """Run nested cross-validation for Binary Relevance and/or Classifier Chains.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    Y : pd.DataFrame
        Binary label matrix (columns = short drug names)
    config : Config
        Pipeline configuration

    Returns
    -------
    CVResults
        Aggregated per-fold results for all methods
    """
    labels = list(Y.columns)
    feature_names = list(X.columns)
    model_type = config.model

    all_orderings = list(permutations(labels)) if config.run_classifier_chain else []

    n_methods = 0
    if config.run_binary_relevance:
        n_methods += 1
    if config.run_classifier_chain:
        n_methods += len(all_orderings)

    total_folds = config.n_splits_outer * config.n_repeats_outer
    total_iterations = total_folds * n_methods

    logger.info("NESTED CROSS-VALIDATION")
    logger.info(f"  Methods: BR={'yes' if config.run_binary_relevance else 'no'}, "
                f"Chain={'yes' if config.run_classifier_chain else 'no'}")
    if config.run_classifier_chain:
        logger.info(f"  Chain orderings: {len(all_orderings)}")
    logger.info(f"  Total iterations: {total_iterations}")
    logger.info("")

    results = CVResults(
        config=config, labels=labels, feature_names=feature_names,
        n_samples=X.shape[0],
    )
    model_dir = Path(config.output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Multi-label stratification: preserves resistance profile frequencies
    # across folds (e.g., "010" = susceptible-resistant-susceptible)
    stratify_col = Y.apply(lambda row: "".join(row.astype(str).values), axis=1)
    if groups is None:
        outer_cv = RepeatedStratifiedKFold(
            n_splits=config.n_splits_outer,
            n_repeats=config.n_repeats_outer,
            random_state=config.seed,
        )
        splits = list(outer_cv.split(X, stratify_col))
    else:
        # patient-grouped outer folds; the inner folds below are grouped as well
        groups = np.asarray(groups)
        splits = outer_splits(stratify_col.values, groups, config.n_splits_outer, config.n_repeats_outer, config.seed)
        check_no_patient_overlap(splits, groups)
        logger.info(f"  Patient-grouped folds: {len(set(groups))} patients")

    # the model cache is valid only for these folds and settings: refuse to resume with anything else
    ensure_signature(model_dir, {
        "model": model_type, "n_samples": int(X.shape[0]), "group_by_patient": groups is not None,
        "test_folds": digest(*[te for _, te in splits]), "param_search_mode": config.param_search_mode,
        "n_splits_inner": config.n_splits_inner, "chain_cv": getattr(config, "chain_cv", None),
        "chain_method": getattr(config, "chain_method", "predict"),
    }, name=f"SIGNATURE_{model_type}.json")   # one signature per learner: the folder is shared by all learners

    pbar = tqdm(total=total_iterations, desc="Training", ncols=100)
    failed_runs = []

    # Worker mode (env NESTED_CV_FOLDS="0,5,10"): fit and cache only these outer folds, so several processes can share
    # one model folder; a final run without the variable then loads every cached model and writes the results.
    only_folds = os.environ.get("NESTED_CV_FOLDS")
    only_folds = {int(i) for i in only_folds.split(",")} if only_folds else None
    for fold_global_idx, (train_idx, test_idx) in enumerate(splits):
        repeat_idx = fold_global_idx // config.n_splits_outer
        fold_idx = fold_global_idx % config.n_splits_outer
        if only_folds is not None and fold_global_idx not in only_folds:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        inner_cv = None
        if groups is not None:
            inner_cv = inner_splits(stratify_col.values[train_idx], groups[train_idx], config.n_splits_inner, config.seed)
            check_no_patient_overlap(inner_cv, groups[train_idx])

        if config.run_binary_relevance:
            pbar.set_description(f"R{repeat_idx+1} F{fold_idx+1} | BR")
            try:
                br = load_model(model_dir, "binary_relevance", repeat_idx, fold_idx, model_type)
                if br is None:
                    base_est = _build_search_estimator(model_type, config, inner_cv)
                    br = BinaryRelevanceWrapper(base_est, labels, config.seed)
                    br.fit(X_train, Y_train)
                    if config.save_fitted_models:
                        save_model(br, model_dir, "binary_relevance", repeat_idx, fold_idx,
                                   model_type, compression=config.model_compression)

                result = _evaluate_method(
                    br, X_test, Y_test, "binary_relevance", model_type,
                    feature_names, repeat_idx, fold_idx, labels,
                )
                results.fold_results.append(result)
            except Exception as e:
                failed_runs.append({"repeat": repeat_idx, "fold": fold_idx, "method": "BR", "error": str(e), "traceback": traceback.format_exc()})
                logger.error(f"Failed: R{repeat_idx+1} F{fold_idx+1} | BR: {e}")
            pbar.update(1)

        if config.run_classifier_chain:
            ensemble_probs = {label: [] for label in labels}

            for ordering in all_orderings:
                order_key = "_".join(ordering)
                pbar.set_description(f"R{repeat_idx+1} F{fold_idx+1} | {' '.join(ordering)}")
                try:
                    chain = load_model(model_dir, "chain", repeat_idx, fold_idx, model_type, list(ordering))
                    if chain is None:
                        base_est = _build_search_estimator(model_type, config, inner_cv)
                        chain = ClassifierChainWrapper(base_est, list(ordering), config.seed,
                                                       cv=getattr(config, "chain_cv", None),
                                                       chain_method=getattr(config, "chain_method", "predict"))
                        chain.fit(X_train, Y_train)
                        if config.save_fitted_models:
                            save_model(chain, model_dir, "chain", repeat_idx, fold_idx,
                                       model_type, list(ordering), config.model_compression)

                    result = _evaluate_method(
                        chain, X_test, Y_test, f"chain_{order_key}", model_type,
                        feature_names, repeat_idx, fold_idx, labels,
                    )
                    results.fold_results.append(result)

                    for label in labels:
                        ensemble_probs[label].append(result.y_prob[label])

                except Exception as e:
                    failed_runs.append({"repeat": repeat_idx, "fold": fold_idx, "method": f"chain_{order_key}", "error": str(e), "traceback": traceback.format_exc()})
                    logger.error(f"Failed: R{repeat_idx+1} F{fold_idx+1} | {order_key}: {e}")
                pbar.update(1)

            if all(len(ensemble_probs[l]) == len(all_orderings) for l in labels):
                ens_result = FoldResult(
                    repeat_idx=repeat_idx,
                    fold_idx=fold_idx,
                    method="ensemble",
                    model_type=model_type,
                )
                for label in labels:
                    y_prob_ens = np.mean(ensemble_probs[label], axis=0)
                    y_pred_ens = (y_prob_ens >= 0.5).astype(int)
                    y_true = Y_test[label].values

                    ens_result.y_true[label] = y_true
                    ens_result.y_pred[label] = y_pred_ens
                    ens_result.y_prob[label] = y_prob_ens
                    ens_result.metrics[label] = calc_binary_metrics(y_true, y_pred_ens, y_prob_ens)
                    ens_result.clinical_metrics[label] = calc_clinical_metrics(y_true, y_pred_ens)

                results.fold_results.append(ens_result)

    pbar.close()

    if failed_runs:
        logger.warning(f"{len(failed_runs)} runs failed")

    return results
