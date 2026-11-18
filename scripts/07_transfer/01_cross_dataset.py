#!/usr/bin/env python3
"""Step 4: Cross-dataset validation (train on one site, test on another).

Fits Binary Relevance and every chain ordering on each site, then evaluates all
ordered train/test site pairs to test whether the models transfer across sites
(external validity). Fitted models are cached so reruns are fast. Results are
written as JSON for the figure scripts.

Usage:
    python scripts/07_transfer/01_cross_dataset.py
    python scripts/07_transfer/01_cross_dataset.py --model catboost
    python scripts/07_transfer/01_cross_dataset.py --no-cache
"""

import argparse
import json
import logging
from itertools import permutations
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mecor.config import Config
from mecor.data.loader import load_dataset
from mecor.evaluation.metrics import calc_binary_metrics, calc_clinical_metrics
from mecor.models.binary_relevance import BinaryRelevanceWrapper
from mecor.models.chain import ClassifierChainWrapper
from mecor.models.factory import get_base_estimator, get_param_grid, needs_scaling
from mecor.models.persistence import load_cross_model, save_cross_model

CROSS_CONFIGS = [
    "configs/s_orsola.yaml",
    "configs/driams_a.yaml",
    "configs/driams_b.yaml",
    "configs/driams_c.yaml",
    "configs/driams_d.yaml",
]

# CTX (Cefotaxime) and CRO (Ceftriaxone) are interchangeable ESBL markers
# with identical EUCAST breakpoints. Map to canonical labels for alignment.
CANONICAL_LABELS = ["CIP", "CRO", "CAZ"]
LABEL_MAP = {"CTX": "CRO"}  # local uses CTX, DRIAMS uses CRO


def _build_search_estimator(model_type: str, config) -> object:
    """Build a hyperparameter-tuned pipeline for cross-dataset eval."""
    steps = []
    if needs_scaling(model_type):
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", get_base_estimator(model_type, config.seed)))
    pipe = Pipeline(steps)
    param_grid = get_param_grid(model_type, config.param_search_mode)

    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=config.n_splits_inner, shuffle=True,
                           random_state=config.seed),
        n_jobs=config.n_jobs,
        refit=True,
        verbose=0,
    )



def _fit_or_load_models(train_name, X_train, Y_train, config_train, model_type,
                        labels, model_dir, logger):
    """Fit BR + all chain orderings for one train dataset, or load from cache.

    Returns (br, chains_dict) where chains_dict maps ordering tuple -> fitted chain.
    """
    all_orderings = list(permutations(labels))

    br = load_cross_model(model_dir, train_name, "binary_relevance", model_type)
    if br is None:
        logger.info(f"  Fitting Binary Relevance on {train_name}...")
        base_est = _build_search_estimator(model_type, config_train)
        br = BinaryRelevanceWrapper(base_est, labels, config_train.seed)
        br.fit(X_train, Y_train)
        save_cross_model(br, model_dir, train_name, "binary_relevance", model_type,
                         compression=config_train.model_compression)
    else:
        logger.info(f"  Loaded cached BR for {train_name}")

    chains = {}
    for ordering in all_orderings:
        order_list = list(ordering)
        chain = load_cross_model(model_dir, train_name, "chain", model_type,
                                 chain_order=order_list)
        if chain is None:
            logger.info(f"  Fitting chain {order_list} on {train_name}...")
            base_est = _build_search_estimator(model_type, config_train)
            chain = ClassifierChainWrapper(base_est, order_list, config_train.seed)
            chain.fit(X_train, Y_train)
            save_cross_model(chain, model_dir, train_name, "chain", model_type,
                             chain_order=order_list,
                             compression=config_train.model_compression)
        else:
            logger.info(f"  Loaded cached chain {order_list} for {train_name}")
        chains[ordering] = chain

    return br, chains


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset validation")
    parser.add_argument("--model", default="logistic",
                        help="Model type (logistic, catboost, tabpfn)")
    parser.add_argument("--output-dir", default="results/cross_dataset",
                        help="Output directory")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force retraining (ignore cached models)")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.no_cache:
        import shutil
        if model_dir.exists():
            shutil.rmtree(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared model cache (--no-cache)")

    datasets = {}
    for config_path in CROSS_CONFIGS:
        if not Path(config_path).exists():
            continue
        config = Config.from_yaml(config_path)
        config.model = args.model
        try:
            X, Y = load_dataset(config)
            # Map labels to canonical names (CTX -> CRO)
            Y = Y.rename(columns=LABEL_MAP)
            datasets[config.dataset_name] = (X, Y, config)
            logger.info(f"Loaded {config.dataset_name}: {X.shape}, labels: {list(Y.columns)}")
        except Exception as e:
            logger.error(f"Failed to load {config_path}: {e}")

    if len(datasets) < 2:
        logger.error("Need at least 2 datasets for cross-dataset validation")
        return

    labels = CANONICAL_LABELS

    # Phase 1: Fit (or load) models for each training dataset
    trained_models = {}
    for train_name, (X_train, Y_train, config_train) in datasets.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training models on: {train_name} (n={X_train.shape[0]})")
        br, chains = _fit_or_load_models(
            train_name, X_train, Y_train, config_train,
            args.model, labels, model_dir, logger,
        )
        trained_models[train_name] = (br, chains)

    # Phase 2: Evaluate all train->test pairs.
    all_results = []
    for train_name, (X_train, Y_train, config_train) in datasets.items():
        br, chains = trained_models[train_name]

        for test_name, (X_test, Y_test, config_test) in datasets.items():
            if train_name == test_name:
                continue

            logger.info(f"\n  Evaluating: {train_name} -> {test_name}")

            result_entry = {
                "train": train_name,
                "test": test_name,
                "model": args.model,
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "methods": {},
            }

            try:
                y_probs = br.predict_proba(X_test)
                y_preds = br.predict(X_test)
                br_metrics = {}
                for label in labels:
                    m = calc_binary_metrics(Y_test[label].values, y_preds[label], y_probs[label])
                    m.update(calc_clinical_metrics(Y_test[label].values, y_preds[label]))
                    br_metrics[label] = m
                result_entry["methods"]["binary_relevance"] = br_metrics
                logger.info("    BR AUROC: " + ' | '.join(f'{l}={br_metrics[l]["AUROC"]:.3f}' for l in labels))
            except Exception as e:
                logger.error(f"    BR eval failed: {e}")

            try:
                ensemble_probs = {label: [] for label in labels}
                for _, chain in chains.items():
                    y_probs = chain.predict_proba(X_test)
                    for label in labels:
                        ensemble_probs[label].append(y_probs[label])

                chain_metrics = {}
                for label in labels:
                    y_prob_ens = np.mean(ensemble_probs[label], axis=0)
                    y_pred_ens = (y_prob_ens >= 0.5).astype(int)
                    m = calc_binary_metrics(Y_test[label].values, y_pred_ens, y_prob_ens)
                    m.update(calc_clinical_metrics(Y_test[label].values, y_pred_ens))
                    chain_metrics[label] = m
                result_entry["methods"]["ensemble"] = chain_metrics
                logger.info("    Ens AUROC: " + ' | '.join(f'{l}={chain_metrics[l]["AUROC"]:.3f}' for l in labels))
            except Exception as e:
                logger.error(f"    Chain eval failed: {e}")

            all_results.append(result_entry)

    with open(output_dir / "cross_dataset_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved results to {output_dir / 'cross_dataset_results.json'}")

    _print_summary(all_results, labels)


def _print_summary(results, labels):
    """Print cross-dataset AUROC summary."""
    print(f"\n{'='*70}")
    print("CROSS-DATASET AUROC SUMMARY")
    print(f"{'='*70}")

    for label in labels:
        print(f"\n--- {label} ---")
        print(f"{'Train -> Test':<30} {'BR AUROC':>12} {'Chain AUROC':>12} {'Delta':>10}")
        print("-" * 64)

        for r in results:
            pair = f"{r['train']} -> {r['test']}"
            br_auc = r["methods"].get("binary_relevance", {}).get(label, {}).get("AUROC", float("nan"))
            ch_auc = r["methods"].get("ensemble", {}).get(label, {}).get("AUROC", float("nan"))
            delta = ch_auc - br_auc if not (np.isnan(br_auc) or np.isnan(ch_auc)) else float("nan")
            print(f"{pair:<30} {br_auc:>12.3f} {ch_auc:>12.3f} {delta:>+10.3f}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
