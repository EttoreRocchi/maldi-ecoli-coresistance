#!/usr/bin/env python3
"""Step 3: Within-site nested cross-validation (Binary Relevance + Classifier Chain).

Runs 5x3 repeated stratified nested CV for one dataset and one model family,
fitting Binary Relevance and all six classifier-chain orderings (plus their
ensemble). The full CVResults object is pickled for the downstream statistics and
figure scripts; the config is saved alongside it.

Usage:
    python scripts/02_chain_vs_br/01_nested_cv.py --config configs/s_orsola.yaml --model logistic
    python scripts/02_chain_vs_br/01_nested_cv.py --config configs/driams_c.yaml --model catboost
"""

import argparse
import logging
import os
import time
from pathlib import Path

from mecor.config import Config
from mecor.data.loader import load_dataset
from mecor.data.quality import validate_data
from mecor.evaluation.checkpoint import atomic_pickle_dump
from mecor.evaluation.cv import run_nested_cv
from mecor.evaluation.grouping import load_patient_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Run the within-site AMR prediction pipeline")
    parser.add_argument("--config", required=True, help="Path to a dataset config YAML")
    parser.add_argument("--model", default=None,
                        help="Override the model type (logistic, catboost, tabpfn)")
    parser.add_argument("--br-only", action="store_true",
                        help="Run only Binary Relevance")
    parser.add_argument("--chain-only", action="store_true",
                        help="Run only the Classifier Chains")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override the config's n_jobs")
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config.from_yaml(args.config)
    if args.model:
        config.model = args.model
    if args.br_only:
        config.run_classifier_chain = False
    if args.chain_only:
        config.run_binary_relevance = False
    if args.n_jobs:
        config.n_jobs = args.n_jobs

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(config.output_dir) / config.model
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading: {config.dataset_name}")
    X, Y = load_dataset(config)
    validate_data(X, Y, config)
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Labels: {list(Y.columns)}")
    logger.info(f"Model: {config.model}")

    groups = load_patient_groups(config, X.index)   # None unless config.group_by_patient
    t0 = time.time()
    cv_results = run_nested_cv(X, Y, config, groups=groups)
    elapsed = time.time() - t0
    logger.info(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Total fold results: {len(cv_results.fold_results)}")
    if os.environ.get("NESTED_CV_FOLDS"):
        logger.info(f"Worker run (folds {os.environ['NESTED_CV_FOLDS']}): models cached, results not saved")
        return

    results_path = output_dir / "cv_results.pkl"
    atomic_pickle_dump(cv_results, results_path)
    logger.info(f"Saved: {results_path}")

    config.to_yaml(output_dir / "config.yaml")
    _print_summary(cv_results, config)


def _print_summary(cv_results, config):
    """Print a per-method performance table for a quick console sanity check."""
    import numpy as np

    from mecor.evaluation.metrics import ci_halfwidth

    labels = cv_results.labels
    methods = sorted(set(r.method for r in cv_results.fold_results))
    metrics = ["AUROC", "MCC", "Balanced Acc", "F1 (weighted)"]

    print(f"\n{'='*80}")
    print(f"RESULTS: {config.dataset_name} / {config.model}")
    print(f"{'='*80}")

    for metric in metrics:
        print(f"\n--- {metric} ---")
        header = f"{'Method':<25}"
        for label in labels:
            header += f"{label:>15}"
        print(header)
        print("-" * (25 + 15 * len(labels)))

        for method in methods:
            method_results = [r for r in cv_results.fold_results if r.method == method]
            row = f"{method:<25}"
            for label in labels:
                values = [r.metrics[label][metric] for r in method_results if label in r.metrics]
                if values:
                    mean = np.mean(values)
                    ci = ci_halfwidth(np.std(values, ddof=1), len(values),
                                      k=config.n_splits_outer)
                    row += f"{mean:.3f}±{ci:.3f}".rjust(15)
                else:
                    row += "N/A".rjust(15)
            print(row)

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
