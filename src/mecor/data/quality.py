"""Data quality assessment and validation."""

import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)


def validate_data(X: pd.DataFrame, Y: pd.DataFrame, config) -> None:
    """Validate feature matrix and labels for common issues."""
    logger.info("DATA VALIDATION")

    if X.isnull().any().any():
        n_missing = X.isnull().sum().sum()
        raise ValueError(f"Features contain {n_missing} missing values")
    logger.info("  No missing values in features")

    if np.isinf(X.values).any():
        raise ValueError("Features contain infinite values")
    logger.info("  No infinite values in features")

    for label in Y.columns:
        pos_rate = Y[label].mean()
        if pos_rate < config.min_class_frequency or pos_rate > config.max_class_frequency:
            logger.warning(
                f"  {label}: Severe class imbalance ({pos_rate:.1%} positive)"
            )
        else:
            logger.info(f"  {label}: {pos_rate:.1%} positive - OK")


def quality_report(X: pd.DataFrame, Y: pd.DataFrame) -> Dict:
    """Generate dataset quality summary for the Methods section.

    Returns
    -------
    dict
        Summary statistics including sample count, feature count,
        resistance rates, class balance, and feature distributions.
    """
    report = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "labels": list(Y.columns),
        "resistance_rates": {},
        "feature_stats": {},
    }

    for label in Y.columns:
        n_r = int(Y[label].sum())
        n_s = int((Y[label] == 0).sum())
        report["resistance_rates"][label] = {
            "resistant": n_r,
            "susceptible": n_s,
            "rate": float(Y[label].mean()),
        }

    report["feature_stats"] = {
        "mean_intensity": float(X.values.mean()),
        "std_intensity": float(X.values.std()),
        "min_intensity": float(X.values.min()),
        "max_intensity": float(X.values.max()),
        "zero_fraction": float((X.values == 0).mean()),
    }

    return report
