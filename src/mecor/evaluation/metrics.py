"""Classification metrics including clinical AMR metrics."""

from typing import Dict

import numpy as np
from scipy.stats import t as t_dist
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ci_halfwidth(std, n, confidence: float = 0.95, k: int = None) -> float:
    """Calculate confidence interval half-width using t-distribution.

    When k is provided, applies the Nadeau & Bengio (2003) correction
    for repeated k-fold CV: SE = std * sqrt(1/n + 1/(k-1)).
    This accounts for the correlation between fold estimates that share
    training data, producing wider (more conservative) intervals.
    """
    if isinstance(n, (int, float, np.integer)):
        if n <= 1:
            return 0.0
        t_crit = t_dist.ppf((1 + confidence) / 2, n - 1)
        if k is not None and k > 1:
            return t_crit * std * np.sqrt(1.0 / n + 1.0 / (k - 1))
        return t_crit * std / np.sqrt(n)
    result = np.zeros_like(std, dtype=float)
    mask = n > 1
    if np.any(mask):
        t_crit = t_dist.ppf((1 + confidence) / 2, n[mask] - 1)
        if k is not None and k > 1:
            result[mask] = t_crit * std[mask] * np.sqrt(1.0 / n[mask] + 1.0 / (k - 1))
        else:
            result[mask] = t_crit * std[mask] / np.sqrt(n[mask])
    return result


def error_bar_halfwidth(std, n, mode: str = "sd", confidence: float = 0.95, k: int = None):
    """Compute error bar half-width.

    mode='sd': standard deviation (returns std directly, default)
    mode='ci': Nadeau & Bengio corrected 95% CI
    """
    if mode == "sd":
        return std if isinstance(std, (int, float, np.integer)) else std.copy()
    return ci_halfwidth(std, n, confidence=confidence, k=k)


def calc_binary_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Calculate comprehensive binary classification metrics."""
    metrics = {}

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics["Precision (0)"] = precision[0] if len(precision) > 0 else 0.0
    metrics["Precision (1)"] = precision[1] if len(precision) > 1 else 0.0
    metrics["Recall (0)"] = recall[0] if len(recall) > 0 else 0.0
    metrics["Recall (1)"] = recall[1] if len(recall) > 1 else 0.0
    metrics["F1 (0)"] = f1[0] if len(f1) > 0 else 0.0
    metrics["F1 (1)"] = f1[1] if len(f1) > 1 else 0.0
    metrics["F1 (weighted)"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    metrics["Balanced Acc"] = balanced_accuracy_score(y_true, y_pred)

    try:
        metrics["AUROC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["AUROC"] = np.nan

    return metrics


def calc_clinical_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate clinical AMR metrics (VME, ME, sensitivity, specificity).

    VME (Very Major Error): Resistant classified as susceptible (false negative for resistance)
    ME (Major Error): Susceptible classified as resistant (false positive for resistance)

    These use the clinical convention where:
    - y_true/y_pred: 1 = Resistant, 0 = Susceptible
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # VME: among truly resistant, fraction predicted susceptible
    n_resistant = (y_true == 1).sum()
    if n_resistant > 0:
        vme = ((y_true == 1) & (y_pred == 0)).sum() / n_resistant
    else:
        vme = 0.0

    # ME: among truly susceptible, fraction predicted resistant
    n_susceptible = (y_true == 0).sum()
    if n_susceptible > 0:
        me = ((y_true == 0) & (y_pred == 1)).sum() / n_susceptible
    else:
        me = 0.0

    sensitivity = 1.0 - vme

    specificity = 1.0 - me

    return {
        "VME": vme,
        "ME": me,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


def calc_clinical_metrics_at_thresholds(y_true, y_prob, thresholds=None) -> dict:
    """Calculate VME/ME at multiple probability thresholds.

    Returns
    -------
    dict with keys 'thresholds', 'vme', 'me', 'sensitivity', 'specificity'
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    results = {"thresholds": thresholds, "vme": [], "me": [], "sensitivity": [], "specificity": []}

    for t in thresholds:
        y_pred = (np.asarray(y_prob) >= t).astype(int)
        clinical = calc_clinical_metrics(y_true, y_pred)
        results["vme"].append(clinical["VME"])
        results["me"].append(clinical["ME"])
        results["sensitivity"].append(clinical["Sensitivity"])
        results["specificity"].append(clinical["Specificity"])

    for key in ["vme", "me", "sensitivity", "specificity"]:
        results[key] = np.array(results[key])

    return results
