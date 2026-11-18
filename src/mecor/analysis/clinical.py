"""Clinical evaluation: VME/ME analysis at different thresholds."""

from typing import Dict, List

import numpy as np

from mecor.evaluation.metrics import calc_clinical_metrics, calc_clinical_metrics_at_thresholds


def clinical_evaluation(
    fold_results: list,
    labels: List[str],
    method: str = "ensemble",
) -> Dict:
    """Aggregate clinical metrics (VME/ME) across CV folds for a method.

    Parameters
    ----------
    fold_results : list of FoldResult
        Results from nested CV
    labels : list of str
        Drug short names
    method : str
        Method name to filter (e.g., 'ensemble', 'binary_relevance')

    Returns
    -------
    dict mapping label -> {
        'vme_mean', 'vme_std', 'me_mean', 'me_std',
        'sensitivity_mean', 'specificity_mean',
        'threshold_analysis': dict from calc_clinical_metrics_at_thresholds
    }
    """
    method_results = [r for r in fold_results if r.method == method]

    summary = {}
    for label in labels:
        vme_vals = []
        me_vals = []
        sens_vals = []
        spec_vals = []
        all_y_true = []
        all_y_prob = []

        for r in method_results:
            if label in r.clinical_metrics:
                cm = r.clinical_metrics[label]
                vme_vals.append(cm["VME"])
                me_vals.append(cm["ME"])
                sens_vals.append(cm["Sensitivity"])
                spec_vals.append(cm["Specificity"])
            if label in r.y_true and label in r.y_prob:
                all_y_true.extend(r.y_true[label].tolist())
                all_y_prob.extend(r.y_prob[label].tolist())

        summary[label] = {
            "vme_mean": float(np.mean(vme_vals)) if vme_vals else np.nan,
            "vme_std": float(np.std(vme_vals, ddof=1)) if len(vme_vals) > 1 else 0.0,
            "me_mean": float(np.mean(me_vals)) if me_vals else np.nan,
            "me_std": float(np.std(me_vals, ddof=1)) if len(me_vals) > 1 else 0.0,
            "sensitivity_mean": float(np.mean(sens_vals)) if sens_vals else np.nan,
            "specificity_mean": float(np.mean(spec_vals)) if spec_vals else np.nan,
            "n_folds": len(vme_vals),
        }

        if all_y_true and all_y_prob:
            summary[label]["threshold_analysis"] = calc_clinical_metrics_at_thresholds(
                np.array(all_y_true), np.array(all_y_prob)
            )

    return summary


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_vme: float = 0.015,
    max_me: float = 0.03,
) -> Dict:
    """Find threshold minimizing VME while keeping ME below limit.

    Parameters
    ----------
    y_true : array
        True binary labels
    y_prob : array
        Predicted probabilities
    max_vme : float
        Maximum acceptable VME
    max_me : float
        Maximum acceptable ME

    Returns
    -------
    dict with 'threshold', 'vme', 'me', 'sensitivity', 'specificity'
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best = {"threshold": 0.5, "vme": 1.0, "me": 1.0}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = calc_clinical_metrics(y_true, y_pred)

        if cm["VME"] <= max_vme and cm["ME"] <= max_me:
            if cm["VME"] < best["vme"]:
                best = {"threshold": float(t), **cm}

    # If no threshold satisfies both constraints, minimize VME
    if best["vme"] == 1.0:
        best_vme = 1.0
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            cm = calc_clinical_metrics(y_true, y_pred)
            if cm["VME"] < best_vme:
                best_vme = cm["VME"]
                best = {"threshold": float(t), **cm}

    return best
