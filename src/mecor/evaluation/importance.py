"""Feature importance extraction and chain gain analysis."""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _extract_single_importance(clf, estimator_type: str, n_features: int) -> np.ndarray:
    """Extract feature importance from a single estimator.

    Extraction strategy per model type:
    - logistic: |coef_| (absolute value of coefficients; loses directionality
      but allows ranking features by importance magnitude)
    - catboost: native feature_importances_ (prediction value change)
    - tabpfn: returns zeros (TabPFN is a foundation model with no interpretable
      per-feature importance; use SHAP or mutual information scores instead)
    """
    if estimator_type == "logistic":
        if hasattr(clf, "coef_"):
            return np.abs(clf.coef_[0])
    elif estimator_type == "catboost":
        if hasattr(clf, "feature_importances_"):
            return clf.feature_importances_
    # tabpfn and unknown: no native importance
    return np.zeros(n_features)


def extract_feature_importances(
    classifier,
    feature_names: List[str],
    estimator_type: str,
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Extract feature importances from a trained classifier.

    Works for both ClassifierChainWrapper and BinaryRelevanceWrapper.

    For chains: feature names include prediction_* entries for previous labels.
    For binary relevance: feature names are only the original features.

    When a MutualInfoSelector is in the pipeline (TabPFN), importances are
    mapped back to the original feature space with zeros for unselected features.

    Returns
    -------
    dict
        Maps label name to (importance_array, feature_names) tuple
    """
    from mecor.models.binary_relevance import BinaryRelevanceWrapper
    from mecor.models.chain import ClassifierChainWrapper

    importances = {}

    if isinstance(classifier, BinaryRelevanceWrapper):
        for label in classifier.labels:
            est = classifier.estimators_[label]
            if hasattr(est, "best_estimator_"):
                best = est.best_estimator_
            else:
                best = est
            if hasattr(best, "named_steps"):
                clf = best.named_steps["clf"]
                selector = best.named_steps.get("selector", None)
            else:
                clf = best
                selector = None

            n_out = len(feature_names)
            if selector is not None and hasattr(selector, "top_indices_"):
                # Map importance back to original feature space
                raw_imp = _extract_single_importance(clf, estimator_type, len(selector.top_indices_))
                imp = np.zeros(n_out)
                imp[selector.top_indices_] = raw_imp
            else:
                imp = _extract_single_importance(clf, estimator_type, n_out)

            current_names = feature_names.copy()
            if len(imp) != len(current_names):
                if len(imp) > len(current_names):
                    current_names.extend([f"Feature_{i}" for i in range(len(current_names), len(imp))])
                else:
                    current_names = current_names[:len(imp)]

            importances[label] = (imp, current_names)

    elif isinstance(classifier, ClassifierChainWrapper):
        for idx, label in enumerate(classifier.chain_order):
            estimator = classifier.chain.estimators_[idx]

            if hasattr(estimator, "best_estimator_"):
                best = estimator.best_estimator_
            else:
                best = estimator
            if hasattr(best, "named_steps"):
                clf = best.named_steps["clf"]
                selector = best.named_steps.get("selector", None)
            else:
                clf = best
                selector = None

            # Build feature names including previous predictions
            current_names = feature_names.copy()
            for prev_idx in range(idx):
                current_names.append(f"prediction_{classifier.chain_order[prev_idx]}")

            n_out = len(current_names)
            if selector is not None and hasattr(selector, "top_indices_"):
                raw_imp = _extract_single_importance(clf, estimator_type, len(selector.top_indices_))
                imp = np.zeros(n_out)
                imp[selector.top_indices_] = raw_imp
            else:
                imp = _extract_single_importance(clf, estimator_type, n_out)

            if len(imp) != len(current_names):
                if len(imp) > len(current_names):
                    current_names.extend([f"Feature_{i}" for i in range(len(current_names), len(imp))])
                else:
                    current_names = current_names[:len(imp)]

            importances[label] = (imp, current_names)

    return importances


def compute_chain_gain(
    importance_store: dict,
    labels: List[str],
) -> dict:
    """Quantify the contribution of chain predictions to each label.

    For each label (except the first in each ordering), computes:
    - Absolute importance of prediction_* features
    - Relative importance (% of total from chain predictions)

    Parameters
    ----------
    importance_store : dict
        Nested dict: importance_store[ordering_key][label] = list of (imp, names) tuples
    labels : list
        Label names

    Returns
    -------
    dict
        Mapping label -> {'abs_importance': float, 'rel_importance': float, 'n_orderings': int}
    """
    chain_gains = {label: [] for label in labels}

    for order_key, label_data in importance_store.items():
        for label in labels:
            if label not in label_data:
                continue

            for imp_array, feat_names in label_data[label]:
                pred_mask = np.array(["prediction_" in n for n in feat_names])
                if not pred_mask.any():
                    continue

                total_imp = np.sum(np.abs(imp_array))
                chain_imp = np.sum(np.abs(imp_array[pred_mask]))

                if total_imp > 0:
                    chain_gains[label].append({
                        "abs": chain_imp,
                        "rel": chain_imp / total_imp,
                        "ordering": order_key,
                    })

    summary = {}
    for label in labels:
        if chain_gains[label]:
            abs_vals = [g["abs"] for g in chain_gains[label]]
            rel_vals = [g["rel"] for g in chain_gains[label]]
            summary[label] = {
                "mean_abs_importance": float(np.mean(abs_vals)),
                "mean_rel_importance": float(np.mean(rel_vals)),
                "std_rel_importance": float(np.std(rel_vals, ddof=1)) if len(rel_vals) > 1 else 0.0,
                "n_observations": len(chain_gains[label]),
            }
        else:
            summary[label] = {
                "mean_abs_importance": 0.0,
                "mean_rel_importance": 0.0,
                "std_rel_importance": 0.0,
                "n_observations": 0,
            }

    return summary
