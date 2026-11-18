"""Binary Relevance baseline: independent classifiers per antibiotic."""

import logging
from typing import Dict, List

import numpy as np
from sklearn.base import clone

logger = logging.getLogger(__name__)


class BinaryRelevanceWrapper:
    """Independent classifiers per antibiotic (no label dependencies).

    Same interface as ClassifierChainWrapper for fair comparison.
    Each label sees only the original features X (no prediction_* columns).
    """

    def __init__(self, base_estimator, labels: List[str], seed: int = 42):
        """
        Parameters
        ----------
        base_estimator : estimator object
            Base estimator (typically a Pipeline with GridSearchCV)
        labels : List[str]
            Label names (e.g., ['CIP', 'CTX', 'CAZ'])
        seed : int
            Random state for reproducibility
        """
        self.base_estimator = base_estimator
        self.labels = labels
        self.seed = seed
        self.estimators_ = {}

    def fit(self, X, Y):
        """Fit independent classifier per label.

        Parameters
        ----------
        X : array-like
            Feature matrix
        Y : pd.DataFrame
            Multi-label target matrix
        """
        for label in self.labels:
            logger.debug(f"  Training BR for {label}...")
            est = clone(self.base_estimator)
            est.fit(X, Y[label].values)
            self.estimators_[label] = est
        return self

    def predict_proba(self, X) -> Dict[str, np.ndarray]:
        """Predict probabilities independently for each label.

        Returns
        -------
        dict
            Maps label names to probability arrays
        """
        predictions = {}
        for label in self.labels:
            est = self.estimators_[label]
            if hasattr(est, "predict_proba"):
                predictions[label] = est.predict_proba(X)[:, 1]
            elif hasattr(est, "decision_function"):
                z = est.decision_function(X)
                predictions[label] = 1 / (1 + np.exp(-z))
            else:
                predictions[label] = est.predict(X).astype(float)
        return predictions

    def predict(self, X) -> Dict[str, np.ndarray]:
        """Predict binary labels independently for each label.

        Returns
        -------
        dict
            Maps label names to binary prediction arrays
        """
        probs = self.predict_proba(X)
        return {label: (probs[label] >= 0.5).astype(int) for label in self.labels}
