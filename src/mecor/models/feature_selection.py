"""Feature selection transformers for MALDI-TOF AMR prediction.

Provides sklearn-compatible transformers for dimensionality reduction,
primarily designed for models with feature count limits (e.g., TabPFN ≤ 500).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """Select top-k features by mutual information with the target.

    Designed for TabPFN which has a 500 feature pre-training limit.
    Fits on training data only (no data leakage when used in a Pipeline).

    Since this is used per-label inside BR/Chain wrappers (where y is a single
    column), mutual information is computed per-label. Different labels may
    select different features, this is scientifically correct as CIP resistance
    may depend on different spectral peaks than CTX resistance.

    Parameters
    ----------
    k : int, default=500
        Number of top features to select.
    random_state : int, default=42
        Random state for mutual_info_classif estimation.
    """

    def __init__(self, k=500, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        """Compute mutual information scores and select top-k features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels.

        Returns
        -------
        self
        """
        n_features = X.shape[1]
        k = min(self.k, n_features)

        mi = mutual_info_classif(X, y, random_state=self.random_state)
        self.top_indices_ = np.argsort(mi)[-k:]
        self.top_indices_.sort()  # Preserve original column order
        self.mi_scores_ = mi
        return self

    def transform(self, X):
        """Select the top-k features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        X_selected : array-like of shape (n_samples, k)
            Reduced feature matrix.
        """
        if hasattr(X, "iloc"):
            return X.iloc[:, self.top_indices_]
        return X[:, self.top_indices_]

    def get_feature_names_out(self, input_features=None):
        """Get feature names for the selected features.

        Parameters
        ----------
        input_features : array-like of str, optional
            Input feature names. If None, generates default names.

        Returns
        -------
        feature_names : ndarray of str
            Selected feature names.
        """
        if input_features is not None:
            return np.array(input_features)[self.top_indices_]
        return np.array([f"f{i}" for i in self.top_indices_])
