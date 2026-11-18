"""Classifier Chain wrapper for multi-label AMR prediction."""

import logging
from typing import Dict, List

import numpy as np
from sklearn.multioutput import ClassifierChain

logger = logging.getLogger(__name__)


class ClassifierChainWrapper:
    """Wrapper around sklearn's ClassifierChain for multi-label AMR prediction.

    Trains classifiers sequentially where each model sees predictions from
    previous models in the chain, enabling learning of label co-occurrence.
    """

    def __init__(self, base_estimator, chain_order: List[str], seed: int = 42, cv=None, chain_method: str = "predict"):
        """
        Parameters
        ----------
        base_estimator : estimator object
            Base estimator (typically a Pipeline with GridSearchCV)
        chain_order : List[str]
            Order of labels in the chain (e.g., ['CTX', 'CAZ', 'CIP'])
        seed : int
            Random state for reproducibility
        cv : int or None
            Passed to sklearn ``ClassifierChain``. None (default) trains each
            downstream link on the true upstream labels; an int K uses out-of-fold
            upstream predictions at train time (sensitivity analysis).
        """
        self.base_estimator = base_estimator
        self.chain_order = chain_order
        self.seed = seed
        self.cv = cv
        # "predict": hard 0/1 upstream labels at prediction time (classical chain, manuscript main analysis);
        # "predict_proba": upstream predicted probabilities are passed instead (soft chain). With cv set, sklearn
        # uses the same method for the out-of-fold upstream inputs at training time.
        self.chain_method = chain_method
        self.chain = None

    def __setstate__(self, state):
        # models pickled before chain_method existed used hard upstream labels, the default
        state.setdefault("chain_method", "predict")
        self.__dict__.update(state)

    def fit(self, X, Y):
        """Fit the classifier chain.

        Parameters
        ----------
        X : array-like
            Feature matrix
        Y : pd.DataFrame
            Multi-label target matrix
        """
        Y_ordered = Y[self.chain_order].values
        order = list(range(len(self.chain_order)))

        try:
            self.chain = ClassifierChain(
                estimator=self.base_estimator,
                order=order,
                cv=self.cv,
                chain_method=self.chain_method,
                random_state=self.seed,
            )
        except TypeError:
            # sklearn <1.4 uses base_estimator instead of estimator
            self.chain = ClassifierChain(
                base_estimator=self.base_estimator,
                order=order,
                cv=self.cv,
                random_state=self.seed,
            )
        self.chain.fit(X, Y_ordered)
        return self

    def predict_proba(self, X) -> Dict[str, np.ndarray]:
        """Predict probabilities for each label in the chain.

        Manually iterates through chain estimators, augmenting features
        with hard binary predictions from previous labels. This matches
        sklearn's ClassifierChain behavior (chain_method_="predict"),
        which is consistent with training on true binary labels.

        Returns
        -------
        dict
            Maps label names to probability arrays
        """
        if self.chain is None:
            raise ValueError("Chain must be fitted before prediction")

        predictions = {}
        X_augmented = np.asarray(X)

        for idx, label in enumerate(self.chain_order):
            estimator = self.chain.estimators_[idx]

            if hasattr(estimator, "predict_proba"):
                y_prob = estimator.predict_proba(X_augmented)[:, 1]
            elif hasattr(estimator, "decision_function"):
                z = estimator.decision_function(X_augmented)
                y_prob = 1 / (1 + np.exp(-z))
            else:
                y_prob = estimator.predict(X_augmented).astype(float)

            predictions[label] = y_prob

            if idx < len(self.chain_order) - 1:
                if self.chain_method == "predict_proba":
                    y_pass = y_prob.reshape(-1, 1)
                else:
                    y_pass = (y_prob >= 0.5).astype(int).reshape(-1, 1)
                X_augmented = np.column_stack([X_augmented, y_pass])

        return predictions

    def predict(self, X) -> Dict[str, np.ndarray]:
        """Predict binary labels using sklearn's predict.

        Returns
        -------
        dict
            Maps label names to binary prediction arrays
        """
        if self.chain is None:
            raise ValueError("Chain must be fitted before prediction")

        y_pred_array = self.chain.predict(X)
        return {
            label: y_pred_array[:, idx]
            for idx, label in enumerate(self.chain_order)
        }

    def get_augmented_feature_names(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Get feature names for each position in the chain.

        Returns
        -------
        dict
            Maps label name to list of feature names (original + prediction_* features)
        """
        augmented = {}
        for idx, label in enumerate(self.chain_order):
            names = feature_names.copy()
            for prev_idx in range(idx):
                names.append(f"prediction_{self.chain_order[prev_idx]}")
            augmented[label] = names
        return augmented
