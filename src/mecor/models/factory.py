"""Model factory and hyperparameter grids."""

from sklearn.linear_model import LogisticRegression

# NOTE: catboost and tabpfn are imported lazily inside get_base_estimator so that
# logistic-regression-only runs (and post-processing) do not require the heavy
# gradient-boosting / GPU stack to be installed.


def get_base_estimator(estimator_type: str, random_state: int):
    """Create base estimator instance.

    Parameters
    ----------
    estimator_type : str
        One of 'logistic', 'catboost', 'tabpfn'.
    random_state : int
        Random seed for reproducibility.
    """
    if estimator_type == "logistic":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
            n_jobs=1,
        )
    elif estimator_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            subsample=0.8,
            l2_leaf_reg=5,
            od_type="Iter",
            od_wait=50,
            auto_class_weights="Balanced",
            verbose=0,
            allow_writing_files=False,
            thread_count=1,
            random_state=random_state,
        )
    elif estimator_type == "tabpfn":
        import os

        from tabpfn import TabPFNClassifier
        kwargs = {}
        # Pin the checkpoint used for the manuscript (TabPFN v2.5 default classifier) so that
        # reruns do not silently pick up a newer default model. Set TABPFN_MODEL_PATH to the
        # local .ckpt file; when unset, the package default is used.
        model_path = os.environ.get("TABPFN_MODEL_PATH")
        if model_path:
            kwargs["model_path"] = model_path
        return TabPFNClassifier(
            n_estimators=8,
            balance_probabilities=True,
            random_state=random_state,
            device="auto",
            ignore_pretraining_limits=True,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown estimator_type: {estimator_type}")


def needs_scaling(estimator_type: str) -> bool:
    """Determine if model requires feature scaling."""
    return estimator_type in ("logistic",)


def get_model_display_name(model_type: str) -> str:
    """Get human-readable display name for model type."""
    names = {
        "logistic": "Logistic Regression",
        "catboost": "CatBoost",
        "tabpfn": "TabPFN",
    }
    return names.get(model_type, model_type)


def get_param_grid(estimator_type: str, mode: str = "full"):
    """Get hyperparameter grid for model tuning.

    Parameters
    ----------
    estimator_type : str
        Model type identifier
    mode : str
        'quick' for fast exploration, 'full' for comprehensive search
    """
    if estimator_type == "logistic":
        if mode == "quick":
            return {"clf__C": [0.1, 1.0, 10.0], "clf__penalty": ["l2"]}
        if mode == "reduced":
            # L2-only, liblinear grid (sensitivity only: with this grid the chain cannot lean on the upstream label)
            return {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l2"], "clf__solver": ["liblinear"]}
        if mode == "liblinear":
            # L1 and L2 with the fast liblinear solver (8 combinations); used for the secondary analyses
            return {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l1", "l2"], "clf__solver": ["liblinear"]}
        if mode == "liblinear_full":
            # the main-analysis grid without the saga solver (same C values and penalties; saga took 98% of the fit time)
            return {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], "clf__penalty": ["l1", "l2"], "clf__solver": ["liblinear"]}
        return {
            "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
        }

    elif estimator_type == "catboost":
        if mode == "quick":
            return {
                "clf__learning_rate": [0.1],
                "clf__depth": [4],
                "clf__rsm": [0.5],
            }
        return {
            "clf__learning_rate": [0.03, 0.1],
            "clf__depth": [4, 6],
            "clf__rsm": [0.3, 0.5],
        }

    elif estimator_type == "tabpfn":
        # TabPFN is a pre-trained model, no hyperparameter tuning needed.
        # Feature selection k is fixed in the pipeline step.
        return {}

    else:
        raise ValueError(f"Unknown estimator: {estimator_type}")
