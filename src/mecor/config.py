"""Configuration management for the AMR prediction pipeline."""

import os
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# data roots used in the configs as ${NAME}; override with environment variables
PATH_DEFAULTS = {"SORSOLA_ROOT": "data/s_orsola", "DRIAMS_ROOT": "data"}


def _expand_paths(value):
    """Substitute ${SORSOLA_ROOT} and ${DRIAMS_ROOT} in string config values (environment first, then defaults)."""
    if not isinstance(value, str) or "${" not in value:
        return value
    return string.Template(value).safe_substitute({**PATH_DEFAULTS, **os.environ})


@dataclass
class Config:
    """Configuration for the Classifier Chain AMR prediction pipeline."""

    dataset_name: str = "s_orsola"

    data_path: str = ""          # Path to pre-binned CSV (local)
    spectra_dir: str = ""        # Raw spectra directory (for preprocessing)
    metadata_path: str = ""      # Metadata CSV (for DRIAMS)
    binned_dir: str = ""         # Pre-binned spectra directory (for DRIAMS)

    targets: List[str] = field(default_factory=lambda: [
        "Ciprofloxacina", "Cefotaxime", "Ceftazidime"
    ])
    target_short_names: Dict[str, str] = field(default_factory=lambda: {
        "Ciprofloxacina": "CIP",
        "Ciprofloxacin": "CIP",
        "Cefotaxime": "CTX",
        "Ceftriaxone": "CRO",
        "Ceftazidime": "CAZ",
    })

    # Species filter (for DRIAMS)
    species: str = "Escherichia coli"

    # DRIAMS-specific
    driams_years: List[str] = field(default_factory=lambda: ["2018"])

    output_dir: str = "results"

    model: str = "logistic"

    run_binary_relevance: bool = True
    run_classifier_chain: bool = True

    # Classifier-chain training mode. None (default) = train downstream links on the
    # observed (true) upstream labels (sklearn cv=None). An int K makes sklearn use
    # out-of-fold upstream predictions (cross_val_predict, K folds) at train time:
    # a sensitivity analysis for optimistic bias from teacher-forced labels.
    chain_cv: Optional[int] = None
    # What the upstream links pass to the downstream links: "predict" (hard labels, classical) or
    # "predict_proba" (probabilities, soft chain). Used at training time when chain_cv is set.
    chain_method: str = "predict"

    seed: int = 42
    n_jobs: int = -1
    n_splits_outer: int = 5
    n_repeats_outer: int = 3
    n_splits_inner: int = 3
    # Keep all isolates of a patient on the same side of every split (DRIAMS-A, patient_no of id/<year>/<year>_strat.csv)
    group_by_patient: bool = False

    param_search_mode: str = "full"
    random_search_iterations: int = 20

    save_fitted_models: bool = True
    load_saved_models: bool = True
    model_compression: int = 3

    # Preprocessing (maldiamrkit)
    bin_width: int = 3
    mz_min: int = 2000
    mz_max: int = 20000

    confidence_level: float = 0.95

    dpi: int = 300
    top_features_to_plot: int = 20
    plot_ylimits: Dict[str, List[float]] = field(default_factory=lambda: {
        "AUROC": [0.5, 1.05],
        "MCC": [0.0, 1.05],
        "Balanced Acc": [0.5, 1.05],
        "F1 (weighted)": [0.5, 1.05],
    })

    min_class_frequency: float = 0.05
    max_class_frequency: float = 0.95

    log_level: str = "INFO"

    def __post_init__(self):
        valid_models = ["logistic", "catboost", "tabpfn"]
        if self.model not in valid_models:
            raise ValueError(f"model must be one of {valid_models}. Got '{self.model}'")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if not 0 < self.min_class_frequency < self.max_class_frequency < 1:
            raise ValueError("Invalid class frequency bounds")

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file, supporting inheritance via _base."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        if "_base" in raw:
            base_path = Path(path).parent / raw.pop("_base")
            with open(base_path, "r") as f:
                base = yaml.safe_load(f) or {}
            base.update(raw)
            raw = base

        # Backward compatibility
        if "models" in raw and isinstance(raw["models"], list):
            raw["model"] = raw["models"][0]
            del raw["models"]

        raw = {k: _expand_paths(v) for k, v in raw.items()}
        return cls(**raw)

    def to_yaml(self, path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=False)

    @property
    def labels(self) -> List[str]:
        """Get short names for targets in consistent order."""
        return [self.target_short_names.get(t, t) for t in self.targets]
