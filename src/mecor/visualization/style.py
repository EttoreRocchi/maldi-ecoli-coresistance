"""Publication-quality visualization style (legacy API kept; theme and palette now come from paper_style)."""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

from . import paper_style as ps

MSYSTEMS_STYLE = dict(ps.RC)

PALETTE = {
    "binary_relevance": ps.ROLE["br"],
    "chain_ordering": ps.LIGHT,
    "chain_ensemble": ps.ROLE["ensemble"],
    "best_chain": ps.ROLE["best"],
    "chain_prediction": ps.ROLE["oracle"],
    "original_feature": ps.ROLE["best"],
    "dataset_local":    ps.SITE["s_orsola"],
    "dataset_s_orsola": ps.SITE["s_orsola"],
    "dataset_driams_a": ps.SITE["driams_a"],
    "dataset_driams_b": ps.SITE["driams_b"],
    "dataset_driams_c": ps.SITE["driams_c"],
    "dataset_driams_d": ps.SITE["driams_d"],
}

DATASET_DISPLAY_NAMES = {
    "s_orsola": "Sant\u2019Orsola",
    "local":    "Sant\u2019Orsola",   # backward compat (pkl files store "local")
    "driams_a": "DRIAMS A",
    "driams_b": "DRIAMS B",
    "driams_c": "DRIAMS C",
    "driams_d": "DRIAMS D",
}


def get_display_name(name: str) -> str:
    """Return the human-readable display name for a dataset identifier."""
    return DATASET_DISPLAY_NAMES.get(name, name)

DRUG_COLORS = dict(ps.DRUG)


def setup_style():
    """Apply the shared paper style globally."""
    ps.apply()


def save_fig(fig, path, dpi: int = 300):
    """Save figure in both PNG and PDF formats (shared style)."""
    ps.save(fig, Path(path))
