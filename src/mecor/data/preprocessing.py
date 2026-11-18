"""MALDI-TOF spectrum preprocessing using maldiamrkit v0.7.0."""

import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def preprocess_spectra(
    spectra_dir: str,
    metadata_path: str,
    antibiotics: list,
    output_path: str,
    bin_width: int = 3,
    mz_min: int = 2000,
    mz_max: int = 20000,
) -> pd.DataFrame:
    """Preprocess raw MALDI-TOF spectra using maldiamrkit v0.7.0.

    Applies the default preprocessing pipeline:
    1. Clip negatives to zero
    2. Square root variance stabilization
    3. Savitzky-Golay smoothing
    4. SNIP baseline correction
    5. m/z trimming to [mz_min, mz_max]
    6. TIC normalization

    Then bins spectra to fixed m/z grid.

    Parameters
    ----------
    spectra_dir : str
        Directory containing raw spectrum .txt files
    metadata_path : str
        Path to metadata CSV with antibiotic resistance labels
    antibiotics : list
        List of antibiotic column names to include
    output_path : str
        Path to save the processed CSV
    bin_width : int
        Width of m/z bins in Daltons
    mz_min, mz_max : int
        m/z range to keep

    Returns
    -------
    pd.DataFrame
        Combined feature + label dataframe
    """
    from maldiamrkit.dataset import MaldiSet

    logger.info(f"Preprocessing spectra from {spectra_dir}")
    logger.info(f"  Antibiotics: {antibiotics}")
    logger.info(f"  Binning: {mz_min}-{mz_max} Da, width={bin_width}")

    data = MaldiSet.from_directory(
        spectra_dir,
        metadata_path,
        aggregate_by=dict(antibiotics=antibiotics),
        verbose=False,
    )

    X, y = data.X, data.y

    logger.info(f"  Processed: {X.shape[0]} spectra, {X.shape[1]} features")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined = pd.concat([X, y], axis=1)
    combined.to_csv(output_path, index=True)

    logger.info(f"  Saved to {output_path}")

    return combined


def generate_pseudogels(
    spectra_dir: str,
    metadata_path: str,
    antibiotics: list,
    output_dir: str,
    dpi: int = 300,
) -> None:
    """Generate pseudogel visualizations for each antibiotic."""
    import matplotlib.pyplot as plt
    from maldiamrkit.dataset import MaldiSet

    data = MaldiSet.from_directory(
        spectra_dir,
        metadata_path,
        aggregate_by=dict(antibiotics=antibiotics),
        verbose=False,
    )

    os.makedirs(output_dir, exist_ok=True)

    for antibiotic in antibiotics:
        try:
            fig, axes = data.plot_pseudogel(antibiotic=antibiotic, show=False)
            plt.savefig(
                os.path.join(output_dir, f"pseudogel_{antibiotic}.png"),
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)
            logger.info(f"  Saved pseudogel for {antibiotic}")
        except Exception as e:
            logger.warning(f"  Failed to generate pseudogel for {antibiotic}: {e}")
