"""Unified data loading for local and DRIAMS datasets."""

import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def load_dataset(config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset based on config.dataset_name."""
    if config.dataset_name in ("local", "s_orsola"):
        return load_local_dataset(config)
    elif config.dataset_name.startswith("driams"):
        return load_driams_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")


def build_Y(df: pd.DataFrame, targets: List[str], short_names: Dict[str, str]) -> pd.DataFrame:
    """Build binary target matrix from S/I/R labels.

    Maps: S -> 0, I -> 0, R -> 1.
    Since EUCAST 2019, "I" is redefined as "Susceptible, Increased exposure"
    (not intermediate). Mapping I->0 is standard practice in MALDI-TOF AMR
    prediction (e.g., DRIAMS/AMR-kit default) and consistent with clinical
    decision-making where "I" isolates are treated as susceptible.
    Renames columns to short names (e.g., Ciprofloxacina -> CIP).
    """
    Y = df[targets].copy()
    Y.columns = [short_names.get(col, col) for col in Y.columns]
    Y = Y.dropna()
    label_map = {"S": 0, "I": 0, "R": 1}
    for col in Y.columns:
        Y[col] = Y[col].map(label_map)
    Y = Y.astype(int)
    return Y


def load_local_dataset(config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-binned local CSV dataset.

    Expected CSV format: index column, then ~6000 m/z feature columns,
    then target columns (Ciprofloxacina, Cefotaxime, Ceftazidime) with S/I/R values.
    """
    logger.info(f"Loading local dataset from {config.data_path}")
    df = pd.read_csv(config.data_path, index_col=0)

    Y = build_Y(df, config.targets, config.target_short_names)
    X = df.drop(columns=config.targets).loc[Y.index]

    logger.info(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features")
    return X, Y


def load_driams_dataset(config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load DRIAMS dataset from binned_6000 files + metadata CSV.

    Steps:
    1. Read metadata CSV, filter to target species
    2. Require non-null for all target drugs
    3. Load binned spectra from binned_6000/{code}.txt
    4. Map S/I/R -> binary labels

    Multi-year support: when config.driams_years has >1 entry,
    metadata_path and binned_dir are treated as base directories.
    Per-year paths are constructed as {base}/{year}/{year}_clean.csv
    and {base}/{year}/ respectively.
    """
    logger.info(f"Loading DRIAMS dataset: {config.dataset_name}")

    if len(config.driams_years) > 1:
        all_X, all_Y = [], []
        for year in config.driams_years:
            meta_path = Path(config.metadata_path) / year / f"{year}_clean.csv"
            bin_dir = Path(config.binned_dir) / year
            logger.info(f"  Loading year {year} from {meta_path}")
            meta = pd.read_csv(str(meta_path), low_memory=False)
            X_y, Y_y = _load_driams_from_meta(meta, str(bin_dir), config)
            all_X.append(X_y)
            all_Y.append(Y_y)
        X = pd.concat(all_X)
        Y = pd.concat(all_Y)
        logger.info(f"  Combined {len(config.driams_years)} years: {X.shape[0]} samples total")
        return X, Y

    meta = pd.read_csv(config.metadata_path, low_memory=False)
    return _load_driams_from_meta(meta, config.binned_dir, config)


def _load_driams_from_meta(
    meta: pd.DataFrame,
    binned_dir: str,
    config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Core DRIAMS loading: filter metadata, load binned spectra, build labels."""
    ecoli = meta[meta["species"] == config.species].copy()
    logger.info(f"  {config.species}: {len(ecoli)} samples")

    for drug in config.targets:
        if drug not in ecoli.columns:
            raise ValueError(
                f"Drug '{drug}' not found in metadata columns: {list(ecoli.columns)}"
            )
        ecoli = ecoli[ecoli[drug].notna()]

    logger.info(f"  With all target drugs: {len(ecoli)} samples")

    X_rows = []
    valid_codes = []

    for _, row in ecoli.iterrows():
        code = str(row["code"])
        fpath = os.path.join(binned_dir, f"{code}.txt")

        if not os.path.exists(fpath):
            continue

        try:
            spectrum = pd.read_csv(fpath, sep=" ")
            X_rows.append(spectrum["binned_intensity"].values)
            valid_codes.append(code)
        except Exception:
            continue

    if len(X_rows) == 0:
        raise ValueError(f"No valid spectra found in {binned_dir}")

    # Build feature matrix with m/z bin column names matching local dataset
    X = pd.DataFrame(np.array(X_rows), index=valid_codes)
    X.columns = [str(config.mz_min + i * config.bin_width) for i in range(X.shape[1])]

    Y_data = ecoli.set_index("code").loc[valid_codes, config.targets].copy()
    Y_data.columns = [config.target_short_names.get(col, col) for col in Y_data.columns]
    # Normalize case before mapping (DRIAMS may have lowercase s/i/r)
    label_map = {"S": 0, "I": 0, "R": 1}
    for col in Y_data.columns:
        Y_data[col] = Y_data[col].astype(str).str.strip().str.upper().map(label_map)
    # Drop rows with unmappable values (not S/I/R)
    n_before = len(Y_data)
    Y_data = Y_data.dropna()
    X = X.loc[Y_data.index]
    if n_before > len(Y_data):
        logger.warning(f"  Dropped {n_before - len(Y_data)} samples with unmappable labels")
    Y_data = Y_data.astype(int)

    logger.info(f"  Loaded {X.shape[0]} spectra with {X.shape[1]} bins")

    for label in Y_data.columns:
        n_r = int(Y_data[label].sum())
        pct = Y_data[label].mean() * 100
        logger.info(f"    {label}: {n_r} resistant ({pct:.1f}%)")

    return X, Y_data
