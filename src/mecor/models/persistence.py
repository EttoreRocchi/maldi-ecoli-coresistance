"""Model persistence utilities."""

import logging
from pathlib import Path
from typing import List, Optional

from joblib import dump, load

logger = logging.getLogger(__name__)


def get_model_path(
    model_dir: Path,
    method: str,
    repeat_idx: int,
    fold_idx: int,
    model_type: str,
    chain_order: Optional[List[str]] = None,
) -> Path:
    """Get file path for a saved model.

    Parameters
    ----------
    model_dir : Path
        Directory for saved models
    method : str
        'chain' or 'binary_relevance'
    repeat_idx, fold_idx : int
        CV indices
    model_type : str
        Model type name
    chain_order : list, optional
        Chain ordering (only for chain method)
    """
    if method == "chain" and chain_order:
        order_str = "_".join(chain_order)
        filename = f"chain_r{repeat_idx:02d}_f{fold_idx:02d}_{model_type}_{order_str}.joblib"
    else:
        filename = f"br_r{repeat_idx:02d}_f{fold_idx:02d}_{model_type}.joblib"
    return model_dir / filename


def save_model(
    model,
    model_dir: Path,
    method: str,
    repeat_idx: int,
    fold_idx: int,
    model_type: str,
    chain_order: Optional[List[str]] = None,
    compression: int = 3,
) -> Optional[Path]:
    """Save fitted model to disk."""
    model_path = get_model_path(
        model_dir, method, repeat_idx, fold_idx, model_type, chain_order
    )
    try:
        # atomic: an interrupted save never leaves a truncated model file behind
        from mecor.evaluation.checkpoint import atomic_joblib_dump
        atomic_joblib_dump(model, model_path, compress=compression)
        logger.debug(f"Saved: {model_path.name}")
        return model_path
    except Exception as e:
        logger.warning(f"Failed to save {model_path.name}: {e}")
        return None


def load_model(
    model_dir: Path,
    method: str,
    repeat_idx: int,
    fold_idx: int,
    model_type: str,
    chain_order: Optional[List[str]] = None,
) -> Optional[object]:
    """Load fitted model from disk."""
    model_path = get_model_path(
        model_dir, method, repeat_idx, fold_idx, model_type, chain_order
    )
    if not model_path.exists():
        return None
    # Remove corrupt/empty files from interrupted saves
    if model_path.stat().st_size == 0:
        model_path.unlink()
        logger.info(f"Removed empty model file: {model_path.name}")
        return None
    try:
        model = load(model_path)
        logger.debug(f"Loaded: {model_path.name}")
        return model
    except Exception:
        logger.warning(f"Corrupt model file, removing: {model_path.name}")
        model_path.unlink(missing_ok=True)
        return None


def _cross_model_path(
    model_dir: Path,
    train_name: str,
    method: str,
    model_type: str,
    chain_order: Optional[List[str]] = None,
) -> Path:
    """Get file path for a cross-dataset model.

    Naming: cross_{train}_{model}_{method}[_{order}].joblib
    Models are keyed by train_name only (not test), since features are
    identical across all datasets (100% m/z bin overlap).
    """
    if method == "chain" and chain_order:
        order_str = "_".join(chain_order)
        filename = f"cross_{train_name}_{model_type}_chain_{order_str}.joblib"
    else:
        filename = f"cross_{train_name}_{model_type}_br.joblib"
    return model_dir / filename


def save_cross_model(
    model,
    model_dir: Path,
    train_name: str,
    method: str,
    model_type: str,
    chain_order: Optional[List[str]] = None,
    compression: int = 3,
) -> Optional[Path]:
    """Save a cross-dataset fitted model to disk."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = _cross_model_path(
        model_dir, train_name, method, model_type, chain_order
    )
    try:
        dump(model, model_path, compress=compression)
        logger.debug(f"Saved cross-dataset model: {model_path.name}")
        return model_path
    except Exception as e:
        logger.warning(f"Failed to save {model_path.name}: {e}")
        return None


def load_cross_model(
    model_dir: Path,
    train_name: str,
    method: str,
    model_type: str,
    chain_order: Optional[List[str]] = None,
) -> Optional[object]:
    """Load a cross-dataset fitted model from disk."""
    model_path = _cross_model_path(
        model_dir, train_name, method, model_type, chain_order
    )
    if not model_path.exists():
        return None
    if model_path.stat().st_size == 0:
        model_path.unlink()
        logger.info(f"Removed empty model file: {model_path.name}")
        return None
    try:
        model = load(model_path)
        logger.debug(f"Loaded cross-dataset model: {model_path.name}")
        return model
    except Exception:
        logger.warning(f"Corrupt model file, removing: {model_path.name}")
        model_path.unlink(missing_ok=True)
        return None
