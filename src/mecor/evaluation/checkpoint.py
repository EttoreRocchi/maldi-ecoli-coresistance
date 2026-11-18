"""Crash-safe checkpoints: atomic writes, tolerant reads and run signatures.

Every write goes to a temporary file in the same folder, is flushed to disk and then renamed over the
target, so an interrupted job leaves either the old file or the complete new one, never a truncated file.
A checkpoint folder records a signature of what it was computed with (folds, grouping, grid); resuming
with different settings stops the job instead of silently reusing incompatible checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)
SIGNATURE_FILE = "SIGNATURE.json"


def _atomic_write(path, write: Callable) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def atomic_json_dump(obj: Any, path, **kwargs) -> Path:
    return _atomic_write(path, lambda f: f.write(json.dumps(obj, **kwargs).encode("utf8")))


def atomic_pickle_dump(obj: Any, path) -> Path:
    return _atomic_write(path, lambda f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL))


def atomic_joblib_dump(obj: Any, path, compress: int = 3) -> Path:
    from joblib import dump
    return _atomic_write(path, lambda f: dump(obj, f, compress=compress))


def atomic_savez(path, **arrays) -> Path:
    return _atomic_write(path, lambda f: np.savez_compressed(f, **arrays))


def load_json_or_none(path) -> Optional[Any]:
    """Return the JSON content, or None if the file is missing or unreadable (an unreadable file is removed)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(f"Unreadable checkpoint removed: {path}")
        path.unlink(missing_ok=True)
        return None


def npz_ok(path) -> bool:
    """True if the .npz file exists and can be read (an unreadable file is removed)."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            for k in z.files:
                z[k]
        return True
    except Exception:
        logger.warning(f"Unreadable checkpoint removed: {path}")
        path.unlink(missing_ok=True)
        return False


def digest(*arrays) -> str:
    """Short hash of index arrays (for example the test indices of every fold, in order)."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes())
        h.update(b"|")
    return h.hexdigest()[:16]


def ensure_signature(directory, payload: dict, name: str = SIGNATURE_FILE) -> None:
    """Record the settings of a checkpoint folder, or stop if the folder was computed with other settings.

    ``name`` separates the signatures of learners that share one folder (the nested-CV model folder holds the fold models
    of every learner, told apart by a file suffix).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sig = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    f = directory / name
    old = load_json_or_none(f)
    if old is None:
        if any(not p.name.startswith("SIGNATURE") for p in directory.iterdir()):
            logger.warning(f"{directory} has checkpoints without a signature; adopting the current settings")
        atomic_json_dump({"sha256": sig, "payload": payload}, f, indent=1, default=str)
    elif old.get("sha256") != sig:
        raise SystemExit(
            f"Checkpoint folder {directory} was computed with different settings:\n"
            f"  stored : {old.get('payload')}\n  current: {payload}\n"
            "Refusing to resume; use another output folder or remove the old checkpoints."
        )
