"""Lightweight reader for cv_results.pkl produced by the nested-CV pipeline.

The pickles are ``mecor.evaluation.cv.CVResults`` / ``FoldResult`` dataclasses
holding numpy arrays and plain dicts. Importing the real module pulls in the
training stack (torch / catboost / tabpfn); since post-processing only needs the
stored arrays, we register a no-op stand-in for ``src.models.factory`` so
unpickling stays lightweight.

Note: the TabPFN v3 results were pickled with numpy >= 2.0, so a numpy >= 2.0
environment is required to read all three model pickles.
"""
from __future__ import annotations

import pickle
import sys
import types

import numpy as np


def _prepare_imports():
    class _Stub(types.ModuleType):
        def __getattr__(self, name):  # any symbol -> harmless no-op
            return lambda *a, **k: None

    sys.modules.setdefault("mecor.models.factory", _Stub("mecor.models.factory"))


def load_cv_results(path):
    """Unpickle a cv_results.pkl without importing the heavy training stack."""
    _prepare_imports()
    with open(path, "rb") as f:
        return pickle.load(f)


def methods(cv):
    return sorted({fr.method for fr in cv.fold_results})


def chain_methods(cv):
    return [m for m in methods(cv) if m.startswith("chain")]


def metric_by_fold(cv, method, drug, metric, clinical=False):
    """Map (repeat_idx, fold_idx) -> metric value for one method/drug/metric."""
    out = {}
    for fr in cv.fold_results:
        if fr.method != method:
            continue
        table = fr.clinical_metrics if clinical else fr.metrics
        out[(fr.repeat_idx, fr.fold_idx)] = float(table[drug][metric])
    return out


def aligned(a, b):
    """Return paired arrays for the folds present in both dicts."""
    keys = sorted(set(a) & set(b))
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def best_chain(cv, drug, metric):
    """Name of the chain ordering with the highest mean metric for a drug."""
    best, best_mean = None, -np.inf
    for m in chain_methods(cv):
        vals = list(metric_by_fold(cv, m, drug, metric).values())
        mu = float(np.mean(vals))
        if mu > best_mean:
            best, best_mean = m, mu
    return best


def pooled_predictions(cv, method, drug):
    """Concatenate out-of-fold y_true and y_prob across all folds."""
    yt, yp = [], []
    for fr in cv.fold_results:
        if fr.method != method:
            continue
        yt.append(np.asarray(fr.y_true[drug]))
        yp.append(np.asarray(fr.y_prob[drug]))
    return np.concatenate(yt), np.concatenate(yp)


def bca_mean_ci(vals, seed=42, n_resamples=9999):
    """BCa bootstrap 95% CI for the mean of a 1-D sample.

    Returns (mean, low, high). For a degenerate (zero-variance) sample the
    interval collapses to the mean. Shared by the statistics and figure scripts.
    """
    vals = np.asarray(vals, float)
    m = float(vals.mean())
    if np.allclose(vals.std(), 0.0):
        return m, m, m
    from scipy.stats import bootstrap
    res = bootstrap(
        (vals,), np.mean, method="BCa", confidence_level=0.95,
        n_resamples=n_resamples, random_state=np.random.default_rng(seed),
    )
    return m, float(res.confidence_interval.low), float(res.confidence_interval.high)
