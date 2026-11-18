"""Helpers shared by several analysis scripts (moved here from the scripts that first defined them).

sign_flip_p, metrics, predict_variant: directed-link ablation; exact_sign_flip_pvalue, holm_adjust,
best_chain_max_stat_pvalue: fold-level inference; _bca, clinical_intervals: figure intervals;
LEARNING_CURVE_PLAN: subsample sizes and chain of the learning curves.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from mecor.cvio import bca_mean_ci, best_chain, metric_by_fold

SEED = 42
DRUGS = ["CIP", "CTX", "CAZ"]


def sign_flip_p(d):
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    cnt = 0
    for signs in product([1.0, -1.0], repeat=n):
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / 2 ** n

def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    out = {"AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
           "MCC": float(matthews_corrcoef(y, yhat))}
    r = y == 1
    s = y == 0
    out["VME"] = float((yhat[r] == 0).mean()) if r.any() else np.nan
    out["ME"] = float((yhat[s] == 1).mean()) if s.any() else np.nan
    return out

def predict_variant(ests, order, k, X, Yt, variant, rng=None, permute_only=None):
    """Probability for downstream position k under a given upstream variant.

    permute_only: name of a single upstream drug to permute (others realized)."""
    Xa = np.asarray(X, float)
    cols = []
    probs_up = []
    for j in range(k):
        pj = ests[j].predict_proba(np.column_stack([Xa] + cols) if cols else Xa)[:, 1]
        probs_up.append(pj)
        if variant == "proba":
            cols.append(pj.reshape(-1, 1))
        else:
            cols.append((pj >= 0.5).astype(float).reshape(-1, 1))
    if variant == "oracle":
        cols = [Yt[order[j]].values.astype(float).reshape(-1, 1) for j in range(k)]
    elif variant == "permuted":
        cols = [Yt[order[j]].values.astype(float)[rng.permutation(len(Yt))].reshape(-1, 1) for j in range(k)]
    elif variant == "single_perm":
        cols = [c.copy() for c in cols]
        j = order.index(permute_only)
        cols[j] = cols[j][rng.permutation(len(Yt))]
    Xk = np.column_stack([Xa] + cols)
    return ests[k].predict_proba(Xk)[:, 1]

def exact_sign_flip_pvalue(d) -> tuple[float, float]:
    """Two-sided p-value via exact enumeration of 2^n sign flips on paired diffs.

    H0: each d_i is symmetric about 0 (sign exchangeability). Test statistic is
    mean(d); p = P(|mean(d_perm)| >= |mean(d_obs)|).
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n > 22:
        raise ValueError(f"n={n} too large for exact enumeration; use Monte Carlo.")
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)
    signs = 1 - 2 * bits  # 0 -> +1, 1 -> -1
    null_means = (signs * d).mean(axis=1)
    obs = d.mean()
    p = float(np.mean(np.abs(null_means) >= np.abs(obs) - 1e-15))
    return obs, p

def holm_adjust(pvals) -> np.ndarray:
    """Holm step-down adjustment of a vector of p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj_sorted = pvals[order] * (n - np.arange(n))
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj

def best_chain_max_stat_pvalue(D) -> tuple[int, float, float, float]:
    """Max-statistic permutation test for BR vs best-of-K chain orderings.

    D has shape (K, n) with D[c, i] = chain_c[i] - BR[i]. Test stat T = max_c
    mean(D[c]). The same per-fold sign flip is applied to all K chains (preserving
    their joint dependence and correcting for picking the best ordering). Returns
    (best_idx, observed_T, two_sided_p, one_sided_p_greater).
    """
    K, n = D.shape
    if n > 22:
        raise ValueError(f"n={n} too large for exact enumeration; use Monte Carlo.")
    means = D.mean(axis=1)
    obs_T = float(means.max())
    best_idx = int(means.argmax())
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)
    signs = (1 - 2 * bits).astype(float)            # (2^n, n)
    null_means = signs @ D.T / n                    # (2^n, K)
    null_T_max = null_means.max(axis=1)
    p_one_sided = float(np.mean(null_T_max >= obs_T - 1e-15))
    null_T_absmax = np.maximum(null_means.max(axis=1), -null_means.min(axis=1))
    p_two_sided = float(np.mean(null_T_absmax >= abs(obs_T) - 1e-15))
    return best_idx, obs_T, p_two_sided, p_one_sided

def _bca(values):
    return bca_mean_ci(values, seed=SEED)

def clinical_intervals(cv, drugs=DRUGS, best_metric="MCC"):
    """{"br" | "ensemble" | "best": {drug: {"VME" | "ME": (mean, low, high)}}} with BCa 95% intervals over folds."""
    out = {}
    for key in ("br", "ensemble", "best"):
        out[key] = {}
        for drug in drugs:
            method = {"br": "binary_relevance", "ensemble": "ensemble", "best": best_chain(cv, drug, best_metric)}[key]
            out[key][drug] = {m: _bca(list(metric_by_fold(cv, method, drug, m, clinical=True).values()))
                              for m in ("VME", "ME")}
    return out

LEARNING_CURVE_PLAN = {"s_orsola": {"sizes": [60, 90, 120, 150, 187], "draws": 5, "repeats": 2, "up": "CIP", "mid": "CTX", "last": "CAZ"},
        "driams_a": {"sizes": [187, 500, 1000, 2000], "draws": 3, "repeats": 1, "up": "CIP", "mid": "CRO", "last": "CAZ"}}
