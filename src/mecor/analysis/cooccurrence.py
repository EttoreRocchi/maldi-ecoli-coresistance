"""AMR co-occurrence analysis: phi coefficients, chi-squared tests, UpSet data."""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def phi_coefficient(y1: np.ndarray, y2: np.ndarray) -> float:
    """Compute phi coefficient (binary correlation) between two binary arrays."""
    n11 = ((y1 == 1) & (y2 == 1)).sum()
    n10 = ((y1 == 1) & (y2 == 0)).sum()
    n01 = ((y1 == 0) & (y2 == 1)).sum()
    n00 = ((y1 == 0) & (y2 == 0)).sum()

    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / denom


def compute_cooccurrence_analysis(Y: pd.DataFrame) -> Dict:
    """Comprehensive co-occurrence analysis for multi-label AMR data.

    Parameters
    ----------
    Y : pd.DataFrame
        Binary label matrix (columns = drug short names, values = 0/1)

    Returns
    -------
    dict with keys:
        'phi_matrix' : pd.DataFrame - pairwise phi coefficients
        'chi2_results' : list of dict - chi-squared test results per pair
        'profile_counts' : pd.Series - resistance profile frequencies
        'observed_vs_expected' : dict - co-occurrence observed vs expected rates
        'n_samples' : int
    """
    labels = list(Y.columns)
    n = len(Y)

    phi_matrix = pd.DataFrame(
        np.eye(len(labels)), index=labels, columns=labels, dtype=float
    )
    chi2_results = []

    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i >= j:
                continue
            phi = phi_coefficient(Y[l1].values, Y[l2].values)
            phi_matrix.loc[l1, l2] = phi
            phi_matrix.loc[l2, l1] = phi

            ct = pd.crosstab(Y[l1], Y[l2])
            if ct.shape == (2, 2):
                chi2, p_val, _dof, expected = chi2_contingency(ct, correction=False)
            else:
                chi2, p_val, _dof = 0.0, 1.0, 0

            obs = ((Y[l1] == 1) & (Y[l2] == 1)).sum() / n
            exp = Y[l1].mean() * Y[l2].mean()

            chi2_results.append({
                "drug_1": l1,
                "drug_2": l2,
                "phi": phi,
                "chi2": chi2,
                "p_value": p_val,
                "observed_cooccurrence": obs,
                "expected_cooccurrence": exp,
                "ratio_obs_exp": obs / exp if exp > 0 else np.inf,
            })

    profile_counts = Y.apply(
        lambda row: "_".join(
            [f"{c}={'R' if v == 1 else 'S'}" for c, v in row.items()]
        ),
        axis=1,
    ).value_counts()

    for r in chi2_results:
        l1, l2 = r["drug_1"], r["drug_2"]
        intersection = ((Y[l1] == 1) & (Y[l2] == 1)).sum()
        union = ((Y[l1] == 1) | (Y[l2] == 1)).sum()
        r["jaccard"] = intersection / union if union > 0 else 0.0

    return {
        "phi_matrix": phi_matrix,
        "chi2_results": chi2_results,
        "profile_counts": profile_counts,
        "n_samples": n,
        "resistance_rates": {c: float(Y[c].mean()) for c in labels},
    }
