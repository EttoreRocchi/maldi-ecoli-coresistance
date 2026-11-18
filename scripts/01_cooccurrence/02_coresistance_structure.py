#!/usr/bin/env python3
"""Co-resistance structure across sites and the Sant'Orsola instrument batch check.

Questions
  - Is the cross-site "phi gradient" of CIP with the third-generation cephalosporins (3GC: CTX at Sant'Orsola,
        CRO at DRIAMS) a difference in label dependency or mostly a marginal (prevalence) artefact? For each of the
        five sites: 2 x 2 tables, P(CIP-R | 3GC-R), P(CIP-R | 3GC-S), odds ratio with 95 percent Woolf interval,
        P(CAZ-R | 3GC-R), P(3GC-R | CAZ-R), discordant cephalosporin phenotypes, phi for the three pairs (asserted
        against results/cooccurrence/*), the maximum phi attainable given the two marginals and the ratio phi / phi_max,
        prevalences, n, the eight resistance-profile counts; DRIAMS-A additionally per year (2015 to 2018).
  - EUCAST "I" handling: raw value counts of Ciprofloxacin, Ceftriaxone and Ceftazidime per DRIAMS site and year
        among E. coli rows passing the inclusion rule except the S/I/R check, rows dropped by the loader, and the
        three phi coefficients and prevalences under the alternative mapping I -> R (the manuscript maps I -> S).
  - Full-panel dependency structure of E. coli at each DRIAMS site and pooled: every drug column with at least
        50 R and 50 S results after mapping (I -> S), pairwise phi on pairwise-complete observations, average-linkage
        clustering on 1 - phi, heatmaps, top 15 pairs with n, within- and between-class mean phi, chaining candidates.
  - Sant'Orsola instrument (Spectrometer 1 or 2 in data/metadata.csv): resistance prevalence per instrument with
        Fisher exact p, AUROC of the instrument indicator alone, per-instrument AUROC and MCC of binary relevance and
        of the chain ensemble from the logistic out-of-fold predictions (outer folds reconstructed and verified),
        chain gain within instrument (isolate-level cluster bootstrap and per-fold sign-flip test), and a spectral
        batch check (logistic regression predicting the instrument from the 6000 bins, 5-fold CV x 3 repeats).

Method notes
  The DRIAMS inclusion rule reproduces src/data/loader.py (E. coli, non-null CIP/CRO/CAZ, binned spectrum file
  present, value in S/I/R after strip and upper-casing); n and phi are asserted against results/cooccurrence/*.
  phi_max for marginals p <= q is sqrt(p (1 - q) / (q (1 - p))). Odds ratios use Woolf intervals, with 0.5 added
  to every cell when a cell is zero. Deterministic (seed 42). Refitting is limited to the instrument classifier
  (n_jobs = 8). Nothing is written under results/driams_* or results/cvsens.

Usage
  python scripts/01_cooccurrence/02_coresistance_structure.py              # full analysis, then the figures
  python scripts/01_cooccurrence/02_coresistance_structure.py --plot-only  # figures only, from results/stats/coresistance_structure.json

Outputs
  results/stats/coresistance_structure.json
  tables/coresistance_structure.md
  figures/supplementary/coresistance_phi_marginals.png (+ pdf)
  figures/supplementary/driams_full_panel_phi_<site>.png (+ pdf; driams_a, driams_b, driams_c, driams_d, pooled)
  figures/supplementary/instrument_batch_check.png (+ pdf)
"""
from __future__ import annotations

import json
import os
from itertools import combinations, product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")  # with n_jobs = 8 this keeps the refit under 16 cores

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)

from mecor.config import Config  # noqa: E402
from mecor.cvio import load_cv_results  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402

SEED = 42
N_JOBS = 8
N_BOOT = 2000
MIN_RS = 50          #: minimum R and minimum S results for a drug column
MIN_PAIR_N = 30      #: minimum pairwise-complete n to report a phi
SITES = ["s_orsola", "driams_a", "driams_b", "driams_c", "driams_d"]
SITE_LABEL = {"s_orsola": "Sant'Orsola", "driams_a": "DRIAMS-A", "driams_b": "DRIAMS-B",
              "driams_c": "DRIAMS-C", "driams_d": "DRIAMS-D", "pooled": "DRIAMS pooled"}
DRIAMS_TARGETS = ["Ciprofloxacin", "Ceftriaxone", "Ceftazidime"]
SHORT = {"Ciprofloxacin": "CIP", "Ceftriaxone": "CRO", "Ceftazidime": "CAZ"}
MAP_IS = {"S": 0, "I": 0, "R": 1}
MAP_IR = {"S": 0, "I": 1, "R": 1}
FIGDIR = ROOT / "figures/supplementary"


def phi_coef(a, b):
    """Phi coefficient and the 2 x 2 cells (RR, RS, SR, SS) for two 0/1 arrays."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    den = np.sqrt(float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    ph = (n11 * n00 - n10 * n01) / den if den > 0 else np.nan
    return float(ph), (n11, n10, n01, n00)


def phi_max(p, q):
    """Largest phi two binary variables with prevalences p and q can attain."""
    p, q = sorted([float(p), float(q)])
    if p <= 0 or q >= 1:
        return np.nan
    return float(np.sqrt(p * (1 - q) / (q * (1 - p))))


def odds_ratio(n11, n10, n01, n00):
    cells = np.array([n11, n10, n01, n00], float)
    if (cells == 0).any():
        cells = cells + 0.5
    a, b, c, d = cells
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(orr), float(orr * np.exp(-1.959964 * se)), float(orr * np.exp(1.959964 * se))


def safe_div(a, b):
    return float(a) / float(b) if b else np.nan


def cohort_stats(Y, ceph):
    """All co-resistance quantities for one cohort. Y has columns CIP, <ceph>, CAZ with 0/1 values."""
    n = len(Y)
    out = {"n": int(n),
           "prevalence": {d: {"n_R": int(Y[d].sum()), "frac": float(Y[d].mean())} for d in Y.columns}}
    out["pairs"] = {}
    for a, b in [("CIP", ceph), ("CIP", "CAZ"), (ceph, "CAZ")]:
        ph, (n11, n10, n01, n00) = phi_coef(Y[a], Y[b])
        pm = phi_max(Y[a].mean(), Y[b].mean())
        orr, lo, hi = odds_ratio(n11, n10, n01, n00)
        fp = float(fisher_exact([[n11, n10], [n01, n00]])[1])
        out["pairs"][f"{a}-{b}"] = {
            "phi": ph, "phi_max": pm, "phi_ratio": ph / pm if pm and not np.isnan(pm) else np.nan,
            "OR": orr, "OR_ci95": [lo, hi], "fisher_p": fp,
            "cells": {"RR": n11, "RS": n10, "SR": n01, "SS": n00},
            f"P({a}R|{b}R)": safe_div(n11, n11 + n01), f"P({a}R|{b}S)": safe_div(n10, n10 + n00),
            f"P({b}R|{a}R)": safe_div(n11, n11 + n10), f"P({b}R|{a}S)": safe_div(n01, n01 + n00),
        }
    disc_a = int(((Y[ceph] == 1) & (Y["CAZ"] == 0)).sum())
    disc_b = int(((Y["CAZ"] == 1) & (Y[ceph] == 0)).sum())
    out["ceph_discordant"] = {f"{ceph}R_CAZS": disc_a, f"CAZR_{ceph}S": disc_b,
                              "frac_of_any_ceph_R": safe_div(disc_a + disc_b, int(((Y[ceph] == 1) | (Y["CAZ"] == 1)).sum()))}
    prof = {}
    for combo in product([0, 1], repeat=3):
        key = "".join("R" if v else "S" for v in combo)
        prof[key] = int(((Y["CIP"] == combo[0]) & (Y[ceph] == combo[1]) & (Y["CAZ"] == combo[2])).sum())
    out["profiles_CIP_3GC_CAZ"] = prof
    return out


def sign_flip_p(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return np.nan
    obs = abs(d.mean())
    cnt = 0
    for signs in product([1.0, -1.0], repeat=n):
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / 2 ** n


def clean(o):
    """Recursively convert numpy scalars / arrays and NaN for JSON."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return lines


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def fp(x):
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def driams_year_paths(cfg):
    if len(cfg.driams_years) > 1:
        return [(y, Path(cfg.metadata_path) / y / f"{y}_clean.csv", Path(cfg.binned_dir) / y) for y in cfg.driams_years]
    y = cfg.driams_years[0]
    return [(y, Path(cfg.metadata_path), Path(cfg.binned_dir))]


def load_driams_ecoli(site):
    """E. coli rows of one DRIAMS site (all years), with year, has3 (three target drugs non-null) and
    spectrum_exists flags, and the raw (stripped) target strings."""
    cfg = Config.from_yaml(str(ROOT / f"configs/{site}.yaml"))
    frames = []
    for year, meta_path, bin_dir in driams_year_paths(cfg):
        meta = pd.read_csv(meta_path, low_memory=False)
        e = meta[meta["species"] == cfg.species].copy()
        e["year"] = year
        e["site"] = site
        e["has3"] = e[DRIAMS_TARGETS].notna().all(axis=1)
        e["spectrum_exists"] = [os.path.exists(bin_dir / f"{c}.txt") if h else False
                                for c, h in zip(e["code"].astype(str), e["has3"])]
        frames.append(e)
    return pd.concat(frames, ignore_index=True)


def included_rows(e):
    """Loader inclusion rule: has3, spectrum present, all three values in S/I/R after strip + upper."""
    ok = e["has3"] & e["spectrum_exists"]
    for d in DRIAMS_TARGETS:
        ok &= e[d].astype(str).str.strip().str.upper().isin(["S", "I", "R"])
    return ok


def labels_from(e, mapping):
    Y = pd.DataFrame({SHORT[d]: e[d].astype(str).str.strip().str.upper().map(mapping) for d in DRIAMS_TARGETS},
                     index=e.index)
    return Y.astype(int)


NON_DRUG = {"code", "species", "laboratory_species", "combined_code", "genus", "ESBL", "MRSA", "year", "site",
            "has3", "spectrum_exists", "Quinolones", "Aminoglycosides"}
RENAME_POOL = {"Cotrimoxazol": "Cotrimoxazole", "Ampicillin-Amoxicillin": "Ampicillin",
               "Fosfomycin-Trometamol": "Fosfomycin", "Cefalotin-Cefazolin": "Cefazolin"}
CLASS_ORDER = ["Penicillin/BLI", "Cephalosporin", "Carbapenem", "Monobactam", "Fluoroquinolone",
               "Aminoglycoside", "TMP-SMX", "Other"]
CLASS_ABBR = {"Penicillin/BLI": "PEN", "Cephalosporin": "CEPH", "Carbapenem": "CARB", "Monobactam": "MONO",
              "Fluoroquinolone": "FQ", "Aminoglycoside": "AG", "TMP-SMX": "SXT", "Other": "OTH"}


def drug_class(name):
    n = name.lower()
    if any(k in n for k in ["ampicillin", "amoxicillin", "piperacillin", "ticarcillin", "penicillin"]):
        return "Penicillin/BLI"
    if n.startswith("cef"):
        return "Cephalosporin"
    if any(k in n for k in ["imipenem", "meropenem", "ertapenem"]):
        return "Carbapenem"
    if "aztreonam" in n:
        return "Monobactam"
    if any(k in n for k in ["ciprofloxacin", "levofloxacin", "norfloxacin", "moxifloxacin", "ofloxacin",
                            "pefloxacin", "sparfloxacin"]):
        return "Fluoroquinolone"
    if any(k in n for k in ["gentamicin", "tobramycin", "amikacin", "streptomycin"]):
        return "Aminoglycoside"
    if "cotrimoxazol" in n:
        return "TMP-SMX"
    return "Other"


def is_drug_column(c):
    if c in NON_DRUG:
        return False
    return not any(t in c for t in ["_screen", "high_level", "GRD", ".1"])


def panel_matrix(e, mapping=MAP_IS):
    cols = [c for c in e.columns if is_drug_column(c)]
    M = pd.DataFrame(index=e.index)
    for c in cols:
        v = e[c].astype(str).str.strip().str.upper().map(mapping)
        M[c] = v
    keep = [c for c in M.columns if (M[c] == 1).sum() >= MIN_RS and (M[c] == 0).sum() >= MIN_RS]
    return M[keep]


def pairwise_phi(M):
    cols = list(M.columns)
    k = len(cols)
    P = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    N = pd.DataFrame(np.zeros((k, k), int), index=cols, columns=cols)
    pairs = []
    for i, j in combinations(range(k), 2):
        a, b = cols[i], cols[j]
        m = M[a].notna() & M[b].notna()
        n = int(m.sum())
        ph = phi_coef(M.loc[m, a], M.loc[m, b])[0] if n >= MIN_PAIR_N else np.nan
        P.iloc[i, j] = P.iloc[j, i] = ph
        N.iloc[i, j] = N.iloc[j, i] = n
        pairs.append({"drug_1": a, "drug_2": b, "phi": ph, "n": n,
                      "prev_1": float(M.loc[m, a].mean()) if n else np.nan,
                      "prev_2": float(M.loc[m, b].mean()) if n else np.nan})
    for c in cols:
        N.loc[c, c] = int(M[c].notna().sum())
    return P, N, pairs


def cluster_order(P):
    D = 1.0 - P.fillna(0.0).values
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return [P.index[i] for i in dendrogram(Z, no_plot=True)["leaves"]]


from mecor.visualization import paper_style as ps  # noqa: E402

PAIR_LABELS = ["CIP with CTX/CRO", "CIP with CAZ", "CTX/CRO with CAZ"]


def setup_mpl():
    """Apply the shared paper style and return pyplot (kept for the callers in main)."""
    import matplotlib.pyplot as plt
    ps.apply()
    return plt


def save(fig, stem):
    ps.save(fig, FIGDIR / stem)


def _pair_keys(pairs):
    """Keys of the three pairs in a site's 'pairs' dict (CTX at Sant'Orsola, CRO at DRIAMS)."""
    ceph = "CTX" if "CIP-CTX" in pairs else "CRO"
    return [f"CIP-{ceph}", "CIP-CAZ", f"{ceph}-CAZ"]


def heatmap_phi(P, N, order, title, stem, plt=None):
    """Lower-triangle phi heatmap of the full panel, SEQ_CMAP from 0 to 1, value printed in every cell.

    P is a DataFrame (or dict of dicts) of phi, None or NaN where fewer than MIN_PAIR_N isolates were tested for
    both drugs; `title` and `N` are accepted for call compatibility and not drawn (the caption carries them).
    """
    import matplotlib.pyplot as plt
    ps.apply()
    P = P if isinstance(P, pd.DataFrame) else pd.DataFrame(P)
    Po = P.loc[order, order].astype(float).values
    k = len(order)
    M = np.full((k - 1, k - 1), np.nan)           # rows = order[1:], columns = order[:-1]
    for i in range(1, k):
        for j in range(i):
            M[i - 1, j] = Po[i, j]
    cell = 0.30 if k > 12 else 0.36
    fig, ax = plt.subplots(figsize=(min(ps.WIDTH["double"], cell * (k - 1) + 2.7), cell * (k - 1) + 1.5))
    cmap = ps.SEQ_CMAP.copy(); cmap.set_bad(ps.SURFACE)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="equal")
    fs = 6 if k > 12 else 6.5
    for i in range(1, k):
        for j in range(i):
            v = Po[i, j]
            if np.isnan(v):
                ax.text(j, i - 1, "n/a", ha="center", va="center", fontsize=fs - 0.5, color=ps.MUTED)
            else:
                ax.text(j, i - 1, f"{v:.2f}", ha="center", va="center", fontsize=fs, color=ps.SURFACE if v > 0.62 else ps.INK)
    lab = [f"{d} ({CLASS_ABBR[drug_class(d)]})" for d in order]
    ax.set_xticks(range(k - 1)); ax.set_xticklabels(lab[:-1], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(k - 1)); ax.set_yticklabels(lab[1:])
    ax.set_xticks(np.arange(-0.5, k - 1, 1), minor=True); ax.set_yticks(np.arange(-0.5, k - 1, 1), minor=True)
    ax.grid(which="minor", color=ps.SURFACE, linewidth=1.2); ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, shrink=0.6, anchor=(0.0, 1.0), ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Phi coefficient (pairwise complete)"); cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.6)
    save(fig, stem)
    plt.close(fig)


def plot_phi_marginals(res):
    """coresistance_phi_marginals: rows = the three drug pairs; (a) observed phi and attainable maximum per site,
    (b) DRIAMS-A per year, (c) change in phi when I is mapped to R instead of S."""
    import matplotlib.pyplot as plt
    x10a = res["X10a"]["sites"]; years = res["X10a"]["driams_a_per_year"]; i2r = res["X10b"]["I_to_R"]
    sites = [s for s in ps.SITE_ORDER if s in x10a]
    dsites = [s for s in ps.SITE_ORDER if s in i2r]
    yrs = sorted(years)
    fig, axes = ps.figure("double", height=5.6, nrows=3, ncols=3, gridspec_kw={"width_ratios": [1.15, 1.0, 0.95]})
    for r, lab in enumerate(PAIR_LABELS):
        # (a) observed phi (filled) and the maximum given the marginals (hollow), per site
        ax = axes[r, 0]
        for xi, s in enumerate(sites):
            rec = x10a[s]["pairs"][_pair_keys(x10a[s]["pairs"])[r]]
            ax.plot([xi, xi], [rec["phi"], rec["phi_max"]], color=ps.LIGHT, lw=0.8, zorder=1)
            ps.ci_point(ax, xi, rec["phi_max"], color=ps.INK, hollow=True, ms=6.0, zorder=2,
                        label="Maximum phi given the marginals" if (r, xi) == (0, 0) else None)
            ps.ci_point(ax, xi, rec["phi"], color=ps.INK, ms=4.0, zorder=3, label="Observed phi" if (r, xi) == (0, 0) else None)
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels([ps.SITE_NAME[s] for s in sites], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.5, len(sites) - 0.5); ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if r == 1:
            ax.set_ylabel("Phi coefficient")
        ax.text(0.98, 0.97, lab, transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
        ps.despine(ax)
        # (b) DRIAMS-A per year
        ax = axes[r, 1]
        key = ["CIP-CRO", "CIP-CAZ", "CRO-CAZ"][r]
        xs = [int(y) for y in yrs]
        ax.plot(xs, [years[y]["pairs"][key]["phi_max"] for y in yrs], color=ps.INK, lw=1.2, ls="--",
                marker="o", ms=6.0, mfc=ps.SURFACE, mew=1.0, zorder=2)
        ax.plot(xs, [years[y]["pairs"][key]["phi"] for y in yrs], color=ps.INK, lw=1.2, marker="o", ms=4.0, zorder=3)
        ax.set_xticks(xs); ax.set_xticklabels([f"{y}\nn = {years[y]['n']:,}" for y in yrs])
        ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4); ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if r == 0:
            ax.text(0.98, 0.97, "DRIAMS-A per year", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=ps.INK2)
        ps.despine(ax)
        # (c) effect of the I mapping
        ax = axes[r, 2]
        ps.hline(ax, 0, color=ps.LIGHT)
        ps.ci_point(ax, list(range(len(dsites))), [i2r[s]["phi"][key]["diff"] for s in dsites], color=ps.INK)
        ax.set_xticks(range(len(dsites)))
        ax.set_xticklabels([ps.SITE_NAME[s] for s in dsites], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.5, len(dsites) - 0.5); ax.set_ylim(-0.06, 0.26); ax.set_yticks([0, 0.1, 0.2])
        if r == 1:
            ax.set_ylabel("Phi (I to R) minus phi (I to S)")
        ps.despine(ax)
    for r in (0, 1):
        for c in range(3):
            axes[r, c].tick_params(labelbottom=False)
    for c, letter in enumerate("abc"):
        ps.panel_label(axes[0, c], letter)
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [l.index("Observed phi"), l.index("Maximum phi given the marginals")]
    ps.legend_above(fig, handles=[h[i] for i in order], labels=[l[i] for i in order], y=1.0)
    save(fig, "coresistance_phi_marginals")
    plt.close(fig)


def plot_instrument_check(res):
    """instrument_batch_check: (a) prevalence per instrument, (b) AUROC of BR and the chain ensemble within instrument,
    (c) chain gain within instrument with the isolate-level bootstrap interval, (d) spectral batch check."""
    import matplotlib.pyplot as plt
    x13 = res["X13"]; labels = list(x13["prevalence"]); n_i = {int(k): v for k, v in x13["n_per_instrument"].items()}
    groups = [("instr1", 0.0, f"Instrument 1 (n = {n_i[1]})"), ("instr2", 3.7, f"Instrument 2 (n = {n_i[2]})")]
    xt = [x0 + j for _, x0, _ in groups for j in range(len(labels))]
    fig, axes = ps.figure("double", height=5.0, nrows=2, ncols=2)
    # (a) prevalence per instrument with the Fisher exact p
    ax = axes[0, 0]
    for j, d in enumerate(labels):
        p1, p2 = x13["prevalence"][d]["instr1"]["frac"], x13["prevalence"][d]["instr2"]["frac"]
        ps.ci_point(ax, j - 0.12, p1, color=ps.INK, hollow=True, label=groups[0][2] if j == 0 else None)
        ps.ci_point(ax, j + 0.12, p2, color=ps.INK, label=groups[1][2] if j == 0 else None)
        ax.text(j, max(p1, p2) + 0.035, f"p = {fp(x13['prevalence'][d]['fisher_p'])}", ha="center", va="bottom",
                fontsize=6.5, color=ps.INK2)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels); ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 0.7); ax.set_ylabel("Resistance prevalence")
    ax.legend(loc="upper left", frameon=False); ps.despine(ax)
    # (b) AUROC of BR and the ensemble within instrument (mean over the three repeats, thin line = range over repeats)
    ax = axes[0, 1]
    allv = []
    for g, (grp, x0, gl) in enumerate(groups):
        for j, d in enumerate(labels):
            rec = x13["per_instrument_performance"][d][grp]
            for m, role, off in [("BR", "br", -0.15), ("Ens", "ensemble", 0.15)]:
                v = [x for x in rec[m]["AUROC_per_repeat"] if x is not None]
                allv += v
                ps.ci_point(ax, x0 + j + off, rec[m]["AUROC_mean_over_repeats"], min(v), max(v), color=ps.ROLE[role],
                            label=ps.ROLE_NAME[role] if (g, j) == (0, 0) else None)
        ax.text(x0 + 1, -0.19, gl, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=ps.INK2)
    ax.axvline(2.85, color=ps.GRID, lw=0.6)
    ax.set_xticks(xt); ax.set_xticklabels(labels * 2); ax.set_xlim(-0.6, 6.3)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC (out of fold)")
    ps.legend_above(ax, y=1.01); ps.despine(ax)
    # (c) chain gain within instrument, isolate-level cluster bootstrap 95 percent interval
    ax = axes[1, 0]
    ps.hline(ax, 0, color=ps.LIGHT)
    for g, (grp, x0, gl) in enumerate(groups):
        for j, d in enumerate(labels):
            gn = x13["per_instrument_performance"][d][grp]["chain_gain"]
            lo, hi = gn["dAUROC_boot_ci95"]
            ps.ci_point(ax, x0 + j, gn["dAUROC_mean_over_repeats"], lo, hi, color=ps.ROLE["ensemble"])
        ax.text(x0 + 1, -0.19, gl, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=ps.INK2)
    ax.axvline(2.85, color=ps.GRID, lw=0.6)
    ax.set_xticks(xt); ax.set_xticklabels(labels * 2); ax.set_xlim(-0.6, 6.3)
    ax.set_ylabel("Gain over binary relevance\n(AUROC)"); ps.despine(ax)
    lo_all = min(x13["per_instrument_performance"][d][g]["chain_gain"]["dAUROC_boot_ci95"][0] for d in labels for g in ("instr1", "instr2"))
    hi_all = max(x13["per_instrument_performance"][d][g]["chain_gain"]["dAUROC_boot_ci95"][1] for d in labels for g in ("instr1", "instr2"))
    ax.set_ylim(lo_all - 0.02, hi_all + 0.02)
    # (d) spectral batch check: one point per outer fold
    ax = axes[1, 1]
    fa = x13["spectral_instrument_classifier"]["fold_AUROC"]
    jit = np.random.default_rng(SEED).uniform(-0.3, 0.3, size=len(fa))
    ax.plot(fa, jit, linestyle="none", marker="o", ms=4.5, color=ps.INK, mec=ps.SURFACE, mew=0.6, zorder=3)
    mean = float(np.mean(fa))
    ax.axvline(mean, color=ps.INK2, lw=1.0, zorder=2)
    ax.text(mean + 0.008, 0.62, f"mean {mean:.3f}", ha="left", va="bottom", fontsize=7, color=ps.INK2)
    ax.axvline(0.5, color=ps.MUTED, lw=0.7, zorder=1)
    ax.text(0.508, 0.62, "chance", ha="left", va="bottom", fontsize=7, color=ps.MUTED)
    ax.set_xlim(0.45, 1.0); ax.set_ylim(-0.8, 0.8); ax.set_yticks([])
    ax.set_xlabel("AUROC of the instrument classifier (one point per outer fold)")
    ax.spines["left"].set_visible(False); ps.despine(ax, grid="x")
    for ax, letter in zip(axes.ravel(), "abcd"):
        ps.panel_label(ax, letter, dx=-0.16, dy=1.03)
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, "instrument_batch_check")
    plt.close(fig)


def plot_from_results(res):
    """Draw every figure of this script from the stored results (results/stats/coresistance_structure.json)."""
    setup_mpl()
    plot_phi_marginals(res)
    plot_instrument_check(res)
    for s, r in res["X10c"].items():
        heatmap_phi(pd.DataFrame(r["phi_matrix"]), None, r["cluster_order"], "", f"driams_full_panel_phi_{s}")
    print("saved figures")


def main():
    res = {"notes": {}}
    tables = ["# Co-resistance structure and instrument batch check", ""]
    plt = setup_mpl()

    cfg_local = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml"))
    X, Y = load_dataset(cfg_local)
    meta = pd.read_csv(Config.from_yaml(str(ROOT / "configs/s_orsola.yaml")).metadata_path)
    assert meta["ID"].is_unique and X.index.isin(meta["ID"]).all(), "X.index not found in metadata ID"
    mloc = meta.set_index("ID").loc[X.index]
    for s, lng in [("CIP", "Ciprofloxacina"), ("CTX", "Cefotaxime"), ("CAZ", "Ceftazidime")]:
        assert (Y[s].values == (mloc[lng].values == "R").astype(int)).all(), f"label mismatch {s}"
    local_raw = {lng: mloc[lng].value_counts(dropna=False).to_dict()
                 for lng in ["Ciprofloxacina", "Cefotaxime", "Ceftazidime"]}
    res["notes"]["s_orsola_raw_values_analysed_isolates"] = local_raw
    res["notes"]["s_orsola_has_I"] = any("I" in d for d in local_raw.values())
    print(f"Sant'Orsola: {len(Y)} isolates, raw values {local_raw}")

    ecoli = {s: load_driams_ecoli(s) for s in ["driams_a", "driams_b", "driams_c", "driams_d"]}
    cohorts = {"s_orsola": (Y.copy(), "CTX")}
    for s, e in ecoli.items():
        inc = included_rows(e)
        cohorts[s] = (labels_from(e[inc], MAP_IS), "CRO")

    coocc_check = {}
    x10a = {}
    for s in SITES:
        Yc, ceph = cohorts[s]
        st = cohort_stats(Yc, ceph)
        x10a[s] = st
        ref = json.load(open(ROOT / f"results/cooccurrence/{s}/cooccurrence_results.json"))
        assert ref["n_samples"] == st["n"], f"{s}: n {st['n']} differs from stored {ref['n_samples']}"
        diffs = []
        for r in ref["pairwise_tests"]:
            key = f"{r['drug_1']}-{r['drug_2']}"
            diffs.append(abs(st["pairs"][key]["phi"] - r["phi"]))
        coocc_check[s] = {"n_stored": ref["n_samples"], "n_here": st["n"], "max_abs_phi_diff": float(max(diffs))}
        assert max(diffs) < 1e-9, f"{s}: phi mismatch {max(diffs)}"
        print(f"{s}: n={st['n']} reproduces results/cooccurrence (max |dphi| = {max(diffs):.1e})")
    res["X10a"] = {"sites": x10a, "cooccurrence_reproduction": coocc_check}

    # DRIAMS-A per year
    ea = ecoli["driams_a"]
    inc_a = included_rows(ea)
    x10a_years = {}
    for y in sorted(ea["year"].unique()):
        m = inc_a & (ea["year"] == y)
        x10a_years[y] = cohort_stats(labels_from(ea[m], MAP_IS), "CRO")
    res["X10a"]["driams_a_per_year"] = x10a_years

    def x10a_rows(name, st, ceph):
        p = st["pairs"]
        a = p[f"CIP-{ceph}"]
        b = p[f"{ceph}-CAZ"]
        return [name, st["n"],
                f"{st['prevalence']['CIP']['frac']:.3f} / {st['prevalence'][ceph]['frac']:.3f} / {st['prevalence']['CAZ']['frac']:.3f}",
                f3(a[f"P(CIPR|{ceph}R)"]), f3(a[f"P(CIPR|{ceph}S)"]),
                f"{a['OR']:.2f} [{a['OR_ci95'][0]:.2f}, {a['OR_ci95'][1]:.2f}]",
                f3(b[f"P(CAZR|{ceph}R)"]), f3(b[f"P({ceph}R|CAZR)"]),
                f"{st['ceph_discordant'][f'{ceph}R_CAZS']} / {st['ceph_discordant'][f'CAZR_{ceph}S']}"]

    def phi_rows(name, st, ceph):
        rows = []
        for key in [f"CIP-{ceph}", "CIP-CAZ", f"{ceph}-CAZ"]:
            q = st["pairs"][key]
            rows.append([name, key, f3(q["phi"]), f3(q["phi_max"]), f3(q["phi_ratio"]),
                         f"{q['OR']:.2f} [{q['OR_ci95'][0]:.2f}, {q['OR_ci95'][1]:.2f}]", q["cells"]["RR"], q["cells"]["RS"],
                         q["cells"]["SR"], q["cells"]["SS"]])
        return rows

    tables += ["##. Conditional probabilities of CIP resistance given the 3GC phenotype (3GC = CTX at Sant'Orsola, CRO at DRIAMS)", "",
               "Discordant = count of 3GC-R and CAZ-S / count of CAZ-R and 3GC-S.", ""]
    rows = [x10a_rows(SITE_LABEL[s], x10a[s], cohorts[s][1]) for s in SITES]
    rows += [x10a_rows(f"DRIAMS-A {y}", x10a_years[y], "CRO") for y in sorted(x10a_years)]
    tables += md_table(["Site", "n", "Prevalence CIP / 3GC / CAZ", "P(CIP-R | 3GC-R)", "P(CIP-R | 3GC-S)", "OR [95% CI]",
                        "P(CAZ-R | 3GC-R)", "P(3GC-R | CAZ-R)", "Discordant cephalosporins"], rows)
    tables += ["", "##. Phi, maximum phi attainable given the marginals, and the 2 x 2 cells (RR, RS, SR, SS in the order drug_1, drug_2)", ""]
    rows = []
    for s in SITES:
        rows += phi_rows(SITE_LABEL[s], x10a[s], cohorts[s][1])
    for y in sorted(x10a_years):
        rows += phi_rows(f"DRIAMS-A {y}", x10a_years[y], "CRO")
    tables += md_table(["Site", "Pair", "phi", "phi_max", "phi / phi_max", "OR [95% CI]", "RR", "RS", "SR", "SS"], rows)
    tables += ["", "##. Eight resistance profiles (order CIP, 3GC, CAZ; R = resistant, S = susceptible)", ""]
    prof_keys = ["SSS", "RSS", "SRS", "SSR", "RRS", "RSR", "SRR", "RRR"]
    rows = []
    for s in SITES:
        pr = x10a[s]["profiles_CIP_3GC_CAZ"]
        rows.append([SITE_LABEL[s], x10a[s]["n"]] + [f"{pr[k]} ({100 * pr[k] / x10a[s]['n']:.1f}%)" for k in prof_keys])
    for y in sorted(x10a_years):
        pr = x10a_years[y]["profiles_CIP_3GC_CAZ"]
        rows.append([f"DRIAMS-A {y}", x10a_years[y]["n"]] + [f"{pr[k]} ({100 * pr[k] / x10a_years[y]['n']:.1f}%)" for k in prof_keys])
    tables += md_table(["Site", "n"] + prof_keys, rows)

    x10b = {"raw_counts": {}, "dropped": {}, "I_to_R": {}}
    rows_raw, rows_ir = [], []
    for s, e in ecoli.items():
        pre = e["has3"] & e["spectrum_exists"]
        for y in sorted(e["year"].unique()):
            m = pre & (e["year"] == y)
            sub = e[m]
            counts = {d: sub[d].astype(str).str.strip().value_counts().to_dict() for d in DRIAMS_TARGETS}
            inc = included_rows(sub)
            drop = int((~inc).sum())
            x10b["raw_counts"][f"{s}_{y}"] = counts
            x10b["dropped"][f"{s}_{y}"] = {"rows_before_SIR_check": int(m.sum()), "rows_dropped": drop,
                                           "rows_included": int(inc.sum())}
            for d in DRIAMS_TARGETS:
                cnt = counts[d]
                std = {k: cnt.get(k, 0) for k in ["S", "I", "R"]}
                other = {k: v for k, v in cnt.items() if k.upper() not in ("S", "I", "R")}
                rows_raw.append([SITE_LABEL[s], y, SHORT[d], std["S"], std["I"], std["R"],
                                 "; ".join(f"{k}: {v}" for k, v in sorted(other.items())) or "none",
                                 int(m.sum()), drop if d == DRIAMS_TARGETS[0] else ""])
        # site-level: I -> R versus I -> S on the included rows
        inc = included_rows(e)
        Y_is = labels_from(e[inc], MAP_IS)
        Y_ir = labels_from(e[inc], MAP_IR)
        caz_i = inc & (e["Ceftazidime"].astype(str).str.strip().str.upper() == "I")
        cro_of_caz_i = e.loc[caz_i, "Ceftriaxone"].astype(str).str.strip().str.upper().value_counts().to_dict()
        cip_of_caz_i = e.loc[caz_i, "Ciprofloxacin"].astype(str).str.strip().str.upper().value_counts().to_dict()
        rec = {"n": int(inc.sum()), "n_I": {SHORT[d]: int((e.loc[inc, d].astype(str).str.strip().str.upper() == "I").sum())
                                             for d in DRIAMS_TARGETS},
               "CAZ_I_isolates_by_CRO": cro_of_caz_i, "CAZ_I_isolates_by_CIP": cip_of_caz_i,
               "prevalence_I_to_S": {c: float(Y_is[c].mean()) for c in Y_is.columns},
               "prevalence_I_to_R": {c: float(Y_ir[c].mean()) for c in Y_ir.columns}, "phi": {}}
        for a, b in [("CIP", "CRO"), ("CIP", "CAZ"), ("CRO", "CAZ")]:
            p_is = phi_coef(Y_is[a], Y_is[b])[0]
            p_ir = phi_coef(Y_ir[a], Y_ir[b])[0]
            rec["phi"][f"{a}-{b}"] = {"I_to_S": p_is, "I_to_R": p_ir, "diff": p_ir - p_is,
                                      "phi_max_I_to_S": phi_max(Y_is[a].mean(), Y_is[b].mean()),
                                      "phi_max_I_to_R": phi_max(Y_ir[a].mean(), Y_ir[b].mean())}
            rows_ir.append([SITE_LABEL[s], f"{a}-{b}", f3(p_is), f3(p_ir), f"{p_ir - p_is:+.3f}",
                            f3(rec["phi"][f"{a}-{b}"]["phi_max_I_to_S"]), f3(rec["phi"][f"{a}-{b}"]["phi_max_I_to_R"])])
        x10b["I_to_R"][s] = rec
        # DRIAMS-A per year
        if s == "driams_a":
            rec_y = {}
            for y in sorted(e["year"].unique()):
                m = inc & (e["year"] == y)
                Yi, Yr = labels_from(e[m], MAP_IS), labels_from(e[m], MAP_IR)
                rec_y[y] = {"n": int(m.sum()),
                            "n_I_CAZ": int((e.loc[m, "Ceftazidime"].astype(str).str.strip().str.upper() == "I").sum()),
                            "phi": {f"{a}-{b}": {"I_to_S": phi_coef(Yi[a], Yi[b])[0], "I_to_R": phi_coef(Yr[a], Yr[b])[0]}
                                    for a, b in [("CIP", "CRO"), ("CIP", "CAZ"), ("CRO", "CAZ")]},
                            "prev_CAZ_I_to_S": float(Yi["CAZ"].mean()), "prev_CAZ_I_to_R": float(Yr["CAZ"].mean())}
            x10b["driams_a_per_year"] = rec_y
    x10b["s_orsola"] = {"n_I": 0, "note": "No I results among the analysed Sant'Orsola isolates for CIP, CTX or CAZ "
                                           "(raw values are S or R only), so the I mapping has no effect there."}
    res["X10b"] = x10b
    tables += ["", "##. Raw S/I/R values of the three target drugs per DRIAMS site and year (E. coli rows with all three drugs non-null and a spectrum file; before the S/I/R check)", "",
               "Rows dropped = rows with at least one non-standard value (for any of the three drugs); given once per site-year.", ""]
    tables += md_table(["Site", "Year", "Drug", "S", "I", "R", "Non-standard values", "Rows before check", "Rows dropped"], rows_raw)
    tables += ["", "##. Phi under the manuscript mapping (I to S) and under I to R, per DRIAMS site (included cohorts)", ""]
    tables += md_table(["Site", "Pair", "phi (I to S)", "phi (I to R)", "difference", "phi_max (I to S)", "phi_max (I to R)"], rows_ir)
    rows = []
    for s in ["driams_a", "driams_b", "driams_c", "driams_d"]:
        rec = x10b["I_to_R"][s]
        rows.append([SITE_LABEL[s], rec["n"], rec["n_I"]["CIP"], rec["n_I"]["CRO"], rec["n_I"]["CAZ"],
                     f"{rec['CAZ_I_isolates_by_CRO'].get('R', 0)} / {rec['CAZ_I_isolates_by_CRO'].get('S', 0) + rec['CAZ_I_isolates_by_CRO'].get('I', 0)}"] +
                    [f"{rec['prevalence_I_to_S'][c]:.3f} / {rec['prevalence_I_to_R'][c]:.3f}" for c in ["CIP", "CRO", "CAZ"]])
    tables += ["", "##. Number of I results among included isolates, CRO status of the CAZ-I isolates, and prevalence under I to S / I to R", ""]
    tables += md_table(["Site", "n", "I CIP", "I CRO", "I CAZ", "CAZ-I isolates: CRO-R / CRO-S", "Prev CIP (I to S / I to R)", "Prev CRO", "Prev CAZ"], rows)
    rows = []
    for y, r in x10b["driams_a_per_year"].items():
        rows.append([y, r["n"], r["n_I_CAZ"], f"{r['prev_CAZ_I_to_S']:.3f} / {r['prev_CAZ_I_to_R']:.3f}"] +
                    [f"{r['phi'][k]['I_to_S']:.3f} / {r['phi'][k]['I_to_R']:.3f}" for k in ["CIP-CRO", "CIP-CAZ", "CRO-CAZ"]])
    tables += ["", "##. DRIAMS-A per year: CAZ I results and phi under I to S / I to R", ""]
    tables += md_table(["Year", "n", "I CAZ", "Prev CAZ (I to S / I to R)", "phi CIP-CRO", "phi CIP-CAZ", "phi CRO-CAZ"], rows)

    x10c = {}
    panels = {}
    for s, e in ecoli.items():
        panels[s] = e
    pooled_frames = []
    for s, e in ecoli.items():
        f = e.rename(columns=RENAME_POOL)
        f = f.loc[:, ~f.columns.duplicated()]
        pooled_frames.append(f)
    panels["pooled"] = pd.concat(pooled_frames, ignore_index=True)
    for s, e in panels.items():
        M = panel_matrix(e)
        P, N, pairs = pairwise_phi(M)
        order = cluster_order(P)
        prev = {c: {"n": int(M[c].notna().sum()), "n_R": int((M[c] == 1).sum()), "prev": float(M[c].mean())} for c in M.columns}
        top = sorted([p for p in pairs if not np.isnan(p["phi"])], key=lambda p: -p["phi"])[:15]
        # class block means
        cls = {c: drug_class(c) for c in M.columns}
        within, between = [], []
        block = {}
        for p in pairs:
            if np.isnan(p["phi"]):
                continue
            c1, c2 = cls[p["drug_1"]], cls[p["drug_2"]]
            (within if c1 == c2 else between).append(p["phi"])
            key = " vs ".join(sorted([c1, c2]))
            block.setdefault(key, []).append(p["phi"])
        # beta-lactam super-block
        bl = {"Penicillin/BLI", "Cephalosporin", "Carbapenem", "Monobactam"}
        bl_within = [p["phi"] for p in pairs if not np.isnan(p["phi"]) and cls[p["drug_1"]] in bl and cls[p["drug_2"]] in bl]
        # chaining candidates: strong phi and one member clearly more prevalent
        cands = []
        for p in pairs:
            if np.isnan(p["phi"]) or p["phi"] < 0.5:
                continue
            hi, lo = (p["drug_1"], p["drug_2"]) if p["prev_1"] >= p["prev_2"] else (p["drug_2"], p["drug_1"])
            ph_hi, ph_lo = max(p["prev_1"], p["prev_2"]), min(p["prev_1"], p["prev_2"])
            ratio = ph_hi / ph_lo if ph_lo > 0 else np.inf
            if ratio >= 1.5:
                cands.append({"upstream_candidate": hi, "downstream": lo, "phi": p["phi"], "n": p["n"],
                              "prev_upstream": ph_hi, "prev_downstream": ph_lo, "prev_ratio": ratio,
                              "phi_max": phi_max(ph_hi, ph_lo)})
        cands = sorted(cands, key=lambda c: -c["phi"])
        x10c[s] = {"n_ecoli_rows": int(len(e)), "drugs": list(M.columns), "n_drugs": int(M.shape[1]),
                   "drug_summary": prev, "cluster_order": order, "phi_matrix": P.loc[order, order].round(4).to_dict(),
                   "n_matrix": N.loc[order, order].to_dict(), "top15": top,
                   "class_means": {"within_class": float(np.mean(within)) if within else np.nan,
                                   "between_class": float(np.mean(between)) if between else np.nan,
                                   "beta_lactam_superblock": float(np.mean(bl_within)) if bl_within else np.nan,
                                   "blocks": {k: {"mean_phi": float(np.mean(v)), "n_pairs": len(v)} for k, v in sorted(block.items())}},
                   "chaining_candidates": cands}
        heatmap_phi(P, N, order, f"{SITE_LABEL[s]}: E. coli, {M.shape[1]} drugs with >= {MIN_RS} R and >= {MIN_RS} S "
                                 f"(n rows = {len(e)})", f"driams_full_panel_phi_{s}", plt)
        print(f"{s}: {len(e)} E. coli rows, {M.shape[1]} drugs, within-class mean phi "
              f"{x10c[s]['class_means']['within_class']:.3f}, between {x10c[s]['class_means']['between_class']:.3f}")
    res["X10c"] = x10c
    for s in ["driams_a", "driams_b", "driams_c", "driams_d", "pooled"]:
        r = x10c[s]
        tables += ["", f"##. {SITE_LABEL[s]}: top 15 drug pairs by phi (E. coli rows = {r['n_ecoli_rows']}, "
                       f"{r['n_drugs']} drugs with >= {MIN_RS} R and >= {MIN_RS} S)", ""]
        tables += md_table(["Rank", "Drug 1", "Drug 2", "phi", "n (pairwise complete)", "prev 1", "prev 2"],
                           [[i + 1, p["drug_1"], p["drug_2"], f3(p["phi"]), p["n"], f3(p["prev_1"]), f3(p["prev_2"])]
                            for i, p in enumerate(r["top15"])])
        tables += ["", "Drugs included: " + ", ".join(f"{d} (R {r['drug_summary'][d]['n_R']}/{r['drug_summary'][d]['n']})"
                                                     for d in r["cluster_order"]), "",
                   f"Mean phi within class {f3(r['class_means']['within_class'])}, between classes "
                   f"{f3(r['class_means']['between_class'])}, within the beta-lactam super-block "
                   f"{f3(r['class_means']['beta_lactam_superblock'])}.", ""]
        tables += md_table(["Class block", "mean phi", "n pairs"],
                           [[k, f3(v["mean_phi"]), v["n_pairs"]] for k, v in r["class_means"]["blocks"].items()])
        tables += ["", f"Chaining candidates at {SITE_LABEL[s]} (phi >= 0.5 and prevalence ratio >= 1.5):", ""]
        tables += md_table(["Upstream (more prevalent)", "Downstream", "phi", "n", "prev upstream", "prev downstream", "ratio", "phi_max"],
                           [[c["upstream_candidate"], c["downstream"], f3(c["phi"]), c["n"], f3(c["prev_upstream"]),
                             f3(c["prev_downstream"]), f"{c['prev_ratio']:.2f}", f3(c["phi_max"])] for c in r["chaining_candidates"]] or
                           [["none", "", "", "", "", "", "", ""]])

    instr = mloc["Spectrometer"].astype(int).values
    assert sorted(set(instr)) == [1, 2]
    n_i = {k: int((instr == k).sum()) for k in (1, 2)}
    assert n_i == {1: 67, 2: 120}, n_i
    labels = list(Y.columns)
    x13 = {"n_per_instrument": n_i, "prevalence": {}, "instrument_indicator_auroc": {}}
    rows = []
    for d in labels:
        y = Y[d].values
        r1, r2 = int(y[instr == 1].sum()), int(y[instr == 2].sum())
        tab = [[r1, n_i[1] - r1], [r2, n_i[2] - r2]]
        p = float(fisher_exact(tab)[1])
        auc_raw = float(roc_auc_score(y, (instr == 2).astype(float)))
        x13["prevalence"][d] = {"instr1": {"n_R": r1, "frac": r1 / n_i[1]}, "instr2": {"n_R": r2, "frac": r2 / n_i[2]},
                                "fisher_p": p, "OR_instr1_vs_instr2": list(odds_ratio(r1, n_i[1] - r1, r2, n_i[2] - r2))}
        x13["instrument_indicator_auroc"][d] = {"auroc_instr2_as_score": auc_raw, "auroc_folded": max(auc_raw, 1 - auc_raw),
                                                "resistance_enriched_on": 2 if auc_raw >= 0.5 else 1}
        rows.append([d, f"{r1}/{n_i[1]} ({100 * r1 / n_i[1]:.1f}%)", f"{r2}/{n_i[2]} ({100 * r2 / n_i[2]:.1f}%)", fp(p),
                     f3(auc_raw), f3(max(auc_raw, 1 - auc_raw))])
    tables += ["", "##. Sant'Orsola resistance prevalence per instrument (Fisher exact p) and AUROC of the instrument indicator alone", "",
               "AUROC (instrument 2 as score): values above 0.5 mean resistance is enriched on instrument 2, below 0.5 on instrument 1; "
               "folded = max(AUROC, 1 - AUROC) is the discrimination obtainable from the instrument alone.", ""]
    tables += md_table(["Drug", "Instrument 1 R", "Instrument 2 R", "Fisher p", "AUROC (instr 2 as score)", "AUROC folded"], rows)
    # profile distribution per instrument
    prof_i = {}
    for k in (1, 2):
        Yk = Y[instr == k]
        prof_i[k] = {"".join("R" if v else "S" for v in combo): int(((Yk["CIP"] == combo[0]) & (Yk["CTX"] == combo[1]) & (Yk["CAZ"] == combo[2])).sum())
                     for combo in product([0, 1], repeat=3)}
    x13["profiles_per_instrument"] = prof_i

    # out-of-fold predictions mapped to isolates
    cv = load_cv_results(ROOT / "results/s_orsola/logistic/cv_results.pkl")
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    folds = {(g // 5, g % 5): te for g, (tr, te) in enumerate(outer.split(X, strat))}
    for fr in cv.fold_results:
        te = folds[(fr.repeat_idx, fr.fold_idx)]
        for d in labels:
            assert np.array_equal(Y.iloc[te][d].values, np.asarray(fr.y_true[d])), "fold mismatch"
    print("Outer folds reconstructed and verified for every stored fold result")
    methods = {"BR": "binary_relevance", "Ens": "ensemble"}
    prob = {m: {d: np.full((3, len(Y)), np.nan) for d in labels} for m in methods}
    pred = {m: {d: np.full((3, len(Y)), np.nan) for d in labels} for m in methods}
    for fr in cv.fold_results:
        if fr.method not in methods.values():
            continue
        m = [k for k, v in methods.items() if v == fr.method][0]
        te = folds[(fr.repeat_idx, fr.fold_idx)]
        for d in labels:
            prob[m][d][fr.repeat_idx, te] = np.asarray(fr.y_prob[d])
            pred[m][d][fr.repeat_idx, te] = np.asarray(fr.y_pred[d])
    for m in methods:
        for d in labels:
            assert not np.isnan(prob[m][d]).any()

    def metr(y, p, yh):
        return (float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
                float(matthews_corrcoef(y, yh.astype(int))))

    rng = np.random.default_rng(SEED)
    x13["per_instrument_performance"] = {}
    rows = []
    for d in labels:
        y = Y[d].values
        x13["per_instrument_performance"][d] = {}
        for grp, mask in [("all", np.ones(len(y), bool)), ("instr1", instr == 1), ("instr2", instr == 2)]:
            rec = {"n": int(mask.sum()), "n_R": int(y[mask].sum())}
            for m in methods:
                per_rep = [metr(y[mask], prob[m][d][r, mask], pred[m][d][r, mask]) for r in range(3)]
                pooled = metr(np.tile(y[mask], 3), prob[m][d][:, mask].ravel(), pred[m][d][:, mask].ravel())
                rec[m] = {"AUROC_mean_over_repeats": float(np.nanmean([x[0] for x in per_rep])),
                          "AUROC_per_repeat": [x[0] for x in per_rep],
                          "MCC_mean_over_repeats": float(np.mean([x[1] for x in per_rep])),
                          "MCC_per_repeat": [x[1] for x in per_rep],
                          "AUROC_pooled": pooled[0], "MCC_pooled": pooled[1]}
            # chain gain: isolate-level cluster bootstrap on the pooled repeats
            idx = np.where(mask)[0]
            d_auc, d_mcc = [], []
            for _ in range(N_BOOT):
                b = rng.choice(idx, size=len(idx), replace=True)
                yb = np.tile(y[b], 3)
                if len(np.unique(yb)) < 2:
                    continue
                a_e = roc_auc_score(yb, prob["Ens"][d][:, b].ravel())
                a_b = roc_auc_score(yb, prob["BR"][d][:, b].ravel())
                m_e = matthews_corrcoef(yb, pred["Ens"][d][:, b].ravel().astype(int))
                m_b = matthews_corrcoef(yb, pred["BR"][d][:, b].ravel().astype(int))
                d_auc.append(a_e - a_b)
                d_mcc.append(m_e - m_b)
            # per-fold within-group deltas with exact sign-flip test
            fd_auc, fd_mcc = [], []
            for (r, f), te in sorted(folds.items()):
                sub = te[mask[te]]
                ys = y[sub]
                if len(sub) < 4 or len(np.unique(ys)) < 2:
                    continue
                fd_auc.append(roc_auc_score(ys, prob["Ens"][d][r, sub]) - roc_auc_score(ys, prob["BR"][d][r, sub]))
                fd_mcc.append(matthews_corrcoef(ys, pred["Ens"][d][r, sub].astype(int)) -
                              matthews_corrcoef(ys, pred["BR"][d][r, sub].astype(int)))
            rec["chain_gain"] = {
                "dAUROC_mean_over_repeats": rec["Ens"]["AUROC_mean_over_repeats"] - rec["BR"]["AUROC_mean_over_repeats"],
                "dMCC_mean_over_repeats": rec["Ens"]["MCC_mean_over_repeats"] - rec["BR"]["MCC_mean_over_repeats"],
                "dAUROC_pooled": rec["Ens"]["AUROC_pooled"] - rec["BR"]["AUROC_pooled"],
                "dMCC_pooled": rec["Ens"]["MCC_pooled"] - rec["BR"]["MCC_pooled"],
                "dAUROC_boot_ci95": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))],
                "dMCC_boot_ci95": [float(np.percentile(d_mcc, 2.5)), float(np.percentile(d_mcc, 97.5))],
                "per_fold_dAUROC_mean": float(np.mean(fd_auc)), "per_fold_dAUROC_signflip_p": sign_flip_p(fd_auc),
                "per_fold_dMCC_mean": float(np.mean(fd_mcc)), "per_fold_dMCC_signflip_p": sign_flip_p(fd_mcc),
                "n_folds_used": len(fd_auc)}
            x13["per_instrument_performance"][d][grp] = rec
            g = rec["chain_gain"]
            rows.append([d, grp, rec["n"], rec["n_R"], f3(rec["BR"]["AUROC_mean_over_repeats"]), f3(rec["Ens"]["AUROC_mean_over_repeats"]),
                         f"{g['dAUROC_mean_over_repeats']:+.3f} [{g['dAUROC_boot_ci95'][0]:+.3f}, {g['dAUROC_boot_ci95'][1]:+.3f}] p={fp(g['per_fold_dAUROC_signflip_p'])}",
                         f3(rec["BR"]["MCC_mean_over_repeats"]), f3(rec["Ens"]["MCC_mean_over_repeats"]),
                         f"{g['dMCC_mean_over_repeats']:+.3f} [{g['dMCC_boot_ci95'][0]:+.3f}, {g['dMCC_boot_ci95'][1]:+.3f}] p={fp(g['per_fold_dMCC_signflip_p'])}"])
    tables += ["", "##. Per-instrument AUROC and MCC of binary relevance (BR) and chain ensemble (Ens), logistic regression out-of-fold predictions", "",
               "Values are means over the three CV repeats of the metric computed on the pooled out-of-fold predictions of the group; "
               "delta = Ens minus BR with an isolate-level cluster bootstrap 95% interval (2000 resamples) and the exact sign-flip p over the per-fold within-group differences.", ""]
    tables += md_table(["Drug", "Group", "n", "n R", "BR AUROC", "Ens AUROC", "dAUROC [CI] p", "BR MCC", "Ens MCC", "dMCC [CI] p"], rows)

    # spectral batch check
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))])
    grid = {"clf__C": [0.001, 0.01, 0.1, 1.0]}
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    outer_i = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    yi = (instr == 2).astype(int)
    Xa = np.asarray(X, float)
    fold_auc, fold_bacc, chosen = [], [], []
    oof = np.full((3, len(yi)), np.nan)
    for g, (tr, te) in enumerate(outer_i.split(Xa, yi)):
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=N_JOBS)
        gs.fit(Xa[tr], yi[tr])
        p = gs.predict_proba(Xa[te])[:, 1]
        oof[g // 5, te] = p
        fold_auc.append(float(roc_auc_score(yi[te], p)))
        fold_bacc.append(float(balanced_accuracy_score(yi[te], (p >= 0.5).astype(int))))
        chosen.append(float(gs.best_params_["clf__C"]))
    rep_auc = [float(roc_auc_score(yi, oof[r])) for r in range(3)]
    x13["spectral_instrument_classifier"] = {
        "model": "StandardScaler + L2 logistic regression (class_weight balanced), C in {0.001, 0.01, 0.1, 1} by inner 3-fold grid search",
        "cv": "RepeatedStratifiedKFold(5, 3, random_state=42)", "fold_AUROC": fold_auc,
        "AUROC_mean": float(np.mean(fold_auc)), "AUROC_sd": float(np.std(fold_auc, ddof=1)),
        "AUROC_min": float(np.min(fold_auc)), "AUROC_max": float(np.max(fold_auc)),
        "balanced_accuracy_mean": float(np.mean(fold_bacc)), "pooled_AUROC_per_repeat": rep_auc, "chosen_C": chosen}
    print(f"Instrument classifier AUROC mean {np.mean(fold_auc):.3f} (sd {np.std(fold_auc, ddof=1):.3f}, "
          f"min {np.min(fold_auc):.3f}, max {np.max(fold_auc):.3f}), balanced accuracy {np.mean(fold_bacc):.3f}")
    tables += ["", "##. Spectral batch check: logistic regression predicting the instrument from the 6000 bins (5-fold CV x 3 repeats)", ""]
    tables += md_table(["AUROC mean (sd)", "min", "max", "Balanced accuracy", "Pooled AUROC per repeat"],
                       [[f"{np.mean(fold_auc):.3f} ({np.std(fold_auc, ddof=1):.3f})", f3(np.min(fold_auc)), f3(np.max(fold_auc)),
                         f3(np.mean(fold_bacc)), ", ".join(f3(a) for a in rep_auc)]])
    res["X13"] = x13

    (ROOT / "results/stats").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/stats/coresistance_structure.json").write_text(json.dumps(clean(res), indent=1))
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables/coresistance_structure.md").write_text("\n".join(tables) + "\n")
    print("\n".join(tables))

    plot_from_results(clean(res))
    print("saved figures, JSON and tables")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Co-resistance structure and instrument batch check")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figures from results/stats/coresistance_structure.json without recomputing anything")
    if ap.parse_args().plot_only:
        plot_from_results(json.load(open(ROOT / "results/stats/coresistance_structure.json")))
    else:
        main()
