"""Does chain benefit scale with co-resistance strength? A dependence-aware test.

The within-site dose-response (Fig. 6) is fragile: phi_max takes only two distinct
values across the 9 within-site points (CIP 0.67; CTX=CAZ 0.894), so a 9-point
Spearman/OLS is pseudoreplication -- effectively a low-phi-vs-high-phi contrast.

This script tests the graded relationship where phi genuinely varies: the 60
cross-site evaluations (20 train->test pairs x 3 drugs), with phi_max taken at the
*test* site (0.31 to 1.00). Because these 60 points are not independent (shared
train site, shared test site, shared drug), we fit a linear mixed-effects model

    Delta_metric ~ phi_max + (random intercepts),

with crossed random intercepts for train site, test site and drug (via variance
components). The fixed-effect slope on phi_max, its 95% CI and p-value are the
dependence-aware analogue of the OLS slope. We report the naive Spearman/OLS
alongside for transparency, and also a within-site two-group contrast (low-phi
CIP vs high-phi ESBL drugs) instead of the 9-point regression.

Outputs:
  results/stats/cross_site_mixedeffects.json
  figures/supplementary/cross_site_phi_benefit.png (+pdf)
Run with --plot-only to redraw the figure from the stored results without refitting.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import linregress, mannwhitneyu, spearmanr

matplotlib.use("Agg")
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D

DRUG_COLORS = {"CIP": "#4C72B0", "CEPH": "#C44E52", "CAZ": "#DD8452"}
DRUG_LABEL = {"CIP": "CIP", "CEPH": "CTX/CRO", "CAZ": "CAZ"}

from mecor.cvio import load_cv_results, metric_by_fold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
SITES = ["s_orsola", "driams_a", "driams_b", "driams_c", "driams_d"]
SITE_FROM_CROSS = {"local": "s_orsola", "driams_a": "driams_a", "driams_b": "driams_b",
                   "driams_c": "driams_c", "driams_d": "driams_d"}
MODELS = ["logistic", "catboost", "tabpfn"]


def load_phi():
    out = {}
    for s in SITES:
        pm = json.loads((ROOT / f"results/cooccurrence/{s}/cooccurrence_results.json").read_text())["phi_matrix"]
        ceph = "CTX" if "CTX" in pm else "CRO"
        canon = {"CIP": "CIP", ceph: "CEPH", "CAZ": "CAZ"}
        out[s] = {canon[r]: max(pm[r][o] for o in pm[r] if o != r) for r in canon}
    return out


def cross_frame(phi):
    cd = json.loads((ROOT / "results/cross_dataset/logistic/cross_dataset_results.json").read_text())
    rows = []
    for e in cd:
        train, test = SITE_FROM_CROSS[e["train"]], SITE_FROM_CROSS[e["test"]]
        br, en = e["methods"]["binary_relevance"], e["methods"]["ensemble"]
        for d in br:
            canon = "CEPH" if d in ("CRO", "CTX") else d
            rows.append(dict(train=train, test=test, drug=canon,
                             phi=phi[test][canon],
                             dAUROC=en[d]["AUROC"] - br[d]["AUROC"],
                             dMCC=en[d]["MCC"] - br[d]["MCC"]))
    return pd.DataFrame(rows)


def within_frame(phi):
    rows = []
    for model in MODELS:
        cv = load_cv_results(ROOT / f"results/s_orsola/{model}/cv_results.pkl")
        for raw, canon in (("CIP", "CIP"), ("CTX", "CEPH"), ("CAZ", "CAZ")):
            rec = dict(model=model, drug=canon, phi=phi["s_orsola"][canon])
            for metric, key in (("AUROC", "dAUROC"), ("MCC", "dMCC")):
                br = np.mean(list(metric_by_fold(cv, "binary_relevance", raw, metric).values()))
                en = np.mean(list(metric_by_fold(cv, "ensemble", raw, metric).values()))
                rec[key] = en - br
            rows.append(rec)
    return pd.DataFrame(rows)


def _fit_one(df, ycol, groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = smf.mixedlm(f"{ycol} ~ phi", df, groups=df[groups]).fit(reml=True, method="lbfgs")
    beta = float(r.fe_params["phi"]); se = float(r.bse["phi"])
    return {"slope": beta, "se": se, "ci95": [beta - 1.96 * se, beta + 1.96 * se],
            "p": float(r.pvalues["phi"]), "converged": bool(r.converged),
            "groups": groups, "n_groups": int(df[groups].nunique())}


def fit_mixed(df, ycol):
    """Random-intercept mixed models. Primary: test-site clustering (the main
    source of non-independence -- spectral drift/prevalence). Sensitivity: train
    site and drug. We do not add drug as a *fixed* effect because phi is nearly
    collinear with drug (CIP is always low-phi), which would absorb the phi term."""
    primary = _fit_one(df, ycol, "test")
    sens = {g: _fit_one(df, ycol, g) for g in ("train", "drug")}
    return {"primary_test_site_RE": primary, "sensitivity": sens}


def naive(df, ycol):
    rho, p = spearmanr(df["phi"], df[ycol])
    lr = linregress(df["phi"], df[ycol])
    return {"spearman_rho": float(rho), "spearman_p": float(p),
            "ols_slope": float(lr.slope),
            "ols_slope_ci95": [float(lr.slope - 1.96 * lr.stderr),
                               float(lr.slope + 1.96 * lr.stderr)],
            "n": int(len(df))}


def within_contrast(dfw, ycol):
    """Two-group contrast: low-phi CIP vs high-phi ESBL drugs (per model means)."""
    low = dfw[dfw["drug"] == "CIP"][ycol].values
    high = dfw[dfw["drug"].isin(["CEPH", "CAZ"])][ycol].values
    u, p = mannwhitneyu(high, low, alternative="greater")
    return {"low_phi_CIP_mean": float(low.mean()), "n_low": int(len(low)),
            "high_phi_ESBL_mean": float(high.mean()), "n_high": int(len(high)),
            "diff": float(high.mean() - low.mean()),
            "mannwhitney_u": float(u), "mannwhitney_p_one_sided": float(p),
            "n_distinct_phi_within": int(dfw["phi"].nunique())}


def make_figure(dfc, stats, dest_stem):
    """Chain benefit against the test-site co-resistance for the 60 cross-site evaluations: points by drug (ps.DRUG),
    mixed-effects line (slope from the stored stats through the centroid), zero line, one legend above; no in-panel stats
    (the caption carries them). Saves <dest_stem>.png and .pdf."""
    from mecor.visualization import paper_style as ps
    col = {"CIP": ps.DRUG["CIP"], "CEPH": ps.DRUG["CTX"], "CAZ": ps.DRUG["CAZ"]}
    fig, axes = ps.figure("double", height=2.8, ncols=2, constrained_layout=True)
    panels = [("dAUROC", "AUROC"), ("dMCC", "MCC")]
    for ax, (ycol, key), letter in zip(axes, panels, "ab"):
        ps.hline(ax, 0.0)
        for drug in ("CIP", "CEPH", "CAZ"):
            sub = dfc[dfc["drug"] == drug]
            ax.plot(sub["phi"], sub[ycol], ls="none", marker="o", ms=4, color=col[drug], mec=ps.SURFACE, mew=0.4, alpha=0.9, zorder=3)
        mx = stats["cross_site"][key]["mixed"]["primary_test_site_RE"]
        xs = np.linspace(dfc["phi"].min(), dfc["phi"].max(), 50)
        xbar, ybar = dfc["phi"].mean(), dfc[ycol].mean()
        ax.plot(xs, ybar + mx["slope"] * (xs - xbar), color=ps.INK, lw=1.2, zorder=2)
        ax.set_xlabel("Co-resistance $\\varphi$ at the test site")
        ax.set_ylabel(f"Chain ensemble minus binary relevance ({key})")
        ax.margins(y=0.1); ps.despine(ax); ps.panel_label(ax, letter)
    handles = [Line2D([0], [0], marker="o", ls="", ms=4, color=col[d], label=DRUG_LABEL[d]) for d in ("CIP", "CEPH", "CAZ")]
    handles.append(Line2D([0], [0], ls="-", lw=1.2, color=ps.INK, label="Mixed-effects fit (random intercept per test site)"))
    ps.legend_above(fig, handles, [h.get_label() for h in handles], ncol=4, y=1.0)
    out = ps.save(fig, dest_stem)
    print(f"saved {out}.png")


def plot_from_results(stats_path=ROOT / "results/stats/cross_site_mixedeffects.json",
                      dest_stem=ROOT / "figures/supplementary/cross_site_phi_benefit"):
    """Redraw the figure from stored results only (cross-dataset results, co-occurrence phi and the saved mixed-model stats)."""
    stats = json.loads(Path(stats_path).read_text())
    make_figure(cross_frame(load_phi()), stats, dest_stem)


def main():
    phi = load_phi()
    dfc = cross_frame(phi)
    dfw = within_frame(phi)

    out = {
        "within_site": {
            "note": "9 points, only %d distinct phi values -> reported as a 2-group "
                    "contrast, not a 9-point regression." % dfw["phi"].nunique(),
            "AUROC_contrast": within_contrast(dfw, "dAUROC"),
            "MCC_contrast": within_contrast(dfw, "dMCC"),
        },
        "cross_site": {
            "note": "60 evaluations (20 pairs x 3 drugs); phi varies 0.31-1.00. "
                    "Mixed-effects slope on phi is the dependence-aware test.",
            "phi_range": [float(dfc["phi"].min()), float(dfc["phi"].max())],
            "AUROC": {"naive": naive(dfc, "dAUROC"), "mixed": fit_mixed(dfc, "dAUROC")},
            "MCC": {"naive": naive(dfc, "dMCC"), "mixed": fit_mixed(dfc, "dMCC")},
        },
    }

    dest = ROOT / "results/stats/cross_site_mixedeffects.json"
    dest.write_text(json.dumps(out, indent=2))
    make_figure(dfc, out, ROOT / "figures/supplementary/cross_site_phi_benefit")

    print("WITHIN-SITE (reframed as 2-group contrast; only %d distinct phi):"
          % dfw["phi"].nunique())
    for met in ("AUROC", "MCC"):
        c = out["within_site"][f"{met}_contrast"]
        print(f"  d{met}: low-phi CIP mean={c['low_phi_CIP_mean']:+.3f} vs "
              f"high-phi ESBL mean={c['high_phi_ESBL_mean']:+.3f} "
              f"(diff {c['diff']:+.3f}, Mann-Whitney p={c['mannwhitney_p_one_sided']:.3f})")
    print("\nCROSS-SITE (phi varies 0.31-1.00):")
    for met in ("AUROC", "MCC"):
        nv = out["cross_site"][met]["naive"]; mx = out["cross_site"][met]["mixed"]["primary_test_site_RE"]
        print(f"  d{met} naive: Spearman rho={nv['spearman_rho']:+.2f} p={nv['spearman_p']:.3f}; "
              f"OLS slope={nv['ols_slope']:+.3f} {[round(v,3) for v in nv['ols_slope_ci95']]}")
        print(f"  d{met} mixed [test-site RE, {mx['n_groups']} groups]: slope={mx['slope']:+.3f} "
              f"CI95={[round(v,3) for v in mx['ci95']]} p={mx['p']:.3f} converged={mx['converged']}")
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv[1:]:
        plot_from_results()
    else:
        main()
