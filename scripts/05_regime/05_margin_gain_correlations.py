"""Margin-gain correlations along the learning curves: pooled, within each cohort and partial.

Reads results/stats/cross_prediction_lc.json (scripts/05_regime/02_cross_prediction_lc.py: for each size-by-label point, the proxy margin, the stacking and
hard-chain gains over binary relevance and the AUROC of the label's own classifier). Over the 18 subsample points it reports
the Spearman correlation of the margin with each gain, pooled and within each cohort; the correlation of each gain with the
headroom alone (minus the own AUROC); and the partial Spearman correlation of the margin with each gain controlling for the
own AUROC (Pearson partial correlation on ranks, t test with n - 3 degrees of freedom). The points are nested subsamples, so
the p-values are descriptive.
Outputs: results/stats/margin_gain_correlations.json, tables/margin_gain_correlations.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
# --grouped: points from the patient-grouped DRIAMS-A learning curves (scripts/05_regime/02_cross_prediction_lc.py --sites driams_a_grouped,s_orsola);
# outputs get the suffix _grouped (the DRIAMS-A analyses of the paper)
GROUPED = "--grouped" in sys.argv
SFX = "_grouped" if GROUPED else ""
COHORTS = {"pooled": None, "DRIAMS-A": "driams_a_grouped" if GROUPED else "driams_a", "Sant'Orsola": "s_orsola"}
GAINS = {"stacking": "gain_stack", "chain (hard)": "gain_hard"}


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((rx, ry), (rx, rz), (ry, rz)))
    r = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)); df = len(x) - 3
    return float(r), float(2 * t.sf(abs(r) * np.sqrt(df / (1 - r ** 2)), df))


def main():
    rows = [r for r in json.load(open(ROOT / f"results/stats/cross_prediction_lc{SFX}.json"))["rows"] if isinstance(r.get("gain_hard"), dict)]
    out, lines = [], ["# Margin-gain correlations along the learning curves (18 subsample points; scripts/05_regime/02_cross_prediction_lc.py)", "",
                      "Spearman rho (p). Headroom alone: correlation of the gain with minus the own-classifier AUROC. Partial: margin with gain, controlling for the own-classifier AUROC. Nested subsamples: p-values are descriptive.", "",
                      "| Cohort | Points | Gain | Margin vs gain | Headroom vs gain | Margin vs gain, own AUROC fixed (partial) |", "|---|---|---|---|---|---|"]
    for cname, site in COHORTS.items():
        rr = [r for r in rows if site is None or r["site"] == site]
        if len(rr) < 4: print("too few points", cname); continue
        margin = np.array([r["margin"]["delta"] for r in rr]); own = np.array([r["own"] for r in rr])
        for gname, key in GAINS.items():
            g = np.array([r[key]["delta"] for r in rr])
            rho, p = spearmanr(margin, g); rh, ph = spearmanr(-own, g); rp, pp = partial_spearman(margin, g, own)
            row = {"cohort": cname, "n": len(rr), "gain": gname, "rho_margin": [float(rho), float(p)], "rho_headroom": [float(rh), float(ph)], "partial_rho_margin": [rp, pp]}
            out.append(row)
            fp = lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"
            lines.append(f"| {cname} | {len(rr)} | {gname} | {rho:+.2f} ({fp(p)}) | {rh:+.2f} ({fp(ph)}) | {rp:+.2f} ({fp(pp)}) |")
    (ROOT / f"results/stats/margin_gain_correlations{SFX}.json").write_text(json.dumps(out, indent=1))
    (ROOT / f"tables/margin_gain_correlations{SFX}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
