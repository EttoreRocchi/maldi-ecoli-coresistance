"""Within-site replication of the chain effect across sites (logistic regression nested CV).

Question: does the within-site chain benefit observed at Sant'Orsola replicate at the DRIAMS sites, and is the
site-level pattern explained by the refined mechanism (co-resistance of the upstream label x its standalone
predictive accuracy x headroom of the downstream label)?

For every site with a within-site run (results/<site>/logistic/cv_results.pkl: s_orsola, driams_b, driams_c,
driams_d, driams_a when available) the script reports per drug: BR, chain ensemble, best ordering (selected by
MCC), deltas with BCa intervals and exact sign-flip p-values over the 15 folds, VME and ME, together with the
site's phi coefficients (results/cooccurrence), prevalences and n. A site-level table relates the cephalosporin
ensemble gain to phi(CIP, cephalosporin), the CIP standalone AUROC and the cephalosporin BR AUROC.

Outputs: results/stats/driams_replication.json, tables/driams_replication.md,
         figures/supplementary/driams_replication.png (+pdf)
Run with --plot-only to redraw the figure from results/stats/driams_replication.json without recomputing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor import shared as la
from mecor.cvio import aligned, bca_mean_ci, chain_methods, load_cv_results, metric_by_fold  # noqa: E402

SITES = [("s_orsola", "Sant'Orsola"), ("driams_b", "DRIAMS-B"), ("driams_c", "DRIAMS-C"), ("driams_d", "DRIAMS-D"), ("driams_a", "DRIAMS-A")]
# --grouped: DRIAMS-A from the patient-grouped within-site run; outputs get the suffix _grouped (the DRIAMS-A analyses of the paper)
GROUPED = "--grouped" in sys.argv
SFX = "_grouped" if GROUPED else ""
if GROUPED:
    SITES = [("driams_a_grouped", n) if k == "driams_a" else (k, n) for k, n in SITES]


def best_by_mcc(cv, drug):
    best, bm = None, -np.inf
    for m in chain_methods(cv):
        mu = float(np.mean(list(metric_by_fold(cv, m, drug, "MCC").values())))
        if mu > bm:
            best, bm = m, mu
    return best


def main():
    out = {"sites": {}}
    for key, name in SITES:
        pkl = ROOT / f"results/{key}/logistic/cv_results.pkl"
        if not pkl.exists():
            print(f"{name}: not available yet")
            continue
        cv = load_cv_results(pkl)
        co = json.load(open(ROOT / f"results/cooccurrence/{key.replace('_grouped', '')}/cooccurrence_results.json"))   # co-occurrence is descriptive, unchanged by grouping
        labels = list(cv.labels)
        ceph = "CTX" if "CTX" in labels else "CRO"
        phi = {}
        for pr in co.get("pairs", []):
            phi[tuple(sorted((pr["drug1"], pr["drug2"])))] = pr["phi"]
        if not phi:  # fall back on the phi matrix if present
            for a, row in co.get("phi_matrix", {}).items():
                for b, v in row.items():
                    if a != b:
                        phi[tuple(sorted((a, b)))] = v
        site = {"n": cv.n_samples, "labels": labels, "phi": {"+".join(k): v for k, v in phi.items()},
                "prevalence": co.get("prevalence", {}), "drugs": {}}
        for d in labels:
            rec = {}
            br = {m: metric_by_fold(cv, "binary_relevance", d, m) for m in ("AUROC", "MCC")}
            en = {m: metric_by_fold(cv, "ensemble", d, m) for m in ("AUROC", "MCC")}
            bc = best_by_mcc(cv, d)
            bs = {m: metric_by_fold(cv, bc, d, m) for m in ("AUROC", "MCC")}
            for m in ("AUROC", "MCC"):
                a, b = aligned(br[m], en[m])
                a2, b2 = aligned(br[m], bs[m])
                _, lo, hi = bca_mean_ci(b - a)
                _, lo2, hi2 = bca_mean_ci(b2 - a2)
                rec[m] = {"BR": float(a.mean()), "Ens": float(b.mean()), "Best": float(b2.mean()),
                          "dEns": float((b - a).mean()), "dEns_ci": [lo, hi], "dEns_p": la.sign_flip_p(b - a),
                          "dBest": float((b2 - a2).mean()), "dBest_ci": [lo2, hi2], "dBest_p_naive": la.sign_flip_p(b2 - a2)}
            for m in ("VME", "ME"):
                a, b = aligned(metric_by_fold(cv, "binary_relevance", d, m, clinical=True), metric_by_fold(cv, "ensemble", d, m, clinical=True))
                rec[m] = {"BR": float(a.mean()), "Ens": float(b.mean()), "dEns": float((b - a).mean())}
            rec["best_ordering"] = bc.replace("chain_", "").replace("_", ">")
            site["drugs"][d] = rec
        out["sites"][name] = site
        print(f"{name}: n={cv.n_samples} done")

    # site-level mechanism table
    rows = []
    for name, s in out["sites"].items():
        labels = s["labels"]
        ceph = "CTX" if "CTX" in labels else "CRO"
        cip_auc = s["drugs"]["CIP"]["AUROC"]["BR"]
        for d in (ceph, "CAZ"):
            rows.append({"site": name, "drug": d, "n": s["n"],
                         "phi_cip_drug": s["phi"].get("+".join(sorted(("CIP", d)))),
                         "phi_ceph_pair": s["phi"].get("+".join(sorted((ceph, "CAZ")))),
                         "cip_auroc": cip_auc, "drug_br_auroc": s["drugs"][d]["AUROC"]["BR"],
                         "headroom": 1 - s["drugs"][d]["AUROC"]["BR"],
                         "dEns_auroc": s["drugs"][d]["AUROC"]["dEns"], "dEns_auroc_p": s["drugs"][d]["AUROC"]["dEns_p"],
                         "dEns_mcc": s["drugs"][d]["MCC"]["dEns"], "dEns_mcc_p": s["drugs"][d]["MCC"]["dEns_p"]})
    out["site_level"] = rows
    (ROOT / f"results/stats/driams_replication{SFX}.json").write_text(json.dumps(out, indent=1))

    lines = ["# Within-site replication across sites (logistic regression, 5x3 repeated nested CV)", ""]
    for name, s in out["sites"].items():
        lines += [f"## {name} (n = {s['n']}; phi: " + ", ".join(f"{k} {v:.3f}" for k, v in s["phi"].items()) + ")", "",
                  "| Drug | BR AUROC | Ens AUROC | dAUROC [CI] p | BR MCC | Ens MCC | dMCC [CI] p | Best ordering | Best dMCC (naive p) | VME BR>Ens | ME BR>Ens |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for d, r in s["drugs"].items():
            A, M = r["AUROC"], r["MCC"]
            lines.append(f"| {d} | {A['BR']:.3f} | {A['Ens']:.3f} | {A['dEns']:+.3f} [{A['dEns_ci'][0]:+.3f}, {A['dEns_ci'][1]:+.3f}] p={A['dEns_p']:.3f} | {M['BR']:.3f} | {M['Ens']:.3f} | {M['dEns']:+.3f} [{M['dEns_ci'][0]:+.3f}, {M['dEns_ci'][1]:+.3f}] p={M['dEns_p']:.3f} | {r['best_ordering']} | {M['dBest']:+.3f} ({M['dBest_p_naive']:.3f}) | {r['VME']['BR']:.3f}>{r['VME']['Ens']:.3f} | {r['ME']['BR']:.3f}>{r['ME']['Ens']:.3f} |")
        lines.append("")
    lines += ["## Site-level mechanism table (cephalosporin labels)", "",
              "| Site | Drug | n | phi(CIP, drug) | phi(ceph pair) | CIP BR AUROC | drug BR AUROC | headroom | ensemble dAUROC (p) | ensemble dMCC (p) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['site']} | {r['drug']} | {r['n']} | {r['phi_cip_drug']:.3f} | {r['phi_ceph_pair']:.3f} | {r['cip_auroc']:.3f} | {r['drug_br_auroc']:.3f} | {r['headroom']:.3f} | {r['dEns_auroc']:+.3f} ({r['dEns_auroc_p']:.3f}) | {r['dEns_mcc']:+.3f} ({r['dEns_mcc_p']:.3f}) |")
    (ROOT / f"tables/driams_replication{SFX}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_from_results()


def plot_from_results(stats_path=ROOT / f"results/stats/driams_replication{SFX}.json", out_stem=ROOT / f"figures/supplementary/driams_replication{SFX}"):
    """Supplementary figure from the stored JSON only: chain ensemble minus BR per site (AUROC, MCC) as dot-and-interval
    (BCa 95%), drugs as hue (ps.DRUG), sites ordered by cohort size, zero line, one legend above."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.lines import Line2D

    from mecor.visualization import paper_style as ps
    out = json.load(open(stats_path))
    sites = list(out["sites"])
    keys = [("CIP", "CIP"), ("ceph", "CTX/CRO"), ("CAZ", "CAZ")]
    col = {"CIP": ps.DRUG["CIP"], "ceph": ps.DRUG["CTX"], "CAZ": ps.DRUG["CAZ"]}
    fig, axes = ps.figure("double", height=2.8, ncols=2, constrained_layout=True)
    for ax, metric, letter in zip(axes, ("AUROC", "MCC"), "ab"):
        ps.hline(ax, 0.0)
        for i, (dkey, _) in enumerate(keys):
            for j, name in enumerate(sites):
                s_ = out["sites"][name]
                d = ("CTX" if "CTX" in s_["labels"] else "CRO") if dkey == "ceph" else dkey
                r = s_["drugs"][d][metric]
                ps.ci_point(ax, j + (i - 1) * 0.22, r["dEns"], r["dEns_ci"][0], r["dEns_ci"][1], color=col[dkey])
        ax.set_xticks(range(len(sites))); ax.set_xticklabels([ps.SITE_NAME.get(n, n) if n != "Sant'Orsola" else ps.SITE_NAME["s_orsola"] for n in sites], rotation=30, ha="right")
        ax.set_xlim(-0.6, len(sites) - 0.4); ax.margins(y=0.1)
        ax.set_ylabel(f"Chain ensemble minus binary relevance ({metric})"); ps.despine(ax); ps.panel_label(ax, letter)
    handles = [Line2D([], [], ls="none", marker="o", ms=4.5, color=col[k], label=lab) for k, lab in keys]
    ps.legend_above(fig, handles, [h.get_label() for h in handles], ncol=3, y=1.0)
    return ps.save(fig, out_stem)


if __name__ == "__main__":
    if "--plot-only" in sys.argv[1:]:
        plot_from_results()
    else:
        main()
