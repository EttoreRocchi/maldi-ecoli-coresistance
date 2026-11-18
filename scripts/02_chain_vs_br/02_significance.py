#!/usr/bin/env python3
"""Step 7: Within-site significance and confidence intervals (Sant'Orsola).

Two complementary analyses of the saved nested-CV folds, for every model
(logistic, catboost, tabpfn) and drug (CIP, CTX, CAZ):

  ci          BCa bootstrap 95% confidence intervals (used for inference in the
              main text). For BR, the chain ensemble, and the best chain ordering
              it reports AUROC, MCC, clinical VME/ME, and the chain-minus-BR
              deltas. A delta CI excluding zero indicates a chain effect.
              Writes results/stats/within_site_ci.json.

  permutation Exact paired sign-flip permutation tests (reported in the Supplement).
              For each (model, label, metric) it enumerates all 2^15 = 32,768 sign
              assignments on the per-fold differences and reports: BR vs ensemble
              (main comparison), BR vs each of the 6 fixed orderings (Holm-corrected), and
              BR vs the best-of-6 ordering (max-statistic, selection-corrected).
              Writes tables/paired_permutation_*.{md,csv}.

Main comparison: chain ensemble vs BR for each drug, learner and metric (18 comparisons, adjusted
together in scripts/02_chain_vs_br/06_chain_vs_br_table.py); only the CTX gains survive the adjustment.

Usage:
    python scripts/02_chain_vs_br/02_significance.py                 # both analyses
    python scripts/02_chain_vs_br/02_significance.py --analysis ci
    python scripts/02_chain_vs_br/02_significance.py --analysis permutation
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from mecor.cvio import aligned, bca_mean_ci, best_chain, load_cv_results, metric_by_fold
from mecor.shared import best_chain_max_stat_pvalue, exact_sign_flip_pvalue, holm_adjust  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
MODELS = ["logistic", "catboost", "tabpfn"]
DRUGS = ["CIP", "CTX", "CAZ"]
METRICS = ["AUROC", "MCC", "F1 (weighted)", "Balanced Acc"]
CHAIN_ORDERINGS = [
    "chain_CIP_CTX_CAZ",
    "chain_CIP_CAZ_CTX",
    "chain_CTX_CIP_CAZ",
    "chain_CTX_CAZ_CIP",
    "chain_CAZ_CIP_CTX",
    "chain_CAZ_CTX_CIP",
]


def paired_diff(cv, method, label, metric, ref="binary_relevance"):
    """Per-fold (method - ref) differences, aligned on the shared folds."""
    a, b = aligned(metric_by_fold(cv, method, label, metric),
                   metric_by_fold(cv, ref, label, metric))
    return a - b


def mean_ci(vals):
    m, lo, hi = bca_mean_ci(vals)
    return {"mean": m, "ci95": [lo, hi]}


def delta_block(chain_vals, br_vals):
    a, b = aligned(chain_vals, br_vals)
    diffs = a - b
    m, lo, hi = bca_mean_ci(diffs)
    _, perm_p = exact_sign_flip_pvalue(diffs)
    return {"delta": m, "ci95": [lo, hi], "perm_p": perm_p, "n_folds": int(len(diffs))}


def summarize(cv, drug, metric):
    br = metric_by_fold(cv, "binary_relevance", drug, metric)
    ens = metric_by_fold(cv, "ensemble", drug, metric)
    bc_name = best_chain(cv, drug, metric)
    bc = metric_by_fold(cv, bc_name, drug, metric)
    return {
        "BR": mean_ci(list(br.values())),
        "Ens": mean_ci(list(ens.values())),
        "Best": mean_ci(list(bc.values())),
        "best_chain": bc_name,
        "delta_Ens": delta_block(ens, br),
        "delta_Best": delta_block(bc, br),
    }


def clinical(cv, drug):
    out = {}
    best = best_chain(cv, drug, "MCC")
    for meth, key in (("binary_relevance", "BR"), ("ensemble", "Ens"), (best, "Best")):
        out[key] = {
            "VME": mean_ci(list(metric_by_fold(cv, meth, drug, "VME", clinical=True).values())),
            "ME": mean_ci(list(metric_by_fold(cv, meth, drug, "ME", clinical=True).values())),
        }
    return out


def _fmt_ci(d):
    return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def _fmt_delta(d):
    return f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] (perm p={d['perm_p']:.3f})"


def run_confidence_intervals(results_dir, stats_dir):
    results = {}
    for model in MODELS:
        path = results_dir / model / "cv_results.pkl"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        cv = load_cv_results(path)
        results[model] = {"drugs": {}}
        for drug in DRUGS:
            results[model]["drugs"][drug] = {
                "AUROC": summarize(cv, drug, "AUROC"),
                "MCC": summarize(cv, drug, "MCC"),
                "clinical": clinical(cv, drug),
            }

    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "within_site_ci.json").write_text(json.dumps(results, indent=2))

    print("# Within-site nested-CV (BCa bootstrap 95% CI; permutation p for Supplement)\n")
    for model in results:
        print(f"## {model}")
        for drug in DRUGS:
            for metric in ("AUROC", "MCC"):
                s = results[model]["drugs"][drug][metric]
                print(f"- {drug} {metric}: BR {_fmt_ci(s['BR'])} | Ens {_fmt_ci(s['Ens'])} | "
                      f"Best {_fmt_ci(s['Best'])} [{s['best_chain']}]")
                print(f"    dEns {_fmt_delta(s['delta_Ens'])} | dBest {_fmt_delta(s['delta_Best'])}")
            c = results[model]["drugs"][drug]["clinical"]
            print(f"    VME BR {_fmt_ci(c['BR']['VME'])} Ens {_fmt_ci(c['Ens']['VME'])} Best {_fmt_ci(c['Best']['VME'])}")
            print(f"    ME  BR {_fmt_ci(c['BR']['ME'])} Ens {_fmt_ci(c['Ens']['ME'])} Best {_fmt_ci(c['Best']['ME'])}")
        print()
    print(f"saved {stats_dir / 'within_site_ci.json'}")


def df_to_markdown(df) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [head, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_permutation(results_dir, tables_dir):
    primary_rows = []       # BR vs Ensemble
    per_ordering_rows = []   # BR vs each fixed ordering
    best_chain_rows = []     # BR vs best-of-K chain (max-stat corrected)

    for model in MODELS:
        path = results_dir / model / "cv_results.pkl"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        cv = load_cv_results(path)
        labels = list(cv.labels)

        for label, metric in product(labels, METRICS):
            br = metric_by_fold(cv, "binary_relevance", label, metric)
            br_mean = float(np.mean(list(br.values())))

            # Primary: BR vs Ensemble.
            d = paired_diff(cv, "ensemble", label, metric)
            assert len(d) == 15, len(d)
            mean_d, p = exact_sign_flip_pvalue(d)
            ens_mean = br_mean + mean_d
            primary_rows.append({
                "model": model, "label": label, "metric": metric,
                "BR_mean": br_mean, "Ensemble_mean": ens_mean,
                "mean_diff (Ens-BR)": mean_d, "n_folds": len(d), "p_value": p,
            })

            # Secondary: BR vs each ordering (Holm-corrected within model/label/metric).
            ord_rows = []
            for ordering in CHAIN_ORDERINGS:
                d_ord = paired_diff(cv, ordering, label, metric)
                obs, p_raw = exact_sign_flip_pvalue(d_ord)
                ord_rows.append({
                    "model": model, "label": label, "metric": metric,
                    "ordering": ordering.replace("chain_", "").replace("_", "->"),
                    "Chain_mean": br_mean + obs, "mean_diff (Chain-BR)": obs, "p_raw": p_raw,
                })
            p_holm = holm_adjust([r["p_raw"] for r in ord_rows])
            for r, p_adj in zip(ord_rows, p_holm):
                r["p_holm"] = float(p_adj)
            per_ordering_rows.extend(ord_rows)

            # Tertiary: BR vs best-of-K chain (max-stat permutation).
            D = np.stack([paired_diff(cv, ordering, label, metric) for ordering in CHAIN_ORDERINGS])
            best_idx, obs_T, p_two, p_one = best_chain_max_stat_pvalue(D)
            best_ordering = CHAIN_ORDERINGS[best_idx].replace("chain_", "").replace("_", "->")
            _, p_naive = exact_sign_flip_pvalue(D[best_idx])
            best_chain_rows.append({
                "model": model, "label": label, "metric": metric,
                "BR_mean": br_mean, "BestChain_mean": br_mean + obs_T,
                "best_ordering": best_ordering, "mean_diff (Best-BR)": obs_T,
                "p_naive (biased)": p_naive, "p_maxstat": p_two, "p_maxstat_one_sided": p_one,
            })

    df_primary = pd.DataFrame(primary_rows)
    df_secondary = pd.DataFrame(per_ordering_rows)
    df_best = pd.DataFrame(best_chain_rows)

    tables_dir.mkdir(parents=True, exist_ok=True)
    out_md = tables_dir / "paired_permutation_br_vs_chain.md"
    with open(out_md, "w") as f:
        f.write("# Paired Permutation Test: BR vs Chain methods\n\n")
        f.write(
            "Per-fold metrics from 5x3 repeated stratified CV (n=15 paired folds). "
            "Two-sided p-value by exact enumeration of all 2^15 = 32,768 sign "
            "assignments on the per-fold differences (chain - BR). H0: differences "
            "are symmetrically distributed around 0.\n\n"
        )

        f.write("## Main comparison: BR vs Chain Ensemble (mean of 6 orderings)\n\n")
        f.write("Fixed comparison (the ensemble averages the six orderings); no post hoc selection of an ordering.\n\n")
        df_p = df_primary.copy()
        for col in ["BR_mean", "Ensemble_mean", "mean_diff (Ens-BR)"]:
            df_p[col] = df_p[col].map(lambda x: f"{x:+.4f}" if "diff" in col else f"{x:.4f}")
        df_p["p_value"] = df_p["p_value"].map(lambda x: f"{x:.4f}")
        f.write(df_to_markdown(df_p))
        f.write("\n\n")

        f.write("## Secondary: BR vs each fixed chain ordering\n\n")
        f.write(
            "p_raw is the raw permutation p-value; p_holm is Holm-adjusted across "
            "the 6 orderings within each (model, label, metric).\n\n"
        )
        df_s = df_secondary.copy()
        df_s["Chain_mean"] = df_s["Chain_mean"].map(lambda x: f"{x:.4f}")
        df_s["mean_diff (Chain-BR)"] = df_s["mean_diff (Chain-BR)"].map(lambda x: f"{x:+.4f}")
        df_s["p_raw"] = df_s["p_raw"].map(lambda x: f"{x:.4f}")
        df_s["p_holm"] = df_s["p_holm"].map(lambda x: f"{x:.4f}")
        f.write(df_to_markdown(df_s))
        f.write("\n\n")

        f.write("## Tertiary: BR vs best-of-6 chain (max-statistic, selection-corrected)\n\n")
        f.write(
            "Test stat: max over the 6 orderings of mean per-fold diff (chain - BR). "
            "Null: same sign-flip per fold applied to all orderings (preserves joint "
            "dependence; corrects for picking the best ordering post-hoc). Exact 2^15 "
            "enumeration. p_naive is the biased per-ordering test on the selected best "
            "(do NOT report this as the headline) and is shown only to quantify the "
            "selection bias.\n\n"
        )
        df_b = df_best.copy()
        df_b["BR_mean"] = df_b["BR_mean"].map(lambda x: f"{x:.4f}")
        df_b["BestChain_mean"] = df_b["BestChain_mean"].map(lambda x: f"{x:.4f}")
        df_b["mean_diff (Best-BR)"] = df_b["mean_diff (Best-BR)"].map(lambda x: f"{x:+.4f}")
        df_b["p_naive (biased)"] = df_b["p_naive (biased)"].map(lambda x: f"{x:.4f}")
        df_b["p_maxstat"] = df_b["p_maxstat"].map(lambda x: f"{x:.4f}")
        df_b["p_maxstat_one_sided"] = df_b["p_maxstat_one_sided"].map(lambda x: f"{x:.4f}")
        f.write(df_to_markdown(df_b))
        f.write("\n")

    df_primary.to_csv(tables_dir / "paired_permutation_primary.csv", index=False)
    df_secondary.to_csv(tables_dir / "paired_permutation_per_ordering.csv", index=False)
    df_best.to_csv(tables_dir / "paired_permutation_best_chain.csv", index=False)

    print(f"Wrote {out_md}")
    print("\n=== Primary (BR vs Ensemble) ===")
    print(df_primary.round(4).to_string(index=False))
    print("\n=== Tertiary (BR vs Best Chain, max-stat corrected) ===")
    print(df_best.round(4).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Within-site significance and confidence intervals")
    parser.add_argument("--analysis", choices=["ci", "permutation", "all"], default="all",
                        help="Which analysis to run (default: all)")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "s_orsola"),
                        help="Directory holding <model>/cv_results.pkl")
    parser.add_argument("--stats-dir", default=str(ROOT / "results" / "stats"),
                        help="Output directory for within_site_ci.json")
    parser.add_argument("--tables-dir", default=str(ROOT / "tables"),
                        help="Output directory for the permutation Markdown/CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if args.analysis in ("ci", "all"):
        run_confidence_intervals(results_dir, Path(args.stats_dir))
    if args.analysis in ("permutation", "all"):
        run_permutation(results_dir, Path(args.tables_dir))


if __name__ == "__main__":
    main()
