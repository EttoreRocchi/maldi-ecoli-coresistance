#!/usr/bin/env python3
"""Step 2: AMR co-occurrence analysis (Figure 1, Figure 2 and the DRIAMS UpSet plots).

For each dataset, compute pairwise phi coefficients, chi-squared tests, and
resistance-profile counts, then save per-dataset figures and JSON. When more than
one dataset is given, also save the multi-site phi matrix (Figure 1). The paper
copies (figures/main/fig1_coresistance_phi_heatmaps, fig2_upset_s_orsola
and figures/supplementary/upset_driams_*) are written alongside.

Usage:
    python scripts/01_cooccurrence/01_cooccurrence.py
    python scripts/01_cooccurrence/01_cooccurrence.py --configs configs/s_orsola.yaml configs/driams_c.yaml
    python scripts/01_cooccurrence/01_cooccurrence.py --plot-only      # redraw from results/cooccurrence/*/cooccurrence_results.json
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from mecor.visualization.cooccurrence import (
    plot_phi_heatmap,
    plot_phi_heatmaps_multi,
    plot_resistance_profiles,
    plot_upset,
    plot_upset_from_counts,
    profile_counts_from_Y,
)

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)

# All five sites, in the order used for the paper figure.
DEFAULT_CONFIGS = [
    "configs/s_orsola.yaml",
    "configs/driams_a.yaml",
    "configs/driams_b.yaml",
    "configs/driams_c.yaml",
    "configs/driams_d.yaml",
]

# Where the LaTeX sources include the figures (file names fixed).
PAPER_FIG1 = ROOT / "figures/main/fig1_coresistance_phi_heatmaps"
PAPER_UPSET = {
    "s_orsola": ROOT / "figures/main/fig2_upset_s_orsola",
    "driams_a": ROOT / "figures/supplementary/upset_driams_a",
    "driams_b": ROOT / "figures/supplementary/upset_driams_b",
    "driams_c": ROOT / "figures/supplementary/upset_driams_c",
    "driams_d": ROOT / "figures/supplementary/upset_driams_d",
}


def parse_args():
    parser = argparse.ArgumentParser(description="AMR co-occurrence analysis")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help="Dataset config files to analyze")
    parser.add_argument("--output-dir", default="results/cooccurrence",
                        help="Output directory for figures and JSON")
    parser.add_argument("--plot-only", action="store_true",
                        help="Do not load the spectra; redraw every figure from the stored JSON results")
    return parser.parse_args()


def copy_to_paper(stem_src: Path, stem_dst: Path):
    """Copy <stem>.png and <stem>.pdf to the paper folder under the fixed paper name."""
    stem_dst.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        src = stem_src.with_suffix(f".{ext}")
        if src.exists():
            shutil.copy2(src, stem_dst.with_suffix(f".{ext}"))


def analyze_one(config, output_dir, logger):
    """Run the co-occurrence analysis for a single dataset and save its outputs.

    Returns (analysis, results_json) or None if the dataset could not be loaded.
    """
    from mecor.analysis.cooccurrence import compute_cooccurrence_analysis
    from mecor.data.loader import load_dataset

    try:
        X, Y = load_dataset(config)
    except Exception as e:
        logger.error(f"Failed to load {config.dataset_name}: {e}")
        return None

    analysis = compute_cooccurrence_analysis(Y)

    ds_dir = output_dir / config.dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    plot_phi_heatmap(analysis["phi_matrix"], output_path=ds_dir / "phi_heatmap.png")
    plot_resistance_profiles(analysis["profile_counts"], output_path=ds_dir / "resistance_profiles.png")
    plot_upset(Y, output_path=ds_dir / "upset.png")

    results_json = {
        "dataset": config.dataset_name,
        "n_samples": analysis["n_samples"],
        "resistance_rates": analysis["resistance_rates"],
        "pairwise_tests": analysis["chi2_results"],
        "phi_matrix": analysis["phi_matrix"].to_dict(),
        "labels": list(Y.columns),
        "joint_profile_counts": profile_counts_from_Y(Y),   # R/S string in label order, e.g. "RSS"
    }
    with open(ds_dir / "cooccurrence_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\n--- {config.dataset_name} (n={analysis['n_samples']}) ---")
    for r in analysis["chi2_results"]:
        sig = _stars(r["p_value"])
        print(f"  {r['drug_1']}-{r['drug_2']}: phi={r['phi']:.3f}, "
              f"chi2={r['chi2']:.1f}, p={r['p_value']:.2e} {sig}")

    return analysis, results_json


def _stars(p):
    """Significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _stored_profile_counts(name, results_json):
    """Joint profile counts for the UpSet plot: from the per-site JSON, else from results/stats/coresistance_structure.json."""
    if "joint_profile_counts" in results_json:
        return results_json["joint_profile_counts"], results_json["labels"]
    stats_path = ROOT / "results/stats/coresistance_structure.json"
    if stats_path.exists():
        stats = json.load(open(stats_path))
        site = stats.get("X10a", {}).get("sites", {}).get(name)
        if site is not None:
            labels = list(results_json["phi_matrix"].keys())   # CIP, CTX or CRO, CAZ
            return site["profiles_CIP_3GC_CAZ"], labels
    return None, None


def replot_one(name, output_dir, logger):
    """Redraw the per-site figures from results/cooccurrence/<name>/cooccurrence_results.json."""
    ds_dir = output_dir / name
    path = ds_dir / "cooccurrence_results.json"
    if not path.exists():
        logger.warning(f"No stored results for {name} ({path}), skipping")
        return None
    results_json = json.load(open(path))
    phi = pd.DataFrame(results_json["phi_matrix"]).astype(float)
    plot_phi_heatmap(phi, output_path=ds_dir / "phi_heatmap.png")
    counts, labels = _stored_profile_counts(name, results_json)
    if counts is None:
        logger.warning(f"No joint profile counts stored for {name}; UpSet plot not redrawn")
    else:
        plot_upset_from_counts(counts, labels, output_path=ds_dir / "upset.png")
    analysis = {"phi_matrix": phi, "n_samples": results_json["n_samples"]}
    return analysis, results_json


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_analyses = {}
    all_results = {}
    for config_path in args.configs:
        if not Path(config_path).exists():
            logger.warning(f"Config not found: {config_path}, skipping")
            continue
        if args.plot_only:
            name = Path(config_path).stem
            logger.info(f"Replotting from stored results: {name}")
            out = replot_one(name, output_dir, logger)
        else:
            from mecor.config import Config
            config = Config.from_yaml(config_path)
            name = config.dataset_name
            logger.info(f"Analyzing: {name}")
            out = analyze_one(config, output_dir, logger)
        if out is None:
            continue
        analysis, results_json = out
        all_analyses[name] = analysis
        all_results[name] = results_json
        if name in PAPER_UPSET and (output_dir / name / "upset.png").exists():
            copy_to_paper(output_dir / name / "upset", PAPER_UPSET[name])

    # Multi-dataset phi matrix (Figure 1).
    if len(all_analyses) > 1:
        plot_phi_heatmaps_multi(
            all_analyses,
            output_path=output_dir / "phi_heatmaps_comparison.png",
        )
        copy_to_paper(output_dir / "phi_heatmaps_comparison", PAPER_FIG1)
        logger.info("Saved multi-dataset phi matrix (Figure 1)")

    if not args.plot_only:
        with open(output_dir / "all_cooccurrence_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
