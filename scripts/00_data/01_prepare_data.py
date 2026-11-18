#!/usr/bin/env python3
"""Step 1: Prepare and validate a dataset.

Sant'Orsola: optionally preprocess raw spectra with maldiamrkit into a binned
CSV, then load, validate, and write a quality report.
DRIAMS: load the pre-binned spectra plus metadata, validate, and report stats.

Usage:
    python scripts/00_data/01_prepare_data.py --config configs/s_orsola.yaml
    python scripts/00_data/01_prepare_data.py --config configs/s_orsola.yaml --preprocess
    python scripts/00_data/01_prepare_data.py --config configs/driams_c.yaml
"""

import argparse
import json
import logging
from pathlib import Path

from mecor.config import Config
from mecor.data.loader import load_dataset
from mecor.data.preprocessing import preprocess_spectra
from mecor.data.quality import quality_report, validate_data


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a dataset for the AMR pipeline")
    parser.add_argument("--config", required=True,
                        help="Path to a dataset config YAML (see configs/)")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run maldiamrkit preprocessing on the local raw spectra first")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.preprocess and config.dataset_name in ("local", "s_orsola"):
        logger.info("Preprocessing local spectra with maldiamrkit...")
        processed_path = Path(config.data_path)   # the loader reads the binned matrix from data_path
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        combined = preprocess_spectra(
            spectra_dir=config.spectra_dir,
            metadata_path=config.metadata_path,
            antibiotics=config.targets,
            output_path=str(processed_path),
            bin_width=config.bin_width,
            mz_min=config.mz_min,
            mz_max=config.mz_max,
        )
        # the folds depend on the row order: restore the order of the matrix used in the paper when it is recorded
        order_file = Path(args.config).with_name(f"{config.dataset_name}_sample_order.txt")
        if order_file.exists():
            order = [ln.strip() for ln in order_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
            assert set(order) == set(combined.index), f"{order_file} does not list the same samples as the spectra"
            combined.loc[order].to_csv(processed_path)
            logger.info(f"Rows reordered as in {order_file}")
        logger.info(f"Saved processed data to {processed_path}")

    logger.info(f"Loading dataset: {config.dataset_name}")
    X, Y = load_dataset(config)
    logger.info(f"Dataset shape: X={X.shape}, Y={Y.shape}")
    validate_data(X, Y, config)

    report = quality_report(X, Y)
    report["dataset_name"] = config.dataset_name
    report["config_file"] = args.config

    report_path = output_dir / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Quality report saved to {report_path}")

    print(f"\n{'='*60}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Samples: {report['n_samples']}")
    print(f"Features: {report['n_features']}")
    print(f"{'='*60}")
    for label, rates in report["resistance_rates"].items():
        print(f"  {label}: {rates['resistant']}R / {rates['susceptible']}S "
              f"({rates['rate']:.1%} resistant)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
