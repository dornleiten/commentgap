#!/usr/bin/env python3
"""Freeze Paper 1 factorial winners from development CV only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commentgap_analysis.factorial_winners import freeze_development_cv_winners


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-cv", type=Path, default=Path("model_output/selection_2025/factorial_rankers/development_cv_results.csv"))
    parser.add_argument("--experiment-variants", type=Path, help="Defaults to experiment_variants.csv beside --development-cv.")
    parser.add_argument("--output-root", type=Path, default=Path("model_output/selection_2025/paper1/factorial_winners"))
    parser.add_argument("--scope", action="append", choices=("all", "root"), dest="scopes", help="Repeat to include root appendix winners (default: all).")
    parser.add_argument("--expected-folds", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = freeze_development_cv_winners(
        development_cv_path=args.development_cv,
        experiment_variants_path=args.experiment_variants,
        output_root=args.output_root,
        scopes=args.scopes or ("all",),
        expected_folds=args.expected_folds,
    )
    print(json.dumps({"winners": manifest["winners"], "held_out_artifacts_read": False}, indent=2))


if __name__ == "__main__":
    main()
