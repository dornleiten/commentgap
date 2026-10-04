#!/usr/bin/env python3
"""Freeze Paper 1 factorial winners from development CV only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commentgap_analysis.factorial_winners import freeze_development_cv_winners
from commentgap_analysis.paths import ExecutionContext, add_execution_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-cv", type=Path, default=None)
    parser.add_argument("--experiment-variants", type=Path, help="Defaults to experiment_variants.csv beside --development-cv.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scope", action="append", choices=("all", "root"), dest="scopes", help="Repeat to include root appendix winners (default: all).")
    parser.add_argument("--expected-folds", type=int, default=5)
    add_execution_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    context = ExecutionContext.from_values(mode=args.mode, run_id=args.run_id, repo_root=args.repo_root)
    factorial_root = (
        context.read_root("frozen_cg1_factorial_rankers")
        if context.mode == "frozen"
        else context.run_input("CG1/rankers/factorial")
    )
    development_cv = args.development_cv or factorial_root / "development_cv_results.csv"
    output_root = context.output_root("CG1/winners", explicit=args.output_root, env_var="COMMENTGAP_WINNER_ROOT")
    manifest = freeze_development_cv_winners(
        development_cv_path=development_cv,
        experiment_variants_path=args.experiment_variants,
        output_root=output_root,
        scopes=args.scopes or ("all",),
        expected_folds=args.expected_folds,
    )
    print(json.dumps({"winners": manifest["winners"], "held_out_artifacts_read": False}, indent=2))


if __name__ == "__main__":
    main()
