#!/usr/bin/env python3
"""Build Paper 1 tables and plots from frozen factorial winners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commentgap_analysis.paper1_reporting import run_paper1_reporting


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-root", type=Path, default=Path("model_output/selection_2025/model_data"))
    parser.add_argument("--factorial-root", type=Path, default=Path("model_output/selection_2025/factorial_rankers"))
    parser.add_argument("--winner-root", type=Path, default=Path("model_output/selection_2025/paper1/factorial_winners"))
    parser.add_argument("--regression-root", type=Path, default=Path("model_output/selection_2025/regression"))
    parser.add_argument("--output-root", type=Path, default=Path("model_output/selection_2025/paper1/reporting"))
    parser.add_argument("--scope", action="append", choices=("all", "root"), dest="scopes", help="Repeat to include root appendix outputs (default: all).")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--permutation-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_paper1_reporting(
        model_data_root=args.model_data_root,
        factorial_root=args.factorial_root,
        winner_root=args.winner_root,
        regression_root=args.regression_root,
        output_root=args.output_root,
        scopes=args.scopes or ("all",),
        bootstrap_draws=args.bootstrap_draws,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
        make_figures=not args.no_figures,
    )
    print(json.dumps({"manifest": str(args.output_root / "report_manifest.json"), "outputs": len(manifest["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
