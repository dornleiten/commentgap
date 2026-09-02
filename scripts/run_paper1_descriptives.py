#!/usr/bin/env python3
"""Build Paper 1 descriptive/topic artifacts (all comments by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commentgap_analysis.paper1_descriptives import run_descriptive_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-root", type=Path, default=Path("model_output/selection_2025/model_data"))
    parser.add_argument("--data-root", type=Path, default=Path("data/scrape_2025"))
    parser.add_argument("--output-root", type=Path, default=Path("model_output/selection_2025/paper1/descriptives"))
    parser.add_argument("--scope", action="append", choices=("all", "root"), dest="scopes", help="Repeat to include root appendix (default: all).")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_descriptive_analysis(
        model_data_root=args.model_data_root,
        data_root=args.data_root,
        output_root=args.output_root,
        scopes=args.scopes or ("all",),
        threads=args.threads,
        make_figures=not args.no_figures,
    )
    print(json.dumps({"manifest": str(args.output_root / "descriptive_manifest.json"), "scopes": manifest["scopes"]}, indent=2))


if __name__ == "__main__":
    main()
