#!/usr/bin/env python3
"""Build Paper 1 descriptive/topic artifacts (all comments by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commentgap_analysis.paper1_descriptives import run_descriptive_analysis
from commentgap_analysis.paths import ExecutionContext, add_execution_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scope", action="append", choices=("all", "root"), dest="scopes", help="Repeat to include root appendix (default: all).")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-figures", action="store_true")
    add_execution_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    context = ExecutionContext.from_values(mode=args.mode, run_id=args.run_id, repo_root=args.repo_root)
    model_data_root = context.read_root("model_data", explicit=args.model_data_root, env_var="COMMENTGAP_MODEL_DATA_ROOT")
    data_root = context.read_root("raw_scrape", explicit=args.data_root, env_var="COMMENTGAP_DATA_ROOT")
    output_root = context.output_root("CG1/descriptives", explicit=args.output_root, env_var="COMMENTGAP_DESCRIPTIVES_ROOT")
    manifest = run_descriptive_analysis(
        model_data_root=model_data_root,
        data_root=data_root,
        output_root=output_root,
        scopes=args.scopes or ("all",),
        threads=args.threads,
        make_figures=not args.no_figures,
    )
    print(json.dumps({"manifest": str(output_root / "descriptive_manifest.json"), "scopes": manifest["scopes"]}, indent=2))


if __name__ == "__main__":
    main()
