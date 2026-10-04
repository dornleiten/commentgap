#!/usr/bin/env python3
"""Build the Paper 1 2025 comment-gap score (all comments by default)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from commentgap_analysis.comment_gap import run_comment_gap_analysis
from commentgap_analysis.paths import ExecutionContext, add_execution_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-root", type=Path, default=None)
    parser.add_argument("--descriptives-root", type=Path, default=None)
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
    if args.descriptives_root is not None:
        descriptives_root = context.read_path(args.descriptives_root)
    else:
        run_descriptives = context.run_path("CG1/descriptives")
        if os.environ.get("COMMENTGAP_DESCRIPTIVES_ROOT"):
            descriptives_root = context.read_root(
                "frozen_cg1_descriptives", env_var="COMMENTGAP_DESCRIPTIVES_ROOT"
            )
        elif context.mode != "frozen" and run_descriptives.exists():
            descriptives_root = context.run_input("CG1/descriptives")
        else:
            descriptives_root = context.read_root("frozen_cg1_descriptives")
    output_root = context.output_root("CG1/comment_gap", explicit=args.output_root, env_var="COMMENTGAP_COMMENT_GAP_ROOT")
    manifest = run_comment_gap_analysis(
        model_data_root=model_data_root,
        descriptives_root=descriptives_root,
        output_root=output_root,
        scopes=args.scopes or ("all",),
        threads=args.threads,
        make_figures=not args.no_figures,
    )
    print(json.dumps({"manifest": str(output_root / "comment_gap_manifest.json"), "scopes": manifest["scopes"]}, indent=2))


if __name__ == "__main__":
    main()
