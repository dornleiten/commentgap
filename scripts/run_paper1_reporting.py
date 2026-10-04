#!/usr/bin/env python3
"""Build Paper 1 tables and plots from frozen factorial winners."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from commentgap_analysis.paper1_reporting import run_paper1_reporting
from commentgap_analysis.paths import ExecutionContext, add_execution_arguments


def _input_root(context: ExecutionContext, *, explicit: Path | None, key: str, run_area: str, env_var: str) -> Path:
    if explicit is not None:
        return context.read_path(explicit)
    if os.environ.get(env_var):
        return context.read_root(key, env_var=env_var)
    candidate = context.run_path(run_area)
    if context.mode != "frozen" and candidate.exists():
        return context.run_input(run_area)
    return context.read_root(key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-root", type=Path, default=None)
    parser.add_argument("--factorial-root", type=Path, default=None)
    parser.add_argument("--winner-root", type=Path, default=None)
    parser.add_argument("--regression-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scope", action="append", choices=("all", "root"), dest="scopes", help="Repeat to include root appendix outputs (default: all).")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--permutation-repeats", type=int, default=1)
    parser.add_argument("--shap-test-rows", type=int, default=50_000)
    parser.add_argument("--shap-background-rows", type=int, default=2_048)
    parser.add_argument("--shap-nsamples", type=int, default=100)
    parser.add_argument("--shap-chunk-rows", type=int, default=500)
    parser.add_argument("--shap-force-recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--no-figures", action="store_true")
    add_execution_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    context = ExecutionContext.from_values(mode=args.mode, run_id=args.run_id, repo_root=args.repo_root)
    model_data_root = context.read_root("model_data", explicit=args.model_data_root, env_var="COMMENTGAP_MODEL_DATA_ROOT")
    factorial_root = _input_root(
        context, explicit=args.factorial_root, key="frozen_cg1_factorial_rankers",
        run_area="CG1/rankers/factorial", env_var="COMMENTGAP_FACTORIAL_ROOT",
    )
    winner_root = _input_root(
        context, explicit=args.winner_root, key="frozen_cg1_winners",
        run_area="CG1/winners", env_var="COMMENTGAP_WINNER_ROOT",
    )
    regression_root = _input_root(
        context, explicit=args.regression_root, key="frozen_cg1_regression",
        run_area="CG1/regression", env_var="COMMENTGAP_REGRESSION_ROOT",
    )
    if context.mode == "frozen":
        output_root = context.staging_output("rendered/CG1/reporting", explicit=args.output_root)
    else:
        output_root = context.output_root("CG1/reporting", explicit=args.output_root, env_var="COMMENTGAP_REPORTING_ROOT")
    manifest = run_paper1_reporting(
        model_data_root=model_data_root,
        factorial_root=factorial_root,
        winner_root=winner_root,
        regression_root=regression_root,
        output_root=output_root,
        scopes=args.scopes or ("all",),
        bootstrap_draws=args.bootstrap_draws,
        permutation_repeats=args.permutation_repeats,
        shap_test_rows=args.shap_test_rows,
        shap_background_rows=args.shap_background_rows,
        shap_nsamples=args.shap_nsamples,
        shap_chunk_rows=args.shap_chunk_rows,
        shap_force_recompute=args.shap_force_recompute,
        seed=args.seed,
        make_figures=not args.no_figures,
    )
    print(json.dumps({"manifest": str(output_root / "report_manifest.json"), "outputs": len(manifest["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
