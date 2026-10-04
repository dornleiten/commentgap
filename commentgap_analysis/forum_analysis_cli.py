"""Command-line entry point for the FORUM/ranking analysis pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .forum_scores import (
    ALL_OUTCOMES,
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_RANDOM_DRAWS,
    DEFAULT_SEED,
    DEFAULT_TIE_DRAWS,
    DEFAULT_WORKERS,
    FORUM_OUTCOMES,
    build_forum_analysis_table,
    freeze_ranker_handoff,
    run_policy_inference,
    run_policy_scoring,
)
from .ranking_algorithm_effects import run_ranking_algorithm_effects
from .paths import ExecutionContext, add_execution_arguments


def resolve_forum_canonical_path(canonical_path: str | Path, forum_root: Path) -> Path:
    """Map a saved FORUM pipeline target into the selected analysis root."""
    canonical_path = str(canonical_path).replace("\\", "/")
    relative_prefix = "CG2/forum/"
    if canonical_path.startswith(relative_prefix):
        relative_path = canonical_path[len(relative_prefix):]
    elif "/CG2/forum/" in canonical_path:
        relative_path = canonical_path.split("/CG2/forum/", 1)[1]
    elif canonical_path.endswith("/CG2/forum"):
        relative_path = ""
    else:
        raise ValueError(f"Unrecognized canonical FORUM path: {canonical_path}")
    return Path(forum_root) / relative_path


def _input_path(
    context: ExecutionContext,
    *,
    explicit: Path | None,
    env_var: str,
    run_area: str,
    frozen_key: str,
    suffix: str = "",
) -> Path:
    """Resolve an explicit override before same-run and saved artifacts."""
    if explicit is not None:
        return context.read_path(explicit)
    if (value := os.environ.get(env_var)):
        return context.read_path(value)
    same_run = context.run_path(run_area)
    if context.mode != "frozen" and (same_run / suffix).exists():
        base = context.run_input(run_area)
    else:
        base = context.read_root(frozen_key)
    return base / suffix if suffix else base


def _add_freeze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--factorial-root", type=Path)
    parser.add_argument(
        "--regression-scores", type=Path
    )
    parser.add_argument("--handoff-root", type=Path)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument(
        "--allow-active-factorial",
        action="store_true",
        help="Override the safety check that the factorial ranker process is idle.",
    )


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--handoff-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--choice-set",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--embedding-store",
        type=Path,
        required=True,
        help="Frozen comment-embedding store used for static novelty.",
    )
    parser.add_argument("--analysis-root", type=Path, default=None)
    parser.add_argument("--min-comments", type=int, default=11)


def _add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--analysis-comments",
        type=Path,
        default=None,
    )
    parser.add_argument("--policy-root", type=Path, default=None)
    parser.add_argument("--tie-draws", type=int, default=DEFAULT_TIE_DRAWS)
    parser.add_argument("--random-draws", type=int, default=DEFAULT_RANDOM_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of independent story-scoring processes (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=list(ALL_OUTCOMES),
        choices=list(FORUM_OUTCOMES),
    )


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy-scores",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--analysis-comments",
        type=Path,
        default=None,
    )
    parser.add_argument("--inference-root", type=Path, default=None)
    parser.add_argument(
        "--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def _add_reporting_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inference-root", type=Path, default=None)
    parser.add_argument("--reporting-root", type=Path, default=None)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-forum-analysis",
        description=(
            "Freeze Paper 1 rankers, construct the FORUM held-out table, "
            "score the 90 policy bundles, run paired inference, and build "
            "paper outputs."
        ),
    )
    add_execution_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_freeze_arguments(
        subparsers.add_parser("freeze", help="Freeze development-selected rankers.")
    )
    _add_build_arguments(
        subparsers.add_parser("build", help="Build the immutable analysis table.")
    )
    _add_score_arguments(
        subparsers.add_parser("score", help="Score the full policy factorial.")
    )
    _add_inference_arguments(
        subparsers.add_parser("infer", help="Run paired bootstrap inference.")
    )
    _add_reporting_arguments(
        subparsers.add_parser("report", help="Build figures and LaTeX tables.")
    )
    all_parser = subparsers.add_parser(
        "all", help="Run freeze, build, score, inference, and reporting in sequence."
    )
    _add_freeze_arguments(all_parser)
    _add_build_arguments(all_parser)
    all_parser.set_defaults(
        handoff_manifest=None,
        analysis_comments=None,
    )
    all_parser.add_argument("--policy-root", type=Path, default=None)
    all_parser.add_argument("--inference-root", type=Path, default=None)
    all_parser.add_argument("--reporting-root", type=Path, default=None)
    all_parser.add_argument("--tie-draws", type=int, default=DEFAULT_TIE_DRAWS)
    all_parser.add_argument("--random-draws", type=int, default=DEFAULT_RANDOM_DRAWS)
    all_parser.add_argument(
        "--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS
    )
    all_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    all_parser.add_argument("--progress-every", type=int, default=10)
    all_parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of independent story-scoring processes (default: {DEFAULT_WORKERS}).",
    )
    all_parser.add_argument(
        "--outcomes",
        nargs="+",
        default=list(ALL_OUTCOMES),
        choices=list(FORUM_OUTCOMES),
    )
    return parser


def _freeze(arguments: argparse.Namespace) -> dict[str, Any]:
    return freeze_ranker_handoff(
        factorial_root=arguments.factorial_root,
        regression_scores_path=arguments.regression_scores,
        output_root=arguments.handoff_root,
        expected_folds=arguments.expected_folds,
        require_idle=not arguments.allow_active_factorial,
    )


def _build(arguments: argparse.Namespace) -> dict[str, Any]:
    handoff_manifest = arguments.handoff_manifest
    if handoff_manifest is None:
        handoff_manifest = arguments.handoff_root / "ranker_handoff_manifest.json"
    return build_forum_analysis_table(
        handoff_manifest_path=handoff_manifest,
        choice_set_path=arguments.choice_set,
        split_path=arguments.split,
        data_root=arguments.data_root,
        embedding_store=arguments.embedding_store,
        output_root=arguments.analysis_root,
        min_comments=arguments.min_comments,
    )


def _score(arguments: argparse.Namespace) -> dict[str, Any]:
    analysis_comments = arguments.analysis_comments
    if analysis_comments is None:
        analysis_comments = arguments.analysis_root / "analysis_comments.parquet"
    return run_policy_scoring(
        analysis_path=analysis_comments,
        output_root=arguments.policy_root,
        outcomes=arguments.outcomes,
        tie_draws=arguments.tie_draws,
        random_draws=arguments.random_draws,
        seed=arguments.seed,
        progress_every_stories=arguments.progress_every,
        workers=arguments.workers,
    )


def _infer(arguments: argparse.Namespace) -> dict[str, Any]:
    policy_scores = getattr(arguments, "policy_scores", None)
    if policy_scores is None:
        policy_scores = arguments.policy_root / "policy_scores.parquet"
    analysis_comments = arguments.analysis_comments
    if analysis_comments is None:
        analysis_comments = arguments.analysis_root / "analysis_comments.parquet"
    return run_policy_inference(
        policy_scores_path=policy_scores,
        analysis_comments_path=analysis_comments,
        output_root=arguments.inference_root,
        bootstrap_draws=arguments.bootstrap_draws,
        seed=arguments.seed,
    )


def _report(arguments: argparse.Namespace) -> dict[str, Any]:
    return run_ranking_algorithm_effects(
        inference_root=arguments.inference_root,
        output_root=arguments.reporting_root,
    )



def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    context = ExecutionContext.from_values(
        mode=arguments.mode, run_id=arguments.run_id, repo_root=arguments.repo_root
    )
    command = arguments.command
    # Historical stores remain inputs. Every newly produced FORUM stage is
    # rooted beneath one fresh run, including an `all` pipeline.
    if hasattr(arguments, "factorial_root"):
        arguments.factorial_root = _input_path(
            context, explicit=arguments.factorial_root,
            env_var="COMMENTGAP_FACTORIAL_ROOT", run_area="CG1/rankers/factorial",
            frozen_key="frozen_cg1_factorial_rankers",
        )
        arguments.regression_scores = _input_path(
            context, explicit=arguments.regression_scores,
            env_var="COMMENTGAP_REGRESSION_ROOT", run_area="CG1/regression",
            frozen_key="frozen_cg1_regression", suffix="all/test_scores_wide.parquet",
        )
        arguments.handoff_root = context.output_root(
            "CG2/forum/ranker_handoff",
            explicit=arguments.handoff_root,
            env_var="COMMENTGAP_FORUM_HANDOFF_ROOT",
        )
    if command in {"build", "all"}:
        if arguments.handoff_manifest is not None:
            arguments.handoff_manifest = context.read_path(arguments.handoff_manifest)
        model_data_root = context.read_root(
            "model_data", env_var="COMMENTGAP_MODEL_DATA_ROOT"
        )
        arguments.choice_set = context.read_path(arguments.choice_set) if arguments.choice_set else model_data_root / "choice_set_all.parquet"
        arguments.split = context.read_path(arguments.split) if arguments.split else model_data_root / "master_article_split.parquet"
        arguments.data_root = context.read_root(
            "raw_scrape", explicit=arguments.data_root,
            env_var="COMMENTGAP_DATA_ROOT",
        )
        arguments.analysis_root = context.output_root(
            "CG2/forum", explicit=arguments.analysis_root,
            env_var="COMMENTGAP_FORUM_ANALYSIS_ROOT",
        )
    if command == "score":
        arguments.analysis_comments = _input_path(
            context, explicit=arguments.analysis_comments,
            env_var="COMMENTGAP_ANALYSIS_COMMENTS_PATH", run_area="CG2/forum",
            frozen_key="frozen_cg2_forum", suffix="analysis_comments.parquet",
        )
    if command == "infer":
        arguments.policy_scores = _input_path(
            context, explicit=arguments.policy_scores,
            env_var="COMMENTGAP_FORUM_SCORES_PATH", run_area="CG2/forum/policy_scores",
            frozen_key="frozen_cg2_forum", suffix="policy_scores.parquet",
        )
        arguments.analysis_comments = _input_path(
            context, explicit=arguments.analysis_comments,
            env_var="COMMENTGAP_ANALYSIS_COMMENTS_PATH", run_area="CG2/forum",
            frozen_key="frozen_cg2_forum", suffix="analysis_comments.parquet",
        )
    if command == "report":
        arguments.inference_root = _input_path(
            context, explicit=arguments.inference_root,
            env_var="COMMENTGAP_FORUM_INFERENCE_ROOT", run_area="CG2/forum/inference",
            frozen_key="frozen_cg2_forum", suffix="inference",
        )
    if hasattr(arguments, "policy_root"):
        arguments.policy_root = context.output_root(
            "CG2/forum/policy_scores", explicit=arguments.policy_root,
            env_var="COMMENTGAP_FORUM_POLICY_ROOT",
        )
    if hasattr(arguments, "inference_root") and command in {"infer", "all"}:
        arguments.inference_root = context.output_root(
            "CG2/forum/inference", explicit=arguments.inference_root,
            env_var="COMMENTGAP_FORUM_INFERENCE_ROOT",
        )
    if hasattr(arguments, "reporting_root"):
        arguments.reporting_root = context.output_root(
            "CG2/forum/reporting", explicit=arguments.reporting_root,
            env_var="COMMENTGAP_FORUM_REPORTING_ROOT",
        )
    if arguments.command == "freeze":
        result = _freeze(arguments)
    elif arguments.command == "build":
        result = _build(arguments)
    elif arguments.command == "score":
        result = _score(arguments)
    elif arguments.command == "infer":
        result = _infer(arguments)
    elif arguments.command == "report":
        result = _report(arguments)
    else:
        result = {"freeze": _freeze(arguments)}
        result["build"] = _build(arguments)
        result["score"] = _score(arguments)
        result["infer"] = _infer(arguments)
        result["report"] = _report(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
