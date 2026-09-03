"""Command-line entry point for the FORUM/ranking analysis pipeline."""

from __future__ import annotations

import argparse
import json
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


DEFAULT_FACTORIAL_ROOT = Path("model_output/selection_2025/factorial_rankers")
DEFAULT_REGRESSION_SCORES = Path(
    "model_output/selection_2025/regression/all/test_scores_wide.parquet"
)
DEFAULT_HANDOFF_ROOT = Path(
    "model_output/selection_2025/forum_ranking_analysis/ranker_handoff"
)
DEFAULT_ANALYSIS_ROOT = Path("model_output/selection_2025/forum_ranking_analysis")
DEFAULT_POLICY_ROOT = DEFAULT_ANALYSIS_ROOT / "policy_scores"
DEFAULT_INFERENCE_ROOT = DEFAULT_ANALYSIS_ROOT / "inference"
DEFAULT_REPORTING_ROOT = DEFAULT_ANALYSIS_ROOT / "reporting"


def _add_freeze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--factorial-root", type=Path, default=DEFAULT_FACTORIAL_ROOT)
    parser.add_argument(
        "--regression-scores", type=Path, default=DEFAULT_REGRESSION_SCORES
    )
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
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
        default=DEFAULT_HANDOFF_ROOT / "ranker_handoff_manifest.json",
    )
    parser.add_argument(
        "--choice-set",
        type=Path,
        default=Path(
            "model_output/selection_2025/model_data/choice_set_all.parquet"
        ),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(
            "model_output/selection_2025/model_data/master_article_split.parquet"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/scrape_2025"))
    parser.add_argument(
        "--embedding-store",
        type=Path,
        required=True,
        help="Frozen comment-embedding store used for static novelty.",
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--min-comments", type=int, default=11)


def _add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--analysis-comments",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "analysis_comments.parquet",
    )
    parser.add_argument("--policy-root", type=Path, default=DEFAULT_POLICY_ROOT)
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
        default=DEFAULT_POLICY_ROOT / "policy_scores.parquet",
    )
    parser.add_argument(
        "--analysis-comments",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "analysis_comments.parquet",
    )
    parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT)
    parser.add_argument(
        "--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def _add_reporting_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT)
    parser.add_argument("--reporting-root", type=Path, default=DEFAULT_REPORTING_ROOT)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-forum-analysis",
        description=(
            "Freeze Paper 1 rankers, construct the FORUM held-out table, "
            "score the 90 policy bundles, run paired inference, and build "
            "paper outputs."
        ),
    )
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
    all_parser.add_argument("--policy-root", type=Path, default=DEFAULT_POLICY_ROOT)
    all_parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT)
    all_parser.add_argument("--reporting-root", type=Path, default=DEFAULT_REPORTING_ROOT)
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
