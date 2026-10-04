"""Command-line interface for model-ready analysis features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import FeatureBuildConfig, build_analysis_features
from .nlp import (
    DEFAULT_SENTIMENT_MODEL_ID,
    DEFAULT_SENTIMENT_MODEL_REVISION,
    DEFAULT_TOXICITY_MODEL_ID,
    DEFAULT_TOXICITY_MODEL_REVISION,
    resolve_hf_model_revision,
)
from .paths import ExecutionContext, add_execution_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-features",
        description=(
            "Build resumable model features and root/all-comment choice sets from "
            "the normalized collection and precomputed semantic similarities."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--similarity-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--similarity-store", type=Path, default=None)
    parser.add_argument(
        "--aqua-store",
        type=Path,
        default=None,
        help="Optional completed AQuA build/root to validate and merge by key and text hash.",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--lookback-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--nlp-mode", choices=("real", "pilot"), default="real")
    parser.add_argument("--sentiment-model-id", default=DEFAULT_SENTIMENT_MODEL_ID)
    parser.add_argument("--sentiment-revision", default=None)
    parser.add_argument("--sentiment-batch-size", type=int, default=32)
    parser.add_argument("--toxicity-model-id", default=DEFAULT_TOXICITY_MODEL_ID)
    parser.add_argument("--toxicity-revision", default=None)
    parser.add_argument("--toxicity-batch-size", type=int, default=16)
    parser.add_argument("--embedding-model-id", default="BAAI/bge-m3")
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--tie-draws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument("--progress-every-rows", type=int, default=250_000)
    parser.add_argument("--progress-every-stories", type=int, default=100)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Watermark outputs as pilot-only and permit --max-stories/pilot NLP.",
    )
    parser.add_argument(
        "--include-january-without-lookback",
        action="store_true",
        help="Do not exclude January when no prior-December lookback root is supplied.",
    )
    parser.add_argument(
        "--allow-non-page-publication-time",
        action="store_true",
        help="Permit publication timestamps not extracted from article pages.",
    )
    parser.add_argument("--overwrite", action="store_true")
    add_execution_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = ExecutionContext.from_values(
        mode=args.mode, run_id=args.run_id, repo_root=args.repo_root
    )
    sentiment_revision = args.sentiment_revision or (
        DEFAULT_SENTIMENT_MODEL_REVISION
        if args.sentiment_model_id == DEFAULT_SENTIMENT_MODEL_ID
        else None
    )
    toxicity_revision = args.toxicity_revision or (
        DEFAULT_TOXICITY_MODEL_REVISION
        if args.toxicity_model_id == DEFAULT_TOXICITY_MODEL_ID
        else None
    )
    if not args.pilot and args.nlp_mode == "real":
        sentiment_revision = resolve_hf_model_revision(
            args.sentiment_model_id,
            sentiment_revision,
        )
        toxicity_revision = resolve_hf_model_revision(
            args.toxicity_model_id,
            toxicity_revision,
        )
    config = FeatureBuildConfig(
        data_root=context.read_root("raw_scrape", explicit=args.data_root, env_var="COMMENTGAP_DATA_ROOT"),
        output_root=context.output_root("shared/features", explicit=args.output_root, env_var="COMMENTGAP_FEATURE_ROOT"),
        similarity_root=context.read_root("similarities", explicit=args.similarity_root, env_var="COMMENTGAP_SIMILARITY_ROOT"),
        similarity_store=context.read_path(args.similarity_store) if args.similarity_store else None,
        aqua_store=context.read_path(args.aqua_store) if args.aqua_store else None,
        year=args.year,
        lookback_root=context.read_path(args.lookback_root) if args.lookback_root else None,
        allow_incomplete=args.allow_incomplete,
        inference_mode=not args.pilot,
        nlp_mode=args.nlp_mode,
        device=args.device,
        sentiment_model_id=args.sentiment_model_id,
        sentiment_revision=sentiment_revision,
        toxicity_model_id=args.toxicity_model_id,
        toxicity_revision=toxicity_revision,
        embedding_model_id=args.embedding_model_id,
        embedding_revision=args.embedding_revision,
        sentiment_batch_size=args.sentiment_batch_size,
        toxicity_batch_size=args.toxicity_batch_size,
        tie_draws=args.tie_draws,
        seed=args.seed,
        require_page_publication_time=not args.allow_non_page_publication_time,
        exclude_january_without_lookback=not args.include_january_without_lookback,
        overwrite=args.overwrite,
        max_stories=args.max_stories,
        progress_every_rows=args.progress_every_rows,
        progress_every_stories=args.progress_every_stories,
    )
    print(json.dumps(build_analysis_features(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
