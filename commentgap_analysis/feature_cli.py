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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-features",
        description=(
            "Build resumable model features and root/all-comment choice sets from "
            "the normalized collection and precomputed semantic similarities."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/scrape_2025"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("model_output/selection_2025/features"),
    )
    parser.add_argument(
        "--similarity-root",
        type=Path,
        default=Path("model_output/selection_2025/similarities"),
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        data_root=args.data_root,
        output_root=args.output_root,
        similarity_root=args.similarity_root,
        similarity_store=args.similarity_store,
        aqua_store=args.aqua_store,
        year=args.year,
        lookback_root=args.lookback_root,
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
