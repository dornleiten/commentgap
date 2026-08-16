"""Command-line interface for the reusable embedding store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import (
    EmbeddingBuildConfig,
    build_embedding_store,
    build_token_length_diagnostics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-embed",
        description=(
            "Create resumable, model-versioned comment and article-passage embeddings "
            "from the normalized Parquet collection."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/scrape_2025"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("model_output/selection_2025/embeddings"),
    )
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        dest="years",
        default=None,
        help=(
            "Year to embed; repeat for multiple years. By default every matching "
            "article/comment year partition is discovered."
        ),
    )
    parser.add_argument("--model-id", default="BAAI/bge-m3")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Initial embedding batch size.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Adaptive ceiling; defaults to the initial batch size.",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="Smallest automatic OOM retry batch.",
    )
    parser.add_argument(
        "--batch-growth-successes",
        type=int,
        default=25,
        help="Successful full-batch calls required before doubling toward the ceiling.",
    )
    parser.add_argument(
        "--no-adaptive-batching",
        action="store_true",
        help="Disable automatic accelerator OOM backoff and recovery.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Vector precision on disk; float16 roughly halves vector storage.",
    )
    parser.add_argument(
        "--prompt-name",
        default=None,
        help="Optional Sentence Transformers prompt name required by some checkpoints.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a QA-valid but unfinished crawl; the store is watermarked.",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Deterministic smoke-test subset; the store is watermarked.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute checkpoints in the matching model/data build namespace.",
    )
    parser.add_argument(
        "--progress-every-stories",
        type=int,
        default=25,
        help="Print elapsed time, throughput, and ETA after this many articles.",
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=2048,
        help="CPU tokenizer batch size used for truncation diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Write token-length diagnostics without loading model weights or using a GPU.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = EmbeddingBuildConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        years=tuple(args.years) if args.years else None,
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        min_batch_size=args.min_batch_size,
        adaptive_batching=not args.no_adaptive_batching,
        batch_growth_successes=args.batch_growth_successes,
        max_length=args.max_length,
        storage_dtype=args.storage_dtype,
        prompt_name=args.prompt_name,
        allow_incomplete=args.allow_incomplete,
        max_stories=args.max_stories,
        overwrite=args.overwrite,
        progress_every_stories=args.progress_every_stories,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )
    summary = (
        build_token_length_diagnostics(config)
        if args.diagnostics_only
        else build_embedding_store(config)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
