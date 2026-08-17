"""Command-line interface for scalar semantic similarities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .similarities import SimilarityBuildConfig, build_similarity_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-similarity",
        description=(
            "Calculate article similarity and strict-prior novelty from a completed, "
            "model-versioned embedding store without rerunning the transformer."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/scrape_2025"))
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path("model_output/selection_2025/embeddings"),
        help="Parent containing model/build embedding stores.",
    )
    parser.add_argument(
        "--embedding-store",
        type=Path,
        default=None,
        help="Optional exact embedding build directory containing embedding_manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("model_output/selection_2025/similarities"),
    )
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        dest="years",
        default=None,
        help="Target year; repeat as needed. Defaults to 2025 because lookback similarities are not model inputs.",
    )
    parser.add_argument("--model-id", default="BAAI/bge-m3")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--exact-novelty-threshold", type=int, default=5_000)
    parser.add_argument("--hnsw-validation-sample", type=int, default=100)
    parser.add_argument("--hnsw-required-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument(
        "--allow-subset-source",
        action="store_true",
        help="Permit a watermarked subset embedding store for smoke testing only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every-stories", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimilarityBuildConfig(
        data_root=args.data_root,
        embedding_root=args.embedding_root,
        embedding_store=args.embedding_store,
        output_root=args.output_root,
        years=tuple(args.years) if args.years else (2025,),
        model_id=args.model_id,
        revision=args.revision,
        exact_novelty_threshold=args.exact_novelty_threshold,
        hnsw_validation_sample=args.hnsw_validation_sample,
        hnsw_required_recall=args.hnsw_required_recall,
        seed=args.seed,
        max_stories=args.max_stories,
        allow_subset_source=args.allow_subset_source,
        overwrite=args.overwrite,
        progress_every_stories=args.progress_every_stories,
    )
    print(json.dumps(build_similarity_store(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

