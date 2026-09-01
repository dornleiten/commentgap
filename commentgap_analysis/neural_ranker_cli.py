"""Command-line entry point shared by the neural ranking workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .neural_ranking import (
    default_recipe,
    run_neural_ranker_workflow,
)


DEFAULT_OUTPUTS = {
    "frozen_bge": Path("model_output/selection_2025/neural_rankers/frozen_bge_m3"),
    "metadata_mlp": Path("model_output/selection_2025/neural_rankers/metadata_mlp"),
}


def build_parser(default_approach: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approach",
        choices=tuple(DEFAULT_OUTPUTS),
        default=default_approach,
        required=default_approach is None,
    )
    parser.add_argument(
        "--model-data-root",
        type=Path,
        default=Path("model_output/selection_2025/model_data"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/scrape_2025"))
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path("model_output/selection_2025/embeddings"),
    )
    parser.add_argument("--embedding-store", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument(
        "--progress-every-stories",
        type=int,
        default=100,
        help="Print training/inference progress and ETA every N completed articles.",
    )
    parser.add_argument(
        "--training-mode",
        choices=("cv", "fixed_split", "full"),
        default="cv",
        help=(
            "Development training strategy: cv uses five folds; fixed_split "
            "uses predefined fold 0 and refits; full trains all development "
            "articles for max_epochs without validation."
        ),
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore resumable workflow checkpoints and start incomplete scopes afresh.",
    )
    return parser


def main(argv: list[str] | None = None, *, default_approach: str | None = None) -> None:
    args = build_parser(default_approach).parse_args(argv)
    approach = args.approach
    recipe = default_recipe(approach)
    output_root = args.output_root or DEFAULT_OUTPUTS[approach]
    result = run_neural_ranker_workflow(
        args.model_data_root,
        args.data_root,
        output_root,
        approach=approach,
        embedding_root=args.embedding_root,
        embedding_store=args.embedding_store,
        device=args.device,
        bootstrap_draws=args.bootstrap_draws,
        progress_every_stories=args.progress_every_stories,
        training_mode=args.training_mode,
        resume=not args.no_resume,
        force_recompute=args.force_recompute,
        recipe=recipe,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
