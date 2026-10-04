"""Command-line entry point shared by the neural ranking workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .neural_ranking import (
    default_recipe,
    run_neural_ranker_workflow,
)
from .paths import ExecutionContext, add_execution_arguments


SUPPORTED_APPROACHES = ("frozen_bge", "metadata_mlp")


def build_parser(default_approach: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approach",
        choices=SUPPORTED_APPROACHES,
        default=default_approach,
        required=default_approach is None,
    )
    parser.add_argument(
        "--model-data-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=None,
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
    add_execution_arguments(parser)
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
    context = ExecutionContext.from_values(
        mode=args.mode, run_id=args.run_id, repo_root=args.repo_root
    )
    approach = args.approach
    recipe = default_recipe(approach)
    model_data_root = context.read_root("model_data", explicit=args.model_data_root, env_var="COMMENTGAP_MODEL_DATA_ROOT")
    data_root = context.read_root("raw_scrape", explicit=args.data_root, env_var="COMMENTGAP_DATA_ROOT")
    embedding_root = context.read_root("embeddings", explicit=args.embedding_root, env_var="COMMENTGAP_EMBEDDING_ROOT")
    output_root = context.output_root(f"CG1/rankers/neural/{approach}", explicit=args.output_root, env_var="COMMENTGAP_NEURAL_RANKER_ROOT")
    result = run_neural_ranker_workflow(
        model_data_root,
        data_root,
        output_root,
        approach=approach,
        embedding_root=embedding_root,
        embedding_store=context.read_path(args.embedding_store) if args.embedding_store else None,
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
