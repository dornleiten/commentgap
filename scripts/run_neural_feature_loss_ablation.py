#!/usr/bin/env python3
"""Run the four neural feature-by-loss ablation models with live progress."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

from commentgap_analysis.neural_ranking import (
    NeuralTrainingRecipe,
    default_recipe,
    run_neural_ranker_workflow,
)


DEFAULT_OUTPUT_ROOT = Path(
    "model_output/selection_2025/neural_rankers/feature_loss_ablation"
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


def ablation_recipes() -> dict[str, NeuralTrainingRecipe]:
    """Return the same four recipes declared by the ablation notebook."""
    metadata_base = default_recipe("metadata_mlp")
    frozen_base = default_recipe("frozen_bge")
    return {
        "metadata_pairwise": replace(
            metadata_base,
            ranking_loss="pairwise_logistic",
            hard_negatives_per_positive=0,
        ),
        "metadata_lambda": replace(
            metadata_base,
            ranking_loss="lambda_ndcg",
            hard_negatives_per_positive=1,
        ),
        "frozen_bge_pairwise": replace(
            frozen_base,
            ranking_loss="pairwise_logistic",
            hard_negatives_per_positive=0,
        ),
        "frozen_bge_lambda": replace(
            frozen_base,
            ranking_loss="lambda_ndcg",
            hard_negatives_per_positive=1,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-data-root",
        type=Path,
        default=Path(
            os.getenv(
                "COMMENTGAP_MODEL_DATA_ROOT",
                "model_output/selection_2025/model_data",
            )
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.getenv("COMMENTGAP_DATA_ROOT", "data/scrape_2025")),
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(
            os.getenv(
                "COMMENTGAP_EMBEDDING_ROOT",
                "model_output/selection_2025/embeddings",
            )
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.getenv("COMMENTGAP_NEURAL_ABLATION_ROOT", str(DEFAULT_OUTPUT_ROOT))
        ),
    )
    parser.add_argument("--device", default=os.getenv("COMMENTGAP_DEVICE", "auto"))
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=int(os.getenv("COMMENTGAP_BOOTSTRAP_DRAWS", "1000")),
    )
    parser.add_argument(
        "--progress-every-stories",
        type=int,
        default=int(os.getenv("COMMENTGAP_NEURAL_PROGRESS_EVERY_STORIES", "100")),
    )
    parser.add_argument(
        "--training-mode",
        choices=("cv", "fixed_split", "full"),
        default=os.getenv("COMMENTGAP_TRAINING_MODE", "cv"),
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        default=_env_flag("COMMENTGAP_FORCE_RECOMPUTE"),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching incomplete checkpoints and restart each model.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    results: dict[str, object] = {}
    recipes = ablation_recipes()
    for index, (model_name, recipe) in enumerate(recipes.items(), start=1):
        print(
            f"\n=== model {index}/{len(recipes)}: {model_name} "
            f"({recipe.approach}, {recipe.ranking_loss}) ===",
            flush=True,
        )
        results[model_name] = run_neural_ranker_workflow(
            args.model_data_root,
            args.data_root,
            args.output_root / model_name,
            approach=recipe.approach,
            embedding_root=args.embedding_root,
            device=args.device,
            bootstrap_draws=args.bootstrap_draws,
            training_mode=args.training_mode,
            progress_every_stories=args.progress_every_stories,
            force_recompute=args.force_recompute,
            resume=not args.no_resume,
            recipe=recipe,
        )
        print(
            json.dumps({model_name: results[model_name]}, indent=2, sort_keys=True),
            flush=True,
        )
    print("\n=== all four neural models complete ===", flush=True)
    return results


if __name__ == "__main__":
    main()
