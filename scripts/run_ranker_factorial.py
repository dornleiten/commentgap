#!/usr/bin/env python3
"""Run the 68-model resumable XGBoost/neural ranking factorial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from commentgap_analysis.factorial_rankers import (
    factorial_variants,
    run_factorial_experiment,
    select_variants,
)


DEFAULT_OUTPUT_ROOT = Path(
    "model_output/selection_2025/factorial_rankers"
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


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
    parser.add_argument("--embedding-store", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.getenv("COMMENTGAP_FACTORIAL_ROOT", str(DEFAULT_OUTPUT_ROOT))
        ),
    )
    parser.add_argument(
        "--device",
        default=os.getenv("COMMENTGAP_DEVICE", "auto"),
        help="Default device for both families unless overridden below.",
    )
    parser.add_argument(
        "--neural-device",
        help="Override the device only for neural variants (for example cuda).",
    )
    parser.add_argument(
        "--xgb-device",
        help="Override the device only for XGBoost variants (for example cpu).",
    )
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
        "--xgb-verbose-every",
        type=int,
        default=50,
        help="Print an XGBoost evaluation line every N boosting rounds; 0 disables.",
    )
    parser.add_argument(
        "--include",
        action="append",
        metavar="GLOB",
        help="Run only matching stable variant IDs; repeatable (default: all).",
    )
    parser.add_argument(
        "--exclude",
        "--skip",
        action="append",
        default=[],
        dest="exclude",
        metavar="GLOB",
        help="Skip matching stable variant IDs; repeatable.",
    )
    parser.add_argument(
        "--scope",
        action="append",
        choices=("root", "all"),
        dest="scopes",
        help="Run only this candidate scope; repeat to select both (default: both).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        default=_env_flag("COMMENTGAP_FORCE_RECOMPUTE"),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching incomplete neural epoch checkpoints.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Record an ordinary Python failure and continue with the next model.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print all stable variant IDs and exit.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the variants selected by include/exclude filters and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_parser().parse_args(argv)
    variants = factorial_variants()
    include = tuple(args.include or ["*"])
    exclude = tuple(args.exclude)
    selected = select_variants(variants, include=include, exclude=exclude)
    if args.list_models:
        for index, variant in enumerate(variants, start=1):
            print(
                f"{index:02d}/{len(variants)} {variant.variant_id} "
                + json.dumps(variant.as_record(), sort_keys=True)
            )
        return {}
    if args.plan:
        selected_ids = {variant.variant_id for variant in selected}
        for index, variant in enumerate(variants, start=1):
            status = "RUN" if variant.variant_id in selected_ids else "FILTERED"
            print(f"{index:02d}/{len(variants)} {status:8s} {variant.variant_id}")
        print(f"selected={len(selected)}/{len(variants)}")
        return {}
    if not selected:
        raise SystemExit("No variants match the include/exclude filters")
    scopes = tuple(dict.fromkeys(args.scopes or ["root", "all"]))
    return run_factorial_experiment(
        model_data_root=args.model_data_root,
        data_root=args.data_root,
        embedding_root=args.embedding_root,
        embedding_store=args.embedding_store,
        output_root=args.output_root,
        neural_device=args.neural_device or args.device,
        xgb_device=args.xgb_device or args.device,
        bootstrap_draws=args.bootstrap_draws,
        progress_every_stories=args.progress_every_stories,
        xgb_verbose_every=args.xgb_verbose_every,
        training_mode="cv",
        force_recompute=args.force_recompute,
        resume=not args.no_resume,
        include=include,
        exclude=exclude,
        scopes=scopes,
        keep_going=args.keep_going,
    )


if __name__ == "__main__":
    main()
