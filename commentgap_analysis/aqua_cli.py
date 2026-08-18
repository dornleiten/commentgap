"""CLI for the main-environment AQuA feature-store orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aqua import AquaBuildConfig, build_aqua_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-aqua",
        description="Build a resumable AQuA feature store via an isolated legacy runtime.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/scrape_2025"))
    parser.add_argument("--output-root", type=Path, default=Path("model_output/selection_2025/aqua"))
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--runtime-python", type=Path, default=Path(".venv-aqua/bin/python"))
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=Path(".cache/aqua-upstream-637914d/trained adapters"),
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=Path("aqua_runtime/artifacts.json"),
    )
    parser.add_argument(
        "--requirements-lock",
        type=Path,
        default=None,
        help="Runtime lockfile (default: CUDA lock for --device cuda, CPU lock otherwise).",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--execution-mode", choices=("parallel", "sequential"), default="parallel")
    parser.add_argument(
        "--no-sequential-fallback",
        action="store_true",
        help="Do not retry an accelerator OOM with deterministic sequential adapters.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Maximum adaptive batch size (default: 64).",
    )
    parser.add_argument(
        "--adaptive-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Length-bucket under a padded-token budget (default: enabled).",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=2048,
        help="Maximum batch-size × capped-token-length budget (default: 2048).",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every-stories", type=int, default=25)
    parser.add_argument(
        "--allow-unverified-parity",
        action="store_true",
        help="Permit only a pilot-watermarked build before upstream parity is frozen.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AquaBuildConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        year=args.year,
        runtime_python=args.runtime_python,
        adapter_root=args.adapter_root,
        artifact_manifest=args.artifact_manifest,
        requirements_lock=args.requirements_lock,
        device=args.device,
        execution_mode=args.execution_mode,
        sequential_fallback=not args.no_sequential_fallback,
        batch_size=args.batch_size,
        adaptive_batches=args.adaptive_batches,
        max_batch_tokens=args.max_batch_tokens,
        max_length=args.max_length,
        allow_incomplete=args.allow_incomplete,
        max_stories=args.max_stories,
        overwrite=args.overwrite,
        progress_every_stories=args.progress_every_stories,
        require_parity=not args.allow_unverified_parity,
    )
    print(json.dumps(build_aqua_store(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
