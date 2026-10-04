"""CLI for the main-environment AQuA feature-store orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aqua import AquaBuildConfig, build_aqua_store
from .paths import ExecutionContext, add_execution_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-aqua",
        description="Build a resumable AQuA feature store via an isolated legacy runtime.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--runtime-python", type=Path, default=None)
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=None,
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
    parser.add_argument(
        "--window-max-stories",
        type=int,
        default=100,
        help="Maximum stories pooled for cross-story length bucketing (default: 100).",
    )
    parser.add_argument(
        "--window-max-rows",
        type=int,
        default=50000,
        help="Regular-RAM guard for one cross-story window (default: 50000 rows).",
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
    add_execution_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = ExecutionContext.from_values(
        mode=args.mode, run_id=args.run_id, repo_root=args.repo_root
    )
    config = AquaBuildConfig(
        data_root=context.read_root("raw_scrape", explicit=args.data_root, env_var="COMMENTGAP_DATA_ROOT"),
        output_root=context.output_root("shared/aqua", explicit=args.output_root, env_var="COMMENTGAP_AQUA_ROOT"),
        year=args.year,
        runtime_python=context.read_path(args.runtime_python) if args.runtime_python else context.paths.root / ".venv-aqua/bin/python",
        adapter_root=context.read_path(args.adapter_root) if args.adapter_root else context.paths.root / ".cache/aqua-upstream-637914d/trained adapters",
        artifact_manifest=context.read_path(args.artifact_manifest) if args.artifact_manifest else context.paths.root / "aqua_runtime/artifacts.json",
        requirements_lock=context.read_path(args.requirements_lock) if args.requirements_lock else None,
        device=args.device,
        execution_mode=args.execution_mode,
        sequential_fallback=not args.no_sequential_fallback,
        batch_size=args.batch_size,
        adaptive_batches=args.adaptive_batches,
        max_batch_tokens=args.max_batch_tokens,
        window_max_stories=args.window_max_stories,
        window_max_rows=args.window_max_rows,
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
