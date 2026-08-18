"""CLI executed only by the isolated AQuA Python environment."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import resource
import time

import pandas as pd

from .model import AquaModel, predictions_to_frame
from .schema import AQUA_FEATURES, feature_by_adapter, text_hash


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _package_versions() -> dict[str, str]:
    packages = ("adapter-transformers", "huggingface-hub", "numpy", "pandas", "pyarrow", "torch", "tqdm")
    output = {}
    for package in packages:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m aqua_runtime.cli")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--job-manifest", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--execution-mode", choices=("parallel", "sequential"), default="parallel")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Maximum rows per adaptive batch, or exact row batch size in fixed mode.",
    )
    parser.add_argument(
        "--adaptive-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Length-bucket rows under a padded-token budget (default: enabled).",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=2048,
        help="Maximum batch rows multiplied by the longest capped token length.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--progress-every-shards", type=int, default=25)
    parser.add_argument("--adapter", action="append", default=None)
    parser.add_argument("--build-signature", required=True)
    parser.add_argument("--watermark", choices=("PRODUCTION", "PILOT_NOT_FOR_INFERENCE"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args(argv)
    if (
        args.batch_size < 1
        or args.max_batch_tokens < 1
        or args.max_length < 1
        or args.progress_every_shards < 1
    ):
        raise ValueError(
            "batch size, token budget, maximum length, and progress interval "
            "must be positive"
        )
    if (args.job_manifest is None) == (args.input is None):
        raise ValueError("Specify either --input/--output or --job-manifest")
    if args.input is not None and args.output is None:
        raise ValueError("--output is required with --input")
    if args.job_manifest is not None and args.summary_output is None:
        raise ValueError("--summary-output is required with --job-manifest")
    jobs = (
        json.loads(args.job_manifest.read_text())["jobs"]
        if args.job_manifest is not None
        else [{"input": str(args.input), "output": str(args.output)}]
    )
    if not jobs:
        raise ValueError("AQuA job manifest cannot be empty")
    started = time.monotonic()
    features = (
        tuple(feature_by_adapter(adapter) for adapter in args.adapter)
        if args.adapter
        else AQUA_FEATURES
    )
    model = AquaModel(
        adapter_root=args.adapter_root,
        artifact_manifest=args.artifact_manifest,
        device=args.device,
        execution_mode=args.execution_mode,
        max_length=args.max_length,
        features=features,
    )
    total_rows = 0
    total_truncated = 0
    batching_totals = {
        "planned_batches": 0,
        "executed_batches": 0,
        "oom_backoffs": 0,
        "effective_unpadded_tokens": 0,
        "planned_padded_tokens": 0,
        "executed_padded_tokens": 0,
    }
    minimum_executed_batch_size: int | None = None
    maximum_executed_batch_size = 0
    for job_index, job in enumerate(jobs, start=1):
        job_started = time.monotonic()
        input_path = Path(job["input"])
        output_path = Path(job["output"])
        source = pd.read_parquet(input_path)
        required = {"story_id", "comment_id", "effective_text", "effective_text_hash"}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"AQuA input is missing columns: {sorted(missing)}")
        if source[["story_id", "comment_id", "effective_text", "effective_text_hash"]].isna().any().any():
            raise ValueError("AQuA input keys, text, and text hashes cannot be null")
        if source.duplicated(["story_id", "comment_id"]).any():
            raise ValueError("AQuA input contains duplicate keys")
        text = source["effective_text"].astype(str)
        if text.str.strip().eq("").any():
            raise ValueError("AQuA input contains empty comment text")
        if not text.map(text_hash).eq(source["effective_text_hash"]).all():
            raise ValueError("AQuA input contains an invalid effective_text_hash")
        logits, token_counts, truncated, batching = model.predict(
            text.tolist(),
            args.batch_size,
            adaptive_batches=args.adaptive_batches,
            max_batch_tokens=args.max_batch_tokens,
            return_diagnostics=True,
        )
        output = predictions_to_frame(
            source[["story_id", "comment_id", "effective_text_hash"]],
            logits,
            token_counts,
            truncated,
            build_signature=args.build_signature,
            watermark=args.watermark,
            features=features,
        )
        _atomic_parquet(output, output_path)
        total_rows += len(output)
        total_truncated += int(output["aqua_input_truncated"].sum())
        for key in batching_totals:
            batching_totals[key] += int(batching[key])
        job_minimum = int(batching["minimum_executed_batch_size"])
        if job_minimum:
            minimum_executed_batch_size = (
                job_minimum
                if minimum_executed_batch_size is None
                else min(minimum_executed_batch_size, job_minimum)
            )
        maximum_executed_batch_size = max(
            maximum_executed_batch_size,
            int(batching["maximum_executed_batch_size"]),
        )
        _atomic_json(
            {
                "rows": len(output),
                "truncated_rows": int(output["aqua_input_truncated"].sum()),
                "elapsed_seconds": time.monotonic() - job_started,
                "device": args.device,
                "execution_mode": args.execution_mode,
                "batch_size": args.batch_size,
                "batching": batching,
                "job_index": job_index,
                "jobs": len(jobs),
                "build_signature": args.build_signature,
                "watermark": args.watermark,
            },
            output_path.with_suffix(".runtime.json"),
        )
        if job_index % args.progress_every_shards == 0 or job_index == len(jobs):
            print(
                f"AQuA runtime: {job_index:,}/{len(jobs):,} shards "
                f"rows={total_rows:,}",
                flush=True,
            )
    executed_padded_tokens = batching_totals["executed_padded_tokens"]
    batching_summary = {
        "strategy": (
            "length_bucketed_token_budget"
            if args.adaptive_batches
            else "fixed_rows"
        ),
        "max_batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens if args.adaptive_batches else None,
        **batching_totals,
        "minimum_executed_batch_size": minimum_executed_batch_size or 0,
        "maximum_executed_batch_size": maximum_executed_batch_size,
        "mean_executed_batch_size": (
            total_rows / batching_totals["executed_batches"]
            if batching_totals["executed_batches"]
            else 0.0
        ),
        "executed_padding_fraction": (
            1.0
            - batching_totals["effective_unpadded_tokens"] / executed_padded_tokens
            if executed_padded_tokens
            else 0.0
        ),
    }
    summary = {
        "rows": total_rows,
        "shards": len(jobs),
        "device": args.device,
        "execution_mode": args.execution_mode,
        "batch_size": args.batch_size,
        "batching": batching_summary,
        "max_length": args.max_length,
        "truncated_rows": total_truncated,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "adapters": [feature.repository_adapter for feature in features],
        "artifact_hashes": model.artifact_hashes,
        "build_signature": args.build_signature,
        "watermark": args.watermark,
    }
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary["peak_memory_mib"] = peak_rss / (1024 * 1024 if platform.system() == "Darwin" else 1024)
    summary_path = args.summary_output or args.output.with_suffix(".runtime-summary.json")
    _atomic_json(summary, summary_path)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
