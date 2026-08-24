"""Freeze and verify AQuA upstream/isolated-runtime parity without raw text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from aqua_runtime.schema import (
    AQUA_FEATURES,
    AQUA_UPSTREAM_COMMIT,
    label_column,
    logit_column,
    sha256_file,
)


PARITY_LOGIT_ATOL = 2e-6
PARITY_LOGIT_RTOL = 1e-6


def _normalize_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    key_columns = ["story_id", "comment_id"]
    missing = [column for column in key_columns if column not in frame]
    if missing:
        raise ValueError(f"{label} parity input is missing keys: {missing}")
    if frame[key_columns].isna().any().any():
        raise ValueError(f"{label} parity input contains null keys")
    normalized = frame.copy()
    for column in key_columns:
        normalized[column] = normalized[column].astype(str)
    return normalized


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_parity(
    upstream_output: Path,
    runtime_output: Path,
    *,
    repeat_output: Path | None = None,
    sequential_output: Path | None = None,
) -> dict:
    upstream = _normalize_keys(
        pd.read_csv(upstream_output, sep="\t"), "Upstream"
    )
    runtime = _normalize_keys(pd.read_parquet(runtime_output), "Runtime")
    key_columns = ["story_id", "comment_id"]
    if upstream.duplicated(key_columns).any() or runtime.duplicated(key_columns).any():
        raise ValueError("Parity inputs contain duplicate keys")
    merged = upstream.merge(
        runtime,
        on=key_columns,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Parity key mismatch: {merged['_merge'].value_counts().to_dict()}")
    adapter_results = {}
    for feature in AQUA_FEATURES:
        upstream_column = f"{feature.repository_adapter}_ad"
        runtime_column = label_column(feature.stem)
        if upstream_column not in merged or runtime_column not in merged:
            raise ValueError(f"Missing parity columns for {feature.repository_adapter}")
        matches = merged[upstream_column].astype(int).eq(merged[runtime_column].astype(int))
        adapter_results[feature.repository_adapter] = {
            "rows": len(matches),
            "matching_hard_labels": int(matches.sum()),
            "exact": bool(matches.all()),
        }
    if "score" not in merged:
        raise ValueError("Upstream parity output is missing the published score column")
    score_difference = np.abs(
        merged["score"].to_numpy(float) - merged["aqua_score_hard"].to_numpy(float)
    )
    comparisons = {}
    for label, path in (("repeat_cpu", repeat_output), ("sequential", sequential_output)):
        if path is None:
            continue
        other = _normalize_keys(pd.read_parquet(path), label)
        if other.duplicated(key_columns).any():
            raise ValueError(f"{label} parity output contains duplicate keys")
        comparison = runtime.merge(
            other,
            on=key_columns,
            how="outer",
            suffixes=("_reference", "_comparison"),
            indicator=True,
            validate="one_to_one",
        )
        if not comparison["_merge"].eq("both").all():
            raise ValueError(
                f"{label} parity key mismatch: "
                f"{comparison['_merge'].value_counts().to_dict()}"
            )
        labels_equal = all(
            comparison[f"{label_column(feature.stem)}_reference"].equals(
                comparison[f"{label_column(feature.stem)}_comparison"]
            )
            for feature in AQUA_FEATURES
        )
        adapter_maximum_differences = {}
        logit_differences = []
        for feature in AQUA_FEATURES:
            feature_differences = []
            for ordinal in range(4):
                difference = np.abs(
                    comparison[
                        f"{logit_column(feature.stem, ordinal)}_reference"
                    ].to_numpy(float)
                    - comparison[
                        f"{logit_column(feature.stem, ordinal)}_comparison"
                    ].to_numpy(float)
                )
                feature_differences.append(difference)
                logit_differences.append(difference)
            adapter_maximum_differences[feature.repository_adapter] = float(
                np.concatenate(feature_differences).max(initial=0.0)
            )
        all_logit_differences = np.concatenate(logit_differences)
        logits_close = all(
            np.allclose(
                comparison[f"{logit_column(feature.stem, ordinal)}_reference"],
                comparison[f"{logit_column(feature.stem, ordinal)}_comparison"],
                atol=PARITY_LOGIT_ATOL,
                rtol=PARITY_LOGIT_RTOL,
            )
            for feature in AQUA_FEATURES
            for ordinal in range(4)
        )
        comparisons[label] = {
            "hard_labels_exact": labels_equal,
            "logits_within_tolerance": logits_close,
            "logit_absolute_tolerance": PARITY_LOGIT_ATOL,
            "logit_relative_tolerance": PARITY_LOGIT_RTOL,
            "maximum_absolute_logit_difference": float(
                all_logit_differences.max(initial=0.0)
            ),
            "adapter_maximum_absolute_logit_differences": (
                adapter_maximum_differences
            ),
            "sha256": sha256_file(path),
        }
    verified = (
        all(result["exact"] for result in adapter_results.values())
        and bool((score_difference <= 1e-6).all())
        and all(
            result["hard_labels_exact"] and result["logits_within_tolerance"]
            for result in comparisons.values()
        )
    )
    return {
        "status": "verified" if verified else "failed",
        "upstream_commit": AQUA_UPSTREAM_COMMIT,
        "rows": len(merged),
        "adapter_results": adapter_results,
        "maximum_hard_score_difference": float(score_difference.max(initial=0.0)),
        "comparisons": comparisons,
        "files": {
            "upstream_output_sha256": sha256_file(upstream_output),
            "runtime_output_sha256": sha256_file(runtime_output),
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "contains_raw_comment_text": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="commentgap-aqua-parity")
    parser.add_argument("--upstream-output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--repeat-output", type=Path, default=None)
    parser.add_argument("--sequential-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, default=Path("aqua_runtime/artifacts.json"))
    parser.add_argument("--update-artifact-manifest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_parity(
        args.upstream_output,
        args.runtime_output,
        repeat_output=args.repeat_output,
        sequential_output=args.sequential_output,
    )
    _atomic_json(report, args.report)
    if args.update_artifact_manifest:
        if report["status"] != "verified":
            raise RuntimeError("Refusing to mark a failed parity report as verified")
        if args.repeat_output is None or args.sequential_output is None:
            raise RuntimeError(
                "Freezing production parity requires both --repeat-output and "
                "--sequential-output"
            )
        artifact_manifest = json.loads(args.artifact_manifest.read_text())
        artifact_manifest["parity"] = {
            "status": "verified",
            "fixture": str(args.report),
            "fixture_sha256": sha256_file(args.report),
        }
        _atomic_json(artifact_manifest, args.artifact_manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
