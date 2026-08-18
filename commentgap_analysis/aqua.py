"""Main-environment orchestration and validation for isolated AQuA inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from aqua_runtime.schema import (
    AQUA_FEATURES,
    AQUA_PROBABILITY_STATUS,
    AQUA_SCHEMA_VERSION,
    AQUA_SCORE_MAX,
    AQUA_SCORE_MIN,
    AQUA_UPSTREAM_COMMIT,
    composite_raw,
    downstream_feature_columns,
    expected_alias_column,
    expected_raw_column,
    label_column,
    logit_column,
    normalize_score,
    probability_column,
    sha256_file,
    text_hash,
)

from .features import (
    _atomic_json,
    _atomic_parquet,
    _read_dataset,
    dataset_fingerprint,
    validate_qa_summary,
)


DEFAULT_AQUA_OUTPUT_ROOT = Path("model_output/selection_2025/aqua")
DEFAULT_AQUA_RUNTIME_PYTHON = Path(".venv-aqua/bin/python")
DEFAULT_AQUA_ADAPTER_ROOT = Path(
    f".cache/aqua-upstream-{AQUA_UPSTREAM_COMMIT[:7]}/trained adapters"
)
DEFAULT_AQUA_ARTIFACT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "aqua_runtime" / "artifacts.json"
)
DEFAULT_AQUA_REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements-aqua-legacy.txt"
DEFAULT_AQUA_CUDA_REQUIREMENTS = (
    Path(__file__).resolve().parents[1] / "requirements-aqua-cuda113.txt"
)


@dataclass(frozen=True)
class AquaBuildConfig:
    data_root: Path = Path("data/scrape_2025")
    output_root: Path = DEFAULT_AQUA_OUTPUT_ROOT
    year: int = 2025
    runtime_python: Path = DEFAULT_AQUA_RUNTIME_PYTHON
    adapter_root: Path = DEFAULT_AQUA_ADAPTER_ROOT
    artifact_manifest: Path = DEFAULT_AQUA_ARTIFACT_MANIFEST
    requirements_lock: Path | None = None
    device: str = "cpu"
    execution_mode: str = "parallel"
    sequential_fallback: bool = True
    batch_size: int = 64
    adaptive_batches: bool = True
    max_batch_tokens: int = 2048
    max_length: int = 512
    allow_incomplete: bool = False
    max_stories: int | None = None
    overwrite: bool = False
    progress_every_stories: int = 25
    require_parity: bool = True

    def __post_init__(self) -> None:
        for name in (
            "data_root",
            "output_root",
            "runtime_python",
            "adapter_root",
            "artifact_manifest",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("AQuA device must be explicitly 'cpu' or 'cuda'")
        requirements_lock = self.requirements_lock
        if requirements_lock is None:
            requirements_lock = (
                DEFAULT_AQUA_CUDA_REQUIREMENTS
                if self.device == "cuda"
                else DEFAULT_AQUA_REQUIREMENTS
            )
        object.__setattr__(self, "requirements_lock", Path(requirements_lock))
        if self.execution_mode not in {"parallel", "sequential"}:
            raise ValueError("execution_mode must be 'parallel' or 'sequential'")
        if self.batch_size < 1 or self.max_batch_tokens < 1 or self.max_length < 1:
            raise ValueError(
                "AQuA batch size, token budget, and maximum length must be positive"
            )
        if self.max_stories is not None and self.max_stories < 1:
            raise ValueError("max_stories must be positive")
        if self.progress_every_stories < 1:
            raise ValueError("progress_every_stories must be positive")


def build_runtime_command(
    config: AquaBuildConfig,
    *,
    build_signature: str,
    watermark: str,
    input_path: Path | None = None,
    output_path: Path | None = None,
    execution_mode: str | None = None,
    job_manifest: Path | None = None,
    summary_output: Path | None = None,
) -> list[str]:
    """Construct the legacy subprocess command without importing its packages."""
    if job_manifest is None and (input_path is None or output_path is None):
        raise ValueError("Single-shard AQuA commands require input and output paths")
    if job_manifest is not None and summary_output is None:
        raise ValueError("Job-manifest AQuA commands require a summary output")
    input_arguments = (
        ["--job-manifest", str(job_manifest), "--summary-output", str(summary_output)]
        if job_manifest is not None
        else ["--input", str(input_path), "--output", str(output_path)]
    )
    return [
        str(config.runtime_python),
        "-m",
        "aqua_runtime.cli",
        *input_arguments,
        "--adapter-root",
        str(config.adapter_root),
        "--artifact-manifest",
        str(config.artifact_manifest),
        "--device",
        config.device,
        "--execution-mode",
        execution_mode or config.execution_mode,
        "--batch-size",
        str(config.batch_size),
        "--adaptive-batches" if config.adaptive_batches else "--no-adaptive-batches",
        "--max-batch-tokens",
        str(config.max_batch_tokens),
        "--max-length",
        str(config.max_length),
        "--progress-every-shards",
        str(config.progress_every_stories),
        "--build-signature",
        build_signature,
        "--watermark",
        watermark,
    ]


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        maximum = float(np.max(np.abs(actual - expected)))
        raise ValueError(f"AQuA {label} does not recompute (maximum difference {maximum})")


def validate_aqua_frame(
    frame: pd.DataFrame,
    *,
    expected: pd.DataFrame | None = None,
    build_signature: str | None = None,
    watermark: str | None = None,
) -> dict[str, Any]:
    """Validate full runtime output and optionally its exact source universe."""
    base_required = {
        "story_id",
        "comment_id",
        "effective_text_hash",
        "aqua_score_hard_raw",
        "aqua_score_hard",
        "aqua_score_expected_raw_unscaled",
        "aqua_score_expected_raw",
        "aqua_input_token_count",
        "aqua_input_truncated",
        "aqua_runtime_status",
        "aqua_probability_status",
        "aqua_build_signature",
        "aqua_watermark",
    }
    for feature in AQUA_FEATURES:
        base_required.add(label_column(feature.stem))
        base_required.add(expected_raw_column(feature.stem))
        base_required.update(logit_column(feature.stem, value) for value in range(4))
        base_required.update(probability_column(feature.stem, value) for value in range(4))
    missing = base_required - set(frame.columns)
    if missing:
        raise ValueError(f"AQuA output is missing columns: {sorted(missing)}")
    if frame.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("AQuA output contains duplicate keys")
    if frame[list(base_required)].isna().any().any():
        raise ValueError("AQuA output contains null required values")
    if not frame["aqua_runtime_status"].eq("ok").all():
        raise ValueError("AQuA output contains non-success runtime statuses")
    if not frame["aqua_probability_status"].eq(AQUA_PROBABILITY_STATUS).all():
        raise ValueError("AQuA raw probabilities are not marked uncalibrated")
    if any("calibrated" in column and "uncalibrated" not in column for column in frame.columns):
        raise ValueError("Calibrated AQuA columns require a separate calibration manifest")
    if build_signature is not None and not frame["aqua_build_signature"].eq(build_signature).all():
        raise ValueError("AQuA shard has the wrong build signature")
    if watermark is not None and not frame["aqua_watermark"].eq(watermark).all():
        raise ValueError("AQuA shard has the wrong watermark")

    hard_matrix: list[np.ndarray] = []
    expected_matrix: list[np.ndarray] = []
    class_frequencies: dict[str, dict[str, int]] = {}
    for feature in AQUA_FEATURES:
        labels = frame[label_column(feature.stem)].to_numpy()
        if not np.equal(labels, np.floor(labels)).all() or not np.isin(labels, range(4)).all():
            raise ValueError(f"Invalid ordinal labels for AQuA {feature.stem}")
        probabilities = frame[
            [probability_column(feature.stem, value) for value in range(4)]
        ].to_numpy(dtype=np.float64)
        logits = frame[
            [logit_column(feature.stem, value) for value in range(4)]
        ].to_numpy(dtype=np.float64)
        if not np.isfinite(probabilities).all() or not np.isfinite(logits).all():
            raise ValueError(f"Non-finite AQuA outputs for {feature.stem}")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError(f"Out-of-range AQuA probabilities for {feature.stem}")
        _assert_close(probabilities.sum(axis=1), np.ones(len(frame)), f"{feature.stem} probability sums")
        shifted_logits = logits - logits.max(axis=1, keepdims=True)
        recomputed_probabilities = np.exp(shifted_logits)
        recomputed_probabilities /= recomputed_probabilities.sum(axis=1, keepdims=True)
        _assert_close(
            probabilities,
            recomputed_probabilities,
            f"{feature.stem} probabilities from logits",
        )
        if not np.array_equal(probabilities.argmax(axis=1), labels.astype(int)):
            raise ValueError(f"AQuA {feature.stem} argmax does not equal the hard label")
        expected_values = probabilities @ np.arange(4, dtype=np.float64)
        stored_expected = frame[expected_raw_column(feature.stem)].to_numpy(dtype=np.float64)
        _assert_close(stored_expected, expected_values, f"{feature.stem} expected score")
        if (stored_expected < 0).any() or (stored_expected > 3).any():
            raise ValueError(f"AQuA expected score outside 0..3 for {feature.stem}")
        hard_matrix.append(labels.astype(np.float64))
        expected_matrix.append(stored_expected)
        counts = pd.Series(labels).value_counts().reindex(range(4), fill_value=0)
        class_frequencies[feature.stem] = {str(key): int(value) for key, value in counts.items()}

    hard_raw = composite_raw(np.column_stack(hard_matrix))
    expected_raw = composite_raw(np.column_stack(expected_matrix))
    _assert_close(frame["aqua_score_hard_raw"].to_numpy(float), hard_raw, "hard raw score")
    _assert_close(frame["aqua_score_hard"].to_numpy(float), normalize_score(hard_raw), "hard score")
    _assert_close(
        frame["aqua_score_expected_raw_unscaled"].to_numpy(float),
        expected_raw,
        "expected raw score",
    )
    _assert_close(
        frame["aqua_score_expected_raw"].to_numpy(float),
        normalize_score(expected_raw),
        "expected score",
    )
    for column in ("aqua_score_hard", "aqua_score_expected_raw"):
        values = frame[column].to_numpy(float)
        if (values < -1e-6).any() or (values > 5 + 1e-6).any():
            raise ValueError(f"{column} lies outside 0..5")
    if (frame["aqua_input_token_count"].to_numpy() < 1).any():
        raise ValueError("AQuA token counts must be positive")

    if expected is not None:
        expected_required = {"story_id", "comment_id", "effective_text_hash"}
        missing_expected = expected_required - set(expected.columns)
        if missing_expected:
            raise ValueError(f"Expected AQuA keys are missing: {sorted(missing_expected)}")
        if expected.duplicated(["story_id", "comment_id"]).any():
            raise ValueError("Expected AQuA candidate universe contains duplicate keys")
        comparison = expected[list(expected_required)].merge(
            frame[["story_id", "comment_id", "effective_text_hash"]],
            on=["story_id", "comment_id"],
            how="outer",
            suffixes=("_expected", "_actual"),
            indicator=True,
            validate="one_to_one",
        )
        if not comparison["_merge"].eq("both").all():
            counts = comparison["_merge"].value_counts().to_dict()
            raise ValueError(f"AQuA key coverage mismatch: {counts}")
        if not comparison["effective_text_hash_expected"].eq(
            comparison["effective_text_hash_actual"]
        ).all():
            raise ValueError("AQuA effective_text_hash mismatch")

    return {
        "rows": len(frame),
        "unique_keys": int(frame[["story_id", "comment_id"]].drop_duplicates().shape[0]),
        "truncated_rows": int(frame["aqua_input_truncated"].sum()),
        "class_frequencies": class_frequencies,
    }


def _candidate_comments(config: AquaBuildConfig) -> pd.DataFrame:
    columns = (
        "story_id",
        "comment_id",
        "effective_text",
        "lifecycle_status",
        "created_at",
        "year",
        "month",
    )
    comments = _read_dataset(config.data_root / "comments", config.year, list(columns))
    candidates = comments.loc[
        comments["lifecycle_status"].eq("Published")
        & comments["effective_text"].fillna("").astype(str).str.strip().ne("")
        & comments["created_at"].notna()
    ].copy()
    candidates["story_id"] = candidates["story_id"].astype(str)
    candidates["comment_id"] = candidates["comment_id"].astype(str)
    if config.max_stories is not None:
        selected = sorted(candidates["story_id"].unique())[: config.max_stories]
        candidates = candidates[candidates["story_id"].isin(selected)].copy()
    if candidates.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("Candidate AQuA input contains duplicate keys")
    month_counts = candidates.groupby("story_id")["month"].nunique()
    if (month_counts > 1).any():
        raise ValueError("AQuA expects every story to occupy one source month partition")
    candidates["effective_text_hash"] = candidates["effective_text"].astype(str).map(text_hash)
    return candidates.sort_values(["story_id", "comment_id"]).reset_index(drop=True)


def _build_identity(config: AquaBuildConfig, fingerprint: str) -> tuple[str, dict[str, Any]]:
    if not config.artifact_manifest.is_file():
        raise FileNotFoundError(config.artifact_manifest)
    if not config.requirements_lock.is_file():
        raise FileNotFoundError(config.requirements_lock)
    artifacts = json.loads(config.artifact_manifest.read_text())
    parity = artifacts.get("parity", {})
    if config.require_parity and parity.get("status") != "verified":
        raise RuntimeError(
            "Production AQuA requires a verified upstream parity fixture; run the "
            "isolated parity workflow or pass --allow-unverified-parity for a pilot."
        )
    identity = {
        "schema_version": AQUA_SCHEMA_VERSION,
        "upstream_commit": AQUA_UPSTREAM_COMMIT,
        "artifact_manifest_sha256": sha256_file(config.artifact_manifest),
        "requirements_lock_sha256": sha256_file(config.requirements_lock),
        "source_fingerprint": fingerprint,
        "year": config.year,
        "device": config.device,
        "execution_mode": config.execution_mode,
        "sequential_fallback": config.sequential_fallback,
        "batch_size": config.batch_size,
        "adaptive_batches": config.adaptive_batches,
        "max_batch_tokens": config.max_batch_tokens,
        "max_length": config.max_length,
        "allow_incomplete": config.allow_incomplete,
        "max_stories": config.max_stories,
        "parity_fixture_sha256": parity.get("fixture_sha256"),
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return signature, identity


def _runtime_env() -> dict[str, str]:
    environment = os.environ.copy()
    repository_root = str(Path(__file__).resolve().parents[1])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = repository_root if not existing else f"{repository_root}{os.pathsep}{existing}"
    return environment


def _distribution_summary(values: pd.Series) -> dict[str, float]:
    quantiles = values.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "mean": float(values.mean()),
        **{f"q{int(index * 100):02d}": float(value) for index, value in quantiles.items()},
    }


def _aggregate_validation(
    shard_paths: list[Path],
    *,
    expected_rows: int,
    runtime_seconds: float,
    device: str,
    batch_size: int,
    parity_status: str,
) -> dict[str, Any]:
    class_counts = {
        feature.stem: np.zeros(4, dtype=np.int64) for feature in AQUA_FEATURES
    }
    correlation_sums = {
        feature.stem: np.zeros(5, dtype=np.float64) for feature in AQUA_FEATURES
    }
    sample_parts: list[pd.DataFrame] = []
    total_rows = 0
    total_truncated = 0
    token_sum = 0.0
    for path in shard_paths:
        compact_columns = ["comment_id", "aqua_input_token_count", "aqua_input_truncated", "aqua_score_hard", "aqua_score_expected_raw"]
        for feature in AQUA_FEATURES:
            compact_columns.extend((label_column(feature.stem), expected_raw_column(feature.stem)))
        frame = pd.read_parquet(path, columns=compact_columns)
        total_rows += len(frame)
        total_truncated += int(frame["aqua_input_truncated"].sum())
        token_sum += float(frame["aqua_input_token_count"].sum())
        for feature in AQUA_FEATURES:
            hard = frame[label_column(feature.stem)].to_numpy(float)
            expected = frame[expected_raw_column(feature.stem)].to_numpy(float)
            class_counts[feature.stem] += np.bincount(hard.astype(int), minlength=4)
            correlation_sums[feature.stem] += np.asarray(
                [hard.sum(), expected.sum(), np.square(hard).sum(), np.square(expected).sum(), (hard * expected).sum()]
            )
        sample_mask = frame["comment_id"].astype(str).map(
            lambda value: int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big") % 1000 == 0
        )
        if sample_mask.any():
            sample_parts.append(frame.loc[sample_mask].drop(columns="comment_id"))
    if total_rows != expected_rows:
        raise ValueError(f"AQuA store has {total_rows} rows; expected {expected_rows}")
    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.read_parquet(shard_paths[0], columns=[
        "aqua_input_token_count", "aqua_score_hard", "aqua_score_expected_raw",
        *[column for feature in AQUA_FEATURES for column in (label_column(feature.stem), expected_raw_column(feature.stem))],
    ]).head(1)
    correlations: dict[str, float | None] = {}
    component_distributions: dict[str, Any] = {}
    warnings: list[str] = []
    for feature in AQUA_FEATURES:
        sx, sy, sxx, syy, sxy = correlation_sums[feature.stem]
        numerator = total_rows * sxy - sx * sy
        denominator = math.sqrt(max(0.0, (total_rows * sxx - sx * sx) * (total_rows * syy - sy * sy)))
        correlations[feature.stem] = float(numerator / denominator) if denominator else None
        counts = class_counts[feature.stem]
        if counts.max() / total_rows > 0.99:
            warnings.append(f"Severe class collapse for {feature.stem}: {int(counts.max())}/{total_rows}")
        component_distributions[feature.stem] = {
            "class_frequencies": {str(index): int(value) for index, value in enumerate(counts)},
            "expected_raw": _distribution_summary(sample[expected_raw_column(feature.stem)]),
        }
    return {
        "schema_version": AQUA_SCHEMA_VERSION,
        "rows": total_rows,
        "unique_keys": total_rows,
        "missing_rows": 0,
        "duplicate_rows": 0,
        "error_rows": 0,
        "truncated_rows": total_truncated,
        "truncated_fraction": total_truncated / total_rows if total_rows else 0.0,
        "mean_input_tokens": token_sum / total_rows if total_rows else 0.0,
        "sampled_quantile_rows": len(sample),
        "quantiles_are_deterministic_sample_estimates": True,
        "components": component_distributions,
        "hard_expected_correlations": correlations,
        "hard_score": _distribution_summary(sample["aqua_score_hard"]),
        "expected_score_raw": _distribution_summary(sample["aqua_score_expected_raw"]),
        "runtime_seconds": runtime_seconds,
        "throughput_rows_per_second": total_rows / runtime_seconds if runtime_seconds else None,
        "device": device,
        "batch_size": batch_size,
        "parity_test_status": parity_status,
        "probability_status": AQUA_PROBABILITY_STATUS,
        "warnings": warnings,
    }


def _write_qa_table(shard_paths: list[Path], destination: Path) -> None:
    """Write a privacy-safe sample and global score extremes without raw text."""
    selected_stems = ("justification", "question", "respect", "insult", "sarcasm")
    columns = [
        "story_id",
        "comment_id",
        "effective_text_hash",
        "aqua_score_hard",
        "aqua_score_expected_raw",
        "aqua_input_token_count",
        "aqua_input_truncated",
        *[
            column
            for stem in selected_stems
            for column in (label_column(stem), expected_raw_column(stem))
        ],
    ]
    random_records: list[dict[str, Any]] = []
    extreme_records: list[dict[str, Any]] = []
    for path in shard_paths:
        frame = pd.read_parquet(path, columns=columns)
        random_mask = frame["comment_id"].astype(str).map(
            lambda value: int.from_bytes(
                hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big"
            )
            % 100_000
            == 0
        )
        for record in frame.loc[random_mask].to_dict("records"):
            record["aqua_qa_reason"] = "deterministic_random"
            random_records.append(record)
        if not frame.empty:
            for index, reason in (
                (frame["aqua_score_expected_raw"].idxmin(), "story_minimum"),
                (frame["aqua_score_expected_raw"].idxmax(), "story_maximum"),
            ):
                record = frame.loc[index].to_dict()
                record["aqua_qa_reason"] = reason
                extreme_records.append(record)
    random_frame = pd.DataFrame(random_records).head(50)
    extremes = pd.DataFrame(extreme_records)
    if not extremes.empty:
        extremes = pd.concat(
            [
                extremes.nsmallest(10, "aqua_score_expected_raw").assign(
                    aqua_qa_reason="global_minimum"
                ),
                extremes.nlargest(10, "aqua_score_expected_raw").assign(
                    aqua_qa_reason="global_maximum"
                ),
            ],
            ignore_index=True,
        )
    qa = pd.concat([random_frame, extremes], ignore_index=True)
    qa = qa.drop_duplicates(["story_id", "comment_id", "aqua_qa_reason"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    qa.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def build_aqua_store(
    config: AquaBuildConfig,
    *,
    subprocess_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Build a resumable AQuA store using only subprocess communication."""
    started = time.monotonic()
    qa_path = config.data_root / "qa_summary" / f"year={config.year}" / "summary.json"
    if not qa_path.is_file():
        raise FileNotFoundError(qa_path)
    qa = json.loads(qa_path.read_text())
    validate_qa_summary(qa, allow_incomplete=config.allow_incomplete)
    if not config.runtime_python.is_file():
        raise FileNotFoundError(
            f"Missing isolated AQuA Python executable: {config.runtime_python}"
        )
    fingerprint = dataset_fingerprint(config.data_root, config.year)
    build_signature, identity = _build_identity(config, fingerprint)
    build_root = config.output_root / f"build={build_signature[:16]}"
    watermark = (
        "PRODUCTION"
        if not config.allow_incomplete and config.max_stories is None and config.require_parity
        else "PILOT_NOT_FOR_INFERENCE"
    )
    artifact_data = json.loads(config.artifact_manifest.read_text())
    manifest_path = build_root / "aqua_manifest.json"
    state = {
        "status": "running",
        "schema_version": AQUA_SCHEMA_VERSION,
        "build_signature": build_signature,
        "build_root": str(build_root.resolve()),
        "watermark": watermark,
        "identity": identity,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "source_dataset_fingerprint": fingerprint,
        "probability_status": AQUA_PROBABILITY_STATUS,
        "upstream": artifact_data.get("upstream"),
        "base_model": artifact_data.get("base_model"),
        "ordinal_class_mapping": artifact_data.get("ordinal_class_mapping"),
        "adapter_artifact_hashes": artifact_data.get("adapter_files"),
        "parity": artifact_data.get("parity"),
        "adapters": [feature.repository_adapter for feature in AQUA_FEATURES],
    }
    _atomic_json(state, manifest_path)
    candidates = _candidate_comments(config)
    if candidates.empty:
        raise ValueError("No eligible comments were found for AQuA inference")
    shard_paths: list[Path] = []
    new_shards = 0
    skipped_shards = 0
    sequential_fallbacks = 0
    processed_rows = 0
    grouped = candidates.groupby("story_id", sort=True)
    jobs: list[dict[str, str]] = []
    expected_by_destination: dict[str, pd.DataFrame] = {}
    for index, (story_id, story) in enumerate(grouped, start=1):
        month = int(story["month"].iloc[0])
        destination = build_root / f"year={config.year}" / f"month={month:02d}" / f"{story_id}.parquet"
        expected = story[["story_id", "comment_id", "effective_text_hash"]]
        shard_paths.append(destination)
        expected_by_destination[str(destination)] = expected
        if destination.exists() and not config.overwrite:
            validate_aqua_frame(
                pd.read_parquet(destination),
                expected=expected,
                build_signature=build_signature,
                watermark=watermark,
            )
            skipped_shards += 1
        else:
            input_path = (
                build_root
                / "inputs"
                / f"year={config.year}"
                / f"month={month:02d}"
                / f"{story_id}.parquet"
            )
            _atomic_parquet(
                story[["story_id", "comment_id", "effective_text", "effective_text_hash"]],
                input_path,
            )
            jobs.append({"input": str(input_path.resolve()), "output": str(destination.resolve())})
        processed_rows += len(story)
        if index % config.progress_every_stories == 0 or index == grouped.ngroups:
            print(
                f"AQuA export: {index:,}/{grouped.ngroups:,} stories | "
                f"rows={processed_rows:,}/{len(candidates):,} "
                f"pending={len(jobs):,} skipped={skipped_shards:,}",
                flush=True,
            )

    if jobs:
        jobs_path = build_root / "aqua_jobs.json"
        runtime_summary_path = build_root / "aqua_runtime_summary.json"
        _atomic_json({"build_signature": build_signature, "jobs": jobs}, jobs_path)
        command = build_runtime_command(
            config,
            build_signature=build_signature,
            watermark=watermark,
            job_manifest=jobs_path,
            summary_output=runtime_summary_path,
        )
        completed = subprocess_runner(
            command,
            check=False,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            env=_runtime_env(),
        )
        error_text = str(completed.stderr or "").lower()
        if (
            completed.returncode
            and config.execution_mode == "parallel"
            and config.sequential_fallback
            and (
                completed.returncode in {-9, 137}
                or any(marker in error_text for marker in ("out of memory", "cuda oom", "cuda error"))
            )
        ):
            remaining = [job for job in jobs if not Path(job["output"]).is_file()]
            if remaining:
                fallback_path = build_root / "aqua_jobs_sequential_fallback.json"
                fallback_summary = build_root / "aqua_runtime_summary_sequential.json"
                _atomic_json(
                    {"build_signature": build_signature, "jobs": remaining}, fallback_path
                )
                fallback_command = build_runtime_command(
                    config,
                    build_signature=build_signature,
                    watermark=watermark,
                    execution_mode="sequential",
                    job_manifest=fallback_path,
                    summary_output=fallback_summary,
                )
                completed = subprocess_runner(
                    fallback_command,
                    check=False,
                    stdout=None,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=Path(__file__).resolve().parents[1],
                    env=_runtime_env(),
                )
                sequential_fallbacks = len(remaining)
        if completed.returncode:
            raise RuntimeError(
                f"AQuA subprocess failed with exit {completed.returncode}: "
                f"{str(completed.stderr or '')[-4000:]}"
            )
        new_shards = len(jobs)

    created_destinations = {str(Path(job["output"]).resolve()) for job in jobs}
    for destination in shard_paths:
        if not destination.is_file():
            raise RuntimeError(f"AQuA runtime did not create {destination}")
        try:
            validate_aqua_frame(
                pd.read_parquet(destination),
                expected=expected_by_destination[str(destination)],
                build_signature=build_signature,
                watermark=watermark,
            )
        except Exception:
            if str(destination.resolve()) in created_destinations:
                destination.unlink(missing_ok=True)
            raise
    validation = _aggregate_validation(
        shard_paths,
        expected_rows=len(candidates),
        runtime_seconds=time.monotonic() - started,
        device=config.device,
        batch_size=config.batch_size,
        parity_status=artifact_data.get("parity", {}).get("status", "not_recorded"),
    )
    runtime_summary_paths = [
        path
        for path in (
            build_root / "aqua_runtime_summary.json",
            build_root / "aqua_runtime_summary_sequential.json",
        )
        if path.is_file()
    ]
    runtime_summaries = [json.loads(path.read_text()) for path in runtime_summary_paths]
    validation["runtime_processes"] = [
        {
            "path": str(path.resolve()),
            "device": summary.get("device"),
            "execution_mode": summary.get("execution_mode"),
            "rows": summary.get("rows"),
            "shards": summary.get("shards"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "batching": summary.get("batching"),
            "peak_memory_mib": summary.get("peak_memory_mib"),
            "python": summary.get("python"),
            "platform": summary.get("platform"),
            "packages": summary.get("packages"),
        }
        for path, summary in zip(runtime_summary_paths, runtime_summaries)
    ]
    validation["peak_memory_mib"] = max(
        (float(summary.get("peak_memory_mib", 0.0)) for summary in runtime_summaries),
        default=None,
    )
    validation["batching"] = [
        summary["batching"] for summary in runtime_summaries if summary.get("batching")
    ]
    qa_table_path = build_root / "aqua_qa_sample.csv"
    _write_qa_table(shard_paths, qa_table_path)
    validation["qa_table_path"] = str(qa_table_path.resolve())
    _atomic_json(validation, build_root / "aqua_validation.json")
    state.update(
        {
            "status": "complete",
            "rows": len(candidates),
            "stories": int(candidates["story_id"].nunique()),
            "new_story_checkpoints": new_shards,
            "skipped_story_checkpoints": skipped_shards,
            "sequential_fallback_story_checkpoints": sequential_fallbacks,
            "validation_path": str((build_root / "aqua_validation.json").resolve()),
            "runtime_processes": validation["runtime_processes"],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    _atomic_json(state, manifest_path)
    _atomic_json(state, config.output_root / "aqua_manifest.json")
    return state


def resolve_aqua_store(
    store: Path,
    *,
    year: int,
    source_fingerprint: str,
    require_production: bool,
) -> tuple[Path, dict[str, Any]]:
    store = Path(store)
    manifest_path = store / "aqua_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing AQuA manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("AQuA store is not complete")
    if manifest.get("schema_version") != AQUA_SCHEMA_VERSION:
        raise RuntimeError("AQuA store has an incompatible schema version")
    if manifest.get("source_dataset_fingerprint") != source_fingerprint:
        raise RuntimeError("AQuA store does not match the current source collection")
    if manifest.get("probability_status") != AQUA_PROBABILITY_STATUS:
        raise RuntimeError("AQuA store has an unsupported probability status")
    validation_path = Path(manifest.get("validation_path", ""))
    if not validation_path.is_file():
        raise FileNotFoundError("AQuA store is missing its completed validation report")
    validation = json.loads(validation_path.read_text())
    invalid_validation = {
        key: validation.get(key)
        for key in ("missing_rows", "duplicate_rows", "error_rows")
        if validation.get(key, 0) != 0
    }
    if invalid_validation:
        raise RuntimeError(f"AQuA store validation failed: {invalid_validation}")
    if validation.get("rows") != manifest.get("rows", validation.get("rows")):
        raise RuntimeError("AQuA manifest and validation row counts disagree")
    if validation.get("schema_version", AQUA_SCHEMA_VERSION) != AQUA_SCHEMA_VERSION:
        raise RuntimeError("AQuA validation report has an incompatible schema version")
    if validation.get("probability_status", AQUA_PROBABILITY_STATUS) != AQUA_PROBABILITY_STATUS:
        raise RuntimeError("AQuA validation report has an unsupported probability status")
    if require_production and manifest.get("watermark") != "PRODUCTION":
        raise RuntimeError("Inference requires a production AQuA store")
    build_root = Path(manifest.get("build_root", store))
    if not build_root.is_absolute():
        build_root = store / build_root
    year_root = build_root / f"year={year}"
    if not year_root.is_dir():
        raise FileNotFoundError(year_root)
    return build_root, manifest


def load_aqua_for_candidates(
    store: Path,
    candidates: pd.DataFrame,
    *,
    year: int,
    source_fingerprint: str,
    require_production: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load validated compact aliases and enforce exact candidate coverage."""
    import pyarrow.dataset as ds

    build_root, manifest = resolve_aqua_store(
        store,
        year=year,
        source_fingerprint=source_fingerprint,
        require_production=require_production,
    )
    raw_columns = [
        "story_id",
        "comment_id",
        "effective_text_hash",
        "aqua_score_hard",
        "aqua_score_expected_raw",
        "aqua_input_token_count",
        "aqua_input_truncated",
        "aqua_runtime_status",
        "aqua_probability_status",
        "aqua_build_signature",
        "aqua_watermark",
    ]
    for feature in AQUA_FEATURES:
        raw_columns.extend((label_column(feature.stem), expected_raw_column(feature.stem)))
    compact = ds.dataset(
        build_root / f"year={year}", format="parquet", partitioning=None
    ).to_table(columns=raw_columns).to_pandas()
    if compact.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("AQuA store contains duplicate keys")
    expected = candidates[["story_id", "comment_id", "effective_text"]].copy()
    expected["effective_text_hash"] = expected["effective_text"].astype(str).map(text_hash)
    compact = expected[["story_id", "comment_id"]].merge(
        compact,
        on=["story_id", "comment_id"],
        how="left",
        validate="one_to_one",
    )
    coverage = expected[["story_id", "comment_id", "effective_text_hash"]].merge(
        compact[["story_id", "comment_id", "effective_text_hash"]],
        on=["story_id", "comment_id"],
        how="outer",
        suffixes=("_expected", "_actual"),
        indicator=True,
        validate="one_to_one",
    )
    if not coverage["_merge"].eq("both").all():
        raise ValueError(f"AQuA candidate coverage mismatch: {coverage['_merge'].value_counts().to_dict()}")
    if not coverage["effective_text_hash_expected"].eq(coverage["effective_text_hash_actual"]).all():
        raise ValueError("AQuA candidate text hashes do not match")
    if not compact["aqua_build_signature"].eq(manifest["build_signature"]).all():
        raise ValueError("AQuA compact rows have inconsistent build signatures")
    if require_production and not compact["aqua_watermark"].eq("PRODUCTION").all():
        raise ValueError("AQuA compact rows are not production watermarked")
    if not compact["aqua_runtime_status"].eq("ok").all():
        raise ValueError("AQuA compact rows contain runtime failures")
    if not compact["aqua_probability_status"].eq(AQUA_PROBABILITY_STATUS).all():
        raise ValueError("AQuA compact rows are not explicitly uncalibrated")
    if (compact["aqua_input_token_count"] < 1).any():
        raise ValueError("AQuA compact rows contain invalid token counts")
    if not compact["aqua_score_hard"].between(0, 5).all() or not compact[
        "aqua_score_expected_raw"
    ].between(0, 5).all():
        raise ValueError("AQuA compact composite scores lie outside 0..5")
    for feature in AQUA_FEATURES:
        labels = compact[label_column(feature.stem)]
        expected_values = compact[expected_raw_column(feature.stem)]
        if not labels.isin(range(4)).all() or not expected_values.between(0, 3).all():
            raise ValueError(f"Invalid compact AQuA values for {feature.stem}")
        compact[expected_alias_column(feature.stem)] = expected_values
    compact["aqua_score_expected"] = compact["aqua_score_expected_raw"]
    selected = ["story_id", "comment_id", *downstream_feature_columns()]
    return compact[selected], manifest
