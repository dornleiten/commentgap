"""Audit and promote a complete pilot AQuA store without rerunning inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import pandas as pd

from aqua_runtime.schema import (
    AQUA_FEATURES,
    AQUA_PROBABILITY_STATUS,
    AQUA_SCHEMA_VERSION,
    AQUA_UPSTREAM_COMMIT,
    sha256_file,
)

from .aqua import (
    AquaBuildConfig,
    DEFAULT_AQUA_ARTIFACT_MANIFEST,
    _aggregate_validation,
    _build_identity,
    _candidate_comments,
    _write_qa_table,
    validate_aqua_frame,
)
from .features import (
    _atomic_json,
    _atomic_parquet,
    dataset_fingerprint,
    validate_qa_summary,
)


PILOT_WATERMARK = "PILOT_NOT_FOR_INFERENCE"
PRODUCTION_WATERMARK = "PRODUCTION"
PROMOTION_METHOD = "validated_metadata_promotion_without_reinference"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


@dataclass(frozen=True)
class AquaPromotionConfig:
    source_store: Path
    output_root: Path
    data_root: Path = Path("data/scrape_2025")
    year: int = 2025
    artifact_manifest: Path = DEFAULT_AQUA_ARTIFACT_MANIFEST
    overwrite: bool = False
    progress_every_stories: int = 25

    def __post_init__(self) -> None:
        for name in ("source_store", "output_root", "data_root", "artifact_manifest"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.progress_every_stories < 1:
            raise ValueError("progress_every_stories must be positive")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve_recorded_path(value: Any, *anchors: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest contains a missing or invalid recorded path")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(anchor / path for anchor in anchors)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _validated_parity(
    artifact_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    artifacts = _load_json(artifact_manifest, "AQuA artifact manifest")
    parity = artifacts.get("parity")
    if not isinstance(parity, dict) or parity.get("status") != "verified":
        raise RuntimeError(
            "Promotion requires verified parity in the artifact manifest. Run "
            "commentgap-aqua-parity with --update-artifact-manifest first."
        )
    fixture_hash = parity.get("fixture_sha256")
    if not _is_sha256(fixture_hash):
        raise ValueError("Verified parity is missing a valid fixture SHA-256")
    repository_root = Path(__file__).resolve().parents[1]
    fixture = _resolve_recorded_path(
        parity.get("fixture"), repository_root, artifact_manifest.resolve().parent
    )
    report = _load_json(fixture, "AQuA parity report")
    if sha256_file(fixture) != fixture_hash:
        raise ValueError("AQuA parity report does not match its recorded SHA-256")
    if report.get("status") != "verified":
        raise ValueError("AQuA parity report is not verified")
    if report.get("upstream_commit") != AQUA_UPSTREAM_COMMIT:
        raise ValueError("AQuA parity report targets the wrong upstream commit")
    if report.get("contains_raw_comment_text") is not False:
        raise ValueError("AQuA parity report must explicitly exclude raw comment text")
    rows = report.get("rows")
    if not isinstance(rows, int) or rows < 1:
        raise ValueError("AQuA parity report has no verified rows")
    hard_score_difference = float(
        report.get("maximum_hard_score_difference", float("inf"))
    )
    if not math.isfinite(hard_score_difference) or hard_score_difference > 1e-6:
        raise ValueError("AQuA parity hard-score comparison exceeds 1e-6")

    adapter_results = report.get("adapter_results", {})
    expected_adapters = {feature.repository_adapter for feature in AQUA_FEATURES}
    if set(adapter_results) != expected_adapters:
        raise ValueError("AQuA parity report does not cover exactly all adapters")
    if any(
        result.get("exact") is not True
        or result.get("rows") != rows
        or result.get("matching_hard_labels") != rows
        for result in adapter_results.values()
    ):
        raise ValueError("AQuA parity report contains an inexact adapter comparison")

    comparisons = report.get("comparisons", {})
    for comparison_name in ("repeat_cpu", "sequential"):
        comparison = comparisons.get(comparison_name, {})
        if (
            comparison.get("hard_labels_exact") is not True
            or comparison.get("logits_within_1e_6") is not True
            or not _is_sha256(comparison.get("sha256"))
        ):
            raise ValueError(
                f"AQuA parity report lacks verified {comparison_name} outputs"
            )
    return artifacts, report, fixture


def _validate_artifact_continuity(
    source_manifest: dict[str, Any], artifacts: dict[str, Any]
) -> None:
    comparisons = (
        ("upstream", source_manifest.get("upstream"), artifacts.get("upstream")),
        ("base model", source_manifest.get("base_model"), artifacts.get("base_model")),
        (
            "ordinal class mapping",
            source_manifest.get("ordinal_class_mapping"),
            artifacts.get("ordinal_class_mapping"),
        ),
        (
            "adapter artifact hashes",
            source_manifest.get("adapter_artifact_hashes"),
            artifacts.get("adapter_files"),
        ),
    )
    for label, source_value, verified_value in comparisons:
        if source_value != verified_value:
            raise ValueError(
                f"Refusing promotion because the verified {label} differs from "
                "the pilot build"
            )
    upstream = artifacts.get("upstream", {})
    if upstream.get("commit") != AQUA_UPSTREAM_COMMIT:
        raise ValueError("AQuA artifact manifest targets the wrong upstream commit")
    expected_adapters = {feature.repository_adapter for feature in AQUA_FEATURES}
    if set(artifacts.get("adapter_files", {})) != expected_adapters:
        raise ValueError("AQuA artifact manifest does not contain exactly all adapters")
    if set(source_manifest.get("adapters", [])) != expected_adapters:
        raise ValueError("Pilot manifest does not record exactly all AQuA adapters")


def _source_build_config(
    source_manifest: dict[str, Any], config: AquaPromotionConfig
) -> AquaBuildConfig:
    raw = source_manifest.get("config")
    if not isinstance(raw, dict):
        raise ValueError("Pilot manifest is missing its build configuration")
    allowed = {field.name for field in fields(AquaBuildConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Pilot manifest contains unknown configuration: {sorted(unknown)}")
    values = dict(raw)
    values.update(
        {
            "data_root": config.data_root,
            "output_root": config.output_root,
            "artifact_manifest": config.artifact_manifest,
            "allow_incomplete": False,
            "max_stories": None,
            "overwrite": config.overwrite,
            "progress_every_stories": config.progress_every_stories,
            "require_parity": True,
        }
    )
    return AquaBuildConfig(**values)


def _validate_source_manifest(
    source_manifest: dict[str, Any],
    *,
    source_store: Path,
    data_root: Path,
    year: int,
) -> tuple[Path, Path, dict[str, Any], str]:
    if source_manifest.get("status") != "complete":
        raise RuntimeError("Pilot AQuA store is not complete")
    if source_manifest.get("schema_version") != AQUA_SCHEMA_VERSION:
        raise ValueError("Pilot AQuA store has an incompatible schema version")
    if source_manifest.get("watermark") != PILOT_WATERMARK:
        raise ValueError("Source store is not pilot-watermarked")
    if source_manifest.get("probability_status") != AQUA_PROBABILITY_STATUS:
        raise ValueError("Pilot AQuA store has an unsupported probability status")
    source_config = source_manifest.get("config", {})
    if source_config.get("allow_incomplete") is not False:
        raise ValueError("Incomplete AQuA collections cannot be promoted")
    if source_config.get("max_stories") is not None:
        raise ValueError("Story-limited AQuA pilots cannot be promoted")
    if int(source_config.get("year", year)) != year:
        raise ValueError("Pilot year does not match --year")

    identity = source_manifest.get("identity")
    signature = source_manifest.get("build_signature")
    if not isinstance(identity, dict) or not isinstance(signature, str):
        raise ValueError("Pilot manifest lacks its build identity")
    recomputed_signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if recomputed_signature != signature:
        raise ValueError("Pilot manifest build signature does not recompute")
    expected_identity = {
        "schema_version": AQUA_SCHEMA_VERSION,
        "upstream_commit": AQUA_UPSTREAM_COMMIT,
        "source_fingerprint": source_manifest.get("source_dataset_fingerprint"),
        "year": year,
        **{
            key: source_config.get(key)
            for key in (
                "device",
                "execution_mode",
                "sequential_fallback",
                "batch_size",
                "adaptive_batches",
                "max_batch_tokens",
                "window_max_stories",
                "window_max_rows",
                "max_length",
                "allow_incomplete",
                "max_stories",
            )
        },
    }
    mismatches = {
        key: {"manifest": identity.get(key), "expected": value}
        for key, value in expected_identity.items()
        if identity.get(key) != value
    }
    if identity.get("require_parity", False) is not False:
        mismatches["require_parity"] = {
            "manifest": identity.get("require_parity"),
            "expected": False,
        }
    if mismatches:
        raise ValueError(f"Pilot configuration and identity disagree: {mismatches}")

    requirements_lock = _resolve_recorded_path(
        source_config.get("requirements_lock"), Path(__file__).resolve().parents[1]
    )
    if not requirements_lock.is_file():
        raise FileNotFoundError(f"Missing pilot requirements lock: {requirements_lock}")
    if identity.get("requirements_lock_sha256") != sha256_file(requirements_lock):
        raise ValueError("Pilot requirements lock no longer matches its recorded hash")

    fingerprint = dataset_fingerprint(data_root, year)
    if source_manifest.get("source_dataset_fingerprint") != fingerprint:
        raise ValueError("Pilot store does not match the current source collection")
    qa_path = data_root / "qa_summary" / f"year={year}" / "summary.json"
    qa = _load_json(qa_path, "collection QA summary")
    validate_qa_summary(qa, allow_incomplete=False)

    source_build_root = _resolve_recorded_path(
        source_manifest.get("build_root"), source_store
    )
    if not source_build_root.is_dir():
        raise FileNotFoundError(f"Missing pilot build directory: {source_build_root}")
    build_manifest_path = source_build_root / "aqua_manifest.json"
    build_manifest = _load_json(build_manifest_path, "pilot build manifest")
    if build_manifest != source_manifest:
        raise ValueError("Pilot root and retained build manifests disagree")

    validation_path = _resolve_recorded_path(
        source_manifest.get("validation_path"), source_build_root, source_store
    )
    validation = _load_json(validation_path, "pilot validation report")
    invalid = {
        key: validation.get(key)
        for key in ("missing_rows", "duplicate_rows", "error_rows")
        if validation.get(key, 0) != 0
    }
    if invalid:
        raise ValueError(f"Pilot validation report contains failures: {invalid}")
    if validation.get("rows") != source_manifest.get("rows"):
        raise ValueError("Pilot manifest and validation row counts disagree")
    if validation.get("schema_version", AQUA_SCHEMA_VERSION) != AQUA_SCHEMA_VERSION:
        raise ValueError("Pilot validation report has an incompatible schema")
    return source_build_root, validation_path, validation, fingerprint


def _assert_only_metadata_changed(source: pd.DataFrame, promoted: pd.DataFrame) -> None:
    metadata = ["aqua_build_signature", "aqua_watermark"]
    try:
        pd.testing.assert_frame_equal(
            source.drop(columns=metadata),
            promoted.drop(columns=metadata),
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError(
            "Promoted shard differs from the pilot in prediction data"
        ) from exc


def promote_aqua_store(config: AquaPromotionConfig) -> dict[str, Any]:
    """Promote a complete, verified pilot store by rewriting metadata only."""
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    source_store = config.source_store.resolve()
    source_manifest_path = source_store / "aqua_manifest.json"
    source_manifest = _load_json(source_manifest_path, "pilot root manifest")
    (
        source_build_root,
        source_validation_path,
        source_validation,
        fingerprint,
    ) = _validate_source_manifest(
        source_manifest,
        source_store=source_store,
        data_root=config.data_root,
        year=config.year,
    )
    artifacts, parity_report, parity_fixture = _validated_parity(
        config.artifact_manifest
    )
    _validate_artifact_continuity(source_manifest, artifacts)

    build_config = _source_build_config(source_manifest, config)
    build_signature, identity = _build_identity(build_config, fingerprint)
    if build_signature == source_manifest["build_signature"]:
        raise RuntimeError(
            "Production and pilot build signatures collide; update the code so "
            "require_parity is part of the AQuA build identity"
        )
    build_root = config.output_root / f"build={build_signature[:16]}"
    if build_root.resolve() == source_build_root.resolve():
        raise RuntimeError("Promotion destination resolves to the pilot build directory")

    candidates = _candidate_comments(build_config)
    if candidates.empty:
        raise ValueError("No eligible comments were found for AQuA promotion")
    expected_rows = len(candidates)
    expected_stories = int(candidates["story_id"].nunique())
    if source_manifest.get("rows") != expected_rows:
        raise ValueError(
            f"Pilot has {source_manifest.get('rows')} rows; current collection has "
            f"{expected_rows} eligible rows"
        )
    if source_manifest.get("stories") != expected_stories:
        raise ValueError(
            f"Pilot has {source_manifest.get('stories')} stories; current collection "
            f"has {expected_stories} eligible stories"
        )

    promotion_audit = {
        "method": PROMOTION_METHOD,
        "started_at_utc": started_at,
        "source_store": str(source_store),
        "source_build_root": str(source_build_root.resolve()),
        "source_build_signature": source_manifest["build_signature"],
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_build_manifest_path": str(
            (source_build_root / "aqua_manifest.json").resolve()
        ),
        "source_build_manifest_sha256": sha256_file(
            source_build_root / "aqua_manifest.json"
        ),
        "source_validation_path": str(source_validation_path.resolve()),
        "source_validation_sha256": sha256_file(source_validation_path),
        "verified_parity_fixture": str(parity_fixture.resolve()),
        "verified_parity_fixture_sha256": sha256_file(parity_fixture),
        "verified_parity_rows": parity_report["rows"],
        "numeric_predictions_recomputed": False,
        "per_shard_full_validation": True,
        "only_signature_and_watermark_rewritten": True,
    }
    state = {
        "status": "running",
        "schema_version": AQUA_SCHEMA_VERSION,
        "build_signature": build_signature,
        "build_root": str(build_root.resolve()),
        "watermark": PRODUCTION_WATERMARK,
        "identity": identity,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(build_config).items()
        },
        "source_dataset_fingerprint": fingerprint,
        "probability_status": AQUA_PROBABILITY_STATUS,
        "upstream": artifacts.get("upstream"),
        "base_model": artifacts.get("base_model"),
        "ordinal_class_mapping": artifacts.get("ordinal_class_mapping"),
        "adapter_artifact_hashes": artifacts.get("adapter_files"),
        "parity": artifacts.get("parity"),
        "adapters": [feature.repository_adapter for feature in AQUA_FEATURES],
        "promotion": promotion_audit,
    }
    manifest_path = build_root / "aqua_manifest.json"
    _atomic_json(state, manifest_path)

    shard_paths: list[Path] = []
    new_shards = 0
    skipped_shards = 0
    processed_rows = 0
    grouped = candidates.groupby("story_id", sort=True)
    for index, (story_id, story) in enumerate(grouped, start=1):
        month = int(story["month"].iloc[0])
        relative = Path(f"year={config.year}") / f"month={month:02d}" / f"{story_id}.parquet"
        source_path = source_build_root / relative
        destination = build_root / relative
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing pilot shard: {source_path}")
        expected = story[["story_id", "comment_id", "effective_text_hash"]]
        source_frame = pd.read_parquet(source_path)
        validate_aqua_frame(
            source_frame,
            expected=expected,
            build_signature=source_manifest["build_signature"],
            watermark=PILOT_WATERMARK,
        )
        if destination.exists() and not config.overwrite:
            promoted = pd.read_parquet(destination)
            validate_aqua_frame(
                promoted,
                expected=expected,
                build_signature=build_signature,
                watermark=PRODUCTION_WATERMARK,
            )
            _assert_only_metadata_changed(source_frame, promoted)
            skipped_shards += 1
        else:
            promoted = source_frame.copy()
            promoted["aqua_build_signature"] = build_signature
            promoted["aqua_watermark"] = PRODUCTION_WATERMARK
            _assert_only_metadata_changed(source_frame, promoted)
            validate_aqua_frame(
                promoted,
                expected=expected,
                build_signature=build_signature,
                watermark=PRODUCTION_WATERMARK,
            )
            _atomic_parquet(promoted, destination)
            try:
                persisted = pd.read_parquet(destination)
                validate_aqua_frame(
                    persisted,
                    expected=expected,
                    build_signature=build_signature,
                    watermark=PRODUCTION_WATERMARK,
                )
                _assert_only_metadata_changed(source_frame, persisted)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            new_shards += 1
        shard_paths.append(destination)
        processed_rows += len(story)
        if index % config.progress_every_stories == 0 or index == grouped.ngroups:
            elapsed = time.monotonic() - started
            rate = processed_rows / elapsed if elapsed else 0.0
            eta = (expected_rows - processed_rows) / rate if rate else 0.0
            print(
                f"AQuA promotion: {index:,}/{grouped.ngroups:,} stories | "
                f"rows={processed_rows:,}/{expected_rows:,} "
                f"new={new_shards:,} skipped={skipped_shards:,} | "
                f"elapsed={_format_duration(elapsed)} rate={rate:,.1f} rows/s "
                f"eta={_format_duration(eta)}",
                flush=True,
            )

    source_pipeline_elapsed = float(
        source_manifest.get("elapsed_seconds", 0.0) or 0.0
    )
    source_runtime_seconds = float(
        source_validation.get("runtime_seconds", source_pipeline_elapsed) or 0.0
    )
    validation = _aggregate_validation(
        shard_paths,
        expected_rows=expected_rows,
        runtime_seconds=source_runtime_seconds,
        device=build_config.device,
        batch_size=build_config.batch_size,
        parity_status="verified",
    )
    qa_table_path = build_root / "aqua_qa_sample.csv"
    _write_qa_table(shard_paths, qa_table_path)
    validation["qa_table_path"] = str(qa_table_path.resolve())
    promotion_audit["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    promotion_audit["elapsed_seconds"] = time.monotonic() - started
    validation.update(
        {
            "runtime_seconds_provenance": "recorded_by_source_pilot_manifest",
            "promotion_elapsed_seconds": promotion_audit["elapsed_seconds"],
            "promotion_throughput_rows_per_second": (
                expected_rows / promotion_audit["elapsed_seconds"]
                if promotion_audit["elapsed_seconds"]
                else None
            ),
            "promotion": promotion_audit,
            "runtime_processes": source_manifest.get("runtime_processes", []),
            "peak_memory_mib": source_validation.get("peak_memory_mib"),
            "batching": source_validation.get("batching", []),
        }
    )
    validation_path = build_root / "aqua_validation.json"
    _atomic_json(validation, validation_path)

    state.update(
        {
            "status": "complete",
            "rows": expected_rows,
            "stories": expected_stories,
            "new_story_checkpoints": new_shards,
            "skipped_story_checkpoints": skipped_shards,
            "sequential_fallback_story_checkpoints": 0,
            "validation_path": str(validation_path.resolve()),
            "runtime_processes": source_manifest.get("runtime_processes", []),
            "source_pipeline_elapsed_seconds": source_pipeline_elapsed,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    _atomic_json(state, manifest_path)
    _atomic_json(state, config.output_root / "aqua_manifest.json")
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentgap-aqua-promote",
        description=(
            "Audit and promote a complete pilot AQuA store after parity verification, "
            "without rerunning model inference."
        ),
    )
    parser.add_argument("--source-store", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/scrape_2025"))
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=Path("aqua_runtime/artifacts.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every-stories", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = promote_aqua_store(
        AquaPromotionConfig(
            source_store=args.source_store,
            output_root=args.output_root,
            data_root=args.data_root,
            year=args.year,
            artifact_manifest=args.artifact_manifest,
            overwrite=args.overwrite,
            progress_every_stories=args.progress_every_stories,
        )
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
