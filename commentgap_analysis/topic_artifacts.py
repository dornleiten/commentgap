"""Reproducibility, signatures, and cache helpers for topic artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, Iterable, Mapping, Sequence
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TOPIC_COMPLETION_SCHEMA_VERSION = 1
TOPIC_COMPLETION_MANIFEST = "topic_agenda_completion_manifest.json"


def topic_calculation_config(
    *, score_policies: Mapping[str, object], policy_specs: Sequence[object],
    analysis_models: Sequence[str] = ("spline", "power_law"),
    distance_measures: Sequence[str] = ("cosine", "jensen_shannon"),
    seed: int = 2025, tie_draws: int = 10, random_draws: int = 25,
    oracle_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the canonical notebook-15 calculation configuration."""
    return {
        # These are sets of required result families; canonical ordering keeps
        # equivalent notebook runs from differing only in model iteration order.
        "analysis_models": sorted(set(analysis_models)),
        "distance_measures": sorted(set(distance_measures)),
        "score_policies": dict(score_policies),
        "policy_specs": [repr(spec) for spec in policy_specs],
        "seed": seed,
        "tie_draws": tie_draws,
        "random_draws": random_draws,
        "oracle": dict(oracle_kwargs or {
            "max_iterations": 5, "n_starts": 8, "random_state": seed,
            "n_perturbations": 4, "perturbation_fraction": 0.10,
            "n_local_moves": 2, "allow_invalid_placement": True,
            "exact_max_valid_comments": 8, "progress_every": 25,
        }),
    }


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions(names: Iterable[str]) -> dict[str, str]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def signature_equal(path: Path | str, signature: Mapping[str, object]) -> bool:
    try:
        return Path(path).exists() and json.loads(Path(path).read_text()) == signature
    except (OSError, ValueError, TypeError):
        return False


def write_signature(path: Path | str, signature: Mapping[str, object]) -> None:
    Path(path).write_text(json.dumps(signature, indent=2, sort_keys=True, default=str) + "\n")


def _atomic_json_write(path: Path | str, payload: Mapping[str, object]) -> None:
    """Write a JSON marker atomically, leaving the old marker on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".partial", delete=False,
    )
    try:
        with temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True, default=str)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary.name, path)
    except BaseException:
        Path(temporary.name).unlink(missing_ok=True)
        raise


def _relative_artifact_path(output_root: Path, path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(output_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Topic artifact is outside output root: {path}") from exc


def _artifact_record(output_root: Path, path: Path | str) -> dict[str, object]:
    path = Path(path)
    record: dict[str, object] = {
        "path": _relative_artifact_path(output_root, path),
        "present": path.is_file(),
    }
    if path.is_file():
        record["sha256"] = file_sha256(path)
        record["size"] = path.stat().st_size
    else:
        record["sha256"] = None
        record["size"] = None
    return record


def topic_calculation_artifacts(
    output_root: Path | str, *,
    analysis_models: Sequence[str] = ("spline", "power_law"),
    distance_measures: Sequence[str] = ("cosine", "jensen_shannon"),
) -> dict[str, Path]:
    """Return the complete, explicit artifact inventory produced by notebook 15.

    Presentation outputs written by notebook 16 are deliberately absent.  The
    inventory is shared by the producer and consumer so a completion marker
    cannot be published with an accidentally incomplete or over-broad file set.
    """
    output_root = Path(output_root)
    artifacts: dict[str, Path] = {
        "attention_curve": output_root / "topic_policy_vote_attention_curve.csv",
        "attention_fit": output_root / "topic_policy_vote_attention_fit.json",
        "attention_comparison": output_root / "topic_policy_vote_attention_comparison.csv",
        "attention_cv": output_root / "topic_policy_vote_attention_cv.csv",
        "attention_tuning": output_root / "topic_policy_vote_attention_tuning.csv",
        "baseline_story": output_root / "topic_agenda_baseline_story_metrics.parquet",
        "baseline_summary": output_root / "topic_agenda_baseline_summary.csv",
        "rarefaction_story": output_root / "topic_agenda_rarefaction_story_metrics.parquet",
        "rarefaction_summary": output_root / "topic_agenda_rarefaction_summary.csv",
        "attention_sensitivity": output_root / "topic_policy_fitted_attention_sensitivity.csv",
    }
    for model in analysis_models:
        prefix = f"vote_{model}_"
        stem = f"{model}"
        artifacts.update({
            f"{stem}_visible_distributions": output_root / f"topic_policy_{prefix}visible_distributions.parquet",
            f"{stem}_metrics_draws": output_root / f"topic_policy_{prefix}metrics_draws.parquet",
            f"{stem}_metrics": output_root / f"topic_policy_{prefix}metrics.parquet",
            f"{stem}_concentration_story": output_root / f"{prefix}topic_policy_concentration_story_metrics.parquet",
            f"{stem}_concentration_summary": output_root / f"{prefix}topic_policy_concentration_summary.csv",
            f"{stem}_exposure_story": output_root / f"{prefix}topic_policy_exposure_coverage_story_metrics.parquet",
            f"{stem}_exposure_summary": output_root / f"{prefix}topic_policy_exposure_coverage_summary.csv",
            f"{stem}_reference_targets": output_root / f"{prefix}topic_policy_reference_target_metrics.parquet",
            f"{stem}_run_metadata": output_root / f"topic_policy_{prefix}run_metadata.json",
        })
        for distance in distance_measures:
            oracle_stem = "cosine" if distance == "cosine" else "js"
            oracle = f"topic_policy_{prefix}{oracle_stem}_oracle"
            for artifact in ("visible_distributions", "metrics", "summary", "skipped", "exact_validation"):
                suffix = "parquet" if artifact in {"visible_distributions", "metrics"} else "csv"
                artifacts[f"{stem}_{distance}_oracle_{artifact}"] = output_root / f"{oracle}_{artifact}.{suffix}"
    return artifacts


def topic_calculation_source_manifests(
    output_root: Path | str, *,
    analysis_models: Sequence[str] = ("spline", "power_law"),
    distance_measures: Sequence[str] = ("cosine", "jensen_shannon"),
) -> dict[str, Path]:
    """Return provenance cache manifests, whose presence is recorded when available."""
    output_root = Path(output_root)
    manifests = {
        "attention_cache_manifest": output_root / "topic_policy_vote_attention_cache_manifest.json",
        "baseline_cache_manifest": output_root / "topic_agenda_baseline_cache_manifest.json",
        "rarefaction_cache_manifest": output_root / "topic_agenda_rarefaction_cache_manifest.json",
    }
    for model in analysis_models:
        prefix = f"vote_{model}_"
        manifests[f"{model}_visible_cache_manifest"] = output_root / f"topic_policy_{prefix}cache_manifest.json"
        manifests[f"{model}_metrics_cache_manifest"] = output_root / f"topic_policy_{prefix}metrics_cache_manifest.json"
        for distance in distance_measures:
            oracle_stem = "cosine" if distance == "cosine" else "js"
            manifests[f"{model}_{distance}_oracle_cache_manifest"] = output_root / (
                f"topic_policy_{prefix}{oracle_stem}_oracle_visible_distributions.manifest.json"
            )
    return manifests


def begin_topic_run(
    output_root: Path | str, provenance: Mapping[str, object], *,
    manifest_name: str = TOPIC_COMPLETION_MANIFEST,
) -> str:
    """Mark a topic calculation run in progress before any result is written."""
    output_root = Path(output_root)
    token = uuid.uuid4().hex
    _atomic_json_write(output_root / manifest_name, {
        "schema_version": TOPIC_COMPLETION_SCHEMA_VERSION,
        "status": "in_progress",
        "run_token": token,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": dict(provenance),
    })
    return token


def publish_topic_run(
    output_root: Path | str, *, run_token: str,
    provenance: Mapping[str, object],
    required_artifacts: Mapping[str, Path | str],
    optional_artifacts: Mapping[str, Path | str] | None = None,
    manifest_name: str = TOPIC_COMPLETION_MANIFEST,
) -> dict[str, object]:
    """Publish a complete topic run only after every required output is present."""
    output_root = Path(output_root)
    manifest_path = output_root / manifest_name
    try:
        current = json.loads(manifest_path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("Topic run has no readable in-progress completion marker") from exc
    if current.get("status") != "in_progress":
        raise RuntimeError("Topic run completion marker is not in progress")
    if current.get("run_token") != run_token:
        raise RuntimeError("Topic run completion marker belongs to a different run")
    optional_artifacts = optional_artifacts or {}
    missing = [name for name, path in required_artifacts.items() if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"Cannot publish incomplete topic run; missing artifacts: {', '.join(sorted(missing))}")
    records = {name: _artifact_record(output_root, path) for name, path in required_artifacts.items()}
    optional_records = {name: _artifact_record(output_root, path) for name, path in optional_artifacts.items()}
    payload = {
        "schema_version": TOPIC_COMPLETION_SCHEMA_VERSION,
        "status": "complete",
        "run_token": run_token,
        "started_at_utc": current.get("started_at_utc"),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": dict(provenance),
        "artifacts": records,
        "optional_artifacts": optional_records,
    }
    _atomic_json_write(manifest_path, payload)
    return payload


def _check_schema(path: Path, columns: Sequence[str]) -> bool:
    if not columns:
        return True
    try:
        if path.suffix == ".parquet":
            available = set(pq.read_schema(path).names)
        else:
            available = set(pd.read_csv(path, nrows=0).columns)
    except (OSError, ValueError, TypeError, pa.ArrowInvalid):
        return False
    return set(columns).issubset(available)


def validate_topic_run(
    output_root: Path | str, *,
    expected_provenance: Mapping[str, object] | None = None,
    required_artifacts: Mapping[str, Path | str],
    optional_artifacts: Mapping[str, Path | str] | None = None,
    schema_checks: Mapping[str, Sequence[str]] | None = None,
    allow_older_results: bool = False,
    allow_legacy_missing: bool = False,
    manifest_name: str = TOPIC_COMPLETION_MANIFEST,
) -> dict[str, object]:
    """Validate a complete topic run and its hashes before consumer reads.

    A known in-progress marker or any recorded hash mismatch is always an
    error.  The older-results option only relaxes provenance matching.  Runs
    created before completion markers existed can be inspected explicitly with
    ``allow_legacy_missing`` after basic file/schema checks.
    """
    output_root = Path(output_root)
    optional_artifacts = optional_artifacts or {}
    schema_checks = schema_checks or {}
    manifest_path = output_root / manifest_name
    if not manifest_path.exists():
        if not (allow_older_results and allow_legacy_missing):
            raise RuntimeError(f"Topic run completion manifest is missing: {manifest_path}")
        for name, path in required_artifacts.items():
            path = Path(path)
            if not path.is_file() or not _check_schema(path, schema_checks.get(name, ())):
                raise RuntimeError(f"Legacy topic run is missing or has invalid artifact: {name}")
        return {"status": "legacy_complete", "manifest": str(manifest_path)}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Topic run completion manifest is unreadable: {manifest_path}") from exc
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Topic run is not complete (status={manifest.get('status')!r})")
    if manifest.get("schema_version") != TOPIC_COMPLETION_SCHEMA_VERSION:
        raise RuntimeError("Unsupported topic run completion manifest schema")
    records = manifest.get("artifacts")
    if not isinstance(records, Mapping):
        raise RuntimeError("Topic run completion manifest has no artifact inventory")
    for name, expected_path in required_artifacts.items():
        record = records.get(name)
        if not isinstance(record, Mapping) or not record.get("present"):
            raise RuntimeError(f"Topic run completion inventory is missing required artifact: {name}")
        path = Path(expected_path)
        if record.get("path") != _relative_artifact_path(output_root, path):
            raise RuntimeError(f"Topic run artifact path changed: {name}")
        if not path.is_file() or record.get("sha256") != file_sha256(path):
            raise RuntimeError(f"Topic run artifact hash mismatch: {name}")
        if record.get("size") != path.stat().st_size:
            raise RuntimeError(f"Topic run artifact size mismatch: {name}")
        if not _check_schema(path, schema_checks.get(name, ())):
            raise RuntimeError(f"Topic run artifact schema mismatch: {name}")
    optional_records = manifest.get("optional_artifacts", {})
    if not isinstance(optional_records, Mapping):
        raise RuntimeError("Topic run completion manifest has invalid optional inventory")
    for name, expected_path in optional_artifacts.items():
        record = optional_records.get(name)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Topic run completion inventory lacks optional artifact record: {name}")
        path = Path(expected_path)
        if record.get("path") != _relative_artifact_path(output_root, path):
            raise RuntimeError(f"Topic run optional artifact path changed: {name}")
        present = path.is_file()
        if bool(record.get("present")) != present:
            raise RuntimeError(f"Topic run optional artifact presence changed: {name}")
        if present and record.get("sha256") != file_sha256(path):
            raise RuntimeError(f"Topic run optional artifact hash mismatch: {name}")
        if present and record.get("size") != path.stat().st_size:
            raise RuntimeError(f"Topic run optional artifact size mismatch: {name}")
    if expected_provenance is not None and not allow_older_results:
        stored = manifest.get("provenance", {})
        stored_reuse = stored.get("reuse", {}) if isinstance(stored, Mapping) else {}
        if isinstance(stored_reuse, Mapping) and stored_reuse.get("allow_older_run_results"):
            raise RuntimeError(
                "Topic run was published from older calculations; set allow_older_results=True"
            )
        for key, value in expected_provenance.items():
            if stored.get(key) != value:
                raise RuntimeError(f"Topic run provenance mismatch: {key}")
    return manifest


def read_cached_parquet(
    path: Path | str, *, manifest_path: Path | str,
    signature: Mapping[str, object], required_columns: Sequence[str] = (),
) -> pd.DataFrame | None:
    if not signature_equal(manifest_path, signature) or not Path(path).exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError, pa.ArrowInvalid):
        return None
    if frame.empty or not set(required_columns).issubset(frame.columns):
        return None
    return frame


def read_existing_run_parquet(
    path: Path | str, *, required_columns: Sequence[str] = (),
    reuse_existing: bool = False,
) -> pd.DataFrame | None:
    """Load an existing run artifact when it has the expected basic schema.

    This is an explicit opt-in for inspecting older calculations. Normal
    execution must use ``read_cached_parquet`` with a matching signature.
    The stored provenance is never rewritten by this loader.
    """

    path = Path(path)
    if not reuse_existing or not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError, TypeError, pa.ArrowInvalid):
        return None
    if frame.empty or not set(required_columns).issubset(frame.columns):
        return None
    return frame


def reusable_artifacts(
    paths: Iterable[Path | str], *, manifest_path: Path | str,
    signature: Mapping[str, object], force_recompute: bool = False,
    allow_older_results: bool = False,
) -> bool:
    """Require a complete artifact set and current provenance unless opted out."""
    return (
        not force_recompute
        and all(Path(path).is_file() for path in paths)
        and (allow_older_results or signature_equal(manifest_path, signature))
    )


def input_signature(
    *, repo_root: Path | str, membership_path: Path | str,
    comments_path: Path | str, topic_root: Path | str,
) -> dict[str, object]:
    repo_root, membership_path, comments_path, topic_root = map(
        Path, (repo_root, membership_path, comments_path, topic_root)
    )
    split_path = repo_root / "model_output/selection_2025/model_data/master_article_split.parquet"
    model_manifest = topic_root / "topic_model_manifest.json"
    return {
        "membership_path": str(membership_path),
        "membership_sha256": file_sha256(membership_path),
        "comments_path": str(comments_path),
        "comments_sha256": file_sha256(comments_path),
        "split_path": str(split_path),
        "split_sha256": file_sha256(split_path) if split_path.exists() else "missing",
        "model_manifest_path": str(model_manifest),
        "model_manifest_sha256": file_sha256(model_manifest) if model_manifest.exists() else "missing",
        "code_revision": code_revision(),
        "topic_modeling_code_hash": file_sha256(repo_root / "commentgap_analysis/topic_modeling.py"),
        "topic_metrics_code_hash": file_sha256(repo_root / "commentgap_analysis/topic_metrics.py"),
        "topic_policy_code_hash": file_sha256(repo_root / "commentgap_analysis/topic_policy.py"),
        "policy_code_hash": file_sha256(repo_root / "commentgap_analysis/forum_scores.py"),
        "packages": package_versions(["bertopic", "umap-learn", "hdbscan", "scikit-learn", "pandas", "numpy"]),
    }


def compute_draw_metrics_in_batches(
    visible_path: Path | str, metrics_path: Path | str, *,
    compute_metrics: Callable[[pd.DataFrame], pd.DataFrame], batch_size: int = 25_000,
) -> None:
    """Compute and atomically assemble draw metrics, resuming complete parts."""
    visible_path, metrics_path = Path(visible_path), Path(metrics_path)
    parquet = pq.ParquetFile(visible_path)
    total_rows = parquet.metadata.num_rows
    total_batches = (total_rows + batch_size - 1) // batch_size
    parts_path = metrics_path.with_suffix(".parts")
    parts_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for batch_number, batch in enumerate(parquet.iter_batches(batch_size=batch_size), start=1):
        part_path = parts_path / f"part_{batch_number:05d}.parquet"
        if part_path.exists():
            continue
        metrics_batch = compute_metrics(batch.to_pandas())
        temporary_part = part_path.with_suffix(".partial.parquet")
        pq.write_table(pa.Table.from_pandas(metrics_batch, preserve_index=False), temporary_part)
        temporary_part.replace(part_path)
        elapsed = time.perf_counter() - started
        rate = batch_number / elapsed if elapsed else 0.0
        remaining = (total_batches - batch_number) / rate if rate else float("nan")
        eta = f"{remaining / 60:.1f} min" if remaining == remaining else "n/a"
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
            print(f"Computed draw metrics batch {batch_number}/{total_batches}: {len(metrics_batch):,} rows; ETA {eta}", flush=True)
    temporary_output = metrics_path.with_suffix(".partial.parquet")
    if temporary_output.exists():
        temporary_output.unlink()
    writer = None
    try:
        for batch_number in range(1, total_batches + 1):
            table = pq.read_table(parts_path / f"part_{batch_number:05d}.parquet")
            if writer is None:
                writer = pq.ParquetWriter(temporary_output, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary_output.replace(metrics_path)


def required(path: Path | str, *, message: str = "Required artifact is missing") -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")
    return path
