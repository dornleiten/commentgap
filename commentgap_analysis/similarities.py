"""Resumable scalar similarities derived from a completed embedding store.

The expensive transformer pass is deliberately separate from this module.  Each
comment/article vector is read from the versioned embedding store and used only
for the three semantic quantities required by the preference models:

* mean cosine similarity to the three closest article passages;
* novelty relative to strictly earlier comments; and
* novelty relative to strictly earlier root comments.

For ordinary discussions one exact comment-comment matrix is shared by both
novelty definitions.  Exceptionally large discussions use separate incremental
HNSW indexes because their all-comment and root-only candidate histories differ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from .embeddings import (
    _analysis_package_versions,
    _format_duration,
    _multi_year_fingerprint,
    _safe_model_key,
)
from .features import (
    _novelty_hnsw,
    article_similarity_top3,
    dataset_fingerprint,
    validate_approximate_novelty,
)
from .nlp import DEFAULT_EMBEDDING_MODEL_ID, DEFAULT_EMBEDDING_MODEL_REVISION


BASE_SEED = 20260813


@dataclass(frozen=True)
class SimilarityBuildConfig:
    """Configuration that uniquely identifies a scalar-similarity build."""

    data_root: Path = Path("data/scrape_2025")
    embedding_root: Path = Path("model_output/selection_2025/embeddings")
    embedding_store: Path | None = None
    output_root: Path = Path("model_output/selection_2025/similarities")
    years: tuple[int, ...] = (2025,)
    model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    revision: str | None = None
    exact_novelty_threshold: int = 5_000
    hnsw_validation_sample: int = 100
    hnsw_required_recall: float = 0.95
    seed: int = BASE_SEED
    max_stories: int | None = None
    allow_subset_source: bool = False
    overwrite: bool = False
    progress_every_stories: int = 100

    def __post_init__(self) -> None:
        for name in ("data_root", "embedding_root", "output_root"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.embedding_store is not None:
            object.__setattr__(self, "embedding_store", Path(self.embedding_store))
        object.__setattr__(self, "years", tuple(sorted(set(map(int, self.years)))))
        if self.model_id == DEFAULT_EMBEDDING_MODEL_ID and self.revision is None:
            object.__setattr__(self, "revision", DEFAULT_EMBEDDING_MODEL_REVISION)
        if not self.years:
            raise ValueError("At least one target year is required")
        if self.exact_novelty_threshold < 1:
            raise ValueError("exact_novelty_threshold must be positive")
        if self.hnsw_validation_sample < 1:
            raise ValueError("hnsw_validation_sample must be positive")
        if not 0 <= self.hnsw_required_recall <= 1:
            raise ValueError("hnsw_required_recall must be between zero and one")
        if self.max_stories is not None and self.max_stories < 1:
            raise ValueError("max_stories must be positive")
        if self.progress_every_stories < 1:
            raise ValueError("progress_every_stories must be positive")


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_embedding_store(
    root: Path,
    *,
    model_id: str,
    revision: str | None,
    years: tuple[int, ...],
    allow_subset: bool,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one compatible completed embedding-store manifest."""
    root = Path(root)
    direct = root / "embedding_manifest.json"
    manifests = [direct] if direct.exists() else sorted(root.glob("model=*/build=*/embedding_manifest.json"))
    compatible: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[str] = []
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        model = manifest.get("model", {})
        source_years = set(map(int, manifest.get("source", {}).get("years", [])))
        reasons: list[str] = []
        if manifest.get("status") != "complete":
            reasons.append("not complete")
        if model.get("model_id") != model_id:
            reasons.append("model mismatch")
        if revision and model.get("resolved_revision") != revision:
            reasons.append("revision mismatch")
        if not set(years).issubset(source_years):
            reasons.append("missing target year")
        watermark = str(manifest.get("watermark", ""))
        if not allow_subset and watermark != "COMPLETE_SOURCE":
            reasons.append(f"watermark={watermark or 'missing'}")
        if reasons:
            rejected.append(f"{manifest_path}: {', '.join(reasons)}")
        else:
            compatible.append((manifest_path.parent, manifest))
    if not compatible:
        detail = "\n".join(rejected[-10:]) or f"No manifests found below {root}"
        raise FileNotFoundError(
            "No compatible completed embedding store was found. "
            f"Expected model={model_id}, revision={revision}, years={years}.\n{detail}"
        )
    if len(compatible) > 1:
        choices = "\n".join(str(path) for path, _ in compatible)
        raise RuntimeError(
            "Multiple compatible embedding stores were found; pass --embedding-store "
            f"with the intended run root:\n{choices}"
        )
    return compatible[0]


def resolve_similarity_store(
    root: Path,
    *,
    model_id: str,
    revision: str | None,
    year: int,
    require_complete_source: bool,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one compatible completed scalar-similarity store."""
    root = Path(root)
    direct = root / "similarity_manifest.json"
    manifests = [direct] if direct.exists() else sorted(root.glob("model=*/build=*/similarity_manifest.json"))
    compatible: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[str] = []
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        model = manifest.get("embedding_model", {})
        reasons: list[str] = []
        if manifest.get("status") != "complete":
            reasons.append("not complete")
        if model.get("model_id") != model_id:
            reasons.append("model mismatch")
        if revision and model.get("resolved_revision") != revision:
            reasons.append("revision mismatch")
        if int(year) not in set(map(int, manifest.get("target_years", []))):
            reasons.append("missing target year")
        if require_complete_source and manifest.get("watermark") != "COMPLETE_SOURCE":
            reasons.append(f"watermark={manifest.get('watermark')}")
        if reasons:
            rejected.append(f"{manifest_path}: {', '.join(reasons)}")
        else:
            compatible.append((manifest_path.parent, manifest))
    if not compatible:
        detail = "\n".join(rejected[-10:]) or f"No manifests found below {root}"
        raise FileNotFoundError(
            "No compatible completed similarity store was found. Run "
            "scripts/build_similarity_features.py first.\n" + detail
        )
    if len(compatible) > 1:
        choices = "\n".join(str(path) for path, _ in compatible)
        raise RuntimeError(
            "Multiple compatible similarity stores were found; set "
            f"COMMENTGAP_SIMILARITY_STORE explicitly:\n{choices}"
        )
    return compatible[0]


def _fixed_vectors(table: Any) -> np.ndarray:
    column = table.column("embedding").combine_chunks()
    dimension = int(column.type.list_size)
    values = column.values.to_numpy(zero_copy_only=False).reshape(len(column), dimension)
    vectors = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding store contains non-finite vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embedding store contains zero-length vectors")
    return vectors / norms


def _exact_novelty_from_matrix(
    similarities: np.ndarray,
    timestamps: pd.Series,
    eligible: np.ndarray,
) -> np.ndarray:
    """Derive strict-prior novelty from a precomputed cosine matrix."""
    novelty = np.full(len(similarities), np.nan, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce"),
            "position": np.arange(len(similarities)),
        }
    )
    previous: list[int] = []
    valid = frame[frame["timestamp"].notna()]
    for _, batch in valid.groupby("timestamp", sort=True):
        positions = batch["position"].to_numpy(dtype=int)
        query_positions = positions[eligible[positions]]
        if previous and len(query_positions):
            novelty[query_positions] = 1.0 - similarities[
                np.ix_(query_positions, np.asarray(previous, dtype=int))
            ].max(axis=1)
        previous.extend(int(position) for position in query_positions)
    return novelty


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Materialize one exact normalized-vector cosine matrix."""
    return vectors @ vectors.T


def required_temporal_novelties(
    vectors: np.ndarray,
    timestamps: pd.Series,
    is_root: np.ndarray,
    *,
    exact_threshold: int,
    seed: int,
    validation_sample: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, str]]:
    """Calculate only the all-comment and root-only novelty features.

    When the complete discussion is below the exact threshold, one cosine matrix
    is reused for both definitions rather than recalculating root-root products.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    is_root = np.asarray(is_root, dtype=bool)
    all_eligible = np.ones(len(vectors), dtype=bool)
    diagnostics: list[dict[str, Any]] = []
    methods: dict[str, str] = {}

    if len(vectors) <= exact_threshold:
        similarities = _cosine_matrix(vectors)
        novelty_all = _exact_novelty_from_matrix(similarities, timestamps, all_eligible)
        novelty_root = _exact_novelty_from_matrix(similarities, timestamps, is_root)
        methods.update(all="exact_shared_matrix", root="exact_shared_matrix")
        return novelty_all, novelty_root, diagnostics, methods

    novelty_all = _novelty_hnsw(vectors, timestamps, all_eligible)
    methods["all"] = "hnsw"
    diagnostics.append(
        {
            **validate_approximate_novelty(
                vectors,
                timestamps,
                all_eligible,
                novelty_all,
                seed=seed,
                sample_size=validation_sample,
            ),
            "candidate_scope": "all",
        }
    )

    root_positions = np.flatnonzero(is_root)
    if len(root_positions) <= exact_threshold:
        root_vectors = vectors[root_positions]
        root_similarities = _cosine_matrix(root_vectors)
        root_values = _exact_novelty_from_matrix(
            root_similarities,
            pd.Series(timestamps).reset_index(drop=True).iloc[root_positions].reset_index(drop=True),
            np.ones(len(root_positions), dtype=bool),
        )
        novelty_root = np.full(len(vectors), np.nan, dtype=np.float32)
        novelty_root[root_positions] = root_values
        methods["root"] = "exact_root_matrix"
    else:
        novelty_root = _novelty_hnsw(vectors, timestamps, is_root)
        methods["root"] = "hnsw"
        diagnostics.append(
            {
                **validate_approximate_novelty(
                    vectors,
                    timestamps,
                    is_root,
                    novelty_root,
                    seed=seed + 1,
                    sample_size=validation_sample,
                ),
                "candidate_scope": "root",
            }
        )
    return novelty_all, novelty_root, diagnostics, methods


def _story_similarity(
    comment_path: Path,
    passage_path: Path,
    source_path: Path,
    *,
    exact_threshold: int,
    seed: int,
    validation_sample: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    comment_table = pq.read_table(comment_path)
    passage_table = pq.read_table(passage_path)
    source = pq.read_table(source_path, columns=["story_id", "comment_id", "is_root"]).to_pandas()
    comments = comment_table.select(["story_id", "comment_id", "created_at"]).to_pandas()
    if comments.duplicated(["story_id", "comment_id"]).any():
        raise ValueError(f"Duplicate embedding keys in {comment_path}")
    if source.duplicated(["story_id", "comment_id"]).any():
        raise ValueError(f"Duplicate source keys in {source_path}")
    comments["_position"] = np.arange(len(comments))
    comments = comments.merge(source, on=["story_id", "comment_id"], how="left", validate="one_to_one")
    if comments["is_root"].isna().any():
        raise ValueError(f"Embedding comments are missing from source data for {comment_path}")
    order = comments.sort_values(["created_at", "comment_id"])["_position"].to_numpy(dtype=int)
    comments = comments.iloc[order].reset_index(drop=True)
    comment_vectors = _fixed_vectors(comment_table)[order]
    passage_vectors = _fixed_vectors(passage_table)

    article_similarity = article_similarity_top3(comment_vectors, passage_vectors)
    novelty_all, novelty_root, diagnostics, methods = required_temporal_novelties(
        comment_vectors,
        comments["created_at"],
        comments["is_root"].to_numpy(dtype=bool),
        exact_threshold=exact_threshold,
        seed=seed,
        validation_sample=validation_sample,
    )
    output = comments[["story_id", "comment_id"]].copy()
    output["article_similarity_top3"] = article_similarity
    output["novelty_prior_all"] = novelty_all
    output["novelty_prior_roots"] = novelty_root
    if output.duplicated(["story_id", "comment_id"]).any():
        raise AssertionError("Similarity output keys are not unique")
    summary = {
        "story_id": str(output["story_id"].iloc[0]),
        "comments": len(output),
        "roots": int(comments["is_root"].sum()),
        "article_passages": len(passage_vectors),
        "methods": methods,
    }
    return output, diagnostics, summary


def build_similarity_store(config: SimilarityBuildConfig) -> dict[str, Any]:
    """Build resumable per-story scalar similarities from saved embeddings."""
    requested_store = config.embedding_store or config.embedding_root
    embedding_store, embedding_manifest = resolve_embedding_store(
        requested_store,
        model_id=config.model_id,
        revision=config.revision,
        years=config.years,
        allow_subset=config.allow_subset_source,
    )
    embedding_source_years = tuple(
        sorted(map(int, embedding_manifest.get("source", {}).get("years", [])))
    )
    current_embedding_fingerprint, _ = _multi_year_fingerprint(
        config.data_root, embedding_source_years
    )
    if current_embedding_fingerprint != embedding_manifest["dataset_fingerprint"]:
        raise RuntimeError(
            "The current source collection does not match the completed embedding "
            "store. Do not combine vectors and metadata from different snapshots."
        )
    target_dataset_fingerprints = {
        str(year): dataset_fingerprint(config.data_root, year) for year in config.years
    }
    embedding_model = embedding_manifest["model"]
    identity = {
        "embedding_build_signature": embedding_manifest["build_signature"],
        "embedding_dataset_fingerprint": embedding_manifest["dataset_fingerprint"],
        "target_dataset_fingerprints": target_dataset_fingerprints,
        "target_years": list(config.years),
        "model_id": embedding_model["model_id"],
        "resolved_revision": embedding_model["resolved_revision"],
        "exact_novelty_threshold": config.exact_novelty_threshold,
        "hnsw_validation_sample": config.hnsw_validation_sample,
        "hnsw_required_recall": config.hnsw_required_recall,
        "seed": config.seed,
        "max_stories": config.max_stories,
        "semantic_features": [
            "article_similarity_top3",
            "novelty_prior_all",
            "novelty_prior_roots",
        ],
    }
    build_signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_root = (
        config.output_root
        / f"model={_safe_model_key(config.model_id)}"
        / f"build={build_signature[:12]}-{embedding_manifest['build_signature'][:12]}"
    )
    _atomic_json(
        {
            "status": "running",
            "build_signature": build_signature,
            "identity": identity,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
                if key != "overwrite"
            },
        },
        run_root / "build_state.json",
    )

    comment_files: list[tuple[int, int, Path]] = []
    for year in config.years:
        for path in sorted((embedding_store / "comments" / f"year={year}").glob("month=*/*.parquet")):
            month = int(path.parent.name.split("=", 1)[1])
            comment_files.append((year, month, path))
    if config.max_stories is not None:
        comment_files = comment_files[: config.max_stories]
    if not comment_files:
        raise FileNotFoundError(f"No comment embedding checkpoints found in {embedding_store}")

    started = time.monotonic()
    written = 0
    skipped = 0
    total_rows = 0
    method_counts = {
        "all_exact_shared_matrix": 0,
        "all_hnsw": 0,
        "root_exact_shared_matrix": 0,
        "root_exact_root_matrix": 0,
        "root_hnsw": 0,
    }
    large_stories: list[dict[str, Any]] = []
    diagnostics_all: list[dict[str, Any]] = []
    diagnostics_root = run_root / "novelty_validation_parts"

    for index, (year, month, comment_path) in enumerate(comment_files, start=1):
        story_id = comment_path.stem
        passage_path = (
            embedding_store / "article_passages" / f"year={year}" / f"month={month:02d}" / comment_path.name
        )
        source_path = config.data_root / "comments" / f"year={year}" / f"month={month:02d}" / comment_path.name
        destination = run_root / "scalars" / f"year={year}" / f"month={month:02d}" / comment_path.name
        diagnostic_path = diagnostics_root / f"year={year}" / f"month={month:02d}" / f"{story_id}.json"
        if not passage_path.exists():
            raise FileNotFoundError(passage_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if destination.exists() and diagnostic_path.exists() and not config.overwrite:
            cached = _read_json(diagnostic_path)
            summary = cached["summary"]
            diagnostics = cached["diagnostics"]
            skipped += 1
        else:
            output, diagnostics, summary = _story_similarity(
                comment_path,
                passage_path,
                source_path,
                exact_threshold=config.exact_novelty_threshold,
                seed=config.seed + index,
                validation_sample=config.hnsw_validation_sample,
            )
            for diagnostic in diagnostics:
                diagnostic.update({"story_id": story_id, "year": year, "month": month})
                if (
                    diagnostic.get("sample_size")
                    and float(diagnostic["recall_within_1e_3"]) < config.hnsw_required_recall
                ):
                    raise RuntimeError(
                        f"HNSW novelty validation failed for {story_id}: {diagnostic}"
                    )
            _atomic_parquet(output, destination)
            _atomic_json({"summary": summary, "diagnostics": diagnostics}, diagnostic_path)
            written += 1
        total_rows += int(summary["comments"])
        all_method = summary["methods"]["all"]
        root_method = summary["methods"]["root"]
        method_counts[f"all_{all_method}"] += 1
        method_counts[f"root_{root_method}"] += 1
        if int(summary["comments"]) > config.exact_novelty_threshold:
            large_stories.append(summary)
        diagnostics_all.extend(diagnostics)
        if index % config.progress_every_stories == 0 or index == len(comment_files):
            elapsed = time.monotonic() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(comment_files) - index) / rate if rate else 0.0
            print(
                "Similarity progress: "
                f"{index:,}/{len(comment_files):,} stories "
                f"({100 * index / len(comment_files):.1f}%) | "
                f"rows={total_rows:,} new={written:,} skipped={skipped:,} | "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
                f"rate={rate:.2f} stories/s",
                flush=True,
            )

    validation = {
        "method": "Incremental HNSW cosine distance checked against exact strict-prior neighbors",
        "required_recall_within_1e_3": config.hnsw_required_recall,
        "sample_per_story_scope": config.hnsw_validation_sample,
        "stories": diagnostics_all,
    }
    _atomic_json(validation, run_root / "novelty_validation.json")
    source_watermark = embedding_manifest["watermark"]
    watermark = "SUBSET" if config.max_stories is not None else source_watermark
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "watermark": watermark,
        "run_root": str(run_root),
        "build_signature": build_signature,
        "embedding_store": str(embedding_store),
        "embedding_build_signature": embedding_manifest["build_signature"],
        "embedding_dataset_fingerprint": embedding_manifest["dataset_fingerprint"],
        "target_dataset_fingerprints": target_dataset_fingerprints,
        "embedding_model": embedding_model,
        "target_years": list(config.years),
        "features": identity["semantic_features"],
        "algorithm": {
            "article_similarity": "mean of top-three normalized passage cosine similarities",
            "novelty": "one minus maximum cosine similarity to strictly earlier eligible comments",
            "timestamp_ties": "comments sharing a timestamp do not count one another",
            "exact_novelty_threshold": config.exact_novelty_threshold,
            "exact_reuse": "one all-comment cosine matrix supplies both novelty scopes when possible",
            "hnsw": {"space": "cosine", "M": 32, "ef_construction": 200, "ef": 100},
        },
        "storage": {
            "format": "Parquet",
            "compression": "zstd",
            "files": len(comment_files),
            "rows": total_rows,
            "key": ["story_id", "comment_id"],
            "scalar_columns": identity["semantic_features"],
        },
        "method_counts": method_counts,
        "stories_over_exact_threshold": sorted(
            large_stories, key=lambda value: int(value["comments"]), reverse=True
        ),
        "validation": {
            "path": str(run_root / "novelty_validation.json"),
            "hnsw_story_scope_checks": len(diagnostics_all),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _analysis_package_versions(),
        },
        "execution": {
            "elapsed_seconds": time.monotonic() - started,
            "new_story_checkpoints": written,
            "skipped_story_checkpoints": skipped,
        },
    }
    _atomic_json(manifest, run_root / "similarity_manifest.json")
    _atomic_json(
        {"status": "complete", "build_signature": build_signature, "manifest": str(run_root / "similarity_manifest.json")},
        run_root / "build_state.json",
    )
    return manifest
