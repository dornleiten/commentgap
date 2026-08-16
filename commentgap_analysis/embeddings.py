"""Resumable, model-versioned dense embedding stores for the 2025 corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    _analysis_package_versions,
    _atomic_json,
    _atomic_parquet,
    _read_dataset,
    dataset_fingerprint,
    validate_qa_summary,
)
from .nlp import (
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    SentenceTransformerEmbedder,
    TextEmbedder,
    TokenLengthInspector,
    TransformerTokenLengthInspector,
    select_torch_device,
)


@dataclass(frozen=True)
class EmbeddingBuildConfig:
    """Configuration that uniquely identifies an embedding-store build."""

    data_root: Path = Path("data/scrape_2025")
    output_root: Path = Path("model_output/selection_2025/embeddings")
    years: tuple[int, ...] | None = None
    model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    revision: str | None = None
    device: str = "auto"
    batch_size: int = 64
    max_batch_size: int | None = None
    min_batch_size: int = 1
    adaptive_batching: bool = True
    batch_growth_successes: int = 25
    max_length: int = 512
    prompt_name: str | None = None
    storage_dtype: str = "float32"
    allow_incomplete: bool = False
    max_stories: int | None = None
    overwrite: bool = False
    progress_every_stories: int = 25
    tokenizer_batch_size: int = 2048

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.years is not None:
            normalized_years = tuple(sorted({int(year) for year in self.years}))
            if not normalized_years:
                raise ValueError("years cannot be empty")
            object.__setattr__(self, "years", normalized_years)
        if self.model_id == DEFAULT_EMBEDDING_MODEL_ID and self.revision is None:
            object.__setattr__(self, "revision", DEFAULT_EMBEDDING_MODEL_REVISION)
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_batch_size is not None and self.max_batch_size < self.batch_size:
            raise ValueError("max_batch_size cannot be smaller than batch_size")
        if self.min_batch_size < 1 or self.min_batch_size > self.batch_size:
            raise ValueError("min_batch_size must be between 1 and batch_size")
        if self.batch_growth_successes < 1:
            raise ValueError("batch_growth_successes must be positive")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.max_stories is not None and self.max_stories < 1:
            raise ValueError("max_stories must be positive")
        if self.storage_dtype not in {"float32", "float16"}:
            raise ValueError("storage_dtype must be 'float32' or 'float16'")
        if self.progress_every_stories < 1:
            raise ValueError("progress_every_stories must be positive")
        if self.tokenizer_batch_size < 1:
            raise ValueError("tokenizer_batch_size must be positive")


def _safe_model_key(model_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "__", model_id).strip("._-")
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:80]}--{digest}"


def discover_embedding_years(data_root: Path) -> tuple[int, ...]:
    """Discover years that have both article and comment Parquet partitions."""
    articles = {
        int(path.name.split("=", 1)[1])
        for path in (data_root / "articles").glob("year=*")
        if path.is_dir() and path.name.split("=", 1)[1].isdigit()
    }
    comments = {
        int(path.name.split("=", 1)[1])
        for path in (data_root / "comments").glob("year=*")
        if path.is_dir() and path.name.split("=", 1)[1].isdigit()
    }
    years = tuple(sorted(articles & comments))
    if not years:
        raise FileNotFoundError(
            f"No matching article/comment year partitions found under {data_root}"
        )
    return years


def _qa_paths(data_root: Path, year: int) -> list[Path]:
    year_root = data_root / "qa_summary" / f"year={year}"
    annual = year_root / "summary.json"
    if annual.exists():
        return [annual]
    monthly = sorted(year_root.glob("month=*/summary.json"))
    if not monthly:
        raise FileNotFoundError(
            f"No annual or month-level QA summary found for year {year} under {year_root}"
        )
    return monthly


def _multi_year_fingerprint(
    data_root: Path, years: tuple[int, ...]
) -> tuple[str, dict[str, str]]:
    year_fingerprints: dict[str, str] = {}
    for year in years:
        digest = hashlib.sha256(dataset_fingerprint(data_root, year).encode("ascii"))
        for path in _qa_paths(data_root, year):
            digest.update(str(path.relative_to(data_root)).encode("utf-8"))
            digest.update(path.read_bytes())
        year_fingerprints[str(year)] = digest.hexdigest()
    combined = hashlib.sha256(
        json.dumps(year_fingerprints, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return combined, year_fingerprints


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _is_accelerator_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "mps backend out of memory",
            "mps out of memory",
        )
    )


def _clear_accelerator_cache(device: str) -> None:
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
    except Exception:
        # Cache clearing is best effort; retrying may still succeed after local
        # tensors from the failed encode have been released.
        pass


class AdaptiveBatchEncoder:
    """Retry accelerator OOMs with backoff and cautiously recover throughput."""

    def __init__(
        self,
        embedder: TextEmbedder,
        *,
        device: str,
        initial_batch_size: int,
        maximum_batch_size: int,
        minimum_batch_size: int,
        growth_successes: int,
        adaptive: bool,
    ) -> None:
        self.embedder = embedder
        self.device = device
        self.current_batch_size = initial_batch_size
        self.maximum_batch_size = maximum_batch_size
        self.minimum_batch_size = minimum_batch_size
        self.growth_successes = growth_successes
        self.adaptive = adaptive and device in {"cuda", "mps"}
        self._saturated_successes = 0
        self.history: list[dict[str, Any]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        while True:
            attempted = self.current_batch_size
            try:
                output = self.embedder.encode(texts, attempted)
            except (RuntimeError, MemoryError) as error:
                if (
                    not self.adaptive
                    or not _is_accelerator_oom(error)
                    or attempted <= self.minimum_batch_size
                ):
                    raise
                reduced = max(self.minimum_batch_size, attempted // 2)
                if reduced == attempted:
                    raise
                self.current_batch_size = reduced
                self._saturated_successes = 0
                event = {
                    "event": "decrease_after_oom",
                    "from": attempted,
                    "to": reduced,
                    "records_in_call": len(texts),
                }
                self.history.append(event)
                print(
                    f"Adaptive batching: accelerator OOM at batch={attempted}; "
                    f"clearing cache and retrying the same records at batch={reduced}.",
                    flush=True,
                )
                _clear_accelerator_cache(self.device)
                continue

            if len(texts) >= attempted:
                self._saturated_successes += 1
            if (
                self.adaptive
                and self.current_batch_size < self.maximum_batch_size
                and self._saturated_successes >= self.growth_successes
            ):
                increased = min(
                    self.maximum_batch_size,
                    max(self.current_batch_size + 1, self.current_batch_size * 2),
                )
                event = {
                    "event": "increase_after_successes",
                    "from": self.current_batch_size,
                    "to": increased,
                    "successful_full_batches": self._saturated_successes,
                }
                self.history.append(event)
                self.current_batch_size = increased
                self._saturated_successes = 0
                print(
                    f"Adaptive batching: increasing batch to {increased} after "
                    f"{self.growth_successes} successful full-batch calls.",
                    flush=True,
                )
            return np.asarray(output)

    def summary(self) -> dict[str, Any]:
        return {
            "adaptive": self.adaptive,
            "initial_batch_size": self.history[0]["from"] if self.history else self.current_batch_size,
            "final_batch_size": self.current_batch_size,
            "maximum_batch_size": self.maximum_batch_size,
            "minimum_batch_size": self.minimum_batch_size,
            "growth_successes": self.growth_successes,
            "events": self.history,
        }


def _passage_records(article: pd.Series) -> list[dict[str, Any]]:
    """Return stable title/subtitle/paragraph keys matching the feature workflow."""
    records: list[dict[str, Any]] = []
    story_id = str(article["story_id"])
    for kind in ("title", "subtitle"):
        value = article.get(kind)
        text = re.sub(r"\s+", " ", str(value)).strip() if pd.notna(value) else ""
        if text:
            records.append(
                {
                    "story_id": story_id,
                    "passage_id": f"{story_id}:{kind}",
                    "passage_kind": kind,
                    "passage_index": 0,
                    "source_text_sha256": _text_hash(text),
                    "text": text,
                }
            )
    body = article.get("body")
    body_text = str(body) if pd.notna(body) else ""
    paragraph_index = 0
    for value in re.split(r"\n\s*\n+", body_text):
        text = re.sub(r"\s+", " ", value).strip()
        if not text:
            continue
        records.append(
            {
                "story_id": story_id,
                "passage_id": f"{story_id}:body:{paragraph_index:04d}",
                "passage_kind": "body",
                "passage_index": paragraph_index,
                "source_text_sha256": _text_hash(text),
                "text": text,
            }
        )
        paragraph_index += 1
    return records


def _vector_array(vectors: np.ndarray, storage_dtype: str):
    import pyarrow as pa

    if vectors.ndim != 2 or vectors.shape[1] < 1:
        raise ValueError(f"Expected a non-empty two-dimensional vector matrix, got {vectors.shape}")
    numpy_dtype = np.float16 if storage_dtype == "float16" else np.float32
    arrow_dtype = pa.float16() if storage_dtype == "float16" else pa.float32()
    flat = pa.array(np.asarray(vectors, dtype=numpy_dtype).reshape(-1), type=arrow_dtype)
    return pa.FixedSizeListArray.from_arrays(flat, vectors.shape[1])


def _write_comment_vectors(
    frame: pd.DataFrame,
    vectors: np.ndarray,
    path: Path,
    storage_dtype: str,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "story_id": pa.array(frame["story_id"].astype(str)),
            "comment_id": pa.array(frame["comment_id"].astype(str)),
            "created_at": pa.array(frame["created_at"]),
            "source_text_sha256": pa.array(frame["effective_text"].map(_text_hash)),
            "embedding": _vector_array(vectors, storage_dtype),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _write_passage_vectors(
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    path: Path,
    storage_dtype: str,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "story_id": pa.array([row["story_id"] for row in records]),
            "passage_id": pa.array([row["passage_id"] for row in records]),
            "passage_kind": pa.array([row["passage_kind"] for row in records]),
            "passage_index": pa.array([row["passage_index"] for row in records], type=pa.int32()),
            "source_text_sha256": pa.array([row["source_text_sha256"] for row in records]),
            "embedding": _vector_array(vectors, storage_dtype),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _parquet_inventory(root: Path) -> tuple[int, int, int | None]:
    import pyarrow.parquet as pq

    files = sorted(root.glob("year=*/month=*/*.parquet")) if root.exists() else []
    rows = 0
    dimensions: set[int] = set()
    for path in files:
        parquet = pq.ParquetFile(path)
        rows += parquet.metadata.num_rows
        vector_type = parquet.schema_arrow.field("embedding").type
        dimensions.add(int(vector_type.list_size))
    if len(dimensions) > 1:
        raise RuntimeError(f"Mixed embedding dimensions found under {root}: {dimensions}")
    return len(files), rows, next(iter(dimensions), None)


def _validated_embedding_sources(
    config: EmbeddingBuildConfig,
) -> tuple[tuple[int, ...], dict[str, list[dict[str, Any]]]]:
    years = config.years or discover_embedding_years(config.data_root)
    qa_by_year: dict[str, list[dict[str, Any]]] = {}
    for year in years:
        summaries = [json.loads(path.read_text()) for path in _qa_paths(config.data_root, year)]
        for summary in summaries:
            validate_qa_summary(summary, allow_incomplete=config.allow_incomplete)
        qa_by_year[str(year)] = summaries
    return years, qa_by_year


def _histogram_quantile(histogram: Counter[int], quantile: float) -> int | None:
    total = sum(histogram.values())
    if not total:
        return None
    target = max(1, math.ceil(total * quantile))
    cumulative = 0
    for length, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= target:
            return int(length)
    return int(max(histogram))


def _length_summary(histogram: Counter[int], max_length: int) -> dict[str, Any]:
    total = sum(histogram.values())
    over_limit = sum(count for length, count in histogram.items() if length > max_length)
    return {
        "count": total,
        "over_max_length": over_limit,
        "percent_over_max_length": 100.0 * over_limit / total if total else 0.0,
        "p50_tokens": _histogram_quantile(histogram, 0.50),
        "p95_tokens": _histogram_quantile(histogram, 0.95),
        "p99_tokens": _histogram_quantile(histogram, 0.99),
        "maximum_tokens": int(max(histogram)) if histogram else None,
    }


def build_token_length_diagnostics(
    config: EmbeddingBuildConfig,
    *,
    inspector: TokenLengthInspector | None = None,
) -> dict[str, Any]:
    """Measure exact tokenizer lengths without retaining or emitting raw text."""
    years, _ = _validated_embedding_sources(config)
    if inspector is None:
        inspector = TransformerTokenLengthInspector(
            model_id=config.model_id,
            revision=config.revision,
        )
    if inspector.model_id != config.model_id:
        raise ValueError(
            f"Configured model {config.model_id!r} does not match tokenizer {inspector.model_id!r}"
        )
    resolved_revision = (
        getattr(inspector, "resolved_revision", None)
        or config.revision
        or "unresolved"
    )
    fingerprint, year_fingerprints = _multi_year_fingerprint(config.data_root, years)
    diagnostic_identity = {
        "years": list(years),
        "model_id": config.model_id,
        "requested_revision": config.revision,
        "resolved_revision": resolved_revision,
        "max_length": config.max_length,
        "candidate_filter": "Published and non-empty effective_text",
        "max_stories": config.max_stories,
    }
    signature = hashlib.sha256(
        json.dumps(diagnostic_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    diagnostic_root = (
        config.output_root
        / "token_diagnostics"
        / f"model={_safe_model_key(config.model_id)}"
        / f"build={signature[:12]}-{fingerprint[:12]}"
    )
    summary_path = diagnostic_root / "token_length_summary.json"
    truncated_path = diagnostic_root / "truncated_records.parquet"
    if summary_path.exists() and truncated_path.exists() and not config.overwrite:
        cached = json.loads(summary_path.read_text())
        print(
            f"Token diagnostics: reusing {summary_path} "
            f"({cached['combined']['over_max_length']:,} records over limit)",
            flush=True,
        )
        return cached

    articles = pd.concat(
        [
            _read_dataset(
                config.data_root / "articles",
                year,
                ["story_id", "title", "subtitle", "body", "year", "month"],
            )
            for year in years
        ],
        ignore_index=True,
        sort=False,
    )
    articles["story_id"] = articles["story_id"].astype(str)
    story_ids = sorted(articles["story_id"].unique())
    if config.max_stories is not None:
        story_ids = story_ids[: config.max_stories]
    articles = articles[articles["story_id"].isin(set(story_ids))].copy()

    import pyarrow.dataset as ds

    histograms = {"comments": Counter(), "article_passages": Counter()}
    truncated: list[dict[str, Any]] = []
    started_at = time.monotonic()
    for year in years:
        year_articles = articles[articles["year"].astype(int).eq(year)]
        comment_dataset = ds.dataset(
            config.data_root / "comments" / f"year={year}",
            format="parquet",
            partitioning=None,
        )
        for month in sorted(year_articles["month"].dropna().astype(int).unique()):
            month_articles = year_articles[
                year_articles["month"].astype(int).eq(month)
            ]
            month_story_ids = set(month_articles["story_id"])
            comments = comment_dataset.to_table(
                columns=[
                    "comment_id",
                    "story_id",
                    "effective_text",
                    "lifecycle_status",
                    "month",
                ],
                filter=(ds.field("month") == int(month))
                & (ds.field("lifecycle_status") == "Published"),
            ).to_pandas()
            comments["story_id"] = comments["story_id"].astype(str)
            comments = comments[
                comments["story_id"].isin(month_story_ids)
                & comments["effective_text"].fillna("").str.strip().ne("")
            ].copy()
            comment_texts = comments["effective_text"].astype(str).tolist()
            comment_lengths = inspector.token_lengths(
                comment_texts, config.tokenizer_batch_size
            )
            if len(comment_lengths) != len(comments):
                raise ValueError(f"Tokenizer length mismatch for {year}-{month:02d} comments")
            histograms["comments"].update(map(int, comment_lengths))
            for row, text, length in zip(
                comments.itertuples(index=False), comment_texts, comment_lengths
            ):
                if int(length) > config.max_length:
                    truncated.append(
                        {
                            "record_type": "comment",
                            "year": year,
                            "month": month,
                            "story_id": str(row.story_id),
                            "record_id": str(row.comment_id),
                            "token_length": int(length),
                            "source_text_sha256": _text_hash(text),
                        }
                    )

            passage_metadata: list[dict[str, Any]] = []
            for _, article in month_articles.iterrows():
                passage_metadata.extend(_passage_records(article))
            passage_texts = [record["text"] for record in passage_metadata]
            passage_lengths = inspector.token_lengths(
                passage_texts, config.tokenizer_batch_size
            )
            if len(passage_lengths) != len(passage_metadata):
                raise ValueError(f"Tokenizer length mismatch for {year}-{month:02d} passages")
            histograms["article_passages"].update(map(int, passage_lengths))
            for record, length in zip(passage_metadata, passage_lengths):
                if int(length) > config.max_length:
                    truncated.append(
                        {
                            "record_type": "article_passage",
                            "year": year,
                            "month": month,
                            "story_id": record["story_id"],
                            "record_id": record["passage_id"],
                            "token_length": int(length),
                            "source_text_sha256": record["source_text_sha256"],
                        }
                    )
            elapsed = time.monotonic() - started_at
            inspected = sum(sum(histogram.values()) for histogram in histograms.values())
            print(
                f"Token diagnostics: completed {year}-{month:02d} | "
                f"records={inspected:,} over_limit={len(truncated):,} "
                f"elapsed={_format_duration(elapsed)}",
                flush=True,
            )

    combined = histograms["comments"] + histograms["article_passages"]
    summary = {
        "status": "complete",
        "diagnostic_root": str(diagnostic_root),
        "identity": diagnostic_identity,
        "dataset_fingerprint": fingerprint,
        "year_fingerprints": year_fingerprints,
        "comments": _length_summary(histograms["comments"], config.max_length),
        "article_passages": _length_summary(
            histograms["article_passages"], config.max_length
        ),
        "combined": _length_summary(combined, config.max_length),
        "truncated_records_path": str(truncated_path),
        "elapsed_seconds": time.monotonic() - started_at,
    }
    columns = [
        "record_type",
        "year",
        "month",
        "story_id",
        "record_id",
        "token_length",
        "source_text_sha256",
    ]
    _atomic_parquet(pd.DataFrame(truncated, columns=columns), truncated_path)
    _atomic_json(summary, summary_path)
    return summary


def build_embedding_store(
    config: EmbeddingBuildConfig,
    *,
    embedder: TextEmbedder | None = None,
) -> dict[str, Any]:
    """Build a keyed, resumable comment and article-passage vector store."""
    years = config.years or discover_embedding_years(config.data_root)
    qa_by_year: dict[str, list[dict[str, Any]]] = {}
    for year in years:
        summaries = [json.loads(path.read_text()) for path in _qa_paths(config.data_root, year)]
        for summary in summaries:
            validate_qa_summary(summary, allow_incomplete=config.allow_incomplete)
        qa_by_year[str(year)] = summaries

    if embedder is None:
        embedder = SentenceTransformerEmbedder(
            model_id=config.model_id,
            revision=config.revision,
            device=config.device,
            max_length=config.max_length,
            prompt_name=config.prompt_name,
        )
    if embedder.model_id != config.model_id:
        raise ValueError(
            f"Configured model {config.model_id!r} does not match adapter {embedder.model_id!r}"
        )
    resolved_device = getattr(embedder, "device", None) or select_torch_device(config.device)
    print(
        "Embedding preflight: "
        f"requested_device={config.device} resolved_device={resolved_device} "
        f"model={config.model_id} years={','.join(map(str, years))} "
        f"storage={config.storage_dtype}",
        flush=True,
    )
    if not hasattr(embedder, "token_lengths"):
        raise TypeError(
            "The embedding adapter must expose token_lengths() so truncation is audited"
        )
    token_diagnostics = build_token_length_diagnostics(config, inspector=embedder)
    print(
        "Token-length result: "
        f"comments_over_limit={token_diagnostics['comments']['over_max_length']:,} "
        f"passages_over_limit={token_diagnostics['article_passages']['over_max_length']:,} "
        f"max_length={config.max_length}",
        flush=True,
    )
    batch_encoder = AdaptiveBatchEncoder(
        embedder,
        device=resolved_device,
        initial_batch_size=config.batch_size,
        maximum_batch_size=config.max_batch_size or config.batch_size,
        minimum_batch_size=config.min_batch_size,
        growth_successes=config.batch_growth_successes,
        adaptive=config.adaptive_batching,
    )
    resolved_revision = getattr(embedder, "resolved_revision", None) or config.revision or "unresolved"
    fingerprint, year_fingerprints = _multi_year_fingerprint(config.data_root, years)
    identity = {
        "years": list(years),
        "model_id": config.model_id,
        "requested_revision": config.revision,
        "resolved_revision": resolved_revision,
        "max_length": config.max_length,
        "prompt_name": config.prompt_name,
        "candidate_filter": "Published and non-empty effective_text",
        "normalization": "L2",
        "storage_dtype": config.storage_dtype,
        "max_stories": config.max_stories,
    }
    build_signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_root = (
        config.output_root
        / f"model={_safe_model_key(config.model_id)}"
        / f"build={build_signature[:12]}-{fingerprint[:12]}"
    )
    state_path = run_root / "build_state.json"
    state = {
        "status": "running",
        "dataset_fingerprint": fingerprint,
        "build_signature": build_signature,
        "identity": identity,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
            if key != "overwrite"
        },
    }
    _atomic_json(state, state_path)

    articles = pd.concat(
        [
            _read_dataset(
                config.data_root / "articles",
                year,
                ["story_id", "title", "subtitle", "body", "year", "month"],
            )
            for year in years
        ],
        ignore_index=True,
        sort=False,
    )
    articles["story_id"] = articles["story_id"].astype(str)
    if articles.duplicated("story_id").any():
        raise ValueError("story_id is not unique in article data")
    story_ids = sorted(set(articles["story_id"]))
    if config.max_stories is not None:
        story_ids = story_ids[: config.max_stories]
    selected_story_ids = set(story_ids)
    articles = articles[articles["story_id"].isin(selected_story_ids)].copy()
    print(
        f"Embedding workload: {len(story_ids):,} articles; "
        "comment totals are reported as each month is scanned.",
        flush=True,
    )

    import pyarrow.dataset as ds

    comment_columns = [
        "comment_id",
        "story_id",
        "created_at",
        "effective_text",
        "lifecycle_status",
        "year",
        "month",
    ]

    processed = 0
    eligible_comment_count = 0
    eligible_comments_by_year: dict[str, int] = {}
    embedded_comment_rows = 0
    embedded_passage_rows = 0
    skipped_comment_checkpoints = 0
    skipped_passage_checkpoints = 0
    started_at = time.monotonic()
    for year in years:
        year_articles = articles[articles["year"].astype(int).eq(year)]
        comment_dataset = ds.dataset(
            config.data_root / "comments" / f"year={year}",
            format="parquet",
            partitioning=None,
        )
        missing = set(comment_columns) - set(comment_dataset.schema.names)
        if missing:
            raise ValueError(f"Missing comment columns for {year}: {sorted(missing)}")
        year_comment_count = 0
        for month in sorted(year_articles["month"].dropna().astype(int).unique()):
            month_articles = year_articles[
                year_articles["month"].astype(int).eq(month)
            ].set_index("story_id", drop=False)
            month_story_ids = set(month_articles.index)
            month_table = comment_dataset.to_table(
                columns=comment_columns,
                filter=(ds.field("month") == int(month))
                & (ds.field("lifecycle_status") == "Published"),
            )
            month_comments = month_table.to_pandas()
            month_comments["story_id"] = month_comments["story_id"].astype(str)
            month_comments["comment_id"] = month_comments["comment_id"].astype(str)
            month_comments = month_comments[
                month_comments["story_id"].isin(month_story_ids)
                & month_comments["effective_text"].fillna("").str.strip().ne("")
            ].copy()
            month_comments["effective_text"] = month_comments["effective_text"].astype(str)
            month_comments["created_at"] = pd.to_datetime(
                month_comments["created_at"], utc=True, errors="coerce"
            )
            if month_comments.duplicated("comment_id").any():
                raise ValueError(f"comment_id is not unique within {year}-{month:02d}")
            eligible_comment_count += len(month_comments)
            year_comment_count += len(month_comments)
            comment_groups = {
                story_id: group
                for story_id, group in month_comments.groupby("story_id", sort=False)
            }

            for story_id, article in month_articles.iterrows():
                group = comment_groups.get(story_id)
                comment_path = (
                    run_root
                    / "comments"
                    / f"year={year}"
                    / f"month={month:02d}"
                    / f"{story_id}.parquet"
                )
                passage_path = (
                    run_root
                    / "article_passages"
                    / f"year={year}"
                    / f"month={month:02d}"
                    / f"{story_id}.parquet"
                )
                if group is not None and (config.overwrite or not comment_path.exists()):
                    ordered = group.sort_values(["created_at", "comment_id"]).copy()
                    vectors = np.asarray(
                        batch_encoder.encode(ordered["effective_text"].tolist()),
                        dtype=np.float32,
                    )
                    if vectors.shape[0] != len(ordered):
                        raise ValueError(f"Comment embedding row mismatch for story {story_id}")
                    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
                        raise ValueError(
                            f"Comment embeddings are not normalized for story {story_id}"
                        )
                    _write_comment_vectors(
                        ordered, vectors, comment_path, config.storage_dtype
                    )
                    embedded_comment_rows += len(ordered)
                elif group is not None:
                    skipped_comment_checkpoints += 1
                if config.overwrite or not passage_path.exists():
                    passages = _passage_records(article)
                    if passages:
                        vectors = np.asarray(
                            batch_encoder.encode([row["text"] for row in passages]),
                            dtype=np.float32,
                        )
                        if vectors.shape[0] != len(passages):
                            raise ValueError(
                                f"Passage embedding row mismatch for story {story_id}"
                            )
                        if not np.allclose(
                            np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3
                        ):
                            raise ValueError(
                                f"Passage embeddings are not normalized for story {story_id}"
                            )
                        _write_passage_vectors(
                            passages, vectors, passage_path, config.storage_dtype
                        )
                        embedded_passage_rows += len(passages)
                else:
                    skipped_passage_checkpoints += 1
                processed += 1
                if (
                    processed % config.progress_every_stories == 0
                    or processed == len(story_ids)
                ):
                    elapsed = time.monotonic() - started_at
                    rate = processed / elapsed if elapsed else 0.0
                    eta = (len(story_ids) - processed) / rate if rate else 0.0
                    print(
                        "Embedding progress: "
                        f"{processed:,}/{len(story_ids):,} articles "
                        f"({100 * processed / len(story_ids):.1f}%) | "
                        f"new comments={embedded_comment_rows:,} "
                        f"new passages={embedded_passage_rows:,} | "
                        f"elapsed={_format_duration(elapsed)} "
                        f"eta={_format_duration(eta)} "
                        f"rate={rate:.2f} articles/s "
                        f"batch={batch_encoder.current_batch_size}",
                        flush=True,
                    )
            del month_table, month_comments, comment_groups
        eligible_comments_by_year[str(year)] = year_comment_count

    comment_files, comment_rows, comment_dimensions = _parquet_inventory(run_root / "comments")
    passage_files, passage_rows, passage_dimensions = _parquet_inventory(run_root / "article_passages")
    dimensions = {value for value in (comment_dimensions, passage_dimensions) if value is not None}
    if len(dimensions) != 1:
        raise RuntimeError(f"Expected one embedding dimension across the store, found {dimensions}")
    dimension = next(iter(dimensions))
    watermark_parts = []
    if config.allow_incomplete and any(
        summary.get("nonterminal_stories", 0)
        for summaries in qa_by_year.values()
        for summary in summaries
    ):
        watermark_parts.append("INCOMPLETE_SOURCE")
    if config.max_stories is not None:
        watermark_parts.append("SUBSET")
    watermark = "+".join(watermark_parts) if watermark_parts else "COMPLETE_SOURCE"
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "watermark": watermark,
        "run_root": str(run_root),
        "dataset_fingerprint": fingerprint,
        "year_fingerprints": year_fingerprints,
        "build_signature": build_signature,
        "model": identity,
        "token_length_diagnostics": token_diagnostics,
        "embedding_dimension": dimension,
        "storage": {
            "format": "Parquet",
            "compression": "zstd",
            "vector_column": "embedding",
            "vector_type": f"fixed_size_list<{config.storage_dtype}>[{dimension}]",
            "comments": {"files": comment_files, "rows": comment_rows},
            "article_passages": {"files": passage_files, "rows": passage_rows},
        },
        "source": {
            "years": list(years),
            "articles": len(articles),
            "eligible_comments": eligible_comment_count,
            "eligible_comments_by_year": eligible_comments_by_year,
            "selected_stories": len(story_ids),
            "qa": qa_by_year,
        },
        "environment": {
            "requested_device": config.device,
            "resolved_device": resolved_device,
            "packages": _analysis_package_versions(),
        },
        "execution": {
            "elapsed_seconds": time.monotonic() - started_at,
            "embedded_comment_rows_this_run": embedded_comment_rows,
            "embedded_passage_rows_this_run": embedded_passage_rows,
            "skipped_comment_checkpoints": skipped_comment_checkpoints,
            "skipped_passage_checkpoints": skipped_passage_checkpoints,
            "batching": batch_encoder.summary(),
        },
    }
    _atomic_json(manifest, run_root / "embedding_manifest.json")
    state["status"] = "complete"
    state["embedding_dimension"] = dimension
    _atomic_json(state, state_path)
    return manifest
