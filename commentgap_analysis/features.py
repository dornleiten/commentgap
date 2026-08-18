"""Build model-ready features from the normalized 2025 Parquet collection."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .nlp import (
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    GermanSentimentEncoder,
    PilotLexiconSentiment,
    SentimentEncoder,
    select_torch_device,
)


URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
WORD_RE = re.compile(r"\b[\wÄÖÜäöüß]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
VOWEL_GROUP_RE = re.compile(r"[aeiouyäöü]+", re.IGNORECASE)
VIENNA = ZoneInfo("Europe/Vienna")

BASE_SEED = 20260813
DEFAULT_TIE_DRAWS = 10


ROOT_MODEL_FEATURES = [
    "log_words",
    "sentiment_positive",
    "sentiment_negative",
    "lexdiv_length_adjusted",
    "reading_level_length_adjusted",
    "url_present",
    "article_similarity_top3",
    "novelty_prior_roots_model",
    "log_hours_since_article",
    "log_prior_roots",
    "log_prior_comments",
    "log_comments_prev_hour",
    "vienna_overnight",
    "vienna_weekday_shoulder_evening",
    "vienna_weekend_day_evening",
    "log_author_prior_30d_comments",
    "log_author_prior_30d_snapshot_upvotes",
    "log_author_prior_30d_snapshot_downvotes",
    "log_author_prior_comments_story",
]

ALL_MODEL_FEATURES = [
    feature.replace("novelty_prior_roots_model", "novelty_prior_all_model")
    for feature in ROOT_MODEL_FEATURES
] + [
    "is_reply",
    "log_depth",
    "log_branch_prior_comments",
    "log_branch_comments_prev_hour",
]

BINARY_FEATURES = {
    "url_present",
    "vienna_overnight",
    "vienna_weekday_shoulder_evening",
    "vienna_weekend_day_evening",
    "is_reply",
}


@dataclass(frozen=True)
class FeatureBuildConfig:
    data_root: Path = Path("data/scrape_2025")
    output_root: Path = Path("model_output/selection_2025/features")
    similarity_root: Path = Path("model_output/selection_2025/similarities")
    similarity_store: Path | None = None
    year: int = 2025
    lookback_root: Path | None = None
    allow_incomplete: bool = False
    inference_mode: bool = True
    nlp_mode: str = "real"
    device: str = "auto"
    sentiment_revision: str | None = None
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    embedding_revision: str | None = None
    sentiment_batch_size: int = 32
    tie_draws: int = DEFAULT_TIE_DRAWS
    seed: int = BASE_SEED
    require_page_publication_time: bool = True
    exclude_january_without_lookback: bool = True
    overwrite: bool = False
    max_stories: int | None = None
    progress_every_rows: int = 250_000
    progress_every_stories: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "similarity_root", Path(self.similarity_root))
        if self.similarity_store is not None:
            object.__setattr__(self, "similarity_store", Path(self.similarity_store))
        if self.lookback_root is not None:
            object.__setattr__(self, "lookback_root", Path(self.lookback_root))
        if self.nlp_mode not in {"real", "pilot"}:
            raise ValueError("nlp_mode must be 'real' or 'pilot'")
        if (
            self.embedding_model_id == DEFAULT_EMBEDDING_MODEL_ID
            and self.embedding_revision is None
        ):
            object.__setattr__(
                self, "embedding_revision", DEFAULT_EMBEDDING_MODEL_REVISION
            )
        if not self.embedding_model_id.strip():
            raise ValueError("embedding_model_id cannot be empty")
        if self.inference_mode and (self.allow_incomplete or self.nlp_mode != "real"):
            raise ValueError("Inference mode requires complete data and production NLP")
        if self.inference_mode and self.max_stories is not None:
            raise ValueError("Inference mode cannot limit the number of stories")
        if self.tie_draws < 1:
            raise ValueError("tie_draws must be positive")
        if self.progress_every_rows < 1:
            raise ValueError("progress_every_rows must be positive")
        if self.progress_every_stories < 1:
            raise ValueError("progress_every_stories must be positive")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _linux_rss_gib() -> float | None:
    """Return current Linux resident memory without adding a dependency."""
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024**2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _progress_line(
    label: str,
    completed: int,
    total: int,
    started: float,
    *,
    unit: str = "rows",
    detail: str = "",
) -> None:
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed and completed else 0.0
    eta = (total - completed) / rate if rate else 0.0
    rss = _linux_rss_gib()
    diagnostics = [detail] if detail else []
    if rss is not None:
        diagnostics.append(f"rss={rss:.1f}GiB")
    suffix = f" | {' '.join(diagnostics)}" if diagnostics else ""
    print(
        f"{label}: {completed:,}/{total:,} {unit} "
        f"({100 * completed / total if total else 100:.1f}%) | "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
        f"rate={rate:,.1f} {unit}/s{suffix}",
        flush=True,
    )


def validate_qa_summary(summary: dict[str, Any], allow_incomplete: bool = False) -> None:
    integrity_fields = (
        "duplicate_comment_ids",
        "negative_reaction_rows",
        "invalid_author_hash_rows",
        "sticky_count_mismatch_stories",
        "missing_parent_rows",
        "missing_root_rows",
        "invalid_tree_relationship_rows",
        "comments_without_forum_rows",
        "manifest_count_mismatches",
        "invalid_discrepancy_status_rows",
        "unexpected_forum_count_mismatches",
        "invalid_forum_discrepancy_status_rows",
        "invalid_cursor_progression_pages",
        "incomplete_terminal_page_walks",
        "invalid_cursor_hash_rows",
        "forum_page_aggregate_mismatches",
        "reply_depth_limit_rows",
    )
    failures = {key: summary.get(key) for key in integrity_fields if summary.get(key, 0) != 0}
    if summary.get("raw_author_identifier_columns"):
        failures["raw_author_identifier_columns"] = summary["raw_author_identifier_columns"]
    if failures:
        raise ValueError(f"Collection QA integrity checks failed: {failures}")
    status_counts = summary.get("status_counts", {})
    if status_counts.get("failed", 0):
        raise ValueError("Collection contains failed stories")
    if not allow_incomplete and summary.get("nonterminal_stories", 0):
        raise ValueError(
            f"Collection is incomplete: {summary['nonterminal_stories']} nonterminal stories"
        )
    if not bool(summary.get("passed")):
        raise ValueError("Collection QA summary is not marked as passed")


def _hash_file_inventory(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def dataset_fingerprint(data_root: Path, year: int) -> str:
    paths: list[Path] = []
    for table in ("articles", "forums", "comments", "forum_pages"):
        paths.extend((data_root / table / f"year={year}").glob("month=*/*.parquet"))
    qa = data_root / "qa_summary" / f"year={year}" / "summary.json"
    if qa.exists():
        paths.append(qa)
    return _hash_file_inventory(paths, data_root)


def _analysis_package_versions() -> dict[str, str]:
    packages = (
        "duckdb",
        "pyarrow",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "torch",
        "transformers",
        "sentence-transformers",
        "hnswlib",
        "pyphen",
    )
    output: dict[str, str] = {}
    for package in packages:
        try:
            output[package] = package_version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output


def _stable_tie_key(story_id: str, comment_id: str, seed: int, draw: int) -> int:
    payload = f"{seed}|{draw}|{story_id}|{comment_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def assign_audience_labels(
    candidates: pd.DataFrame,
    *,
    draws: int = DEFAULT_TIE_DRAWS,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"story_id", "comment_id", "is_sticky", "relative_votes"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing audience-label columns: {sorted(missing)}")
    output = candidates.copy()
    diagnostics: list[dict[str, Any]] = []
    for draw in range(1, draws + 1):
        output[f"audience_selected_draw_{draw:02d}"] = False

    for story_id, indices in output.groupby("story_id", sort=False).groups.items():
        group = output.loc[indices]
        n_candidates = len(group)
        n_picks = int(group["is_sticky"].astype(bool).sum())
        if not 0 < n_picks < n_candidates:
            raise ValueError(f"Uninformative choice set {story_id}: k={n_picks}, N={n_candidates}")
        cutoff = float(group["relative_votes"].nlargest(n_picks).iloc[-1])
        above = int((group["relative_votes"] > cutoff).sum())
        tied = int((group["relative_votes"] == cutoff).sum())
        slots = n_picks - above
        diagnostics.append(
            {
                "story_id": story_id,
                "n_candidates": n_candidates,
                "n_picks": n_picks,
                "cutoff_relative_votes": cutoff,
                "n_above_cutoff": above,
                "n_tied_at_cutoff": tied,
                "slots_within_cutoff_tie": slots,
                "cutoff_inclusion_probability": slots / tied,
                "ambiguous_cutoff": tied > slots,
            }
        )
        for draw in range(1, draws + 1):
            ordered = sorted(
                indices,
                key=lambda idx: (
                    -float(output.at[idx, "relative_votes"]),
                    _stable_tie_key(
                        str(story_id), str(output.at[idx, "comment_id"]), seed, draw
                    ),
                ),
            )
            output.loc[ordered[:n_picks], f"audience_selected_draw_{draw:02d}"] = True
    return output, pd.DataFrame(diagnostics)


def _as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def compute_discussion_history(
    comments: pd.DataFrame,
    *,
    progress_every_rows: int | None = None,
    consume_input: bool = False,
) -> pd.DataFrame:
    """Compute strictly prior discussion and branch activity without tie leakage."""
    required = {
        "comment_id",
        "story_id",
        "created_at",
        "is_root",
        "root_comment_id",
        "author_hash",
    }
    missing = required - set(comments.columns)
    if missing:
        raise ValueError(f"Missing discussion-history columns: {sorted(missing)}")
    # Production builds hand ownership of the stage input to this function.
    # Keeping only the ordered version avoids retaining an unordered full-size
    # copy throughout the multi-hour pass. Tests and external callers retain
    # the non-mutating default.
    output = comments if consume_input else comments.copy()
    output["created_at"] = _as_utc(output["created_at"])
    output.sort_values(
        ["story_id", "created_at", "comment_id"],
        kind="stable",
        na_position="last",
        ignore_index=True,
        inplace=True,
    )
    for name in (
        "prior_roots",
        "prior_comments",
        "comments_prev_hour",
        "branch_prior_comments",
        "branch_comments_prev_hour",
        "author_prior_comments_story",
    ):
        output[name] = np.int64(0)

    started = time.monotonic()
    processed = 0
    next_report = progress_every_rows or 0
    for _, story_indices in output.groupby("story_id", sort=False).groups.items():
        # The complete frame is already in stable story/time order. Only this
        # bounded per-story view is materialized while calculating histories.
        story = output.loc[story_indices]
        roots_before = 0
        comments_before = 0
        recent: deque[pd.Timestamp] = deque()
        branch_counts: defaultdict[str, int] = defaultdict(int)
        branch_recent: defaultdict[str, deque[pd.Timestamp]] = defaultdict(deque)
        author_counts: defaultdict[str, int] = defaultdict(int)
        for timestamp, batch in story.groupby("created_at", sort=True, dropna=False):
            if pd.isna(timestamp):
                continue
            hour_start = timestamp - pd.Timedelta(hours=1)
            while recent and recent[0] < hour_start:
                recent.popleft()
            for idx, row in batch.iterrows():
                branch = str(row["root_comment_id"] or row["comment_id"])
                branch_queue = branch_recent[branch]
                while branch_queue and branch_queue[0] < hour_start:
                    branch_queue.popleft()
                author = str(row["author_hash"] or "")
                output.at[idx, "prior_roots"] = roots_before
                output.at[idx, "prior_comments"] = comments_before
                output.at[idx, "comments_prev_hour"] = len(recent)
                output.at[idx, "branch_prior_comments"] = 0 if row["is_root"] else branch_counts[branch]
                output.at[idx, "branch_comments_prev_hour"] = 0 if row["is_root"] else len(branch_queue)
                output.at[idx, "author_prior_comments_story"] = author_counts[author] if author else 0
            for idx, row in batch.iterrows():
                branch = str(row["root_comment_id"] or row["comment_id"])
                author = str(row["author_hash"] or "")
                comments_before += 1
                roots_before += int(bool(row["is_root"]))
                recent.append(timestamp)
                branch_counts[branch] += 1
                branch_recent[branch].append(timestamp)
                if author:
                    author_counts[author] += 1
        processed += len(story)
        if progress_every_rows and (processed >= next_report or processed == len(output)):
            _progress_line("Discussion history", processed, len(output), started)
            next_report = processed + progress_every_rows
    return output


def compute_author_history(
    comments: pd.DataFrame,
    *,
    target_mask: pd.Series | None = None,
    window_days: int = 30,
    progress_every_rows: int | None = None,
    consume_input: bool = False,
) -> pd.DataFrame:
    """Compute prior cross-article history using collection-snapshot vote totals."""
    required = {
        "comment_id",
        "story_id",
        "created_at",
        "author_hash",
        "votes_positive",
        "votes_negative",
    }
    missing = required - set(comments.columns)
    if missing:
        raise ValueError(f"Missing author-history columns: {sorted(missing)}")
    output = comments if consume_input else comments.copy()
    output["created_at"] = _as_utc(output["created_at"])
    if target_mask is None:
        target_mask = pd.Series(True, index=output.index)
    target_mask = target_mask.reindex(output.index, fill_value=False)
    output["_history_target"] = target_mask.to_numpy(dtype=bool)
    output.sort_values(
        ["created_at", "comment_id"],
        kind="stable",
        na_position="last",
        ignore_index=True,
        inplace=True,
    )
    output["author_prior_30d_comments"] = np.int64(0)
    output["author_prior_30d_snapshot_upvotes"] = np.int64(0)
    output["author_prior_30d_snapshot_downvotes"] = np.int64(0)

    histories: defaultdict[str, deque[tuple[pd.Timestamp, str, int, int]]] = defaultdict(deque)
    totals: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_story: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0, 0])
    )
    window = pd.Timedelta(days=window_days)
    started = time.monotonic()
    processed = 0
    next_report = progress_every_rows or 0
    for timestamp, batch in output.groupby("created_at", sort=False, dropna=False):
        if pd.isna(timestamp):
            continue
        for idx, row in batch.iterrows():
            if not bool(row["_history_target"]):
                continue
            author = str(row["author_hash"] or "")
            if not author:
                continue
            history = histories[author]
            cutoff = timestamp - window
            while history and history[0][0] < cutoff:
                _, old_story, up, down = history.popleft()
                totals[author][0] -= 1
                totals[author][1] -= up
                totals[author][2] -= down
                by_story[author][old_story][0] -= 1
                by_story[author][old_story][1] -= up
                by_story[author][old_story][2] -= down
            story = str(row["story_id"])
            focal = by_story[author][story]
            output.at[idx, "author_prior_30d_comments"] = totals[author][0] - focal[0]
            output.at[idx, "author_prior_30d_snapshot_upvotes"] = totals[author][1] - focal[1]
            output.at[idx, "author_prior_30d_snapshot_downvotes"] = totals[author][2] - focal[2]
        # Add the whole timestamp batch only after all focal rows were evaluated.
        for _, row in batch.iterrows():
            author = str(row["author_hash"] or "")
            if not author:
                continue
            story = str(row["story_id"])
            up = max(0, int(row["votes_positive"] or 0))
            down = max(0, int(row["votes_negative"] or 0))
            histories[author].append((timestamp, story, up, down))
            totals[author][0] += 1
            totals[author][1] += up
            totals[author][2] += down
            by_story[author][story][0] += 1
            by_story[author][story][1] += up
            by_story[author][story][2] += down
        processed += len(batch)
        if progress_every_rows and (processed >= next_report or processed == len(output)):
            _progress_line("Author history", processed, len(output), started)
            next_report = processed + progress_every_rows
    return output.drop(columns="_history_target")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def cttr(text: str) -> float:
    tokens = [token.lower() for token in WORD_RE.findall(text or "")]
    return len(set(tokens)) / math.sqrt(2 * len(tokens)) if tokens else 0.0


@lru_cache(maxsize=100_000)
def _syllables_de(word: str) -> int:
    try:
        import pyphen

        dictionary = getattr(_syllables_de, "_dictionary", None)
        if dictionary is None:
            dictionary = pyphen.Pyphen(lang="de_DE")
            setattr(_syllables_de, "_dictionary", dictionary)
        pieces = dictionary.inserted(word.lower()).split("-")
        return max(1, len([piece for piece in pieces if piece]))
    except ImportError:
        return max(1, len(VOWEL_GROUP_RE.findall(word.lower())))


def smog_de(text: str) -> float:
    words = WORD_RE.findall(text or "")
    if not words:
        return 0.0
    sentences = max(1, len(SENTENCE_RE.findall((text or "").strip() + " ")))
    polysyllables = sum(_syllables_de(word) >= 3 for word in words)
    return math.sqrt(polysyllables * 30.0 / sentences) - 2.0


def vienna_period(timestamp: pd.Timestamp) -> str | None:
    if pd.isna(timestamp):
        return None
    local = timestamp.to_pydatetime().astimezone(VIENNA)
    hour = local.hour
    if hour < 6:
        return "overnight"
    if local.weekday() >= 5:
        return "weekend_day_evening"
    if 9 <= hour < 18:
        return "weekday_work"
    return "weekday_shoulder_evening"


def split_article_passages(article: pd.Series) -> list[str]:
    values = [article.get("title"), article.get("subtitle")]
    body = str(article.get("body") or "")
    values.extend(re.split(r"\n\s*\n+", body))
    return [re.sub(r"\s+", " ", str(value)).strip() for value in values if value and str(value).strip()]


def article_similarity_top3(
    comment_vectors: np.ndarray,
    passage_vectors: np.ndarray,
    *,
    block_size: int = 4_096,
) -> np.ndarray:
    """Mean top-three passage cosine, calculated once in bounded row blocks."""
    if passage_vectors.size == 0:
        return np.full(len(comment_vectors), np.nan, dtype=np.float32)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    k = min(3, len(passage_vectors))
    output = np.empty(len(comment_vectors), dtype=np.float32)
    for start in range(0, len(comment_vectors), block_size):
        stop = min(start + block_size, len(comment_vectors))
        similarities = comment_vectors[start:stop] @ passage_vectors.T
        top = np.partition(similarities, similarities.shape[1] - k, axis=1)[:, -k:]
        output[start:stop] = top.mean(axis=1)
    return output


def _novelty_exact(vectors: np.ndarray, timestamps: pd.Series, eligible: np.ndarray) -> np.ndarray:
    novelty = np.full(len(vectors), np.nan, dtype=np.float32)
    previous: list[int] = []
    frame = pd.DataFrame({"timestamp": timestamps, "position": np.arange(len(vectors))})
    for _, batch in frame.groupby("timestamp", sort=True, dropna=False):
        positions = batch["position"].to_numpy()
        query_positions = positions[eligible[positions]]
        if previous and len(query_positions):
            similarities = vectors[query_positions] @ vectors[np.asarray(previous)].T
            novelty[query_positions] = 1.0 - similarities.max(axis=1)
        previous.extend(int(pos) for pos in query_positions)
    return novelty


def _novelty_hnsw(vectors: np.ndarray, timestamps: pd.Series, eligible: np.ndarray) -> np.ndarray:
    try:
        import hnswlib
    except ImportError as exc:
        raise RuntimeError(
            "Large-discussion novelty requires hnswlib; install requirements-analysis.txt"
        ) from exc
    novelty = np.full(len(vectors), np.nan, dtype=np.float32)
    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=int(eligible.sum()), ef_construction=200, M=32, random_seed=BASE_SEED)
    index.set_ef(100)
    added = 0
    frame = pd.DataFrame({"timestamp": timestamps, "position": np.arange(len(vectors))})
    for _, batch in frame.groupby("timestamp", sort=True, dropna=False):
        positions = batch["position"].to_numpy()
        query_positions = positions[eligible[positions]]
        if added and len(query_positions):
            _, distances = index.knn_query(vectors[query_positions], k=1)
            novelty[query_positions] = distances[:, 0]
        if len(query_positions):
            index.add_items(vectors[query_positions], query_positions)
            added += len(query_positions)
    return novelty


def temporal_novelty(
    vectors: np.ndarray,
    timestamps: pd.Series,
    eligible: np.ndarray,
    *,
    exact_threshold: int,
) -> np.ndarray:
    if int(eligible.sum()) <= exact_threshold:
        return _novelty_exact(vectors, timestamps, eligible)
    return _novelty_hnsw(vectors, timestamps, eligible)


def validate_approximate_novelty(
    vectors: np.ndarray,
    timestamps: pd.Series,
    eligible: np.ndarray,
    approximate: np.ndarray,
    *,
    seed: int,
    sample_size: int = 100,
) -> dict[str, Any]:
    """Compare approximate distances with exact prior-neighbor distances."""
    timestamp_values = pd.Series(timestamps).reset_index(drop=True)
    candidates = np.flatnonzero(eligible & np.isfinite(approximate))
    if not len(candidates):
        return {"sample_size": 0, "mean_absolute_error": None, "max_absolute_error": None, "recall_within_1e_3": None}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(candidates, size=min(sample_size, len(candidates)), replace=False)
    errors: list[float] = []
    for position in sampled:
        previous = np.flatnonzero(
            eligible & (timestamp_values < timestamp_values.iloc[position]).to_numpy()
        )
        if not len(previous):
            continue
        exact = 1.0 - float((vectors[position] @ vectors[previous].T).max())
        errors.append(abs(exact - float(approximate[position])))
    if not errors:
        return {"sample_size": 0, "mean_absolute_error": None, "max_absolute_error": None, "recall_within_1e_3": None}
    error_array = np.asarray(errors)
    return {
        "sample_size": len(errors),
        "mean_absolute_error": float(error_array.mean()),
        "max_absolute_error": float(error_array.max()),
        "recall_within_1e_3": float((error_array <= 1e-3).mean()),
    }


def _read_dataset(table_root: Path, year: int, columns: list[str]) -> pd.DataFrame:
    import pyarrow.dataset as ds

    root = table_root / f"year={year}"
    if not root.exists():
        raise FileNotFoundError(root)
    dataset = ds.dataset(root, format="parquet", partitioning=None)
    available = set(dataset.schema.names)
    missing = set(columns) - available
    if missing:
        raise ValueError(f"Missing columns under {root}: {sorted(missing)}")
    return dataset.to_table(columns=columns).to_pandas()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd")
    os.replace(temporary, path)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _length_residual(frame: pd.DataFrame, outcome: str) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    valid = np.isfinite(frame[outcome]) & np.isfinite(frame["log_words"])
    if valid.sum() < 20:
        raise ValueError(f"Too few observations to length-adjust {outcome}")
    model = make_pipeline(
        SplineTransformer(n_knots=6, degree=3, include_bias=False), Ridge(alpha=1.0)
    )
    x = frame.loc[valid, ["log_words"]].to_numpy()
    y = frame.loc[valid, outcome].to_numpy()
    model.fit(x, y)
    predicted = np.full(len(frame), np.nan)
    predicted[valid] = model.predict(x)
    residual = frame[outcome].to_numpy(dtype=float) - predicted
    return residual, {"outcome": outcome, "n_knots": 6, "degree": 3, "ridge_alpha": 1.0}


def _feature_registry() -> dict[str, Any]:
    labels = {
        "log_words": "Comment length (log words)",
        "sentiment_positive": "Positive sentiment",
        "sentiment_negative": "Negative sentiment",
        "lexdiv_length_adjusted": "Length-adjusted lexical diversity",
        "reading_level_length_adjusted": "Length-adjusted reading difficulty",
        "url_present": "URL present",
        "article_similarity_top3": "Article similarity (top-three passages)",
        "novelty_prior_roots_model": "Novelty from earlier roots",
        "novelty_prior_all_model": "Novelty from earlier comments",
        "log_hours_since_article": "Hours since article publication",
        "log_prior_roots": "Earlier roots",
        "log_prior_comments": "Earlier comments",
        "log_comments_prev_hour": "Comments in previous hour",
        "vienna_overnight": "Overnight posting period",
        "vienna_weekday_shoulder_evening": "Weekday shoulder/evening",
        "vienna_weekend_day_evening": "Weekend daytime/evening",
        "log_author_prior_30d_comments": "Author comments in prior 30 days",
        "log_author_prior_30d_snapshot_upvotes": "Snapshot upvotes on prior comments",
        "log_author_prior_30d_snapshot_downvotes": "Snapshot downvotes on prior comments",
        "log_author_prior_comments_story": "Author's earlier comments in article",
        "is_reply": "Reply rather than root",
        "log_depth": "Reply depth",
        "log_branch_prior_comments": "Earlier comments in branch",
        "log_branch_comments_prev_hour": "Branch comments in previous hour",
    }
    return {
        "version": 1,
        "models": {
            "root": {"features": ROOT_MODEL_FEATURES},
            "all": {"features": ALL_MODEL_FEATURES},
        },
        "features": {
            name: {
                "label": labels[name],
                "standardize": name not in BINARY_FEATURES,
                "deferred": False,
            }
            for name in sorted(set(ROOT_MODEL_FEATURES + ALL_MODEL_FEATURES))
        }
        | {
            name: {"label": label, "standardize": False, "deferred": True}
            for name, label in {
                "toxicity_probability": "Toxicity probability",
                "engagement_probability": "Engaging-comment probability",
                "fact_claim_probability": "Fact-claim probability",
            }.items()
        },
        "categorical_reference": {"vienna_period": "weekday_work"},
    }


def _prepare_model_columns(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    output = frame.copy()
    output["relative_votes"] = output["votes_positive"] - output["votes_negative"]
    for raw in (
        "hours_since_article",
        "prior_roots",
        "prior_comments",
        "comments_prev_hour",
        "author_prior_30d_comments",
        "author_prior_30d_snapshot_upvotes",
        "author_prior_30d_snapshot_downvotes",
        "author_prior_comments_story",
        "depth",
        "branch_prior_comments",
        "branch_comments_prev_hour",
    ):
        output[f"log_{raw}"] = np.log1p(output[raw].clip(lower=0).astype(float))
    output["is_reply"] = (~output["is_root"].astype(bool)).astype(int)
    for period in ("overnight", "weekday_shoulder_evening", "weekend_day_evening"):
        output[f"vienna_{period}"] = (output["vienna_period"] == period).astype(int)
    novelty = "novelty_prior_roots" if scope == "root" else "novelty_prior_all"
    finite = np.isfinite(output[novelty])
    if not finite.any():
        raise ValueError(f"No finite values for {novelty}")
    output[f"{novelty}_model"] = output[novelty].fillna(output.loc[finite, novelty].mean())
    return output


def _make_choice_set(
    features: pd.DataFrame,
    raw_comments: pd.DataFrame,
    *,
    scope: str,
    config: FeatureBuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if scope not in {"root", "all"}:
        raise ValueError(scope)
    scope_raw = raw_comments[raw_comments["is_root"].astype(bool)] if scope == "root" else raw_comments
    scope_features = features[features["is_root"].astype(bool)] if scope == "root" else features
    invalid_sticky_stories = set(
        scope_raw.loc[
            scope_raw["is_sticky"].astype(bool)
            & (
                ~scope_raw["lifecycle_status"].eq("Published")
                |
                scope_raw["effective_text"].fillna("").str.strip().eq("")
                | scope_raw["created_at"].isna()
                | scope_raw["comment_id"].fillna("").astype(str).str.strip().eq("")
            ),
            "story_id",
        ].astype(str)
    )
    output = scope_features[~scope_features["story_id"].astype(str).isin(invalid_sticky_stories)].copy()
    if config.require_page_publication_time:
        validity = output.groupby("story_id").agg(
            page_time=("published_at_source", lambda x: bool(x.eq("page").all())),
            invalid_time=("invalid_posting_time", "any"),
        )
        valid_stories = validity.index[validity["page_time"] & ~validity["invalid_time"]]
        output = output[output["story_id"].isin(valid_stories)]
    if config.lookback_root is None and config.exclude_january_without_lookback:
        output = output[~((output["article_year"] == config.year) & (output["article_month"] == 1))]

    counts = output.groupby("story_id").agg(
        n_candidates=("comment_id", "size"), n_picks=("is_sticky", "sum")
    )
    eligible = counts[(counts["n_picks"] > 0) & (counts["n_picks"] < counts["n_candidates"])]
    output = output.merge(eligible, left_on="story_id", right_index=True, validate="many_to_one")
    output = _prepare_model_columns(output, scope)
    output["curator_selected"] = output["is_sticky"].astype(bool)
    output, ties = assign_audience_labels(
        output, draws=config.tie_draws, seed=config.seed
    )
    features_used = ROOT_MODEL_FEATURES if scope == "root" else ALL_MODEL_FEATURES
    missingness = output[features_used].isna().sum()
    if int(missingness.sum()):
        raise ValueError(f"Missing model features in {scope}: {missingness[missingness > 0].to_dict()}")
    if output.duplicated(["story_id", "comment_id"]).any():
        raise ValueError(f"Duplicate candidate keys in {scope}")
    output["candidate_scope"] = scope
    keep = [
        "story_id",
        "comment_id",
        "article_year",
        "article_month",
        "candidate_scope",
        "n_candidates",
        "n_picks",
        "curator_selected",
        "relative_votes",
    ] + [f"audience_selected_draw_{draw:02d}" for draw in range(1, config.tie_draws + 1)]
    raw_descriptive = [
        "word_count",
        "cttr",
        "smog_de",
        "sentiment_neutral",
        "hours_since_article",
        "prior_roots",
        "prior_comments",
        "comments_prev_hour",
        "branch_prior_comments",
        "branch_comments_prev_hour",
        "author_prior_30d_comments",
        "author_prior_30d_snapshot_upvotes",
        "author_prior_30d_snapshot_downvotes",
        "author_prior_comments_story",
        "vienna_period",
    ]
    keep += [name for name in raw_descriptive + features_used if name not in keep]
    summary = {
        "scope": scope,
        "candidate_rows": len(output),
        "eligible_stories": int(output["story_id"].nunique()),
        "sticky_comments": int(output["curator_selected"].sum()),
        "invalid_sticky_stories_excluded": len(invalid_sticky_stories),
        "ambiguous_vote_cutoffs": int(ties["ambiguous_cutoff"].sum()),
    }
    return output[keep].sort_values(["story_id", "comment_id"]), ties, summary


def _load_sentiment_encoder(config: FeatureBuildConfig) -> SentimentEncoder:
    if config.nlp_mode == "pilot":
        return PilotLexiconSentiment()
    return GermanSentimentEncoder(device=config.device, revision=config.sentiment_revision)


def build_analysis_features(
    config: FeatureBuildConfig,
    *,
    sentiment_encoder: SentimentEncoder | None = None,
) -> dict[str, Any]:
    """Run the resumable feature build using precomputed semantic scalars."""
    build_started = time.monotonic()
    print(
        "Feature preflight: "
        f"year={config.year} requested_device={config.device} "
        f"inference_mode={config.inference_mode} nlp_mode={config.nlp_mode}",
        flush=True,
    )
    qa_path = config.data_root / "qa_summary" / f"year={config.year}" / "summary.json"
    if not qa_path.exists():
        raise FileNotFoundError(qa_path)
    qa = json.loads(qa_path.read_text())
    validate_qa_summary(qa, allow_incomplete=config.allow_incomplete)
    from .similarities import resolve_similarity_store

    requested_similarity_store = config.similarity_store or config.similarity_root
    similarity_store, similarity_manifest = resolve_similarity_store(
        requested_similarity_store,
        model_id=config.embedding_model_id,
        revision=config.embedding_revision,
        year=config.year,
        require_complete_source=config.inference_mode,
    )
    fingerprint = dataset_fingerprint(config.data_root, config.year)
    expected_similarity_fingerprint = similarity_manifest.get(
        "target_dataset_fingerprints", {}
    ).get(str(config.year))
    if expected_similarity_fingerprint != fingerprint:
        raise RuntimeError(
            "The precomputed similarity store does not match the current source "
            f"collection for {config.year}."
        )
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
        if key not in {"overwrite", "output_root"}
    }
    config_payload["resolved_similarity_store"] = str(similarity_store)
    config_payload["similarity_build_signature"] = similarity_manifest["build_signature"]
    build_signature = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    state_path = config.output_root / "build_state.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("dataset_fingerprint") != fingerprint and not config.overwrite:
            raise RuntimeError(
                "Input collection changed since checkpoints were created; use a new output "
                "directory or set overwrite=True"
            )
        if previous.get("build_signature") != build_signature and not config.overwrite:
            raise RuntimeError(
                "Feature-build configuration changed; use a new output directory or "
                "set overwrite=True"
            )
    watermark = "INFERENCE" if config.inference_mode else "PILOT_NOT_FOR_INFERENCE"
    state = {
        "status": "running",
        "dataset_fingerprint": fingerprint,
        "build_signature": build_signature,
        "watermark": watermark,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    }
    _atomic_json(state, state_path)

    comment_columns = [
        "comment_id",
        "story_id",
        "parent_comment_id",
        "root_comment_id",
        "depth",
        "is_root",
        "created_at",
        "effective_text",
        "lifecycle_status",
        "is_sticky",
        "votes_positive",
        "votes_negative",
        "author_hash",
        "year",
        "month",
    ]
    article_columns = [
        "story_id",
        "year",
        "month",
        "published_at",
        "published_at_source",
    ]
    stage_started = time.monotonic()
    print("Feature stage: loading source comments and articles", flush=True)
    target_raw = _read_dataset(config.data_root / "comments", config.year, comment_columns)
    articles = _read_dataset(config.data_root / "articles", config.year, article_columns)
    print(
        "Feature stage complete: source load | "
        f"comments={len(target_raw):,} articles={len(articles):,} "
        f"elapsed={_format_duration(time.monotonic() - stage_started)}",
        flush=True,
    )
    if articles.duplicated("story_id").any():
        raise ValueError("story_id is not unique in article data")
    if config.max_stories is not None:
        sticky_stories = sorted(
            target_raw.loc[target_raw["is_sticky"].astype(bool), "story_id"].astype(str).unique()
        )
        other_stories = sorted(set(target_raw["story_id"].astype(str)) - set(sticky_stories))
        selected_stories = (sticky_stories + other_stories)[: config.max_stories]
        target_raw = target_raw[target_raw["story_id"].astype(str).isin(selected_stories)].copy()
        articles = articles[articles["story_id"].astype(str).isin(selected_stories)].copy()
    source_comment_count = len(target_raw)
    if target_raw.duplicated("comment_id").any():
        raise ValueError("comment_id is not unique in source data")
    target_raw["created_at"] = _as_utc(target_raw["created_at"])
    target_raw["is_target"] = True

    candidate_mask = (
        target_raw["lifecycle_status"].eq("Published")
        & target_raw["effective_text"].fillna("").str.strip().ne("")
        & target_raw["created_at"].notna()
    )
    target_raw["_candidate"] = candidate_mask.to_numpy(dtype=bool)

    history_checkpoint_root = (
        config.output_root
        / "history_checkpoints"
        / f"build={build_signature[:8]}-{fingerprint[:8]}"
    )
    discussion_checkpoint = history_checkpoint_root / "discussion_history.parquet"
    author_checkpoint = history_checkpoint_root / "author_history.parquet"
    discussion_feature_columns = [
        "comment_id",
        "prior_roots",
        "prior_comments",
        "comments_prev_hour",
        "branch_prior_comments",
        "branch_comments_prev_hour",
        "author_prior_comments_story",
    ]
    author_feature_columns = [
        "comment_id",
        "author_prior_30d_comments",
        "author_prior_30d_snapshot_upvotes",
        "author_prior_30d_snapshot_downvotes",
    ]

    if discussion_checkpoint.exists() and not config.overwrite:
        print(
            f"Feature stage: reusing discussion-history checkpoint {discussion_checkpoint}",
            flush=True,
        )
        discussion_features = pd.read_parquet(
            discussion_checkpoint,
            columns=discussion_feature_columns + ["_candidate"],
        )
        discussion_features = discussion_features.loc[
            discussion_features["_candidate"].astype(bool), discussion_feature_columns
        ].copy()
    else:
        print("Feature stage: calculating strictly-prior discussion activity", flush=True)
        discussion_source = target_raw.loc[target_raw["created_at"].notna()].copy()
        discussion_all = compute_discussion_history(
            discussion_source,
            progress_every_rows=config.progress_every_rows,
            consume_input=True,
        )
        _atomic_parquet(discussion_all, discussion_checkpoint)
        print(
            "Feature stage checkpointed: discussion history | "
            f"rows={len(discussion_all):,} path={discussion_checkpoint}",
            flush=True,
        )
        discussion_features = discussion_all.loc[
            discussion_all["_candidate"].astype(bool), discussion_feature_columns
        ].copy()
        del discussion_all, discussion_source
        gc.collect()
    if discussion_features.duplicated("comment_id").any():
        raise ValueError("Duplicate comment_id values in discussion-history checkpoint")
    state.setdefault("stages", {})["discussion_history"] = {
        "status": "complete",
        "path": str(discussion_checkpoint),
        "candidate_rows": len(discussion_features),
    }
    _atomic_json(state, state_path)

    # Activity counts include every recorded posting with a timestamp, including
    # later-deleted tombstones. Text/NLP candidate eligibility is applied only
    # after these posting-time histories are computed.
    if author_checkpoint.exists() and not config.overwrite:
        print(
            f"Feature stage: reusing author-history checkpoint {author_checkpoint}",
            flush=True,
        )
        author_features = pd.read_parquet(
            author_checkpoint,
            columns=author_feature_columns + ["is_target", "_candidate"],
        )
        author_features = author_features.loc[
            author_features["is_target"].astype(bool)
            & author_features["_candidate"].astype(bool),
            author_feature_columns,
        ].copy()
    else:
        author_source_columns = [
            "comment_id",
            "story_id",
            "created_at",
            "author_hash",
            "votes_positive",
            "votes_negative",
            "is_target",
            "_candidate",
        ]
        author_source = target_raw.loc[
            target_raw["created_at"].notna(), author_source_columns
        ].copy()
        if config.lookback_root is not None:
            lookback_columns = [
                "comment_id",
                "story_id",
                "created_at",
                "author_hash",
                "votes_positive",
                "votes_negative",
            ]
            lookback = _read_dataset(
                config.lookback_root / "comments",
                config.year - 1,
                lookback_columns,
            )
            lookback["created_at"] = _as_utc(lookback["created_at"])
            lookback = lookback.loc[lookback["created_at"].notna()].copy()
            lookback["is_target"] = False
            lookback["_candidate"] = False
            author_source = pd.concat(
                [lookback[author_source_columns], author_source],
                ignore_index=True,
                sort=False,
            )
            del lookback
            gc.collect()
        print("Feature stage: calculating 30-day author history", flush=True)
        author_all = compute_author_history(
            author_source,
            target_mask=author_source["is_target"].astype(bool),
            progress_every_rows=config.progress_every_rows,
            consume_input=True,
        )
        _atomic_parquet(author_all, author_checkpoint)
        print(
            "Feature stage checkpointed: author history | "
            f"rows={len(author_all):,} path={author_checkpoint}",
            flush=True,
        )
        author_features = author_all.loc[
            author_all["is_target"].astype(bool)
            & author_all["_candidate"].astype(bool),
            author_feature_columns,
        ].copy()
        del author_all, author_source
        gc.collect()
    if author_features.duplicated("comment_id").any():
        raise ValueError("Duplicate comment_id values in author-history checkpoint")
    state.setdefault("stages", {})["author_history"] = {
        "status": "complete",
        "path": str(author_checkpoint),
        "candidate_rows": len(author_features),
    }
    _atomic_json(state, state_path)

    print("Feature stage: joining compact history results to candidates", flush=True)
    stage_started = time.monotonic()
    base = target_raw.loc[candidate_mask].drop(columns=["_candidate"]).copy()
    del target_raw, candidate_mask
    gc.collect()
    history_features = discussion_features.merge(
        author_features,
        on="comment_id",
        validate="one_to_one",
    )
    del discussion_features, author_features
    gc.collect()
    base = base.merge(history_features, on="comment_id", validate="one_to_one")
    del history_features
    gc.collect()
    print(
        "Feature stage complete: compact history joins | "
        f"rows={len(base):,} elapsed={_format_duration(time.monotonic() - stage_started)}",
        flush=True,
    )
    print("Feature stage: attaching compact article controls", flush=True)
    # Article text has already been represented by the precomputed semantic
    # similarities. Never replicate title/subtitle/body over millions of
    # comment rows: only these compact article controls are needed downstream.
    article_data = articles[
        [
            "story_id",
            "year",
            "month",
            "published_at",
            "published_at_source",
        ]
    ].rename(columns={"year": "article_year", "month": "article_month"})
    del articles
    gc.collect()
    if article_data.duplicated("story_id").any():
        raise ValueError("story_id is not unique in compact article controls")
    article_lookup = article_data.set_index("story_id")
    missing_article_stories = set(base["story_id"].unique()) - set(article_lookup.index)
    if missing_article_stories:
        raise ValueError(
            f"Comments reference {len(missing_article_stories)} missing articles"
        )
    article_columns_to_map = [
        "article_year",
        "article_month",
        "published_at",
        "published_at_source",
    ]
    for column in article_columns_to_map:
        base[column] = base["story_id"].map(article_lookup[column])
    del article_lookup
    del article_data
    gc.collect()
    rss = _linux_rss_gib()
    print(
        "Feature stage complete: compact article controls"
        + (f" | rss={rss:.1f}GiB" if rss is not None else ""),
        flush=True,
    )
    base["published_at"] = _as_utc(base["published_at"])
    base["hours_since_article"] = (
        base["created_at"] - base["published_at"]
    ).dt.total_seconds() / 3600
    base["invalid_posting_time"] = base["hours_since_article"].isna() | (base["hours_since_article"] < 0)
    base["vienna_period"] = base["created_at"].map(vienna_period)
    print(
        f"Feature stage: calculating local text measures for {len(base):,} candidates",
        flush=True,
    )
    stage_started = time.monotonic()
    base["word_count"] = base["effective_text"].map(word_count)
    base["log_words"] = np.log1p(base["word_count"])
    base["cttr"] = base["effective_text"].map(cttr)
    base["smog_de"] = base["effective_text"].map(smog_de)
    base["url_present"] = base["effective_text"].str.contains(URL_RE).astype(int)
    print(
        "Feature stage complete: local text measures | "
        f"elapsed={_format_duration(time.monotonic() - stage_started)}",
        flush=True,
    )

    if sentiment_encoder is None:
        sentiment_encoder = _load_sentiment_encoder(config)
    if config.inference_mode and "PILOT_ONLY" in sentiment_encoder.model_id:
        raise ValueError("Pilot sentiment adapters cannot be used for inference")
    print(
        "Feature NLP preflight: "
        f"model={sentiment_encoder.model_id} "
        f"revision={getattr(sentiment_encoder, 'resolved_revision', 'unresolved')} "
        f"device={getattr(sentiment_encoder, 'device', select_torch_device(config.device))} "
        f"batch={config.sentiment_batch_size}",
        flush=True,
    )

    checkpoint_root = (
        config.output_root
        / "scalar_features"
        / f"build={build_signature[:8]}-{fingerprint[:8]}"
        / f"year={config.year}"
    )
    grouped_stories = base.groupby("story_id", sort=True)
    total_stories = grouped_stories.ngroups
    total_candidates = len(base)
    processed_candidates = 0
    new_candidates = 0
    written_stories = 0
    skipped_stories = 0
    stage_started = time.monotonic()
    print(
        "Feature stage: sentiment, semantic joins, and scalar checkpoints | "
        f"stories={total_stories:,} candidates={total_candidates:,}",
        flush=True,
    )
    for story_index, (story_id, story) in enumerate(grouped_stories, start=1):
        month = int(story["article_month"].iloc[0])
        destination = checkpoint_root / f"month={month:02d}" / f"{story_id}.parquet"
        if destination.exists() and not config.overwrite:
            skipped_stories += 1
            processed_candidates += len(story)
            if story_index % config.progress_every_stories == 0 or story_index == total_stories:
                _progress_line(
                    "Scalar features",
                    story_index,
                    total_stories,
                    stage_started,
                    unit="stories",
                    detail=(
                        f"candidates={processed_candidates:,}/{total_candidates:,} "
                        f"new_rows={new_candidates:,} written={written_stories:,} "
                        f"skipped={skipped_stories:,}"
                    ),
                )
            continue
        story = story.sort_values(["created_at", "comment_id"]).copy()
        texts = story["effective_text"].astype(str).tolist()
        sentiments = sentiment_encoder.predict(texts, config.sentiment_batch_size)
        if sentiments.shape != (len(story), 3):
            raise ValueError("Sentiment encoder returned an unexpected shape")
        story[["sentiment_positive", "sentiment_negative", "sentiment_neutral"]] = sentiments
        similarity_path = (
            similarity_store
            / "scalars"
            / f"year={config.year}"
            / f"month={month:02d}"
            / f"{story_id}.parquet"
        )
        if not similarity_path.exists():
            raise FileNotFoundError(
                f"Missing precomputed semantic scalars for story {story_id}: {similarity_path}"
            )
        semantic_columns = [
            "story_id",
            "comment_id",
            "article_similarity_top3",
            "novelty_prior_all",
            "novelty_prior_roots",
        ]
        semantic = pd.read_parquet(similarity_path, columns=semantic_columns)
        if semantic.duplicated(["story_id", "comment_id"]).any():
            raise ValueError(f"Duplicate semantic-similarity keys in {similarity_path}")
        before = len(story)
        story = story.merge(
            semantic,
            on=["story_id", "comment_id"],
            how="left",
            validate="one_to_one",
            indicator="_semantic_join",
        )
        if len(story) != before or not story["_semantic_join"].eq("both").all():
            raise ValueError(f"Incomplete semantic-similarity join for story {story_id}")
        story = story.drop(columns="_semantic_join")
        story["nlp_watermark"] = watermark
        _atomic_parquet(story, destination)
        written_stories += 1
        processed_candidates += len(story)
        new_candidates += len(story)
        if story_index % config.progress_every_stories == 0 or story_index == total_stories:
            _progress_line(
                "Scalar features",
                story_index,
                total_stories,
                stage_started,
                unit="stories",
                detail=(
                    f"candidates={processed_candidates:,}/{total_candidates:,} "
                    f"new_rows={new_candidates:,} written={written_stories:,} "
                    f"skipped={skipped_stories:,}"
                ),
            )

    import pyarrow.dataset as ds

    print("Feature stage: assembling scalar dataset and length adjustments", flush=True)
    stage_started = time.monotonic()
    scalar = ds.dataset(checkpoint_root, format="parquet", partitioning=None).to_table().to_pandas()
    scalar["lexdiv_length_adjusted"], lex_meta = _length_residual(scalar, "cttr")
    scalar["reading_level_length_adjusted"], reading_meta = _length_residual(scalar, "smog_de")
    print(
        "Feature stage complete: scalar assembly | "
        f"rows={len(scalar):,} elapsed={_format_duration(time.monotonic() - stage_started)}",
        flush=True,
    )

    print("Feature stage: constructing root and all-comment choice sets", flush=True)
    stage_started = time.monotonic()
    # The full raw table was deliberately released before NLP inference. Reload
    # it only for final curator-set validation and immediately release it again.
    choice_raw = _read_dataset(config.data_root / "comments", config.year, comment_columns)
    choice_raw["created_at"] = _as_utc(choice_raw["created_at"])
    scalar_story_ids = set(scalar["story_id"].astype(str).unique())
    choice_raw = choice_raw[
        choice_raw["story_id"].astype(str).isin(scalar_story_ids)
    ].copy()
    root, root_ties, root_summary = _make_choice_set(
        scalar, choice_raw, scope="root", config=config
    )
    all_comments, all_ties, all_summary = _make_choice_set(
        scalar, choice_raw, scope="all", config=config
    )
    del choice_raw, scalar_story_ids
    gc.collect()
    _atomic_parquet(root, config.output_root / "choice_set_root.parquet")
    _atomic_parquet(all_comments, config.output_root / "choice_set_all.parquet")
    _atomic_parquet(root_ties.assign(candidate_scope="root"), config.output_root / "tie_diagnostics_root.parquet")
    _atomic_parquet(all_ties.assign(candidate_scope="all"), config.output_root / "tie_diagnostics_all.parquet")
    print(
        "Feature stage complete: choice sets | "
        f"root_rows={len(root):,} all_rows={len(all_comments):,} "
        f"elapsed={_format_duration(time.monotonic() - stage_started)}",
        flush=True,
    )
    registry = _feature_registry()
    registry["length_adjustment"] = [lex_meta, reading_meta]
    registry["nlp"] = {
        "sentiment_model": sentiment_encoder.model_id,
        "sentiment_revision": getattr(sentiment_encoder, "resolved_revision", "unresolved"),
        "embedding_model": similarity_manifest["embedding_model"]["model_id"],
        "embedding_revision": similarity_manifest["embedding_model"]["resolved_revision"],
        "similarity_store": str(similarity_store),
        "similarity_build_signature": similarity_manifest["build_signature"],
        "watermark": watermark,
    }
    _atomic_json(registry, config.output_root / "feature_manifest.json")
    similarity_validation = similarity_store / "novelty_validation.json"
    if not similarity_validation.exists():
        raise FileNotFoundError(similarity_validation)
    _atomic_json(
        json.loads(similarity_validation.read_text()),
        config.output_root / "novelty_validation.json",
    )
    summary = {
        "watermark": watermark,
        "qa": {
            "passed": qa.get("passed"),
            "allow_incomplete": qa.get("allow_incomplete"),
            "nonterminal_stories": qa.get("nonterminal_stories"),
            "status_counts": qa.get("status_counts"),
        },
        "source": {
            "comments": source_comment_count,
            "articles": len(articles),
            "dataset_fingerprint": fingerprint,
            "similarity_store": str(similarity_store),
            "similarity_build_signature": similarity_manifest["build_signature"],
        },
        "root": root_summary,
        "all": all_summary,
        "models": registry["nlp"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": select_torch_device(config.device),
            "packages": _analysis_package_versions(),
        },
        "execution": {
            "elapsed_seconds": time.monotonic() - build_started,
            "new_story_checkpoints": written_stories,
            "skipped_story_checkpoints": skipped_stories,
        },
    }
    _atomic_json(summary, config.output_root / "provenance_manifest.json")
    _atomic_json(
        {
            **state,
            "status": "complete",
            "manifest": str(config.output_root / "provenance_manifest.json"),
        },
        state_path,
    )
    print(
        "Feature build complete: "
        f"elapsed={_format_duration(time.monotonic() - build_started)} "
        f"output={config.output_root}",
        flush=True,
    )
    return summary
