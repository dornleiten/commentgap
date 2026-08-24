"""Build model-ready features from the normalized 2025 Parquet collection."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import gc
import hashlib
import inspect
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

from aqua_runtime.schema import text_hash

from .nlp import (
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    DEFAULT_SENTIMENT_MODEL_ID,
    DEFAULT_SENTIMENT_MODEL_REVISION,
    DEFAULT_TOXICITY_MODEL_ID,
    DEFAULT_TOXICITY_MODEL_REVISION,
    PilotLexiconSentiment,
    PilotLexiconToxicity,
    SentimentEncoder,
    TextDetoxToxicityEncoder,
    ToxicityEncoder,
    XLMTwitterSentimentEncoder,
    select_torch_device,
)


URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
WORD_RE = re.compile(r"\b[\wÄÖÜäöüß]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
VOWEL_GROUP_RE = re.compile(r"[aeiouyäöü]+", re.IGNORECASE)
VIENNA = ZoneInfo("Europe/Vienna")

BASE_SEED = 20260813
DEFAULT_TIE_DRAWS = 10
LOCAL_TEXT_OUTPUT_SEMANTICS_VERSION = 1
LOCAL_TEXT_COLUMNS = ["word_count", "log_words", "cttr", "smog_de", "url_present"]
CHOICE_WRITE_BATCH_ROWS = 25_000
CHOICE_READ_WINDOW_STORIES = 250


ROOT_MODEL_FEATURES = [
    "log_words",
    "sentiment_positive",
    "sentiment_negative",
    "toxicity_probability",
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
    aqua_store: Path | None = None
    year: int = 2025
    lookback_root: Path | None = None
    allow_incomplete: bool = False
    inference_mode: bool = True
    nlp_mode: str = "real"
    device: str = "auto"
    sentiment_model_id: str = DEFAULT_SENTIMENT_MODEL_ID
    sentiment_revision: str | None = None
    toxicity_model_id: str = DEFAULT_TOXICITY_MODEL_ID
    toxicity_revision: str | None = None
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    embedding_revision: str | None = None
    sentiment_batch_size: int = 32
    toxicity_batch_size: int = 16
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
        if self.aqua_store is not None:
            object.__setattr__(self, "aqua_store", Path(self.aqua_store))
        if self.lookback_root is not None:
            object.__setattr__(self, "lookback_root", Path(self.lookback_root))
        if self.nlp_mode not in {"real", "pilot"}:
            raise ValueError("nlp_mode must be 'real' or 'pilot'")
        if not self.sentiment_model_id.strip():
            raise ValueError("sentiment_model_id cannot be empty")
        if not self.toxicity_model_id.strip():
            raise ValueError("toxicity_model_id cannot be empty")
        if (
            self.sentiment_model_id == DEFAULT_SENTIMENT_MODEL_ID
            and self.sentiment_revision is None
        ):
            object.__setattr__(
                self, "sentiment_revision", DEFAULT_SENTIMENT_MODEL_REVISION
            )
        if (
            self.toxicity_model_id == DEFAULT_TOXICITY_MODEL_ID
            and self.toxicity_revision is None
        ):
            object.__setattr__(
                self, "toxicity_revision", DEFAULT_TOXICITY_MODEL_REVISION
            )
        if (
            self.embedding_model_id == DEFAULT_EMBEDDING_MODEL_ID
            and self.embedding_revision is None
        ):
            object.__setattr__(
                self, "embedding_revision", DEFAULT_EMBEDDING_MODEL_REVISION
            )
        if not self.embedding_model_id.strip():
            raise ValueError("embedding_model_id cannot be empty")
        if self.sentiment_batch_size < 1:
            raise ValueError("sentiment_batch_size must be positive")
        if self.toxicity_batch_size < 1:
            raise ValueError("toxicity_batch_size must be positive")
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


def _identity_signature(identity: dict[str, Any]) -> str:
    """Return a stable cache identity for one independently reusable stage."""
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _adapter_implementation_signature(adapter: Any) -> str:
    """Identify numerical NLP semantics while permitting safe cache reuse.

    Production adapters expose a frozen compatibility signature. Diagnostic,
    logging, batching and validation changes can then reuse existing predictions;
    a change that can alter feature values must deliberately bump the signature.
    Injected/test adapters retain source-based invalidation.
    """
    frozen_signature = getattr(adapter, "cache_compatibility_signature", None)
    if frozen_signature:
        return str(frozen_signature)
    adapter_type = type(adapter)
    identity = f"{adapter_type.__module__}.{adapter_type.__qualname__}"
    try:
        source = inspect.getsource(adapter_type)
    except (OSError, TypeError):
        source = identity
    return hashlib.sha256(f"{identity}\n{source}".encode("utf-8")).hexdigest()


def _functions_implementation_signature(*functions: Any) -> str:
    parts: list[str] = []
    for function in functions:
        identity = f"{function.__module__}.{function.__qualname__}"
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            source = identity
        parts.append(f"{identity}\n{source}")
    return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()


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


def _release_unused_memory() -> None:
    """Return unused Arrow/glibc allocations to the OS when supported."""
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except (ImportError, AttributeError):
        pass
    if platform.system() == "Linux":
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass


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


def _local_text_identity(
    config: FeatureBuildConfig, fingerprint: str
) -> dict[str, Any]:
    try:
        pyphen_version = package_version("pyphen")
    except PackageNotFoundError:
        pyphen_version = "not-installed-vowel-fallback"
    return {
        "schema_version": 1,
        "output_semantics_version": LOCAL_TEXT_OUTPUT_SEMANTICS_VERSION,
        "year": config.year,
        "dataset_fingerprint": fingerprint,
        "candidate_filter": (
            "Published, non-empty effective_text, non-missing created_at"
        ),
        "max_stories": config.max_stories,
        "columns": LOCAL_TEXT_COLUMNS,
        "word_pattern": WORD_RE.pattern,
        "sentence_pattern": SENTENCE_RE.pattern,
        "vowel_group_pattern": VOWEL_GROUP_RE.pattern,
        "url_pattern": URL_RE.pattern,
        "url_pattern_flags": URL_RE.flags,
        "syllabification": {"language": "de_DE", "pyphen_version": pyphen_version},
        "smog_formula": "sqrt(polysyllables * 30 / sentences) - 2",
        "implementation_signature": _functions_implementation_signature(
            word_count,
            cttr,
            _syllables_de,
            smog_de,
        ),
    }


def compute_local_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic comment-local measures without retaining raw text."""
    required = {"story_id", "comment_id", "effective_text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing local-text input columns: {sorted(missing)}")
    if frame[["story_id", "comment_id", "effective_text"]].isna().any().any():
        raise ValueError("Local-text input keys and text cannot be null")
    if frame.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("Local-text input contains duplicate keys")
    output = frame[["story_id", "comment_id"]].copy()
    output["story_id"] = output["story_id"].astype(str)
    output["comment_id"] = output["comment_id"].astype(str)
    text = frame["effective_text"].astype(str)
    output["effective_text_hash"] = text.map(text_hash)
    output["word_count"] = text.map(word_count)
    output["log_words"] = np.log1p(output["word_count"])
    output["cttr"] = text.map(cttr)
    output["smog_de"] = text.map(smog_de)
    output["url_present"] = text.str.contains(URL_RE).astype(int)
    return output


def validate_local_text_features(
    frame: pd.DataFrame, expected: pd.DataFrame
) -> pd.DataFrame:
    """Validate a local-text checkpoint and restore the expected key order."""
    required_columns = [
        "story_id",
        "comment_id",
        "effective_text_hash",
        *LOCAL_TEXT_COLUMNS,
    ]
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Local-text checkpoint is missing columns: {sorted(missing)}")
    expected_required = {"story_id", "comment_id", "effective_text"}
    missing_expected = expected_required - set(expected.columns)
    if missing_expected:
        raise ValueError(
            f"Local-text validation input is missing columns: {sorted(missing_expected)}"
        )
    actual = frame[required_columns].copy()
    wanted = expected[["story_id", "comment_id", "effective_text"]].copy()
    for candidate in (actual, wanted):
        candidate["story_id"] = candidate["story_id"].astype(str)
        candidate["comment_id"] = candidate["comment_id"].astype(str)
    if actual.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("Local-text checkpoint contains duplicate keys")
    if wanted.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("Local-text validation input contains duplicate keys")
    wanted["effective_text_hash"] = wanted["effective_text"].astype(str).map(text_hash)
    wanted = wanted.drop(columns="effective_text")
    joined = wanted.merge(
        actual,
        on=["story_id", "comment_id"],
        how="outer",
        suffixes=("_expected", "_actual"),
        indicator=True,
        validate="one_to_one",
        sort=False,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            "Local-text checkpoint key coverage mismatch: "
            f"{joined['_merge'].value_counts().to_dict()}"
        )
    if not joined["effective_text_hash_expected"].eq(
        joined["effective_text_hash_actual"]
    ).all():
        raise ValueError("Local-text checkpoint effective_text_hash mismatch")
    if joined[LOCAL_TEXT_COLUMNS].isna().any().any():
        raise ValueError("Local-text checkpoint contains null feature values")
    if (joined["word_count"] < 0).any():
        raise ValueError("Local-text checkpoint contains a negative word count")
    if not np.allclose(
        joined["log_words"].to_numpy(float),
        np.log1p(joined["word_count"].to_numpy(float)),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("Local-text log_words does not recompute")
    if not joined["url_present"].isin((0, 1)).all():
        raise ValueError("Local-text URL indicator is not binary")
    if not np.isfinite(joined[["cttr", "smog_de"]].to_numpy(float)).all():
        raise ValueError("Local-text checkpoint contains non-finite values")
    return joined[["story_id", "comment_id", *LOCAL_TEXT_COLUMNS]].copy()


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


def _fit_length_adjuster(
    frame: pd.DataFrame, outcome: str
) -> tuple[Any, dict[str, Any]]:
    """Fit the existing length adjustment without retaining its residual array."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    valid = np.isfinite(frame[outcome]) & np.isfinite(frame["log_words"])
    if valid.sum() < 20:
        raise ValueError(f"Too few observations to length-adjust {outcome}")
    model = make_pipeline(
        SplineTransformer(n_knots=6, degree=3, include_bias=False), Ridge(alpha=1.0)
    )
    model.fit(
        frame.loc[valid, ["log_words"]].to_numpy(),
        frame.loc[valid, outcome].to_numpy(),
    )
    return model, {"outcome": outcome, "n_knots": 6, "degree": 3, "ridge_alpha": 1.0}


def _apply_length_adjuster(
    frame: pd.DataFrame, outcome: str, model: Any
) -> np.ndarray:
    valid = np.isfinite(frame[outcome]) & np.isfinite(frame["log_words"])
    predicted = np.full(len(frame), np.nan)
    if valid.any():
        predicted[valid] = model.predict(
            frame.loc[valid, ["log_words"]].to_numpy()
        )
    residual = frame[outcome].to_numpy(dtype=float) - predicted
    return residual


def _feature_registry(*, aqua_available: bool = False) -> dict[str, Any]:
    from aqua_runtime.schema import AQUA_FEATURES, expected_alias_column, label_column

    labels = {
        "log_words": "Comment length (log words)",
        "sentiment_positive": "Positive sentiment",
        "sentiment_negative": "Negative sentiment",
        "toxicity_probability": "Maximum toxicity probability",
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
    aqua_features: dict[str, dict[str, Any]] = {}
    for feature in AQUA_FEATURES:
        aqua_features[label_column(feature.stem)] = {
            "label": f"AQuA {feature.description} (hard ordinal label)",
            "standardize": False,
            "deferred": not aqua_available,
            "descriptive_only": True,
            "scale": "0..3",
        }
        aqua_features[expected_alias_column(feature.stem)] = {
            "label": f"AQuA {feature.description} (raw expected ordinal score)",
            "standardize": True,
            "deferred": not aqua_available,
            "descriptive_only": True,
            "scale": "0..3",
            "probability_status": "uncalibrated",
        }
    aqua_features.update(
        {
            "aqua_score_hard": {
                "label": "Published hard-label AQuA composite score",
                "standardize": True,
                "deferred": not aqua_available,
                "descriptive_only": True,
                "scale": "0..5",
            },
            "aqua_score_expected": {
                "label": "Raw expected AQuA composite score",
                "standardize": True,
                "deferred": not aqua_available,
                "descriptive_only": True,
                "scale": "0..5",
                "probability_status": "uncalibrated",
            },
        }
    )
    return {
        "version": 3,
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
                "engagement_probability": "Engaging-comment probability",
                "fact_claim_probability": "Fact-claim probability",
            }.items()
        }
        | {
            "toxicity_mean_probability": {
                "label": "Mean toxicity probability across chunks",
                "standardize": False,
                "deferred": False,
                "descriptive_only": True,
            }
        }
        | aqua_features,
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


def _prepare_model_columns_with_novelty_mean(
    frame: pd.DataFrame,
    scope: str,
    novelty_fill_value: float,
) -> pd.DataFrame:
    """Prepare one bounded story frame using the previously computed global mean."""
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
    if not np.isfinite(novelty_fill_value):
        raise ValueError(f"No finite values for {novelty}")
    output[f"{novelty}_model"] = output[novelty].fillna(novelty_fill_value)
    return output


def _finalize_choice_set(
    output: pd.DataFrame,
    *,
    scope: str,
    config: FeatureBuildConfig,
    novelty_fill_value: float | None = None,
    invalid_sticky_stories_excluded: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create model columns and labels after story eligibility is established."""
    output = (
        _prepare_model_columns(output, scope)
        if novelty_fill_value is None
        else _prepare_model_columns_with_novelty_mean(
            output, scope, novelty_fill_value
        )
    )
    output["curator_selected"] = output["is_sticky"].astype(bool)
    output, ties = assign_audience_labels(
        output, draws=config.tie_draws, seed=config.seed
    )
    features_used = ROOT_MODEL_FEATURES if scope == "root" else ALL_MODEL_FEATURES
    missingness = output[features_used].isna().sum()
    if int(missingness.sum()):
        raise ValueError(
            f"Missing model features in {scope}: "
            f"{missingness[missingness > 0].to_dict()}"
        )
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
    ] + [
        f"audience_selected_draw_{draw:02d}"
        for draw in range(1, config.tie_draws + 1)
    ]
    raw_descriptive = [
        "word_count",
        "cttr",
        "smog_de",
        "sentiment_neutral",
        "toxicity_mean_probability",
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
    if "aqua_score_hard" in output.columns:
        from aqua_runtime.schema import downstream_feature_columns

        raw_descriptive.extend(downstream_feature_columns())
    keep += [name for name in raw_descriptive + features_used if name not in keep]
    summary = {
        "scope": scope,
        "candidate_rows": len(output),
        "eligible_stories": int(output["story_id"].nunique()),
        "sticky_comments": int(output["curator_selected"].sum()),
        "invalid_sticky_stories_excluded": invalid_sticky_stories_excluded,
        "ambiguous_vote_cutoffs": int(ties["ambiguous_cutoff"].sum()),
    }
    return output[keep].sort_values(["story_id", "comment_id"]), ties, summary


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
    return _finalize_choice_set(
        output,
        scope=scope,
        config=config,
        invalid_sticky_stories_excluded=len(invalid_sticky_stories),
    )


class _BufferedParquetSink:
    """Write one atomic Parquet file without retaining the complete table."""

    def __init__(self, destination: Path, *, batch_rows: int) -> None:
        self.destination = Path(destination)
        self.temporary = self.destination.with_suffix(self.destination.suffix + ".tmp")
        self.batch_rows = batch_rows
        self.frames: list[pd.DataFrame] = []
        self.rows = 0
        self.total_rows = 0
        self.writer: Any | None = None
        self.schema: Any | None = None

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.frames.append(frame)
        self.rows += len(frame)
        self.total_rows += len(frame)
        if self.rows >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.frames:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        combined = pd.concat(self.frames, ignore_index=True)
        self.frames.clear()
        self.rows = 0
        table = pa.Table.from_pandas(combined, preserve_index=False)
        table = table.replace_schema_metadata()
        del combined
        if self.writer is None:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.temporary,
                self.schema,
                compression="zstd",
            )
        elif not table.schema.equals(self.schema, check_metadata=False):
            table = table.cast(self.schema)
        self.writer.write_table(table)

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            raise ValueError(f"Refusing to write an empty Parquet file: {self.destination}")
        self.writer.close()
        self.writer = None

    def commit(self) -> None:
        os.replace(self.temporary, self.destination)

    def abort(self) -> None:
        self.frames.clear()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.temporary.unlink(missing_ok=True)


def _eligible_story_scope(
    frame: pd.DataFrame,
    *,
    scope: str,
    invalid_sticky_stories: set[str],
    config: FeatureBuildConfig,
) -> pd.DataFrame | None:
    """Apply the existing choice-set eligibility rules to one story."""
    if scope not in {"root", "all"}:
        raise ValueError(scope)
    story_ids = frame["story_id"].astype(str).unique()
    if len(story_ids) != 1:
        raise ValueError("A scalar story shard must contain exactly one story_id")
    story_id = str(story_ids[0])
    output = frame[frame["is_root"].astype(bool)].copy() if scope == "root" else frame.copy()
    if output.empty or story_id in invalid_sticky_stories:
        return None
    if config.require_page_publication_time:
        if not output["published_at_source"].eq("page").all():
            return None
        if output["invalid_posting_time"].astype(bool).any():
            return None
    if config.lookback_root is None and config.exclude_january_without_lookback:
        january = (
            output["article_year"].eq(config.year)
            & output["article_month"].eq(1)
        )
        if january.all():
            return None
    n_candidates = len(output)
    n_picks = int(output["is_sticky"].astype(bool).sum())
    if not 0 < n_picks < n_candidates:
        return None
    output["n_candidates"] = n_candidates
    output["n_picks"] = n_picks
    return output


def _invalid_sticky_story_sets(
    data_root: Path,
    year: int,
    included_story_ids: set[str],
) -> dict[str, set[str]]:
    """Scan only sticky raw comments, in batches, for invalid curator picks."""
    import pyarrow.dataset as ds

    root = Path(data_root) / "comments" / f"year={year}"
    dataset = ds.dataset(root, format="parquet", partitioning=None)
    columns = [
        "story_id",
        "comment_id",
        "is_root",
        "lifecycle_status",
        "effective_text",
        "created_at",
    ]
    scanner = dataset.scanner(
        columns=columns,
        filter=ds.field("is_sticky") == True,  # noqa: E712 - PyArrow expression
        batch_size=65_536,
    )
    invalid_all: set[str] = set()
    invalid_root: set[str] = set()
    for batch in scanner.to_batches():
        sticky = batch.to_pandas()
        sticky["story_id"] = sticky["story_id"].astype(str)
        sticky = sticky[sticky["story_id"].isin(included_story_ids)]
        if sticky.empty:
            continue
        created_at = _as_utc(sticky["created_at"])
        invalid = sticky[
            ~sticky["lifecycle_status"].eq("Published")
            | sticky["effective_text"].fillna("").str.strip().eq("")
            | created_at.isna()
            | sticky["comment_id"].fillna("").astype(str).str.strip().eq("")
        ]
        invalid_ids = set(invalid["story_id"].astype(str))
        invalid_all.update(invalid_ids)
        invalid_root.update(
            invalid.loc[invalid["is_root"].astype(bool), "story_id"].astype(str)
        )
    return {"root": invalid_root, "all": invalid_all}


def _invalid_sticky_story_sets_from_frame(
    comments: pd.DataFrame,
) -> dict[str, set[str]]:
    """Derive invalid curator stories while the raw frame is already resident."""
    required = {
        "story_id",
        "comment_id",
        "is_root",
        "is_sticky",
        "lifecycle_status",
        "effective_text",
        "created_at",
    }
    missing = required - set(comments.columns)
    if missing:
        raise ValueError(f"Missing invalid-sticky columns: {sorted(missing)}")
    sticky = comments.loc[
        comments["is_sticky"].fillna(False).astype(bool),
        list(required - {"is_sticky"}),
    ].copy()
    if sticky.empty:
        return {"root": set(), "all": set()}
    sticky["story_id"] = sticky["story_id"].astype(str)
    invalid = sticky[
        ~sticky["lifecycle_status"].eq("Published")
        | sticky["effective_text"].fillna("").str.strip().eq("")
        | _as_utc(sticky["created_at"]).isna()
        | sticky["comment_id"].fillna("").astype(str).str.strip().eq("")
    ]
    invalid_all = set(invalid["story_id"])
    invalid_root = set(
        invalid.loc[invalid["is_root"].astype(bool), "story_id"]
    )
    return {"root": invalid_root, "all": invalid_all}


def _story_path_windows(
    paths: list[Path], window_stories: int = CHOICE_READ_WINDOW_STORIES
) -> Iterable[list[Path]]:
    for start in range(0, len(paths), window_stories):
        yield paths[start : start + window_stories]


def _read_scalar_window(
    paths: list[Path], columns: list[str] | None = None
) -> pd.DataFrame:
    """Read many story shards concurrently with one Arrow-to-pandas conversion."""
    import pyarrow.dataset as ds

    table = ds.dataset(
        [str(path) for path in paths],
        format="parquet",
        partitioning=None,
    ).to_table(columns=columns, use_threads=True)
    frame = table.to_pandas(split_blocks=True)
    del table
    expected = {path.stem for path in paths}
    actual = set(frame["story_id"].astype(str).unique())
    if actual != expected:
        raise ValueError(
            "Scalar read-window story coverage mismatch: "
            f"missing={sorted(expected - actual)[:5]} "
            f"unexpected={sorted(actual - expected)[:5]}"
        )
    return frame


def _build_choice_sets_bounded(
    *,
    checkpoint_root: Path,
    data_root: Path,
    output_root: Path,
    config: FeatureBuildConfig,
    invalid_stories: dict[str, set[str]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Fit global adjustments and stream final choice sets by story."""
    import pyarrow.dataset as ds

    scalar_paths = sorted(
        Path(checkpoint_root).glob("month=*/*.parquet"),
        key=lambda path: (path.stem, path.parent.name),
    )
    if not scalar_paths:
        raise FileNotFoundError(f"No scalar story checkpoints under {checkpoint_root}")
    story_ids = [path.stem for path in scalar_paths]
    if len(story_ids) != len(set(story_ids)):
        raise ValueError("Duplicate story checkpoint names in scalar feature store")
    included_story_ids = set(story_ids)

    print(
        "Feature assembly: fitting length adjustments from three numeric columns | "
        f"stories={len(scalar_paths):,}",
        flush=True,
    )
    length_table = ds.dataset(
        checkpoint_root, format="parquet", partitioning=None
    ).to_table(columns=["log_words", "cttr", "smog_de"])
    length_frame = length_table.to_pandas(split_blocks=True)
    del length_table
    lex_model, lex_meta = _fit_length_adjuster(length_frame, "cttr")
    reading_model, reading_meta = _fit_length_adjuster(length_frame, "smog_de")
    length_rows = len(length_frame)
    del length_frame
    _release_unused_memory()
    print(
        "Feature assembly: length adjustments fitted | "
        f"rows={length_rows:,}",
        flush=True,
    )

    if invalid_stories is None:
        print("Feature assembly: scanning invalid sticky comments", flush=True)
        invalid_stories = _invalid_sticky_story_sets(
            data_root,
            config.year,
            included_story_ids,
        )
    else:
        invalid_stories = {
            scope: set(values) & included_story_ids
            for scope, values in invalid_stories.items()
        }
        print(
            "Feature assembly: reusing invalid-sticky results from source load | "
            f"root={len(invalid_stories['root']):,} "
            f"all={len(invalid_stories['all']):,}",
            flush=True,
        )
    compact_columns = [
        "story_id",
        "comment_id",
        "is_root",
        "is_sticky",
        "published_at_source",
        "invalid_posting_time",
        "article_year",
        "article_month",
        "novelty_prior_roots",
        "novelty_prior_all",
    ]
    eligible_ids: dict[str, set[str]] = {"root": set(), "all": set()}
    novelty_sums = {"root": 0.0, "all": 0.0}
    novelty_counts = {"root": 0, "all": 0}
    eligibility_started = time.monotonic()
    processed_stories = 0
    for window_paths in _story_path_windows(scalar_paths):
        compact_window = _read_scalar_window(window_paths, compact_columns)
        compact_groups = {
            str(story_id): story
            for story_id, story in compact_window.groupby("story_id", sort=False)
        }
        for path in window_paths:
            compact = compact_groups[path.stem]
            story_id = str(compact["story_id"].iloc[0])
            for scope, novelty_column in (
                ("root", "novelty_prior_roots"),
                ("all", "novelty_prior_all"),
            ):
                eligible = _eligible_story_scope(
                    compact,
                    scope=scope,
                    invalid_sticky_stories=invalid_stories[scope],
                    config=config,
                )
                if eligible is None:
                    continue
                eligible_ids[scope].add(story_id)
                values = eligible[novelty_column].to_numpy(dtype=float)
                finite = np.isfinite(values)
                novelty_sums[scope] += float(values[finite].sum())
                novelty_counts[scope] += int(finite.sum())
        processed_stories += len(window_paths)
        del compact_groups, compact_window
        _progress_line(
            "Choice eligibility",
            processed_stories,
            len(scalar_paths),
            eligibility_started,
            unit="stories",
            detail=(
                f"eligible(root/all)={len(eligible_ids['root']):,}/"
                f"{len(eligible_ids['all']):,}"
            ),
        )
    novelty_means: dict[str, float] = {}
    for scope in ("root", "all"):
        if novelty_counts[scope] == 0:
            novelty_name = (
                "novelty_prior_roots" if scope == "root" else "novelty_prior_all"
            )
            raise ValueError(f"No finite values for {novelty_name}")
        novelty_means[scope] = novelty_sums[scope] / novelty_counts[scope]

    sinks = {
        "root": _BufferedParquetSink(
            Path(output_root) / "choice_set_root.parquet",
            batch_rows=CHOICE_WRITE_BATCH_ROWS,
        ),
        "all": _BufferedParquetSink(
            Path(output_root) / "choice_set_all.parquet",
            batch_rows=CHOICE_WRITE_BATCH_ROWS,
        ),
        "root_ties": _BufferedParquetSink(
            Path(output_root) / "tie_diagnostics_root.parquet",
            batch_rows=CHOICE_WRITE_BATCH_ROWS,
        ),
        "all_ties": _BufferedParquetSink(
            Path(output_root) / "tie_diagnostics_all.parquet",
            batch_rows=CHOICE_WRITE_BATCH_ROWS,
        ),
    }
    summaries = {
        scope: {
            "scope": scope,
            "candidate_rows": 0,
            "eligible_stories": 0,
            "sticky_comments": 0,
            "invalid_sticky_stories_excluded": len(invalid_stories[scope]),
            "ambiguous_vote_cutoffs": 0,
        }
        for scope in ("root", "all")
    }
    writing_started = time.monotonic()
    try:
        processed_stories = 0
        for window_paths in _story_path_windows(scalar_paths):
            story_window = _read_scalar_window(window_paths)
            story_window["lexdiv_length_adjusted"] = _apply_length_adjuster(
                story_window, "cttr", lex_model
            )
            story_window["reading_level_length_adjusted"] = _apply_length_adjuster(
                story_window, "smog_de", reading_model
            )
            story_groups = {
                str(story_id): story
                for story_id, story in story_window.groupby("story_id", sort=False)
            }
            eligible_frames: dict[str, list[pd.DataFrame]] = {
                "root": [],
                "all": [],
            }
            for path in window_paths:
                story = story_groups[path.stem]
                story_id = str(story["story_id"].iloc[0])
                for scope in ("root", "all"):
                    if story_id not in eligible_ids[scope]:
                        continue
                    eligible = _eligible_story_scope(
                        story,
                        scope=scope,
                        invalid_sticky_stories=invalid_stories[scope],
                        config=config,
                    )
                    if eligible is None:
                        raise RuntimeError(
                            "Choice eligibility changed between passes for "
                            f"{scope} {story_id}"
                        )
                    eligible_frames[scope].append(eligible)
            for scope in ("root", "all"):
                if not eligible_frames[scope]:
                    continue
                eligible_window = pd.concat(
                    eligible_frames[scope], ignore_index=True
                )
                output, ties, window_summary = _finalize_choice_set(
                    eligible_window,
                    scope=scope,
                    config=config,
                    novelty_fill_value=novelty_means[scope],
                )
                sinks[scope].append(output)
                sinks[f"{scope}_ties"].append(
                    ties.assign(candidate_scope=scope)
                )
                for field in (
                    "candidate_rows",
                    "eligible_stories",
                    "sticky_comments",
                    "ambiguous_vote_cutoffs",
                ):
                    summaries[scope][field] += window_summary[field]
            processed_stories += len(window_paths)
            del eligible_frames, story_groups, story_window
            _progress_line(
                "Choice-set writing",
                processed_stories,
                len(scalar_paths),
                writing_started,
                unit="stories",
                detail=(
                    f"rows(root/all)={summaries['root']['candidate_rows']:,}/"
                    f"{summaries['all']['candidate_rows']:,}"
                ),
            )
        for sink in sinks.values():
            sink.close()
        for sink in sinks.values():
            sink.commit()
    except BaseException:
        for sink in sinks.values():
            sink.abort()
        raise
    return summaries["root"], summaries["all"], [lex_meta, reading_meta]


def _load_sentiment_encoder(config: FeatureBuildConfig) -> SentimentEncoder:
    if config.nlp_mode == "pilot":
        return PilotLexiconSentiment()
    return XLMTwitterSentimentEncoder(
        model_id=config.sentiment_model_id,
        device=config.device,
        revision=config.sentiment_revision,
    )


def _load_toxicity_encoder(config: FeatureBuildConfig) -> ToxicityEncoder:
    if config.nlp_mode == "pilot":
        return PilotLexiconToxicity()
    return TextDetoxToxicityEncoder(
        model_id=config.toxicity_model_id,
        device=config.device,
        revision=config.toxicity_revision,
    )


def build_analysis_features(
    config: FeatureBuildConfig,
    *,
    sentiment_encoder: SentimentEncoder | None = None,
    toxicity_encoder: ToxicityEncoder | None = None,
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
    aqua_build_root = None
    aqua_manifest = None
    if config.aqua_store is not None:
        from .aqua import resolve_aqua_store

        aqua_build_root, aqua_manifest = resolve_aqua_store(
            config.aqua_store,
            year=config.year,
            source_fingerprint=fingerprint,
            require_production=config.inference_mode,
        )
    lookback_fingerprint = (
        dataset_fingerprint(config.lookback_root, config.year - 1)
        if config.lookback_root is not None
        else None
    )
    history_identity = {
        "schema_version": 2,
        "year": config.year,
        "target_dataset_fingerprint": fingerprint,
        "lookback_dataset_fingerprint": lookback_fingerprint,
        "max_stories": config.max_stories,
        "discussion_semantics": "strict-prior timestamp batches; one-hour inclusive boundary",
        "author_semantics": "strict-prior 30-day window; focal article excluded; snapshot votes",
        "candidate_filter": "Published, non-empty effective_text, non-missing created_at",
        "implementation_signature": _functions_implementation_signature(
            compute_discussion_history,
            compute_author_history,
        ),
    }
    history_signature = _identity_signature(history_identity)
    local_text_identity = _local_text_identity(config, fingerprint)
    local_text_signature = _identity_signature(local_text_identity)

    # The final assembly identity composes independent upstream identities. It
    # intentionally excludes batch size, device and progress frequency because
    # those alter execution rather than feature values.
    build_identity = {
        "schema_version": 2,
        "year": config.year,
        "target_dataset_fingerprint": fingerprint,
        "history_signature": history_signature,
        "local_text_signature": local_text_signature,
        "similarity_build_signature": similarity_manifest["build_signature"],
        "aqua_build_signature": (
            aqua_manifest["build_signature"] if aqua_manifest is not None else None
        ),
        "sentiment": {
            "model_id": config.sentiment_model_id,
            "revision": config.sentiment_revision,
            "aggregation": "token_weighted_chunk_mean",
        },
        "toxicity": {
            "model_id": config.toxicity_model_id,
            "revision": config.toxicity_revision,
            "aggregation": "maximum_and_token_weighted_chunk_mean",
        },
        "choice_sets": {
            "tie_draws": config.tie_draws,
            "seed": config.seed,
            "require_page_publication_time": config.require_page_publication_time,
            "exclude_january_without_lookback": config.exclude_january_without_lookback,
        },
        "inference_mode": config.inference_mode,
        "nlp_mode": config.nlp_mode,
        "max_stories": config.max_stories,
        "local_feature_implementation_signature": _functions_implementation_signature(
            word_count,
            cttr,
            smog_de,
            vienna_period,
            _length_residual,
            _prepare_model_columns,
            assign_audience_labels,
        ),
    }
    build_signature = _identity_signature(build_identity)
    state_path = (
        config.output_root
        / "build_states"
        / f"build={build_signature[:12]}.json"
    )
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("dataset_fingerprint") != fingerprint and not config.overwrite:
            raise RuntimeError("The matching feature-build state has a different dataset fingerprint")
    watermark = "INFERENCE" if config.inference_mode else "PILOT_NOT_FOR_INFERENCE"
    state = {
        "status": "running",
        "dataset_fingerprint": fingerprint,
        "build_signature": build_signature,
        "build_identity": build_identity,
        "history_signature": history_signature,
        "history_identity": history_identity,
        "local_text_signature": local_text_signature,
        "local_text_identity": local_text_identity,
        "watermark": watermark,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    }
    legacy_state_path = config.output_root / "build_state.json"
    legacy_state = (
        json.loads(legacy_state_path.read_text())
        if legacy_state_path.exists()
        else None
    )
    _atomic_json(state, state_path)
    _atomic_json(state, legacy_state_path)

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
    source_article_count = len(articles)
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
    invalid_sticky_stories = _invalid_sticky_story_sets_from_frame(target_raw)
    print(
        "Feature preflight: invalid sticky stories identified during source load | "
        f"root={len(invalid_sticky_stories['root']):,} "
        f"all={len(invalid_sticky_stories['all']):,}",
        flush=True,
    )

    history_checkpoint_root = (
        config.output_root
        / "history_checkpoints"
        / f"build={history_signature[:12]}-{fingerprint[:8]}"
    )
    # One-time compatibility bridge for checkpoints produced before cache
    # identities were split by feature family.
    if legacy_state and not history_checkpoint_root.exists():
        legacy_config = legacy_state.get("config", {})
        legacy_signature = str(legacy_state.get("build_signature", ""))
        same_history_inputs = (
            legacy_state.get("dataset_fingerprint") == fingerprint
            and legacy_config.get("year") == config.year
            and legacy_config.get("max_stories") == config.max_stories
            and str(legacy_config.get("lookback_root"))
            == str(config.lookback_root)
        )
        legacy_history_root = (
            config.output_root
            / "history_checkpoints"
            / f"build={legacy_signature[:8]}-{fingerprint[:8]}"
        )
        if (
            same_history_inputs
            and (legacy_history_root / "discussion_history.parquet").exists()
            and (legacy_history_root / "author_history.parquet").exists()
        ):
            history_checkpoint_root = legacy_history_root
            print(
                "Feature stage: adopting compatible legacy history checkpoints | "
                f"path={history_checkpoint_root}",
                flush=True,
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
    print("Feature stage: validating and attaching AQuA features", flush=True)
    stage_started = time.monotonic()
    if config.aqua_store is not None:
        from .aqua import load_aqua_for_candidates

        aqua_features, aqua_manifest = load_aqua_for_candidates(
            config.aqua_store,
            base[["story_id", "comment_id", "effective_text"]],
            year=config.year,
            source_fingerprint=fingerprint,
            require_production=config.inference_mode,
        )
        before = len(base)
        base = base.merge(
            aqua_features,
            on=["story_id", "comment_id"],
            how="left",
            validate="one_to_one",
        )
        if len(base) != before or base["aqua_runtime_status"].isna().any():
            raise ValueError("Incomplete AQuA feature merge")
    print(
        "Feature stage complete: AQuA feature attachment | "
        f"elapsed={_format_duration(time.monotonic() - stage_started)}",
        flush=True,
    )

    if sentiment_encoder is None:
        sentiment_encoder = _load_sentiment_encoder(config)
    if config.inference_mode and "PILOT_ONLY" in sentiment_encoder.model_id:
        raise ValueError("Pilot sentiment adapters cannot be used for inference")
    if toxicity_encoder is None:
        toxicity_encoder = _load_toxicity_encoder(config)
    if config.inference_mode and "PILOT_ONLY" in toxicity_encoder.model_id:
        raise ValueError("Pilot toxicity adapters cannot be used for inference")
    print(
        "Feature NLP preflight: "
        f"sentiment_model={sentiment_encoder.model_id} "
        f"sentiment_revision={getattr(sentiment_encoder, 'resolved_revision', 'unresolved')} "
        f"toxicity_model={toxicity_encoder.model_id} "
        f"toxicity_revision={getattr(toxicity_encoder, 'resolved_revision', 'unresolved')} "
        f"device={getattr(sentiment_encoder, 'device', select_torch_device(config.device))} "
        f"sentiment_batch={config.sentiment_batch_size} "
        f"toxicity_batch={config.toxicity_batch_size}",
        flush=True,
    )

    sentiment_identity = {
        "schema_version": 1,
        "year": config.year,
        "dataset_fingerprint": fingerprint,
        "candidate_filter": "Published, non-empty effective_text, non-missing created_at",
        "model_id": sentiment_encoder.model_id,
        "resolved_revision": getattr(sentiment_encoder, "resolved_revision", "unresolved"),
        "adapter_implementation_signature": _adapter_implementation_signature(
            sentiment_encoder
        ),
        "aggregation": "token_weighted_chunk_mean",
        "chunk_tokens": getattr(sentiment_encoder, "chunk_tokens", None),
        "columns": ["sentiment_positive", "sentiment_negative", "sentiment_neutral"],
        "max_stories": config.max_stories,
    }
    toxicity_identity = {
        "schema_version": 1,
        "year": config.year,
        "dataset_fingerprint": fingerprint,
        "candidate_filter": "Published, non-empty effective_text, non-missing created_at",
        "model_id": toxicity_encoder.model_id,
        "resolved_revision": getattr(toxicity_encoder, "resolved_revision", "unresolved"),
        "adapter_implementation_signature": _adapter_implementation_signature(
            toxicity_encoder
        ),
        "aggregation": "maximum_and_token_weighted_chunk_mean",
        "chunk_tokens": getattr(toxicity_encoder, "chunk_tokens", None),
        "columns": ["toxicity_probability", "toxicity_mean_probability"],
        "max_stories": config.max_stories,
    }
    sentiment_signature = _identity_signature(sentiment_identity)
    toxicity_signature = _identity_signature(toxicity_identity)
    build_identity["sentiment"]["feature_family_signature"] = sentiment_signature
    build_identity["toxicity"]["feature_family_signature"] = toxicity_signature
    build_signature = _identity_signature(build_identity)
    state["build_signature"] = build_signature
    state["build_identity"] = build_identity
    state_path = (
        config.output_root
        / "build_states"
        / f"build={build_signature[:12]}.json"
    )
    _atomic_json(state, state_path)
    _atomic_json(state, config.output_root / "build_state.json")
    sentiment_checkpoint_root = (
        config.output_root
        / "feature_families"
        / "sentiment"
        / f"build={sentiment_signature[:12]}-{fingerprint[:8]}"
        / f"year={config.year}"
    )
    toxicity_checkpoint_root = (
        config.output_root
        / "feature_families"
        / "toxicity"
        / f"build={toxicity_signature[:12]}-{fingerprint[:8]}"
        / f"year={config.year}"
    )
    local_text_checkpoint_root = (
        config.output_root
        / "feature_families"
        / "local_text"
        / f"build={local_text_signature[:12]}-{fingerprint[:8]}"
        / f"year={config.year}"
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
    sentiment_written = 0
    sentiment_reused = 0
    toxicity_written = 0
    toxicity_reused = 0
    local_text_written = 0
    local_text_reused = 0
    stage_started = time.monotonic()
    print(
        "Feature stage: local text, sentiment, toxicity, semantic joins, and "
        "scalar checkpoints | "
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
        family_key_columns = ["story_id", "comment_id"]
        local_text_path = (
            local_text_checkpoint_root / f"month={month:02d}" / f"{story_id}.parquet"
        )
        if local_text_path.exists() and not config.overwrite:
            local_text_frame = validate_local_text_features(
                pd.read_parquet(local_text_path), story
            )
            local_text_reused += 1
        else:
            local_text_output = compute_local_text_features(story)
            local_text_frame = validate_local_text_features(local_text_output, story)
            _atomic_parquet(local_text_output, local_text_path)
            del local_text_output
            local_text_written += 1
        before = len(story)
        story = story.merge(
            local_text_frame,
            on=family_key_columns,
            how="left",
            validate="one_to_one",
        )
        if len(story) != before or story[LOCAL_TEXT_COLUMNS].isna().any().any():
            raise ValueError(f"Incomplete local-text feature join for story {story_id}")
        del local_text_frame
        sentiment_columns = [
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_neutral",
        ]
        sentiment_path = (
            sentiment_checkpoint_root / f"month={month:02d}" / f"{story_id}.parquet"
        )
        if sentiment_path.exists() and not config.overwrite:
            sentiment_frame = pd.read_parquet(
                sentiment_path, columns=family_key_columns + sentiment_columns
            )
            sentiment_reused += 1
        else:
            sentiments = sentiment_encoder.predict(texts, config.sentiment_batch_size)
            if sentiments.shape != (len(story), 3):
                raise ValueError("Sentiment encoder returned an unexpected shape")
            sentiment_frame = story[family_key_columns].copy()
            sentiment_frame[sentiment_columns] = sentiments
            _atomic_parquet(sentiment_frame, sentiment_path)
            sentiment_written += 1

        toxicity_columns = ["toxicity_probability", "toxicity_mean_probability"]
        toxicity_path = (
            toxicity_checkpoint_root / f"month={month:02d}" / f"{story_id}.parquet"
        )
        if toxicity_path.exists() and not config.overwrite:
            toxicity_frame = pd.read_parquet(
                toxicity_path, columns=family_key_columns + toxicity_columns
            )
            toxicity_reused += 1
        else:
            toxicities = toxicity_encoder.predict(texts, config.toxicity_batch_size)
            if toxicities.shape != (len(story), 2):
                raise ValueError("Toxicity encoder returned an unexpected shape")
            toxicity_frame = story[family_key_columns].copy()
            toxicity_frame[toxicity_columns] = toxicities
            _atomic_parquet(toxicity_frame, toxicity_path)
            toxicity_written += 1

        for family_name, family_frame, family_columns in (
            ("sentiment", sentiment_frame, sentiment_columns),
            ("toxicity", toxicity_frame, toxicity_columns),
        ):
            if family_frame.duplicated(family_key_columns).any():
                raise ValueError(f"Duplicate {family_name} keys for story {story_id}")
            before = len(story)
            story = story.merge(
                family_frame,
                on=family_key_columns,
                how="left",
                validate="one_to_one",
            )
            if len(story) != before or story[family_columns].isna().any().any():
                raise ValueError(f"Incomplete {family_name} feature join for story {story_id}")
        del sentiment_frame, toxicity_frame
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
                    f"skipped={skipped_stories:,} "
                    f"local_text(new/reused)={local_text_written:,}/{local_text_reused:,} "
                    f"sentiment(new/reused)={sentiment_written:,}/{sentiment_reused:,} "
                    f"toxicity(new/reused)={toxicity_written:,}/{toxicity_reused:,}"
                ),
            )

    # The scalar checkpoints now own all candidate-level values needed below.
    # Releasing the full in-memory candidate frame before assembly prevents a
    # second 9M-row table from doubling peak ordinary-RAM usage.
    del grouped_stories, base
    story = None
    texts = None
    _release_unused_memory()

    local_text_manifest = {
        "status": "complete",
        "build_signature": local_text_signature,
        "identity": local_text_identity,
        "output_semantics_version": LOCAL_TEXT_OUTPUT_SEMANTICS_VERSION,
        "root": str(local_text_checkpoint_root),
        "files": total_stories,
        "rows": total_candidates,
        "new_story_checkpoints": local_text_written,
        "reused_story_checkpoints": local_text_reused,
    }
    sentiment_manifest = {
        "status": "complete",
        "build_signature": sentiment_signature,
        "identity": sentiment_identity,
        "output_semantics_version": getattr(
            sentiment_encoder, "output_semantics_version", None
        ),
        "root": str(sentiment_checkpoint_root),
        "files": total_stories,
        "rows": total_candidates,
        "new_story_checkpoints": sentiment_written,
        "reused_story_checkpoints": sentiment_reused,
        "sequence_diagnostics": (
            sentiment_encoder.sequence_diagnostics()
            if hasattr(sentiment_encoder, "sequence_diagnostics")
            else None
        ),
    }
    toxicity_manifest = {
        "status": "complete",
        "build_signature": toxicity_signature,
        "identity": toxicity_identity,
        "output_semantics_version": getattr(
            toxicity_encoder, "output_semantics_version", None
        ),
        "root": str(toxicity_checkpoint_root),
        "files": total_stories,
        "rows": total_candidates,
        "new_story_checkpoints": toxicity_written,
        "reused_story_checkpoints": toxicity_reused,
        "sequence_diagnostics": (
            toxicity_encoder.sequence_diagnostics()
            if hasattr(toxicity_encoder, "sequence_diagnostics")
            else None
        ),
    }
    _atomic_json(
        local_text_manifest, local_text_checkpoint_root.parent / "manifest.json"
    )
    _atomic_json(sentiment_manifest, sentiment_checkpoint_root.parent / "manifest.json")
    _atomic_json(toxicity_manifest, toxicity_checkpoint_root.parent / "manifest.json")
    state.setdefault("stages", {})["local_text"] = local_text_manifest
    state.setdefault("stages", {})["sentiment"] = sentiment_manifest
    state.setdefault("stages", {})["toxicity"] = toxicity_manifest
    _atomic_json(state, state_path)
    _atomic_json(state, config.output_root / "build_state.json")

    # Metadata needed below has already been frozen in the family manifests.
    # Free both transformer objects (and their CPU-side weights) before the
    # bounded-memory scalar pass.
    del sentiment_encoder, toxicity_encoder
    gc.collect()

    print("Feature stage: bounded-memory scalar assembly and choice sets", flush=True)
    stage_started = time.monotonic()
    root_summary, all_summary, length_adjustment = _build_choice_sets_bounded(
        checkpoint_root=checkpoint_root,
        data_root=config.data_root,
        output_root=config.output_root,
        config=config,
        invalid_stories=invalid_sticky_stories,
    )
    print(
        "Feature stage complete: choice sets | "
        f"root_rows={root_summary['candidate_rows']:,} "
        f"all_rows={all_summary['candidate_rows']:,} "
        f"elapsed={_format_duration(time.monotonic() - stage_started)}",
        flush=True,
    )
    registry = _feature_registry(aqua_available=aqua_manifest is not None)
    registry["length_adjustment"] = length_adjustment
    registry["local_text"] = {
        "feature_store": str(local_text_checkpoint_root),
        "build_signature": local_text_signature,
        "output_semantics_version": LOCAL_TEXT_OUTPUT_SEMANTICS_VERSION,
        "columns": LOCAL_TEXT_COLUMNS,
    }
    registry["nlp"] = {
        "sentiment_model": sentiment_identity["model_id"],
        "sentiment_revision": sentiment_identity["resolved_revision"],
        "sentiment_aggregation": "token_weighted_chunk_mean",
        "sentiment_feature_store": str(sentiment_checkpoint_root),
        "sentiment_build_signature": sentiment_signature,
        "toxicity_model": toxicity_identity["model_id"],
        "toxicity_revision": toxicity_identity["resolved_revision"],
        "toxicity_aggregation": {
            "toxicity_probability": "maximum_chunk_probability",
            "toxicity_mean_probability": "token_weighted_chunk_mean",
        },
        "toxicity_feature_store": str(toxicity_checkpoint_root),
        "toxicity_build_signature": toxicity_signature,
        "embedding_model": similarity_manifest["embedding_model"]["model_id"],
        "embedding_revision": similarity_manifest["embedding_model"]["resolved_revision"],
        "similarity_store": str(similarity_store),
        "similarity_build_signature": similarity_manifest["build_signature"],
        "watermark": watermark,
        "aqua": (
            {
                "store": str(aqua_build_root),
                "build_signature": aqua_manifest["build_signature"],
                "schema_version": aqua_manifest["schema_version"],
                "watermark": aqua_manifest["watermark"],
                "probability_status": aqua_manifest["probability_status"],
                "expected_alias_source": "*_expected_raw",
            }
            if aqua_manifest is not None
            else None
        ),
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
            "articles": source_article_count,
            "dataset_fingerprint": fingerprint,
            "similarity_store": str(similarity_store),
            "similarity_build_signature": similarity_manifest["build_signature"],
        },
        "root": root_summary,
        "all": all_summary,
        "models": registry["nlp"],
        "local_text": registry["local_text"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": select_torch_device(config.device),
            "packages": _analysis_package_versions(),
        },
        "execution": {
            "elapsed_seconds": time.monotonic() - build_started,
            "choice_assembly_strategy": "bounded_cross_story_windows",
            "choice_write_batch_rows": CHOICE_WRITE_BATCH_ROWS,
            "choice_read_window_stories": CHOICE_READ_WINDOW_STORIES,
            "length_fit_columns": ["log_words", "cttr", "smog_de"],
            "new_story_checkpoints": written_stories,
            "skipped_story_checkpoints": skipped_stories,
            "sentiment_story_checkpoints_written": sentiment_written,
            "sentiment_story_checkpoints_reused": sentiment_reused,
            "toxicity_story_checkpoints_written": toxicity_written,
            "toxicity_story_checkpoints_reused": toxicity_reused,
            "local_text_story_checkpoints_written": local_text_written,
            "local_text_story_checkpoints_reused": local_text_reused,
        },
    }
    _atomic_json(summary, config.output_root / "provenance_manifest.json")
    completed_state = {
        **state,
        "status": "complete",
        "manifest": str(config.output_root / "provenance_manifest.json"),
    }
    _atomic_json(completed_state, state_path)
    _atomic_json(completed_state, config.output_root / "build_state.json")
    print(
        "Feature build complete: "
        f"elapsed={_format_duration(time.monotonic() - build_started)} "
        f"output={config.output_root}",
        flush=True,
    )
    return summary
