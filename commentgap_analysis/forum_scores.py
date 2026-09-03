"""FORUM policy construction, scoring, and inference.

This module consumes frozen Paper 1 artifacts but owns a separate output
namespace. Development cross-validation selects predictive rankers; held-out
scores are opened only after that selection has been frozen.  Legacy FORUM
files are design references and are never read here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .factorial_winners import assert_factorial_idle, rank_development_cv_variants


FORUM_ANALYSIS_VERSION = 3
DEFAULT_SEED = 20260813
PRIMARY_MIN_COMMENTS = 11
SENSITIVITY_MIN_COMMENTS = 100
DEFAULT_TIE_DRAWS = 10
DEFAULT_RANDOM_DRAWS = 100
DEFAULT_BOOTSTRAP_DRAWS = 2_000


def _available_cpu_count() -> int:
    """Return CPUs available to this process, respecting CPU affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


DEFAULT_WORKERS = _available_cpu_count()

SUBSTANTIVE_ORDERINGS = (
    "relative_votes",
    "upvotes",
    "chronological",
    "reverse_chronological",
    "regression_audience",
    "regression_editor",
    "xgb_metadata_audience",
    "xgb_metadata_editor",
    "xgb_metadata_text_audience",
    "xgb_metadata_text_editor",
    "neural_metadata_audience",
    "neural_metadata_editor",
    "neural_metadata_text_audience",
    "neural_metadata_text_editor",
)
CONTROL_ORDERING = "random"
ALL_ORDERINGS = (*SUBSTANTIVE_ORDERINGS, CONTROL_ORDERING)
REPLY_MODES = ("loose", "trees", "hidden")

PRIMARY_OUTCOMES = (
    "aqua_score_expected",
    "article_similarity_top3",
    "toxicity_probability",
    "log_author_prior_30d_comments",
    "author_prior_30d_reception_balance",
    "semantic_novelty_knn5",
    "sentiment_positive",
    "sentiment_negative",
    "cttr",
    "smog_de",
)
SECONDARY_OUTCOMES: tuple[str, ...] = ()
ALL_OUTCOMES = (*PRIMARY_OUTCOMES, *SECONDARY_OUTCOMES)

# Raw feature columns used by the Paper 1 regression.  Interaction terms in
# the fitted model (for example ``log_words:curator``) are not separate
# comment-level outcomes and therefore are not included here.
REGRESSION_FEATURE_COLUMNS = (
    "log_words",
    "sentiment_positive",
    "sentiment_negative",
    "toxicity_probability",
    "lexdiv_length_adjusted",
    "reading_level_length_adjusted",
    "url_present",
    "article_similarity_top3",
    "novelty_prior_all_model",
    "log_hours_since_article",
    "prior_reply_composition",
    "discussion_pace",
    "vienna_overnight",
    "vienna_weekday_shoulder_evening",
    "vienna_weekend_day_evening",
    "log_author_prior_30d_comments",
    "author_prior_30d_upvote_reception",
    "author_prior_30d_downvote_reception",
    "log_author_prior_comments_story",
    "is_reply",
    "reply_depth_centered",
    "aqua_relevance_expected",
    "aqua_fact_expected",
    "aqua_opinion_expected",
    "aqua_justification_expected",
    "aqua_solution_proposal_expected",
    "aqua_additional_knowledge_expected",
    "aqua_question_expected",
    "aqua_referencing_users_expected",
    "aqua_referencing_medium_expected",
    "aqua_referencing_contents_expected",
    "aqua_referencing_personal_expected",
    "aqua_referencing_format_expected",
    "aqua_polite_address_expected",
    "aqua_respect_expected",
    "aqua_screaming_expected",
    "aqua_vulgarity_expected",
    "aqua_insult_expected",
    "aqua_sarcasm_expected",
    "aqua_discrimination_expected",
    "aqua_storytelling_expected",
)
# Notebook 10 scores only the explicitly selected FORUM outcomes.
FORUM_OUTCOMES = ALL_OUTCOMES
# Additional FORUM axes retained for the regression-feature diagnostic.
FORUM_ANALYSIS_OUTCOMES = tuple(dict.fromkeys((*FORUM_OUTCOMES, *REGRESSION_FEATURE_COLUMNS)))

STRUCTURAL_COLUMNS = (
    "story_id",
    "comment_id",
    "is_root",
    "root_comment_id",
    "parent_comment_id",
    "created_at",
    "preorder_position",
    "display_order",
    "root_order",
    "is_sticky",
    "votes_positive",
    "votes_negative",
)

CHOICE_COLUMNS = (
    "story_id",
    "comment_id",
    "article_month",
    "n_candidates",
    "curator_selected",
    "relative_votes",
    "sentiment_positive",
    "sentiment_negative",
    "article_similarity_top3",
    "lexdiv_length_adjusted",
    "reading_level_length_adjusted",
    "toxicity_probability",
    "aqua_score_expected",
    "cttr",
    "smog_de",
    "aqua_score_hard",
    *REGRESSION_FEATURE_COLUMNS,
)
CHOICE_COLUMNS = tuple(dict.fromkeys(CHOICE_COLUMNS))


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        temporary,
        compression="zstd",
    )
    os.replace(temporary, path)


def _stable_uint64(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _validate_unique_keys(frame: pd.DataFrame, label: str) -> None:
    _require_columns(frame, ("story_id", "comment_id"), label)
    if frame.duplicated(["story_id", "comment_id"]).any():
        examples = frame.loc[
            frame.duplicated(["story_id", "comment_id"], keep=False),
            ["story_id", "comment_id"],
        ].head().to_dict("records")
        raise ValueError(f"{label} has duplicate comment keys: {examples}")


@dataclass(frozen=True)
class SelectedRanker:
    family: str
    feature_set: str
    variant_id: str
    scope: str
    score_path: str
    audience_column: str
    editor_column: str


def freeze_ranker_handoff(
    *,
    factorial_root: Path = Path("model_output/selection_2025/factorial_rankers"),
    regression_scores_path: Path = Path(
        "model_output/selection_2025/regression/all/test_scores_wide.parquet"
    ),
    output_root: Path = Path(
        "model_output/selection_2025/forum_ranking_analysis/ranker_handoff"
    ),
    expected_folds: int = 5,
    draw_policies: Sequence[str] | None = ("draw1",),
    require_idle: bool = True,
) -> dict[str, Any]:
    """Freeze four all-comment predictive winners without reading test values.

    Only development-CV tables determine the selected variants.  Once the
    winner identities are fixed, this function checks that their score files
    and the regression score file exist and records their hashes; it does not
    open those Parquet files.  FORUM/ranking analysis uses the same ``draw1`` winner contract
    as notebook 08 by default; ``mean10`` may be requested explicitly for a
    sensitivity handoff.
    """
    factorial_root = Path(factorial_root)
    output_root = Path(output_root)
    if require_idle:
        assert_factorial_idle()
    development_path = factorial_root / "development_cv_results.csv"
    variants_path = factorial_root / "experiment_variants.csv"
    if not development_path.exists():
        raise FileNotFoundError(development_path)
    if not variants_path.exists():
        raise FileNotFoundError(variants_path)
    plan = pd.read_csv(variants_path)
    _require_columns(
        plan,
        ("variant_id", "family", "scope", "feature_set", "status"),
        "factorial experiment plan",
    )
    requested_draw_policies = (
        None if draw_policies is None else tuple(dict.fromkeys(draw_policies))
    )
    if requested_draw_policies is not None:
        if not requested_draw_policies or not set(requested_draw_policies).issubset(
            {"draw1", "mean10"}
        ):
            raise ValueError("draw_policies must contain draw1 and/or mean10")
        _require_columns(plan, ("draw_policy",), "factorial experiment plan")
    requested = plan[plan["scope"].eq("all")].copy()
    if requested_draw_policies is not None:
        requested = requested[
            requested["draw_policy"].isin(requested_draw_policies)
        ]
    if requested.empty or not requested["status"].eq("complete").all():
        incomplete = requested.loc[
            ~requested["status"].eq("complete"),
            ["variant_id", "status"],
        ].head().to_dict("records")
        raise RuntimeError(f"All FORUM candidate rankers must be complete: {incomplete}")

    development = pd.read_csv(development_path)
    ranking, winners, excluded = rank_development_cv_variants(
        development,
        scopes=("all",),
        expected_folds=expected_folds,
        draw_policies=requested_draw_policies,
    )
    if not excluded.empty:
        raise RuntimeError(
            "FORUM ranker selection found incomplete development folds: "
            f"{excluded.head().to_dict('records')}"
        )
    expected_cells = {
        (family, feature_set)
        for family in ("xgboost", "neural")
        for feature_set in ("metadata", "metadata_bge")
    }
    observed_cells = set(
        winners[["family", "feature_set"]].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells or len(winners) != 4:
        raise RuntimeError(
            "Expected one winner in each family x feature-set cell; "
            f"observed={sorted(observed_cells)}"
        )

    selected: list[SelectedRanker] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for row in winners.itertuples(index=False):
        score_path = factorial_root / str(row.variant_id) / "all" / "test_scores_wide.parquet"
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        feature_label = "metadata_text" if row.feature_set == "metadata_bge" else "metadata"
        prefix = f"{'xgb' if row.family == 'xgboost' else 'neural'}_{feature_label}"
        selected.append(
            SelectedRanker(
                family=str(row.family),
                feature_set=str(row.feature_set),
                variant_id=str(row.variant_id),
                scope="all",
                score_path=str(score_path),
                audience_column="audience_score",
                editor_column="curator_score",
            )
        )
        artifacts[str(score_path)] = {
            "sha256": _sha256(score_path),
            "role": f"selected_{prefix}_held_out_scores",
        }
    regression_scores_path = Path(regression_scores_path)
    if not regression_scores_path.exists():
        raise FileNotFoundError(regression_scores_path)
    artifacts[str(regression_scores_path)] = {
        "sha256": _sha256(regression_scores_path),
        "role": "stacked_regression_held_out_scores",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    ranking_path = output_root / "development_cv_rankings.csv"
    winners_path = output_root / "selected_rankers.csv"
    ranking.to_csv(ranking_path, index=False)
    pd.DataFrame([asdict(item) for item in selected]).to_csv(winners_path, index=False)
    manifest = {
        "version": FORUM_ANALYSIS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_source": "development_cv_results.csv only",
        "held_out_values_read_for_selection": False,
        "scope": "all",
        "draw_policies": (
            list(requested_draw_policies)
            if requested_draw_policies is not None
            else None
        ),
        "winner_dimensions": ["family", "feature_set"],
        "expected_folds": expected_folds,
        "selection_rule": [
            "descending mean macro nDCG@k",
            "ascending fold SD",
            "descending minimum-fold macro nDCG@k",
            "lexicographic variant_id",
        ],
        "inputs": {
            str(development_path): _sha256(development_path),
            str(variants_path): _sha256(variants_path),
        },
        "selected_rankers": [asdict(item) for item in selected],
        "held_out_artifacts": artifacts,
        "outputs": {
            str(ranking_path): _sha256(ranking_path),
            str(winners_path): _sha256(winners_path),
        },
    }
    _atomic_json(manifest, output_root / "ranker_handoff_manifest.json")
    return manifest


def assemble_analysis_comments(
    choice_set: pd.DataFrame,
    raw_comments: pd.DataFrame,
    article_split: pd.DataFrame,
    regression_scores: pd.DataFrame,
    selected_scores: Mapping[str, tuple[pd.DataFrame, str, str]],
    *,
    novelty: pd.DataFrame | None = None,
    min_comments: int = PRIMARY_MIN_COMMENTS,
) -> pd.DataFrame:
    """Join the exact held-out Paper 1 universe to raw structure and scores.

    ``selected_scores`` maps a prefix such as ``xgb_metadata`` to a tuple of
    (frame, audience source column, editor source column).
    """
    _require_columns(choice_set, CHOICE_COLUMNS, "all-comment choice set")
    _require_columns(raw_comments, STRUCTURAL_COLUMNS, "raw comments")
    _require_columns(article_split, ("story_id", "split_role"), "article split")
    _validate_unique_keys(choice_set, "all-comment choice set")
    _validate_unique_keys(raw_comments, "raw comments")
    if min_comments < 2:
        raise ValueError("min_comments must be at least 2")

    split = article_split[["story_id", "split_role"]].copy()
    split["story_id"] = split["story_id"].astype(str)
    if split["story_id"].duplicated().any():
        raise ValueError("article split has duplicate story IDs")
    held_out = set(split.loc[split["split_role"].eq("paper2_test"), "story_id"])
    choice = choice_set.copy()
    choice["story_id"] = choice["story_id"].astype(str)
    choice["comment_id"] = choice["comment_id"].astype(str)
    choice = choice[
        choice["story_id"].isin(held_out)
        & (pd.to_numeric(choice["n_candidates"], errors="raise") >= min_comments)
    ].copy()
    if choice.empty:
        raise ValueError("No held-out discussions satisfy the FORUM size rule")
    expected_keys = pd.MultiIndex.from_frame(choice[["story_id", "comment_id"]])

    raw = raw_comments.copy()
    raw["story_id"] = raw["story_id"].astype(str)
    raw["comment_id"] = raw["comment_id"].astype(str)
    raw = raw[raw["story_id"].isin(set(choice["story_id"]))]
    merged = choice.merge(
        raw[list(STRUCTURAL_COLUMNS)],
        on=["story_id", "comment_id"],
        how="left",
        validate="one_to_one",
    )
    if merged[list(set(STRUCTURAL_COLUMNS) - {"story_id", "comment_id", "parent_comment_id"})].isna().any().any():
        raise ValueError("Raw structural fields do not cover the held-out choice set")
    if not np.array_equal(
        merged["curator_selected"].astype(bool).to_numpy(),
        merged["is_sticky"].astype(bool).to_numpy(),
    ):
        raise ValueError("Paper 1 curator labels disagree with raw sticky status")
    computed_relative = (
        pd.to_numeric(merged["votes_positive"], errors="raise")
        - pd.to_numeric(merged["votes_negative"], errors="raise")
    )
    if not np.array_equal(computed_relative.to_numpy(), merged["relative_votes"].to_numpy()):
        raise ValueError("Paper 1 relative votes disagree with raw vote counts")

    _validate_unique_keys(regression_scores, "regression held-out scores")
    _require_columns(
        regression_scores,
        ("regression_audience_score", "regression_curator_score"),
        "regression held-out scores",
    )
    reg = regression_scores.rename(
        columns={"regression_curator_score": "regression_editor_score"}
    ).copy()
    reg["story_id"] = reg["story_id"].astype(str)
    reg["comment_id"] = reg["comment_id"].astype(str)
    merged = merged.merge(
        reg[
            ["story_id", "comment_id", "regression_audience_score", "regression_editor_score"]
        ],
        on=["story_id", "comment_id"],
        how="left",
        validate="one_to_one",
    )

    for prefix, (score_frame, audience_source, editor_source) in selected_scores.items():
        _validate_unique_keys(score_frame, f"{prefix} held-out scores")
        _require_columns(
            score_frame,
            (audience_source, editor_source),
            f"{prefix} held-out scores",
        )
        scores = score_frame[["story_id", "comment_id", audience_source, editor_source]].copy()
        scores["story_id"] = scores["story_id"].astype(str)
        scores["comment_id"] = scores["comment_id"].astype(str)
        scores = scores.rename(
            columns={
                audience_source: f"{prefix}_audience_score",
                editor_source: f"{prefix}_editor_score",
            }
        )
        merged = merged.merge(
            scores,
            on=["story_id", "comment_id"],
            how="left",
            validate="one_to_one",
        )

    if novelty is not None:
        _validate_unique_keys(novelty, "semantic novelty")
        _require_columns(novelty, ("semantic_novelty_knn5",), "semantic novelty")
        novelty_frame = novelty[["story_id", "comment_id", "semantic_novelty_knn5"]].copy()
        novelty_frame["story_id"] = novelty_frame["story_id"].astype(str)
        novelty_frame["comment_id"] = novelty_frame["comment_id"].astype(str)
        merged = merged.merge(
            novelty_frame,
            on=["story_id", "comment_id"],
            how="left",
            validate="one_to_one",
        )

    merged["author_prior_30d_reception_balance"] = (
        pd.to_numeric(
            merged["author_prior_30d_upvote_reception"], errors="raise"
        )
        - pd.to_numeric(
            merged["author_prior_30d_downvote_reception"], errors="raise"
        )
    )
    merged["primary_sample"] = True
    merged["large_thread_sensitivity"] = merged["n_candidates"].ge(
        SENSITIVITY_MIN_COMMENTS
    )
    required_scores = [f"{name}_score" for name in (
        "regression_audience",
        "regression_editor",
        "xgb_metadata_audience",
        "xgb_metadata_editor",
        "xgb_metadata_text_audience",
        "xgb_metadata_text_editor",
        "neural_metadata_audience",
        "neural_metadata_editor",
        "neural_metadata_text_audience",
        "neural_metadata_text_editor",
    )]
    required_outcomes = [name for name in FORUM_OUTCOMES if name in merged.columns]
    missing_values = merged[required_scores + required_outcomes].isna().sum()
    if int(missing_values.sum()):
        raise ValueError(
            "FORUM handoff has missing scores/outcomes: "
            f"{missing_values[missing_values > 0].to_dict()}"
        )
    observed_keys = pd.MultiIndex.from_frame(merged[["story_id", "comment_id"]])
    if not observed_keys.equals(expected_keys):
        raise ValueError("FORUM handoff changed held-out comment key order or coverage")
    return merged.sort_values(["story_id", "display_order", "comment_id"]).reset_index(drop=True)


def static_knn_novelty(
    vectors: np.ndarray,
    *,
    neighbors: int = 5,
    block_size: int = 512,
    approximate_threshold: int = 5_000,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return mean distance to the nearest other comments in a discussion."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Semantic novelty requires at least two embedding vectors")
    if not np.isfinite(values).all():
        raise ValueError("Embedding vectors must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embedding vectors must have positive norm")
    values = values / norms
    n_rows, dimension = values.shape
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    effective_neighbors = min(neighbors, n_rows - 1)
    if block_size < 1:
        raise ValueError("block_size must be positive")

    if n_rows > approximate_threshold:
        try:
            import hnswlib
        except ImportError as error:
            raise RuntimeError(
                "hnswlib is required for large-discussion semantic novelty"
            ) from error
        index = hnswlib.Index(space="cosine", dim=dimension)
        index.init_index(
            max_elements=n_rows,
            ef_construction=200,
            M=32,
            random_seed=seed,
        )
        labels = np.arange(n_rows, dtype=np.int64)
        index.add_items(values, labels)
        index.set_ef(min(n_rows, 200))
        neighbours, distances = index.knn_query(
            values, k=min(effective_neighbors + 1, n_rows)
        )
        novelty = np.empty(n_rows, dtype=float)
        for row in range(n_rows):
            candidates = distances[row, neighbours[row] != row]
            if len(candidates) < effective_neighbors:
                raise RuntimeError("Approximate novelty returned too few neighbours")
            novelty[row] = float(candidates[:effective_neighbors].mean())
        sample = np.linspace(0, n_rows - 1, min(100, n_rows), dtype=int)
        exact = np.empty(len(sample), dtype=float)
        for offset, row in enumerate(sample):
            similarities = values[row] @ values.T
            similarities[row] = -np.inf
            nearest = np.partition(similarities, -effective_neighbors)[
                -effective_neighbors:
            ]
            exact[offset] = float(np.mean(1.0 - nearest))
        maximum_error = float(np.max(np.abs(exact - novelty[sample])))
        recall = float(np.mean(np.abs(exact - novelty[sample]) <= 1e-4))
        if recall < 0.95:
            raise ValueError(
                f"Approximate semantic novelty recall {recall:.3f} is below 0.95"
            )
        return novelty, {
            "method": "hnsw_cosine",
            "rows": n_rows,
            "dimension": dimension,
            "requested_neighbors": neighbors,
            "effective_neighbors": effective_neighbors,
            "validation_rows": len(sample),
            "recall_at_tolerance": recall,
            "maximum_absolute_error": maximum_error,
        }

    novelty = np.empty(n_rows, dtype=float)
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        similarities = values[start:stop] @ values.T
        local = np.arange(stop - start)
        similarities[local, np.arange(start, stop)] = -np.inf
        nearest = np.partition(
            similarities, -effective_neighbors, axis=1
        )[:, -effective_neighbors:]
        novelty[start:stop] = np.mean(1.0 - nearest, axis=1)
    return novelty, {
        "method": "exact_blockwise_cosine",
        "rows": n_rows,
        "dimension": dimension,
        "requested_neighbors": neighbors,
        "effective_neighbors": effective_neighbors,
        "block_size": block_size,
    }


def static_nearest_neighbor_novelty(
    vectors: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible single-neighbour novelty helper."""
    return static_knn_novelty(vectors, neighbors=1, **kwargs)



@dataclass(frozen=True)
class PolicySpec:
    ordering: str
    reply_mode: str
    pinned: bool

    @property
    def policy_id(self) -> str:
        return f"{self.ordering}__{self.reply_mode}__{'pinned' if self.pinned else 'unpinned'}"

    @property
    def deployable(self) -> bool:
        return self.ordering != CONTROL_ORDERING


def policy_specs() -> tuple[PolicySpec, ...]:
    specs = tuple(
        PolicySpec(ordering, reply_mode, pinned)
        for ordering in ALL_ORDERINGS
        for reply_mode in REPLY_MODES
        for pinned in (False, True)
    )
    if len(specs) != 90 or sum(spec.deployable for spec in specs) != 84:
        raise AssertionError("FORUM policy factorial must contain 90/84 bundles")
    return specs


def _score_column(ordering: str) -> str | None:
    if ordering in {
        "relative_votes",
        "upvotes",
        "chronological",
        "reverse_chronological",
        CONTROL_ORDERING,
    }:
        return None
    return f"{ordering}_score"


def _primary_order(
    story: pd.DataFrame,
    indices: Sequence[int],
    ordering: str,
    *,
    seed: int,
    draw: int,
) -> np.ndarray:
    candidates = np.asarray(indices, dtype=int)
    if not len(candidates):
        return candidates
    story_id = str(story["story_id"].iloc[0])
    comment_ids = story["comment_id"].astype(str).to_numpy()
    display = pd.to_numeric(story["display_order"], errors="raise").to_numpy()
    tie = np.asarray(
        [_stable_uint64(seed, draw, story_id, comment_ids[index]) for index in candidates],
        dtype=np.uint64,
    )
    if ordering == CONTROL_ORDERING:
        return candidates[np.argsort(tie, kind="stable")]
    if ordering in {"chronological", "reverse_chronological"}:
        timestamps = pd.to_datetime(story["created_at"], utc=True, errors="raise").astype("int64").to_numpy()
        primary = timestamps[candidates]
        if ordering == "reverse_chronological":
            primary = -primary
        order = np.lexsort((comment_ids[candidates], display[candidates], primary))
        return candidates[order]
    if ordering == "relative_votes":
        values = pd.to_numeric(story["relative_votes"], errors="raise").to_numpy(float)
    elif ordering == "upvotes":
        values = pd.to_numeric(story["votes_positive"], errors="raise").to_numpy(float)
    else:
        column = _score_column(ordering)
        if column is None or column not in story:
            raise ValueError(f"Missing score column for ordering {ordering}: {column}")
        values = pd.to_numeric(story[column], errors="raise").to_numpy(float)
    if not np.isfinite(values[candidates]).all():
        raise ValueError(f"Non-finite ordering scores for {ordering}")
    order = np.lexsort((tie, -values[candidates]))
    return candidates[order]



def _induced_visible_forest(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return visible roots and each row's root after absent ancestors are removed."""
    comment_ids = frame["comment_id"].astype(str).to_numpy()
    if len(set(comment_ids)) != len(comment_ids):
        raise ValueError("Discussion contains duplicate comment IDs")
    location = {comment_id: index for index, comment_id in enumerate(comment_ids)}
    parent_index = np.full(len(frame), -1, dtype=int)
    for index, parent_id in enumerate(frame["parent_comment_id"]):
        if pd.notna(parent_id) and str(parent_id) in location:
            parent_index[index] = location[str(parent_id)]
    effective_root = np.full(len(frame), -1, dtype=int)
    for start in range(len(frame)):
        current = start
        path: list[int] = []
        seen: set[int] = set()
        while effective_root[current] < 0:
            if current in seen:
                raise ValueError("Visible comment graph contains a parent cycle")
            seen.add(current)
            path.append(current)
            parent = int(parent_index[current])
            if parent < 0:
                root = current
                break
            current = parent
        else:
            root = int(effective_root[current])
        if effective_root[current] >= 0:
            root = int(effective_root[current])
        for node in path:
            effective_root[node] = root
    roots = np.flatnonzero(parent_index < 0)
    if not len(roots) or np.any(effective_root < 0):
        raise ValueError("Could not construct the induced visible comment forest")
    return roots, effective_root


def make_policy_order(
    story: pd.DataFrame,
    spec: PolicySpec,
    *,
    seed: int = DEFAULT_SEED,
    draw: int = 1,
    visible_forest: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Construct one policy order while preserving the declared UI semantics."""
    _require_columns(
        story,
        (
            "story_id",
            "comment_id",
            "is_root",
            "root_comment_id",
            "parent_comment_id",
            "created_at",
            "preorder_position",
            "display_order",
            "is_sticky",
            "relative_votes",
            "votes_positive",
        ),
        "FORUM story",
    )
    if story["story_id"].nunique() != 1:
        raise ValueError("make_policy_order expects exactly one discussion")
    frame = story.reset_index(drop=True)
    all_indices = np.arange(len(frame), dtype=int)
    roots, effective_root = (
        _induced_visible_forest(frame)
        if visible_forest is None else visible_forest
    )
    if not len(roots):
        raise ValueError("Discussion has no root comments")
    display = pd.to_numeric(frame["display_order"], errors="raise").to_numpy()
    sticky = frame["is_sticky"].astype(bool).to_numpy()

    candidates = all_indices if spec.reply_mode == "loose" else roots
    ranked = _primary_order(
        frame, candidates, spec.ordering, seed=seed, draw=draw
    )
    if spec.pinned:
        pinned_candidates = candidates[sticky[candidates]]
        pinned_candidates = pinned_candidates[
            np.argsort(display[pinned_candidates], kind="stable")
        ]
        pinned_set = set(map(int, pinned_candidates))
        ranked = np.concatenate(
            [pinned_candidates, np.asarray([i for i in ranked if int(i) not in pinned_set])]
        )

    if spec.reply_mode in {"loose", "hidden"}:
        return ranked
    if spec.reply_mode != "trees":
        raise ValueError(f"Unknown reply mode: {spec.reply_mode}")
    preorder = pd.to_numeric(frame["preorder_position"], errors="raise").to_numpy()
    output: list[int] = []
    for root_index in ranked:
        members = all_indices[effective_root == root_index]
        members = members[np.argsort(preorder[members], kind="stable")]
        output.extend(map(int, members))
    ordered = np.asarray(output, dtype=int)
    if len(ordered) != len(frame) or len(set(map(int, ordered))) != len(frame):
        raise ValueError("Tree ordering did not preserve every comment exactly once")
    return ordered


def _complete_order_for_ndcg(
    frame: pd.DataFrame,
    order: Sequence[int],
    visible_forest: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Complete a hidden root order using frozen preorder for nDCG.

    Hidden-reply policies expose roots first and leave replies unobserved. For
    metric agreement only, those replies are appended in their frozen preorder
    within each ranked root, producing a deterministic complete permutation.
    FORUM still uses its separate hidden-reply interpolation estimand.
    """
    indices = np.asarray(order, dtype=int)
    if len(indices) == len(frame):
        return indices
    roots, effective_root = visible_forest
    all_indices = np.arange(len(frame), dtype=int)
    preorder = pd.to_numeric(frame["preorder_position"], errors="raise").to_numpy()
    output: list[int] = []
    for root_index in indices:
        members = all_indices[effective_root == root_index]
        members = members[np.argsort(preorder[members], kind="stable")]
        output.extend(map(int, members))
    completed = np.asarray(output, dtype=int)
    if len(completed) != len(frame) or len(set(map(int, completed))) != len(frame):
        raise ValueError("Could not complete hidden order for nDCG")
    return completed


@dataclass(frozen=True)
class ForumReference:
    values: np.ndarray
    best_cumulative: np.ndarray
    worst_cumulative: np.ndarray
    random_cumulative: np.ndarray


def forum_reference(values: Sequence[float]) -> ForumReference:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("FORUM values must be a finite one-dimensional vector")
    total = float(array.sum())
    n_rows = len(array)
    return ForumReference(
        values=array,
        best_cumulative=np.cumsum(np.sort(array)[::-1]),
        worst_cumulative=np.cumsum(np.sort(array)),
        random_cumulative=np.arange(1, n_rows + 1, dtype=float) * total / n_rows,
    )


def _policy_cumulative(
    reference: ForumReference,
    order: Sequence[int],
    *,
    hidden: bool,
) -> np.ndarray:
    indices = np.asarray(order, dtype=int)
    n_rows = len(reference.values)
    if not len(indices) or len(set(map(int, indices))) != len(indices):
        raise ValueError("Policy order must contain unique comment indices")
    if np.any(indices < 0) or np.any(indices >= n_rows):
        raise ValueError("Policy order contains an out-of-range comment index")
    if not hidden and len(indices) != n_rows:
        raise ValueError("Visible policy orders must contain every comment")
    cumulative = np.cumsum(reference.values[indices])
    if hidden and len(indices) < n_rows:
        visible = len(indices)
        remaining_positions = np.arange(1, n_rows - visible + 1, dtype=float)
        final_visible = float(cumulative[-1])
        total = float(reference.values.sum())
        continuation = final_visible + (total - final_visible) * (
            remaining_positions / (n_rows - visible)
        )
        cumulative = np.concatenate([cumulative, continuation])
    if len(cumulative) != n_rows:
        raise ValueError("Policy cumulative trajectory does not reach full discussion length")
    return cumulative


def forum_score(
    values: Sequence[float],
    order: Sequence[int],
    *,
    depth: int,
    hidden: bool = False,
    tolerance: float = 1e-12,
) -> float:
    """Calculate FORUM over the first ``depth`` prefixes."""
    reference = forum_reference(values)
    n_rows = len(reference.values)
    if depth < 1 or depth >= n_rows:
        raise ValueError(f"FORUM depth must be in 1..N-1; got depth={depth}, N={n_rows}")
    policy = _policy_cumulative(reference, order, hidden=hidden)
    delta = (policy - reference.random_cumulative)[:depth]
    upper = (reference.best_cumulative - reference.random_cumulative)[:depth]
    lower = (reference.random_cumulative - reference.worst_cumulative)[:depth]
    gamma = np.zeros(depth, dtype=float)
    positive = delta > tolerance
    negative = delta < -tolerance
    if np.any(positive & (upper <= tolerance)):
        raise ValueError("Positive policy delta has no attainable best-side denominator")
    if np.any(negative & (lower <= tolerance)):
        raise ValueError("Negative policy delta has no attainable worst-side denominator")
    gamma[positive] = delta[positive] / upper[positive]
    gamma[negative] = delta[negative] / lower[negative]
    if not np.isfinite(gamma).all() or np.any(np.abs(gamma) > 1 + 1e-8):
        raise ValueError("FORUM normalized deltas fall outside [-1, 1]")
    return float(np.clip(gamma, -1, 1).mean())


def ndcg_score(values: Sequence[float], order: Sequence[int], *, depth: int) -> float:
    """Calculate direct continuous-relevance nDCG for a complete ordering."""
    array = np.asarray(values, dtype=float)
    indices = np.asarray(order, dtype=int)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("nDCG values must be finite and one-dimensional")
    if len(indices) != len(array) or len(set(map(int, indices))) != len(array):
        raise ValueError("nDCG requires a complete permutation")
    if depth < 1 or depth >= len(array):
        raise ValueError("nDCG depth must be in 1..N-1")
    value_range = float(array.max() - array.min())
    if value_range <= 0:
        return math.nan
    relevance = (array - array.min()) / value_range
    discounts = 1.0 / np.log2(np.arange(2, depth + 2, dtype=float))
    observed = float((relevance[indices[:depth]] * discounts).sum())

    ideal = float((np.sort(relevance)[::-1][:depth] * discounts).sum())
    return observed / ideal if ideal > 0 else math.nan

@dataclass(frozen=True)
class _ForumMatrixReference:
    values: np.ndarray
    best_cumulative: np.ndarray
    worst_cumulative: np.ndarray
    random_cumulative: np.ndarray


def _forum_matrix_reference(values: np.ndarray) -> _ForumMatrixReference:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all():
        raise ValueError("FORUM outcome matrix must be finite and two-dimensional")
    ordered = np.sort(matrix, axis=0)
    total = matrix.sum(axis=0)
    n_rows = len(matrix)
    return _ForumMatrixReference(
        values=matrix,
        best_cumulative=np.cumsum(ordered[::-1], axis=0),
        worst_cumulative=np.cumsum(ordered, axis=0),
        random_cumulative=(
            np.arange(1, n_rows + 1, dtype=float)[:, None]
            * total[None, :]
            / n_rows
        ),
    )


def _forum_matrix_scores(
    reference: _ForumMatrixReference,
    order: Sequence[int],
    *,
    depths: Mapping[str, int],
    hidden: bool,
    tolerance: float = 1e-12,
) -> dict[str, np.ndarray]:
    indices = np.asarray(order, dtype=int)
    n_rows = len(reference.values)
    if not len(indices) or len(set(map(int, indices))) != len(indices):
        raise ValueError("Policy order must contain unique comment indices")
    if np.any(indices < 0) or np.any(indices >= n_rows):
        raise ValueError("Policy order contains an out-of-range comment index")
    if not hidden and len(indices) != n_rows:
        raise ValueError("Visible policy orders must contain every comment")
    cumulative = np.cumsum(reference.values[indices], axis=0)
    if hidden and len(indices) < n_rows:
        visible = len(indices)
        remaining = np.arange(1, n_rows - visible + 1, dtype=float)[:, None]
        final_visible = cumulative[-1][None, :]
        total = reference.values.sum(axis=0)[None, :]
        continuation = final_visible + (total - final_visible) * (
            remaining / (n_rows - visible)
        )
        cumulative = np.vstack([cumulative, continuation])
    if len(cumulative) != n_rows:
        raise ValueError("Policy cumulative matrix does not reach full length")
    maximum_depth = max(depths.values())
    minimum_depth = min(depths.values())
    if minimum_depth < 1 or maximum_depth >= n_rows:
        raise ValueError("FORUM matrix depths must be in 1..N-1")
    delta = (cumulative - reference.random_cumulative)[:maximum_depth]
    upper = (
        reference.best_cumulative - reference.random_cumulative
    )[:maximum_depth]
    lower = (
        reference.random_cumulative - reference.worst_cumulative
    )[:maximum_depth]
    gamma = np.zeros_like(delta)
    positive = delta > tolerance
    negative = delta < -tolerance
    if np.any(positive & (upper <= tolerance)):
        raise ValueError("Positive policy delta has no attainable denominator")
    if np.any(negative & (lower <= tolerance)):
        raise ValueError("Negative policy delta has no attainable denominator")
    gamma[positive] = delta[positive] / upper[positive]
    gamma[negative] = delta[negative] / lower[negative]
    if not np.isfinite(gamma).all() or np.any(np.abs(gamma) > 1 + 1e-8):
        raise ValueError("FORUM normalized matrix falls outside [-1, 1]")
    gamma = np.clip(gamma, -1, 1)
    return {
        name: gamma[:depth].mean(axis=0)
        for name, depth in depths.items()
    }


def _ndcg_matrix_scores(
    values: np.ndarray,
    order: Sequence[int],
    *,
    depths: Mapping[str, int],
) -> dict[str, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    indices = np.asarray(order, dtype=int)
    if len(indices) != len(matrix) or len(set(map(int, indices))) != len(matrix):
        raise ValueError("nDCG requires a complete permutation")
    ranges = matrix.max(axis=0) - matrix.min(axis=0)
    valid = ranges > 0
    relevance = np.zeros_like(matrix)
    relevance[:, valid] = (
        matrix[:, valid] - matrix[:, valid].min(axis=0)
    ) / ranges[valid]
    results: dict[str, np.ndarray] = {}
    for name, depth in depths.items():
        discounts = 1.0 / np.log2(np.arange(2, depth + 2, dtype=float))
        observed = (relevance[indices[:depth]] * discounts[:, None]).sum(axis=0)
        ideal = (
            np.sort(relevance, axis=0)[::-1][:depth] * discounts[:, None]
        ).sum(axis=0)
        scores = np.full(matrix.shape[1], np.nan, dtype=float)
        usable = valid & (ideal > 0)
        scores[usable] = observed[usable] / ideal[usable]
        results[name] = scores
    return results


def score_story_policies(
    story: pd.DataFrame,
    *,
    outcomes: Sequence[str] = ALL_OUTCOMES,
    tie_draws: int = DEFAULT_TIE_DRAWS,
    random_draws: int = DEFAULT_RANDOM_DRAWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Score all 90 bundles for one discussion and aggregate ordering draws."""
    if tie_draws < 1 or random_draws < 1:
        raise ValueError("tie_draws and random_draws must be positive")
    frame = story.reset_index(drop=True)
    if frame["story_id"].nunique() != 1:
        raise ValueError("score_story_policies expects exactly one discussion")
    n_rows = len(frame)
    if n_rows < PRIMARY_MIN_COMMENTS:
        raise ValueError(f"FORUM primary discussion requires at least {PRIMARY_MIN_COMMENTS} comments")
    _require_columns(frame, outcomes, "FORUM story outcomes")
    outcome_names = list(outcomes)
    outcome_matrix = np.column_stack([
        pd.to_numeric(frame[outcome], errors="raise").to_numpy(float)
        for outcome in outcome_names
    ])
    if not np.isfinite(outcome_matrix).all():
        raise ValueError("FORUM outcomes must be finite")
    depths = {"top10": 10, "full": n_rows - 1}
    forum_reference_matrix = _forum_matrix_reference(outcome_matrix)
    visible_forest = _induced_visible_forest(frame)
    rows: list[dict[str, Any]] = []
    for spec in policy_specs():
        draws = random_draws if spec.ordering == CONTROL_ORDERING else tie_draws
        forum_draws: dict[tuple[str, str], list[float]] = {}
        ndcg_draws: dict[tuple[str, str], list[float]] = {}
        visible_count = None
        trajectory_cache: dict[
            bytes,
            tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None],
        ] = {}
        for draw in range(1, draws + 1):
            order = make_policy_order(
                frame, spec, seed=seed, draw=draw, visible_forest=visible_forest
            )
            visible_count = len(order)
            cache_key = (
                None
                if spec.ordering == CONTROL_ORDERING
                else hashlib.blake2b(
                    np.asarray(order, dtype=np.int64).tobytes(),
                    digest_size=16,
                ).digest()
            )
            cached = (
                trajectory_cache.get(cache_key)
                if cache_key is not None else None
            )
            if cached is None:
                forum_by_depth = _forum_matrix_scores(
                    forum_reference_matrix,
                    order,
                    depths=depths,
                    hidden=spec.reply_mode == "hidden",
                )
                ndcg_order = _complete_order_for_ndcg(
                    frame, order, visible_forest
                )
                ndcg_by_depth = _ndcg_matrix_scores(
                    outcome_matrix, ndcg_order, depths=depths
                )
                if cache_key is not None:
                    trajectory_cache[cache_key] = (
                        forum_by_depth, ndcg_by_depth
                    )
            else:
                forum_by_depth, ndcg_by_depth = cached
            for outcome_position, outcome in enumerate(outcome_names):
                for depth_name in depths:
                    key = (outcome, depth_name)
                    forum_draws.setdefault(key, []).append(
                        float(forum_by_depth[depth_name][outcome_position])
                    )
                    if ndcg_by_depth is not None:
                        ndcg_draws.setdefault(key, []).append(
                            float(ndcg_by_depth[depth_name][outcome_position])
                        )
        for outcome in outcomes:
            for depth_name in ("top10", "full"):
                key = (outcome, depth_name)
                forum_values = np.asarray(forum_draws[key], dtype=float)
                ndcg_values = np.asarray(ndcg_draws.get(key, []), dtype=float)
                finite_ndcg = ndcg_values[np.isfinite(ndcg_values)]
                ndcg_mean = (
                    float(finite_ndcg.mean()) if len(finite_ndcg) else math.nan
                )
                ndcg_sd = (
                    float(finite_ndcg.std(ddof=1)) if len(finite_ndcg) > 1
                    else (0.0 if len(finite_ndcg) == 1 else math.nan)
                )
                rows.append(
                    {
                        "story_id": str(frame["story_id"].iloc[0]),
                        "n_comments": n_rows,
                        "n_visible_roots": len(visible_forest[0]),
                        "large_thread_sensitivity": n_rows >= SENSITIVITY_MIN_COMMENTS,
                        "policy_id": spec.policy_id,
                        "ordering": spec.ordering,
                        "reply_mode": spec.reply_mode,
                        "pinned": spec.pinned,
                        "deployable": spec.deployable,
                        "outcome": outcome,
                        "outcome_role": "primary" if outcome in PRIMARY_OUTCOMES else "secondary",
                        "depth": depth_name,
                        "visible_comments": visible_count,
                        "ordering_draws": draws,
                        "forum": float(forum_values.mean()),
                        "forum_draw_sd": float(forum_values.std(ddof=1)) if len(forum_values) > 1 else 0.0,
                        "forum_draw_min": float(forum_values.min()),
                        "forum_draw_max": float(forum_values.max()),
                        "ndcg": ndcg_mean,
                        "ndcg_draw_sd": ndcg_sd,
                    }
                )
    return pd.DataFrame(rows)


def _fixed_list_vectors(table: Any) -> np.ndarray:
    column = table.column("embedding").combine_chunks()
    dimension = int(column.type.list_size)
    values = column.values.to_numpy(zero_copy_only=False).reshape(len(column), dimension)
    return np.asarray(values, dtype=np.float32)


def build_static_novelty_store(
    choice_set: pd.DataFrame,
    *,
    embedding_store: Path,
    output_root: Path,
    block_size: int = 512,
    approximate_threshold: int = 5_000,
    seed: int = DEFAULT_SEED,
    progress_every_stories: int = 25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build resumable order-independent novelty checkpoints for held-out rows."""
    import pyarrow.parquet as pq

    _require_columns(
        choice_set,
        ("story_id", "comment_id", "article_month"),
        "novelty choice set",
    )
    _validate_unique_keys(choice_set, "novelty choice set")
    embedding_store = Path(embedding_store)
    manifest_path = embedding_store / "embedding_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    embedding_manifest = json.loads(manifest_path.read_text())
    if embedding_manifest.get("status") != "complete":
        raise RuntimeError("Semantic novelty requires a completed embedding store")
    output_root = Path(output_root)
    checkpoint_root = output_root / "semantic_novelty_knn5" / "year=2025"
    pieces: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    groups = list(choice_set.groupby("story_id", sort=True, observed=True))
    for position, (story_id, group) in enumerate(groups, start=1):
        months = group["article_month"].astype(int).unique()
        if len(months) != 1:
            raise ValueError(f"Story {story_id} has multiple article months")
        month = int(months[0])
        source = (
            embedding_store
            / "comments"
            / "year=2025"
            / f"month={month:02d}"
            / f"{story_id}.parquet"
        )
        if not source.exists():
            raise FileNotFoundError(source)
        destination = checkpoint_root / f"month={month:02d}" / f"{story_id}.parquet"
        expected_ids = group["comment_id"].astype(str).tolist()
        source_hash = _sha256(source)
        checkpoint: pd.DataFrame | None = None
        if destination.exists():
            candidate = pd.read_parquet(destination)
            if (
                set(candidate.columns)
                >= {
                    "story_id",
                    "comment_id",
                    "semantic_novelty_knn5",
                    "embedding_source_sha256",
                    "novelty_method",
                }
                and candidate["comment_id"].astype(str).tolist() == expected_ids
                and candidate["embedding_source_sha256"].eq(source_hash).all()
            ):
                checkpoint = candidate
        if checkpoint is None:
            table = pq.read_table(
                source, columns=["story_id", "comment_id", "embedding"]
            )
            source_ids = table.column("comment_id").to_pylist()
            location = {
                str(comment_id): index
                for index, comment_id in enumerate(source_ids)
            }
            missing = [
                comment_id
                for comment_id in expected_ids
                if comment_id not in location
            ]
            if missing:
                raise ValueError(
                    f"Embedding shard for story {story_id} misses "
                    f"FORUM comments: {missing[:5]}"
                )
            vectors = _fixed_list_vectors(table)
            selected = vectors[
                [location[comment_id] for comment_id in expected_ids]
            ]
            novelty, story_diagnostics = static_knn_novelty(
                selected,
                neighbors=5,
                block_size=block_size,
                approximate_threshold=approximate_threshold,
                seed=seed,
            )
            checkpoint = pd.DataFrame(
                {
                    "story_id": str(story_id),
                    "comment_id": expected_ids,
                    "semantic_novelty_knn5": novelty,
                    "novelty_method": story_diagnostics["method"],
                    "embedding_source_sha256": source_hash,
                }
            )
            _atomic_parquet(checkpoint, destination)
            story_diagnostics = {
                "story_id": str(story_id),
                "checkpoint": str(destination),
                **story_diagnostics,
            }
        else:
            story_diagnostics = {
                "story_id": str(story_id),
                "checkpoint": str(destination),
                "method": str(checkpoint["novelty_method"].iloc[0]),
                "rows": len(checkpoint),
                "reused": True,
            }
        pieces.append(
            checkpoint[
                ["story_id", "comment_id", "semantic_novelty_knn5"]
            ]
        )
        diagnostics.append(story_diagnostics)
        if (
            position % progress_every_stories == 0
            or position == len(groups)
        ):
            print(
                f"FORUM semantic novelty: {position:,}/{len(groups):,} stories",
                flush=True,
            )
    combined = pd.concat(pieces, ignore_index=True)
    _validate_unique_keys(combined, "combined semantic novelty")
    manifest = {
        "version": FORUM_ANALYSIS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "Mean cosine distance to the five nearest other comments in the same discussion"
        ),
        "order_independent": True,
        "self_excluded": True,
        "embedding_store": str(embedding_store),
        "embedding_manifest_sha256": _sha256(manifest_path),
        "stories": int(combined["story_id"].nunique()),
        "comments": len(combined),
        "exact_stories": sum(
            item["method"] == "exact_blockwise_cosine"
            for item in diagnostics
        ),
        "approximate_stories": sum(
            item["method"] == "hnsw_cosine" for item in diagnostics
        ),
        "settings": {
            "block_size": block_size,
            "approximate_threshold": approximate_threshold,
            "neighbors": 5,
            "seed": seed,
        },
    }
    _atomic_json(manifest, output_root / "semantic_novelty_manifest.json")
    return combined, manifest


def _read_raw_comments_for_stories(
    data_root: Path,
    story_ids: set[str],
) -> pd.DataFrame:
    import pyarrow.dataset as ds

    root = Path(data_root) / "comments" / "year=2025"
    dataset = ds.dataset(root, format="parquet", partitioning=None)
    table = dataset.to_table(
        columns=list(STRUCTURAL_COLUMNS),
        filter=ds.field("story_id").isin(sorted(story_ids)),
    )
    return table.to_pandas()


def build_forum_analysis_table(
    *,
    handoff_manifest_path: Path = Path(
        "model_output/selection_2025/forum_ranking_analysis/ranker_handoff/"
        "ranker_handoff_manifest.json"
    ),
    choice_set_path: Path = Path(
        "model_output/selection_2025/model_data/choice_set_all.parquet"
    ),
    split_path: Path = Path(
        "model_output/selection_2025/model_data/"
        "master_article_split.parquet"
    ),
    data_root: Path = Path("data/scrape_2025"),
    embedding_store: Path,
    output_root: Path = Path("model_output/selection_2025/forum_ranking_analysis"),
    min_comments: int = PRIMARY_MIN_COMMENTS,
) -> dict[str, Any]:
    """Build and manifest the comment-level FORUM handoff table."""
    handoff_manifest_path = Path(handoff_manifest_path)
    if not handoff_manifest_path.exists():
        raise FileNotFoundError(handoff_manifest_path)
    handoff = json.loads(handoff_manifest_path.read_text())
    if handoff.get("held_out_values_read_for_selection") is not False:
        raise RuntimeError(
            "Ranker handoff does not prove development-only selection"
        )
    choice_set_path = Path(choice_set_path)
    split_path = Path(split_path)
    choice = pd.read_parquet(
        choice_set_path, columns=list(CHOICE_COLUMNS)
    )
    split = pd.read_parquet(split_path)
    split["story_id"] = split["story_id"].astype(str)
    held_out = set(
        split.loc[split["split_role"].eq("paper2_test"), "story_id"]
    )
    choice["story_id"] = choice["story_id"].astype(str)
    choice = choice[
        choice["story_id"].isin(held_out)
        & choice["n_candidates"].ge(min_comments)
    ].copy()
    raw = _read_raw_comments_for_stories(
        data_root, set(choice["story_id"])
    )

    regression_path = next(
        Path(path)
        for path, metadata in handoff["held_out_artifacts"].items()
        if metadata["role"] == "stacked_regression_held_out_scores"
    )
    recorded_regression_hash = handoff["held_out_artifacts"][
        str(regression_path)
    ]["sha256"]
    if _sha256(regression_path) != recorded_regression_hash:
        raise RuntimeError(
            "Regression held-out score hash changed after handoff freeze"
        )
    regression_scores = pd.read_parquet(regression_path)
    selected_scores: dict[
        str, tuple[pd.DataFrame, str, str]
    ] = {}
    for selected in handoff["selected_rankers"]:
        score_path = Path(selected["score_path"])
        recorded = handoff["held_out_artifacts"][str(score_path)][
            "sha256"
        ]
        if _sha256(score_path) != recorded:
            raise RuntimeError(
                f"Selected score artifact changed: {score_path}"
            )
        family = (
            "xgb" if selected["family"] == "xgboost" else "neural"
        )
        feature = (
            "metadata_text"
            if selected["feature_set"] == "metadata_bge"
            else "metadata"
        )
        selected_scores[f"{family}_{feature}"] = (
            pd.read_parquet(score_path),
            selected["audience_column"],
            selected["editor_column"],
        )
    novelty, novelty_manifest = build_static_novelty_store(
        choice,
        embedding_store=Path(embedding_store),
        output_root=Path(output_root),
    )
    analysis = assemble_analysis_comments(
        choice,
        raw,
        split,
        regression_scores,
        selected_scores,
        novelty=novelty,
        min_comments=min_comments,
    )
    output_root = Path(output_root)
    analysis_path = output_root / "analysis_comments.parquet"
    _atomic_parquet(analysis, analysis_path)
    sample = analysis[
        ["story_id", "n_candidates", "large_thread_sensitivity"]
    ].drop_duplicates()
    manifest = {
        "version": FORUM_ANALYSIS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "all",
        "partition": "held_out_test",
        "stored_split_role": "paper2_test",
        "primary_min_comments": min_comments,
        "large_thread_min_comments": SENSITIVITY_MIN_COMMENTS,
        "primary_stories": int(sample["story_id"].nunique()),
        "large_thread_stories": int(
            sample["large_thread_sensitivity"].sum()
        ),
        "comments": len(analysis),
        "outcomes": {
            "primary": list(PRIMARY_OUTCOMES),
            "secondary": list(SECONDARY_OUTCOMES),
            "forum_extended": list(FORUM_OUTCOMES),
            "regression_features": list(REGRESSION_FEATURE_COLUMNS),
            "sensitivities": [],
        },
        "inputs": {
            str(handoff_manifest_path): _sha256(
                handoff_manifest_path
            ),
            str(choice_set_path): _sha256(choice_set_path),
            str(split_path): _sha256(split_path),
        },
        "novelty_manifest": novelty_manifest,
        "output": {
            "path": str(analysis_path),
            "sha256": _sha256(analysis_path),
            "rows": len(analysis),
        },
    }
    _atomic_json(manifest, output_root / "analysis_manifest.json")
    return manifest


def _score_story_worker(
    payload: tuple[
        str,
        pd.DataFrame,
        tuple[str, ...],
        int,
        int,
        int,
    ],
) -> tuple[str, pd.DataFrame]:
    """Score one story in a separate process for the optional parallel path."""
    story_id, story, outcomes, tie_draws, random_draws, seed = payload
    return story_id, score_story_policies(
        story,
        outcomes=outcomes,
        tie_draws=tie_draws,
        random_draws=random_draws,
        seed=seed,
    )


def run_policy_scoring(
    *,
    analysis_path: Path = Path(
        "model_output/selection_2025/forum_ranking_analysis/"
        "analysis_comments.parquet"
    ),
    output_root: Path = Path(
        "model_output/selection_2025/forum_ranking_analysis/policy_scores"
    ),
    outcomes: Sequence[str] = ALL_OUTCOMES,
    tie_draws: int = DEFAULT_TIE_DRAWS,
    random_draws: int = DEFAULT_RANDOM_DRAWS,
    seed: int = DEFAULT_SEED,
    progress_every_stories: int = 10,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Score every held-out discussion with per-story checkpoints.

    ``workers`` controls optional process-level parallelism across independent
    discussions.  The default of one preserves the original execution path;
    every worker result is checkpointed by the parent process, so interrupted
    runs remain resumable and output ordering stays deterministic.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    analysis_path = Path(analysis_path)
    if not analysis_path.exists():
        raise FileNotFoundError(analysis_path)
    analysis = pd.read_parquet(analysis_path)
    _require_columns(analysis, outcomes, "FORUM analysis table")
    output_root = Path(output_root)
    checkpoint_root = output_root / "stories"
    stories = list(
        analysis.groupby("story_id", sort=True, observed=True)
    )
    results_by_story: dict[str, pd.DataFrame] = {}
    implementation_sha256 = _sha256(Path(__file__))
    signature = hashlib.sha256(
        json.dumps(
            {
                "version": FORUM_ANALYSIS_VERSION,
                "analysis_sha256": _sha256(analysis_path),
                "outcomes": list(outcomes),
                "tie_draws": tie_draws,
                "random_draws": random_draws,
                "seed": seed,
                "implementation_sha256": implementation_sha256,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    pending: list[tuple[str, pd.DataFrame, tuple[str, ...], int, int, int]] = []

    def checkpoint_result(
        story_id: str, result: pd.DataFrame
    ) -> None:
        destination = checkpoint_root / f"{story_id}.parquet"
        result = result.copy()
        result["scoring_signature"] = signature
        _atomic_parquet(result, destination)
        results_by_story[story_id] = result

    for story_id, story in stories:
        story_key = str(story_id)
        destination = checkpoint_root / f"{story_id}.parquet"
        result: pd.DataFrame | None = None
        if destination.exists():
            candidate = pd.read_parquet(destination)
            if (
                "scoring_signature" in candidate
                and candidate["scoring_signature"].eq(signature).all()
                and len(candidate) == 90 * len(outcomes) * 2
            ):
                result = candidate
        if result is not None:
            results_by_story[story_key] = result
        else:
            pending.append(
                (
                    story_key,
                    story,
                    tuple(outcomes),
                    tie_draws,
                    random_draws,
                    seed,
                )
            )

    processed = len(results_by_story)

    def report_progress() -> None:
        if (
            processed % progress_every_stories == 0
            or processed == len(stories)
        ):
            print(
                f"FORUM policies: {processed:,}/{len(stories):,} stories",
                flush=True,
            )

    if workers == 1:
        for payload in pending:
            story_key, story, story_outcomes, story_tie_draws, story_random_draws, story_seed = payload
            result = score_story_policies(
                story,
                outcomes=story_outcomes,
                tie_draws=story_tie_draws,
                random_draws=story_random_draws,
                seed=story_seed,
            )
            checkpoint_result(story_key, result)
            processed += 1
            report_progress()
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for story_key, result in executor.map(
                _score_story_worker, pending, chunksize=1
            ):
                checkpoint_result(story_key, result)
                processed += 1
                report_progress()

    missing_stories = [
        str(story_id)
        for story_id, _ in stories
        if str(story_id) not in results_by_story
    ]
    if missing_stories:
        raise RuntimeError(
            f"Policy scoring did not produce {len(missing_stories)} story results"
        )
    pieces = [results_by_story[str(story_id)] for story_id, _ in stories]
    combined = pd.concat(pieces, ignore_index=True)
    expected_rows = len(stories) * 90 * len(outcomes) * 2
    if len(combined) != expected_rows:
        raise RuntimeError(
            "Policy-score row count mismatch: "
            f"{len(combined):,} != {expected_rows:,}"
        )
    output_path = output_root / "policy_scores.parquet"
    _atomic_parquet(combined, output_path)
    random_calibration = combined[
        combined["ordering"].eq(CONTROL_ORDERING)
        & combined["reply_mode"].eq("loose")
        & ~combined["pinned"]
    ].groupby(
        ["outcome", "depth"], as_index=False, observed=True
    ).agg(
        mean_forum=("forum", "mean"),
        maximum_absolute_story_mean=(
            "forum",
            lambda value: float(np.abs(value).max()),
        ),
    )
    random_calibration_path = (
        output_root / "random_calibration.csv"
    )
    random_calibration.to_csv(
        random_calibration_path, index=False
    )
    manifest = {
        "version": FORUM_ANALYSIS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scoring_signature": signature,
        "policies_evaluated": 90,
        "substantive_policies": 84,
        "random_controls": 6,
        "orderings": list(ALL_ORDERINGS),
        "reply_modes": list(REPLY_MODES),
        "pinning": [False, True],
        "outcomes": list(outcomes),
        "depths": ["top10", "full"],
        "tie_draws": tie_draws,
        "random_draws": random_draws,
        "seed": seed,
        "workers": workers,
        "implementation_sha256": implementation_sha256,
        "n_stories": len(stories),
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "rows": len(combined),
        },
        "random_calibration": {
            "path": str(random_calibration_path),
            "sha256": _sha256(random_calibration_path),
        },
    }
    _atomic_json(
        manifest, output_root / "policy_score_manifest.json"
    )
    return manifest



POLICY_COLUMNS = (
    "policy_id",
    "ordering",
    "reply_mode",
    "pinned",
    "deployable",
)


def _policy_metadata(scores: pd.DataFrame) -> pd.DataFrame:
    """Return one validated row of metadata per policy."""
    _require_columns(scores, POLICY_COLUMNS, "FORUM policy scores")
    metadata = scores.loc[:, POLICY_COLUMNS].drop_duplicates()
    if metadata["policy_id"].duplicated().any():
        raise ValueError("A policy_id maps to multiple policy definitions")
    expected = [spec.policy_id for spec in policy_specs()]
    missing = sorted(set(expected) - set(metadata["policy_id"]))
    if missing:
        raise ValueError(f"Policy scores are missing {len(missing)} policies")
    return metadata.set_index("policy_id").loc[expected].reset_index()


def _sample_names(scores: pd.DataFrame) -> tuple[str, ...]:
    return (
        "primary",
        *(("large_threads",) if scores["large_thread_sensitivity"].any() else ()),
    )


def _sample_frame(scores: pd.DataFrame, sample: str) -> pd.DataFrame:
    if sample == "primary":
        return scores
    if sample == "large_threads":
        return scores[scores["large_thread_sensitivity"]].copy()
    raise ValueError(f"Unknown FORUM sample: {sample}")


def _complete_panel(
    scores: pd.DataFrame,
    policy_ids: Sequence[str],
    *,
    value: str = "forum",
) -> pd.DataFrame:
    """Make a discussion-by-policy panel and reject incomplete cells."""
    panel = scores.pivot(index="story_id", columns="policy_id", values=value)
    panel = panel.reindex(columns=list(policy_ids)).sort_index()
    if panel.empty or panel.isna().any().any():
        missing = int(panel.isna().sum().sum())
        raise ValueError(
            f"Incomplete {value} panel ({missing} missing discussion-policy cells)"
        )
    return panel


def _bootstrap_panel_means(
    values: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
    chunk_size: int = 25,
) -> np.ndarray:
    """Bootstrap paired discussion means without materializing a huge cube."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Bootstrap panel must be a finite two-dimensional matrix")
    n_stories, n_cells = matrix.shape
    if n_stories < 2 or draws < 1:
        raise ValueError("Bootstrap requires at least two stories and one draw")
    result = np.empty((draws, n_cells), dtype=float)
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        indices = rng.integers(0, n_stories, size=(stop - start, n_stories))
        result[start:stop] = matrix[indices].mean(axis=1)
    return result


def _bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return float(lower), float(upper)


def bootstrap_policy_summary(
    scores: pd.DataFrame,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Estimate policy-cell means and paired discussion-bootstrap intervals."""
    metadata = _policy_metadata(scores)
    policy_ids = metadata["policy_id"].tolist()
    rows: list[dict[str, Any]] = []
    cell_number = 0
    for sample in _sample_names(scores):
        sample_scores = _sample_frame(scores, sample)
        for (outcome, depth), group in sample_scores.groupby(
            ["outcome", "depth"], sort=True, observed=True
        ):
            panel = _complete_panel(group, policy_ids)
            rng = np.random.default_rng(seed + cell_number)
            boot = _bootstrap_panel_means(panel.to_numpy(), draws=draws, rng=rng)
            estimates = panel.mean(axis=0).to_numpy(float)
            ranks = (-boot).argsort(axis=1).argsort(axis=1) + 1
            for position, policy_id in enumerate(policy_ids):
                lower, upper = _bootstrap_interval(boot[:, position])
                record = metadata.iloc[position].to_dict()
                record.update(
                    {
                        "sample": sample,
                        "outcome": outcome,
                        "depth": depth,
                        "n_stories": len(panel),
                        "estimate": float(estimates[position]),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "probability_highest_forum": float(np.mean(ranks[:, position] == 1)),
                        "probability_top_three_forum": float(np.mean(ranks[:, position] <= 3)),
                    }
                )
                rows.append(record)
            cell_number += 1
    return pd.DataFrame(rows)


def _contrast_definitions(metadata: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """Define prespecified average marginal contrasts on the 90 policy cells."""
    definitions: list[tuple[str, str, np.ndarray]] = []
    n_policies = len(metadata)
    for ordering in SUBSTANTIVE_ORDERINGS:
        weight = np.zeros(n_policies, dtype=float)
        treatment = metadata["ordering"].eq(ordering).to_numpy()
        control = metadata["ordering"].eq(CONTROL_ORDERING).to_numpy()
        weight[treatment] = 1.0 / treatment.sum()
        weight[control] = -1.0 / control.sum()
        definitions.append(("ordering_vs_random", ordering, weight))
    substantive = metadata["deployable"].to_numpy(bool)
    for reply_mode in ("trees", "hidden"):
        weight = np.zeros(n_policies, dtype=float)
        treatment = substantive & metadata["reply_mode"].eq(reply_mode).to_numpy()
        control = substantive & metadata["reply_mode"].eq("loose").to_numpy()
        weight[treatment] = 1.0 / treatment.sum()
        weight[control] = -1.0 / control.sum()
        definitions.append(("reply_vs_loose", reply_mode, weight))
    weight = np.zeros(n_policies, dtype=float)
    treatment = substantive & metadata["pinned"].to_numpy(bool)
    control = substantive & ~metadata["pinned"].to_numpy(bool)
    weight[treatment] = 1.0 / treatment.sum()
    weight[control] = -1.0 / control.sum()
    definitions.append(("pinned_vs_unpinned", "pinned", weight))
    return definitions


def bootstrap_marginal_effects(
    scores: pd.DataFrame,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Estimate paired marginal effects for ordering, replies, and pinning."""
    metadata = _policy_metadata(scores)
    policy_ids = metadata["policy_id"].tolist()
    definitions = _contrast_definitions(metadata)
    rows: list[dict[str, Any]] = []
    cell_number = 0
    for sample in _sample_names(scores):
        sample_scores = _sample_frame(scores, sample)
        for (outcome, depth), group in sample_scores.groupby(
            ["outcome", "depth"], sort=True, observed=True
        ):
            panel = _complete_panel(group, policy_ids)
            rng = np.random.default_rng(seed + 10_000 + cell_number)
            boot_means = _bootstrap_panel_means(panel.to_numpy(), draws=draws, rng=rng)
            cell_means = panel.mean(axis=0).to_numpy(float)
            for family, contrast, weight in definitions:
                boot_effect = boot_means @ weight
                lower, upper = _bootstrap_interval(boot_effect)
                rows.append(
                    {
                        "sample": sample,
                        "outcome": outcome,
                        "depth": depth,
                        "contrast_family": family,
                        "contrast": contrast,
                        "reference": (
                            "random" if family == "ordering_vs_random"
                            else "loose" if family == "reply_vs_loose"
                            else "unpinned"
                        ),
                        "n_stories": len(panel),
                        "estimate": float(cell_means @ weight),
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
            cell_number += 1
    return pd.DataFrame(rows)



def policy_design_matrix(metadata: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the two-way design with random/loose/unpinned as the reference."""
    ordering_terms = list(SUBSTANTIVE_ORDERINGS)
    reply_terms = ["trees", "hidden"]
    columns: list[np.ndarray] = [np.ones(len(metadata), dtype=float)]
    names = ["intercept[random,loose,unpinned]"]
    order_indicators: dict[str, np.ndarray] = {}
    reply_indicators: dict[str, np.ndarray] = {}
    for ordering in ordering_terms:
        indicator = metadata["ordering"].eq(ordering).to_numpy(float)
        order_indicators[ordering] = indicator
        columns.append(indicator)
        names.append(f"ordering[{ordering}]")
    for reply_mode in reply_terms:
        indicator = metadata["reply_mode"].eq(reply_mode).to_numpy(float)
        reply_indicators[reply_mode] = indicator
        columns.append(indicator)
        names.append(f"reply[{reply_mode}]")
    pinned = metadata["pinned"].to_numpy(float)
    columns.append(pinned)
    names.append("pinned")
    for ordering in ordering_terms:
        for reply_mode in reply_terms:
            columns.append(order_indicators[ordering] * reply_indicators[reply_mode])
            names.append(f"ordering[{ordering}]:reply[{reply_mode}]")
    for ordering in ordering_terms:
        columns.append(order_indicators[ordering] * pinned)
        names.append(f"ordering[{ordering}]:pinned")
    for reply_mode in reply_terms:
        columns.append(reply_indicators[reply_mode] * pinned)
        names.append(f"reply[{reply_mode}]:pinned")
    matrix = np.column_stack(columns)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise RuntimeError("FORUM policy design matrix is rank deficient")
    return matrix, names


def bootstrap_policy_decomposition(
    scores: pd.DataFrame,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the cell-mean decomposition and bootstrap paired discussions."""
    metadata = _policy_metadata(scores)
    policy_ids = metadata["policy_id"].tolist()
    design, terms = policy_design_matrix(metadata)
    inverse = np.linalg.pinv(design)
    coefficient_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    cell_number = 0
    for sample in _sample_names(scores):
        sample_scores = _sample_frame(scores, sample)
        for (outcome, depth), group in sample_scores.groupby(
            ["outcome", "depth"], sort=True, observed=True
        ):
            panel = _complete_panel(group, policy_ids)
            cell_means = panel.mean(axis=0).to_numpy(float)
            coefficients = inverse @ cell_means
            fitted = design @ coefficients
            rng = np.random.default_rng(seed + 20_000 + cell_number)
            boot_means = _bootstrap_panel_means(panel.to_numpy(), draws=draws, rng=rng)
            boot_coefficients = boot_means @ inverse.T
            for position, term in enumerate(terms):
                lower, upper = _bootstrap_interval(boot_coefficients[:, position])
                coefficient_rows.append(
                    {
                        "sample": sample,
                        "outcome": outcome,
                        "depth": depth,
                        "term": term,
                        "reference_cell": "random | loose | unpinned",
                        "n_stories": len(panel),
                        "estimate": float(coefficients[position]),
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
            for position, policy_id in enumerate(policy_ids):
                record = metadata.iloc[position].to_dict()
                record.update(
                    {
                        "sample": sample,
                        "outcome": outcome,
                        "depth": depth,
                        "cell_mean": float(cell_means[position]),
                        "fitted": float(fitted[position]),
                        "residual": float(cell_means[position] - fitted[position]),
                    }
                )
                residual_rows.append(record)
            cell_number += 1
    return pd.DataFrame(coefficient_rows), pd.DataFrame(residual_rows)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(np.asarray(left, dtype=float)).rank(method="average")
    right_rank = pd.Series(np.asarray(right, dtype=float)).rank(method="average")
    value = left_rank.corr(right_rank)
    return float(value) if pd.notna(value) else math.nan


def _rowwise_rank_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute Spearman correlations for corresponding matrix rows."""
    from scipy.stats import rankdata

    left_rank = rankdata(np.asarray(left, dtype=float), axis=1, method="average")
    right_rank = rankdata(np.asarray(right, dtype=float), axis=1, method="average")
    left_centered = left_rank - left_rank.mean(axis=1, keepdims=True)
    right_centered = right_rank - right_rank.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    result = np.full(len(left_rank), np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def bootstrap_metric_agreement(
    scores: pd.DataFrame,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compare FORUM and nDCG within reply/pinning groups and overall.

    Each point is the Spearman agreement across the 14 substantive ordering
    policies within one reply-mode/pin-state group. The overall point pools all
    84 substantive policy bundles. Random controls are excluded.
    """
    metadata = _policy_metadata(scores)
    substantive = metadata[metadata["deployable"]].copy()
    policy_groups: list[tuple[str, str | None, bool | None, list[str]]] = [
        ("overall", None, None, substantive["policy_id"].tolist())
    ]
    for reply_mode in REPLY_MODES:
        for pinned in (False, True):
            policy_groups.append(
                (
                    f"{reply_mode} | {'pinned' if pinned else 'unpinned'}",
                    reply_mode,
                    pinned,
                    substantive.loc[
                        substantive["reply_mode"].eq(reply_mode)
                        & substantive["pinned"].eq(pinned),
                        "policy_id",
                    ].tolist(),
                )
            )

    rows: list[dict[str, Any]] = []
    cell_number = 0
    for sample in _sample_names(scores):
        sample_scores = _sample_frame(scores, sample)
        for (outcome, depth), group in sample_scores.groupby(
            ["outcome", "depth"], sort=True, observed=True
        ):
            for variant_group, reply_mode, pinned, policy_ids in policy_groups:
                forum = _complete_panel(group, policy_ids, value="forum")
                ndcg = group.pivot(
                    index="story_id", columns="policy_id", values="ndcg"
                ).reindex(index=forum.index, columns=policy_ids)
                complete = ndcg.notna().all(axis=1)
                excluded_constant = int((~complete).sum())
                forum = forum.loc[complete]
                ndcg = ndcg.loc[complete]
                if len(forum) < 2:
                    raise ValueError(
                        f"Too few non-constant discussions for {outcome} / {depth} / {variant_group}"
                    )
                estimate = _rank_correlation(
                    forum.mean(axis=0).to_numpy(),
                    ndcg.mean(axis=0).to_numpy(),
                )
                cell_seed = seed + 30_000 + cell_number
                boot_forum = _bootstrap_panel_means(
                    forum.to_numpy(), draws=draws,
                    rng=np.random.default_rng(cell_seed),
                )
                boot_ndcg = _bootstrap_panel_means(
                    ndcg.to_numpy(), draws=draws,
                    rng=np.random.default_rng(cell_seed),
                )
                boot_correlation = _rowwise_rank_correlation(
                    boot_forum, boot_ndcg
                )
                finite = boot_correlation[np.isfinite(boot_correlation)]
                if not len(finite) or not np.isfinite(estimate):
                    raise ValueError(
                        f"Metric agreement is undefined for {outcome} / {depth} / {variant_group}"
                    )
                lower, upper = _bootstrap_interval(finite)
                rows.append(
                    {
                        "sample": sample,
                        "outcome": outcome,
                        "depth": depth,
                        "variant_group": variant_group,
                        "reply_mode": reply_mode,
                        "pinned": pinned,
                        "n_stories": len(forum),
                        "constant_outcome_stories_excluded": excluded_constant,
                        "n_orderings": len(policy_ids),
                        "spearman_forum_ndcg": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "scope": (
                            "all 84 substantive policy bundles"
                            if variant_group == "overall"
                            else "14 substantive orderings within reply/pin group"
                        ),
                    }
                )
                cell_number += 1
    return pd.DataFrame(rows)


PREDICTIVE_SCORE_COLUMNS = (
    "regression_audience_score",
    "regression_editor_score",
    "xgb_metadata_audience_score",
    "xgb_metadata_editor_score",
    "xgb_metadata_text_audience_score",
    "xgb_metadata_text_editor_score",
    "neural_metadata_audience_score",
    "neural_metadata_editor_score",
    "neural_metadata_text_audience_score",
    "neural_metadata_text_editor_score",
)


def mechanism_alignment(
    comments: pd.DataFrame,
    *,
    outcomes: Sequence[str] = PRIMARY_OUTCOMES,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize within-discussion score/outcome rank correlations."""
    _require_columns(
        comments,
        (
            "story_id",
            "large_thread_sensitivity",
            *PREDICTIVE_SCORE_COLUMNS,
            *outcomes,
        ),
        "FORUM mechanism analysis",
    )
    story_rows: list[dict[str, Any]] = []
    for story_id, story in comments.groupby("story_id", sort=True, observed=True):
        large = bool(story["large_thread_sensitivity"].iloc[0])
        for score_column in PREDICTIVE_SCORE_COLUMNS:
            score = pd.to_numeric(story[score_column], errors="coerce")
            for outcome in outcomes:
                values = pd.to_numeric(story[outcome], errors="coerce")
                complete = score.notna() & values.notna()
                correlation = (
                    _rank_correlation(
                        score.loc[complete].to_numpy(),
                        values.loc[complete].to_numpy(),
                    )
                    if complete.sum() >= 3
                    else math.nan
                )
                story_rows.append(
                    {
                        "story_id": str(story_id),
                        "large_thread_sensitivity": large,
                        "score": score_column,
                        "outcome": outcome,
                        "n_comments": int(complete.sum()),
                        "spearman": correlation,
                    }
                )
    story_correlations = pd.DataFrame(story_rows)
    summary_rows: list[dict[str, Any]] = []
    cell_number = 0
    samples = (
        "primary",
        *(
            ("large_threads",)
            if story_correlations["large_thread_sensitivity"].any()
            else ()
        ),
    )
    for sample in samples:
        sample_frame = (
            story_correlations
            if sample == "primary"
            else story_correlations[story_correlations["large_thread_sensitivity"]]
        )
        for (score_column, outcome), group in sample_frame.groupby(
            ["score", "outcome"], sort=True, observed=True
        ):
            values = group["spearman"].dropna().to_numpy(float)
            if len(values) < 2:
                continue
            rng = np.random.default_rng(seed + 50_000 + cell_number)
            indices = rng.integers(0, len(values), size=(draws, len(values)))
            boot = values[indices].mean(axis=1)
            lower, upper = _bootstrap_interval(boot)
            summary_rows.append(
                {
                    "sample": sample,
                    "score": score_column,
                    "outcome": outcome,
                    "n_stories": len(values),
                    "mean_within_story_spearman": float(values.mean()),
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
            cell_number += 1
    return story_correlations, pd.DataFrame(summary_rows)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _extend_existing_analysis_features(
    analysis_path: Path,
    choice_set_path: Path,
) -> list[str]:
    """Add newly required regression features without rebuilding novelty.

    Older FORUM analysis tables already contain the held-out universe,
    structural fields, semantic novelty, and frozen model scores.  The extra
    regression outcomes are deterministic columns from the canonical choice
    set, so they can be joined in-place when the embedding store is no longer
    available.
    """
    analysis_path = Path(analysis_path)
    choice_set_path = Path(choice_set_path)
    analysis = pd.read_parquet(analysis_path)
    missing = [
        column
        for column in REGRESSION_FEATURE_COLUMNS
        if column not in analysis.columns
    ]
    if not missing:
        return []
    choice = pd.read_parquet(
        choice_set_path,
        columns=["story_id", "comment_id", *missing],
    )
    analysis["story_id"] = analysis["story_id"].astype(str)
    analysis["comment_id"] = analysis["comment_id"].astype(str)
    choice["story_id"] = choice["story_id"].astype(str)
    choice["comment_id"] = choice["comment_id"].astype(str)
    _validate_unique_keys(choice, "choice set regression feature projection")
    augmented = analysis.reset_index(drop=True).merge(
        choice,
        on=["story_id", "comment_id"],
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if not augmented["_merge"].eq("both").all():
        raise ValueError("Choice set does not cover the existing analysis comments")
    augmented = augmented.drop(columns="_merge")
    if augmented[list(missing)].isna().any().any():
        raise ValueError("Regression feature extension contains missing values")
    _atomic_parquet(augmented, analysis_path)
    manifest_path = analysis_path.parent / "analysis_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["version"] = FORUM_ANALYSIS_VERSION
        manifest["extended_regression_features"] = missing
        manifest.setdefault("outcomes", {})["forum_extended"] = list(FORUM_OUTCOMES)
        manifest["outcomes"]["regression_features"] = list(REGRESSION_FEATURE_COLUMNS)
        manifest["output"] = {
            "path": str(analysis_path),
            "sha256": _sha256(analysis_path),
            "rows": len(augmented),
        }
        _atomic_json(manifest, manifest_path)
    return missing


def run_forum_analysis_pipeline(
    *,
    repo_root: Path | str = Path('.'),
    embedding_store: Path | str | None = None,
    min_comments: int = PRIMARY_MIN_COMMENTS,
    tie_draws: int = DEFAULT_TIE_DRAWS,
    random_draws: int = DEFAULT_RANDOM_DRAWS,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
    progress_every_stories: int = 10,
    outcomes: Sequence[str] = ALL_OUTCOMES,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Run each missing FORUM analysis stage and reuse completed artifacts.

    This is the notebook-facing orchestration entry point. It deliberately
    checks stage outputs independently so rerunning a presentation notebook
    does not repeat completed work. ``outcomes`` can add diagnostic FORUM
    outcomes without changing the primary reporting estimand. Policy scoring
    emits progress every ``progress_every_stories`` discussions; the
    static-novelty builder emits its own checkpoint progress.
    """
    def report_stage(message: str) -> None:
        print(f"FORUM {message}", flush=True)

    repo_root = Path(repo_root)
    analysis_root = repo_root / "model_output/selection_2025/forum_ranking_analysis"
    inference_root = analysis_root / "inference"
    reporting_root = analysis_root / "reporting"
    targets = {
        "freeze": [analysis_root / "ranker_handoff/ranker_handoff_manifest.json"],
        "build": [analysis_root / "analysis_comments.parquet"],
        "score": [analysis_root / "policy_scores/policy_scores.parquet"],
        "infer": [
            inference_root / "policy_summary.csv",
            inference_root / "marginal_effects.csv",
            inference_root / "two_way_decomposition.csv",
            inference_root / "two_way_cell_residuals.csv",
            inference_root / "forum_ndcg_agreement.csv",
            inference_root / "mechanism_alignment.csv",
        ],
        "report": [reporting_root / "reporting_manifest.json"],
    }
    statuses: dict[str, str] = {}
    agreement_path = inference_root / "forum_ndcg_agreement.csv"
    grouped_agreement_current = (
        agreement_path.exists()
        and "variant_group" in pd.read_csv(agreement_path, nrows=0).columns
    )
    analysis_path = analysis_root / "analysis_comments.parquet"
    analysis_features_current = False
    if analysis_path.exists():
        try:
            pd.read_parquet(
                analysis_path,
                columns=list(FORUM_OUTCOMES),
            )
            analysis_features_current = True
        except Exception:
            # Arrow raises backend-specific exceptions when a requested
            # column is absent; any such result means the table needs a
            # rebuild with the extended feature projection.
            analysis_features_current = False
    build_needs_run = (
        not all(path.exists() for path in targets["build"])
        or not analysis_features_current
    )
    extended_features: list[str] = []
    if build_needs_run and analysis_path.exists() and embedding_store is None:
        extended_features = _extend_existing_analysis_features(
            analysis_path,
            repo_root / "model_output/selection_2025/model_data/choice_set_all.parquet",
        )
        try:
            pd.read_parquet(
                analysis_path,
                columns=list(FORUM_OUTCOMES),
            )
            analysis_features_current = True
            build_needs_run = False
        except Exception:
            analysis_features_current = False
            build_needs_run = True

    score_manifest_path = analysis_root / "policy_scores/policy_score_manifest.json"
    score_outcomes_current = False
    if score_manifest_path.exists():
        score_manifest = json.loads(score_manifest_path.read_text())
        score_outcomes_current = score_manifest.get("outcomes") == list(outcomes)
    score_needs_run = (
        not all(path.exists() for path in targets["score"])
        or not grouped_agreement_current
        or not score_outcomes_current
        or build_needs_run
    )

    if all(path.exists() for path in targets["freeze"]):
        statuses["freeze"] = "reused"
        report_stage("freeze: reused")
    else:
        report_stage("freeze: running")
        freeze_ranker_handoff(
            factorial_root=repo_root / "model_output/selection_2025/factorial_rankers",
            regression_scores_path=repo_root / "model_output/selection_2025/regression/all/test_scores_wide.parquet",
            output_root=analysis_root / "ranker_handoff",
        )
        statuses["freeze"] = "ran"
        report_stage("freeze: complete")

    if not build_needs_run:
        statuses["build"] = "reused"
        report_stage("build: reused")
    else:
        report_stage("build: running")
        if embedding_store is None:
            raise FileNotFoundError(
                "Analysis comments are missing or cannot be extended from the choice set. "
                "Set embedding_store to the completed BGE store for a full rebuild."
            )
        build_forum_analysis_table(
            handoff_manifest_path=analysis_root / "ranker_handoff/ranker_handoff_manifest.json",
            choice_set_path=repo_root / "model_output/selection_2025/model_data/choice_set_all.parquet",
            split_path=repo_root / "model_output/selection_2025/model_data/master_article_split.parquet",
            data_root=repo_root / "data/scrape_2025",
            embedding_store=Path(embedding_store),
            output_root=analysis_root,
            min_comments=min_comments,
        )
        statuses["build"] = "ran"
        report_stage("build: complete")

    if not score_needs_run:
        statuses["score"] = "reused"
        report_stage("score: reused")
    else:
        report_stage("score: running")
        run_policy_scoring(
            analysis_path=analysis_root / "analysis_comments.parquet",
            output_root=analysis_root / "policy_scores",
            tie_draws=tie_draws,
            random_draws=random_draws,
            seed=seed,
            progress_every_stories=progress_every_stories,
            outcomes=outcomes,
            workers=workers,
        )
        statuses["score"] = "ran"
        report_stage("score: complete")

    inference_needs_run = score_needs_run or not grouped_agreement_current
    if all(path.exists() for path in targets["infer"]) and not inference_needs_run:
        statuses["infer"] = "reused"
        report_stage("inference: reused")
    else:
        report_stage("inference: running")
        run_policy_inference(
            policy_scores_path=analysis_root / "policy_scores/policy_scores.parquet",
            analysis_comments_path=analysis_root / "analysis_comments.parquet",
            output_root=inference_root,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
        )
        statuses["infer"] = "ran"
        report_stage("inference: complete")

    if all(path.exists() for path in targets["report"]) and not inference_needs_run:
        statuses["report"] = "reused"
        report_stage("report: reused")
    else:
        report_stage("report: running")
        from .ranking_algorithm_effects import run_ranking_algorithm_effects

        run_ranking_algorithm_effects(
            inference_root=inference_root,
            output_root=reporting_root,
        )
        statuses["report"] = "ran"
        report_stage("report: complete")

    if extended_features:
        statuses["build"] = "extended_from_choice_set"
    return {"stage_status": statuses, "targets": targets}


def run_policy_inference(
    *,
    policy_scores_path: Path | str,
    analysis_comments_path: Path | str,
    output_root: Path | str = "model_output/selection_2025/forum_ranking_analysis/inference",
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run the prespecified FORUM/ranking analysis inference suite and freeze its products."""
    policy_scores_path = Path(policy_scores_path)
    analysis_comments_path = Path(analysis_comments_path)
    output_root = Path(output_root)
    scores = pd.read_parquet(policy_scores_path)
    comments = pd.read_parquet(analysis_comments_path)
    _require_columns(
        scores,
        (
            "story_id",
            "large_thread_sensitivity",
            *POLICY_COLUMNS,
            "outcome",
            "depth",
            "forum",
            "ndcg",
        ),
        "FORUM policy scores",
    )
    expected_rows = (
        scores["story_id"].nunique()
        * len(policy_specs())
        * scores["outcome"].nunique()
        * scores["depth"].nunique()
    )
    if len(scores) != expected_rows:
        raise ValueError(
            f"Policy-score panel is incomplete: {len(scores):,} != {expected_rows:,}"
        )

    print("FORUM inference: policy summary running", flush=True)
    policy_summary = bootstrap_policy_summary(
        scores, draws=bootstrap_draws, seed=seed
    )
    print("FORUM inference: policy summary complete", flush=True)
    print("FORUM inference: marginal effects running", flush=True)
    marginal_effects = bootstrap_marginal_effects(
        scores, draws=bootstrap_draws, seed=seed
    )
    print("FORUM inference: marginal effects complete", flush=True)
    print("FORUM inference: two-way decomposition running", flush=True)
    decomposition, residuals = bootstrap_policy_decomposition(
        scores, draws=bootstrap_draws, seed=seed
    )
    print("FORUM inference: two-way decomposition complete", flush=True)
    print("FORUM inference: metric agreement running", flush=True)
    metric_agreement = bootstrap_metric_agreement(
        scores, draws=bootstrap_draws, seed=seed
    )
    print("FORUM inference: metric agreement complete", flush=True)
    print("FORUM inference: mechanism alignment running", flush=True)
    mechanism_stories, mechanism_summary = mechanism_alignment(
        comments, draws=bootstrap_draws, seed=seed
    )
    print("FORUM inference: mechanism alignment complete", flush=True)

    products: dict[str, tuple[pd.DataFrame, Path]] = {
        "policy_summary": (
            policy_summary,
            output_root / "policy_summary.csv",
        ),
        "marginal_effects": (
            marginal_effects,
            output_root / "marginal_effects.csv",
        ),
        "decomposition": (
            decomposition,
            output_root / "two_way_decomposition.csv",
        ),
        "decomposition_residuals": (
            residuals,
            output_root / "two_way_cell_residuals.csv",
        ),
        "metric_agreement": (
            metric_agreement,
            output_root / "forum_ndcg_agreement.csv",
        ),
        "mechanism_summary": (
            mechanism_summary,
            output_root / "mechanism_alignment.csv",
        ),
    }
    for frame, path in products.values():
        _atomic_csv(frame, path)
    mechanism_path = output_root / "mechanism_alignment_by_story.parquet"
    _atomic_parquet(mechanism_stories, mechanism_path)

    random_reference = policy_summary[
        policy_summary["ordering"].eq(CONTROL_ORDERING)
        & policy_summary["reply_mode"].eq("loose")
        & ~policy_summary["pinned"]
    ].copy()
    random_reference_path = output_root / "random_reference.csv"
    _atomic_csv(random_reference, random_reference_path)

    output_manifest: dict[str, Any] = {}
    for name, (frame, path) in products.items():
        output_manifest[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(frame),
        }
    output_manifest["mechanism_by_story"] = {
        "path": str(mechanism_path),
        "sha256": _sha256(mechanism_path),
        "rows": len(mechanism_stories),
    }
    output_manifest["random_reference"] = {
        "path": str(random_reference_path),
        "sha256": _sha256(random_reference_path),
        "rows": len(random_reference),
        "interpretation": (
            "Observed random | loose | unpinned cell; the two-way regression "
            "uses this treatment-coded reference while retaining cell residuals."
        ),
    }
    manifest = {
        "version": FORUM_ANALYSIS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "bootstrap_draws": bootstrap_draws,
        "paired_resampling_unit": "discussion",
        "primary_sample": f"held-out discussions with >= {PRIMARY_MIN_COMMENTS} comments",
        "sensitivity_sample": (
            f"held-out discussions with >= {SENSITIVITY_MIN_COMMENTS} comments"
        ),
        "regression_reference": "random | loose | unpinned",
        "random_control_role": "Explicit grounding control; excluded from deployable policy rankings.",
        "input": {
            "policy_scores": {
                "path": str(policy_scores_path),
                "sha256": _sha256(policy_scores_path),
            },
            "analysis_comments": {
                "path": str(analysis_comments_path),
                "sha256": _sha256(analysis_comments_path),
            },
        },
        "outputs": output_manifest,
    }
    _atomic_json(manifest, output_root / "inference_manifest.json")
    return manifest
