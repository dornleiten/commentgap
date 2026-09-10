"""Policy-level topic exposure, ranking, and oracle calculations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from .topic_modeling import (
    _current_code_revision,
    _frame_fingerprint,
    _topic_columns,
    _normalise_distribution,
)
from .topic_metrics import (
    _normalised_entropy,
    classify_projection,
    compute_topic_metrics,
    cosine_similarity,
    hellinger_distance,
    jensen_shannon_distance,
    shannon_entropy,
)
from .topic_artifacts import read_existing_run_parquet


def read_existing_oracle(
    paths: Mapping[str, Path | str], *, topic_columns: Sequence[str],
    reuse_existing: bool = False,
) -> dict[str, pd.DataFrame] | None:
    """Load a complete oracle result already stored in a selected run.

    Explicit historical reuse checks artifact presence and minimum schema.
    It is disabled by default: ``run_oracle_benchmark`` provides normal
    signature-validated reuse. Incomplete sets return ``None``.
    """

    if not reuse_existing:
        return None
    visible = read_existing_run_parquet(
        paths["visible_path"],
        required_columns=("story_id", "policy_id", "ordering", *topic_columns),
        reuse_existing=True,
    )
    metrics = read_existing_run_parquet(
        paths["metrics_path"], required_columns=("story_id", "policy_id"),
        reuse_existing=True,
    )
    if visible is None or metrics is None:
        return None
    try:
        summary = pd.read_csv(paths["summary_path"])
        skipped = pd.read_csv(paths["skipped_path"])
        exact_validation = pd.read_csv(paths["exact_validation_path"])
    except (OSError, ValueError, TypeError):
        return None
    return {
        "visible": visible,
        "metrics": metrics,
        "summary": summary,
        "skipped": skipped,
        "exact_validation": exact_validation,
    }

def build_topic_policy_concentration_analysis(
    visible_distributions: pd.DataFrame,
    article_distributions: pd.DataFrame,
    discussion_distributions: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    topic_columns: Sequence[str] | None = None,
    relative_votes_policy_id: str = "relative_votes__loose__unpinned",
    output_root: Path | str | None = None,
    output_prefix: str = "",
) -> dict[str, pd.DataFrame]:
    """Summarise policy concentration and differences from relative votes.

    Positive normalized-entropy changes mean a more distributed topic agenda;
    negative changes mean a more concentrated agenda. The effective-topic-count
    columns provide the same comparison in topic-count units.
    """

    article_topics = list(topic_columns or _topic_columns(article_distributions))
    if {
        "visible_normalized_entropy",
        "visible_effective_topics",
        "article_normalized_entropy",
        "discussion_normalized_entropy",
    }.issubset(metrics.columns):
        story = metrics.copy()
        story["story_id"] = story["story_id"].astype(str)
        article_frame = article_distributions[["story_id", *article_topics]].copy()
        discussion_frame = discussion_distributions[["story_id", *article_topics]].copy()
        article_values = _normalise_distribution(
            article_frame.set_index("story_id")[article_topics].to_numpy(float)
        )
        discussion_values = _normalise_distribution(
            discussion_frame.set_index("story_id")[article_topics].to_numpy(float)
        )
        article_reference = pd.DataFrame({
            "story_id": article_frame["story_id"].astype(str),
            "article_normalized_entropy": _normalised_entropy(article_values),
            "article_effective_topic_count": np.exp(shannon_entropy(article_values)),
        })
        discussion_reference = pd.DataFrame({
            "story_id": discussion_frame["story_id"].astype(str),
            "discussion_normalized_entropy": _normalised_entropy(discussion_values),
            "discussion_effective_topic_count": np.exp(shannon_entropy(discussion_values)),
        })
        if (
            "article_normalized_entropy" not in story.columns
            or "discussion_normalized_entropy" not in story.columns
        ):
            story = story.merge(article_reference, on="story_id", how="left", suffixes=("", "_reference"))
            story = story.merge(discussion_reference, on="story_id", how="left", suffixes=("", "_reference"))
        if "article_effective_topic_count" not in story.columns and "article_effective_topics" in story.columns:
            story["article_effective_topic_count"] = story["article_effective_topics"]
        if "discussion_effective_topic_count" not in story.columns and "discussion_effective_topics" in story.columns:
            story["discussion_effective_topic_count"] = story["discussion_effective_topics"]
        relative = story.loc[
            story["policy_id"].eq(relative_votes_policy_id),
            ["story_id", "visible_normalized_entropy", "visible_effective_topics"],
        ].rename(columns={
            "visible_normalized_entropy": "relative_votes_normalized_entropy",
            "visible_effective_topics": "relative_votes_effective_topic_count",
        })
        if relative.empty:
            raise ValueError(
                f"Missing relative-votes reference policy: {relative_votes_policy_id}"
            )
        relative = relative.drop_duplicates("story_id")
        story = story.merge(relative, on="story_id", how="inner", validate="many_to_one")
        story["normalized_entropy"] = story["visible_normalized_entropy"]
        story["effective_topic_count"] = story["visible_effective_topics"]
        story["normalized_entropy_change_vs_discussion"] = (
            story["normalized_entropy"] - story["discussion_normalized_entropy"]
        )
        story["normalized_entropy_change_vs_relative_votes"] = (
            story["normalized_entropy"] - story["relative_votes_normalized_entropy"]
        )
        story["effective_topic_count_change_vs_discussion"] = (
            story["effective_topic_count"] - story["discussion_effective_topic_count"]
        )
        story["effective_topic_count_change_vs_relative_votes"] = (
            story["effective_topic_count"] - story["relative_votes_effective_topic_count"]
        )
        summary_columns = [
            "normalized_entropy",
            "effective_topic_count",
            "normalized_entropy_change_vs_discussion",
            "normalized_entropy_change_vs_relative_votes",
            "effective_topic_count_change_vs_discussion",
            "effective_topic_count_change_vs_relative_votes",
        ]
        summary = story.groupby(
            ["policy_id", "ordering", "reply_mode", "pinned"], as_index=False
        ).agg(
            n_stories=("story_id", "nunique"),
            **{
                f"{column}_mean": (column, "mean")
                for column in summary_columns
            },
            **{
                f"{column}_sd": (column, "std")
                for column in summary_columns
            },
        )
        if output_root is not None:
            output_root = Path(output_root)
            output_root.mkdir(parents=True, exist_ok=True)
            story.to_parquet(
                output_root / f"{output_prefix}topic_policy_concentration_story_metrics.parquet",
                index=False,
            )
            summary.to_csv(
                output_root / f"{output_prefix}topic_policy_concentration_summary.csv",
                index=False,
            )
        return {"story": story, "summary": summary}

    if _topic_columns(discussion_distributions) != article_topics:
        raise ValueError("Article and discussion distributions must share topic columns")
    if _topic_columns(visible_distributions) != article_topics:
        raise ValueError("Visible distributions must share topic columns")

    article = article_distributions[["story_id", *article_topics]].copy()
    discussion = discussion_distributions[["story_id", *article_topics]].copy()
    visible = visible_distributions[["story_id", "policy_id", *article_topics]].copy()
    for frame in (article, discussion, visible):
        frame["story_id"] = frame["story_id"].astype(str)
    article_values = _normalise_distribution(
        article.set_index("story_id")[article_topics].to_numpy(float)
    )
    discussion_values = _normalise_distribution(
        discussion.set_index("story_id")[article_topics].to_numpy(float)
    )
    article_normalized_entropy = pd.Series(
        _normalised_entropy(article_values),
        index=article["story_id"],
        name="article_normalized_entropy",
    )
    discussion_entropy = pd.Series(
        _normalised_entropy(discussion_values),
        index=discussion["story_id"],
        name="discussion_normalized_entropy",
    )
    discussion_effective = pd.Series(
        np.exp(shannon_entropy(discussion_values)),
        index=discussion["story_id"],
        name="discussion_effective_topic_count",
    )

    visible_values = _normalise_distribution(visible[article_topics].to_numpy(float))
    visible["normalized_entropy"] = _normalised_entropy(visible_values)
    visible["effective_topic_count"] = np.exp(shannon_entropy(visible_values))
    visible = visible.merge(
        article_normalized_entropy,
        left_on="story_id",
        right_index=True,
        how="inner",
    ).merge(
        discussion_entropy,
        left_on="story_id",
        right_index=True,
        how="inner",
    ).merge(
        discussion_effective,
        left_on="story_id",
        right_index=True,
        how="inner",
    )
    relative = visible.loc[
        visible["policy_id"].eq(relative_votes_policy_id),
        ["story_id", "normalized_entropy", "effective_topic_count"],
    ].rename(columns={
        "normalized_entropy": "relative_votes_normalized_entropy",
        "effective_topic_count": "relative_votes_effective_topic_count",
    })
    if relative.empty:
        raise ValueError(
            f"Missing relative-votes reference policy: {relative_votes_policy_id}"
        )
    visible = visible.merge(relative, on="story_id", how="inner", validate="many_to_one")
    visible["normalized_entropy_change_vs_discussion"] = (
        visible["normalized_entropy"] - visible["discussion_normalized_entropy"]
    )
    visible["normalized_entropy_change_vs_relative_votes"] = (
        visible["normalized_entropy"]
        - visible["relative_votes_normalized_entropy"]
    )
    visible["effective_topic_count_change_vs_discussion"] = (
        visible["effective_topic_count"]
        - visible["discussion_effective_topic_count"]
    )
    visible["effective_topic_count_change_vs_relative_votes"] = (
        visible["effective_topic_count"]
        - visible["relative_votes_effective_topic_count"]
    )
    policy_metadata = metrics[
        ["policy_id", "ordering", "reply_mode", "pinned"]
    ].drop_duplicates("policy_id")
    story = visible.merge(policy_metadata, on="policy_id", how="left")
    summary_columns = [
        "normalized_entropy",
        "effective_topic_count",
        "normalized_entropy_change_vs_discussion",
        "normalized_entropy_change_vs_relative_votes",
        "effective_topic_count_change_vs_discussion",
        "effective_topic_count_change_vs_relative_votes",
    ]
    summary = story.groupby(
        ["policy_id", "ordering", "reply_mode", "pinned"], as_index=False
    ).agg(
        n_stories=("story_id", "nunique"),
        **{
            f"{column}_mean": (column, "mean")
            for column in summary_columns
        },
        **{
            f"{column}_sd": (column, "std")
            for column in summary_columns
        },
    )
    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        story.to_parquet(
            output_root / f"{output_prefix}topic_policy_concentration_story_metrics.parquet",
            index=False,
        )
        summary.to_csv(
            output_root / f"{output_prefix}topic_policy_concentration_summary.csv",
            index=False,
        )
    return {"story": story, "summary": summary}

def build_topic_policy_exposure_coverage_analysis(
    metrics: pd.DataFrame,
    *,
    output_root: Path | str | None = None,
    output_prefix: str = "",
) -> dict[str, pd.DataFrame]:
    """Summarise classifiable-comment exposure and its alignment relationship.

    assigned_exposure_coverage is the share of rank-weighted attention weight
    carried by comments with a valid topic assignment. It is a diagnostic for
    whether a policy's apparent topic alignment could be affected by selective
    topic-model coverage.
    """

    required = {
        "story_id",
        "policy_id",
        "ordering",
        "reply_mode",
        "pinned",
        "assigned_exposure_coverage",
        "alignment_gain",
        "article_visible_js_distance",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Policy metrics are missing coverage columns: {missing}")
    story = metrics.loc[
        :,
        [
            "story_id",
            "policy_id",
            "ordering",
            "reply_mode",
            "pinned",
            "assigned_exposure_coverage",
            "alignment_gain",
            "article_visible_js_distance",
        ],
    ].copy()
    story["story_id"] = story["story_id"].astype(str)
    if not np.isfinite(story["assigned_exposure_coverage"]).all():
        raise ValueError("Assigned exposure coverage must be finite")
    if ((story["assigned_exposure_coverage"] < 0) | (story["assigned_exposure_coverage"] > 1)).any():
        raise ValueError("Assigned exposure coverage must lie in [0, 1]")

    summary = story.groupby(
        ["policy_id", "ordering", "reply_mode", "pinned"], as_index=False
    ).agg(
        n_stories=("story_id", "nunique"),
        assigned_exposure_coverage_mean=("assigned_exposure_coverage", "mean"),
        assigned_exposure_coverage_sd=("assigned_exposure_coverage", "std"),
        alignment_gain_mean=("alignment_gain", "mean"),
        alignment_gain_sd=("alignment_gain", "std"),
        article_visible_js_distance_mean=("article_visible_js_distance", "mean"),
        article_visible_js_distance_sd=("article_visible_js_distance", "std"),
    )

    correlation_rows = []
    for label, frame in [("story_policy", story), ("policy_means", summary)]:
        coverage_column, alignment_column = (
            ("assigned_exposure_coverage", "alignment_gain")
            if label == "story_policy"
            else ("assigned_exposure_coverage_mean", "alignment_gain_mean")
        )
        coverage_values = frame[coverage_column]
        alignment_values = frame[alignment_column]
        enough_variation = (
            len(frame) >= 2
            and coverage_values.nunique(dropna=True) >= 2
            and alignment_values.nunique(dropna=True) >= 2
        )
        correlation_rows.append({
            "level": label,
            "n": len(frame),
            "pearson_coverage_alignment_gain": (
                coverage_values.corr(alignment_values) if enough_variation else np.nan
            ),
            "spearman_coverage_alignment_gain": (
                coverage_values.corr(alignment_values, method="spearman")
                if enough_variation else np.nan
            ),
        })
    correlations = pd.DataFrame(correlation_rows)

    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        story.to_parquet(
            output_root / f"{output_prefix}topic_policy_exposure_coverage_story_metrics.parquet",
            index=False,
        )
        summary.to_csv(
            output_root / f"{output_prefix}topic_policy_exposure_coverage_summary.csv",
            index=False,
        )
        correlations.to_csv(
            output_root / f"{output_prefix}topic_policy_exposure_coverage_correlations.csv",
            index=False,
        )
    return {"story": story, "summary": summary, "correlations": correlations}

def hellinger_oracle_order(
    article: Sequence[float],
    comment_distributions: Sequence[Sequence[float]] | np.ndarray,
    *,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
    rank_weight_power: float = 1.0,
    rank_weight_mode: str = "inverse_power",
    rank_decay: float = 0.1,
    rank_weight_values: Sequence[float] | None = None,
    allow_invalid_placement: bool = True,
    max_iterations: int = 5,
    objective: str = "hellinger_distance",
    structured_starts: Mapping[str, Sequence[int]] | None = None,
    n_starts: int = 1,
    random_state: int | None = None,
    n_perturbations: int = 0,
    perturbation_fraction: float = 0.10,
    n_local_moves: int = 0,
) -> dict[str, object]:
    """Find an article-aligned ordering for one story.

    The visible distribution uses the same inverse-rank weights as the main
    analysis. Comments are initially sorted by Hellinger affinity to the
    article, then repeatedly reassigned using a first-order square-root-space
    objective until no improvement is found or max_iterations is reached.
    The hellinger_distance and jensen_shannon_distance objectives
    directly minimize the corresponding article-visible distance. The
    cosine_similarity objective maximizes ordinary cosine on the original
    topic proportions; it is reported as a distinct robustness measure and
    is not mathematically equivalent to Hellinger distance.

    The search always includes the deterministic affinity ordering. Additional
    structured starts may be supplied, followed by reproducible random starts.
    n_perturbations adds iterated-local-search steps after each greedy result:
    a rank-aware subset of valid comments is shuffled, optional sampled
    insertion/swap moves are applied, and the greedy refinement is rerun. A
    perturbed result becomes the next local state only when it improves that
    start. The best result across all starts and perturbations is returned.

    This is a best-found oracle benchmark, not a guaranteed global optimum:
    the exact permutation problem is combinatorial. Invalid comments may be
    retained via valid_mask; they carry no topic mass but can be placed
    anywhere when allow_invalid_placement is true. Their positions still
    affect the rank weights applied to valid comments.
    """

    if rank_weight_power <= 0 or not np.isfinite(rank_weight_power):
        raise ValueError("rank_weight_power must be finite and positive")
    _rank_attention_weights(
        1,
        rank_weight_power=rank_weight_power,
        rank_weight_mode=rank_weight_mode,
        rank_decay=rank_decay,
        rank_weight_values=rank_weight_values,
    )
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if not isinstance(n_starts, (int, np.integer)) or n_starts < 1:
        raise ValueError("n_starts must be a positive integer")
    if not isinstance(n_perturbations, (int, np.integer)) or n_perturbations < 0:
        raise ValueError("n_perturbations must be a non-negative integer")
    if not isinstance(n_local_moves, (int, np.integer)) or n_local_moves < 0:
        raise ValueError("n_local_moves must be a non-negative integer")
    if not np.isfinite(perturbation_fraction) or not 0 < perturbation_fraction <= 1:
        raise ValueError("perturbation_fraction must be in (0, 1]")
    if objective not in {
        "hellinger_distance",
        "jensen_shannon_distance",
        "cosine_similarity",
    }:
        raise ValueError(
            "objective must be 'hellinger_distance', "
            "'jensen_shannon_distance', or 'cosine_similarity'"
        )
    article_array = _normalise_distribution(np.asarray(article, dtype=float))
    raw = np.asarray(comment_distributions, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != article_array.shape[0] or raw.shape[0] == 0:
        raise ValueError("comment_distributions must be a non-empty matrix matching article")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("comment_distributions must be finite and non-negative")
    if valid_mask is None:
        valid = raw.sum(axis=1) > np.finfo(float).eps
    else:
        valid = np.asarray(valid_mask, dtype=bool).copy()
        if valid.shape != (len(raw),):
            raise ValueError("valid_mask must have one value per comment")
        valid &= raw.sum(axis=1) > np.finfo(float).eps
    if not valid.any():
        raise ValueError("At least one valid comment distribution is required")

    values = np.zeros_like(raw, dtype=float)
    values[valid] = raw[valid] / raw[valid].sum(axis=1, keepdims=True)
    weights = _rank_attention_weights(
        len(values),
        rank_weight_power=rank_weight_power,
        rank_weight_mode=rank_weight_mode,
        rank_decay=rank_decay,
        rank_weight_values=rank_weight_values,
    )
    article_root = np.sqrt(article_array)
    article_norm = np.linalg.norm(article_array)
    all_indices = np.arange(len(raw), dtype=int)
    valid_indices = np.flatnonzero(valid)
    invalid_indices = np.flatnonzero(~valid)

    def distribution_for(order: np.ndarray) -> np.ndarray:
        weighted = values[order] * weights[:, None]
        distribution = weighted.sum(axis=0)
        total = distribution.sum()
        if total <= np.finfo(float).eps:
            raise ValueError("The ordered comments have no valid topic mass")
        return distribution / total

    affinity_vector = article_array if objective == "cosine_similarity" else article_root
    affinity = values @ affinity_vector
    affinity[~valid] = 0.0 if allow_invalid_placement else -np.inf
    def assign_by_weight(scores: np.ndarray) -> np.ndarray:
        candidate = np.empty(len(scores), dtype=int)
        if allow_invalid_placement:
            slots = np.argsort(-weights, kind="stable")
            candidate[slots] = np.argsort(-scores, kind="stable")
        else:
            slots = np.argsort(-weights[:len(valid_indices)], kind="stable")
            candidate[slots] = valid_indices[np.argsort(-scores[valid_indices], kind="stable")]
            candidate[len(valid_indices):] = invalid_indices
        return candidate

    affinity_order = assign_by_weight(affinity)

    cosine_similarity_metric = globals()["cosine_similarity"]

    def cosine_similarity(distribution: np.ndarray) -> float:
        return float(cosine_similarity_metric(article_array, distribution))

    def objective_value(distribution: np.ndarray) -> float:
        if objective == "cosine_similarity":
            return -cosine_similarity(distribution)
        if objective == "jensen_shannon_distance":
            return float(jensen_shannon_distance(article_array, distribution))
        return float(hellinger_distance(article_array, distribution))

    def refine(start_order: np.ndarray) -> dict[str, object]:
        order = start_order.copy()
        start_distribution = distribution_for(order)
        start_objective = objective_value(start_distribution)
        iterations = 0
        for iteration in range(max_iterations):
            visible = distribution_for(order)
            if objective == "jensen_shannon_distance":
                midpoint = (article_array + visible) / 2.0
                js_gradient = 0.5 * np.log(
                    np.maximum(visible, np.finfo(float).eps)
                    / np.maximum(midpoint, np.finfo(float).eps)
                )
                utility = -js_gradient
            elif objective == "cosine_similarity":
                visible_norm = max(np.linalg.norm(visible), np.finfo(float).eps)
                dot_product = float(np.dot(article_array, visible))
                utility = (
                    article_array / (article_norm * visible_norm)
                    - dot_product * visible / (article_norm * visible_norm**3)
                )
            else:
                utility = article_root / np.sqrt(np.maximum(visible, np.finfo(float).eps))
            scores = values @ utility
            scores[~valid] = 0.0 if allow_invalid_placement else -np.inf
            candidate = assign_by_weight(scores)
            candidate_distribution = distribution_for(candidate)
            candidate_objective = objective_value(candidate_distribution)
            current_objective = objective_value(visible)
            iterations = iteration + 1
            if candidate_objective + 1e-12 >= current_objective:
                break
            order = candidate
        distribution = distribution_for(order)
        return {
            "order": order,
            "distribution": distribution,
            "objective": objective_value(distribution),
            "distance": float(hellinger_distance(article_array, distribution)),
            "iterations": iterations,
            "start_objective": start_objective,
        }

    def normalise_start_order(order: Sequence[int]) -> np.ndarray:
        candidate = np.asarray(order, dtype=int)
        if candidate.shape != (len(raw),) or not np.array_equal(np.sort(candidate), all_indices):
            raise ValueError("Each structured start must be a permutation of comment indices")
        if allow_invalid_placement:
            return candidate
        return np.concatenate([candidate[valid[candidate]], candidate[~valid[candidate]]])

    initial_distribution = distribution_for(affinity_order)
    initial_distance = float(hellinger_distance(article_array, initial_distribution))
    best_order = affinity_order.copy()
    best_distribution = initial_distribution.copy()
    best_distance = initial_distance
    best_objective = objective_value(initial_distribution)
    best_start = 0
    best_start_label = "affinity"
    best_perturbation = 0
    best_iterations = 0
    improved_starts = 0
    accepted_perturbations = 0
    local_moves_attempted = 0
    total_iterations = 0
    rng = np.random.default_rng(random_state)
    perturbation_population = len(raw) if allow_invalid_placement else len(valid_indices)
    perturbation_size = min(
        perturbation_population,
        max(2, int(np.ceil(perturbation_population * perturbation_fraction))),
    ) if perturbation_population >= 2 else perturbation_population
    rank_positions = np.arange(len(raw), dtype=int)
    rank_probabilities = weights.copy()
    if not allow_invalid_placement:
        rank_probabilities[len(valid_indices):] = 0.0
    rank_probabilities /= rank_probabilities.sum()

    seed_specs: list[tuple[str, np.ndarray]] = [("affinity", affinity_order.copy())]
    if structured_starts:
        for label, order in structured_starts.items():
            seed_specs.append((f"structured:{label}", normalise_start_order(order)))
    for random_index in range(int(n_starts)):
        random_order = (
            rng.permutation(all_indices)
            if allow_invalid_placement
            else np.concatenate([rng.permutation(valid_indices), invalid_indices])
        )
        seed_specs.append((f"random:{random_index}", random_order))

    for start_index, (start_label, seed_order) in enumerate(seed_specs):
        local = refine(seed_order)
        total_iterations += int(local["iterations"])
        if local["objective"] + 1e-12 < local["start_objective"]:
            improved_starts += 1
        if local["objective"] + 1e-12 < best_objective:
            best_order = local["order"].copy()
            best_distribution = local["distribution"].copy()
            best_distance = float(local["distance"])
            best_objective = float(local["objective"])
            best_start = start_index
            best_start_label = start_label
            best_perturbation = 0
            best_iterations = int(local["iterations"])

        for perturbation_index in range(1, int(n_perturbations) + 1):
            if perturbation_size < 2:
                break
            perturbed = local["order"].copy()
            positions = rng.choice(
                rank_positions,
                size=perturbation_size,
                replace=False,
                p=rank_probabilities,
            )
            shuffled = rng.permutation(perturbed[positions])
            if np.array_equal(shuffled, perturbed[positions]):
                shuffled = np.roll(shuffled, 1)
            perturbed[positions] = shuffled
            for _ in range(int(n_local_moves)):
                local_moves_attempted += 1
                move_positions = rng.choice(
                    rank_positions,
                    size=2,
                    replace=False,
                    p=rank_probabilities,
                )
                first, second = sorted(move_positions.tolist())
                if rng.random() < 0.5:
                    perturbed[first], perturbed[second] = perturbed[second], perturbed[first]
                else:
                    value = perturbed[second]
                    perturbed[first + 1:second + 1] = perturbed[first:second]
                    perturbed[first] = value
            candidate = refine(perturbed)
            total_iterations += int(candidate["iterations"])
            if candidate["objective"] + 1e-12 < local["objective"]:
                local = candidate
                accepted_perturbations += 1
            if candidate["objective"] + 1e-12 < best_objective:
                best_order = candidate["order"].copy()
                best_distribution = candidate["distribution"].copy()
                best_distance = float(candidate["distance"])
                best_objective = float(candidate["objective"])
                best_start = start_index
                best_start_label = start_label
                best_perturbation = perturbation_index
                best_iterations = int(candidate["iterations"])

    return {
        "order": best_order,
        "distribution": best_distribution,
        "hellinger_distance": best_distance,
        "cosine_similarity": cosine_similarity(best_distribution),
        "initial_hellinger_distance": initial_distance,
        "initial_cosine_similarity": cosine_similarity(initial_distribution),
        "objective": objective,
        "objective_value": -best_objective if objective == "cosine_similarity" else best_objective,
        "iterations": best_iterations,
        "total_iterations": total_iterations,
        "structured_starts": len(structured_starts or {}),
        "n_starts": int(n_starts),
        "random_state": random_state,
        "best_start": best_start,
        "best_start_label": best_start_label,
        "best_perturbation": best_perturbation,
        "improved_starts": improved_starts,
        "accepted_perturbations": accepted_perturbations,
        "local_moves_attempted": local_moves_attempted,
        "n_perturbations": int(n_perturbations),
        "perturbation_fraction": float(perturbation_fraction),
        "n_local_moves": int(n_local_moves),
        "allow_invalid_placement": bool(allow_invalid_placement),
    }

def exact_hellinger_oracle_order(
    article: Sequence[float],
    comment_distributions: Sequence[Sequence[float]] | np.ndarray,
    *,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
    rank_weight_power: float = 1.0,
    rank_weight_mode: str = "inverse_power",
    rank_decay: float = 0.1,
    rank_weight_values: Sequence[float] | None = None,
    objective: str = "hellinger_distance",
    max_valid_comments: int = 8,
    allow_invalid_placement: bool = True,
) -> dict[str, object]:
    """Find the exact best ordering for a small comment set.

    Every permutation of the comments is evaluated when invalid placement is
    enabled, so invalid comments can act as rank-weight spacers. This is
    intended for validation of the heuristic oracle on small threads; it is
    factorial in the number of comments and refuses larger inputs.
    """

    if not isinstance(max_valid_comments, (int, np.integer)) or max_valid_comments < 1:
        raise ValueError("max_valid_comments must be a positive integer")
    if rank_weight_power <= 0 or not np.isfinite(rank_weight_power):
        raise ValueError("rank_weight_power must be finite and positive")
    _rank_attention_weights(
        1,
        rank_weight_power=rank_weight_power,
        rank_weight_mode=rank_weight_mode,
        rank_decay=rank_decay,
        rank_weight_values=rank_weight_values,
    )
    if objective not in {
        "hellinger_distance",
        "jensen_shannon_distance",
        "cosine_similarity",
    }:
        raise ValueError(
            "objective must be 'hellinger_distance', "
            "'jensen_shannon_distance', or 'cosine_similarity'"
        )
    article_array = _normalise_distribution(np.asarray(article, dtype=float))
    raw = np.asarray(comment_distributions, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != article_array.shape[0] or raw.shape[0] == 0:
        raise ValueError("comment_distributions must be a non-empty matrix matching article")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("comment_distributions must be finite and non-negative")
    if valid_mask is None:
        valid = raw.sum(axis=1) > np.finfo(float).eps
    else:
        valid = np.asarray(valid_mask, dtype=bool).copy()
        if valid.shape != (len(raw),):
            raise ValueError("valid_mask must have one value per comment")
        valid &= raw.sum(axis=1) > np.finfo(float).eps
    valid_indices = np.flatnonzero(valid)
    invalid_indices = np.flatnonzero(~valid)
    all_indices = np.arange(len(raw), dtype=int)
    if not len(valid_indices):
        raise ValueError("At least one valid comment distribution is required")
    enumeration_size = len(raw) if allow_invalid_placement else len(valid_indices)
    if enumeration_size > max_valid_comments:
        raise ValueError(
            f"Exact enumeration supports at most {max_valid_comments} comments under "
            f"the selected invalid-placement rule; received {enumeration_size}"
        )

    values = np.zeros_like(raw, dtype=float)
    values[valid] = raw[valid] / raw[valid].sum(axis=1, keepdims=True)
    weights = _rank_attention_weights(
        len(raw),
        rank_weight_power=rank_weight_power,
        rank_weight_mode=rank_weight_mode,
        rank_decay=rank_decay,
        rank_weight_values=rank_weight_values,
    )
    article_root = np.sqrt(article_array)
    cosine_similarity_metric = globals()["cosine_similarity"]

    def distribution_for(order: np.ndarray) -> np.ndarray:
        weighted = values[order] * weights[:, None]
        distribution = weighted.sum(axis=0)
        total = distribution.sum()
        if total <= np.finfo(float).eps:
            raise ValueError("The ordered comments have no valid topic mass")
        return distribution / total

    def cosine_similarity(distribution: np.ndarray) -> float:
        return float(cosine_similarity_metric(article_array, distribution))

    def objective_value(distribution: np.ndarray) -> float:
        if objective == "cosine_similarity":
            return -cosine_similarity(distribution)
        if objective == "jensen_shannon_distance":
            return float(jensen_shannon_distance(article_array, distribution))
        return float(hellinger_distance(article_array, distribution))

    best_order = None
    best_distribution = None
    best_objective = np.inf
    n_permutations = 0
    permutation_indices = all_indices.tolist() if allow_invalid_placement else valid_indices.tolist()
    for permutation in permutations(permutation_indices):
        order = np.asarray(
            permutation if allow_invalid_placement else [*permutation, *invalid_indices.tolist()],
            dtype=int,
        )
        distribution = distribution_for(order)
        value = objective_value(distribution)
        n_permutations += 1
        if value + 1e-12 < best_objective:
            best_order = order
            best_distribution = distribution
            best_objective = value

    assert best_order is not None and best_distribution is not None
    return {
        "order": best_order,
        "distribution": best_distribution,
        "hellinger_distance": float(hellinger_distance(article_array, best_distribution)),
        "jensen_shannon_distance": float(jensen_shannon_distance(article_array, best_distribution)),
        "cosine_similarity": cosine_similarity(best_distribution),
        "objective": objective,
        "objective_value": -best_objective if objective == "cosine_similarity" else best_objective,
        "n_permutations": n_permutations,
        "n_valid_comments": int(len(valid_indices)),
        "n_comments": int(len(raw)),
        "allow_invalid_placement": bool(allow_invalid_placement),
        "search_method": "exact",
    }

def ranked_topic_trajectory(
    memberships: pd.DataFrame,
    rankings: pd.DataFrame,
    *,
    ranking_group_columns: Sequence[str] = ("story_id", "policy_id"),
    rank_column: str = "rank",
    comment_id_column: str = "comment_id",
    max_depth: int | None = None,
) -> pd.DataFrame:
    """Calculate cumulative visible topic distributions along each ranking."""

    topic_columns = _topic_columns(memberships)
    required = [*ranking_group_columns, rank_column, comment_id_column]
    missing = [column for column in required if column not in rankings]
    if missing:
        raise ValueError(f"rankings is missing required columns: {missing}")
    membership_columns = ["story_id", comment_id_column, *topic_columns]
    if "story_id" not in memberships:
        raise ValueError("memberships must contain story_id")
    topic_memberships = memberships
    if "valid_topic" in topic_memberships.columns:
        topic_memberships = topic_memberships.loc[topic_memberships["valid_topic"].astype(bool)]
    joined = rankings.merge(
        topic_memberships[membership_columns],
        on=["story_id", comment_id_column],
        how="inner",
        validate="many_to_one",
    )
    if joined[topic_columns].isna().any().any():
        raise ValueError("rankings contains comments without topic memberships")
    rows = []
    for keys, group in joined.sort_values([*ranking_group_columns, rank_column]).groupby(list(ranking_group_columns), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if group[rank_column].duplicated().any():
            raise ValueError("Each ranking must have unique, increasing ranks")
        group = group.sort_values(rank_column)
        if not group[rank_column].is_monotonic_increasing or (group[rank_column] < 1).any():
            raise ValueError("Each ranking must have unique, positive, increasing ranks")
        values = group[topic_columns].to_numpy(dtype=float)
        cumulative = np.cumsum(values, axis=0)
        depth_limit = len(group) if max_depth is None else min(max_depth, len(group))
        for index in range(depth_limit):
            row = dict(zip(ranking_group_columns, keys))
            row["depth"] = index + 1
            row["n_visible"] = index + 1
            row.update(dict(zip(topic_columns, _normalise_distribution(cumulative[index]))))
            rows.append(row)
    return pd.DataFrame(rows, columns=[*ranking_group_columns, "depth", "n_visible", *topic_columns])

def weighted_ranked_topic_distributions(
    memberships: pd.DataFrame,
    comments: pd.DataFrame,
    score_columns: Mapping[str, str | None],
    *,
    ascending: Iterable[str] = (),
    random_draws: int = 25,
    seed: int = 2025,
    rank_weight_power: float = 1.0,
    progress: bool = True,
    progress_every_stories: int = 25,
) -> pd.DataFrame:
    """Compute one full-discussion inverse-rank distribution per story and policy.

    Every comment is ranked, and a comment at position r receives weight
    1 / r**rank_weight_power. Invalid-topic comments retain their rank
    position but contribute no topic mass. Rankings are processed story by
    story so the full long ranking table is never materialised.
    """

    if rank_weight_power <= 0 or not np.isfinite(rank_weight_power):
        raise ValueError("rank_weight_power must be finite and positive")
    if random_draws < 1:
        raise ValueError("random_draws must be at least 1")
    topic_columns = _topic_columns(memberships)
    required_comments = {"story_id", "comment_id"}
    required_comments.update(
        score for policy, score in score_columns.items() if policy != "random" and score is not None
    )
    missing = sorted(required_comments - set(comments.columns))
    if missing:
        raise ValueError(f"comments is missing ranking columns: {missing}")

    comment_frame = comments.loc[:, sorted(required_comments)].copy()
    comment_frame["story_id"] = comment_frame["story_id"].astype(str)
    comment_frame["comment_id"] = comment_frame["comment_id"].astype(str)
    if comment_frame.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("comments must contain one row per story/comment_id")

    topic_memberships = memberships.copy()
    topic_memberships["story_id"] = topic_memberships["story_id"].astype(str)
    topic_memberships["comment_id"] = topic_memberships["comment_id"].astype(str)
    topic_memberships = topic_memberships.loc[
        topic_memberships["doc_type"].eq("comment") if "doc_type" in topic_memberships else slice(None)
    ]
    if topic_memberships.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("memberships must contain one row per story/comment_id")
    topic_memberships = topic_memberships.set_index(["story_id", "comment_id"], drop=False)
    ascending = set(ascending)
    rng = np.random.default_rng(seed)
    rows = []

    for story_id, group in comment_frame.groupby("story_id", sort=True):
        group = group.reset_index(drop=True)
        ids = group["comment_id"].tolist()
        try:
            mapped = topic_memberships.loc[
                [(story_id, comment_id) for comment_id in ids],
                topic_columns + (["valid_topic"] if "valid_topic" in topic_memberships else []),
            ]
        except KeyError as exc:
            raise ValueError(f"comments contains a story/comment without a topic membership: {story_id}") from exc
        values = mapped[topic_columns].to_numpy(dtype=float)
        valid = (
            mapped["valid_topic"].astype(bool).to_numpy()
            if "valid_topic" in mapped
            else np.ones(len(mapped), dtype=bool)
        )

        def add_distribution(policy_id: str, order: np.ndarray) -> None:
            weights = 1.0 / (np.arange(1, len(order) + 1, dtype=float) ** rank_weight_power)
            valid_weights = weights[valid[order]]
            if not len(valid_weights) or valid_weights.sum() <= 0:
                return
            distribution = (values[order][valid[order]] * valid_weights[:, None]).sum(axis=0)
            distribution /= distribution.sum()
            row = {
                "story_id": story_id,
                "policy_id": policy_id,
                "n_documents": len(order),
                "n_valid_documents": int(valid[order].sum()),
                "rank_weight_sum": float(valid_weights.sum()),
            }
            row.update(dict(zip(topic_columns, distribution)))
            rows.append(row)

        for policy, score_column in score_columns.items():
            if policy == "random":
                for draw in range(random_draws):
                    add_distribution(
                        f"random_{draw + 1:04d}",
                        rng.permutation(len(group)),
                    )
                continue
            scores = pd.to_numeric(group[score_column], errors="coerce")
            if not np.isfinite(scores.to_numpy(dtype=float)).all():
                raise ValueError(f"Ranking column {score_column!r} contains non-finite values")
            ordered = group.assign(_score=scores).sort_values(
                ["_score", "comment_id"],
                ascending=[policy in ascending, True],
                kind="mergesort",
            )
            add_distribution(policy, ordered.index.to_numpy(dtype=int))

    return pd.DataFrame(
        rows,
        columns=["story_id", "policy_id", "n_documents", "n_valid_documents", "rank_weight_sum", *topic_columns],
    )

def _rank_attention_weights(
    n_documents: int,
    *,
    rank_weight_power: float = 1.0,
    rank_weight_mode: str = "inverse_power",
    rank_decay: float = 0.1,
    rank_weight_values: Sequence[float] | None = None,
) -> np.ndarray:
    """Return unnormalised attention weights for display ranks."""

    if not isinstance(n_documents, (int, np.integer)) or n_documents < 1:
        raise ValueError("n_documents must be a positive integer")
    if rank_weight_mode not in {"inverse_power", "exponential", "empirical"}:
        raise ValueError("rank_weight_mode must be 'inverse_power', 'exponential', or 'empirical'")
    if rank_weight_mode == "empirical":
        values = np.asarray(rank_weight_values, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Empirical rank_weight_values must be a finite positive vector")
        # Freeze the endpoint beyond development support; never infer from test votes.
        return values[np.minimum(np.arange(n_documents), len(values) - 1)].copy()
    if rank_weight_power <= 0 or not np.isfinite(rank_weight_power):
        raise ValueError("rank_weight_power must be finite and positive")
    if rank_weight_mode == "exponential" and (
        rank_decay <= 0 or not np.isfinite(rank_decay)
    ):
        raise ValueError("rank_decay must be finite and positive for exponential weighting")
    ranks = np.arange(1, n_documents + 1, dtype=np.float64)
    if rank_weight_mode == "inverse_power":
        return 1.0 / (ranks ** rank_weight_power)
    return np.exp(-rank_decay * (ranks - 1.0))

def weighted_policy_topic_distributions(
    memberships: pd.DataFrame,
    comments: pd.DataFrame,
    *,
    tie_draws: int = 10,
    random_draws: int = 25,
    seed: int = 2025,
    rank_weight_power: float = 1.0,
    progress: bool = True,
    progress_every_stories: int = 25,
    n_jobs: int = 1,
    retain_draws: bool = False,
    rank_weight_mode: str = "inverse_power",
    rank_decay: float = 0.1,
    rank_weight_values: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Compute rank-weighted topic distributions for the factorial policies.

    When retain_draws is true, one row is returned for each random or tie draw.
    This allows nonlinear metrics such as entropy and Jensen-Shannon distance
    to be calculated per realized presentation before averaging within story.
    

    Each story is independent, so stories can be processed concurrently. Thread
    workers are used deliberately: they share the read-only topic membership
    frame and avoid copying the large topic matrix into every process.
    """
    from .forum_scores import ALL_ORDERINGS, make_policy_order, policy_specs

    if rank_weight_power <= 0 or not np.isfinite(rank_weight_power):
        raise ValueError("rank_weight_power must be finite and positive")
    if tie_draws < 1 or random_draws < 1:
        raise ValueError("tie_draws and random_draws must be positive")
    if progress_every_stories < 1:
        raise ValueError("progress_every_stories must be positive")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    _rank_attention_weights(
        1,
        rank_weight_power=rank_weight_power,
        rank_weight_mode=rank_weight_mode,
        rank_decay=rank_decay,
        rank_weight_values=rank_weight_values,
    )

    topic_columns = _topic_columns(memberships)
    non_score_orderings = {
        "relative_votes", "upvotes", "chronological",
        "reverse_chronological", "random",
    }
    score_columns = {
        f"{ordering}_score"
        for ordering in ALL_ORDERINGS
        if ordering not in non_score_orderings
    }
    required_comments = {
        "story_id", "comment_id", "is_root", "root_comment_id",
        "parent_comment_id", "created_at", "preorder_position",
        "display_order", "root_order", "is_sticky", "votes_positive",
        "votes_negative", "relative_votes", *score_columns,
    }
    missing = sorted(required_comments - set(comments.columns))
    if missing:
        raise ValueError(f"comments is missing policy columns: {missing}")

    comment_frame = comments.loc[:, sorted(required_comments)].copy()
    comment_frame["story_id"] = comment_frame["story_id"].astype(str)
    comment_frame["comment_id"] = comment_frame["comment_id"].astype(str)
    if comment_frame.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("comments must contain one row per story/comment_id")

    topic_memberships = memberships.copy()
    topic_memberships["story_id"] = topic_memberships["story_id"].astype(str)
    topic_memberships["comment_id"] = topic_memberships["comment_id"].astype(str)
    if "doc_type" in topic_memberships:
        topic_memberships = topic_memberships.loc[topic_memberships["doc_type"].eq("comment")]
    membership_columns = ["story_id", "comment_id", *topic_columns]
    if "valid_topic" in topic_memberships:
        membership_columns.append("valid_topic")
    if topic_memberships.duplicated(["story_id", "comment_id"]).any():
        raise ValueError("memberships must contain one row per story/comment_id")
    topic_memberships = topic_memberships.set_index(["story_id", "comment_id"], drop=False)

    specs = policy_specs()
    story_groups = comment_frame.groupby("story_id", sort=True)
    n_stories = int(comment_frame["story_id"].nunique())
    n_policy_cells = len(specs)
    started = time.perf_counter()
    if progress:
        worker_label = "serial" if n_jobs == 1 else f"{n_jobs} threads"
        print(
            f"Starting factorial topic distributions: {n_stories:,} stories x "
            f"{n_policy_cells} policy cells using {worker_label}; "
            f"random_draws={random_draws}, tie_draws={tie_draws}.",
            flush=True,
        )

    def process_story(item: tuple[str, pd.DataFrame]) -> tuple[list[dict[str, object]], list[np.ndarray]]:
        story_id, group = item
        group = group.reset_index(drop=True)
        ids = group["comment_id"].tolist()
        keys = [(story_id, comment_id) for comment_id in ids]
        try:
            mapped = topic_memberships.loc[keys, membership_columns[2:]]
        except KeyError as exc:
            raise ValueError(
                f"comments contains a story/comment without a topic membership: {story_id}"
            ) from exc
        values = mapped[topic_columns].to_numpy(dtype=np.float32)
        valid = (
            mapped["valid_topic"].astype(bool).to_numpy()
            if "valid_topic" in mapped
            else np.ones(len(mapped), dtype=bool)
        )

        story_metadata: list[dict[str, object]] = []
        story_distributions: list[np.ndarray] = []
        for spec in specs:
            draws = random_draws if spec.ordering == "random" else tie_draws
            draw_distributions = []
            valid_counts = []
            weight_sums = []
            total_weight_sums = []
            for draw in range(1, draws + 1):
                order = make_policy_order(group, spec, seed=seed, draw=draw)
                weights = _rank_attention_weights(
                    len(order),
                    rank_weight_power=rank_weight_power,
                    rank_weight_mode=rank_weight_mode,
                    rank_decay=rank_decay,
                    rank_weight_values=rank_weight_values,
                ).astype(np.float32)
                ordered_valid = valid[order]
                valid_weights = weights[ordered_valid]
                if not len(valid_weights) or valid_weights.sum() <= 0:
                    continue
                distribution = (
                    values[order][ordered_valid] * valid_weights[:, None]
                ).sum(axis=0)
                total = distribution.sum()
                if total <= 0 or not np.isfinite(total):
                    continue
                draw_distributions.append(distribution / total)
                valid_counts.append(int(ordered_valid.sum()))
                weight_sums.append(float(valid_weights.sum()))
                total_weight_sums.append(float(weights.sum()))
            if not draw_distributions:
                continue
            if retain_draws:
                for draw_index, (distribution, valid_count, weight_sum, total_weight_sum) in enumerate(
                    zip(draw_distributions, valid_counts, weight_sums, total_weight_sums),
                    start=1,
                ):
                    story_metadata.append({
                        "story_id": story_id,
                        "policy_id": spec.policy_id,
                        "ordering": spec.ordering,
                        "reply_mode": spec.reply_mode,
                        "pinned": spec.pinned,
                        "deployable": spec.deployable,
                        "draw": draw_index,
                        "n_documents": len(order),
                        "n_valid_documents": valid_count,
                        "rank_weight_sum": weight_sum,
                        "total_rank_weight_sum": total_weight_sum,
                        "assigned_exposure_coverage": weight_sum / total_weight_sum,
                        "ordering_draws": draws,
                        "rank_weight_mode": rank_weight_mode,
                        "rank_decay": rank_decay,
                    })
                    story_distributions.append(np.asarray(distribution, dtype=np.float32))
            else:
                distribution = np.asarray(draw_distributions, dtype=np.float32).mean(axis=0)
                distribution /= distribution.sum()
                story_metadata.append({
                    "story_id": story_id,
                    "policy_id": spec.policy_id,
                    "ordering": spec.ordering,
                    "reply_mode": spec.reply_mode,
                    "pinned": spec.pinned,
                    "deployable": spec.deployable,
                    "n_documents": len(order),
                    "n_valid_documents": int(np.mean(valid_counts)),
                    "rank_weight_sum": float(np.mean(weight_sums)),
                    "total_rank_weight_sum": float(np.mean(total_weight_sums)),
                    "assigned_exposure_coverage": float(
                        np.mean(weight_sums) / np.mean(total_weight_sums)
                    ),
                    "ordering_draws": draws,
                    "rank_weight_mode": rank_weight_mode,
                    "rank_decay": rank_decay,
                })
                story_distributions.append(distribution.astype(np.float32))
        return story_metadata, story_distributions

    metadata_rows: list[dict[str, object]] = []
    distribution_rows: list[np.ndarray] = []
    completed = 0
    if n_jobs == 1:
        results = map(process_story, story_groups)
    else:
        workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(process_story, story_groups)

    try:
        for story_metadata, story_distributions in results:
            metadata_rows.extend(story_metadata)
            distribution_rows.extend(story_distributions)
            completed += 1
            if progress and (
                completed == 1
                or completed % progress_every_stories == 0
                or completed == n_stories
            ):
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (n_stories - completed) / rate if rate > 0 else 0.0
                print(
                    f"[topic policies] {completed:,}/{n_stories:,} stories; "
                    f"{completed * n_policy_cells:,}/{n_stories * n_policy_cells:,} "
                    f"policy cells; {rate:.2f} stories/s; ETA {eta / 60:.1f} min.",
                    flush=True,
                )
    finally:
        if n_jobs != 1:
            executor.shutdown(wait=True)

    if not metadata_rows:
        raise ValueError("No policy topic distributions were produced")

    metadata = pd.DataFrame(metadata_rows)
    topics = pd.DataFrame(
        np.vstack(distribution_rows),
        columns=topic_columns,
        dtype=np.float32,
    )
    return pd.concat([metadata.reset_index(drop=True), topics], axis=1)

def bootstrap_topic_policy_effects(
    metrics: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = (
        "alignment_gain",
        "visible_minus_discussion_entropy",
        "hellinger_alpha",
        "hellinger_residual",
    ),
    bootstrap_draws: int = 2_000,
    seed: int = 2025,
    ordering_reference_reply_mode: str = "loose",
    ordering_reference_pinned: bool = False,
) -> pd.DataFrame:
    """Estimate paired average ordering, reply-mode, and pin effects.

    Ordering effects compare each substantive ordering with random under the
    same reply and pin conditions specified by the reference arguments. Reply
    effects compare trees or hidden with loose, and pin effects compare pinned
    with unpinned, averaging over the other factorial dimensions. All contrasts
    are paired within story.
    """
    from .forum_scores import CONTROL_ORDERING, SUBSTANTIVE_ORDERINGS

    required = {
        "story_id", "policy_id", "ordering", "reply_mode", "pinned",
        *metric_columns,
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"metrics is missing columns: {missing}")
    if bootstrap_draws < 0:
        raise ValueError("bootstrap_draws must be non-negative")
    if ordering_reference_reply_mode not in {"loose", "trees", "hidden"}:
        raise ValueError(
            "ordering_reference_reply_mode must be 'loose', 'trees', or 'hidden'"
        )

    metadata = (
        metrics[["policy_id", "ordering", "reply_mode", "pinned"]]
        .drop_duplicates()
        .set_index("policy_id")
    )
    policy_ids = metadata.index.tolist()
    definitions = []
    for ordering in SUBSTANTIVE_ORDERINGS:
        same_interface = (
            metadata["reply_mode"].eq(ordering_reference_reply_mode)
            & metadata["pinned"].eq(ordering_reference_pinned)
        )
        treatment = metadata["ordering"].eq(ordering).to_numpy() & same_interface.to_numpy()
        control = metadata["ordering"].eq(CONTROL_ORDERING).to_numpy() & same_interface.to_numpy()
        if not treatment.any() or not control.any():
            raise ValueError(
                "Ordering reference condition has no complete treatment/control cells: "
                f"reply_mode={ordering_reference_reply_mode!r}, "
                f"pinned={ordering_reference_pinned!r}"
            )
        weights = np.zeros(len(metadata), dtype=float)
        weights[treatment] = 1.0 / treatment.sum()
        weights[control] = -1.0 / control.sum()
        reference = (
            f"{CONTROL_ORDERING}__{ordering_reference_reply_mode}__"
            f"{'pinned' if ordering_reference_pinned else 'unpinned'}"
        )
        definitions.append(("ordering_vs_random", ordering, reference, weights))
    substantive = metadata["ordering"].ne(CONTROL_ORDERING).to_numpy()
    for reply_mode in ("trees", "hidden"):
        treatment = substantive & metadata["reply_mode"].eq(reply_mode).to_numpy()
        control = substantive & metadata["reply_mode"].eq("loose").to_numpy()
        weights = np.zeros(len(metadata), dtype=float)
        weights[treatment] = 1.0 / treatment.sum()
        weights[control] = -1.0 / control.sum()
        definitions.append(("reply_vs_loose", reply_mode, "loose", weights))
    treatment = substantive & metadata["pinned"].to_numpy(bool)
    control = substantive & ~metadata["pinned"].to_numpy(bool)
    weights = np.zeros(len(metadata), dtype=float)
    weights[treatment] = 1.0 / treatment.sum()
    weights[control] = -1.0 / control.sum()
    definitions.append(("pinned_vs_unpinned", "pinned", "unpinned", weights))

    rows = []
    for metric_number, metric in enumerate(metric_columns):
        panel = metrics.pivot(index="story_id", columns="policy_id", values=metric)
        panel = panel.reindex(columns=policy_ids).dropna(axis=0, how="any")
        if panel.empty:
            continue
        values = panel.to_numpy(dtype=float)
        for family, contrast, reference, weights in definitions:
            differences = values @ weights
            estimate = float(differences.mean())
            if bootstrap_draws:
                rng = np.random.default_rng(seed + metric_number * 1000)
                indices = rng.integers(
                    0, len(differences), size=(bootstrap_draws, len(differences))
                )
                boot = differences[indices].mean(axis=1)
                lower, upper = np.quantile(boot, [0.025, 0.975])
            else:
                lower = upper = np.nan
            rows.append({
                "metric": metric,
                "contrast_family": family,
                "contrast": contrast,
                "reference": reference,
                "n_stories": len(differences),
                "estimate": estimate,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
            })
    return pd.DataFrame(rows)

def build_rankings_from_scores(
    comments: pd.DataFrame,
    score_columns: Mapping[str, str],
    *,
    ascending: Iterable[str] = (),
    random_draws: int = 25,
    seed: int = 2025,
) -> pd.DataFrame:
    """Create the long ranking contract used by :func:`ranked_topic_trajectory`.

    ``score_columns`` maps a policy label to a comment column.  Scores are
    sorted descending unless the policy is listed in ``ascending``.  A policy
    named ``random`` is expanded into deterministic ``random_0001`` ... draws;
    this is important because shallow visible samples have a non-zero sampling
    variance even when the underlying discussion distribution is fixed.
    """

    required = {
        "story_id",
        "comment_id",
        *(score for policy, score in score_columns.items() if policy != "random" and score is not None),
    }
    missing = sorted(required - set(comments.columns))
    if missing:
        raise ValueError(f"comments is missing ranking columns: {missing}")
    if random_draws < 1:
        raise ValueError("random_draws must be at least 1")
    ascending = set(ascending)
    rng = np.random.default_rng(seed)
    rows = []
    for story_id, group in comments.groupby("story_id", sort=True):
        group = group.copy()
        group["comment_id"] = group["comment_id"].astype(str)
        for policy, score_column in score_columns.items():
            if policy == "random":
                for draw in range(random_draws):
                    order = rng.permutation(len(group))
                    ordered = group.iloc[order]
                    rows.extend(
                        {"story_id": story_id, "policy_id": f"random_{draw + 1:04d}", "comment_id": comment_id, "rank": rank}
                        for rank, comment_id in enumerate(ordered["comment_id"], start=1)
                    )
                continue
            if not np.isfinite(pd.to_numeric(group[score_column], errors="coerce")).all():
                raise ValueError(f"Ranking column {score_column!r} contains non-finite values")
            ordered = group.assign(_score=pd.to_numeric(group[score_column], errors="raise"))
            ordered = ordered.sort_values(
                ["_score", "comment_id"],
                ascending=[policy in ascending, True],
                kind="mergesort",
            )
            rows.extend(
                {"story_id": story_id, "policy_id": policy, "comment_id": comment_id, "rank": rank}
                for rank, comment_id in enumerate(ordered["comment_id"], start=1)
            )
    return pd.DataFrame(rows, columns=["story_id", "policy_id", "comment_id", "rank"])

def aggregate_topic_policy_draw_distributions(
    visible_draws: pd.DataFrame,
) -> pd.DataFrame:
    """Average topic distributions across draws for compatibility summaries.

    This helper is deliberately separate from metric aggregation. Primary
    nonlinear metrics should be calculated from each draw first, then averaged
    with aggregate_topic_policy_draw_metrics.
    """

    topic_columns = _topic_columns(visible_draws)
    required = {"story_id", "policy_id", *topic_columns}
    missing = sorted(required - set(visible_draws.columns))
    if missing:
        raise ValueError(f"Visible draws are missing columns: {missing}")
    group_columns = ["story_id", "policy_id"]
    metadata_columns = [
        column for column in (
            "ordering",
            "reply_mode",
            "pinned",
            "deployable",
            "n_documents",
            "n_valid_documents",
            "rank_weight_sum",
            "total_rank_weight_sum",
            "assigned_exposure_coverage",
            "ordering_draws",
            "rank_weight_mode",
            "rank_decay",
        )
        if column in visible_draws.columns
    ]
    grouped = visible_draws.groupby(group_columns, sort=True, dropna=False)
    result = grouped[topic_columns].mean().reset_index()
    result["n_draws"] = grouped.size().to_numpy()
    for column in metadata_columns:
        if pd.api.types.is_numeric_dtype(visible_draws[column]):
            result[column] = grouped[column].mean().to_numpy()
        else:
            result[column] = grouped[column].first().to_numpy()
    return result[
        [*group_columns, *metadata_columns, "n_draws", *topic_columns]
    ]

def aggregate_topic_policy_draw_metrics(
    metrics_draws: pd.DataFrame,
) -> pd.DataFrame:
    """Average nonlinear policy metrics over realized presentation draws."""

    required = {"story_id", "policy_id"}
    missing = sorted(required - set(metrics_draws.columns))
    if missing:
        raise ValueError(f"Draw metrics are missing columns: {missing}")
    group_columns = ["story_id", "policy_id"]
    metadata_columns = [
        column for column in ("ordering", "reply_mode", "pinned", "deployable")
        if column in metrics_draws.columns
    ]
    numeric_columns = [
        column for column in metrics_draws.select_dtypes(include=[np.number]).columns
        if column not in {"story_id", "policy_id", "draw"}
    ]
    grouped = metrics_draws.groupby(group_columns, sort=True, dropna=False)
    result = grouped[numeric_columns].mean().reset_index()
    result["n_draws"] = grouped.size().to_numpy()
    for column in metadata_columns:
        result[column] = grouped[column].first().to_numpy()
    if "hellinger_alpha" in result.columns:
        result["projection_class"] = result["hellinger_alpha"].map(classify_projection)
    return result

def paired_policy_contrasts(
    metrics: pd.DataFrame,
    *,
    metric: str,
    policy_column: str = "policy_id",
    control_policy: str = "random",
    depth_column: str = "depth",
    story_column: str = "story_id",
    bootstrap_draws: int = 2_000,
    seed: int = 2025,
) -> pd.DataFrame:
    """Estimate paired policy-minus-control contrasts with percentile intervals.

    Depth is optional for one-distribution analyses. If present, contrasts
    are calculated separately at each depth; otherwise they are calculated
    once per policy.
    """

    has_depth = depth_column in metrics.columns
    required = {story_column, policy_column, metric}
    if has_depth:
        required.add(depth_column)
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"metrics is missing columns: {missing}")
    if bootstrap_draws < 0:
        raise ValueError("bootstrap_draws must be non-negative")

    grouping = [story_column, depth_column] if has_depth else [story_column]
    policy_grouping = [policy_column, depth_column] if has_depth else [policy_column]
    control = (
        metrics[metrics[policy_column].eq(control_policy)]
        .groupby(grouping, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "control"})
    )
    rows = []
    rng = np.random.default_rng(seed)
    for policy_keys, group in metrics[~metrics[policy_column].eq(control_policy)].groupby(policy_grouping, sort=True):
        if has_depth:
            policy, depth = policy_keys
            control_group = control[control[depth_column].eq(depth)]
        else:
            policy = policy_keys
            depth = None
            control_group = control
        policy_values = group.groupby(story_column, as_index=False)[metric].mean()
        paired = policy_values.merge(control_group, on=story_column, how="inner")
        differences = (paired[metric] - paired["control"]).dropna().to_numpy(dtype=float)
        if not len(differences):
            continue
        if bootstrap_draws:
            draws = rng.choice(differences, size=(bootstrap_draws, len(differences)), replace=True).mean(axis=1)
            ci_lower, ci_upper = np.quantile(draws, [0.025, 0.975])
        else:
            ci_lower = ci_upper = np.nan
        row = {
            policy_column: policy,
            "metric": metric,
            "n_stories": len(differences),
            "estimate": float(differences.mean()),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "control_policy": control_policy,
        }
        if has_depth:
            row[depth_column] = depth
        rows.append(row)
    return pd.DataFrame(rows)

def build_structured_oracle_orders(
    group: pd.DataFrame,
    orderings: Iterable[str],
) -> dict[str, np.ndarray]:
    """Build deterministic score/chronology starts for oracle searches."""

    indices = np.arange(len(group), dtype=int)
    orders: dict[str, np.ndarray] = {}
    for ordering in orderings:
        if ordering == "relative_votes":
            values = pd.to_numeric(group["relative_votes"], errors="raise").to_numpy(float)
            order = np.argsort(-values, kind="stable")
        elif ordering == "upvotes":
            values = pd.to_numeric(group["votes_positive"], errors="raise").to_numpy(float)
            order = np.argsort(-values, kind="stable")
        elif ordering in {"chronological", "reverse_chronological"}:
            timestamps = pd.to_datetime(
                group["created_at"], utc=True, errors="raise"
            ).astype("int64").to_numpy()
            primary = -timestamps if ordering == "reverse_chronological" else timestamps
            display = pd.to_numeric(group["display_order"], errors="raise").to_numpy()
            order = np.lexsort((indices, display, primary))
        else:
            column = f"{ordering}_score"
            values = pd.to_numeric(group[column], errors="raise").to_numpy(float)
            order = np.argsort(-values, kind="stable")
        orders[ordering] = order
    return orders

def run_oracle_benchmark(
    memberships: pd.DataFrame,
    comments: pd.DataFrame,
    article_distributions: pd.DataFrame,
    discussion_distributions: pd.DataFrame,
    topic_columns: Sequence[str],
    *,
    policy_id: str,
    ordering: str,
    objective: str,
    visible_path: Path | str,
    metrics_path: Path | str,
    summary_path: Path | str,
    skipped_path: Path | str,
    exact_validation_path: Path | str,
    structured_start_policies: Sequence[str],
    force_recompute: bool = False,
    max_iterations: int = 5,
    n_starts: int = 8,
    random_state: int = 2025,
    n_perturbations: int = 4,
    perturbation_fraction: float = 0.10,
    n_local_moves: int = 2,
    rank_weight_power: float = 1.0,
    rank_weight_mode: str = "inverse_power",
    rank_decay: float = 0.1,
    rank_weight_values: Sequence[float] | None = None,
    allow_invalid_placement: bool = True,
    exact_max_valid_comments: int = 8,
    progress_every: int = 25,
    label: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit, cache, and validate one article-aware oracle benchmark.

    The oracle uses the observed comments and article topic distribution for
    each story, so it is an upper-bound/normalisation benchmark rather than a
    deployable ranking policy.  This function owns the complete notebook
    workflow: cache validation, multi-start search, progress logging, metric
    calculation, summary writing, and exact enumeration on small stories.
    """

    if objective not in {
        "hellinger_distance",
        "jensen_shannon_distance",
        "cosine_similarity",
    }:
        raise ValueError(
            "objective must be 'hellinger_distance', "
            "'jensen_shannon_distance', or 'cosine_similarity'"
        )
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    _rank_attention_weights(
        1,
        rank_weight_power=rank_weight_power,
        rank_weight_mode=rank_weight_mode,
        rank_decay=rank_decay,
        rank_weight_values=rank_weight_values,
    )
    topic_columns = list(topic_columns)
    visible_path = Path(visible_path)
    metrics_path = Path(metrics_path)
    summary_path = Path(summary_path)
    skipped_path = Path(skipped_path)
    exact_validation_path = Path(exact_validation_path)
    cache_manifest_path = visible_path.with_suffix(".manifest.json")
    membership_cache_columns = [
        column for column in ["story_id", "comment_id", "doc_type", "valid_topic", *topic_columns]
        if column in memberships.columns
    ]
    input_parts = [
        _frame_fingerprint(memberships, membership_cache_columns),
        _frame_fingerprint(comments),
        _frame_fingerprint(article_distributions),
        _frame_fingerprint(discussion_distributions),
    ]
    input_fingerprint = hashlib.sha256("|".join(input_parts).encode()).hexdigest()
    cache_signature = {
        "cache_version": 3,
        "code_revision": _current_code_revision(),
        "code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("topic_modeling.py", "topic_metrics.py", "forum_scores.py")
        },
        "input_fingerprint": input_fingerprint,
        "topic_columns": list(topic_columns),
        "policy_id": policy_id,
        "ordering": ordering,
        "objective": objective,
        "max_iterations": int(max_iterations),
        "n_starts": int(n_starts),
        "random_state": int(random_state),
        "n_perturbations": int(n_perturbations),
        "perturbation_fraction": float(perturbation_fraction),
        "n_local_moves": int(n_local_moves),
        "rank_weight_power": float(rank_weight_power),
        "rank_weight_values": None if rank_weight_values is None else list(map(float, rank_weight_values)),
        "allow_invalid_placement": bool(allow_invalid_placement),
        "exact_max_valid_comments": int(exact_max_valid_comments),
        "structured_start_policies": list(structured_start_policies),
    }
    if rank_weight_mode != "inverse_power" or rank_decay != 0.1:
        cache_signature.update({
            "rank_weight_mode": rank_weight_mode,
            "rank_decay": float(rank_decay),
        })
    for path in (
        visible_path,
        metrics_path,
        summary_path,
        skipped_path,
        exact_validation_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    structured_start_policies = tuple(structured_start_policies)
    config_columns = {
        "oracle_objective",
        "oracle_n_starts",
        "oracle_random_state",
        "oracle_n_perturbations",
        "oracle_perturbation_fraction",
        "oracle_n_local_moves",
        "oracle_structured_start_policies",
        "oracle_allow_invalid_placement",
        "oracle_policy_id",
        "oracle_ordering",
        "oracle_max_iterations",
        "oracle_rank_weight_power",
        "oracle_exact_max_valid_comments",
        "oracle_input_fingerprint",
        "oracle_topic_columns",
        "oracle_code_hash",
    }
    if rank_weight_mode != "inverse_power" or rank_decay != 0.1:
        config_columns.update({"oracle_rank_weight_mode", "oracle_rank_decay"})

    def cache_is_valid() -> bool:
        if force_recompute or not all(
            path.exists()
            for path in (
                visible_path,
                metrics_path,
                summary_path,
                skipped_path,
                exact_validation_path,
            )
        ):
            return False
        try:
            if not cache_manifest_path.exists():
                return False
            cached_manifest = json.loads(cache_manifest_path.read_text())
            if cached_manifest != cache_signature:
                return False
            cached = pd.read_parquet(visible_path)
            if not config_columns.issubset(cached.columns) or cached.empty:
                return False
            return (
                cached["oracle_objective"].eq(objective).all()
                and cached["oracle_n_starts"].eq(n_starts).all()
                and cached["oracle_random_state"].eq(random_state).all()
                and cached["oracle_n_perturbations"].eq(n_perturbations).all()
                and cached["oracle_perturbation_fraction"].eq(perturbation_fraction).all()
                and cached["oracle_n_local_moves"].eq(n_local_moves).all()
                and cached["oracle_structured_start_policies"].eq(
                    "|".join(structured_start_policies)
                ).all()
                and cached["oracle_policy_id"].eq(policy_id).all()
                and cached["oracle_ordering"].eq(ordering).all()
                and cached["oracle_max_iterations"].eq(max_iterations).all()
                and cached["oracle_rank_weight_power"].eq(rank_weight_power).all()
                and (
                    (
                        rank_weight_mode == "inverse_power"
                        and rank_decay == 0.1
                    )
                    or (
                        cached["oracle_rank_weight_mode"].eq(rank_weight_mode).all()
                        and cached["oracle_rank_decay"].eq(rank_decay).all()
                    )
                )
                and cached["oracle_exact_max_valid_comments"].eq(
                    exact_max_valid_comments
                ).all()
                and cached["oracle_input_fingerprint"].eq(input_fingerprint).all()
                and cached["oracle_topic_columns"].eq("|".join(topic_columns)).all()
                and cached["oracle_code_hash"].eq(cache_signature["code_hash"]).all()
            )
        except (KeyError, ValueError, OSError, TypeError):
            return False

    if cache_is_valid():
        return {
            "visible": pd.read_parquet(visible_path),
            "metrics": pd.read_parquet(metrics_path),
            "summary": pd.read_csv(summary_path),
            "skipped": pd.read_csv(skipped_path),
            "exact_validation": pd.read_csv(exact_validation_path),
        }

    membership_frame = memberships.loc[
        memberships["doc_type"].eq("comment")
    ].copy()
    membership_frame["story_id"] = membership_frame["story_id"].astype(str)
    membership_frame["comment_id"] = membership_frame["comment_id"].astype(str)
    membership_index = membership_frame.set_index(
        ["story_id", "comment_id"], drop=False
    )
    if membership_index.index.duplicated().any():
        raise ValueError("Comment topic memberships must have unique story_id/comment_id keys")

    comments_frame = comments.copy()
    comments_frame["story_id"] = comments_frame["story_id"].astype(str)
    comments_frame["comment_id"] = comments_frame["comment_id"].astype(str)
    article_frame = article_distributions.copy()
    article_frame["story_id"] = article_frame["story_id"].astype(str)
    discussion_frame = discussion_distributions.copy()
    discussion_frame["story_id"] = discussion_frame["story_id"].astype(str)
    article_lookup = article_frame.set_index("story_id")
    discussion_ids = set(discussion_frame["story_id"])
    grouped_comments = comments_frame.groupby("story_id", sort=True)
    total_stories = comments_frame["story_id"].nunique()
    display_label = label or ordering
    print(
        f"Starting {display_label} oracle search: {total_stories:,} stories.",
        flush=True,
    )

    oracle_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, str]] = []
    for story_number, (story_id, group) in enumerate(grouped_comments, start=1):
        story_id = str(story_id)
        if story_id not in article_lookup.index or story_id not in discussion_ids:
            skipped_rows.append({
                "story_id": story_id,
                "reason": "missing article or discussion distribution",
            })
            continue
        group = group.reset_index(drop=True)
        structured_starts = build_structured_oracle_orders(
            group, structured_start_policies
        )
        keys = [(story_id, str(comment_id)) for comment_id in group["comment_id"]]
        try:
            mapped = membership_index.loc[
                keys, topic_columns + ["valid_topic"]
            ]
        except KeyError:
            skipped_rows.append({
                "story_id": story_id,
                "reason": "comment without topic membership",
            })
            continue
        try:
            oracle = hellinger_oracle_order(
                article_lookup.loc[story_id, topic_columns].to_numpy(dtype=float),
                mapped[topic_columns].to_numpy(dtype=float),
                valid_mask=mapped["valid_topic"].to_numpy(dtype=bool),
                rank_weight_power=rank_weight_power,
                allow_invalid_placement=allow_invalid_placement,
                max_iterations=max_iterations,
                objective=objective,
                n_starts=n_starts,
                random_state=random_state,
                structured_starts=structured_starts,
                n_perturbations=n_perturbations,
                perturbation_fraction=perturbation_fraction,
                n_local_moves=n_local_moves,
                rank_weight_mode=rank_weight_mode,
                rank_decay=rank_decay,
                rank_weight_values=rank_weight_values,
            )
        except ValueError as exc:
            skipped_rows.append({"story_id": story_id, "reason": str(exc)})
            continue

        order = np.asarray(oracle["order"], dtype=int)
        valid = mapped["valid_topic"].to_numpy(dtype=bool)
        weights = _rank_attention_weights(
            len(order),
            rank_weight_power=rank_weight_power,
            rank_weight_mode=rank_weight_mode,
            rank_decay=rank_decay,
            rank_weight_values=rank_weight_values,
        )
        row: dict[str, object] = {
            "story_id": story_id,
            "policy_id": policy_id,
            "ordering": ordering,
            "reply_mode": "loose",
            "pinned": False,
            "deployable": False,
            "n_documents": len(order),
            "n_valid_documents": int(valid[order].sum()),
            "rank_weight_sum": float(weights[valid[order]].sum()),
            "oracle_objective": objective,
            "oracle_policy_id": policy_id,
            "oracle_ordering": ordering,
            "oracle_max_iterations": int(max_iterations),
            "oracle_rank_weight_power": float(rank_weight_power),
            "oracle_rank_weight_mode": rank_weight_mode,
            "oracle_rank_decay": float(rank_decay),
            "oracle_exact_max_valid_comments": int(exact_max_valid_comments),
            "oracle_input_fingerprint": input_fingerprint,
            "oracle_topic_columns": "|".join(topic_columns),
            "oracle_code_hash": cache_signature["code_hash"],
            "oracle_n_starts": int(oracle["n_starts"]),
            "oracle_random_state": int(oracle["random_state"]),
            "oracle_n_perturbations": int(oracle["n_perturbations"]),
            "oracle_perturbation_fraction": float(oracle["perturbation_fraction"]),
            "oracle_n_local_moves": int(oracle["n_local_moves"]),
            "oracle_structured_start_policies": "|".join(structured_start_policies),
            "oracle_allow_invalid_placement": bool(allow_invalid_placement),
            "oracle_best_start": int(oracle["best_start"]),
            "oracle_best_start_label": str(oracle["best_start_label"]),
            "oracle_best_perturbation": int(oracle["best_perturbation"]),
            "oracle_improved_starts": int(oracle["improved_starts"]),
            "oracle_accepted_perturbations": int(oracle["accepted_perturbations"]),
            "oracle_local_moves_attempted": int(oracle["local_moves_attempted"]),
            "oracle_total_iterations": int(oracle["total_iterations"]),
            "oracle_initial_hellinger_distance": float(
                oracle["initial_hellinger_distance"]
            ),
            "oracle_hellinger_distance": float(oracle["hellinger_distance"]),
            "oracle_iterations": int(oracle["iterations"]),
        }
        row.update(dict(zip(topic_columns, np.asarray(oracle["distribution"], dtype=float))))
        oracle_rows.append(row)
        if (
            story_number == 1
            or story_number % progress_every == 0
            or story_number == total_stories
        ):
            print(
                f"[{display_label}] {story_number:,}/{total_stories:,} stories; "
                f"usable={len(oracle_rows):,}; skipped={len(skipped_rows):,}",
                flush=True,
            )

    visible = pd.DataFrame(
        oracle_rows,
        columns=[
            "story_id", "policy_id", "ordering", "reply_mode", "pinned",
            "deployable", "n_documents", "n_valid_documents", "rank_weight_sum",
            "oracle_objective", "oracle_n_starts", "oracle_random_state",
            "oracle_n_perturbations", "oracle_perturbation_fraction",
            "oracle_n_local_moves", "oracle_structured_start_policies",
            "oracle_allow_invalid_placement",
            "oracle_policy_id", "oracle_ordering", "oracle_max_iterations",
            "oracle_rank_weight_power", "oracle_rank_weight_mode", "oracle_rank_decay",
            "oracle_exact_max_valid_comments",
            "oracle_input_fingerprint", "oracle_topic_columns", "oracle_code_hash",
            "oracle_best_start", "oracle_best_start_label",
            "oracle_best_perturbation", "oracle_improved_starts",
            "oracle_accepted_perturbations", "oracle_local_moves_attempted",
            "oracle_total_iterations", "oracle_initial_hellinger_distance",
            "oracle_hellinger_distance", "oracle_iterations", *topic_columns,
        ],
    )
    if visible.empty:
        raise ValueError(f"The {display_label} oracle produced no usable stories")
    skipped = pd.DataFrame(skipped_rows, columns=["story_id", "reason"])
    # An interrupted rewrite must not leave an old completion signature valid.
    cache_manifest_path.unlink(missing_ok=True)
    visible.to_parquet(visible_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    oracle_metrics = compute_topic_metrics(
        article_frame,
        discussion_frame,
        visible,
    )
    oracle_metrics.to_parquet(metrics_path, index=False)

    summary_row: dict[str, object] = {
        "policy_id": policy_id,
        "n_stories": len(oracle_metrics),
    }
    summary_metrics = [
        "article_visible_hellinger_distance",
        "article_visible_js_distance",
        "alignment_gain",
        "hellinger_alpha",
        "cosine_progress",
    ]
    for metric in summary_metrics:
        if metric not in oracle_metrics:
            continue
        values = oracle_metrics[metric].dropna()
        summary_row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
        summary_row[f"{metric}_sd"] = float(values.std()) if len(values) > 1 else np.nan
        summary_row[f"{metric}_q10"] = float(values.quantile(0.10)) if len(values) else np.nan
        summary_row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
        summary_row[f"{metric}_q90"] = float(values.quantile(0.90)) if len(values) else np.nan
    summary = pd.DataFrame([summary_row])
    summary.to_csv(summary_path, index=False)

    exact_membership_index = membership_index
    exact_article_lookup = article_frame.set_index("story_id")
    exact_rows: list[dict[str, object]] = []
    for exact_story_id, exact_group in comments_frame.groupby("story_id", sort=True):
        exact_story_id = str(exact_story_id)
        if exact_story_id not in exact_article_lookup.index:
            continue
        exact_group = exact_group.reset_index(drop=True)
        exact_keys = [
            (exact_story_id, str(comment_id))
            for comment_id in exact_group["comment_id"]
        ]
        try:
            exact_mapped = exact_membership_index.loc[
                exact_keys, topic_columns + ["valid_topic"]
            ]
        except KeyError:
            continue
        exact_valid = exact_mapped["valid_topic"].to_numpy(dtype=bool).copy()
        exact_values = exact_mapped[topic_columns].to_numpy(dtype=float)
        exact_valid &= exact_values.sum(axis=1) > np.finfo(float).eps
        n_exact_valid = int(exact_valid.sum())
        n_exact_comments = len(exact_values) if allow_invalid_placement else n_exact_valid
        if n_exact_valid < 1 or n_exact_comments > exact_max_valid_comments:
            continue
        exact_result = exact_hellinger_oracle_order(
            exact_article_lookup.loc[
                exact_story_id, topic_columns
            ].to_numpy(dtype=float),
            exact_values,
            valid_mask=exact_valid,
            rank_weight_power=rank_weight_power,
            rank_weight_mode=rank_weight_mode,
            rank_decay=rank_decay,
            rank_weight_values=rank_weight_values,
            objective=objective,
            max_valid_comments=exact_max_valid_comments,
            allow_invalid_placement=allow_invalid_placement,
        )
        heuristic_match = oracle_metrics.loc[
            oracle_metrics["story_id"].astype(str).eq(exact_story_id)
        ]
        if len(heuristic_match) != 1:
            continue
        if objective == "jensen_shannon_distance":
            metric_column = "article_visible_js_distance"
            exact_distance = float(exact_result["jensen_shannon_distance"])
        else:
            metric_column = "article_visible_hellinger_distance"
            exact_distance = float(exact_result["hellinger_distance"])
        heuristic_distance = float(heuristic_match.iloc[0][metric_column])
        exact_rows.append({
            "story_id": exact_story_id,
            "n_valid_comments": n_exact_valid,
            "n_comments": n_exact_comments,
            "n_permutations": int(exact_result["n_permutations"]),
            "heuristic_distance": heuristic_distance,
            "exact_distance": exact_distance,
            "distance_gap": heuristic_distance - exact_distance,
            "heuristic_cosine": float(
                heuristic_match.iloc[0]["article_visible_cosine"]
            ),
            "exact_cosine": float(exact_result["cosine_similarity"]),
            "exact_match": bool(
                abs(heuristic_distance - exact_distance) <= 1e-10
            ),
        })
    exact_validation = pd.DataFrame(exact_rows)
    if exact_validation.empty:
        exact_validation = pd.DataFrame(columns=[
            "story_id", "n_valid_comments", "n_comments", "n_permutations",
            "heuristic_distance", "exact_distance", "distance_gap",
            "heuristic_cosine", "exact_cosine", "exact_match",
        ])
    exact_validation.to_csv(exact_validation_path, index=False)
    cache_manifest_path.write_text(
        json.dumps(cache_signature, indent=2, sort_keys=True) + "\n"
    )
    return {
        "visible": visible,
        "metrics": oracle_metrics,
        "summary": summary,
        "skipped": skipped,
        "exact_validation": exact_validation,
    }

def build_policy_contrast_order(
    score_policies: Mapping[str, object],
    *,
    control_ordering: str = "random",
) -> list[tuple[str, str]]:
    """Return the primary-ordering, reply-status, and pin-status contrasts."""

    orderings = [ordering for ordering in score_policies if ordering != control_ordering]
    return [
        *[("ordering_vs_random", ordering) for ordering in orderings],
        ("reply_vs_loose", "trees"),
        ("reply_vs_loose", "hidden"),
        ("pinned_vs_unpinned", "pinned"),
    ]

def _policy_contrast_mask(
    frame: pd.DataFrame,
    family: str,
    contrast: str,
    *,
    control_ordering: str = "random",
    reference_reply_mode: str = "loose",
    reference_pinned: bool = False,
) -> pd.Series:
    if family == "ordering_vs_random":
        return (
            frame["ordering"].eq(contrast)
            & frame["reply_mode"].eq(reference_reply_mode)
            & frame["pinned"].eq(reference_pinned)
        )
    if family == "reply_vs_loose":
        return frame["ordering"].ne(control_ordering) & frame["reply_mode"].eq(contrast)
    if family == "pinned_vs_unpinned":
        return frame["ordering"].ne(control_ordering) & frame["pinned"].eq(True)
    raise ValueError(f"Unknown contrast family: {family}")

def _policy_contrast_values(
    frame: pd.DataFrame,
    family: str,
    contrast: str,
    value_column: str,
    *,
    control_ordering: str = "random",
    reference_reply_mode: str = "loose",
    reference_pinned: bool = False,
) -> pd.DataFrame:
    """Return story-level treatment, control, and paired differences."""

    required = {"story_id", "ordering", "reply_mode", "pinned", value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Policy frame is missing columns: {missing}")
    working = frame.copy()
    working["story_id"] = working["story_id"].astype(str)
    if family == "ordering_vs_random":
        # Match notebook 11: average each ordering over its six reply/pin
        # variants within story, then subtract the corresponding random mean.
        # Keeping the aggregation story-level preserves the paired comparison.
        keys = ["story_id"]
        treatment = (
            working.loc[
                working["ordering"].eq(contrast),
                keys + [value_column],
            ]
            .groupby(keys, as_index=False, sort=False)[value_column]
            .mean()
            .rename(columns={value_column: "treatment"})
        )
        control = (
            working.loc[
                working["ordering"].eq(control_ordering),
                keys + [value_column],
            ]
            .groupby(keys, as_index=False, sort=False)[value_column]
            .mean()
            .rename(columns={value_column: "control"})
        )
        joined = treatment.merge(control, on=keys, how="inner", validate="one_to_one")
        return joined.assign(difference=joined["treatment"] - joined["control"])
    if family == "reply_vs_loose":
        pair_keys = ["story_id", "ordering", "pinned"]
        treatment = working.loc[
            working["ordering"].ne(control_ordering)
            & working["reply_mode"].eq(contrast),
            pair_keys + [value_column],
        ].rename(columns={value_column: "treatment"})
        control = working.loc[
            working["ordering"].ne(control_ordering)
            & working["reply_mode"].eq(reference_reply_mode),
            pair_keys + [value_column],
        ].rename(columns={value_column: "control"})
        joined = treatment.merge(control, on=pair_keys, how="inner", validate="one_to_one")
        joined["difference"] = joined["treatment"] - joined["control"]
        return joined.groupby("story_id", as_index=False)[["treatment", "control", "difference"]].mean()
    if family == "pinned_vs_unpinned":
        pair_keys = ["story_id", "ordering", "reply_mode"]
        treatment = working.loc[
            working["ordering"].ne(control_ordering) & working["pinned"].eq(True),
            pair_keys + [value_column],
        ].rename(columns={value_column: "treatment"})
        control = working.loc[
            working["ordering"].ne(control_ordering) & working["pinned"].eq(False),
            pair_keys + [value_column],
        ].rename(columns={value_column: "control"})
        joined = treatment.merge(control, on=pair_keys, how="inner", validate="one_to_one")
        joined["difference"] = joined["treatment"] - joined["control"]
        return joined.groupby("story_id", as_index=False)[["treatment", "control", "difference"]].mean()
    raise ValueError(f"Unknown contrast family: {family}")

def _policy_contrast_label(
    family: str,
    contrast: str,
    *,
    ordering_labels: Mapping[str, str],
    reply_labels: Mapping[str, str],
) -> str:
    if family == "ordering_vs_random":
        return ordering_labels.get(contrast, contrast)
    if family == "reply_vs_loose":
        return {"trees": "Reply trees", "hidden": "Replies hidden"}.get(
            contrast, contrast
        )
    return "Pinned Picks"

def _mean_interval(
    values: Iterable[float],
    *,
    bootstrap_draws: int | None = None,
    seed: int | None = None,
) -> tuple[float, float, float, int]:
    """Return a mean and a paired story-bootstrap percentile interval.

    The topic-policy reporting plots use the same 2,000-draw paired
    discussion bootstrap as notebook 11. ``values`` must already represent
    one paired contrast value per story; resampling those values preserves
    the pairing between treatment and control before the contrast was formed.
    """
    array = pd.Series(values, dtype=float).dropna().to_numpy()
    if not len(array):
        return np.nan, np.nan, np.nan, 0
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean, 1
    if bootstrap_draws is None or seed is None:
        from .forum_scores import DEFAULT_BOOTSTRAP_DRAWS, DEFAULT_SEED

        bootstrap_draws = DEFAULT_BOOTSTRAP_DRAWS
        seed = DEFAULT_SEED
    if bootstrap_draws < 1:
        return mean, np.nan, np.nan, len(array)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(bootstrap_draws, len(array)))
    boot_means = array[indices].mean(axis=1)
    lower, upper = np.quantile(boot_means, [0.025, 0.975])
    return mean, float(lower), float(upper), len(array)

def _metric_effect_table(
    metrics: pd.DataFrame,
    *,
    metric: str,
    oracle_frame: pd.DataFrame,
    contrast_order: Sequence[tuple[str, str]],
    ordering_labels: Mapping[str, str],
    reply_labels: Mapping[str, str],
    control_ordering: str = "random",
    reference_reply_mode: str = "loose",
    reference_pinned: bool = False,
) -> pd.DataFrame:
    random_reference = metrics.loc[
        metrics["policy_id"].eq(
            f"{control_ordering}__{reference_reply_mode}__"
            f"{'pinned' if reference_pinned else 'unpinned'}"
        ),
        ["story_id", metric],
    ].copy()
    random_reference["story_id"] = random_reference["story_id"].astype(str)
    random_reference = random_reference.set_index("story_id")[metric]

    oracle_reference = oracle_frame.loc[:, ["story_id", metric]].copy()
    oracle_reference["story_id"] = oracle_reference["story_id"].astype(str)
    oracle_reference = oracle_reference.set_index("story_id")[metric]

    records = []
    working = metrics.copy()
    working["story_id"] = working["story_id"].astype(str)
    for family, contrast in contrast_order:
        paired = _policy_contrast_values(
            working,
            family,
            contrast,
            metric,
            control_ordering=control_ordering,
            reference_reply_mode=reference_reply_mode,
            reference_pinned=reference_pinned,
        )
        paired = paired.merge(
            oracle_reference.rename("oracle"),
            left_on="story_id",
            right_index=True,
            how="inner",
        )
        paired = paired.merge(
            random_reference.rename("random"),
            left_on="story_id",
            right_index=True,
            how="inner",
        )
        effect = paired["difference"]
        absolute_values = paired["treatment"] if family == "ordering_vs_random" else effect
        absolute_mean, absolute_lower, absolute_upper, absolute_n = _mean_interval(absolute_values)
        relative_mean, relative_lower, relative_upper, relative_n = _mean_interval(effect)
        denominator = paired["oracle"] - paired["random"]
        normalized = (effect / denominator).where(
            denominator.abs() > np.finfo(float).eps
        )
        normalized_mean, normalized_lower, normalized_upper, normalized_n = _mean_interval(normalized)
        control_mean, control_lower, control_upper, control_n = _mean_interval(paired["control"])
        zero_denominator = denominator.abs() <= np.finfo(float).eps
        records.append({
            "family": family,
            "contrast": contrast,
            "label": _policy_contrast_label(
                family,
                contrast,
                ordering_labels=ordering_labels,
                reply_labels=reply_labels,
            ),
            "absolute_mean": absolute_mean,
            "absolute_lower": absolute_lower,
            "absolute_upper": absolute_upper,
            "absolute_n": absolute_n,
            "control_mean": control_mean,
            "control_lower": control_lower,
            "control_upper": control_upper,
            "control_n": control_n,
            "relative_mean": relative_mean,
            "relative_lower": relative_lower,
            "relative_upper": relative_upper,
            "relative_n": relative_n,
            "normalized_mean": normalized_mean,
            "normalized_lower": normalized_lower,
            "normalized_upper": normalized_upper,
            "normalized_n": normalized_n,
            "n_joined": int(len(paired)),
            "n_excluded_zero_denominator": int(zero_denominator.sum()),
            "excluded_zero_denominator_fraction": (
                float(zero_denominator.mean()) if len(paired) else np.nan
            ),
        })
    result = pd.DataFrame(records)
    result["contrast_index"] = pd.Categorical(
        list(zip(result["family"], result["contrast"])),
        categories=list(contrast_order),
        ordered=True,
    )
    return result.sort_values("contrast_index").reset_index(drop=True)

def build_policy_reference_target_metrics(
    visible_distributions: pd.DataFrame,
    article_distributions: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    topic_columns: Sequence[str],
    relative_votes_policy_id: str = "relative_votes__loose__unpinned",
    random_policy_id: str = "random__loose__unpinned",
    chunk_size: int = 25_000,
) -> pd.DataFrame:
    """Build per-story raw gains while bounding peak memory use.

    Draw-level metrics are calculated before averaging within story and policy.
    Gains subtract each policy's expected metric from the expected metric of
    random presentations, not the metric of their mean distribution. The
    relative-votes target is its mean distribution across tie draws; the
    directional-cosine diagnostic uses the mean random distribution as origin.
    Policies and draw rows are processed in chunks so the full topic matrix is
    never merged with all three target matrices at once.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    topic_columns = list(topic_columns)
    draw_mode = "draw" in visible_distributions.columns

    def prefixed(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        result = frame[["story_id", *topic_columns]].copy()
        result["story_id"] = result["story_id"].astype(str)
        return result.rename(columns={column: f"{prefix}{column}" for column in topic_columns})

    def normalise(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        totals = values.sum(axis=1, keepdims=True)
        return np.divide(
            values,
            totals,
            out=np.zeros_like(values),
            where=totals > np.finfo(float).eps,
        )

    article_frame = prefixed(article_distributions, "article_")
    random_source = visible_distributions.loc[
        visible_distributions["policy_id"].eq(random_policy_id),
        ["story_id", *topic_columns],
    ]
    relative_source = visible_distributions.loc[
        visible_distributions["policy_id"].eq(relative_votes_policy_id),
        ["story_id", *topic_columns],
    ]
    if draw_mode:
        random_source = random_source.groupby("story_id", as_index=False)[topic_columns].mean()
        relative_source = relative_source.groupby("story_id", as_index=False)[topic_columns].mean()
    random_frame = prefixed(random_source, "random_")
    relative_votes_frame = prefixed(relative_source, "relative_votes_")
    if random_frame.empty:
        raise ValueError("Missing random loose/unpinned reference distribution")
    if relative_votes_frame.empty:
        raise ValueError("Missing relative-votes loose/unpinned reference distribution")

    reference_frame = (
        article_frame
        .merge(random_frame, on="story_id", how="inner")
        .merge(relative_votes_frame, on="story_id", how="inner")
        .drop_duplicates("story_id")
    )
    if reference_frame.empty:
        raise ValueError("No stories have all article, random, and relative-votes distributions")
    reference_frame = reference_frame.set_index("story_id")
    target_columns = {
        target: [f"{prefix}{column}" for column in topic_columns]
        for target, prefix in {
            "relative_votes": "relative_votes_",
            "article": "article_",
        }.items()
    }
    random_columns = [f"random_{column}" for column in topic_columns]
    policy_columns = ["story_id", "policy_id", *topic_columns]

    result_parts = []
    policy_ids = visible_distributions["policy_id"].drop_duplicates().tolist()
    for policy_id in policy_ids:
        policy_rows = visible_distributions.loc[
            visible_distributions["policy_id"].eq(policy_id),
            policy_columns,
        ]
        if not draw_mode:
            policy_rows = policy_rows.drop_duplicates(["story_id", "policy_id"])
        policy_result_parts = []
        for start in range(0, len(policy_rows), chunk_size):
            chunk = policy_rows.iloc[start:start + chunk_size]
            story_ids = chunk["story_id"].astype(str)
            valid = story_ids.isin(reference_frame.index)
            if not valid.any():
                continue
            chunk = chunk.loc[valid]
            story_ids = story_ids.loc[valid]
            policy_values = normalise(chunk[topic_columns].to_numpy(dtype=np.float64))
            random_values = normalise(
                reference_frame.loc[story_ids, random_columns].to_numpy(dtype=np.float64)
            )
            origin_root = np.sqrt(np.maximum(random_values, 0.0))
            policy_root = np.sqrt(np.maximum(policy_values, 0.0))
            movement = policy_root - origin_root
            movement_norm = np.linalg.norm(movement, axis=1)
            chunk_rows = []
            for target, columns in target_columns.items():
                target_values = normalise(
                    reference_frame.loc[story_ids, columns].to_numpy(dtype=np.float64)
                )
                target_root = np.sqrt(np.maximum(target_values, 0.0))
                reference_vector = target_root - origin_root
                reference_norm = np.linalg.norm(reference_vector, axis=1)
                denominator = movement_norm * reference_norm
                directional_cosine = np.divide(
                    np.sum(movement * reference_vector, axis=1),
                    denominator,
                    out=np.full(len(chunk), np.nan, dtype=float),
                    where=denominator > np.finfo(float).eps,
                )
                distance = jensen_shannon_distance(policy_values, target_values)
                random_target_distance = jensen_shannon_distance(random_values, target_values)
                target_similarity = cosine_similarity(policy_values, target_values)
                random_similarity = cosine_similarity(random_values, target_values)
                chunk_rows.append(pd.DataFrame({
                    "story_id": story_ids.to_numpy(),
                    "policy_id": str(policy_id),
                    "target": target,
                    "target_label": {
                        "relative_votes": "Relative-votes distribution",
                        "article": "Article distribution",
                    }[target],
                    "js_distance_to_target": distance,
                    "js_distance_random_to_target": random_target_distance,
                    "js_raw_target_gain": random_target_distance - distance,
                    "cosine_raw_target_gain": target_similarity - random_similarity,
                    "cosine_similarity_to_target": target_similarity,
                    "cosine_similarity_random_to_target": random_similarity,
                    "directional_cosine_to_target": directional_cosine,
                }))
            policy_result_parts.append(pd.concat(chunk_rows, ignore_index=True))
        if not policy_result_parts:
            continue
        policy_result = pd.concat(policy_result_parts, ignore_index=True)
        if draw_mode:
            grouped = policy_result.groupby(
                ["story_id", "policy_id", "target"], as_index=False, sort=False
            )
            numeric_columns = [
                "js_distance_to_target",
                "js_distance_random_to_target",
                "js_raw_target_gain",
                "cosine_raw_target_gain",
                "cosine_similarity_to_target",
                "cosine_similarity_random_to_target",
                "directional_cosine_to_target",
            ]
            averaged = grouped[numeric_columns].mean()
            averaged["n_draws"] = grouped.size()["size"].to_numpy()
            averaged = averaged.merge(
                policy_result[["story_id", "policy_id", "target", "target_label"]].drop_duplicates(
                    ["story_id", "policy_id", "target"]
                ),
                on=["story_id", "policy_id", "target"],
                how="left",
            )
            result_parts.append(averaged)
        else:
            result_parts.append(policy_result)

    if not result_parts:
        raise ValueError("No stories have all policy, article, and reference distributions")
    result = pd.concat(result_parts, ignore_index=True)
    # Use the same draw-then-average estimand for both sides of each gain.
    # Computing a distance to the mean random distribution would give random
    # itself a nonzero gain because these metrics are nonlinear.
    baseline_columns = [
        "js_distance_random_to_target", "cosine_similarity_random_to_target",
    ]
    baseline = result.loc[
        result["policy_id"].eq(random_policy_id),
        ["story_id", "target", "js_distance_to_target", "cosine_similarity_to_target"],
    ].rename(columns={
        "js_distance_to_target": baseline_columns[0],
        "cosine_similarity_to_target": baseline_columns[1],
    })
    result = result.drop(columns=baseline_columns).merge(
        baseline, on=["story_id", "target"], how="inner", validate="many_to_one",
    )
    result["js_raw_target_gain"] = (
        result["js_distance_random_to_target"] - result["js_distance_to_target"]
    )
    result["cosine_raw_target_gain"] = (
        result["cosine_similarity_to_target"] - result["cosine_similarity_random_to_target"]
    )
    metadata = metrics[["policy_id", "ordering", "reply_mode", "pinned"]].drop_duplicates("policy_id")
    return result.merge(metadata, on="policy_id", how="left")

__all__ = [
    "read_existing_oracle",
    "build_topic_policy_concentration_analysis",
    "build_topic_policy_exposure_coverage_analysis",
    "hellinger_oracle_order",
    "exact_hellinger_oracle_order",
    "ranked_topic_trajectory",
    "weighted_ranked_topic_distributions",
    "weighted_policy_topic_distributions",
    "bootstrap_topic_policy_effects",
    "build_rankings_from_scores",
    "aggregate_topic_policy_draw_distributions",
    "aggregate_topic_policy_draw_metrics",
    "paired_policy_contrasts",
    "build_structured_oracle_orders",
    "run_oracle_benchmark",
    "build_policy_contrast_order",
    "build_policy_reference_target_metrics",
]
