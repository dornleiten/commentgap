"""Shared topic-distribution metrics and geometry helpers.

General topic-distribution metrics, agenda summaries, and Hellinger geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .topic_modeling import _distribution_metric_inputs, _normalise_distribution, _topic_columns


def shannon_entropy(distributions: Sequence[float] | np.ndarray) -> float | np.ndarray:
    """Return Shannon entropy using natural logarithms."""

    values = _normalise_distribution(np.asarray(distributions, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(values > 0, values * np.log(values), 0.0)
    result = -terms.sum(axis=-1)
    return float(result) if np.ndim(result) == 0 else result


def effective_topic_count(distributions: Sequence[float] | np.ndarray) -> float | np.ndarray:
    """Return the entropy-derived effective number of topics."""

    return np.exp(shannon_entropy(distributions))


def jensen_shannon_divergence(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Return Jensen-Shannon divergence in nats."""

    left, right = _distribution_metric_inputs(first, second)
    midpoint = (left + right) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        left_term = np.where(left > 0, left * np.log(left / midpoint), 0.0)
        right_term = np.where(right > 0, right * np.log(right / midpoint), 0.0)
    result = 0.5 * (left_term.sum(axis=-1) + right_term.sum(axis=-1))
    return float(result) if np.ndim(result) == 0 else result


def jensen_shannon_distance(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Return square-root Jensen-Shannon distance in nats."""

    return np.sqrt(jensen_shannon_divergence(first, second))


def hellinger_distance(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Return the conventional Hellinger distance."""

    left, right = _distribution_metric_inputs(first, second)
    result = np.linalg.norm(np.sqrt(left) - np.sqrt(right), axis=-1) / np.sqrt(2.0)
    return float(result) if np.ndim(result) == 0 else result


def cosine_similarity(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Return ordinary cosine similarity on topic proportions."""

    left, right = _distribution_metric_inputs(first, second)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    result = np.divide(
        np.sum(left * right, axis=-1),
        denominator,
        out=np.full(np.broadcast_shapes(left.shape[:-1], right.shape[:-1]), np.nan),
        where=denominator > np.finfo(float).eps,
    )
    return float(result) if np.ndim(result) == 0 else result


def _summary_statistics(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = pd.Series(values, dtype=float).dropna().to_numpy()
    if not len(array):
        return {"mean": np.nan, "sd": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0}
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else np.nan
    margin = 1.96 * sd / np.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "n": int(len(array)),
    }


def _normalised_entropy(values: np.ndarray) -> np.ndarray:
    probabilities = _normalise_distribution(np.asarray(values, dtype=float))
    entropy = np.asarray(shannon_entropy(probabilities), dtype=float)
    n_topics = probabilities.shape[-1]
    return entropy / np.log(n_topics) if n_topics > 1 else np.zeros(len(probabilities))


def hellinger_projection(
    article: Sequence[float],
    available: Sequence[float],
    visible: Sequence[float],
) -> dict[str, float]:
    """Decompose visible movement along the available-to-article axis."""

    article, available = _distribution_metric_inputs(article, available)
    article, visible = _distribution_metric_inputs(article, visible)
    root_article = np.sqrt(article)
    root_available = np.sqrt(available)
    root_visible = np.sqrt(visible)
    axis = root_article - root_available
    movement = root_visible - root_available
    axis_norm = float(np.linalg.norm(axis))
    movement_norm = float(np.linalg.norm(movement))
    denominator = float(np.dot(axis, axis))
    if denominator <= np.finfo(float).eps:
        alpha = np.nan
        cosine = np.nan
        angle_degrees = np.nan
        projection = root_available.copy()
        residual = float(np.linalg.norm(movement))
    else:
        alpha = float(np.dot(movement, axis) / denominator)
        projection = root_available + alpha * axis
        residual = float(np.linalg.norm(root_visible - projection))
        if movement_norm <= np.finfo(float).eps:
            cosine = np.nan
            angle_degrees = np.nan
        else:
            cosine = float(np.dot(movement, axis) / (movement_norm * axis_norm))
            angle_degrees = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return {
        "hellinger_alpha": alpha,
        "hellinger_cosine": cosine,
        "hellinger_angle_degrees": angle_degrees,
        "hellinger_residual": residual,
        "hellinger_available_article_distance": float(np.linalg.norm(axis)),
        "overshoot_magnitude": float(max(0.0, alpha - 1.0)) if np.isfinite(alpha) else np.nan,
    }


def classify_projection(alpha: float, *, match_tolerance: float = 0.1) -> str:
    """Translate a Hellinger projection into an interpretable category."""

    if not np.isfinite(alpha):
        return "undefined_baseline"
    if alpha < 0:
        return "away"
    if alpha > 1 + match_tolerance:
        return "overshoot"
    if abs(alpha - 1) <= match_tolerance:
        return "matches_article"
    return "partial_convergence"


def compute_topic_metrics(
    article_distributions: pd.DataFrame,
    discussion_distributions: pd.DataFrame,
    visible_distributions: pd.DataFrame,
    *,
    story_column: str = "story_id",
    match_tolerance: float = 0.1,
) -> pd.DataFrame:
    """Calculate article/discussion similarity, diversity, and ranking effects."""

    article_topics = _topic_columns(article_distributions)
    if _topic_columns(discussion_distributions) != article_topics or _topic_columns(visible_distributions) != article_topics:
        raise ValueError("Article, discussion, and visible distributions must share topic columns")
    for frame, label in ((article_distributions, "article"), (discussion_distributions, "discussion"), (visible_distributions, "visible")):
        if story_column not in frame:
            raise ValueError(f"{label} distributions are missing {story_column!r}")
    if article_distributions[story_column].duplicated().any() or discussion_distributions[story_column].duplicated().any():
        raise ValueError("Article and discussion distributions must have one row per story")
    visible_keys = [column for column in visible_distributions.columns if column not in {story_column, *article_topics, "n_documents"}]
    article = article_distributions.set_index(story_column)
    discussion = discussion_distributions.set_index(story_column)
    rows = []
    for _, visible in visible_distributions.iterrows():
        story_id = visible[story_column]
        if story_id not in article.index or story_id not in discussion.index:
            continue
        a = article.loc[story_id, article_topics].to_numpy(dtype=float)
        c = discussion.loc[story_id, article_topics].to_numpy(dtype=float)
        v = visible[article_topics].to_numpy(dtype=float)
        a_entropy = float(shannon_entropy(a))
        c_entropy = float(shannon_entropy(c))
        v_entropy = float(shannon_entropy(v))
        article_discussion_distance = float(jensen_shannon_distance(a, c))
        article_visible_distance = float(jensen_shannon_distance(a, v))
        alignment_gain = article_discussion_distance - article_visible_distance
        alignment_gain_fraction = (
            alignment_gain / article_discussion_distance
            if article_discussion_distance > np.finfo(float).eps
            else np.nan
        )
        projection = hellinger_projection(a, c, v)
        row = {story_column: story_id}
        row.update({column: visible[column] for column in visible_keys})
        article_root = np.sqrt(a)
        discussion_root = np.sqrt(c)
        visible_root = np.sqrt(v)
        article_discussion_cosine = float(cosine_similarity(a, c))
        article_visible_cosine = float(cosine_similarity(a, v))
        cosine_progress = article_visible_cosine - article_discussion_cosine
        article_discussion_bhattacharyya = float(np.dot(article_root, discussion_root))
        article_visible_bhattacharyya = float(np.dot(article_root, visible_root))
        bhattacharyya_progress = article_visible_bhattacharyya - article_discussion_bhattacharyya
        row.update({
            "article_discussion_js_distance": article_discussion_distance,
            "article_visible_js_distance": article_visible_distance,
            "article_discussion_hellinger_distance": float(hellinger_distance(a, c)),
            "article_visible_hellinger_distance": float(hellinger_distance(a, v)),
            "article_discussion_cosine": article_discussion_cosine,
            "article_visible_cosine": article_visible_cosine,
            "cosine_progress": cosine_progress,
            "article_discussion_bhattacharyya": article_discussion_bhattacharyya,
            "article_visible_bhattacharyya": article_visible_bhattacharyya,
            "bhattacharyya_progress": bhattacharyya_progress,
            "alignment_gain": alignment_gain,
            "alignment_gain_fraction": alignment_gain_fraction,
            "article_entropy": a_entropy,
            "discussion_entropy": c_entropy,
            "visible_entropy": v_entropy,
            "article_normalized_entropy": a_entropy / np.log(len(a)),
            "discussion_normalized_entropy": c_entropy / np.log(len(a)),
            "visible_normalized_entropy": v_entropy / np.log(len(a)),
            "article_effective_topics": float(np.exp(a_entropy)),
            "discussion_effective_topics": float(np.exp(c_entropy)),
            "visible_effective_topics": float(np.exp(v_entropy)),
            "discussion_minus_article_entropy": c_entropy - a_entropy,
            "visible_minus_discussion_entropy": v_entropy - c_entropy,
            "visible_discussion_effective_topic_ratio": float(np.exp(v_entropy - c_entropy)),
            **projection,
            "projection_class": classify_projection(projection["hellinger_alpha"], match_tolerance=match_tolerance),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_hellinger_movement(
    article_distributions: pd.DataFrame,
    discussion_distributions: pd.DataFrame,
    visible_distributions: pd.DataFrame,
    *,
    story_column: str = "story_id",
    policy_column: str = "policy_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise direction and spread of discussion-to-visible movement.

    Movement is calculated in square-root probability space. The first result
    contains policy-level directional summaries; the second contains the mean
    and standard deviation of the movement vector for every topic and policy.
    """
    article_topics = _topic_columns(article_distributions)
    if _topic_columns(discussion_distributions) != article_topics:
        raise ValueError("Article and discussion distributions must share topic columns")
    visible_topics = _topic_columns(visible_distributions)
    if visible_topics != article_topics:
        raise ValueError("Visible distributions must share topic columns")
    for frame, label in (
        (article_distributions, "article"),
        (discussion_distributions, "discussion"),
        (visible_distributions, "visible"),
    ):
        if story_column not in frame:
            raise ValueError(f"{label} distributions are missing {story_column!r}")
    if policy_column not in visible_distributions:
        raise ValueError(f"visible distributions are missing {policy_column!r}")

    article = article_distributions.copy()
    discussion = discussion_distributions.copy()
    article[story_column] = article[story_column].astype(str)
    discussion[story_column] = discussion[story_column].astype(str)
    if article[story_column].duplicated().any() or discussion[story_column].duplicated().any():
        raise ValueError("Article and discussion distributions must have one row per story")
    article = article.set_index(story_column)
    discussion = discussion.set_index(story_column)

    visible = visible_distributions.copy()
    visible[story_column] = visible[story_column].astype(str)
    summary_rows: list[dict[str, object]] = []
    topic_rows: list[dict[str, object]] = []
    topic_metadata = [
        column for column in visible.columns
        if column not in {story_column, policy_column, *article_topics}
    ]

    def add_stats(row: dict[str, object], prefix: str, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        if not len(finite):
            row[f"{prefix}_mean"] = np.nan
            row[f"{prefix}_sd"] = np.nan
            row[f"{prefix}_q10"] = np.nan
            row[f"{prefix}_median"] = np.nan
            row[f"{prefix}_q90"] = np.nan
            return
        row[f"{prefix}_mean"] = float(np.mean(finite))
        row[f"{prefix}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        row[f"{prefix}_q10"], row[f"{prefix}_median"], row[f"{prefix}_q90"] = (
            float(value) for value in np.quantile(finite, [0.10, 0.50, 0.90])
        )

    for policy_id, group in visible.groupby(policy_column, sort=True):
        source_story_count = len(group)
        source_story_ids = group[story_column].astype(str)
        complete_mask = source_story_ids.isin(article.index) & source_story_ids.isin(discussion.index)
        group = group.loc[complete_mask].copy()
        excluded_story_count = source_story_count - len(group)
        if group.empty:
            continue
        story_ids = group[story_column].to_numpy(dtype=str)
        article_root = np.sqrt(article.loc[story_ids, article_topics].to_numpy(dtype=float))
        discussion_root = np.sqrt(discussion.loc[story_ids, article_topics].to_numpy(dtype=float))
        visible_root = np.sqrt(group[article_topics].to_numpy(dtype=float))
        movement = visible_root - discussion_root
        axis = article_root - discussion_root
        axis_norm = np.linalg.norm(axis, axis=1)
        movement_norm = np.linalg.norm(movement, axis=1)
        denominator = axis_norm**2
        dot_product = np.sum(movement * axis, axis=1)
        alpha = np.divide(
            dot_product,
            denominator,
            out=np.full(len(group), np.nan),
            where=denominator > np.finfo(float).eps,
        )
        cosine = np.divide(
            dot_product,
            movement_norm * axis_norm,
            out=np.full(len(group), np.nan),
            where=(movement_norm > np.finfo(float).eps) & (axis_norm > np.finfo(float).eps),
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        residual = np.linalg.norm(movement - np.nan_to_num(alpha)[:, None] * axis, axis=1)
        valid_movement = movement_norm > np.finfo(float).eps
        unit_movement = movement[valid_movement] / movement_norm[valid_movement, None]
        mean_unit = unit_movement.mean(axis=0) if len(unit_movement) else np.full(len(article_topics), np.nan)
        metadata = group.iloc[0]
        valid_cosine = np.isfinite(cosine)
        valid_alpha = np.isfinite(alpha)
        row: dict[str, object] = {
            policy_column: policy_id,
            "n_stories": len(group),
            "n_stories_source": source_story_count,
            "n_stories_excluded_missing_baseline": excluded_story_count,
            "share_toward_article": (
                float(np.mean(cosine[valid_cosine] > 0)) if valid_cosine.any() else np.nan
            ),
            "share_moving_away": (
                float(np.mean(alpha[valid_alpha] < 0)) if valid_alpha.any() else np.nan
            ),
            "mean_unit_movement_resultant": float(np.linalg.norm(mean_unit)) if len(unit_movement) else np.nan,
        }
        for column in topic_metadata:
            row[column] = metadata[column]
        add_stats(row, "movement_norm", movement_norm)
        add_stats(row, "hellinger_cosine", cosine)
        add_stats(row, "hellinger_angle_degrees", angle)
        add_stats(row, "hellinger_alpha", alpha)
        add_stats(row, "hellinger_residual", residual)
        summary_rows.append(row)

        mean_movement = movement.mean(axis=0)
        sd_movement = movement.std(axis=0, ddof=1) if len(movement) > 1 else np.zeros(len(article_topics))
        for topic, mean_value, sd_value in zip(article_topics, mean_movement, sd_movement):
            topic_rows.append({
                policy_column: policy_id,
                "topic": topic,
                "mean_hellinger_change": float(mean_value),
                "sd_hellinger_change": float(sd_value),
                **{column: metadata[column] for column in topic_metadata},
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(topic_rows)

def topicwise_gap_ratios(
    article: Sequence[float],
    available: Sequence[float],
    visible: Sequence[float],
    *,
    minimum_gap: float = 0.02,
) -> pd.DataFrame:
    """Return secondary per-topic convergence/overshoot ratios."""

    article, available = _distribution_metric_inputs(article, available)
    article, visible = _distribution_metric_inputs(article, visible)
    gap = article - available
    change = visible - available
    result = pd.DataFrame({
        "article_share": article,
        "available_share": available,
        "visible_share": visible,
        "article_available_gap": gap,
        "visible_change": change,
    })
    result["alpha_topic"] = np.where(np.abs(gap) > minimum_gap, change / gap, np.nan)
    result["small_baseline_gap"] = np.abs(gap) <= minimum_gap
    return result


def build_topic_agenda_rarefaction_analysis(
    memberships: pd.DataFrame,
    *,
    article_doc_type: str = "article",
    discussion_doc_type: str = "comment",
    sample_size: int | None = None,
    n_draws: int = 100,
    seed: int = 2025,
    output_root: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Compare article and discussion diversity at matched document counts.

    For each story, both agendas are repeatedly sampled without replacement at
    the same number of valid topic-bearing documents. This controls the
    finite-sample bias of empirical entropy separately from normalization by
    the number of model topics.
    """

    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be positive when specified")
    required = {"story_id", "doc_type"}
    missing = sorted(required - set(memberships.columns))
    if missing:
        raise ValueError(f"Memberships are missing columns: {missing}")
    topic_columns = _topic_columns(memberships)
    valid = memberships
    if "valid_topic" in valid.columns:
        valid = valid.loc[valid["valid_topic"].astype(bool)]
    valid = valid.copy()
    valid["story_id"] = valid["story_id"].astype(str)
    rng = np.random.default_rng(seed)
    rows = []
    for story_id, group in valid.groupby("story_id", sort=True):
        article = group.loc[group["doc_type"].eq(article_doc_type), topic_columns].to_numpy(float)
        discussion = group.loc[group["doc_type"].eq(discussion_doc_type), topic_columns].to_numpy(float)
        if not len(article) or not len(discussion):
            continue
        n = min(len(article), len(discussion))
        if sample_size is not None:
            n = min(n, sample_size)
        if n < 1:
            continue
        for draw in range(1, n_draws + 1):
            article_sample = article[rng.choice(len(article), size=n, replace=False)]
            discussion_sample = discussion[rng.choice(len(discussion), size=n, replace=False)]
            article_distribution = _normalise_distribution(article_sample.mean(axis=0))
            discussion_distribution = _normalise_distribution(discussion_sample.mean(axis=0))
            article_entropy = float(shannon_entropy(article_distribution))
            discussion_entropy = float(shannon_entropy(discussion_distribution))
            rows.append({
                "story_id": story_id,
                "draw": draw,
                "sample_size": n,
                "n_valid_article_documents": len(article),
                "n_valid_discussion_documents": len(discussion),
                "article_normalized_entropy": article_entropy / np.log(len(topic_columns)),
                "discussion_normalized_entropy": discussion_entropy / np.log(len(topic_columns)),
                "discussion_minus_article_normalized_entropy": (
                    discussion_entropy - article_entropy
                ) / np.log(len(topic_columns)),
                "article_effective_topic_count": np.exp(article_entropy),
                "discussion_effective_topic_count": np.exp(discussion_entropy),
                "discussion_minus_article_effective_topic_count": (
                    np.exp(discussion_entropy) - np.exp(article_entropy)
                ),
            })
    story = pd.DataFrame(rows)
    if story.empty:
        raise ValueError("No stories have valid article and discussion documents")
    metric_labels = {
        "article_normalized_entropy": "Rarefied article normalized entropy",
        "discussion_normalized_entropy": "Rarefied discussion normalized entropy",
        "discussion_minus_article_normalized_entropy": (
            "Rarefied discussion minus article normalized entropy"
        ),
        "article_effective_topic_count": "Rarefied article effective topic count",
        "discussion_effective_topic_count": "Rarefied discussion effective topic count",
        "discussion_minus_article_effective_topic_count": (
            "Rarefied discussion minus article effective topic count"
        ),
    }
    summary_rows = []
    for metric, label in metric_labels.items():
        row = {"metric": metric, "label": label}
        row.update(_summary_statistics(story[metric]))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary["n_stories"] = story["story_id"].nunique()
    summary["mean_sample_size"] = story.groupby("story_id")["sample_size"].first().mean()
    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        story.to_parquet(
            output_root / "topic_agenda_rarefaction_story_metrics.parquet",
            index=False,
        )
        summary.to_csv(output_root / "topic_agenda_rarefaction_summary.csv", index=False)
    return {"story": story, "summary": summary}

def build_topic_agenda_baseline_analysis(
    article_distributions: pd.DataFrame,
    discussion_distributions: pd.DataFrame,
    *,
    topic_columns: Sequence[str] | None = None,
    output_root: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Summarise article -discussion similarity and baseline concentration.

    Similarity is reported as square-root Jensen -Shannon distance, ordinary
    cosine similarity on topic proportions, and the square-root-space
    Bhattacharyya coefficient. Concentration is reported as
    normalized Shannon entropy, H / log(K), and effective topic count, exp(H).
    All summaries are paired at the story level.
    """

    article_topics = list(topic_columns or _topic_columns(article_distributions))
    if _topic_columns(article_distributions) != article_topics:
        raise ValueError("article topic columns do not match topic_columns")
    if _topic_columns(discussion_distributions) != article_topics:
        raise ValueError("Article and discussion distributions must share topic columns")

    article = article_distributions[["story_id", *article_topics]].copy()
    discussion = discussion_distributions[["story_id", *article_topics]].copy()
    article["story_id"] = article["story_id"].astype(str)
    discussion["story_id"] = discussion["story_id"].astype(str)
    paired = article.merge(
        discussion,
        on="story_id",
        how="inner",
        suffixes=("_article", "_discussion"),
        validate="one_to_one",
    )
    article_values = paired[[f"{column}_article" for column in article_topics]].to_numpy(float)
    discussion_values = paired[
        [f"{column}_discussion" for column in article_topics]
    ].to_numpy(float)
    article_values = _normalise_distribution(article_values)
    discussion_values = _normalise_distribution(discussion_values)
    article_entropy = np.asarray(shannon_entropy(article_values), dtype=float)
    discussion_entropy = np.asarray(shannon_entropy(discussion_values), dtype=float)
    article_normalized_entropy = _normalised_entropy(article_values)
    discussion_normalized_entropy = _normalised_entropy(discussion_values)
    story = pd.DataFrame({
        "story_id": paired["story_id"].to_numpy(),
        "article_discussion_js_distance": jensen_shannon_distance(
            article_values, discussion_values
        ),
        "article_discussion_cosine_similarity": cosine_similarity(
            article_values, discussion_values
        ),
        "article_discussion_bhattacharyya_coefficient": np.sum(
            np.sqrt(article_values) * np.sqrt(discussion_values), axis=1
        ),
        "article_entropy": article_entropy,
        "discussion_entropy": discussion_entropy,
        "article_normalized_entropy": article_normalized_entropy,
        "discussion_normalized_entropy": discussion_normalized_entropy,
        "discussion_minus_article_normalized_entropy": (
            discussion_normalized_entropy - article_normalized_entropy
        ),
        "article_effective_topic_count": np.exp(article_entropy),
        "discussion_effective_topic_count": np.exp(discussion_entropy),
        "discussion_minus_article_effective_topic_count": (
            np.exp(discussion_entropy) - np.exp(article_entropy)
        ),
    })
    metric_labels = {
        "article_discussion_js_distance": "Article -discussion JS distance",
        "article_discussion_cosine_similarity": "Article -discussion raw-proportion cosine similarity",
        "article_discussion_bhattacharyya_coefficient": "Article -discussion Bhattacharyya coefficient (sqrt-space cosine)",
        "article_normalized_entropy": "Article normalized entropy",
        "discussion_normalized_entropy": "Discussion normalized entropy",
        "discussion_minus_article_normalized_entropy": (
            "Discussion minus article normalized entropy"
        ),
        "article_effective_topic_count": "Article effective topic count",
        "discussion_effective_topic_count": "Discussion effective topic count",
        "discussion_minus_article_effective_topic_count": (
            "Discussion minus article effective topic count"
        ),
    }
    summary_rows = []
    for metric, label in metric_labels.items():
        row = {"metric": metric, "label": label}
        row.update(_summary_statistics(story[metric]))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        story.to_parquet(output_root / "topic_agenda_baseline_story_metrics.parquet", index=False)
        summary.to_csv(output_root / "topic_agenda_baseline_summary.csv", index=False)
    return {"story": story, "summary": summary}

__all__ = [
    "shannon_entropy",
    "effective_topic_count",
    "jensen_shannon_divergence",
    "jensen_shannon_distance",
    "hellinger_distance",
    "cosine_similarity",
    "hellinger_projection",
    "classify_projection",
    "compute_topic_metrics",
    "summarize_hellinger_movement",
    "topicwise_gap_ratios",
    "build_topic_agenda_rarefaction_analysis",
    "build_topic_agenda_baseline_analysis",
]
