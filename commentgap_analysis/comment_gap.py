"""2025 Paper 1 comment-gap score.

The score is the curator picks' mean midrank in the audience relative-vote
ranking, normalized so the best possible k selections are 0, the worst are 1,
and a random k-set has expectation 0.5. Ties receive equal midranks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from commentgap_analysis.category_labels import translate_news_category
from commentgap_analysis.plotting import save_display_figure

from commentgap_analysis.paper1_descriptives import (
    AUDIENCE_DRAW_COLUMNS,
    _normalise_scopes,
    _qident,
    _qstring,
    _sha256,
    _write_json,
)


def normalised_selection_rank_gap(
    scores: Sequence[float], selected: Sequence[bool]
) -> float:
    """Return the legacy corrected top-k/bottom-k gap with stable midranks."""
    score_values = pd.to_numeric(pd.Series(scores), errors="coerce")
    selected_values = pd.Series(selected, dtype="boolean")
    if len(score_values) != len(selected_values):
        raise ValueError("scores and selected must have the same length")
    keep = score_values.notna() & selected_values.notna()
    score_values = score_values[keep]
    selected_values = selected_values[keep].astype(bool)
    n_candidates = len(score_values)
    n_selected = int(selected_values.sum())
    if n_candidates < 2 or n_selected < 1 or n_selected >= n_candidates:
        return float("nan")
    ranks = score_values.rank(method="average", ascending=False)
    mean_selected_rank = float(ranks[selected_values].mean())
    best_mean_rank = (n_selected + 1) / 2
    return (mean_selected_rank - best_mean_rank) / (n_candidates - n_selected)


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return the first value whose cumulative positive weight reaches 50%."""
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if value_array.shape != weight_array.shape:
        raise ValueError("values and weights must have the same shape")
    if value_array.ndim != 1 or not len(value_array):
        raise ValueError("values and weights must be non-empty one-dimensional arrays")
    if not np.isfinite(value_array).all() or not np.isfinite(weight_array).all():
        raise ValueError("values and weights must be finite")
    if (weight_array <= 0).any():
        raise ValueError("weights must be positive")
    order = np.argsort(value_array, kind="mergesort")
    ordered_values = value_array[order]
    cumulative = np.cumsum(weight_array[order])
    index = int(np.searchsorted(cumulative, weight_array.sum() / 2, side="left"))
    return float(ordered_values[index])


def _require_gap_columns(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    required = {
        "story_id",
        "comment_id",
        "n_candidates",
        "n_picks",
        "curator_selected",
        "relative_votes",
        *AUDIENCE_DRAW_COLUMNS,
    }
    missing = sorted(required - set(pq.read_schema(path).names))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _score_scope(
    con: duckdb.DuckDBPyConnection,
    choice_path: Path,
    split_path: Path,
    topics_path: Path,
    scope: str,
) -> pd.DataFrame:
    _require_gap_columns(choice_path)
    draw_counts = ",\n".join(
        f"SUM(CAST({_qident(column)} AS BIGINT))::BIGINT AS {_qident(column + '_count')}"
        for column in AUDIENCE_DRAW_COLUMNS
    )
    intersections = ",\n".join(
        f"SUM(CAST(curator_selected AS BIGINT) * CAST({_qident(column)} AS BIGINT))::BIGINT AS {_qident(column + '_intersection')}"
        for column in AUDIENCE_DRAW_COLUMNS
    )
    frame = con.execute(
        f"""
        WITH ranked AS (
            SELECT
                *,
                RANK() OVER (
                    PARTITION BY story_id ORDER BY relative_votes DESC
                ) + (
                    COUNT(*) OVER (PARTITION BY story_id, relative_votes) - 1
                ) / 2.0 AS audience_midrank
            FROM read_parquet({_qstring(choice_path)})
        ), article_scores AS (
            SELECT
                {_qstring(scope)} AS scope,
                CAST(c.story_id AS VARCHAR) AS story_id,
                MIN(c.article_month)::INTEGER AS article_month,
                MIN(s.split_role) AS split_role,
                MIN(s.development_fold)::INTEGER AS development_fold,
                CASE WHEN MIN(s.split_role) = 'paper2_test'
                     THEN 'held_out_test' ELSE MIN(s.split_role) END AS analysis_partition,
                MIN(t.primary_topic) AS primary_topic,
                MIN(t.section_1) AS section_1,
                MIN(t.section_2) AS section_2,
                MIN(t.section_3) AS section_3,
                COUNT(*)::BIGINT AS candidate_rows,
                MIN(c.n_candidates)::BIGINT AS n_candidates_min,
                MAX(c.n_candidates)::BIGINT AS n_candidates_max,
                MIN(c.n_picks)::BIGINT AS n_picks_min,
                MAX(c.n_picks)::BIGINT AS n_picks_max,
                SUM(CAST(c.curator_selected AS BIGINT))::BIGINT AS curator_count,
                AVG(c.audience_midrank) FILTER (WHERE c.curator_selected)
                    AS mean_curator_audience_midrank,
                COUNT(DISTINCT c.relative_votes)::BIGINT AS n_distinct_vote_scores,
                {draw_counts},
                {intersections}
            FROM ranked AS c
            LEFT JOIN read_parquet({_qstring(split_path)}) AS s USING (story_id)
            LEFT JOIN read_parquet({_qstring(topics_path)}) AS t USING (story_id)
            GROUP BY c.story_id
        )
        SELECT * FROM article_scores ORDER BY story_id
        """
    ).df()

    if frame["split_role"].isna().any():
        raise ValueError(f"{scope}: choice-set stories are missing from the shared split")
    if frame["primary_topic"].isna().any():
        raise ValueError(
            f"{scope}: topic metadata is incomplete; rerun stage 5 before stage 6"
        )
    if not np.array_equal(frame["candidate_rows"], frame["n_candidates_min"]):
        raise ValueError(f"{scope}: observed candidate counts do not match n_candidates")
    if not np.array_equal(frame["n_candidates_min"], frame["n_candidates_max"]):
        raise ValueError(f"{scope}: n_candidates is not constant within article")
    if not np.array_equal(frame["n_picks_min"], frame["n_picks_max"]):
        raise ValueError(f"{scope}: n_picks is not constant within article")
    if not np.array_equal(frame["curator_count"], frame["n_picks_min"]):
        raise ValueError(f"{scope}: curator counts do not match n_picks")

    overlap_columns = []
    for column in AUDIENCE_DRAW_COLUMNS:
        count_column = f"{column}_count"
        if not np.array_equal(frame[count_column], frame["n_picks_min"]):
            raise ValueError(f"{scope}: {column} counts do not match n_picks")
        draw = column.rsplit("_", 1)[-1]
        overlap = f"curator_audience_overlap_draw_{draw}"
        jaccard = f"curator_audience_jaccard_draw_{draw}"
        intersection = frame[f"{column}_intersection"].astype(float)
        frame[overlap] = intersection / frame["n_picks_min"]
        frame[jaccard] = intersection / (2 * frame["n_picks_min"] - intersection)
        overlap_columns.append(overlap)

    frame["n_candidates"] = frame.pop("n_candidates_min")
    frame["n_picks"] = frame.pop("n_picks_min")
    frame = frame.drop(columns=["n_candidates_max", "n_picks_max"])
    frame["best_possible_mean_rank"] = (frame["n_picks"] + 1) / 2
    frame["worst_possible_mean_rank"] = frame["n_candidates"] - (
        frame["n_picks"] - 1
    ) / 2
    frame["gap_score"] = (
        frame["mean_curator_audience_midrank"] - frame["best_possible_mean_rank"]
    ) / (frame["n_candidates"] - frame["n_picks"])
    frame["curator_audience_overlap_mean"] = frame[overlap_columns].mean(axis=1)
    frame["curator_audience_overlap_sd"] = frame[overlap_columns].std(axis=1, ddof=1)
    frame["curator_audience_overlap_min"] = frame[overlap_columns].min(axis=1)
    frame["curator_audience_overlap_max"] = frame[overlap_columns].max(axis=1)
    frame["has_any_vote_tie"] = frame["n_distinct_vote_scores"] < frame["n_candidates"]

    invalid = ~frame["gap_score"].between(-1e-12, 1 + 1e-12)
    if invalid.any():
        examples = frame.loc[invalid, ["story_id", "gap_score"]].head().to_dict("records")
        raise ValueError(f"{scope}: gap score outside [0, 1]: {examples}")
    frame["gap_score"] = frame["gap_score"].clip(0, 1)
    drop_counts = [
        f"{column}_{suffix}"
        for column in AUDIENCE_DRAW_COLUMNS
        for suffix in ("count", "intersection")
    ]
    return frame.drop(columns=drop_counts)


def _summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    working = frame.assign(
        _gap_candidate_product=frame["gap_score"] * frame["n_candidates"]
    )
    grouped = working.groupby(groups, dropna=False, observed=True)
    summary = (
        grouped
        .agg(
            n_articles=("story_id", "nunique"),
            gap_mean=("gap_score", "mean"),
            gap_sd=("gap_score", "std"),
            gap_q25=("gap_score", lambda values: values.quantile(0.25)),
            gap_median=("gap_score", "median"),
            gap_q75=("gap_score", lambda values: values.quantile(0.75)),
            overlap_mean=("curator_audience_overlap_mean", "mean"),
            overlap_tie_sd_mean=("curator_audience_overlap_sd", "mean"),
            candidates_mean=("n_candidates", "mean"),
            picks_mean=("n_picks", "mean"),
            articles_with_vote_ties=("has_any_vote_tie", "sum"),
        )
        .reset_index()
    )
    weighted = grouped.agg(
        candidate_comments=("n_candidates", "sum"),
        _gap_candidate_product=("_gap_candidate_product", "sum"),
    ).reset_index()
    weighted["gap_comment_weighted_mean"] = (
        weighted.pop("_gap_candidate_product") / weighted["candidate_comments"]
    )
    median_rows = []
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        median_rows.append(
            {
                **dict(zip(groups, keys)),
                "gap_comment_weighted_median": weighted_median(
                    group["gap_score"], group["n_candidates"]
                ),
            }
        )
    medians = pd.DataFrame(median_rows)
    weighted = weighted.merge(medians, on=groups, validate="one_to_one")
    return summary.merge(weighted, on=groups, validate="one_to_one")


def _save_gap_figures(
    scores: pd.DataFrame, topic_summary: pd.DataFrame, output_root: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")

    scope = "all" if "all" in set(scores["scope"]) else scores["scope"].iloc[0]
    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    plot_comment_gap_distribution(scores, scope, output_root, show=False)
    outputs = [
        figures_root / f"{scope}_comment_gap_distribution.{suffix}"
        for suffix in ("png", "pdf")
    ]
    plot_comment_gap_by_topic(topic_summary, scope, output_root, show=False)
    outputs.extend(
        figures_root / f"{scope}_comment_gap_by_topic.{suffix}"
        for suffix in ("png", "pdf")
    )
    return outputs


def plot_comment_gap_distribution(
    scores: pd.DataFrame,
    scope: str,
    output_root: Path | None = None,
    *,
    show: bool = True,
):
    """Render the article-level comment-gap distribution from scored data."""
    import matplotlib.pyplot as plt

    primary = scores[scores["scope"] == scope]
    fig, axis = plt.subplots(figsize=(6.6, 4.5))
    axis.hist(primary["gap_score"], bins=np.linspace(0, 1, 31), color="#356a8a")
    axis.axvline(
        primary["gap_score"].mean(), color="#263f75", label="equal-article mean"
    )
    axis.axvline(
        np.average(primary["gap_score"], weights=primary["n_candidates"]),
        color="#4b8b6f", linestyle=":", linewidth=2,
        label="candidate-comment-weighted mean",
    )
    axis.set(
        xlabel="Normalized curator–audience comment gap",
        ylabel="Discussions",
        title=f"Comment-gap distribution ({scope})",
    )
    axis.legend(frameon=False)
    return save_display_figure(fig, output_root, f"{scope}_comment_gap_distribution", show=show)


def plot_comment_gap_by_topic(
    topic_summary: pd.DataFrame,
    scope: str,
    output_root: Path | None = None,
    *,
    show: bool = True,
):
    """Render topic-level comment gaps directly from the stage-6 summary."""
    import matplotlib.pyplot as plt

    shown = topic_summary[
        (topic_summary["scope"] == scope)
        & (topic_summary["analysis_partition"] == "all_partitions")
    ].nlargest(15, "n_articles").sort_values("gap_mean")
    label_column = "primary_topic_label" if "primary_topic_label" in shown else "primary_topic"
    fig, axis = plt.subplots(figsize=(6.6, max(4.5, 0.32 * len(shown))))
    y = np.arange(len(shown))
    axis.barh(y - 0.18, shown["gap_mean"], height=0.34, color="#356a8a", label="equal article weight")
    axis.barh(y + 0.18, shown["gap_comment_weighted_mean"], height=0.34, color="#4b8b6f", label="candidate-comment weight")
    axis.set_yticks(y, shown[label_column].map(translate_news_category))
    axis.axvline(0.5, color="#d4744c", linestyle="--")
    axis.set(
        xlim=(0, 1), xlabel="Mean normalized comment gap",
        title=f"Comment gap by primary section_1 topic ({scope})",
    )
    axis.legend(frameon=False)
    return save_display_figure(fig, output_root, f"{scope}_comment_gap_by_topic", show=show)


def run_comment_gap_analysis(
    *,
    model_data_root: Path = Path("model_output/selection_2025/model_data"),
    descriptives_root: Path = Path("model_output/selection_2025/paper1/descriptives"),
    output_root: Path = Path("model_output/selection_2025/paper1/comment_gap"),
    scopes: Iterable[str] = ("all",),
    threads: int = 4,
    make_figures: bool = True,
) -> dict:
    """Build stage-6 article scores, summaries, figures, and a final manifest."""
    scopes = _normalise_scopes(scopes)
    split_path = model_data_root / "master_article_split.parquet"
    topics_path = descriptives_root / "article_topics.parquet"
    descriptive_manifest_path = descriptives_root / "descriptive_manifest.json"
    if not topics_path.exists():
        raise FileNotFoundError(
            f"Missing {topics_path}; run scripts/run_paper1_descriptives.py first"
        )
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    output_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, int(threads))}")
    try:
        frames = [
            _score_scope(
                con,
                model_data_root / f"choice_set_{scope}.parquet",
                split_path,
                topics_path,
                scope,
            )
            for scope in scopes
        ]
    finally:
        con.close()
    scores = pd.concat(frames, ignore_index=True)

    overall = _summary(scores, ["scope", "analysis_partition"])
    overall_all = _summary(
        scores.assign(analysis_partition="all_partitions"),
        ["scope", "analysis_partition"],
    )
    overall = pd.concat([overall, overall_all], ignore_index=True)
    topic = _summary(scores, ["scope", "analysis_partition", "primary_topic"])
    topic_all = _summary(
        scores.assign(analysis_partition="all_partitions"),
        ["scope", "analysis_partition", "primary_topic"],
    )
    topic = pd.concat([topic, topic_all], ignore_index=True)
    topic["primary_topic_label"] = topic["primary_topic"].map(translate_news_category)

    table_paths = {
        "article_gap_scores": output_root / "article_gap_scores.parquet",
        "comment_gap_summary": output_root / "comment_gap_summary.csv",
        "comment_gap_topic_summary": output_root / "comment_gap_topic_summary.csv",
    }
    scores.to_parquet(table_paths["article_gap_scores"], index=False)
    overall.to_csv(table_paths["comment_gap_summary"], index=False)
    topic.to_csv(table_paths["comment_gap_topic_summary"], index=False)
    figure_paths = _save_gap_figures(scores, topic, output_root) if make_figures else []

    preprocessing_path = model_data_root / "preprocessing_manifest.json"
    preprocessing = json.loads(preprocessing_path.read_text()) if preprocessing_path.exists() else {}
    pre_outputs = preprocessing.get("outputs", {})
    inputs = {
        "article_topics": {"path": str(topics_path), "sha256": _sha256(topics_path)},
        "master_article_split": {
            "path": str(split_path),
            "sha256": pre_outputs.get("master_article_split.parquet", {}).get("sha256") or _sha256(split_path),
        },
    }
    if descriptive_manifest_path.exists():
        inputs["descriptive_manifest"] = {
            "path": str(descriptive_manifest_path),
            "sha256": _sha256(descriptive_manifest_path),
        }
    for scope in scopes:
        path = model_data_root / f"choice_set_{scope}.parquet"
        inputs[f"choice_set_{scope}"] = {
            "path": str(path),
            "sha256": pre_outputs.get(path.name, {}).get("sha256") or _sha256(path),
        }
    outputs = {
        name: {"path": str(path), "sha256": _sha256(path), "rows": len(scores) if name == "article_gap_scores" else len(overall) if name == "comment_gap_summary" else len(topic)}
        for name, path in table_paths.items()
    }
    for path in figure_paths:
        outputs[f"figure:{path.name}"] = {"path": str(path), "sha256": _sha256(path)}

    manifest = {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scopes": list(scopes),
        "primary_scope": "all",
        "appendix_scope": "root",
        "metric": {
            "name": "normalised_selection_rank_gap",
            "ranking": "relative_votes descending with average midranks for ties",
            "formula": "(mean_selected_rank - (k + 1) / 2) / (N - k)",
            "interpretation": {"0": "best possible k-set", "0.5": "random k-set expectation", "1": "worst possible k-set"},
        },
        "summary_weighting": {
            "gap_mean": "equal weight per discussion/article",
            "gap_comment_weighted_mean": "weight each discussion by n_candidates; sum(N_j * gap_j) / sum(N_j)",
            "gap_comment_weighted_median": "first ordered article gap where cumulative n_candidates reaches at least 50% of candidate comments",
        },
        "audience_tie_policy": "The gap itself uses midranks; overlap/Jaccard diagnostics report all ten deterministic top-k tie draws and their equal-weight mean.",
        "topic_mapping": "stage-5 section_1 primary topic; section_2 and section_3 retained for audit",
        "split_display_mapping": {"paper2_test": "held_out_test"},
        "legacy_regression_status": "Not ported: its pre-2025 aggregate covariates are not equivalent to the shared 2025 feature contract; stage 7 supplies the paper's inferential selection regressions.",
        "inputs": inputs,
        "outputs": outputs,
    }
    _write_json(manifest, output_root / "comment_gap_manifest.json")
    return manifest
