"""Reusable diagnostics for production model-feature distributions."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


QUANTILES = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)
BINARY_FEATURES = {
    "url_present", "vienna_overnight", "vienna_weekday_shoulder_evening",
    "vienna_weekend_day_evening", "is_reply",
}
PROBABILITY_FEATURES = {"sentiment_positive", "sentiment_negative", "toxicity_probability"}
BOUNDED_RANGES = {
    **{name: (0.0, 1.0) for name in BINARY_FEATURES | PROBABILITY_FEATURES},
    "article_similarity_top3": (-1.0, 1.0),
}
TEXT_NLP = {
    "log_words", "sentiment_positive", "sentiment_negative", "toxicity_probability",
    "lexdiv_length_adjusted", "reading_level_length_adjusted", "url_present",
}
REPLY_STRUCTURE = {"is_reply", "reply_depth_centered", "prior_reply_composition"}


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def expected_range(name: str) -> tuple[float, float] | None:
    if name.startswith("aqua_") and name.endswith("_expected"):
        return (0.0, 3.0)
    return BOUNDED_RANGES.get(name)


def full_distribution_summary(
    scope: str,
    *,
    connection,
    choice_paths: dict[str, Path],
    model_features: dict[str, list[str]],
    registry: dict,
) -> pd.DataFrame:
    expressions = ["COUNT(*) AS total_rows"]
    quantile_sql = "[" + ", ".join(str(value) for value in QUANTILES) + "]"
    for feature in model_features[scope]:
        column = quote_identifier(feature)
        numeric = f"CAST({column} AS DOUBLE)"
        finite = f"isfinite({numeric})"
        expressions.extend(
            [
                f"COUNT_IF({column} IS NULL) AS {quote_identifier(feature + '__nulls')}",
                f"COUNT_IF({column} IS NOT NULL AND NOT {finite}) AS {quote_identifier(feature + '__nonfinite')}",
                f"MIN({numeric}) FILTER (WHERE {finite}) AS {quote_identifier(feature + '__minimum')}",
                f"MAX({numeric}) FILTER (WHERE {finite}) AS {quote_identifier(feature + '__maximum')}",
                f"AVG({numeric}) FILTER (WHERE {finite}) AS {quote_identifier(feature + '__mean')}",
                f"STDDEV_SAMP({numeric}) FILTER (WHERE {finite}) AS {quote_identifier(feature + '__sd')}",
                f"APPROX_COUNT_DISTINCT({numeric}) FILTER (WHERE {finite}) AS {quote_identifier(feature + '__distinct')}",
                f"APPROX_QUANTILE({numeric}, {quantile_sql}) FILTER (WHERE {finite}) AS {quote_identifier(feature + '__quantiles')}",
            ]
        )
    query = "SELECT\n  " + ",\n  ".join(expressions) + "\nFROM read_parquet(?)"
    row = connection.execute(query, [str(choice_paths[scope])]).fetchdf().iloc[0]
    records = []
    for feature in model_features[scope]:
        raw_quantiles = row[f"{feature}__quantiles"]
        quantiles = list(raw_quantiles) if raw_quantiles is not None else [np.nan] * len(QUANTILES)
        record = {
            "scope": scope,
            "feature": feature,
            "label": registry["features"][feature]["label"],
            "rows": int(row["total_rows"]),
            "nulls": int(row[f"{feature}__nulls"]),
            "nonfinite": int(row[f"{feature}__nonfinite"]),
            "distinct_approx": int(row[f"{feature}__distinct"]),
            "minimum": row[f"{feature}__minimum"],
            "maximum": row[f"{feature}__maximum"],
            "mean": row[f"{feature}__mean"],
            "sd": row[f"{feature}__sd"],
            **{f"p{int(q * 1000):03d}": value for q, value in zip(QUANTILES, quantiles)},
        }
        bounds = expected_range(feature)
        record["expected_minimum"] = bounds[0] if bounds else np.nan
        record["expected_maximum"] = bounds[1] if bounds else np.nan
        record["range_violation"] = bool(
            bounds and (record["minimum"] < bounds[0] - 1e-8 or record["maximum"] > bounds[1] + 1e-8)
        )
        record["constant_or_near_constant"] = bool(
            record["distinct_approx"] <= 1 or not np.isfinite(record["sd"]) or record["sd"] <= 1e-12
        )
        lower_span = max(float(record["p500"] - record["p010"]), 1e-12)
        record["heavy_right_tail"] = bool(record["p990"] - record["p500"] > 20 * lower_span)
        records.append(record)
    return pd.DataFrame(records)


def invariant_checks(
    scope: str,
    *,
    connection,
    choice_paths: dict[str, Path],
    model_features: dict[str, list[str]],
) -> pd.DataFrame:
    path = choice_paths[scope]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    expressions = {
        "duplicate_candidate_keys": (
            "COUNT(*) - COUNT(DISTINCT struct_pack(story_id := story_id, comment_id := comment_id))"
        ),
        "uninformative_choice_rows": "COUNT_IF(n_candidates <= 1 OR n_picks <= 0 OR n_picks >= n_candidates)",
    }
    for feature in sorted(BINARY_FEATURES & set(model_features[scope])):
        column = quote_identifier(feature)
        expressions[f"invalid_binary__{feature}"] = f"COUNT_IF({column} NOT IN (0, 1) OR {column} IS NULL)"
    if {"sentiment_positive", "sentiment_negative", "sentiment_neutral"} <= available:
        expressions["invalid_sentiment_simplex"] = (
            "COUNT_IF(abs(sentiment_positive + sentiment_negative + sentiment_neutral - 1.0) > 1e-5)"
        )
    if {"toxicity_probability", "toxicity_mean_probability"} <= available:
        expressions["toxicity_mean_above_maximum"] = (
            "COUNT_IF(toxicity_mean_probability > toxicity_probability + 1e-8)"
        )
    aliases = list(expressions)
    query = "SELECT " + ", ".join(
        f"{expression} AS {quote_identifier(alias)}" for alias, expression in expressions.items()
    ) + " FROM read_parquet(?)"
    values = connection.execute(query, [str(path)]).fetchdf().iloc[0]
    return pd.DataFrame(
        {"scope": scope, "check": aliases, "failures": [int(values[name]) for name in aliases]}
    )


def within_story_variation(
    scope: str,
    *,
    connection,
    choice_paths: dict[str, Path],
    model_features: dict[str, list[str]],
) -> pd.DataFrame:
    span_expressions = []
    for feature in model_features[scope]:
        column = quote_identifier(feature)
        span_expressions.append(
            f"MAX(CAST({column} AS DOUBLE)) - MIN(CAST({column} AS DOUBLE)) AS {quote_identifier(feature)}"
        )
    inner = (
        "SELECT story_id, " + ", ".join(span_expressions)
        + " FROM read_parquet(?) GROUP BY story_id"
    )
    outer = "SELECT COUNT(*) AS stories, " + ", ".join(
        f"COUNT_IF({quote_identifier(feature)} > 1e-12) AS {quote_identifier(feature)}"
        for feature in model_features[scope]
    ) + f" FROM ({inner})"
    values = connection.execute(outer, [str(choice_paths[scope])]).fetchdf().iloc[0]
    stories = int(values["stories"])
    return pd.DataFrame(
        [
            {
                "scope": scope, "feature": feature, "stories": stories,
                "stories_with_variation": int(values[feature]),
                "proportion_stories_with_variation": float(values[feature]) / stories,
            }
            for feature in model_features[scope]
        ]
    )


def feature_group(name: str, model_features: dict[str, list[str]]) -> str:
    author_history = {
        feature
        for feature in set().union(*map(set, model_features.values()))
        if feature.startswith("log_author_") or feature.startswith("author_prior_")
    }
    if name.startswith("aqua_"):
        return "aqua_expected"
    if name in TEXT_NLP:
        return "text_nlp"
    if name in author_history:
        return "author_history"
    if name in REPLY_STRUCTURE:
        return "reply_structure"
    return "semantic_timing_activity"


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    fig.savefig(output_root / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")


def plot_distribution_grid(
    scope: str,
    features: list[str],
    stem: str,
    *,
    samples: dict[str, pd.DataFrame],
    registry: dict,
    output_root: Path,
) -> None:
    columns = 3
    rows = math.ceil(len(features) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.1 * rows), squeeze=False)
    for axis, feature in zip(axes.flat, features):
        values = pd.to_numeric(samples[scope][feature], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        if feature in BINARY_FEATURES:
            counts = pd.Series(values).value_counts(normalize=True).sort_index()
            axis.bar(counts.index.astype(str), counts.values, color="#457b9d")
            axis.set_ylabel("Proportion")
        else:
            low, high = np.quantile(values, [0.005, 0.995])
            shown = np.clip(values, low, high) if high > low else values
            axis.hist(shown, bins=40, density=True, color="#457b9d", alpha=0.8)
            axis.axvline(np.median(values), color="#e76f51", linewidth=1.2)
            axis.set_xlabel(f"display clip [{low:.3g}, {high:.3g}]")
        axis.set_title(registry["features"][feature]["label"], fontsize=9)
    for axis in axes.flat[len(features):]:
        axis.set_visible(False)
    fig.suptitle(f"{scope.title()} candidates: {stem.replace('_', ' ')}", y=1.002)
    fig.tight_layout()
    save_figure(fig, output_root, f"{scope}_{stem}_distributions")
    plt.show()


def plot_aqua_expected_distributions(
    samples: dict[str, pd.DataFrame],
    features: list[str],
    registry: dict,
    output_root: Path,
) -> None:
    """Plot the root/all AQuA expected-score distributions."""
    columns = 4
    rows = math.ceil(len(features) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(14, 2.8 * rows), sharex=True, squeeze=False
    )
    for axis, feature in zip(axes.flat, features):
        for scope, color in (("root", "#457b9d"), ("all", "#e76f51")):
            values = pd.to_numeric(samples[scope][feature], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            axis.hist(
                values,
                bins=np.linspace(0, 3, 31),
                density=True,
                histtype="step",
                linewidth=1.3,
                color=color,
                label=scope,
            )
        axis.set_xlim(0, 3)
        axis.set_title(
            registry["features"][feature]["label"]
            .replace("AQuA ", "")
            .replace(" (raw expected ordinal score)", ""),
            fontsize=9,
        )
    for axis in axes.flat[len(features):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("AQuA expected ordinal distributions", y=1.002)
    fig.tight_layout()
    save_figure(fig, output_root, "aqua_expected_distributions_root_vs_all")
    plt.show()


def plot_spearman_correlation_heatmap(
    correlation: pd.DataFrame,
    scope: str,
    output_root: Path,
) -> None:
    """Render and save one upper-triangular Spearman heatmap."""
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    fig, axis = plt.subplots(figsize=(14, 14))
    import seaborn as sns

    sns.heatmap(
        correlation,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=True,
        yticklabels=True,
        ax=axis,
    )
    axis.tick_params(labelsize=6)
    axis.set_title(f"{scope.title()} candidates: sampled Spearman correlations")
    fig.tight_layout()
    save_figure(fig, output_root, f"{scope}_spearman_correlation_heatmap")
    plt.show()


def plot_selection_contrasts(
    selection_contrasts: pd.DataFrame,
    scope: str,
    registry: dict,
    output_root: Path,
) -> None:
    """Plot selected-versus-unselected standardized mean differences."""
    subset = selection_contrasts[selection_contrasts["scope"] == scope].copy()
    order = (
        subset.groupby("feature")["standardized_mean_difference"]
        .apply(lambda values: values.abs().max())
        .sort_values()
        .index
    )
    positions = {feature: index for index, feature in enumerate(order)}
    fig, axis = plt.subplots(figsize=(14, max(8, 0.25 * len(order))))
    for selector, marker, color, offset in (
        ("audience", "o", "#457b9d", -0.12),
        ("curator", "s", "#e76f51", 0.12),
    ):
        data = subset[subset["selector"] == selector]
        y = np.array([positions[name] for name in data["feature"]]) + offset
        axis.scatter(
            data["standardized_mean_difference"],
            y,
            marker=marker,
            color=color,
            label=selector,
            s=24,
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(
        range(len(order)),
        [registry["features"][name]["label"] for name in order],
        fontsize=7,
    )
    axis.set_xlabel("Selected minus unselected standardized mean difference")
    axis.set_title(f"{scope.title()} candidates: distribution separation")
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_root, f"{scope}_selection_standardized_mean_differences")
    plt.show()


def ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(left[np.isfinite(left)])
    right = np.sort(right[np.isfinite(right)])
    if not len(left) or not len(right):
        return np.nan
    values = np.sort(np.unique(np.concatenate([left, right])))
    left_cdf = np.searchsorted(left, values, side="right") / len(left)
    right_cdf = np.searchsorted(right, values, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))
