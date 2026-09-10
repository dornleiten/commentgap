"""Publication-table and figure assembly for the four-model workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pandas()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _save_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".tex").write_text(
        frame.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}"),
        encoding="utf-8",
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _bootstrap_article_metrics(
    frame: pd.DataFrame, *, draws: int = 1000, seed: int = 20260813
) -> pd.DataFrame:
    """Summarise article-level metrics with an article bootstrap."""
    rng = np.random.default_rng(seed)
    rows = []
    for selector, subset in frame.groupby("selector", sort=True):
        subset = subset.sort_values("story_id")
        for metric in ("ndcg_at_k", "top_k_overlap", "jaccard"):
            values = subset[metric].to_numpy(dtype=float)
            boot = np.mean(
                values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1
            )
            rows.append(
                {
                    "selector": selector,
                    "metric": metric,
                    "estimate": float(values.mean()),
                    "conf_low": float(np.quantile(boot, 0.025)),
                    "conf_high": float(np.quantile(boot, 0.975)),
                    "articles": len(values),
                    "bootstrap_draws": draws,
                }
            )
    return pd.DataFrame(rows)


def _paired_model_differences(
    regression: pd.DataFrame,
    xgboost: pd.DataFrame,
    *,
    draws: int = 1000,
    seed: int = 20260813,
) -> pd.DataFrame:
    """Estimate XGBoost minus regression differences on identical test articles."""
    keys = ["story_id", "selector"]
    regression_keys = regression[keys].sort_values(keys).reset_index(drop=True)
    xgboost_keys = xgboost[keys].sort_values(keys).reset_index(drop=True)
    if regression_keys.duplicated().any() or xgboost_keys.duplicated().any():
        raise ValueError("Test metrics must contain one row per story and selector")
    if not regression_keys.equals(xgboost_keys):
        raise ValueError(
            "Regression and XGBoost test metrics do not contain identical story/selector keys"
        )
    merged = regression.merge(xgboost, on=keys, suffixes=("_regression", "_xgboost"))
    rng = np.random.default_rng(seed)
    rows = []
    for selector, subset in merged.groupby("selector", sort=True):
        for metric in ("ndcg_at_k", "top_k_overlap", "jaccard"):
            differences = (
                subset[f"{metric}_xgboost"] - subset[f"{metric}_regression"]
            ).to_numpy(dtype=float)
            boot = np.mean(
                differences[
                    rng.integers(0, len(differences), size=(draws, len(differences)))
                ],
                axis=1,
            )
            rows.append(
                {
                    "selector": selector,
                    "metric": metric,
                    "estimate_xgboost_minus_regression": float(differences.mean()),
                    "conf_low": float(np.quantile(boot, 0.025)),
                    "conf_high": float(np.quantile(boot, 0.975)),
                    "articles": len(differences),
                    "bootstrap_draws": draws,
                }
            )
    return pd.DataFrame(rows)


def build_reporting_outputs(
    feature_root: Path,
    regression_root: Path,
    ranker_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    feature_root = Path(feature_root)
    regression_root = Path(regression_root)
    ranker_root = Path(ranker_root)
    output_root = Path(output_root)
    table_root = output_root / "tables"
    figure_root = output_root / "figures"
    sns.set_theme(style="whitegrid", context="paper")

    registry = json.loads((feature_root / "feature_manifest.json").read_text())
    provenance = json.loads((feature_root / "provenance_manifest.json").read_text())
    sample_rows = []
    descriptive_rows = []
    regression_rows = []
    probability_rows = []
    metric_rows = []
    regression_metric_rows = []
    paired_difference_rows = []
    regression_diagnostic_rows = []
    regression_tie_rows = []
    regression_chance_rows = []
    regression_above_chance_rows = []
    importance_rows = []
    shap_rows = []
    tie_rows = []
    fold_rows = []
    for scope in ("root", "all"):
        choice = _read_parquet(feature_root / f"choice_set_{scope}.parquet")
        sample_rows.append(
            {
                "scope": scope,
                "articles": choice["story_id"].nunique(),
                "candidate_comments": len(choice),
                "curator_selections": int(choice["curator_selected"].sum()),
                "mean_candidates": choice.groupby("story_id")["n_candidates"].first().mean(),
                "mean_picks": choice.groupby("story_id")["n_picks"].first().mean(),
            }
        )
        for feature in registry["models"][scope]["features"]:
            series = pd.to_numeric(choice[feature], errors="coerce")
            descriptive_rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "label": registry["features"][feature]["label"],
                    "mean": series.mean(),
                    "sd": series.std(),
                    "median": series.median(),
                    "missing": int(series.isna().sum()),
                }
            )
        associations = _read_csv(regression_root / scope / "selector_associations.csv")
        associations["scope"] = scope
        regression_rows.append(associations)
        probabilities = _read_csv(regression_root / scope / "probability_contrasts.csv")
        probabilities["scope"] = scope
        probability_rows.append(probabilities)
        regression_article_metrics = _read_parquet(
            regression_root / scope / "test_article_metrics.parquet"
        )
        xgb_article_metrics = _read_parquet(
            ranker_root / scope / "test_article_metrics.parquet"
        )
        scope_seed = 20260813 + (0 if scope == "root" else 100)
        regression_metric_summary = _read_parquet(
            regression_root / scope / "test_metric_summary.parquet"
        )
        regression_metric_summary["scope"] = scope
        regression_metric_rows.append(regression_metric_summary)
        regression_chance = _read_parquet(
            regression_root / scope / "test_chance_metric_summary.parquet"
        )
        regression_chance["scope"] = scope
        regression_chance_rows.append(regression_chance)
        regression_above_chance = _read_parquet(
            regression_root / scope / "test_above_chance_summary.parquet"
        )
        regression_above_chance["scope"] = scope
        regression_above_chance_rows.append(regression_above_chance)
        regression_diagnostics = _read_csv(
            regression_root / scope / "model_diagnostics.csv"
        )
        regression_diagnostics["scope"] = scope
        regression_diagnostic_rows.append(regression_diagnostics)
        regression_ties = _read_parquet(
            regression_root / scope / "test_tie_sensitivity_metrics.parquet"
        )
        regression_ties["scope"] = scope
        regression_tie_rows.append(regression_ties)
        paired_differences = _paired_model_differences(
            regression_article_metrics, xgb_article_metrics, seed=scope_seed
        )
        paired_differences["scope"] = scope
        paired_difference_rows.append(paired_differences)
        metrics = _read_parquet(ranker_root / scope / "test_metric_summary.parquet")
        metrics["scope"] = scope
        metric_rows.append(metrics)
        importance = _read_parquet(
            ranker_root / scope / "test_grouped_permutation_importance.parquet"
        )
        importance["scope"] = scope
        importance_rows.append(importance)
        shap = _read_parquet(ranker_root / scope / "test_treeshap_sample.parquet")
        id_columns = {"story_id", "comment_id", "selector", "split_role"}
        shap_long = shap.melt(
            id_vars=list(id_columns),
            value_vars=[column for column in shap if column not in id_columns and column != "selector_code"],
            var_name="feature",
            value_name="shap_value",
        )
        shap_long["scope"] = scope
        shap_rows.append(shap_long)
        ties = _read_parquet(ranker_root / scope / "test_tie_sensitivity_metrics.parquet")
        ties["scope"] = scope
        tie_rows.append(ties)
        splits = _read_parquet(ranker_root / scope / "article_split.parquet")
        folds = (
            splits.groupby(["split_role", "development_fold"], as_index=False)
            .agg(articles=("story_id", "nunique"), candidates=("n_candidates", "sum"))
        )
        folds["scope"] = scope
        fold_rows.append(folds)

    sample = pd.DataFrame(sample_rows)
    descriptive = pd.DataFrame(descriptive_rows)
    regression = pd.concat(regression_rows, ignore_index=True)
    probability = pd.concat(probability_rows, ignore_index=True)
    metrics = pd.concat(metric_rows, ignore_index=True)
    regression_metrics = pd.concat(regression_metric_rows, ignore_index=True)
    paired_differences = pd.concat(paired_difference_rows, ignore_index=True)
    regression_diagnostics = pd.concat(regression_diagnostic_rows, ignore_index=True)
    regression_ties = pd.concat(regression_tie_rows, ignore_index=True)
    regression_chance = pd.concat(regression_chance_rows, ignore_index=True)
    regression_above_chance = pd.concat(
        regression_above_chance_rows, ignore_index=True
    )
    importance = pd.concat(importance_rows, ignore_index=True)
    shap = pd.concat(shap_rows, ignore_index=True)
    ties = pd.concat(tie_rows, ignore_index=True)
    folds = pd.concat(fold_rows, ignore_index=True)
    split_balance = _read_csv(feature_root / "split_balance_diagnostics.csv")

    _save_table(sample, table_root / "sample_accounting")
    _save_table(descriptive, table_root / "feature_descriptives")
    _save_table(regression, table_root / "regression_associations")
    _save_table(probability, table_root / "regression_probability_contrasts")
    _save_table(regression_metrics, table_root / "regression_test_performance")
    _save_table(
        regression_chance, table_root / "regression_test_chance_baseline"
    )
    _save_table(
        regression_above_chance, table_root / "regression_test_above_chance"
    )
    _save_table(regression_diagnostics, table_root / "regression_model_diagnostics")
    _save_table(metrics, table_root / "xgb_test_performance")
    model_metrics = pd.concat(
        [
            regression_metrics.assign(model="Conditional logit"),
            metrics.assign(model="XGBoost"),
        ],
        ignore_index=True,
    )
    _save_table(model_metrics, table_root / "model_test_performance")
    _save_table(
        paired_differences, table_root / "regression_vs_xgb_test_differences"
    )
    importance_table = (
        importance.groupby(["scope", "feature"])["importance"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    _save_table(importance_table, table_root / "xgb_permutation_importance")
    _save_table(
        ties.groupby(["scope", "selector", "audience_tie_draw"], as_index=False)[
            ["ndcg_at_k", "top_k_overlap", "jaccard"]
        ].mean(),
        table_root / "tie_sensitivity",
    )
    regression_tie_summary = regression_ties.groupby(
        ["scope", "selector", "audience_tie_draw"], as_index=False
    )[["ndcg_at_k", "top_k_overlap", "jaccard"]].mean()
    _save_table(
        regression_tie_summary, table_root / "regression_test_tie_sensitivity"
    )
    combined_ties = pd.concat(
        [
            regression_tie_summary.assign(model="Conditional logit"),
            ties.groupby(
                ["scope", "selector", "audience_tie_draw"], as_index=False
            )[["ndcg_at_k", "top_k_overlap", "jaccard"]]
            .mean()
            .assign(model="XGBoost"),
        ],
        ignore_index=True,
    )
    _save_table(combined_ties, table_root / "model_test_tie_sensitivity")
    _save_table(folds, table_root / "held_out_split_balance")
    _save_table(split_balance, table_root / "held_out_covariate_balance")

    labels = registry["features"]
    maximum_regression_terms = max(
        int((regression["scope"] == scope).sum()) for scope in ("root", "all")
    )
    regression_figure_height = max(7.0, 0.27 * maximum_regression_terms)
    fig, axes = plt.subplots(
        1, 2, figsize=(14, regression_figure_height), sharex=True
    )
    for axis, scope in zip(axes, ("root", "all")):
        subset = regression[regression["scope"] == scope].copy()
        subset["label"] = subset["term"].map(lambda x: labels.get(x, {}).get("label", x))
        subset = subset.sort_values("curator_minus_audience_log_odds")
        y = np.arange(len(subset))
        axis.errorbar(
            subset["audience_log_odds"],
            y - 0.18,
            xerr=[
                subset["audience_log_odds"] - subset["audience_conf_low"],
                subset["audience_conf_high"] - subset["audience_log_odds"],
            ],
            fmt="o",
            color="#6c757d",
            capsize=2,
            label="Audience",
        )
        axis.errorbar(
            subset["curator_log_odds"],
            y,
            xerr=[
                subset["curator_log_odds"] - subset["curator_conf_low"],
                subset["curator_conf_high"] - subset["curator_log_odds"],
            ],
            fmt="s",
            color="#2a9d8f",
            capsize=2,
            label="Curator",
        )
        axis.errorbar(
            subset["curator_minus_audience_log_odds"],
            y + 0.18,
            xerr=[
                subset["curator_minus_audience_log_odds"] - subset["difference_conf_low"],
                subset["difference_conf_high"] - subset["curator_minus_audience_log_odds"],
            ],
            fmt="o",
            color="#2a6f97",
            capsize=2,
            label="Curator − audience",
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, subset["label"])
        axis.set_title("Root candidates" if scope == "root" else "All comments")
        axis.set_xlabel("Log-odds association")
        axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save_figure(fig, figure_root / "regression_preference_differences")

    plot_metrics = metrics[metrics["metric"].isin(["ndcg_at_k", "top_k_overlap", "jaccard"])].copy()
    fig, axis = plt.subplots(figsize=(6.6, 4.5))
    plot_metrics["model"] = plot_metrics["scope"] + " / " + plot_metrics["selector"]
    metric_order = ["top_k_overlap", "jaccard", "ndcg_at_k"]
    metric_labels = ["Top-k overlap", "Jaccard", "nDCG@k"]
    models = sorted(plot_metrics["model"].unique())
    x = np.arange(len(metric_order), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(models))
    palette = sns.color_palette("deep", len(models))
    for offset, color, model in zip(offsets, palette, models):
        subset = plot_metrics[plot_metrics["model"] == model].set_index("metric").loc[metric_order]
        axis.errorbar(
            x + offset,
            subset["estimate"],
            yerr=[
                subset["estimate"] - subset["conf_low"],
                subset["conf_high"] - subset["estimate"],
            ],
            marker="o",
            linestyle="none",
            capsize=3,
            color=color,
            label=model,
        )
    upper = min(1.0, max(0.1, float(plot_metrics["conf_high"].max()) * 1.2))
    axis.set_ylim(0, upper)
    axis.set_xticks(x, metric_labels)
    axis.set_ylabel("Article-weighted sealed-test estimate")
    axis.set_xlabel("")
    axis.legend(title="Model / selector", frameon=False)
    fig.tight_layout()
    _save_figure(fig, figure_root / "xgb_test_performance")

    comparison = paired_differences.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for axis, scope in zip(axes, ("root", "all")):
        subset = comparison[comparison["scope"] == scope].copy()
        subset["series"] = subset["selector"].str.title()
        x = np.arange(len(metric_order), dtype=float)
        for offset, selector, color in zip(
            (-0.08, 0.08),
            ("Audience", "Curator"),
            sns.color_palette("deep", 2),
        ):
            values = (
                subset[subset["series"] == selector]
                .set_index("metric")
                .loc[metric_order]
            )
            estimate = values["estimate_xgboost_minus_regression"]
            axis.errorbar(
                x + offset,
                estimate,
                yerr=[
                    estimate - values["conf_low"],
                    values["conf_high"] - estimate,
                ],
                marker="o",
                linestyle="none",
                capsize=3,
                color=color,
                label=selector,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, metric_labels)
        axis.set_title("Root candidates" if scope == "root" else "All comments")
        axis.set_xlabel("")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Sealed-test difference (XGBoost − conditional logit)")
    fig.tight_layout()
    _save_figure(fig, figure_root / "regression_vs_xgb_test_performance")

    importance_summary = importance.groupby(["scope", "feature"], as_index=False)["importance"].mean()
    importance_summary["label"] = importance_summary["feature"].map(
        lambda x: labels.get(x, {}).get("label", x)
    )
    keep = (
        importance_summary.sort_values("importance", ascending=False)
        .groupby("scope", group_keys=False)
        .head(10)
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, scope in zip(axes, ("root", "all")):
        subset = keep[keep["scope"] == scope].sort_values("importance")
        axis.barh(subset["label"], subset["importance"], color="#61a5c2")
        axis.set_title("Root candidates" if scope == "root" else "All comments")
        axis.set_xlabel("Decrease in macro nDCG@k")
    fig.tight_layout()
    _save_figure(fig, figure_root / "xgb_grouped_permutation_importance")

    shap_summary = (
        shap.assign(abs_shap=lambda x: x["shap_value"].abs())
        .groupby(["scope", "selector", "feature"], as_index=False)["abs_shap"]
        .mean()
    )
    shap_summary["label"] = shap_summary["feature"].map(
        lambda x: labels.get(x, {}).get("label", x)
    )
    top = (
        shap_summary.sort_values("abs_shap", ascending=False)
        .groupby(["scope", "selector"], group_keys=False)
        .head(5)
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, ((scope, selector), subset) in zip(axes.flat, top.groupby(["scope", "selector"], sort=True)):
        subset = subset.sort_values("abs_shap")
        axis.barh(subset["label"], subset["abs_shap"], color="#e07a5f")
        axis.set_title(f"{scope.title()} / {selector.title()}")
        axis.set_xlabel("Mean |TreeSHAP contribution|")
    fig.tight_layout()
    _save_figure(fig, figure_root / "xgb_treeshap_top_features")

    manifest = {
        "watermark": provenance["watermark"],
        "tables": sorted(path.name for path in table_root.glob("*")),
        "figures": sorted(path.name for path in figure_root.glob("*")),
    }
    (output_root / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
