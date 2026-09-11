"""Reusable Paper 1 model reporting figures.

The plotting functions accept the already prepared reporting tables.  This
keeps notebook display code tied to the same data transformations as the
batch reporter, without reading a previously rendered image back from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .plotting import save_display_figure


_MODEL_VARIANT_SPECS = (
    ("xgboost", "metadata", "XGB", "o", "#08519C"),
    ("xgboost", "metadata_bge", "XGB-T", "^", "#6BAED6"),
    ("neural", "metadata", "NN", "o", "#A63603"),
    ("neural", "metadata_bge", "NN-T", "^", "#FDAE6B"),
)


def _scope_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "scope" not in frame or frame["scope"].eq("all").any():
        return frame[frame["scope"].eq("all")].copy() if "scope" in frame else frame.copy()
    return frame.copy()


def _finish(figure: Any, output_root: Path | None, stem: str, *, show: bool):
    return save_display_figure(figure, output_root, stem, show=show)


def _horizontal_spread(axis: Any, x: pd.Series, y: Any, low: pd.Series, high: pd.Series, color: str):
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(low) & np.isfinite(high)
    axis.hlines(
        np.asarray(y)[finite], np.asarray(low)[finite], np.asarray(high)[finite],
        color=color, linewidth=1.4, alpha=0.85, zorder=1,
    )


def _vertical_spread(axis: Any, x: pd.Series, y: pd.Series, low: pd.Series, high: pd.Series, color: str):
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(low) & np.isfinite(high)
    axis.vlines(
        np.asarray(x)[finite], np.asarray(low)[finite], np.asarray(high)[finite],
        color=color, linewidth=1.1, alpha=0.65, zorder=1,
    )


def _label_column(frame: pd.DataFrame) -> pd.Series:
    if "feature_label" in frame:
        return frame["feature_label"]
    features = frame["feature"].astype(str)
    try:
        from .features import _feature_registry

        labels = {
            name: metadata["label"]
            for name, metadata in _feature_registry(aqua_available=True)["features"].items()
        }
    except (ImportError, KeyError):
        labels = {}
    return features.map(labels).fillna(features.str.replace("_", " ", regex=False).str.capitalize())


def _clean_feature_labels(frame: pd.DataFrame) -> pd.Series:
    """Return feature labels without the implementation detail in AQuA names."""
    return _label_column(frame).astype(str).str.replace(
        " (raw expected ordinal score)", "", regex=False,
    )


def _largest_regression_coefficient_rows(
    associations: pd.DataFrame, *, limit: int = 20
) -> pd.DataFrame:
    """Select coefficient rows from the complete association table."""
    required = {
        "audience_log_odds", "curator_log_odds", "audience_conf_low",
        "audience_conf_high", "curator_conf_low", "curator_conf_high",
    }
    if not required.issubset(associations.columns):
        return associations.iloc[0:0].copy()
    if limit < 1:
        raise ValueError("limit must be positive")
    output = associations.copy()
    output["absolute"] = output[["audience_log_odds", "curator_log_odds"]].abs().max(axis=1)
    return output.nlargest(limit, "absolute").sort_values("audience_log_odds")


def plot_regression_selector_differences(
    associations: pd.DataFrame,
    output_root: Path | None = None,
    *,
    show: bool = True,
    limit: int = 20,
):
    """Plot the largest curator-minus-audience regression associations."""
    import matplotlib.pyplot as plt

    shown = _scope_frame(associations)
    shown = shown.assign(absolute=shown["curator_minus_audience_log_odds"].abs())
    shown = shown.nlargest(limit, "absolute").sort_values("curator_minus_audience_log_odds")
    fig, axis = plt.subplots(figsize=(6.6, max(5, 0.32 * len(shown))))
    y = np.arange(len(shown))
    axis.errorbar(
        shown["curator_minus_audience_log_odds"], y,
        xerr=[
            shown["curator_minus_audience_log_odds"] - shown["difference_conf_low"],
            shown["difference_conf_high"] - shown["curator_minus_audience_log_odds"],
        ], fmt="o", color="#4b8b6f", capsize=2,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="0.9", linewidth=0.6)
    axis.set_yticks(y, _clean_feature_labels(shown))
    axis.set(
        xlabel="Log-odds gap (editor − audience)",
        title="Largest stacked-model selector differences",
    )
    return _finish(fig, output_root, "all_regression_selector_differences", show=show)


def plot_regression_selector_coefficients(
    associations: pd.DataFrame,
    output_root: Path | None = None,
    *,
    show: bool = True,
    limit: int = 20,
):
    """Plot large selector coefficients, selecting from all association rows."""
    import matplotlib.pyplot as plt

    required = {
        "audience_log_odds", "curator_log_odds", "audience_conf_low",
        "audience_conf_high", "curator_conf_low", "curator_conf_high",
    }
    if not required.issubset(associations.columns):
        return None
    shown = _largest_regression_coefficient_rows(_scope_frame(associations), limit=limit)
    fig, axis = plt.subplots(figsize=(6.6, max(5, 0.32 * len(shown))))
    y = np.arange(len(shown))
    for offset, estimate, low, high, color, label in (
        (-0.12, "audience_log_odds", "audience_conf_low", "audience_conf_high", "#356a8a", "Audience"),
        (0.12, "curator_log_odds", "curator_conf_low", "curator_conf_high", "#4b8b6f", "Editor"),
    ):
        axis.errorbar(
            shown[estimate], y + offset,
            xerr=[shown[estimate] - shown[low], shown[high] - shown[estimate]],
            fmt="o", color=color, capsize=2, label=label,
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, _label_column(shown))
    axis.set(xlabel="Coefficient (log-odds; 95% CI)", title="Largest stacked-model coefficients")
    axis.legend(frameon=False)
    return _finish(fig, output_root, "all_regression_selector_coefficients", show=show)


def plot_winner_permutation_importance_gaps(
    permutation_gaps: pd.DataFrame,
    output_root: Path | None = None,
    *,
    show: bool = True,
):
    """Plot selector-specific permutation-importance gaps."""
    import matplotlib.pyplot as plt

    shown = _scope_frame(permutation_gaps)
    if shown.empty:
        return None
    shown = shown.assign(absolute_gap=shown["permutation_importance_gap"].abs())
    shown = (
        shown.sort_values(["model_family", "absolute_gap"], ascending=[True, False])
        .groupby("model_family", group_keys=False).head(15)
        .sort_values("permutation_importance_gap")
    )
    shown = shown.assign(display=_label_column(shown).radd(shown["model_label"] + " — "))
    fig, axis = plt.subplots(figsize=(6.6, max(5, 0.3 * len(shown))))
    y = np.arange(len(shown))
    axis.errorbar(
        shown["permutation_importance_gap"], y,
        xerr=[shown["permutation_importance_gap"] - shown["conf_low"], shown["conf_high"] - shown["permutation_importance_gap"]],
        fmt="o", color="#4b8b6f", capsize=2,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="0.9", linewidth=0.6)
    axis.set_yticks(y, shown["display"])
    axis.set(xlabel="Permutation-importance gap (curator − audience nDCG loss)", title="Largest selector-specific permutation-importance gaps")
    return _finish(fig, output_root, "all_winner_permutation_importance_gaps", show=show)


def plot_winner_permutation_importance(
    permutation_gaps: pd.DataFrame,
    output_root: Path | None = None,
    *,
    show: bool = True,
):
    """Plot large selector-specific permutation importances."""
    import matplotlib.pyplot as plt

    required = {"audience_importance", "curator_importance"}
    if not required.issubset(permutation_gaps.columns):
        return None
    shown = _scope_frame(permutation_gaps)
    if shown.empty:
        return None
    shown = shown.assign(absolute_importance=shown[["audience_importance", "curator_importance"]].abs().max(axis=1))
    shown = (
        shown.sort_values(["model_family", "absolute_importance"], ascending=[True, False])
        .groupby("model_family", group_keys=False).head(15)
        .sort_values("absolute_importance")
    )
    shown = shown.assign(display=_label_column(shown).radd(shown["model_label"] + " — "))
    fig, axis = plt.subplots(figsize=(6.6, max(5, 0.3 * len(shown))))
    y = np.arange(len(shown))
    axis.plot(shown["audience_importance"], y - 0.12, "o", color="#356a8a", label="Audience")
    axis.plot(shown["curator_importance"], y + 0.12, "o", color="#4b8b6f", label="Editor")
    axis.set_yticks(y, shown["display"])
    axis.set(xlabel="Mean permutation importance (nDCG loss)", title="Largest selector-specific permutation importances")
    axis.legend(frameon=False)
    return _finish(fig, output_root, "all_winner_permutation_importance", show=show)


def _shap_rows(shap_summary: pd.DataFrame, value_column: str) -> pd.DataFrame:
    shown = _scope_frame(shap_summary)
    if shown.empty:
        return shown
    groups = [column for column in ("model_family", "feature_set") if column in shown]
    shown = shown.assign(absolute_importance=shown[["mean_shap_audience", "mean_shap_curator"]].abs().max(axis=1))
    return (
        shown.sort_values(groups + [value_column], ascending=[True] * len(groups) + [False])
        .groupby(groups, group_keys=False).head(15)
        .sort_values(value_column)
    )


def _shap_display(frame: pd.DataFrame) -> pd.Series:
    model = frame["model_family"].str.title()
    feature_set = frame["feature_set"].str.replace("_", "+", regex=False)
    return model + " " + feature_set + " — " + _label_column(frame)


def plot_winner_shap_importance(
    shap_summary: pd.DataFrame,
    output_root: Path | None = None,
    *,
    show: bool = True,
):
    """Plot large selector-specific signed SHAP contributions."""
    import matplotlib.pyplot as plt

    required = {"mean_shap_audience", "mean_shap_curator", "mean_shap_gap"}
    if not required.issubset(shap_summary.columns):
        return None
    shown = _shap_rows(shap_summary, "absolute_importance")
    if shown.empty:
        return None
    shown = shown.assign(display=_shap_display(shown))
    fig, axis = plt.subplots(figsize=(6.6, max(5, 0.3 * len(shown))))
    y = np.arange(len(shown))
    has_shap_spreads = {
        "mean_shap_audience_story_q25", "mean_shap_audience_story_q75",
        "mean_shap_curator_story_q25", "mean_shap_curator_story_q75",
    }.issubset(shown.columns)
    if has_shap_spreads:
        for offset, estimate, low, high, color in (
            (-0.12, "mean_shap_audience", "mean_shap_audience_story_q25", "mean_shap_audience_story_q75", "#356a8a"),
            (0.12, "mean_shap_curator", "mean_shap_curator_story_q25", "mean_shap_curator_story_q75", "#4b8b6f"),
        ):
            _horizontal_spread(
                axis, shown[estimate], y + offset, shown[low], shown[high], color,
            )
    axis.plot(shown["mean_shap_audience"], y - 0.12, "o", color="#356a8a", label="Audience")
    axis.plot(shown["mean_shap_curator"], y + 0.12, "o", color="#4b8b6f", label="Editor")
    axis.set_yticks(y, _clean_feature_labels(shown["display"].to_frame(name="feature_label")))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="0.9", linewidth=0.6)
    axis.set(xlabel="Mean signed SHAP contribution (story IQR)", title="Largest selector-specific signed SHAP contributions")
    axis.legend(frameon=False)
    return _finish(fig, output_root, "all_winner_shap_importance", show=show)


def plot_winner_shap_importance_gaps(
    shap_summary: pd.DataFrame,
    output_root: Path | None = None,
    *,
    show: bool = True,
    limit: int = 40,
):
    """Plot large signed SHAP gaps in separate XGB and NN panels."""
    import matplotlib.pyplot as plt

    required = {"mean_shap_audience", "mean_shap_curator", "mean_shap_gap"}
    if not required.issubset(shap_summary.columns):
        return None
    if limit < 2 or limit % 2:
        raise ValueError("limit must be a positive even number")
    shown = _scope_frame(shap_summary)
    if shown.empty:
        return None
    family_specs = (
        ("xgboost", "XGB", "all_winner_shap_importance_gaps_xgb"),
        ("neural", "NN", "all_winner_shap_importance_gaps_nn"),
    )
    family_limit = limit // 2
    figures = []
    for family, family_label, stem in family_specs:
        family_shown = shown[shown["model_family"].eq(family)].copy()
        if family_shown.empty:
            continue
        family_shown = (
            family_shown.assign(absolute_gap=family_shown["mean_shap_gap"].abs())
            .nlargest(family_limit, "absolute_gap")
            .sort_values("mean_shap_gap")
            .reset_index(drop=True)
            .assign(display=lambda frame: _clean_feature_labels(frame))
        )
        fig, axis = plt.subplots(figsize=(6.6, max(5, 0.3 * len(family_shown))))
        y = np.arange(len(family_shown))
        for variant_family, feature_set, label, marker, color in _MODEL_VARIANT_SPECS:
            if variant_family != family:
                continue
            panel = family_shown[family_shown["feature_set"].eq(feature_set)]
            if panel.empty:
                continue
            if {
                "mean_shap_gap_story_q25", "mean_shap_gap_story_q75",
            }.issubset(panel.columns):
                _horizontal_spread(
                    axis, panel["mean_shap_gap"], panel.index.to_numpy(),
                    panel["mean_shap_gap_story_q25"], panel["mean_shap_gap_story_q75"], color,
                )
            axis.plot(
                panel["mean_shap_gap"], panel.index.to_numpy(),
                linestyle="None", marker=marker, color=color, label=label,
            )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="0.9", linewidth=0.6)
        axis.set_yticks(y, family_shown["display"])
        axis.set(
            xlabel="Mean signed SHAP gap (editor − audience; story IQR)",
            title=f"Largest {family_label} selector-specific signed SHAP gaps",
        )
        axis.legend(title="Model variant", frameon=False)
        figures.append(_finish(fig, output_root, stem, show=show))
    return figures


def plot_regression_vs_shap_gaps(
    regression_feature_gaps: pd.DataFrame,
    shap_summary: pd.DataFrame,
    output_root: Path | None = None,
    *,
    show: bool = True,
):
    """Plot regression log-odds gaps against signed SHAP gaps."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    regression_required = {"term", "feature_gap_log_odds"}
    shap_required = {"feature", "model_family", "feature_set", "mean_shap_gap"}
    if not regression_required.issubset(regression_feature_gaps.columns):
        return None
    if not shap_required.issubset(shap_summary.columns):
        return None

    regression = _scope_frame(regression_feature_gaps)[
        ["term", "feature_gap_log_odds"]
    ].rename(columns={"term": "feature_term", "feature_gap_log_odds": "regression_gap"})
    shap = _scope_frame(shap_summary).copy()
    shown = shap.merge(
        regression,
        left_on="feature",
        right_on="feature_term",
        how="inner",
        validate="many_to_one",
    ).dropna(subset=["regression_gap", "mean_shap_gap"])
    if shown.empty:
        return None

    fig, axis = plt.subplots(figsize=(6.6, 6.6))
    for family, feature_set, label, marker, color in _MODEL_VARIANT_SPECS:
        panel = shown[
            shown["model_family"].eq(family)
            & shown["feature_set"].eq(feature_set)
        ]
        if panel.empty:
            continue
        axis.scatter(
            panel["regression_gap"],
            panel["mean_shap_gap"],
            s=42,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
            marker=marker,
            label=label,
        )
    axis.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
    axis.axvline(0, color="0.5", linewidth=0.8, linestyle=":")
    axis.set(
        xlabel="Log-odds gap (editor − audience)",
        ylabel="Mean signed SHAP gap (editor − audience)",
        title="Regression log odds gaps vs ML SHAP gaps",
    )
    axis.grid(color="0.9", linewidth=0.6)
    zoom = inset_axes(
        axis,
        width="43%",
        height="43%",
        loc="lower right",
        bbox_to_anchor=(0, 0.08, 1, 1),
        bbox_transform=axis.transAxes,
        borderpad=1.2,
    )
    for family, feature_set, label, marker, color in _MODEL_VARIANT_SPECS:
        panel = shown[
            shown["model_family"].eq(family)
            & shown["feature_set"].eq(feature_set)
        ]
        if panel.empty:
            continue
        zoom.scatter(
            panel["regression_gap"],
            panel["mean_shap_gap"],
            s=25,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.3,
            marker=marker,
        )
    zoom.axhline(0, color="0.5", linewidth=0.6, linestyle=":")
    zoom.axvline(0, color="0.5", linewidth=0.6, linestyle=":")
    zoom.set_xlim(-0.1, 0.1)
    zoom.set_ylim(-0.1, 0.1)
    zoom.set_xticks(np.arange(-0.1, 0.1001, 0.05))
    zoom.set_yticks(np.arange(-0.1, 0.1001, 0.05))
    zoom.text(
        0.03, 0.97, "Central cluster", transform=zoom.transAxes,
        ha="left", va="top", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )
    zoom.tick_params(labelsize=7, pad=1)
    zoom.grid(color="0.9", linewidth=0.5)
    mark_inset(
        axis, zoom, loc1=1, loc2=3,
        fc="none", ec="0.35", linewidth=0.9,
    )
    legend_handles = [
        Line2D([0], [0], marker=marker, linestyle="", color=color,
               markerfacecolor=color, markeredgecolor=color, label=label)
        for family, feature_set, label, marker, color in _MODEL_VARIANT_SPECS
        if not shown[
            shown["model_family"].eq(family)
            & shown["feature_set"].eq(feature_set)
        ].empty
    ]
    axis.legend(handles=legend_handles, title="Model variant", frameon=False)
    return _finish(fig, output_root, "all_regression_vs_shap_gaps", show=show)
