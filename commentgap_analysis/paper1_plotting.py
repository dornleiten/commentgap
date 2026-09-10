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


def _scope_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "scope" not in frame or frame["scope"].eq("all").any():
        return frame[frame["scope"].eq("all")].copy() if "scope" in frame else frame.copy()
    return frame.copy()


def _finish(figure: Any, output_root: Path | None, stem: str, *, show: bool):
    return save_display_figure(figure, output_root, stem, show=show)


def _label_column(frame: pd.DataFrame) -> pd.Series:
    if "feature_label" in frame:
        return frame["feature_label"]
    return frame["feature"]


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
    axis.set_yticks(y, _label_column(shown))
    axis.set(
        xlabel="Curator minus audience log-odds association",
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
    axis.plot(shown["mean_shap_audience"], y - 0.12, "o", color="#356a8a", label="Audience")
    axis.plot(shown["mean_shap_curator"], y + 0.12, "o", color="#4b8b6f", label="Editor")
    axis.set_yticks(y, shown["display"])
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Mean signed SHAP contribution", title="Largest selector-specific signed SHAP contributions")
    axis.legend(frameon=False)
    return _finish(fig, output_root, "all_winner_shap_importance", show=show)


def plot_winner_shap_importance_gaps(
    shap_summary: pd.DataFrame,
    output_root: Path | None = None,
    *,
    show: bool = True,
):
    """Plot large editor-minus-audience signed SHAP gaps."""
    import matplotlib.pyplot as plt

    required = {"mean_shap_audience", "mean_shap_curator", "mean_shap_gap"}
    if not required.issubset(shap_summary.columns):
        return None
    shown = _scope_frame(shap_summary)
    if shown.empty:
        return None
    groups = [column for column in ("model_family", "feature_set") if column in shown]
    shown = shown.assign(absolute_gap=shown["mean_shap_gap"].abs())
    shown = (
        shown.sort_values(groups + ["absolute_gap"], ascending=[True] * len(groups) + [False])
        .groupby(groups, group_keys=False).head(15)
        .sort_values("mean_shap_gap")
        .assign(display=lambda frame: _shap_display(frame))
    )
    fig, axis = plt.subplots(figsize=(6.6, max(5, 0.3 * len(shown))))
    y = np.arange(len(shown))
    axis.plot(shown["mean_shap_gap"], y, "o", color="#76528b")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, shown["display"])
    axis.set(xlabel="Mean signed SHAP gap (editor − audience)", title="Largest selector-specific signed SHAP gaps")
    return _finish(fig, output_root, "all_winner_shap_importance_gaps", show=show)
