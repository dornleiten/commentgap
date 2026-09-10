"""Plotting helpers for FORUM correlation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import pandas as pd
import seaborn as sns


def _show(show: bool) -> None:
    if show:
        plt.show()


def _identity_line(axis, x, y) -> None:
    values = pd.concat([x, y]).dropna()
    if len(values):
        lo, hi = values.min(), values.max()
        axis.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, alpha=0.7)


def plot_top10_full_forum(
    frame: pd.DataFrame,
    feature_order: list[str],
    feature_palette: Mapping[str, str],
    reply_markers: Mapping[str, str],
    reply_labels: Mapping[str, str],
    *,
    show: bool = True,
):
    """Plot top-10 against full-list FORUM across ranking conditions."""
    fig, axis = plt.subplots(figsize=(14, 10))
    for feature in feature_order:
        for reply_mode, marker in reply_markers.items():
            for pinned in (False, True):
                panel = frame[
                    frame["feature_label"].eq(feature)
                    & frame["reply_mode"].eq(reply_mode)
                    & frame["pinned"].eq(pinned)
                ]
                if panel.empty:
                    continue
                color = feature_palette[feature]
                axis.scatter(
                    panel["forum_full"], panel["forum_top10"], marker=marker,
                    s=58, alpha=0.62,
                    facecolors=color if pinned else "none",
                    edgecolors=color, linewidths=1,
                )
    _identity_line(axis, frame["forum_full"], frame["forum_top10"])
    axis.set_xlabel("Full-list FORUM")
    axis.set_ylabel("Top-10 FORUM")
    axis.set_title("Top-10 versus full-list FORUM across ranking conditions")
    feature_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color,
               markerfacecolor=color, label=feature)
        for feature, color in feature_palette.items()
    ]
    reply_handles = [
        Line2D([0], [0], marker=marker, linestyle="", color="black",
               label=reply_labels[reply])
        for reply, marker in reply_markers.items()
    ]
    pin_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="black",
               markerfacecolor="none", label="Unpinned"),
        Line2D([0], [0], marker="o", linestyle="", color="black",
               markerfacecolor="black", label="Pinned"),
    ]
    fig.legend(handles=feature_handles, title="Feature", loc="upper left",
               bbox_to_anchor=(0.75, 0.9), borderaxespad=0.0, ncol=1,
               handletextpad=0.7, labelspacing=0.5)
    fig.legend(handles=reply_handles, title="Reply mode", loc="upper left",
               bbox_to_anchor=(0.75, 0.48), borderaxespad=0.0,
               handletextpad=0.7, labelspacing=0.6)
    fig.legend(handles=pin_handles, title="Pin status", loc="upper left",
               bbox_to_anchor=(0.75, 0.3), borderaxespad=0.0,
               handletextpad=0.7, labelspacing=0.6)
    fig.tight_layout(rect=(0, 0, 0.75, 1))
    _show(show)
    return fig


def plot_forum_ndcg(
    frame: pd.DataFrame,
    feature_order: list[str],
    feature_palette: Mapping[str, str],
    reply_markers: Mapping[str, str],
    reply_labels: Mapping[str, str],
    *,
    show: bool = True,
):
    """Plot FORUM against nDCG separately for top-10 and full-list depth."""
    depth_order = ["top10", "full"]
    depth_labels = {"top10": "Top 10", "full": "Full list"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharex=True, sharey=True)
    for axis, depth in zip(axes, depth_order):
        panel = frame[frame["depth"].eq(depth)]
        for feature in feature_order:
            for reply_mode, marker in reply_markers.items():
                for pinned in (False, True):
                    points = panel[
                        panel["feature_label"].eq(feature)
                        & panel["reply_mode"].eq(reply_mode)
                        & panel["pinned"].eq(pinned)
                    ]
                    if points.empty:
                        continue
                    color = feature_palette[feature]
                    axis.scatter(
                        points["forum"], points["ndcg"], marker=marker,
                        s=58, alpha=0.62,
                        facecolors=color if pinned else "none",
                        edgecolors=color, linewidths=2,
                    )
            feature_points = panel[panel["feature_label"].eq(feature)]
            if len(feature_points) >= 3 and feature_points["forum"].nunique() > 1:
                sns.regplot(
                    data=feature_points, x="forum", y="ndcg", scatter=False,
                    color=feature_palette[feature],
                    line_kws={"linestyle": "--", "linewidth": 1}, ax=axis,
                )
        axis.set_xlim(-0.6, 0.6)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Mean FORUM")
        axis.set_ylabel("Mean nDCG" if axis is axes[0] else "")
        axis.set_title(f"{depth_labels[depth]}")
    _add_common_legends(fig, feature_palette, reply_markers, reply_labels,
                        bbox_to_anchor=(0.82, 0.90), reply_anchor=(0.82, 0.42),
                        pin_anchor=(0.82, 0.2))
    fig.suptitle("FORUM versus nDCG", y=0.92)
    fig.tight_layout(rect=(0, 0, 0.82, 0.94), w_pad=1.2)
    _show(show)
    return fig


def _common_handles(feature_palette, reply_markers, reply_labels):
    feature_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color,
               markerfacecolor=color, label=feature)
        for feature, color in feature_palette.items()
    ]
    reply_handles = [
        Line2D([0], [0], marker=marker, linestyle="", color="black",
               label=reply_labels[reply])
        for reply, marker in reply_markers.items()
    ]
    pin_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="black",
               markerfacecolor="none", label="Unpinned"),
        Line2D([0], [0], marker="o", linestyle="", color="black",
               markerfacecolor="black", label="Pinned"),
    ]
    return feature_handles, reply_handles, pin_handles


def _add_common_legends(fig, feature_palette, reply_markers, reply_labels,
                        *, bbox_to_anchor, reply_anchor, pin_anchor):
    feature_handles, reply_handles, pin_handles = _common_handles(
        feature_palette, reply_markers, reply_labels
    )
    fig.legend(handles=feature_handles, title="Feature", loc="upper left",
               bbox_to_anchor=bbox_to_anchor, borderaxespad=0.0, ncol=1,
               handletextpad=0.7, labelspacing=0.5)
    fig.legend(handles=reply_handles, title="Reply mode", loc="upper left",
               bbox_to_anchor=reply_anchor, borderaxespad=0.0,
               handletextpad=0.7, labelspacing=0.6)
    fig.legend(handles=pin_handles, title="Pin status", loc="upper left",
               bbox_to_anchor=pin_anchor, borderaxespad=0.0,
               handletextpad=0.7, labelspacing=0.6)


def plot_regression_forum_coefficients(
    frame: pd.DataFrame,
    *,
    show: bool = True,
):
    """Plot regression-ranking FORUM against coefficient odds ratios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    selector_order = ["audience", "editor"]
    selector_labels = {"audience": "Audience model", "editor": "Editor model"}
    depth_order = ["top10", "full"]
    depth_palette = {"top10": "#1f77b4", "full": "#d62728"}
    depth_labels = {"top10": "Top 10", "full": "Full list"}
    category_order = [
        "AQuA", "Text / NLP\nfeatures", "Author history",
        "Timing / discussion\ncontext", "Comment form",
    ]
    category_markers = {
        "AQuA": "o", "Text / NLP\nfeatures": "s", "Author history": "^",
        "Timing / discussion\ncontext": "D", "Comment form": "P",
    }
    for axis, selector in zip(axes, selector_order):
        panel = frame[frame["selector"].eq(selector)]
        sns.scatterplot(
            data=panel, x="forum", y="odds_ratio", hue="depth",
            style="feature_category", hue_order=depth_order,
            style_order=category_order, palette=depth_palette,
            markers=category_markers, s=110, ax=axis,
        )
        x_low, x_high = -1, 1
        y_low, y_high = panel["odds_ratio"].min(), panel["odds_ratio"].max()
        x_margin = 0.08 * max(x_high - x_low, 1e-6)
        y_margin = 0.08 * max(y_high - y_low, 1e-6)
        x_limits = (min(x_low, 0) - x_margin, max(x_high, 0) + x_margin)
        y_limits = (max(0, min(y_low, 1) - y_margin), max(y_high, 1) + y_margin)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        quadrant_specs = [
            (x_limits[0], 0, y_limits[0], 1, "#2ca02c", "Agreement\n(FORUM −, coefficient −)"),
            (0, x_limits[1], 1, y_limits[1], "#2ca02c", "Agreement\n(FORUM +, coefficient +)"),
            (x_limits[0], 0, 1, y_limits[1], "#d62728", "Disagreement\n(FORUM −, coefficient +)"),
            (0, x_limits[1], y_limits[0], 1, "#d62728", "Disagreement\n(FORUM +, coefficient −)"),
        ]
        for left, right, bottom, top, color, label in quadrant_specs:
            axis.add_patch(Rectangle(
                (left, bottom), right - left, top - bottom,
                facecolor=color, edgecolor="none", alpha=0.22, zorder=0,
            ))
            label_y = bottom + (0.42 if bottom >= 1 else 0.50) * (top - bottom)
            axis.text(
                (left + right) / 2, label_y, label,
                ha="center", va="center", fontsize=10,
                fontweight="semibold", color=color, alpha=0.9, zorder=1,
            )
        axis.axhline(1, color="black", linestyle=":", linewidth=1)
        axis.axvline(0, color="black", linestyle=":", linewidth=1)
        axis.set_title(selector_labels[selector])
        axis.set_xlabel("FORUM on feature under regression ranking")
        axis.set_ylabel("Regression odds ratio")
        axis.legend_.remove()
    depth_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=depth_palette[depth],
               markerfacecolor=depth_palette[depth], label=depth_labels[depth])
        for depth in depth_order
    ]
    category_handles = [
        Line2D([0], [0], marker=category_markers[category], linestyle="",
               color="black", markerfacecolor="black", label=category)
        for category in category_order
    ]
    fig.legend(depth_handles, [handle.get_label() for handle in depth_handles],
               loc="center left", bbox_to_anchor=(0.85, 0.67), title="Depth",
               borderaxespad=0.0, labelspacing=0.7)
    fig.legend(category_handles, [handle.get_label() for handle in category_handles],
               loc="center left", bbox_to_anchor=(0.85, 0.29), title="Feature category",
               borderaxespad=0.0, labelspacing=0.7)
    fig.suptitle("Regression FORUM on feature versus coefficient", y=0.9)
    fig.tight_layout(rect=(0, 0, 0.84, 0.94))
    _show(show)
    return fig


def plot_ml_forum_shap(frame: pd.DataFrame, *, show: bool = True):
    """Plot ML FORUM against signed mean SHAP contributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    selector_order = ["audience", "curator"]
    selector_labels = {"audience": "Audience models", "curator": "Editor models"}
    depth_order = ["top10", "full"]
    depth_labels = {"top10": "Top 10", "full": "Full list"}
    model_kind_order = ["XGB", "XGB-T", "NN", "NN-T"]
    model_kind_palette = {
        "XGB": "#0072B2", "XGB-T": "#009E73", "NN": "#D55E00", "NN-T": "#CC79A7",
    }
    category_order = [
        "AQuA", "Text / NLP\nfeatures", "Author history",
        "Timing / discussion\ncontext", "Comment form",
    ]
    category_markers = {
        "AQuA": "o", "Text / NLP\nfeatures": "s", "Author history": "^",
        "Timing / discussion\ncontext": "D", "Comment form": "P",
    }
    x_limits = (-1.08, 1.08)
    shap_low, shap_high = frame["shap_value"].min(), frame["shap_value"].max()
    shap_margin = 0.08 * max(shap_high - shap_low, 1e-6)
    y_limits = (min(shap_low, 0) - shap_margin, max(shap_high, 0) + shap_margin)
    for axis, selector in zip(axes, selector_order):
        panel = frame[frame["selector"].eq(selector)]
        for _, row in panel.iterrows():
            color = model_kind_palette[row["model_kind"]]
            axis.scatter(
                row["forum"], row["shap_value"],
                marker=category_markers[row["feature_category"]], s=110,
                facecolors=color if row["depth"] == "top10" else "none",
                edgecolors=color, linewidths=2, alpha=0.9, zorder=3,
            )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        quadrant_specs = [
            (x_limits[0], 0, y_limits[0], 0, "#2ca02c", "Agreement\n(FORUM −, SHAP −)"),
            (0, x_limits[1], 0, y_limits[1], "#2ca02c", "Agreement\n(FORUM +, SHAP +)"),
            (x_limits[0], 0, 0, y_limits[1], "#d62728", "Disagreement\n(FORUM −, SHAP +)"),
            (0, x_limits[1], y_limits[0], 0, "#d62728", "Disagreement\n(FORUM +, SHAP −)"),
        ]
        for left, right, bottom, top, color, label in quadrant_specs:
            axis.add_patch(Rectangle(
                (left, bottom), right - left, top - bottom,
                facecolor=color, edgecolor="none", alpha=0.22, zorder=0,
            ))
            axis.text(
                (left + right) / 2, bottom + 0.50 * (top - bottom), label,
                ha="center", va="center", fontsize=10,
                fontweight="semibold", color=color, alpha=0.9, zorder=1,
            )
        axis.axhline(0, color="black", linestyle=":", linewidth=1)
        axis.axvline(0, color="black", linestyle=":", linewidth=1)
        axis.set_title(selector_labels[selector])
        axis.set_xlabel("Mean FORUM")
        axis.set_ylabel("Mean SHAP contribution (selector-specific)")
    depth_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.25",
               markerfacecolor="0.25" if depth == "top10" else "none",
               markeredgecolor="0.25", label=depth_labels[depth])
        for depth in depth_order
    ]
    model_handles = [
        Patch(facecolor=model_kind_palette[kind], edgecolor=model_kind_palette[kind], label=kind)
        for kind in model_kind_order
    ]
    category_handles = [
        Line2D([0], [0], marker=category_markers[category], linestyle="",
               color="black", markerfacecolor="black", label=category)
        for category in category_order
    ]
    fig.legend(depth_handles, [handle.get_label() for handle in depth_handles],
               loc="center left", bbox_to_anchor=(0.84, 0.82), title="Depth",
               borderaxespad=0.0, labelspacing=0.7)
    fig.legend(model_handles, [handle.get_label() for handle in model_handles],
               loc="center left", bbox_to_anchor=(0.84, 0.60), title="Model kind",
               borderaxespad=0.0, labelspacing=0.7)
    fig.legend(category_handles, [handle.get_label() for handle in category_handles],
               loc="center left", bbox_to_anchor=(0.84, 0.25), title="Feature category",
               borderaxespad=0.0, labelspacing=0.7)
    fig.suptitle("ML FORUM versus signed mean SHAP by feature", y=0.90)
    fig.tight_layout(rect=(0, 0, 0.82, 0.94))
    _show(show)
    return fig
