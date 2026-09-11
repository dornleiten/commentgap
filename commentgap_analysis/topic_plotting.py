"""Plots for the Paper 2 topic-model search diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


NEIGHBOR_COLORS = {
    5: "#0072B2",   # blue
    10: "#E69F00",  # orange
    15: "#009E73",  # green
    20: "#CC79A7",  # reddish purple
}
SAMPLE_MARKERS = {1: "o", 3: "^", 5: "s", 10: "D", 15: "P"}
OBJECTIVE_COLUMNS = [
    "article_coverage",
    "comment_coverage",
    "raw_pooled_common_ami",
]


def _pareto_mask(values: np.ndarray) -> list[bool]:
    """Return the mask for candidates that maximise every objective."""
    result = []
    for candidate_index in range(len(values)):
        dominated = any(
            other_index != candidate_index
            and np.all(values[other_index] >= values[candidate_index])
            and np.any(values[other_index] > values[candidate_index])
            for other_index in range(len(values))
        )
        result.append(not dominated)
    return result


def _configuration_value(configuration: object, name: str) -> object:
    if isinstance(configuration, Mapping):
        return configuration[name]
    return getattr(configuration, name)


def _final_setup(
    final_configuration: object | None,
    final_fit_corpus: str | None,
) -> tuple[str, int, int, int] | None:
    if final_configuration is None:
        return None
    return (
        final_fit_corpus or "articles_comments",
        int(_configuration_value(final_configuration, "hdbscan_min_cluster_size")),
        int(_configuration_value(final_configuration, "umap_n_neighbors")),
        int(_configuration_value(final_configuration, "hdbscan_min_samples")),
    )


def build_topic_search_plot_data(
    search_report: pd.DataFrame,
    search_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble one row per search setup for the coverage/stability plots."""
    setup_columns = ["fit_corpus", "min_cluster_size", "n_neighbors", "min_samples"]
    article_report = search_report[search_report.doc_type.eq("article_passage")].set_index(setup_columns)
    comment_report = search_report[search_report.doc_type.eq("comment")].set_index(setup_columns)
    raw_report = (
        search_seeds.groupby(setup_columns + ["doc_type"])["raw_coverage_pct"]
        .median()
        .rename("raw_coverage")
        .reset_index()
    )
    article_raw = raw_report[raw_report.doc_type.eq("article_passage")].set_index(setup_columns)
    comment_raw = raw_report[raw_report.doc_type.eq("comment")].set_index(setup_columns)

    records = []
    for setup in article_report.index:
        article = article_report.loc[setup]
        comment = comment_report.loc[setup]
        fit_corpus, min_cluster_size, n_neighbors, min_samples = setup
        records.append(
            {
                "configuration": (
                    f"{fit_corpus}_mcs{min_cluster_size}_nn{n_neighbors}_ms{min_samples}"
                ),
                "fit_corpus": fit_corpus,
                "min_cluster_size": min_cluster_size,
                "n_neighbors": n_neighbors,
                "min_samples": min_samples,
                "article_coverage": article_raw.loc[setup, "raw_coverage"],
                "comment_coverage": comment_raw.loc[setup, "raw_coverage"],
                "raw_pooled_common_ami": article["ami_common_pooled_median"],
            }
        )
    plot_data = pd.DataFrame(records)
    values = plot_data[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
    plot_data["pareto_frontier"] = _pareto_mask(values)
    return plot_data


def build_topic_search_frontier(
    plot_data: pd.DataFrame,
    search_root: Path,
    stability_seeds: Sequence[int],
) -> pd.DataFrame:
    """Add descriptive topic-size diagnostics to the Pareto candidates."""
    topic_rows = []
    for configuration in plot_data.loc[plot_data.pareto_frontier, "configuration"]:
        assignments_path = search_root / configuration / "assignments.parquet"
        if not assignments_path.exists():
            raise FileNotFoundError(assignments_path)
        assignments = pd.read_parquet(assignments_path)
        seed_rows = []
        for seed in stability_seeds:
            labels = assignments[f"raw_seed_{seed}"]
            topic_sizes = labels[labels.ge(0)].value_counts()
            seed_rows.append(
                {
                    "topics": len(topic_sizes),
                    "docs_in_topics_le5": int(topic_sizes[topic_sizes.le(5)].sum()),
                    "docs_in_topics_le10": int(topic_sizes[topic_sizes.le(10)].sum()),
                }
            )
        topic_rows.append(
            {
                "configuration": configuration,
                **{
                    key: int(np.median([row[key] for row in seed_rows]))
                    for key in seed_rows[0]
                },
            }
        )

    frontier = plot_data[plot_data.pareto_frontier].merge(
        pd.DataFrame(topic_rows), on="configuration", validate="one_to_one"
    )
    return frontier.sort_values(
        ["fit_corpus", "article_coverage", "comment_coverage"],
        ascending=[True, False, False],
    )


def _legend(legend_axis, handles, title):
    legend = legend_axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=1,
        frameon=True,
        title=title,
        fontsize=8,
        title_fontsize=8.5,
        handletextpad=0.45,
        borderpad=0.55,
        labelspacing=0.65,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_linewidth(0.8)
    return legend


def _add_legends(figure, legend_axis, cluster_sizes: dict[int, int]) -> None:
    from matplotlib.lines import Line2D

    neighbor_handles = [
        Line2D([], [], linestyle="", marker="o", color=color, markersize=7,
               label=f"{value}")
        for value, color in NEIGHBOR_COLORS.items()
    ]
    sample_handles = [
        Line2D([], [], linestyle="", marker=marker, color="0.25", markersize=7,
               label=f"{value}")
        for value, marker in SAMPLE_MARKERS.items()
    ]
    cluster_handles = [
        Line2D([], [], linestyle="", marker="o", color="0.25",
               markersize=size**0.5, label=f"{value}")
        for value, size in cluster_sizes.items()
    ]
    selection_handles = [
        Line2D(
            [], [], linestyle="", marker="o", markerfacecolor="none",
            markeredgecolor="black", markersize=9, markeredgewidth=1.5,
            label="Pareto frontier",
        ),
        Line2D(
            [], [], linestyle="", marker="o", markerfacecolor="none",
            markeredgecolor="#D62728", markersize=9, markeredgewidth=1.8,
            label="Final model",
        ),
    ]

    legend_axis.set_visible(False)
    left, bottom, width, height = legend_axis.get_position().bounds
    groups = [
        (neighbor_handles, "N neighbors"),
        (sample_handles, "Min samples"),
        (cluster_handles, "Min cluster size"),
        (selection_handles, None),
    ]
    # Separate axes make the group boundaries explicit and prevent a five-row
    # shape legend from colliding with the size legend below it.
    slots = [(0.77, 0.22), (0.48, 0.26), (0.24, 0.20), (0.00, 0.20)]
    for (handles, title), (relative_bottom, relative_height) in zip(groups, slots):
        group_axis = figure.add_axes([
            left,
            bottom + relative_bottom * height,
            width,
            relative_height * height,
        ])
        group_axis.set_axis_off()
        _legend(group_axis, handles, title)


def _plot_metric(
    plot_data: pd.DataFrame,
    search_root: Path,
    x_column: str,
    x_label: str,
    y_column: str,
    y_label: str,
    file_stem: str,
    cluster_sizes: dict[int, int],
    final_setup: tuple[str, int, int, int] | None,
    title: str,
    *,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    figure, (axis, legend_axis) = plt.subplots(
        1,
        2,
        figsize=(6.6, 6.6),
        gridspec_kw={"width_ratios": [1.0, 0.1]},
    )
    axis.set_box_aspect(1)
    for _, row in plot_data.iterrows():
        color = NEIGHBOR_COLORS[int(row.n_neighbors)]
        marker = SAMPLE_MARKERS[int(row.min_samples)]
        size = cluster_sizes[int(row.min_cluster_size)]
        axis.scatter(
            row[x_column], row[y_column], color=color, marker=marker, s=size,
            facecolors=color, edgecolors=color, linewidths=2, zorder=3,
        )

    frontier_rows = plot_data[plot_data.pareto_frontier]
    for _, row in frontier_rows.iterrows():
        axis.scatter(
            row[x_column], row[y_column],
            s=cluster_sizes[int(row.min_cluster_size)] + 55,
            marker=SAMPLE_MARKERS[int(row.min_samples)],
            facecolors="none", edgecolors="black",
            linewidths=1, zorder=4,
        )

    if final_setup is not None:
        fit_corpus, min_cluster_size, n_neighbors, min_samples = final_setup
        selected = plot_data[
            (plot_data.fit_corpus == fit_corpus)
            & (plot_data.min_cluster_size == min_cluster_size)
            & (plot_data.n_neighbors == n_neighbors)
            & (plot_data.min_samples == min_samples)
        ]
        if len(selected) == 1:
            row = selected.iloc[0]
            axis.scatter(
                row[x_column], row[y_column],
                s=cluster_sizes[int(row.min_cluster_size)] + 80,
                marker=SAMPLE_MARKERS[int(row.min_samples)],
                facecolors="none", edgecolors="#D62728",
                linewidths=1, zorder=5,
            )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 100)
    axis.set(
        xlabel=x_label,
        ylabel=y_label,
        title=title,
    )
    axis.grid(alpha=0.2)
    _add_legends(figure, legend_axis, cluster_sizes)
    figure.savefig(search_root / f"{file_stem}.png", dpi=200, bbox_inches="tight")
    figure.savefig(search_root / f"{file_stem}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def _plot_combined_metrics(
    plot_data: pd.DataFrame,
    search_root: Path,
    metrics: Sequence[tuple[str, str, str, str, str, str]],
    cluster_sizes: dict[int, int],
    final_setup: tuple[str, int, int, int] | None,
    *,
    show: bool,
) -> None:
    """Render the article- and comment-coverage diagnostics side by side."""
    import matplotlib.pyplot as plt

    figure, (article_axis, comment_axis, legend_axis) = plt.subplots(
        1,
        3,
        figsize=(14, 6.6),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.24]},
        sharey=True,
    )
    for axis, metric in zip((article_axis, comment_axis), metrics):
        x_column, x_label, y_column, y_label, title, _ = metric
        axis.set_box_aspect(1)
        for _, row in plot_data.iterrows():
            color = NEIGHBOR_COLORS[int(row.n_neighbors)]
            marker = SAMPLE_MARKERS[int(row.min_samples)]
            size = cluster_sizes[int(row.min_cluster_size)]
            axis.scatter(
                row[y_column], row[x_column], color=color, marker=marker, s=size,
                facecolors=color, edgecolors=color, linewidths=2, zorder=3,
            )
        for _, row in plot_data[plot_data.pareto_frontier].iterrows():
            axis.scatter(
                row[y_column], row[x_column],
                s=cluster_sizes[int(row.min_cluster_size)] + 55,
                marker=SAMPLE_MARKERS[int(row.min_samples)],
                facecolors="none", edgecolors="black", linewidths=1, zorder=4,
            )
        if final_setup is not None:
            fit_corpus, min_cluster_size, n_neighbors, min_samples = final_setup
            selected = plot_data[
                (plot_data.fit_corpus == fit_corpus)
                & (plot_data.min_cluster_size == min_cluster_size)
                & (plot_data.n_neighbors == n_neighbors)
                & (plot_data.min_samples == min_samples)
            ]
            if len(selected) == 1:
                row = selected.iloc[0]
                axis.scatter(
                    row[y_column], row[x_column],
                    s=cluster_sizes[int(row.min_cluster_size)] + 80,
                    marker=SAMPLE_MARKERS[int(row.min_samples)],
                    facecolors="none", edgecolors="#D62728",
                    linewidths=1, zorder=5,
                )
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 1)
        axis.set(xlabel=y_label, ylabel=x_label, title=title)
        axis.grid(alpha=0.2)
    comment_axis.set_ylabel("")
    comment_axis.tick_params(axis="y", labelleft=False, labelright=False)
    figure.tight_layout(rect=(0, 0, 0.97, 0.95), w_pad=1.6)
    _add_legends(figure, legend_axis, cluster_sizes)
    figure.savefig(
        search_root / "raw_pooled_common_ami_vs_article_comment_coverage_pareto.png",
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        search_root / "raw_pooled_common_ami_vs_article_comment_coverage_pareto.pdf",
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_topic_search_metrics(
    search_report: pd.DataFrame,
    search_seeds: pd.DataFrame,
    search_root: Path,
    stability_seeds: Sequence[int],
    *,
    final_configuration: object | None = None,
    final_fit_corpus: str | None = None,
    combine: bool = False,
    show: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the search summary and write the AMI/coverage figures."""
    plot_data = build_topic_search_plot_data(search_report, search_seeds)
    frontier = build_topic_search_frontier(plot_data, search_root, stability_seeds)
    frontier_columns = [
        "configuration", "article_coverage", "comment_coverage",
        "raw_pooled_common_ami", "topics",
        "docs_in_topics_le5", "docs_in_topics_le10",
    ]
    frontier[frontier_columns].to_csv(
        search_root / "pareto_frontier_article_comment_coverage_pooled_common_ami.csv",
        index=False,
    )
    print(f"Three-objective Pareto frontier: {len(frontier)} of {len(plot_data)} configurations")

    cluster_size_values = sorted(plot_data.min_cluster_size.unique())
    cluster_sizes = {
        int(value): 60 + 50 * index for index, value in enumerate(cluster_size_values)
    }
    final_setup = _final_setup(final_configuration, final_fit_corpus)
    metrics = [
        (
            "raw_pooled_common_ami",
            "Median overall AMI over commonly clustered documents",
            "article_coverage",
            "Article passage coverage (%)",
            "Clustering stability vs Article passage coverage",
            "raw_pooled_common_ami_vs_article_coverage_pareto",
        ),
        (
            "raw_pooled_common_ami",
            "Median overall AMI over commonly clustered documents",
            "comment_coverage",
            "Comment coverage (%)",
            "Clustering stability vs Comment coverage",
            "raw_pooled_common_ami_vs_comment_coverage_pareto",
        ),
    ]
    if combine:
        _plot_combined_metrics(
            plot_data, search_root, metrics, cluster_sizes, final_setup, show=show,
        )
    else:
        for x_column, x_label, y_column, y_label, title, file_stem in metrics:
            _plot_metric(
                plot_data, search_root, x_column, x_label, y_column, y_label,
                file_stem, cluster_sizes, final_setup, title=title, show=show,
            )
    return plot_data, frontier

from .presentation_labels import (
    ORDERING_DISPLAY_COLORS,
    ORDERING_DISPLAY_LABELS,
    ORDERING_DISPLAY_ORDER,
    REPLY_DISPLAY_MARKERS,
)
from .topic_policy import (
    _policy_contrast_label,
    _policy_contrast_values,
    _mean_interval,
    _metric_effect_table,
)

def plot_topic_policy_exposure_coverage(
    coverage_summary: pd.DataFrame,
    *,
    output_root: Path | str,
    ordering_labels: Mapping[str, str],
    output_prefix: str = "",
    exposure_label: str = "inverse-rank",
) -> pd.DataFrame:
    """Plot policy mean valid exposure against article-alignment gain."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    required = {
        "ordering",
        "assigned_exposure_coverage_mean",
        "alignment_gain_mean",
    }
    missing = sorted(required - set(coverage_summary.columns))
    if missing:
        raise ValueError(f"Coverage summary is missing columns: {missing}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.6, 6))
    available_orderings = set(coverage_summary["ordering"])
    orderings = [ordering for ordering in ORDERING_DISPLAY_ORDER if ordering in available_orderings]
    colors = {
        ordering: ORDERING_DISPLAY_COLORS.get(ordering, "#777777")
        for ordering in orderings
    }
    reply_markers = REPLY_DISPLAY_MARKERS
    for ordering in orderings:
        for reply_mode, marker in reply_markers.items():
            for pinned in (False, True):
                panel = coverage_summary.loc[
                    coverage_summary["ordering"].eq(ordering)
                    & coverage_summary["reply_mode"].eq(reply_mode)
                    & coverage_summary["pinned"].eq(pinned)
                ]
                if panel.empty:
                    continue
                axis.scatter(
                    panel["assigned_exposure_coverage_mean"],
                    panel["alignment_gain_mean"],
                    color=colors[ordering] if pinned else "none",
                    edgecolor=colors[ordering],
                    marker=marker,
                    s=55,
                    linewidth=0.9,
                    alpha=0.9,
                )
    axis.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Alignment")
    axis.set_title("Coverage and alignment")
    axis.grid(color="0.92", linewidth=0.6)
    ordering_handles = [
        Patch(
            facecolor=colors[ordering], edgecolor=colors[ordering], linewidth=0.8,
            label=ordering_labels.get(ordering, ordering),
        )
        for ordering in orderings
    ]
    reply_handles = [
        Line2D(
            [0], [0], marker=marker, color="0.35", markerfacecolor="0.7",
            markeredgecolor="0.25", linestyle="None", markersize=7,
            label={"loose": "Loose replies", "trees": "Thread trees", "hidden": "Hidden replies"}[reply],
        )
        for reply, marker in reply_markers.items()
    ]
    pin_handles = [
        Line2D([0], [0], marker="o", color="0.35", markerfacecolor="0.7",
               markeredgecolor="0.25", linestyle="None", markersize=7, label="Pinned"),
        Line2D([0], [0], marker="o", color="0.35", markerfacecolor="none",
               markeredgecolor="0.25", linestyle="None", markersize=7, label="Unpinned"),
    ]
    ordering_legend = figure.legend(
        handles=ordering_handles, title="Primary ordering", 
        bbox_to_anchor=(0.73, 0.97), loc="upper left",
        handlelength=1.8, handleheight=1.2, handletextpad=0.6,
    )
    reply_legend = figure.legend(
        handles=reply_handles, title="Reply status", 
        bbox_to_anchor=(0.73, 0.36), loc="upper left",
    )
    figure.legend(
        handles=pin_handles, title="Pin status", 
        bbox_to_anchor=(0.73, 0.19), loc="upper left",
    )
    figure.tight_layout(rect=(0, 0, 0.72, 1))
    figure.savefig(
        output_root / f"{output_prefix}topic_policy_exposure_coverage_relationship.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(figure)
    return coverage_summary

def _contrast_colors() -> dict[str, str]:
    return {
        "ordering_vs_random": "#2A62A8",
        "reply_vs_loose": "#D17A22",
        "pinned_vs_unpinned": "#6A3D9A",
    }

def _contrast_style(family: str) -> tuple[str, str, str]:
    """Return the notebook-11 marker, point colour, and interval colour."""
    if family not in {"ordering_vs_random", "reply_vs_loose", "pinned_vs_unpinned"}:
        raise ValueError(f"Unknown contrast family: {family}")
    return "o", "#2457A7", "#7A9AC8"

def _add_contrast_divider(
    axis,
    contrast_order: Sequence[tuple[str, str]],
    *,
    invert: bool = True,
) -> None:
    n_orderings = sum(family == "ordering_vs_random" for family, _ in contrast_order)
    axis.axhline(n_orderings - 0.5, color="0.55", linewidth=0.9, linestyle="-", zorder=0)
    n_reply_effects = sum(family == "reply_vs_loose" for family, _ in contrast_order)
    if n_reply_effects:
        axis.axhline(
            n_orderings + n_reply_effects - 0.5,
            color="0.55",
            linewidth=0.9,
            linestyle="-",
            zorder=0,
        )
    if invert:
        axis.invert_yaxis()

def plot_topic_policy_concentration_effects(
    concentration_story: pd.DataFrame,
    *,
    contrast_order: Sequence[tuple[str, str]],
    output_root: Path | str,
    ordering_labels: Mapping[str, str],
    reply_labels: Mapping[str, str],
    output_stem: str = "topic_policy_concentration_effects",
) -> pd.DataFrame:
    """Plot absolute and comparative normalized-entropy policy effects."""

    import matplotlib.pyplot as plt

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    panel_specs = [
        (
            "normalized_entropy",
            "Absolute entropy",
            "Entropy",
        ),
        (
            "normalized_entropy_change_vs_discussion",
            "Change in entropy",
            "Change in normalized entropy",
        ),
        (
            "normalized_entropy_change_vs_relative_votes",
            "Difference from relative votes",
            "Entropy difference",
        ),
    ]
    rows = []
    for column, _, _ in panel_specs:
        for family, contrast in contrast_order:
            paired = _policy_contrast_values(
                concentration_story,
                family,
                contrast,
                column,
            )
            values = (
                paired["treatment"]
                if family == "ordering_vs_random"
                else paired["difference"]
            )
            mean, lower, upper, n = _mean_interval(values)
            rows.append({
                "metric": column,
                "family": family,
                "contrast": contrast,
                "label": _policy_contrast_label(
                    family,
                    contrast,
                    ordering_labels=ordering_labels,
                    reply_labels=reply_labels,
                ),
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "n": n,
            })
    effects = pd.DataFrame(rows)
    effects["contrast_index"] = pd.Categorical(
        list(zip(effects["family"], effects["contrast"])),
        categories=list(contrast_order),
        ordered=True,
    )
    effects = effects.sort_values(["metric", "contrast_index"]).reset_index(drop=True)
    effects.to_csv(output_root / f"{output_stem}.csv", index=False)

    y = np.arange(len(contrast_order))
    column, title, xlabel = panel_specs[1]
    figure, axis = plt.subplots(
        figsize=(6.6, max(4.8, 0.28 * len(contrast_order))),
    )
    panel = effects.loc[effects["metric"].eq(column)].reset_index(drop=True)
    for row_index, row in panel.iterrows():
        if not np.isfinite(row["mean"]):
            continue
        marker, color, error_color = _contrast_style(row["family"])
        axis.errorbar(
            row["mean"],
            row_index,
            xerr=[[row["mean"] - row["lower"]], [row["upper"] - row["mean"]]],
            fmt=marker,
            markersize=3.5 if marker == "o" else 4.5,
            capsize=3,
            color=color,
            ecolor=error_color,
            alpha=0.9,
        )
    axis.axvline(0, color="0.5", linewidth=0.8, linestyle=":")
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="0.9", linewidth=0.6)
    _add_contrast_divider(axis, contrast_order)
    axis.set_yticks(y)
    axis.set_yticklabels(panel["label"])
    figure.suptitle("Topic Diversity", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.96), pad=0.3)
    path = output_root / f"{output_stem}.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    return effects

def plot_topic_policy_combinations(
    metrics: pd.DataFrame,
    output_root: Path | str,
    *,
    metric_columns: Sequence[str] = (
        "alignment_gain",
        "visible_minus_discussion_entropy",
    ),
) -> list[Path]:
    """Plot mean topic effects for every ordering x reply x pin cell."""
    import matplotlib.pyplot as plt

    from .forum_scores import ALL_ORDERINGS, REPLY_MODES

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = (
        metrics.groupby(["ordering", "reply_mode", "pinned"], as_index=False)[list(metric_columns)]
        .mean()
    )
    columns = [(reply, pinned) for reply in REPLY_MODES for pinned in (False, True)]
    orderings = [ordering for ordering in ORDERING_DISPLAY_ORDER if ordering in ALL_ORDERINGS]
    paths = []
    for metric in metric_columns:
        matrix = (
            summary.pivot_table(
                index="ordering",
                columns=["reply_mode", "pinned"],
                values=metric,
            )
            .reindex(index=orderings, columns=pd.MultiIndex.from_tuples(columns))
        )
        figure, axis = plt.subplots(figsize=(6.6, 7))
        image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r")
        axis.set_xticks(np.arange(len(columns)))
        metric_label = {
            "alignment_gain": "Alignment gain (JS-distance units)",
            "alignment_gain_fraction": "Alignment gain (fraction of gap closed)",
            "visible_minus_discussion_entropy": "Change in topic entropy (nats)",
            "hellinger_alpha": "Hellinger projected movement (alpha)",
            "hellinger_cosine": "Hellinger directional cosine",
            "hellinger_angle_degrees": "Angle to article direction (degrees)",
            "hellinger_residual": "Off-axis Hellinger movement",
        }.get(metric, metric)
        ordering_label = ORDERING_DISPLAY_LABELS
        reply_label = {"loose": "Loose replies", "trees": "Thread trees", "hidden": "Hidden replies"}
        axis.set_xticklabels(
            [
                f"{reply_label.get(reply, reply.title())} / "
                f"{'Pinned' if pinned else 'Unpinned'}"
                for reply, pinned in columns
            ],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(np.arange(len(orderings)))
        axis.set_yticklabels([ordering_label.get(ordering, ordering) for ordering in orderings])
        axis.set_xlabel("Reply visibility and pin status")
        axis.set_ylabel("Comment ordering")
        axis.set_title(f"Absolute mean: {metric_label}")
        figure.colorbar(image, ax=axis, label=metric_label)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix.iloc[row_index, column_index]
                if np.isfinite(value):
                    axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center")
        figure.tight_layout()
        path = output_root / f"topic_policy_combinations_{metric}.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
        paths.append(path)
    return paths

def plot_topic_policy_metric_effects(
    metrics: pd.DataFrame,
    *,
    metric: str,
    oracle_frame: pd.DataFrame,
    contrast_order: Sequence[tuple[str, str]],
    output_root: Path | str,
    output_stem: str,
    metric_label: str,
    x_label: str = "Effect",
    ordering_labels: Mapping[str, str],
    reply_labels: Mapping[str, str],
) -> pd.DataFrame:
    """Plot conditional ordering effects and paired interface effects.

    Ordering rows compare the selected ordering with random after averaging
    the six reply/pin variants within each story. Reply and pin rows compare
    matched cells within story, holding the other policy dimensions fixed.
    """

    import matplotlib.pyplot as plt
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    effect_plot = _metric_effect_table(
        metrics,
        metric=metric,
        oracle_frame=oracle_frame,
        contrast_order=contrast_order,
        ordering_labels=ordering_labels,
        reply_labels=reply_labels,
    )
    effect_plot.to_csv(output_root / f"{output_stem}.csv", index=False)

    y = np.arange(len(effect_plot))
    figure, axis = plt.subplots(
        figsize=(6.6, max(4.8, 0.28 * len(effect_plot))),
    )
    value, lower, upper = "relative_mean", "relative_lower", "relative_upper"
    for row_index, row in effect_plot.iterrows():
        if not np.isfinite(row[value]):
            continue
        marker, color, error_color = _contrast_style(row["family"])
        axis.errorbar(
            row[value],
            row_index,
            xerr=[[row[value] - row[lower]], [row[upper] - row[value]]],
            fmt=marker,
            markersize=3.5 if marker == "o" else 4.5,
            capsize=3,
            color=color,
            ecolor=error_color,
            alpha=0.9,
        )
    random_reference = metrics.loc[
        metrics["policy_id"].eq("random__loose__unpinned"),
        ["story_id", metric],
    ].copy()
    random_reference["story_id"] = random_reference["story_id"].astype(str)
    oracle_reference = oracle_frame.loc[:, ["story_id", metric]].copy()
    oracle_reference["story_id"] = oracle_reference["story_id"].astype(str)
    oracle_gap = (
        oracle_reference.set_index("story_id")[metric]
        - random_reference.set_index("story_id")[metric]
    ).dropna().mean()
    axis.axvline(0, color="0.5", linewidth=0.8, linestyle=":")
    axis.axvline(
        oracle_gap,
        color="0.35",
        linewidth=1.1,
        linestyle="--",
    )
    axis.set_xlabel(x_label)
    axis.grid(axis="x", color="0.9", linewidth=0.6)
    _add_contrast_divider(axis, contrast_order)
    axis.set_yticks(y)
    axis.set_yticklabels(effect_plot["label"])
    x_lower, x_upper = axis.get_xlim()
    x_span = x_upper - x_lower
    oracle_margin = 0.12 * x_span
    if oracle_gap <= x_lower + 0.18 * x_span:
        x_lower = min(x_lower, oracle_gap - oracle_margin)
    elif oracle_gap >= x_upper - 0.18 * x_span:
        x_upper = max(x_upper, oracle_gap + oracle_margin)
    axis.set_xlim(x_lower, x_upper)
    axis.text(
        oracle_gap,
        (len(effect_plot) - 1) / 2,
        "Oracle",
        rotation=90,
        rotation_mode="anchor",
        ha="center",
        va="center",
        color="0.35",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )
    figure.suptitle(metric_label, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.96), pad=0.3)
    path = output_root / f"{output_stem}.png"
    figure.savefig(path, dpi=220)
    plt.show()
    plt.close(figure)
    return effect_plot


def plot_topic_policy_progress_effects(
    metrics: pd.DataFrame,
    *,
    oracle_frames: Mapping[str, pd.DataFrame],
    contrast_order: Sequence[tuple[str, str]],
    output_root: Path | str,
    ordering_labels: Mapping[str, str],
    reply_labels: Mapping[str, str],
    output_prefix: str = "",
) -> pd.DataFrame:
    """Plot raw-proportion cosine and JS progress from random policy to their metric-specific oracles."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_root = Path(output_root)
    specs = [
        ("alignment_gain", "Jensen-Shannon alignment", "JS progress: random (0) -> JS oracle (1)"),
        ("cosine_progress", "Raw-proportion cosine progress", "Raw-proportion cosine progress: random (0) -> raw-cosine oracle (1)"),
    ]
    rows = []
    for metric, metric_label, xlabel in specs:
        random_reference = metrics.loc[
            metrics["policy_id"].eq("random__loose__unpinned"),
            ["story_id", metric],
        ].copy()
        random_reference["story_id"] = random_reference["story_id"].astype(str)
        random_reference = random_reference.set_index("story_id")[metric]
        oracle_reference = oracle_frames[metric].loc[:, ["story_id", metric]].copy()
        oracle_reference["story_id"] = oracle_reference["story_id"].astype(str)
        oracle_reference = oracle_reference.set_index("story_id")[metric]
        working = metrics.copy()
        working["story_id"] = working["story_id"].astype(str)
        for family, contrast in contrast_order:
            paired = _policy_contrast_values(working, family, contrast, metric)
            aligned = paired.merge(
                random_reference.rename("random"),
                left_on="story_id",
                right_index=True,
                how="inner",
            ).merge(
                oracle_reference.rename("oracle"),
                left_on="story_id",
                right_index=True,
                how="inner",
            )
            denominator = aligned["oracle"] - aligned["random"]
            zero_denominator = denominator.abs() <= np.finfo(float).eps
            progress = (aligned["difference"] / denominator).where(
                ~zero_denominator
            )
            mean, lower, upper, n = _mean_interval(progress)
            rows.append({
                "metric": metric,
                "metric_label": metric_label,
                "family": family,
                "contrast": contrast,
                "label": _policy_contrast_label(
                    family,
                    contrast,
                    ordering_labels=ordering_labels,
                    reply_labels=reply_labels,
                ),
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "n": n,
                "n_joined": int(len(aligned)),
                "n_excluded_zero_denominator": int(zero_denominator.sum()),
                "excluded_zero_denominator_fraction": (
                    float(zero_denominator.mean()) if len(aligned) else np.nan
                ),
            })
    effects = pd.DataFrame(rows)
    effects["contrast_index"] = pd.Categorical(
        list(zip(effects["family"], effects["contrast"])),
        categories=list(contrast_order),
        ordered=True,
    )
    effects = effects.sort_values(["metric", "contrast_index"]).reset_index(drop=True)
    effects.to_csv(
        output_root / f"{output_prefix}topic_policy_cosine_js_progress_relative_to_random_oracle.csv",
        index=False,
    )

    figure, axes = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(14, max(6.5, 0.32 * len(contrast_order))),
    )
    colors = _contrast_colors()
    y = np.arange(len(contrast_order))
    for axis, (metric, metric_label, xlabel) in zip(axes, specs):
        panel = effects.loc[effects["metric"].eq(metric)].reset_index(drop=True)
        for row_index, row in panel.iterrows():
            if not np.isfinite(row["mean"]):
                continue
            axis.errorbar(
                row["mean"],
                row_index,
                xerr=[[row["mean"] - row["lower"]], [row["upper"] - row["mean"]]],
                fmt="o",
                markersize=6,
                capsize=3,
                color=colors[row["family"]],
                ecolor=colors[row["family"]],
                alpha=0.9,
            )
        axis.axvline(0, color="0.5", linewidth=0.8, linestyle=":")
        axis.axvline(1, color="0.25", linewidth=0.8, linestyle=":")
        axis.set_title(metric_label)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="0.9", linewidth=0.6)
        _add_contrast_divider(axis, contrast_order, invert=False)
    _add_contrast_divider(axes[0], contrast_order)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        effects.loc[effects["metric"].eq(specs[0][0]), "label"],
    )
    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor=color,
                label={
                    "ordering_vs_random": "Primary ordering",
                    "reply_vs_loose": "Reply status",
                    "pinned_vs_unpinned": "Pin status",
                }[family],
                markersize=7,
            )
            for family, color in colors.items()
        ],
        frameon=False,
        loc="best",
    )
    figure.suptitle("Ordering and interface effects: Jensen-Shannon and raw-proportion cosine progress", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    path = output_root / f"{output_prefix}topic_policy_cosine_js_progress_relative_to_random_oracle.png"
    figure.savefig(path, dpi=220)
    plt.show()
    plt.close(figure)
    return effects

def plot_article_relative_votes_marginal_effects(
    reference_target_metrics: pd.DataFrame,
    *,
    contrast_order: Sequence[tuple[str, str]],
    output_root: Path | str,
    ordering_labels: Mapping[str, str],
    reply_labels: Mapping[str, str],
    output_prefix: str = "",
    normalize_gains: bool = False,
    oracle_gains: Mapping[str, float] | None = None,
    distance_metric: str | None = None,
) -> pd.DataFrame:
    """Plot paired contrasts in article-minus-relative-votes gain.

    The difference is formed within story and policy before the requested
    factorial contrast is averaged. With ``normalize_gains=False`` (the
    default), gains retain their original metric units. With ``True``, each
    gain is divided by the maximum possible gain from the random baseline to
    that target: the random-to-target Jensen-Shannon distance, or
    ``1 - random-to-target cosine similarity``. Thus the plotted difference
    compares the fraction of each target gap closed by the policy.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        (
            "js_raw_target_gain",
            "Jensen-Shannon gain" if normalize_gains else "Jensen-Shannon gain",
        ),
        (
            "cosine_raw_target_gain",
            "Cosine gain",
        ),
    ]
    if distance_metric is not None:
        metric_keys = {"jsd": "js_raw_target_gain", "cosine": "cosine_raw_target_gain"}
        if distance_metric not in metric_keys:
            raise ValueError("distance_metric must be one of: 'jsd', 'cosine'")
        metric_specs = [
            spec for spec in metric_specs if spec[0] == metric_keys[distance_metric]
        ]
    target_index = [
        "story_id",
        "policy_id",
        "ordering",
        "reply_mode",
        "pinned",
    ]
    rows = []
    for metric, metric_label in metric_specs:
        source_columns = target_index + ["target", metric]
        if normalize_gains:
            source_columns += (
                ["js_distance_random_to_target"]
                if metric.startswith("js_")
                else ["cosine_similarity_random_to_target"]
            )
            missing_columns = sorted(set(source_columns) - set(reference_target_metrics.columns))
            if missing_columns:
                raise ValueError(
                    "Normalized gain plotting requires the updated reference-target "
                    f"artifact; missing columns: {missing_columns}. "
                    "Rerun notebook 15 to rebuild reference-target metrics."
                )
        target_frame = (
            reference_target_metrics.loc[
                reference_target_metrics["target"].isin(["article", "relative_votes"]),
                source_columns,
            ]
            .pivot_table(
                index=target_index,
                columns="target",
                values=(
                    [metric, source_columns[-1]]
                    if normalize_gains else metric
                ),
                aggfunc="mean",
            )
            .reset_index()
        )
        if normalize_gains:
            target_frame.columns = [
                "_".join(str(value) for value in column if str(value) != "")
                if isinstance(column, tuple) else str(column)
                for column in target_frame.columns
            ]
            metric_name = metric.removesuffix("_raw_target_gain")
            if metric.startswith("js_"):
                denominator_name = "js_distance_random_to_target"
            else:
                denominator_name = "cosine_similarity_random_to_target"
            for target in ("article", "relative_votes"):
                gain_column = f"{metric}_{target}"
                denominator_column = f"{denominator_name}_{target}"
                if gain_column not in target_frame or denominator_column not in target_frame:
                    raise ValueError(
                        f"Missing normalization inputs for {metric} and {target}; "
                        "rebuild reference-target metrics first"
                    )
                denominator = (
                    target_frame[denominator_column]
                    if metric.startswith("js_")
                    else 1.0 - target_frame[denominator_column]
                )
                target_frame[f"{metric_name}_normalized_gain_{target}"] = np.divide(
                    target_frame[gain_column],
                    denominator,
                    out=np.full(len(target_frame), np.nan, dtype=float),
                    where=denominator > np.finfo(float).eps,
                )
            gain_difference_column = f"{metric_name}_normalized_gain_difference"
            target_frame[gain_difference_column] = (
                target_frame[f"{metric_name}_normalized_gain_article"]
                - target_frame[f"{metric_name}_normalized_gain_relative_votes"]
            )
        else:
            gain_difference_column = "raw_gain_difference"
        if not normalize_gains:
            if not {"article", "relative_votes"}.issubset(target_frame.columns):
                raise ValueError(f"Missing article or relative-votes target for {metric}")
            target_frame["raw_gain_difference"] = target_frame["article"] - target_frame["relative_votes"]
        for family, contrast in contrast_order:
            paired = _policy_contrast_values(
                target_frame,
                family,
                contrast,
                gain_difference_column,
            )
            mean, lower, upper, n = _mean_interval(paired["difference"])
            rows.append({
                "metric": metric,
                "metric_label": (
                    (
                        "Normalized Jensen-Shannon gain"
                        if metric.startswith("js_")
                        else "Normalized raw-proportion cosine gain"
                    )
                    if normalize_gains else metric_label
                ),
                "family": family,
                "contrast": contrast,
                "label": _policy_contrast_label(
                    family,
                    contrast,
                    ordering_labels=ordering_labels,
                    reply_labels=reply_labels,
                ),
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "n": n,
            })
    result = pd.DataFrame(rows)
    result["contrast_index"] = pd.Categorical(
        list(zip(result["family"], result["contrast"])),
        categories=list(contrast_order),
        ordered=True,
    )
    result = result.sort_values(
        ["metric", "contrast_index"]
    ).reset_index(drop=True)
    result.to_csv(
        output_root / (
            f"{output_prefix}topic_policy_article_vs_relative_votes_"
            f"{'normalized' if normalize_gains else 'raw'}_gain_effects.csv"
        ),
        index=False,
    )

    figure, axes = plt.subplots(
        1,
        len(metric_specs),
        sharey=True,
        figsize=(6.6, max(4.8, 0.28 * len(contrast_order))),
        squeeze=False,
    )
    y = np.arange(len(contrast_order))
    for metric_index, (metric, metric_label) in enumerate(metric_specs):
        axis = axes[0, metric_index]
        panel = result.loc[result["metric"].eq(metric)].reset_index(drop=True)
        for row_index, row in panel.iterrows():
            marker, color, error_color = _contrast_style(row["family"])
            axis.errorbar(
                row["mean"],
                row_index,
                xerr=[[row["mean"] - row["lower"]], [row["upper"] - row["mean"]]],
                fmt=marker,
                markersize=3.5 if marker == "o" else 4.5,
                capsize=3,
                color=color,
                ecolor=error_color,
                alpha=0.9,
            )
        axis.axvline(0, color="0.5", linewidth=0.8, linestyle=":")
        if oracle_gains is not None:
            if metric not in oracle_gains:
                raise ValueError(f"Missing oracle gain for {metric}")
            oracle_value = 1.0 if normalize_gains else float(oracle_gains[metric])
            axis.axvline(
                oracle_value,
                color="0.35",
                linewidth=1.1,
                linestyle="--",
                label="Article oracle",
            )
            x_lower, x_upper = axis.get_xlim()
            x_span = x_upper - x_lower
            oracle_margin = 0.12 * x_span
            if oracle_value <= x_lower + 0.18 * x_span:
                x_lower = min(x_lower, oracle_value - oracle_margin)
            elif oracle_value >= x_upper - 0.18 * x_span:
                x_upper = max(x_upper, oracle_value + oracle_margin)
            axis.set_xlim(x_lower, x_upper)
            axis.text(
                oracle_value,
                (len(contrast_order) - 1) / 2,
                "Oracle",
                rotation=90,
                rotation_mode="anchor",
                ha="center",
                va="center",
                color="0.35",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
            )
        axis.set_xlabel("Cosine similarity gain difference" if metric.startswith("cosine") else "Difference in JS gain")
        axis.grid(axis="x", color="0.9", linewidth=0.6)
        _add_contrast_divider(axis, contrast_order, invert=False)
        if metric_index == 0:
            axis.set_yticks(y)
        axis.set_yticklabels(panel["label"])
    _add_contrast_divider(axes[0, 0], contrast_order)
    figure.suptitle(
        (
            "Article vs Relative Votes\nDiscussion Alignment Difference"
        ),
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), pad=0.3)
    path = output_root / (
        f"{output_prefix}topic_policy_article_vs_relative_votes_"
        f"{'normalized' if normalize_gains else 'raw'}_gain_effects.png"
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    return result

def plot_article_relative_votes_progress_scatter(
    reference_target_metrics: pd.DataFrame,
    metrics: pd.DataFrame,
    js_oracle_metrics: pd.DataFrame,
    *,
    score_policies: Mapping[str, object],
    output_root: Path | str,
    ordering_labels: Mapping[str, str],
    output_prefix: str = "",
    distance_metric: str | None = None,
) -> pd.DataFrame:
    """Plot two independent raw-gain axes toward article and relative votes."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source = reference_target_metrics.loc[
        reference_target_metrics["target"].isin(["article", "relative_votes"]),
        [
            "story_id",
            "policy_id",
            "ordering",
            "reply_mode",
            "pinned",
            "target",
            "js_raw_target_gain",
            "cosine_raw_target_gain",
        ],
    ].copy()
    summary = (
        source
        .groupby(["target", "policy_id", "ordering", "reply_mode", "pinned"], as_index=False)
        .agg(
            js_gain=("js_raw_target_gain", "mean"),
            cosine_gain=("cosine_raw_target_gain", "mean"),
            n_stories=("story_id", "nunique"),
        )
        .pivot_table(
            index=["policy_id", "ordering", "reply_mode", "pinned", "n_stories"],
            columns="target",
            values=["js_gain", "cosine_gain"],
        )
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if str(value) != "")
        if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    summary.to_csv(
        output_root / f"{output_prefix}topic_policy_article_relative_votes_raw_gain_scatter.csv",
        index=False,
    )

    random_reference = metrics.loc[
        metrics["policy_id"].eq("random__loose__unpinned"),
        ["story_id", "article_visible_js_distance", "article_visible_cosine"],
    ].copy()
    random_reference["story_id"] = random_reference["story_id"].astype(str)
    oracle_reference = js_oracle_metrics[
        ["story_id", "article_visible_js_distance", "article_visible_cosine"]
    ].copy()
    oracle_reference["story_id"] = oracle_reference["story_id"].astype(str)
    oracle = random_reference.merge(
        oracle_reference,
        on="story_id",
        suffixes=("_random", "_oracle"),
    )
    oracle_gain = {
        "js_gain": float(
            (oracle["article_visible_js_distance_random"]
             - oracle["article_visible_js_distance_oracle"]).mean()
        ),
        "cosine_gain": float(
            (oracle["article_visible_cosine_oracle"]
             - oracle["article_visible_cosine_random"]).mean()
        ),
    }

    reply_markers = REPLY_DISPLAY_MARKERS
    reply_labels = {
        "loose": "Loose replies",
        "trees": "Thread trees",
        "hidden": "Hidden replies",
    }
    colors = {
        ordering: ORDERING_DISPLAY_COLORS.get(ordering, "#777777")
        for ordering in score_policies
    }

    metric_specs = [
        (
            "js_gain",
            "Article vs Relative Votes Discussion Alignment - All Policies",
            "Discussion-Article JS alignment change",
            "Discussion-Relative Votes JS alignment change",
            f"{output_prefix}topic_policy_js_article_relative_votes_raw_gain_scatter.png",
        ),
        (
            "cosine_gain",
            "Article vs Relative Votes Discussion Alignment - All Policies",
            "Discussion-Article cosine similarity change",
            "Discussion-Relative Votes cosine similarity change",
            f"{output_prefix}topic_policy_cosine_article_relative_votes_raw_gain_scatter.png",
        ),
    ]
    if distance_metric is not None:
        metric_keys = {"jsd": "js_gain", "cosine": "cosine_gain"}
        if distance_metric not in metric_keys:
            raise ValueError("distance_metric must be one of: 'jsd', 'cosine'")
        metric_specs = [
            spec for spec in metric_specs if spec[0] == metric_keys[distance_metric]
        ]
    for metric, title, x_label, y_label, output_name in metric_specs:
        x_column = f"{metric}_article"
        y_column = f"{metric}_relative_votes"
        frame = summary.dropna(subset=[x_column, y_column]).copy()
        finite_values = frame[[x_column, y_column]].to_numpy(dtype=float)
        lower_value = float(np.nanmin(finite_values))
        upper_value = float(np.nanmax(finite_values))
        reference_values = np.array(
            [lower_value, upper_value, 0.0, oracle_gain[metric]],
            dtype=float,
        )
        reference_span = max(
            float(np.nanmax(reference_values) - np.nanmin(reference_values)),
            0.05,
        )
        padding = 0.08 * reference_span
        lower = float(np.nanmin(reference_values) - padding)
        upper = float(np.nanmax(reference_values) + padding)

        figure, axis = plt.subplots(figsize=(14, 6))
        for reply_mode, marker in reply_markers.items():
            for pinned in (False, True):
                subset = frame[
                    frame["reply_mode"].eq(reply_mode) & frame["pinned"].eq(pinned)
                ]
                for ordering, policy in subset.groupby("ordering", sort=False):
                    color = colors.get(ordering, "#777777")
                    axis.scatter(
                        policy[x_column],
                        policy[y_column],
                        s=72,
                        marker=marker,
                        color=color if pinned else "none",
                        edgecolor=color,
                        linewidth=0.9,
                        alpha=0.85,
                        zorder=4 if ordering == "random" else 3,
                    )
        axis.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":", zorder=1)
        axis.axvline(0.0, color="0.5", linewidth=0.8, linestyle=":", zorder=1)
        axis.plot(
            [lower, upper],
            [lower, upper],
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )
        axis.axvline(
            oracle_gain[metric],
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.text(
            oracle_gain[metric],
            (lower + upper) / 2.0,
            "Oracle",
            rotation=90,
            va="center",
            ha="center",
            color="0.35",
            backgroundcolor="white",
        )
        label_offset = 0.07 * (upper - lower)
        label_position = lower + 0.52 * (upper - lower)
        axis.text(
            label_position,
            label_position + label_offset,
            "Votes gain larger",
            rotation=45,
            rotation_mode="anchor",
            ha="center",
            va="center",
            color="0.35",
            backgroundcolor="white",
        )
        axis.text(
            label_position,
            label_position - label_offset,
            "Article gain larger",
            rotation=45,
            rotation_mode="anchor",
            ha="center",
            va="center",
            color="0.35",
            backgroundcolor="white",
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.grid(color="0.92", linewidth=0.6)
        ordering_handles = [
            Patch(
                facecolor=colors.get(ordering, "#777777"),
                edgecolor=colors.get(ordering, "#777777"),
                linewidth=0.8,
                label=ordering_labels.get(ordering, ordering),
            )
            for ordering in score_policies
        ]
        reply_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="0.35",
                markerfacecolor="0.7",
                markeredgecolor="0.25",
                linestyle="None",
                markersize=7,
                label=reply_labels[reply_mode],
            )
            for reply_mode, marker in reply_markers.items()
        ]
        pin_handles = [
            Line2D(
                [0], [0], marker="o", color="0.35", markerfacecolor="0.7",
                markeredgecolor="0.25", linestyle="None", markersize=7,
                label="Pinned",
            ),
            Line2D(
                [0], [0], marker="o", color="0.35", markerfacecolor="none",
                markeredgecolor="0.25", linestyle="None", markersize=7,
                label="Unpinned",
            ),
        ]
        ordering_legend = figure.legend(
            handles=ordering_handles,
            title="Primary ordering",
            loc="upper left",
            bbox_to_anchor=(0.63, 0.97),
            labelspacing=0.3,
            borderaxespad=0.0,
            handlelength=1.8,
            handleheight=1.2,
            handletextpad=0.6,
        )
        reply_legend = figure.legend(
            handles=reply_handles,
            title="Reply status",
            loc="upper left",
            bbox_to_anchor=(0.63, 0.43),
            labelspacing=0.35,
            borderaxespad=0.0,
        )
        figure.legend(
            handles=pin_handles,
            title="Pin status",
            loc="upper left",
            bbox_to_anchor=(0.63, 0.27),
            labelspacing=0.35,
            borderaxespad=0.0,
        )
        figure.tight_layout(rect=(0, 0, 0.72, 1))
        figure.savefig(output_root / output_name, dpi=220, bbox_inches="tight")
        plt.show()
        plt.close(figure)
    return summary

_POLICY_PLOT_EXPORTS = {
    "plot_topic_policy_exposure_coverage",
    "plot_topic_policy_concentration_effects",
    "plot_topic_policy_combinations",
    "plot_topic_policy_metric_effects",
    "plot_topic_policy_progress_effects",
    "plot_article_relative_votes_marginal_effects",
    "plot_article_relative_votes_progress_scatter",
}

__all__ = [
    "build_topic_search_plot_data",
    "build_topic_search_frontier",
    "plot_topic_search_metrics",
    *_POLICY_PLOT_EXPORTS,
]
