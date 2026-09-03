"""Publication-facing figures and tables for ranking-algorithm effects."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .forum_scores import PRIMARY_OUTCOMES


OUTCOME_LABELS = {
    "aqua_score_expected": "AQuA deliberative quality",
    "article_similarity_top3": "Article similarity",
    "toxicity_probability": "Toxicity",
    "log_author_prior_30d_comments": "Participant incumbency",
    "author_prior_30d_reception_balance": "Prior audience reception",
    "semantic_novelty_knn5": "Comment novelty (5-NN)",
    "sentiment_positive": "Positive sentiment",
    "sentiment_negative": "Negative sentiment",
    "cttr": "Lexical diversity (raw CTTR)",
    "smog_de": "Reading difficulty (raw SMOG-DE)",
}

ORDERING_LABELS = {
    "relative_votes": "Relative votes",
    "upvotes": "Upvotes",
    "chronological": "Chronological",
    "reverse_chronological": "Reverse chronological",
    "regression_audience": "Regression: audience",
    "regression_editor": "Regression: editor",
    "xgb_metadata_audience": "XGB metadata: audience",
    "xgb_metadata_editor": "XGB metadata: editor",
    "xgb_metadata_text_audience": "XGB text: audience",
    "xgb_metadata_text_editor": "XGB text: editor",
    "neural_metadata_audience": "Neural metadata: audience",
    "neural_metadata_editor": "Neural metadata: editor",
    "neural_metadata_text_audience": "Neural text: audience",
    "neural_metadata_text_editor": "Neural text: editor",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _plot_modules() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    return plt, colors


def _save_figure(figure: Any, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=220, bbox_inches="tight")
    return paths


def _primary_depth(frame: pd.DataFrame, depth: str) -> pd.DataFrame:
    return frame[
        frame["sample"].eq("primary")
        & frame["depth"].eq(depth)
        & frame["outcome"].isin(PRIMARY_OUTCOMES)
    ].copy()


def _primary_top10(frame: pd.DataFrame) -> pd.DataFrame:
    return _primary_depth(frame, "top10")


def plot_ordering_effects(
    effects: pd.DataFrame, output_root: Path, *, depth: str = "top10"
) -> list[Path]:
    """Plot ordering-versus-random effects with aggregate interface effects.

    The first block shows the 14 substantive ordering contrasts. The second
    block adds the three prespecified marginal effects that average over the
    ordering and the other interface dimensions: trees versus loose, hidden
    versus loose, and pinned versus unpinned.
    """
    plt, _ = _plot_modules()
    data = _primary_depth(effects, depth)
    ordering_data = data[data["contrast_family"].eq("ordering_vs_random")]
    interface_definitions = [
        ("pinned_vs_unpinned", "pinned", "Pinned vs unpinned"),
        ("reply_vs_loose", "hidden", "Hidden vs loose"),
        ("reply_vs_loose", "trees", "Trees vs loose"),
    ]
    interface_rows = []
    for family, contrast, label in interface_definitions:
        subset = data[
            data["contrast_family"].eq(family) & data["contrast"].eq(contrast)
        ].copy()
        subset["plot_label"] = label
        interface_rows.append(subset)
    interface_data = pd.concat(interface_rows, ignore_index=True)

    figure, axes = plt.subplots(2, 5, figsize=(18, 11), sharex=True, sharey=True)
    ordering = list(reversed(ORDERING_LABELS))
    for panel_index, (axis, outcome) in enumerate(zip(axes.flat, PRIMARY_OUTCOMES)):
        ordering_panel = (
            ordering_data[ordering_data["outcome"].eq(outcome)]
            .set_index("contrast")
            .reindex(ordering)
        )
        interface_panel = interface_data[interface_data["outcome"].eq(outcome)]
        interface_panel = interface_panel.set_index("plot_label").reindex(
            [label for _, _, label in interface_definitions]
        )
        ordering_positions = np.arange(len(ordering_panel), dtype=float)
        interface_positions = np.arange(
            len(ordering_panel) + 1,
            len(ordering_panel) + 1 + len(interface_panel),
            dtype=float,
        )
        axis.axvline(0, color="0.55", linewidth=0.8)
        axis.errorbar(
            ordering_panel["estimate"],
            ordering_positions,
            xerr=np.vstack(
                [
                    ordering_panel["estimate"] - ordering_panel["ci_lower"],
                    ordering_panel["ci_upper"] - ordering_panel["estimate"],
                ]
            ),
            fmt="o",
            color="#2457A7",
            ecolor="#7A9AC8",
            markersize=3.5,
            capsize=2,
        )
        axis.errorbar(
            interface_panel["estimate"],
            interface_positions,
            xerr=np.vstack(
                [
                    interface_panel["estimate"] - interface_panel["ci_lower"],
                    interface_panel["ci_upper"] - interface_panel["estimate"],
                ]
            ),
            fmt="D",
            color="#333333",
            ecolor="#777777",
            markersize=4.5,
            capsize=2,
        )
        axis.axhline(len(ordering_panel) - 0.5, color="0.82", linewidth=0.8)
        positions = np.concatenate([ordering_positions, interface_positions])
        axis.set_yticks(positions)
        if panel_index % axes.shape[1] == 0:
            axis.set_yticklabels(
                [ORDERING_LABELS.get(value, value) for value in ordering_panel.index]
                + [label for _, _, label in interface_definitions],
                fontsize=7,
            )
        else:
            axis.tick_params(labelleft=False)
        axis.set_title(OUTCOME_LABELS[outcome])
        axis.set_xlabel("FORUM contrast")
        axis.grid(axis="x", color="0.9", linewidth=0.6)
    depth_label = "top 10" if depth == "top10" else "full discussion (N-1)"
    figure.suptitle(
        f"Ordering and average interface effects at {depth_label} "
        "(paired 95% bootstrap intervals)"
    )
    axes[0, 0].invert_yaxis()

    from matplotlib.lines import Line2D

    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", color="#2457A7",
                label="Ordering vs random",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="none", color="#333333",
                label="Average reply / pin effect",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure_name = "figure_ordering_effects" if depth == "top10" else f"figure_ordering_effects_{depth}"
    paths = _save_figure(figure, output_root / figure_name)
    plt.close(figure)
    return paths

def plot_ordering_policy_variants(
    summary: pd.DataFrame,
    output_root: Path,
    *,
    depth: str = "top10",
) -> list[Path]:
    """Plot all substantive reply/pinning variants for each ordering."""
    plt, _ = _plot_modules()
    data = _primary_depth(summary, depth)
    data = data[data["deployable"]].copy()
    orderings = list(reversed(ORDERING_LABELS))
    positions = np.arange(len(orderings), dtype=float)
    reply_colors = {
        "loose": "#2457A7",
        "trees": "#A33F2B",
        "hidden": "#2A8C68",
    }
    pin_markers = {False: "o", True: "^"}
    # Small deterministic offsets keep the six variants legible at each ordering.
    offsets = {
        ("loose", False): -0.18,
        ("loose", True): -0.06,
        ("trees", False): -0.02,
        ("trees", True): 0.10,
        ("hidden", False): 0.14,
        ("hidden", True): 0.22,
    }
    figure, axes = plt.subplots(2, 5, figsize=(18, 10), sharex=True, sharey=True)
    for panel_index, (axis, outcome) in enumerate(zip(axes.flat, PRIMARY_OUTCOMES)):
        panel = data[data["outcome"].eq(outcome)]
        for reply_mode, color in reply_colors.items():
            for pinned, marker in pin_markers.items():
                subset = (
                    panel[panel["reply_mode"].eq(reply_mode) & panel["pinned"].eq(pinned)]
                    .set_index("ordering")
                    .reindex(orderings)
                )
                y = positions + offsets[(reply_mode, pinned)]
                axis.errorbar(
                    subset["estimate"],
                    y,
                    xerr=np.vstack(
                        [
                            subset["estimate"] - subset["ci_lower"],
                            subset["ci_upper"] - subset["estimate"],
                        ]
                    ),
                    fmt=marker,
                    linestyle="none",
                    color=color,
                    ecolor=color,
                    alpha=0.88,
                    markersize=4.5,
                    capsize=1.8,
                    elinewidth=0.8,
                )
        axis.axvline(0, color="0.55", linewidth=0.8)
        axis.set_title(OUTCOME_LABELS[outcome])
        axis.set_xlabel("Mean FORUM")
        axis.grid(axis="x", color="0.9", linewidth=0.6)
        axis.set_yticks(positions)
        if panel_index % axes.shape[1] == 0:
            axis.set_yticklabels(
                [ORDERING_LABELS.get(value, value) for value in orderings],
                fontsize=7,
            )
        else:
            axis.tick_params(labelleft=False)
    axes[0, 0].invert_yaxis()

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=reply)
        for reply, color in reply_colors.items()
    ] + [
        Line2D([0], [0], marker=marker, linestyle="none", color="0.25", label=label)
        for marker, label in (("o", "Unpinned"), ("^", "Pinned"))
    ]
    depth_label = "top 10" if depth == "top10" else "full discussion (N-1)"
    figure.suptitle(f"Ordering × reply × pin variants at {depth_label}")
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(legend_handles),
        frameon=False,
        title="Colour = reply mode; marker = pin state",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure_name = (
        "figure_ordering_policy_variants"
        if depth == "top10"
        else f"figure_ordering_policy_variants_{depth}"
    )
    paths = _save_figure(figure, output_root / figure_name)
    plt.close(figure)
    return paths


def plot_structure_effects(
    effects: pd.DataFrame, output_root: Path, *, depth: str = "top10"
) -> list[Path]:
    plt, _ = _plot_modules()
    data = _primary_depth(effects, depth)
    data = data[~data["contrast_family"].eq("ordering_vs_random")].copy()
    labels = {
        "trees": "Trees vs loose",
        "hidden": "Hidden vs loose",
        "pinned": "Pinned vs unpinned",
    }
    order = ["trees", "hidden", "pinned"]
    figure, axes = plt.subplots(2, 5, figsize=(16, 9), sharex=True)
    for panel_index, (axis, outcome) in enumerate(zip(axes.flat, PRIMARY_OUTCOMES)):
        panel = data[data["outcome"].eq(outcome)].set_index("contrast").reindex(order)
        positions = np.arange(len(panel))
        axis.axvline(0, color="0.55", linewidth=0.8)
        axis.errorbar(
            panel["estimate"],
            positions,
            xerr=np.vstack(
                [
                    panel["estimate"] - panel["ci_lower"],
                    panel["ci_upper"] - panel["estimate"],
                ]
            ),
            fmt="o",
            color="#A33F2B",
            ecolor="#C98D80",
            markersize=4,
            capsize=2,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels([labels[value] for value in panel.index], fontsize=8)
        axis.invert_yaxis()
        axis.set_title(OUTCOME_LABELS[outcome])
        axis.set_xlabel("Average FORUM contrast")
        axis.grid(axis="x", color="0.9", linewidth=0.6)
    depth_label = "top 10" if depth == "top10" else "full discussion (N-1)"
    figure.suptitle(f"Reply and pinning effects at {depth_label} (paired 95% bootstrap intervals)")
    figure.tight_layout()
    figure_name = "figure_structure_effects" if depth == "top10" else f"figure_structure_effects_{depth}"
    paths = _save_figure(figure, output_root / figure_name)
    plt.close(figure)
    return paths


def plot_metric_agreement(
    metric: pd.DataFrame,
    output_root: Path,
    *,
    depth: str = "top10",
) -> list[Path]:
    """Plot FORUM/nDCG agreement by reply/pin group and overall.

    One figure is written per depth. Each outcome has one overall point for
    all substantive policies and six coloured/marked points for the reply x
    pin groups.
    """
    plt, _ = _plot_modules()
    data = metric[
        metric["sample"].eq("primary")
        & metric["depth"].eq(depth)
        & metric["outcome"].isin(PRIMARY_OUTCOMES)
    ].copy()
    positions = np.arange(len(PRIMARY_OUTCOMES), dtype=float)
    reply_colors = {
        "loose": "#2457A7",
        "trees": "#A33F2B",
        "hidden": "#2A8C68",
    }
    pin_markers = {False: "o", True: "^"}
    offsets = {
        ("loose", False): -0.22,
        ("loose", True): -0.13,
        ("trees", False): -0.04,
        ("trees", True): 0.05,
        ("hidden", False): 0.14,
        ("hidden", True): 0.23,
    }
    figure, axis = plt.subplots(figsize=(11, 7))
    for position, outcome in zip(positions, PRIMARY_OUTCOMES):
        panel = data[data["outcome"].eq(outcome)]
        overall = panel[panel["variant_group"].eq("overall")]
        if len(overall) != 1:
            raise ValueError(f"Expected one overall metric-agreement point for {outcome} / {depth}")
        overall_row = overall.iloc[0]
        axis.errorbar(
            overall_row["spearman_forum_ndcg"],
            position,
            xerr=[[overall_row["spearman_forum_ndcg"] - overall_row["ci_lower"]],
                  [overall_row["ci_upper"] - overall_row["spearman_forum_ndcg"]]],
            fmt="D",
            color="0.15",
            ecolor="0.35",
            markersize=5.5,
            capsize=2,
            label="Overall substantive" if position == positions[0] else "_nolegend_",
        )
        for reply_mode, color in reply_colors.items():
            for pinned, marker in pin_markers.items():
                subset = panel[
                    panel["reply_mode"].eq(reply_mode)
                    & panel["pinned"].eq(pinned)
                ]
                if len(subset) != 1:
                    raise ValueError(
                        f"Expected one metric-agreement point for {outcome} / {depth} / {reply_mode} / {pinned}"
                    )
                row = subset.iloc[0]
                axis.errorbar(
                    row["spearman_forum_ndcg"],
                    position + offsets[(reply_mode, pinned)],
                    xerr=[[row["spearman_forum_ndcg"] - row["ci_lower"]],
                          [row["ci_upper"] - row["spearman_forum_ndcg"]]],
                    fmt=marker,
                    linestyle="none",
                    color=color,
                    ecolor=color,
                    markersize=4.5,
                    capsize=1.5,
                    elinewidth=0.8,
                )
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=reply)
        for reply, color in reply_colors.items()
    ] + [
        Line2D([0], [0], marker=marker, linestyle="none", color="0.25", label=label)
        for marker, label in (("o", "Unpinned"), ("^", "Pinned"))
    ] + [
        Line2D([0], [0], marker="D", linestyle="none", color="0.15", label="Overall substantive")
    ]
    depth_label = "top 10" if depth == "top10" else "full discussion (N-1)"
    axis.set_yticks(positions)
    axis.set_yticklabels([OUTCOME_LABELS[value] for value in PRIMARY_OUTCOMES])
    axis.invert_yaxis()
    axis.set_xlim(-1.02, 1.02)
    axis.set_xlabel("Spearman correlation across substantive policy conditions")
    axis.set_title(f"FORUM and direct nDCG agreement at {depth_label}")
    axis.grid(axis="x", color="0.9", linewidth=0.6)
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(legend_handles),
        frameon=False,
        title="Colour = reply mode; marker = pin state",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure_name = (
        "figure_forum_ndcg_agreement"
        if depth == "top10"
        else f"figure_forum_ndcg_agreement_{depth}"
    )
    paths = _save_figure(figure, output_root / figure_name)
    plt.close(figure)
    return paths


def plot_mechanism_heatmap(mechanism: pd.DataFrame, output_root: Path) -> list[Path]:
    plt, colors = _plot_modules()
    data = mechanism[
        mechanism["sample"].eq("primary")
        & mechanism["outcome"].isin(PRIMARY_OUTCOMES)
    ]
    matrix = data.pivot(
        index="score", columns="outcome", values="mean_within_story_spearman"
    ).reindex(columns=PRIMARY_OUTCOMES)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    image = axis.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        norm=colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1),
    )
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(
        [OUTCOME_LABELS[value] for value in matrix.columns],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels([value.replace("_score", "") for value in matrix.index], fontsize=7)
    axis.set_title("Within-discussion score/outcome alignment")
    figure.colorbar(image, ax=axis, label="Mean Spearman correlation")
    figure.tight_layout()
    paths = _save_figure(figure, output_root / "figure_mechanism_alignment")
    plt.close(figure)
    return paths



def write_primary_effects_table(effects: pd.DataFrame, output_root: Path) -> Path:
    data = _primary_top10(effects)
    data = data[data["contrast_family"].eq("ordering_vs_random")].copy()
    data["cell"] = data.apply(
        lambda row: (
            f"{row['estimate']:.3f} "
            f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        ),
        axis=1,
    )
    table = data.pivot(index="contrast", columns="outcome", values="cell")
    table = table.reindex(index=list(ORDERING_LABELS), columns=PRIMARY_OUTCOMES)
    table.index = [ORDERING_LABELS.get(value, value) for value in table.index]
    table.columns = [OUTCOME_LABELS[value] for value in table.columns]
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "table_primary_ordering_effects.tex"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        table.to_latex(
            escape=True,
            caption=(
                "Top-10 FORUM effects relative to random ordering. "
                "Cells show paired-bootstrap means and 95\\% intervals."
            ),
            label="tab:ranking-algorithm-ordering-effects",
        )
    )
    os.replace(temporary, path)
    return path


def run_ranking_algorithm_effects(
    *,
    inference_root: Path | str = "model_output/selection_2025/forum_ranking_analysis/inference",
    output_root: Path | str = "model_output/selection_2025/forum_ranking_analysis/reporting",
) -> dict[str, Any]:
    inference_root = Path(inference_root)
    output_root = Path(output_root)
    inputs = {
        "policy_summary": inference_root / "policy_summary.csv",
        "marginal_effects": inference_root / "marginal_effects.csv",
        "metric_agreement": inference_root / "forum_ndcg_agreement.csv",
        "mechanism": inference_root / "mechanism_alignment.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Ranking-algorithm reporting inputs are missing: {missing}")
    frames = {name: pd.read_csv(path) for name, path in inputs.items()}
    outputs: list[Path] = []
    outputs.extend(plot_ordering_effects(frames["marginal_effects"], output_root))
    outputs.extend(plot_structure_effects(frames["marginal_effects"], output_root))
    for depth in ("top10", "full"):
        outputs.extend(
            plot_metric_agreement(
                frames["metric_agreement"], output_root, depth=depth
            )
        )
    outputs.extend(plot_mechanism_heatmap(frames["mechanism"], output_root))
    outputs.append(
        write_primary_effects_table(frames["marginal_effects"], output_root)
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_sample": "held-out discussions with at least 11 comments",
        "display_depth": "top10 except the metric-agreement depth comparison",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in outputs
        },
    }
    _atomic_json(manifest, output_root / "reporting_manifest.json")
    return manifest

