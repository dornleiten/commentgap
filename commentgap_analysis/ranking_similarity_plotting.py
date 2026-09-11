"""Plotting helpers for UMAP/HDBSCAN ranking-policy similarity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .ranking_policy_plotting import plot_policy_space


def plot_umap_clusters(
    matrices: Mapping[str, pd.DataFrame],
    membership: pd.DataFrame,
    depths: Sequence[str],
    ordering_order: Sequence[str],
    ordering_palette: Mapping[str, str],
    ordering_labels: Mapping[str, str],
    output_path: Path,
    *,
    reply_markers: Mapping[str, str],
    cluster_boundary_padding: float = 0.4,
    cluster_min_gap_fraction: float = 0.015,
    cluster_corner_cut: float = 0.2,
    label_clusters: bool = True,
    show: bool = True,
):
    """Render the fixed 2-D display embedding and save the UMAP figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.spatial import ConvexHull
    import umap

    depth_titles = {"top10": "Top-10 comments", "full": "Full discussion"}
    cluster_ids = [
        cluster for cluster in sorted(membership["umap_cluster"].unique())
        if cluster != -1
    ]
    cluster_numbers = {cluster: index + 1 for index, cluster in enumerate(cluster_ids)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, depth in zip(axes, depths):
        display_embedding = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.15,
            metric="euclidean", random_state=20260902,
        ).fit_transform(matrices[depth])
        display_scale = np.linalg.norm(np.ptp(display_embedding, axis=0))
        minimum_gap = display_scale * cluster_min_gap_fraction
        frame = membership[membership["depth"].eq(depth)].copy()
        frame["umap1"], frame["umap2"] = display_embedding[:, 0], display_embedding[:, 1]
        plot_policy_space(
            axis, frame, "umap1", "umap2", colour_column="ordering",
            colour_order=list(ordering_order), colour_palette=ordering_palette,
            reply_markers=reply_markers,
        )
        cluster_points = {
            cluster: frame.loc[
                frame["umap_cluster"].eq(cluster), ["umap1", "umap2"]
            ].to_numpy()
            for cluster in cluster_ids
        }
        cluster_points = {
            cluster: points for cluster, points in cluster_points.items()
            if len(points) >= 3
        }
        cluster_centers = {
            cluster: points.mean(axis=0)
            for cluster, points in cluster_points.items()
        }
        panel_center = np.vstack(list(cluster_centers.values())).mean(axis=0)
        placed_labels: list[np.ndarray] = []
        for cluster, points in cluster_points.items():

            hull = ConvexHull(points)
            centroid = points.mean(axis=0)
            hull_vectors = points[hull.vertices] - centroid
            hull_radii = np.linalg.norm(hull_vectors, axis=1, keepdims=True)
            radial_padding = np.maximum(hull_radii * cluster_boundary_padding, minimum_gap)
            vertices = centroid + hull_vectors * (1 + radial_padding / hull_radii)
            rounded_boundary = []
            for index, vertex in enumerate(vertices):
                previous_vertex = vertices[index - 1]
                next_vertex = vertices[(index + 1) % len(vertices)]
                start = vertex * (1 - cluster_corner_cut) + previous_vertex * cluster_corner_cut
                end = vertex * (1 - cluster_corner_cut) + next_vertex * cluster_corner_cut
                if index == 0:
                    rounded_boundary.append(start)
                for fraction in np.linspace(0, 1, 16, endpoint=False)[1:]:
                    rounded_boundary.append(
                        (1 - fraction) ** 2 * start
                        + 2 * (1 - fraction) * fraction * vertex
                        + fraction ** 2 * end
                    )
                rounded_boundary.append(end)
            rounded_boundary = np.vstack([rounded_boundary, rounded_boundary[0]])
            axis.plot(
                rounded_boundary[:, 0], rounded_boundary[:, 1], color="black",
                linewidth=1.5, alpha=0.9, zorder=3,
            )
            if label_clusters:
                outward = centroid - panel_center
                if np.linalg.norm(outward) == 0:
                    outward = np.array([1.0, 0.0])
                outward_angle = np.arctan2(outward[1], outward[0])
                candidate_angles = outward_angle + np.linspace(
                    -np.pi, np.pi, 25, endpoint=False,
                )
                candidates = []
                for angle in candidate_angles:
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    boundary_distance = np.max((points - centroid) @ direction)
                    label_position = centroid + direction * (
                        boundary_distance + display_scale * 0.055
                    )
                    other_centers = [
                        other for other_cluster, other in cluster_centers.items()
                        if other_cluster != cluster
                    ]
                    center_gap = min(
                        np.linalg.norm(label_position - other)
                        for other in other_centers
                    ) if other_centers else np.inf
                    placed_gap = min(
                        np.linalg.norm(label_position - other)
                        for other in placed_labels
                    ) if placed_labels else np.inf
                    candidates.append((min(center_gap, placed_gap), label_position, direction))
                _, label_position, direction = max(candidates, key=lambda item: item[0])
                placed_labels.append(label_position)
                axis.annotate(
                    str(cluster_numbers[cluster]),
                    xy=centroid + direction * np.max((points - centroid) @ direction),
                    xytext=label_position,
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="black", zorder=5, clip_on=False,
                    arrowprops=dict(arrowstyle="-", color="0.25", linewidth=0.8),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
                )
        if label_clusters and placed_labels:
            current_x = axis.get_xlim()
            current_y = axis.get_ylim()
            label_array = np.vstack(placed_labels)
            padding = display_scale * 0.04
            axis.set_xlim(
                min(current_x[0], label_array[:, 0].min() - padding),
                max(current_x[1], label_array[:, 0].max() + padding),
            )
            axis.set_ylim(
                min(current_y[0], label_array[:, 1].min() - padding),
                max(current_y[1], label_array[:, 1].max() + padding),
            )
        axis.set(title=depth_titles[depth], xlabel="UMAP 1", ylabel="UMAP 2")
    reply_labels = {"loose": "Loose", "trees": "Trees", "hidden": "Hidden"}
    reply_handles = [
        Line2D([0], [0], marker=marker, linestyle="", color="black", label=reply_labels[reply])
        for reply, marker in reply_markers.items()
    ]
    pin_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="black", markerfacecolor="none", label="Unpinned"),
        Line2D([0], [0], marker="o", linestyle="", color="black", markerfacecolor="black", label="Pinned"),
    ]
    ordering_handles = [
        Patch(facecolor=ordering_palette[ordering], edgecolor=ordering_palette[ordering],
              linewidth=0.8, label=ordering_labels.get(ordering, ordering))
        for ordering in ordering_order
    ]
    boundary_handle = Line2D([0], [0], color="black", linewidth=1.5, label="Cluster boundary")
    fig.legend(handles=ordering_handles, title="Ordering policy", loc="upper left",
               bbox_to_anchor=(0.78, 0.94), borderaxespad=0.0,
               handlelength=1.8, handleheight=1.2, handletextpad=0.6, labelspacing=0.5)
    fig.legend(handles=[boundary_handle], loc="upper left", bbox_to_anchor=(0.78, 0.07),
               borderaxespad=0.0, handletextpad=0.7, labelspacing=0.6)
    fig.legend(handles=reply_handles, title="Reply mode", loc="upper left",
               bbox_to_anchor=(0.78, 0.365), borderaxespad=0.0,
               handletextpad=0.7, labelspacing=0.6)
    fig.legend(handles=pin_handles, title="Pin status", loc="upper left",
               bbox_to_anchor=(0.78, 0.2), borderaxespad=0.0,
               handletextpad=0.7, labelspacing=0.6)
    fig.suptitle("UMAP-HDBSCAN Clustering of Ranking Policies in FORUM Regression Feature Space")
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig
