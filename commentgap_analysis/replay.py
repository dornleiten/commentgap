"""Frozen publication replay from registered scientific artifacts.

This module deliberately has no producer imports.  A replay is allowed to
read registered frozen artifacts and write a new staging directory; it never
creates a model, updates a selection pointer, or repairs a cache.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from commentgap_analysis.paths import ExecutionContext, PathContractError, resolve_artifact_path


class FrozenReplayError(RuntimeError):
    """A registered frozen input is absent or differs from its recorded hash."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, expected: str | None = None) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FrozenReplayError(f"Required frozen artifact is missing: {path}")
    if expected and sha256(path) != expected:
        raise FrozenReplayError(f"Frozen artifact hash differs from its registry: {path}")
    return path


def _registered_source(context: ExecutionContext, row: dict[str, str]) -> Path | None:
    """Locate an asset after or before the relocation without guessing runs."""
    for key in ("canonical_source", "source_path"):
        raw = (row.get(key) or "").strip()
        if raw:
            try:
                candidate = resolve_artifact_path(
                    raw, paths=context.paths,
                    expected_sha256=(row.get("source_sha256") or "").strip() or None,
                )
            except FileNotFoundError:
                continue
            if candidate.is_file():
                return candidate
    return None


def _read_asset_registry(context: ExecutionContext) -> list[dict[str, str]]:
    registry = context.paths.root / "provenance" / "paper-assets.csv"
    if not registry.is_file():
        raise FrozenReplayError(f"Missing publication asset registry: {registry}")
    with registry.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _copy_registered_assets(
    context: ExecutionContext,
    *,
    paper: str,
    output_root: Path,
    skip_stages: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Stage registered non-authored assets with their source identity.

    These are explicitly labelled copied saved presentation assets.  They are
    not passed off as newly calculated results; stages with native frozen
    plotters (currently CG1 stage 09) are skipped and recreated separately.
    """
    skipped = set(skip_stages)
    staged: list[dict[str, str]] = []
    for row in _read_asset_registry(context):
        if row.get("paper") != paper or row.get("stage") in skipped:
            continue
        source = _registered_source(context, row)
        if source is None:
            # A small number of paper figures are authored in the submission
            # package (the CG2 stage-10 conceptual figure). Preserve them in
            # staging with an explicit source record; they are not claimed as
            # regenerated scientific outputs.
            submitted = context.paths.root / row["submitted_path"]
            if row.get("stage") not in {"10", "14"} or not submitted.is_file():
                if not (row.get("source_path") or row.get("canonical_source")):
                    continue
                raise FrozenReplayError(f"No readable registered source for {paper}/{row.get('asset')}")
            destination = output_root / "paper_assets" / row["submitted_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(submitted, destination)
            staged.append({
                "asset": row["asset"],
                "stage": row.get("stage"),
                "destination": str(destination.relative_to(output_root)),
                "source": str(submitted),
                "source_sha256": sha256(submitted),
                "kind": "copied authored/submitted figure; no numeric replay source",
            })
            continue
        expected = (row.get("source_sha256") or "").strip() or None
        _require_file(source, expected)
        destination = output_root / "paper_assets" / row["submitted_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        staged.append({
            "asset": row["asset"],
            "stage": row.get("stage"),
            "destination": str(destination.relative_to(output_root)),
            "source": str(source),
            "source_sha256": sha256(source),
            "kind": "copied saved presentation asset",
        })
    return staged


def replay_cg1(context: ExecutionContext, *, output_root: Path | None = None) -> dict[str, Any]:
    """Recreate CG1 presentation outputs from saved Stage-9 products."""
    if context.mode != "frozen":
        raise PathContractError("replay_cg1 is only available in frozen mode")
    output_root = Path(output_root or context.staging_output("rendered/CG1"))
    source = context.read_root("frozen_cg1_reporting")
    from commentgap_analysis.paper1_reporting import replay_paper1_reporting

    report = replay_paper1_reporting(source_root=source, output_root=output_root / "reporting")
    copied = _copy_registered_assets(context, paper="CG1", output_root=output_root, skip_stages=("09",))
    manifest = {
        "paper": "CG1",
        "mode": "frozen",
        "reporting": report,
        "copied_assets": copied,
        "write_root": str(output_root),
    }
    (output_root / "frozen_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def replay_cg2(context: ExecutionContext, *, output_root: Path | None = None) -> dict[str, Any]:
    """Rebuild CG2 presentation figures from completed FORUM/topic products."""
    if context.mode != "frozen":
        raise PathContractError("replay_cg2 is only available in frozen mode")
    output_root = Path(output_root or context.staging_output("rendered/CG2"))
    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    forum_root = context.read_root("frozen_cg2_forum")
    forum_output = output_root / "forum"

    # Stage 11: all four figures and the primary table are pure transforms of
    # the completed inference CSVs.
    from commentgap_analysis.ranking_algorithm_effects import run_ranking_algorithm_effects
    stage11 = run_ranking_algorithm_effects(
        inference_root=forum_root / "inference", output_root=forum_output
    )
    generated.append({"stage": "11", "kind": "saved inference plotters", "manifest": stage11})

    # Stage 12: aggregate saved per-story policy scores exactly as the
    # notebook does, then use its extracted plotting helpers.
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from commentgap_analysis.forum_plotting import plot_forum_ndcg, plot_top10_full_forum
    from commentgap_analysis.forum_scores import PRIMARY_OUTCOMES
    from commentgap_analysis.presentation_labels import (
        OUTCOME_DISPLAY_LABELS, OUTCOME_DISPLAY_ORDER, REPLY_DISPLAY_MARKERS,
    )
    scores_path = forum_root / "policy_scores" / "policy_scores.parquet"
    _require_file(scores_path)
    score_columns = [
        "story_id", "policy_id", "ordering", "reply_mode", "pinned", "deployable",
        "outcome", "depth", "forum", "ndcg",
    ]
    policy_scores = pd.read_parquet(scores_path, columns=score_columns)
    ranking_metrics = policy_scores.groupby(
        ["policy_id", "ordering", "reply_mode", "pinned", "deployable", "outcome", "depth"],
        as_index=False,
    ).agg(forum=("forum", "mean"), ndcg=("ndcg", "mean"), n_stories=("story_id", "nunique"))
    ranking_metrics["feature_label"] = ranking_metrics["outcome"].map(OUTCOME_DISPLAY_LABELS).fillna(ranking_metrics["outcome"])
    paper_metrics = ranking_metrics[
        ranking_metrics["deployable"] & ranking_metrics["outcome"].isin(PRIMARY_OUTCOMES)
    ].copy()
    paper_wide = paper_metrics.pivot(
        index=["policy_id", "ordering", "reply_mode", "pinned", "outcome", "feature_label"],
        columns="depth", values="forum",
    ).reset_index().rename(columns={"full": "forum_full", "top10": "forum_top10"})
    stage12_root = output_root / "forum_correlations"
    stage12_root.mkdir(parents=True, exist_ok=True)
    feature_order = [OUTCOME_DISPLAY_LABELS[outcome] for outcome in OUTCOME_DISPLAY_ORDER]
    palette = dict(zip(feature_order, sns.color_palette("tab10", len(feature_order))))
    reply_labels = {"loose": "Loose", "trees": "Trees", "hidden": "Hidden"}
    for plotter, frame, name in (
        (plot_top10_full_forum, paper_wide, "paper_forum_top10_full_across_ranking_conditions.pdf"),
        (plot_forum_ndcg, paper_metrics, "paper_forum_vs_ndcg_by_depth.pdf"),
    ):
        figure = plotter(frame, feature_order, palette, REPLY_DISPLAY_MARKERS, reply_labels, show=False)
        figure.savefig(stage12_root / name, bbox_inches="tight")
        plt.close(figure)
    paper_metrics.to_csv(stage12_root / "paper_ranking_metrics.csv", index=False)
    paper_wide.to_csv(stage12_root / "paper_top10_full_metrics.csv", index=False)
    generated.append({"stage": "12", "kind": "saved policy-score plotters", "outputs": [str(path) for path in stage12_root.glob("*.pdf")]})

    # Stage 13: frozen matrices and frozen HDBSCAN labels are preserved.  The
    # display-only two-dimensional UMAP is deterministic and is intentionally
    # recomputed only for drawing; no clusters or policy selection are fitted.
    similarity_root = context.read_root("frozen_cg2_ranking_similarity")
    from commentgap_analysis.ranking_similarity_plotting import plot_umap_clusters
    from commentgap_analysis.presentation_labels import (
        ORDERING_DISPLAY_COLORS, ORDERING_DISPLAY_LABELS, ORDERING_DISPLAY_ORDER,
    )
    matrices = {
        depth: pd.read_csv(similarity_root / f"{depth}_raw_forum_matrix.csv", index_col=0)
        for depth in ("top10", "full")
    }
    membership = pd.read_csv(similarity_root / "umap_cluster_membership.csv")
    available = set(membership["ordering"])
    ordering_order = [value for value in ORDERING_DISPLAY_ORDER if value in available]
    similarity_output = output_root / "ranking_similarity"
    similarity_output.mkdir(parents=True, exist_ok=True)
    figure = plot_umap_clusters(
        matrices, membership, ("top10", "full"), ordering_order,
        {key: ORDERING_DISPLAY_COLORS.get(key, "#777777") for key in ordering_order},
        ORDERING_DISPLAY_LABELS, similarity_output / "raw_umap.png",
        reply_markers=REPLY_DISPLAY_MARKERS, show=False,
    )
    plt.close(figure)
    membership.to_csv(similarity_output / "frozen_umap_cluster_membership.csv", index=False)
    generated.append({"stage": "13", "kind": "seeded display UMAP from saved matrices and frozen labels", "seed": 20260902})

    # Stages 15 and 16: use the selected run recorded by the asset registry.
    topic_row = next(
        row for row in _read_asset_registry(context)
        if row.get("paper") == "CG2" and row.get("asset") == "figures/development_vote_attention.png"
    )
    topic_figure_source = _registered_source(context, topic_row)
    if topic_figure_source is None:
        raise FrozenReplayError("Registered selected topic run is unavailable")
    topic_analysis = topic_figure_source.parent
    from commentgap_analysis.vote_attention_plotting import plot_vote_attention_curve
    curve = pd.read_csv(topic_analysis / "topic_policy_vote_attention_curve.csv")
    fit = json.loads((topic_analysis / "topic_policy_vote_attention_fit.json").read_text())
    topic_output = output_root / "topics"
    topic_output.mkdir(parents=True, exist_ok=True)
    figure = plot_vote_attention_curve(curve, fit, topic_output / "topic_policy_vote_attention_curve.png", show=False)
    plt.close(figure)
    generated.append({"stage": "15", "kind": "saved attention curve and fit plotter"})

    from commentgap_analysis.forum_scores import policy_specs
    from commentgap_analysis.topic_policy import build_policy_contrast_order
    from commentgap_analysis.topic_plotting import (
        plot_article_relative_votes_progress_scatter, plot_combined_article_alignment_effects,
    )
    from commentgap_analysis.presentation_labels import ORDERING_DISPLAY_LABELS
    prefix = "vote_spline_"
    reference = pd.read_parquet(topic_analysis / f"{prefix}topic_policy_reference_target_metrics.parquet")
    metrics = pd.read_parquet(topic_analysis / f"topic_policy_{prefix}metrics.parquet")
    oracle = pd.read_parquet(topic_analysis / f"topic_policy_{prefix}cosine_oracle_metrics.parquet")
    oracle_gains = pd.read_csv(topic_analysis / f"{prefix}topic_policy_combined_article_cosine_gain_effects_oracle_by_story.csv")
    score_policies = {spec.ordering: None for spec in policy_specs()}
    # Preserve the established contrast order; the values are not recomputed.
    contrast_order = build_policy_contrast_order(score_policies)
    plot_combined_article_alignment_effects(
        reference, oracle_gains, contrast_order=contrast_order, output_root=topic_output,
        ordering_labels=dict(ORDERING_DISPLAY_LABELS),
        reply_labels={"loose": "Loose replies", "trees": "Thread trees", "hidden": "Hidden replies"},
        distance_metric="cosine", output_prefix=prefix,
    )
    plot_article_relative_votes_progress_scatter(
        reference, metrics, oracle, score_policies=score_policies, output_root=topic_output,
        ordering_labels=dict(ORDERING_DISPLAY_LABELS), output_prefix=prefix,
        distance_metric="cosine",
    )
    generated.append({"stage": "16", "kind": "saved selected-run metric plotters"})

    # The conceptual illustration and the selected topic-search diagnostic are
    # authored/frozen source assets.  They have no numeric drawing input in the
    # retained bundle, so keep an explicit copied-source record rather than
    # claiming a fresh calculation.
    copied = _copy_registered_assets(context, paper="CG2", output_root=output_root, skip_stages=("11", "12", "13", "15", "16"))
    manifest = {
        "paper": "CG2",
        "mode": "frozen",
        "generated": generated,
        "copied_assets": copied,
        "selection": "registered selected topic run and completed FORUM products; no model fit, policy scoring, clustering, or selection",
        "write_root": str(output_root),
    }
    (output_root / "frozen_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def replay_papers(
    context: ExecutionContext, *, papers: Iterable[str] = ("CG1", "CG2")
) -> dict[str, Any]:
    """Run the requested frozen reporting replays into one run's staging tree."""
    requested = tuple(dict.fromkeys(str(paper).upper() for paper in papers))
    invalid = set(requested) - {"CG1", "CG2"}
    if invalid:
        raise ValueError(f"Unknown paper(s): {sorted(invalid)}")
    result: dict[str, Any] = {}
    if "CG1" in requested:
        result["CG1"] = replay_cg1(context)
    if "CG2" in requested:
        result["CG2"] = replay_cg2(context)
    return result

