#!/usr/bin/env python3
"""Build a small, explicitly selected public replay bundle from private artifacts.

The source tree is never modified. Review the allowlist and generated manifest
before committing the resulting artifacts/canonical directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SELECTED_RUN = "mcs15_nn10_ms5_seed2026_1e1b8fce5deb3e58"
TOPIC = f"CG2/topics/runs/{SELECTED_RUN}"
FORBIDDEN_COLUMNS = {
    "story_id", "comment_id", "doc_id", "root_comment_id", "parent_comment_id",
    "author_id", "author", "username", "title", "canonical_url", "url", "text",
    "body", "content", "representative_docs", "snippet",
}
MAX_FILE_BYTES = 50 * 1024 * 1024


STAGE_FILES: dict[str, list[str]] = {
    "01": [f"shared/scrape_summaries/{name}.parquet" for name in
           ("overview", "monthly", "by_section", "status", "qa")],
    "02": [], "03": [],
    "04": [f"CG1/feature_diagnostics/{name}.csv" for name in (
        "full_distribution_summary", "row_invariant_checks", "sample_accounting",
        "within_story_feature_variation", "all_spearman_correlations",
        "root_spearman_correlations", "high_correlation_pairs",
        "selection_distribution_contrasts")],
    "05": [f"CG1/descriptives/{name}.csv" for name in
           ("sample_flow", "topic_summary", "feature_summary")],
    "06": [f"CG1/comment_gap/{name}.csv" for name in
           ("comment_gap_summary", "comment_gap_topic_summary")],
    "07": [],
    "08": ["CG1/rankers/factorial/experiment_variants.csv"] +
          [f"CG1/winners/{name}.csv" for name in
           ("development_cv_variant_ranking", "development_cv_winners")],
    "09": [f"CG1/reporting/tables/{name}.csv" for name in (
        "development_cv_variant_ranking", "development_cv_winners",
        "figure_26_test_model_performance", "held_out_balanced_macro_f1",
        "held_out_model_performance", "held_out_paired_balanced_macro_f1",
        "held_out_paired_model_differences", "held_out_permutation_importance_gaps",
        "held_out_tie_sensitivity", "regression_coefficients_all",
        "regression_diagnostics_all", "regression_feature_gaps",
        "regression_missing_numbers", "regression_model_diagnostics",
        "regression_selector_associations", "held_out_model_implied_gap_summary",
        "held_out_shap_importance", "cg1_baseline_performance")]
        + ["CG1/reporting/tables/held_out_shap_importance.parquet",
           "CG1/reporting/tables/held_out_shap_story_variability.parquet",
           "CG1/development_scores/figure_26_training_model_performance.csv",
           "CG1/development_scores/training_metrics.parquet"],
    "10": [f"CG2/forum/inference/{name}.csv" for name in
           ("policy_summary", "marginal_effects", "forum_ndcg_agreement",
            "mechanism_alignment")]
          + ["CG2/forum/ranker_handoff/development_cv_rankings.csv"],
    "11": [f"CG2/forum/inference/{name}.csv" for name in
           ("policy_summary", "marginal_effects", "forum_ndcg_agreement",
            "mechanism_alignment")],
    "12": ["CG2/forum/reporting/paper_primary_correlations.csv",
           "CG2/forum/reporting/paper_primary_feature_correlations.csv",
           "CG2/forum/inference/policy_summary.csv",
           "CG1/reporting/tables/regression_selector_associations.csv",
           "CG1/reporting/tables/held_out_shap_importance.parquet"],
    "13": [f"CG2/ranking_similarity/{name}.csv" for name in
           ("policy_feature_mean_forum_outcomes", "top10_raw_forum_matrix",
            "full_raw_forum_matrix", "umap_cluster_membership")]
          + ["CG2/ranking_similarity/display_coordinates.parquet"],
    "14": [f"CG2/topics/diagnostics/searches/7cf9d9ba0750d91b/{name}.csv"
           for name in ("summary", "seeds", "pairs", "pooled_pairs",
                        "pareto_frontier_article_comment_coverage_pooled_common_ami")]
          + [f"{TOPIC}/analysis/{name}.csv" for name in
             ("final_document_coverage", "final_comment_coverage")],
    "15": [f"{TOPIC}/analysis/{name}.csv" for name in (
        "topic_agenda_baseline_summary", "topic_agenda_rarefaction_summary",
        "topic_policy_vote_attention_comparison", "topic_policy_vote_attention_cv",
        "topic_policy_vote_attention_tuning", "topic_policy_vote_attention_curve",
        "topic_policy_fitted_attention_sensitivity")]
          + [f"{TOPIC}/analysis/topic_policy_vote_attention_fit.json"],
    "16": [f"{TOPIC}/analysis/{name}.csv" for name in (
        "vote_spline_topic_policy_combined_article_cosine_gain_effects",
        "vote_spline_topic_policy_concentration_summary",
        "vote_spline_topic_policy_exposure_coverage_summary",
        "vote_spline_topic_policy_concentration_effects",
        "vote_power_law_topic_policy_concentration_summary",
        "vote_power_law_topic_policy_exposure_coverage_summary",
        "vote_spline_topic_policy_article_relative_votes_raw_gain_scatter")],
}
for scope in ("all", "root"):
    STAGE_FILES["07"].extend(
        f"CG1/regression/{scope}/{name}.csv" for name in
        ("model_coefficients", "probability_contrasts", "selector_associations",
         "tie_sensitivity_summary", "feature_scaling", "model_diagnostics")
    )
    STAGE_FILES["07"].extend(
        f"CG1/sensitivity/efron_exact_shared_preprocessing_v4/{scope}/{name}.csv"
        for name in ("sensitivity_summary", "coefficient_comparison")
    )
    STAGE_FILES["07"].extend(
        f"CG1/sensitivity/collinearity_shared_preprocessing_v4/{scope}/{name}.csv"
        for name in ("joint_block_wald_tests", "variant_coefficient_stability",
                     "exact_reparameterisation_checks", "development_cv_paired_differences",
                     "development_cv_metric_summary", "covariance_conditioning",
                     "design_condition_summary", "variant_fit_summary")
    )
for model in ("spline", "power_law"):
    for oracle in ("cosine", "js"):
        STAGE_FILES["15"].append(
            f"{TOPIC}/analysis/topic_policy_vote_{model}_{oracle}_oracle_summary.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def columns(path: Path) -> list[str]:
    if path.suffix in {".png", ".json"}:
        return []
    if path.suffix == ".parquet":
        return pq.read_schema(path).names
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.reader(stream))


def safe_copy(source: Path, target: Path) -> None:
    names = columns(source)
    forbidden = sorted(name for name in names if name.lower() in FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"Sensitive columns in public candidate {source}: {forbidden}")
    if source.stat().st_size >= MAX_FILE_BYTES:
        raise ValueError(f"Public candidate exceeds 50 MiB: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def public_histograms(source_root: Path, target_root: Path) -> list[tuple[str, Path]]:
    """Export exact plotting bins, never the diagnostic comment samples."""
    from commentgap_analysis.feature_diagnostics import BINARY_FEATURES

    root = source_root / "CG1/feature_diagnostics"
    histograms: list[dict] = []
    aqua_histograms: list[dict] = []
    for scope in ("root", "all"):
        sample = pd.read_parquet(root / f"sample_{scope}.parquet")
        excluded = {"story_id", "comment_id", "n_candidates", "n_picks",
                    "curator_selected", "audience_selected_draw_01"}
        for feature in sample.columns:
            if feature in excluded or not pd.api.types.is_numeric_dtype(sample[feature]):
                continue
            values = pd.to_numeric(sample[feature], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            if feature in BINARY_FEATURES:
                unique, counts = np.unique(values, return_counts=True)
                for value, count in zip(unique, counts):
                    histograms.append({"scope": scope, "feature": feature,
                                       "kind": "binary", "left": value,
                                       "right": value, "height": count / len(values),
                                       "median": float(np.median(values)), "clip_low": value,
                                       "clip_high": value})
            else:
                low, high = np.quantile(values, [0.005, 0.995])
                shown = np.clip(values, low, high) if high > low else values
                heights, edges = np.histogram(shown, bins=40, density=True)
                for left, right, height in zip(edges[:-1], edges[1:], heights):
                    histograms.append({"scope": scope, "feature": feature,
                                       "kind": "continuous", "left": left,
                                       "right": right, "height": height,
                                       "median": float(np.median(values)),
                                       "clip_low": low, "clip_high": high})
            if feature.startswith("aqua_") and feature.endswith("_expected"):
                heights, edges = np.histogram(values, bins=np.linspace(0, 3, 31), density=True)
                for left, right, height in zip(edges[:-1], edges[1:], heights):
                    aqua_histograms.append({"scope": scope, "feature": feature,
                                            "left": left, "right": right,
                                            "height": height})
    results = []
    for name, rows in (("feature_histograms.csv", histograms),
                       ("aqua_expected_histograms.csv", aqua_histograms)):
        relative = f"CG1/feature_diagnostics/{name}"
        target = target_root / relative
        pd.DataFrame(rows).to_csv(target, index=False)
        results.append((relative, target))
    return results


def public_gap_plot_inputs(source_root: Path, target_root: Path) -> list[tuple[str, Path]]:
    """Keep only unlinked numeric article plotting values and Jaccard means."""
    scored = pd.read_parquet(source_root / "CG1/comment_gap/article_gap_scores.parquet")
    plot_columns = ["scope", "n_candidates", "n_picks", "gap_score"]
    points = scored.loc[:, plot_columns].copy()
    if points.isna().any().any() or points.duplicated().all():
        raise ValueError("Incomplete public gap plot coordinates")
    jaccard_columns = sorted(name for name in scored if name.startswith("curator_audience_jaccard_draw_"))
    if not jaccard_columns:
        raise ValueError("Saved article scores have no Jaccard tie draws")
    scored["jaccard_mean"] = scored[jaccard_columns].mean(axis=1)
    scored["jaccard_gap_mean"] = 1 - scored["jaccard_mean"]
    both = pd.concat([scored, scored.assign(analysis_partition="all_partitions")], ignore_index=True)
    groups = ["scope", "analysis_partition", "primary_topic"]
    topic = both.groupby(groups, as_index=False)[["jaccard_mean", "jaccard_gap_mean"]].mean()
    overall = both.groupby(groups[:2], as_index=False)[["jaccard_mean", "jaccard_gap_mean"]].mean()
    root = target_root / "CG1/comment_gap"
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, values in (("article_plot_coordinates.parquet", points),
                         ("jaccard_topic_means.csv", topic),
                         ("jaccard_overall_means.csv", overall)):
        target = root / name
        if name.endswith(".parquet"):
            values.to_parquet(target, index=False)
        else:
            values.to_csv(target, index=False)
        outputs.append((f"CG1/comment_gap/{name}", target))
    return outputs


def public_ranking_metrics(source_root: Path, target_root: Path) -> tuple[str, Path]:
    """Aggregate per-story FORUM scores before they enter the public bundle."""
    import duckdb

    source = source_root / "CG2/forum/policy_scores/policy_scores.parquet"
    query = """
        SELECT policy_id, ordering, reply_mode, pinned, deployable, outcome,
               depth, AVG(forum) AS forum, AVG(ndcg) AS ndcg,
               COUNT(DISTINCT story_id) AS n_stories
        FROM read_parquet(?)
        GROUP BY policy_id, ordering, reply_mode, pinned, deployable, outcome, depth
    """
    result = duckdb.connect().execute(query, [str(source)]).fetchdf()
    relative = "CG2/forum/reporting/paper_ranking_metrics.csv"
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    return relative, target


def public_forum_input_counts(source_root: Path, target_root: Path) -> list[tuple[str, Path]]:
    """Retain Stage 10's displayed counts without comment or story rows."""
    import duckdb

    forum_root = source_root / "CG2/forum"
    comments_path = forum_root / "analysis_comments.parquet"
    scores_path = forum_root / "policy_scores/policy_scores.parquet"
    connection = duckdb.connect()
    try:
        comments, discussions = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT story_id) FROM read_parquet(?)",
            [str(comments_path)],
        ).fetchone()
        policy_counts = connection.execute(
            """SELECT policy_id, ordering, reply_mode, pinned, COUNT(*) AS rows
               FROM read_parquet(?)
               GROUP BY policy_id, ordering, reply_mode, pinned""",
            [str(scores_path)],
        ).fetchdf()
        outcomes, depths = connection.execute(
            "SELECT COUNT(DISTINCT outcome), COUNT(DISTINCT depth) FROM read_parquet(?)",
            [str(scores_path)],
        ).fetchone()
    finally:
        connection.close()

    summary = pd.DataFrame({
        "quantity": ["comments", "discussions", "policy definitions", "outcomes", "depths", "score rows"],
        "value": [comments, discussions, len(policy_counts), outcomes, depths,
                  pq.read_metadata(scores_path).num_rows],
    })
    policy_rows = (policy_counts.groupby(["ordering", "reply_mode", "pinned"],
                                         dropna=False, as_index=False)["rows"].sum())
    root = target_root / "CG2/forum"
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, frame in (("input_summary.csv", summary),
                        ("policy_row_counts.csv", policy_rows)):
        relative = f"CG2/forum/{name}"
        target = root / name
        frame.to_csv(target, index=False)
        outputs.append((relative, target))
    return outputs


def public_topic_plot_references(source_root: Path, target_root: Path) -> tuple[str, Path]:
    """Export only the reference lines needed by the saved topic plotters."""
    analysis = source_root / TOPIC / "analysis"
    gains = pd.read_csv(analysis / "vote_spline_topic_policy_combined_article_cosine_gain_effects_oracle_by_story.csv")
    random_metrics = pd.read_parquet(
        analysis / "topic_policy_vote_spline_metrics.parquet",
        columns=["story_id", "policy_id", "article_visible_js_distance", "article_visible_cosine"],
    )
    random_metrics = random_metrics[random_metrics["policy_id"].eq("random__loose__unpinned")]
    oracle = pd.read_parquet(
        analysis / "topic_policy_vote_spline_cosine_oracle_metrics.parquet",
        columns=["story_id", "article_visible_js_distance", "article_visible_cosine"],
    )
    paired = random_metrics.merge(oracle, on="story_id", suffixes=("_random", "_oracle"),
                                  validate="one_to_one")
    if paired.empty:
        raise ValueError("No matching oracle stories for public topic plot references")
    values = {
        "combined_article_cosine": {
            "article": float(gains["article"].mean()),
            "article_minus_votes": float(gains["article_minus_votes"].mean()),
        },
        "scatter_oracle_gain": {
            "js_gain": float((paired["article_visible_js_distance_random"]
                              - paired["article_visible_js_distance_oracle"]).mean()),
            "cosine_gain": float((paired["article_visible_cosine_oracle"]
                                  - paired["article_visible_cosine_random"]).mean()),
        },
    }
    relative = f"{TOPIC}/analysis/public_plot_references.json"
    target = target_root / relative
    target.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    return relative, target


def public_top_stories(source_root: Path, target_root: Path) -> tuple[str, Path]:
    """Retain the top-story counts while omitting titles and direct links."""
    source = source_root / "shared/scrape_summaries/top_stories.parquet"
    values = pd.read_parquet(source, columns=["month", "primary_section", "comments"])
    values.insert(0, "rank", range(1, len(values) + 1))
    relative = "shared/scrape_summaries/top_stories_public.parquet"
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    values.to_parquet(target, index=False)
    return relative, target


def public_preprocessing_diagnostics(target_root: Path) -> list[tuple[str, Path]]:
    """Export the split and scaling diagnostics without article identifiers."""
    private = ROOT / "data/derived/model_data"
    root = target_root / "shared/model_data"
    root.mkdir(parents=True, exist_ok=True)
    balance = pd.read_csv(private / "split_balance_diagnostics.csv")
    if set(balance).intersection(FORBIDDEN_COLUMNS):
        raise ValueError("Identifying split-balance columns")
    split = pd.read_parquet(private / "master_article_split.parquet", columns=["split_role"])
    counts = split["split_role"].value_counts().rename_axis("split_role").reset_index(name="articles")
    parameters = json.loads((private / "preprocessing_parameters.json").read_text())
    results = []
    for name, value in (("split_balance_diagnostics.csv", balance),
                        ("split_role_counts.csv", counts),
                        ("reply_depth_centers.json", parameters["reply_depth_centers"])):
        target = root / name
        if isinstance(value, pd.DataFrame):
            value.to_csv(target, index=False)
        else:
            target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        results.append((f"shared/model_data/{name}", target))
    return results


def public_topic_run_counts(source_root: Path, target_root: Path) -> tuple[str, Path]:
    """Publish selected-fit document counts without memberships or terms."""
    metadata = json.loads((source_root / TOPIC / "topic_model_run_metadata.json").read_text())
    counts = metadata["counts"]
    if not all(isinstance(value, int) and value >= 0 for value in counts.values()):
        raise ValueError("Topic run counts are not nonnegative integers")
    relative = f"{TOPIC}/analysis/public_topic_run_counts.json"
    target = target_root / relative
    target.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
    return relative, target


def public_selected_rankers(source_root: Path, target_root: Path) -> tuple[str, Path]:
    """Publish selected model identities without local score-file paths."""
    source = source_root / "CG2/forum/ranker_handoff/selected_rankers.csv"
    values = pd.read_csv(source).drop(columns=["score_path"])
    relative = "CG2/forum/ranker_handoff/selected_rankers_public.csv"
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(target, index=False)
    return relative, target


def build(source_root: Path, target_root: Path) -> dict:
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"Public bundle target is not empty: {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    products: dict[str, dict] = {}
    stages: dict[str, list[str]] = {}
    for stage, paths in STAGE_FILES.items():
        stages[stage] = paths
        for relative in paths:
            if relative in products:
                continue
            source = source_root / relative
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(source)
            target = target_root / relative
            safe_copy(source, target)
            products[relative] = {"bytes": target.stat().st_size,
                                  "sha256": sha256(target),
                                  "source_sha256": sha256(source),
                                  "columns": columns(target)}
    # The source file contains verbatim example comments. Keep only numeric
    # topic coverage; topic descriptions remain in the private archive.
    source = source_root / TOPIC / "analysis/final_topic_summary.csv"
    target = target_root / TOPIC / "analysis/final_topic_summary_public.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as inp, target.open("w", newline="", encoding="utf-8") as out:
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(out, fieldnames=["Topic", "n_fit_documents"])
        writer.writeheader()
        for row in reader:
            writer.writerow({name: row[name] for name in writer.fieldnames})
    relative = str(target.relative_to(target_root))
    stages["14"].append(relative)
    products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                          "source_sha256": sha256(source), "columns": columns(target),
                          "transformation": "Retained topic number and fit-document count only."}
    # The early stages need inventories, not the underlying comment rows.
    for stage, relative in (("02", "shared/features/feature_inventory_public.json"),
                            ("03", "shared/model_data/model_inventory_public.json")):
        private = (ROOT / "data/derived/features/choice_set_all.parquet" if stage == "02"
                   else ROOT / "data/derived/model_data/choice_set_all.parquet")
        metadata = pq.read_metadata(private)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"row_count": metadata.num_rows,
                                      "column_names": metadata.schema.names,
                                      "source_sha256": sha256(private)}, indent=2) + "\n")
        stages[stage].append(relative)
        products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                              "source_sha256": sha256(private),
                              "transformation": "Schema and row count only; no source rows."}
    for relative, target in public_histograms(source_root, target_root):
        stages["04"].append(relative)
        products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                              "columns": columns(target),
                              "transformation": "Exact plotting bins from private diagnostic sample; no rows or IDs."}
    for relative, target in public_gap_plot_inputs(source_root, target_root):
        stages["06"].append(relative)
        products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                              "columns": columns(target),
                              "transformation": "Numeric article plot coordinates or aggregated Jaccard means; no article IDs."}
    relative, target = public_ranking_metrics(source_root, target_root)
    stages["12"].append(relative)
    products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                          "columns": columns(target),
                          "transformation": "Policy-level means over private story scores; no story IDs."}
    for relative, target in public_forum_input_counts(source_root, target_root):
        stages["10"].append(relative)
        products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                              "columns": columns(target),
                              "transformation": "Aggregate input and policy row counts only; no comment or story IDs."}
    relative, target = public_topic_plot_references(source_root, target_root)
    stages["16"].append(relative)
    products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                          "transformation": "Two oracle reference-line means; no story rows or IDs."}
    relative, target = public_top_stories(source_root, target_root)
    stages["01"].append(relative)
    products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                          "columns": columns(target),
                          "transformation": "Top-story counts and category with titles and URLs removed."}
    for relative, target in public_preprocessing_diagnostics(target_root):
        stages["03"].append(relative)
        products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                              "columns": columns(target),
                              "transformation": "Only split aggregates and numeric preprocessing centers."}
    relative, target = public_topic_run_counts(source_root, target_root)
    stages["14"].append(relative)
    products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                          "transformation": "Selected topic-fit document counts only."}
    relative, target = public_selected_rankers(source_root, target_root)
    stages["10"].append(relative)
    products[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target),
                          "columns": columns(target),
                          "transformation": "Selected model identities with private score paths removed."}
    manifest = {"schema_version": 1, "purpose": "public computation replay inputs",
                "selected_topic_run": SELECTED_RUN, "stages": stages,
                "products": products,
                "privacy_note": "Explicit aggregate and unlinked numeric plotting-input allowlist. No raw comment text, original comment IDs, story IDs, or author IDs. Review article-level plot coordinates and notebook outputs before publication."}
    (target_root / "replay_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (target_root / ".gitignore").write_text(
        "# Only the reviewed public replay products may be tracked.\n"
        "*\n!*/\n!.gitignore\n!replay_manifest.json\n"
        + "".join(f"!/{relative}\n" for relative in sorted(products))
    )
    return {"files": len(products), "bytes": sum(p["bytes"] for p in products.values()),
            "manifest": str(target_root / "replay_manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/archive/recompute")
    parser.add_argument("--target", type=Path, default=ROOT / "artifacts/canonical")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.target), indent=2))


if __name__ == "__main__":
    main()
