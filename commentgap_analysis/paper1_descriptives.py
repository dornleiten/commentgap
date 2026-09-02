"""Paper 1 descriptive tables and figures for the frozen 2025 model sample.

The primary analysis scope is all published comments. Root comments are an
explicit appendix sensitivity. Article sections are joined from the scrape
because they are intentionally absent from the model choice-set contract.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


AUDIENCE_DRAW_COLUMNS = tuple(
    f"audience_selected_draw_{draw:02d}" for draw in range(1, 11)
)
VALID_SCOPES = ("all", "root")
UNKNOWN_TOPIC = "Unknown/other"
TOPIC_MAPPING_VERSION = 1

TEXT_NLP = {
    "log_words",
    "sentiment_positive",
    "sentiment_negative",
    "toxicity_probability",
    "lexdiv_length_adjusted",
    "reading_level_length_adjusted",
    "url_present",
}
REPLY_STRUCTURE = {"is_reply", "reply_depth_centered", "prior_reply_composition"}
RAW_DESCRIPTIVE_FEATURES = (
    "word_count",
    "cttr",
    "smog_de",
    "sentiment_neutral",
    "toxicity_mean_probability",
    "hours_since_article",
    "prior_roots",
    "prior_comments",
    "comments_prev_hour",
    "author_prior_30d_comments",
    "author_prior_30d_snapshot_upvotes",
    "author_prior_30d_snapshot_downvotes",
    "author_prior_comments_story",
    "aqua_score_hard",
    "aqua_score_expected",
)


def _qstring(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _qident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _normalise_scopes(scopes: Iterable[str]) -> tuple[str, ...]:
    resolved = tuple(dict.fromkeys(scopes))
    if not resolved:
        raise ValueError("At least one scope is required")
    invalid = sorted(set(resolved) - set(VALID_SCOPES))
    if invalid:
        raise ValueError(f"Unsupported scopes: {invalid}")
    return resolved


def feature_group(name: str) -> str:
    """Use the same feature-family registry as stage 4 diagnostics."""
    if name.startswith("aqua_"):
        return "aqua_expected"
    if name in TEXT_NLP:
        return "text_nlp"
    if name.startswith("log_author_") or name.startswith("author_prior_"):
        return "author_history"
    if name in REPLY_STRUCTURE:
        return "reply_structure"
    return "semantic_timing_activity"


def _require_columns(path: Path, required: Sequence[str]) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    columns = set(pq.read_schema(path).names)
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return columns


def build_article_topics(
    data_root: Path,
    *,
    year: int = 2025,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Read the article hierarchy and freeze section_1 as the primary topic."""
    article_root = data_root / "articles" / f"year={year}"
    try:
        next(article_root.glob("month=*/*.parquet"))
    except StopIteration as exc:
        raise FileNotFoundError(f"No article shards found below {article_root}") from exc

    glob_path = article_root / "month=*" / "*.parquet"
    own_connection = connection is None
    con = connection or duckdb.connect()
    try:
        topics = con.execute(
            f"""
            SELECT
                CAST(story_id AS VARCHAR) AS story_id,
                MIN(year)::INTEGER AS article_year,
                MIN(month)::INTEGER AS article_month,
                COALESCE(NULLIF(TRIM(MIN(section_1)), ''), {_qstring(UNKNOWN_TOPIC)})
                    AS primary_topic,
                NULLIF(TRIM(MIN(section_1)), '') AS section_1,
                NULLIF(TRIM(MIN(section_2)), '') AS section_2,
                NULLIF(TRIM(MIN(section_3)), '') AS section_3,
                COUNT(DISTINCT section_1) AS n_section_1_values,
                COUNT(DISTINCT section_2) AS n_section_2_values,
                COUNT(DISTINCT section_3) AS n_section_3_values
            FROM read_parquet(
                {_qstring(glob_path)},
                hive_partitioning = false,
                union_by_name = true
            )
            WHERE year = ?
            GROUP BY story_id
            ORDER BY story_id
            """,
            [year],
        ).df()
    finally:
        if own_connection:
            con.close()

    if topics["story_id"].duplicated().any():
        raise ValueError("Article metadata contains duplicate story_id values")
    inconsistent = topics[
        ["n_section_1_values", "n_section_2_values", "n_section_3_values"]
    ].gt(1).any(axis=1)
    if inconsistent.any():
        examples = topics.loc[inconsistent, "story_id"].head(5).tolist()
        raise ValueError(f"Conflicting article section values for stories: {examples}")
    return topics.drop(
        columns=["n_section_1_values", "n_section_2_values", "n_section_3_values"]
    )


def _discussion_table(
    con: duckdb.DuckDBPyConnection,
    choice_path: Path,
    split_path: Path,
    topics: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    required = (
        "story_id",
        "comment_id",
        "article_month",
        "candidate_scope",
        "n_candidates",
        "n_picks",
        "curator_selected",
        "relative_votes",
        *AUDIENCE_DRAW_COLUMNS,
    )
    columns = _require_columns(choice_path, required)
    _require_columns(split_path, ("story_id", "split_role", "development_fold"))
    con.register("article_topics_input", topics)

    draw_counts = ",\n".join(
        f"SUM(CAST({_qident(column)} AS BIGINT)) AS {_qident(column + '_count')}"
        for column in AUDIENCE_DRAW_COLUMNS
    )
    draw_mean = " + ".join(
        f"CAST({_qident(column)} AS DOUBLE)" for column in AUDIENCE_DRAW_COLUMNS
    )
    reply_expression = (
        "SUM(CAST(is_reply AS BIGINT))" if "is_reply" in columns else "NULL"
    )
    frame = con.execute(
        f"""
        SELECT
            {_qstring(scope)} AS scope,
            CAST(c.story_id AS VARCHAR) AS story_id,
            MIN(c.article_month)::INTEGER AS article_month,
            MIN(s.split_role) AS split_role,
            MIN(s.development_fold)::INTEGER AS development_fold,
            CASE WHEN MIN(s.split_role) = 'paper2_test'
                 THEN 'held_out_test' ELSE MIN(s.split_role) END AS analysis_partition,
            MIN(t.primary_topic) AS primary_topic,
            MIN(t.section_1) AS section_1,
            MIN(t.section_2) AS section_2,
            MIN(t.section_3) AS section_3,
            COUNT(*)::BIGINT AS candidate_rows,
            MIN(c.n_candidates)::BIGINT AS n_candidates_declared_min,
            MAX(c.n_candidates)::BIGINT AS n_candidates_declared_max,
            MIN(c.n_picks)::BIGINT AS n_picks_declared_min,
            MAX(c.n_picks)::BIGINT AS n_picks_declared_max,
            SUM(CAST(c.curator_selected AS BIGINT))::BIGINT AS curator_selected_rows,
            SUM(({draw_mean}) / {len(AUDIENCE_DRAW_COLUMNS)})
                AS audience_selected_rows_mean,
            SUM(c.relative_votes)::DOUBLE AS relative_votes_sum,
            AVG(c.relative_votes)::DOUBLE AS relative_votes_mean,
            MEDIAN(c.relative_votes)::DOUBLE AS relative_votes_median,
            {reply_expression} AS reply_candidates,
            {draw_counts}
        FROM read_parquet({_qstring(choice_path)}) AS c
        LEFT JOIN read_parquet({_qstring(split_path)}) AS s USING (story_id)
        LEFT JOIN article_topics_input AS t USING (story_id)
        GROUP BY c.story_id
        ORDER BY c.story_id
        """
    ).df()
    con.unregister("article_topics_input")

    if frame["split_role"].isna().any():
        raise ValueError(f"{scope}: choice-set stories are missing from the shared split")
    if frame["primary_topic"].isna().any():
        raise ValueError(f"{scope}: choice-set stories are missing article topic metadata")
    if not frame["n_candidates_declared_min"].equals(
        frame["n_candidates_declared_max"]
    ) or not np.array_equal(
        frame["candidate_rows"].to_numpy(),
        frame["n_candidates_declared_min"].to_numpy(),
    ):
        raise ValueError(f"{scope}: inconsistent n_candidates values")
    if not frame["n_picks_declared_min"].equals(frame["n_picks_declared_max"]):
        raise ValueError(f"{scope}: inconsistent n_picks values")
    if not np.array_equal(
        frame["curator_selected_rows"].to_numpy(),
        frame["n_picks_declared_min"].to_numpy(),
    ):
        raise ValueError(f"{scope}: curator-selected counts do not equal n_picks")
    for column in AUDIENCE_DRAW_COLUMNS:
        if not np.array_equal(
            frame[f"{column}_count"].to_numpy(),
            frame["n_picks_declared_min"].to_numpy(),
        ):
            raise ValueError(f"{scope}: {column} counts do not equal n_picks")

    frame = frame.rename(
        columns={
            "n_candidates_declared_min": "n_candidates",
            "n_picks_declared_min": "n_picks",
        }
    ).drop(columns=["n_candidates_declared_max", "n_picks_declared_max"])
    reply_candidates = pd.to_numeric(frame["reply_candidates"], errors="coerce")
    frame["reply_candidates"] = reply_candidates
    frame["reply_share"] = reply_candidates / frame["candidate_rows"]
    return frame


def _summarise_discussions(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(groups, dropna=False, observed=True)
        .agg(
            n_articles=("story_id", "nunique"),
            n_candidates=("candidate_rows", "sum"),
            n_curator_picks=("curator_selected_rows", "sum"),
            n_audience_picks_mean=("audience_selected_rows_mean", "sum"),
            candidates_per_article_mean=("candidate_rows", "mean"),
            candidates_per_article_median=("candidate_rows", "median"),
            picks_per_article_mean=("n_picks", "mean"),
            picks_per_article_median=("n_picks", "median"),
            reply_share_mean=("reply_share", "mean"),
            relative_votes_mean=("relative_votes_mean", "mean"),
            relative_votes_sum=("relative_votes_sum", "sum"),
        )
        .reset_index()
    )


def _feature_summary(
    con: duckdb.DuckDBPyConnection,
    choice_path: Path,
    scope: str,
    registry: dict,
) -> pd.DataFrame:
    columns = set(pq.read_schema(choice_path).names)
    model_features = list(registry["models"][scope]["features"])
    raw_features = [name for name in RAW_DESCRIPTIVE_FEATURES if name in columns]
    features = [(name, "transformed_model_predictor") for name in model_features]
    features.extend(
        (name, "raw_descriptive_field")
        for name in raw_features
        if name not in model_features
    )
    missing = [name for name, _ in features if name not in columns]
    if missing:
        raise ValueError(f"{scope}: feature registry columns are missing: {missing}")

    audience_weight = "(" + " + ".join(
        f"CAST({_qident(column)} AS DOUBLE)" for column in AUDIENCE_DRAW_COLUMNS
    ) + f") / {len(AUDIENCE_DRAW_COLUMNS)}"
    expressions: list[str] = []
    for feature, _ in features:
        q = _qident(feature)
        prefix = _qident(feature + "__")[:-1]
        expressions.extend(
            [
                f"COUNT({q}) AS {prefix}candidate_n\"",
                f"AVG({q}) AS {prefix}candidate_mean\"",
                f"STDDEV_SAMP({q}) AS {prefix}candidate_sd\"",
                f"MIN({q}) AS {prefix}candidate_min\"",
                f"QUANTILE_CONT({q}, 0.25) AS {prefix}candidate_q25\"",
                f"MEDIAN({q}) AS {prefix}candidate_median\"",
                f"QUANTILE_CONT({q}, 0.75) AS {prefix}candidate_q75\"",
                f"MAX({q}) AS {prefix}candidate_max\"",
                f"COUNT({q}) FILTER (WHERE curator_selected) AS {prefix}curator_n\"",
                f"AVG({q}) FILTER (WHERE curator_selected) AS {prefix}curator_mean\"",
                f"SUM({q} * {audience_weight}) / NULLIF(SUM({audience_weight}) FILTER (WHERE {q} IS NOT NULL), 0) AS {prefix}audience_mean\"",
                f"SUM({audience_weight}) FILTER (WHERE {q} IS NOT NULL) AS {prefix}audience_effective_n\"",
            ]
        )
    result = con.execute(
        f"SELECT {', '.join(expressions)} FROM read_parquet({_qstring(choice_path)})"
    ).df().iloc[0]

    rows = []
    for feature, source in features:
        prefix = feature + "__"
        details = registry.get("features", {}).get(feature, {})
        rows.append(
            {
                "scope": scope,
                "feature": feature,
                "label": details.get("label", feature.replace("_", " ")),
                "feature_source": source,
                "feature_group": (
                    feature_group(feature) if source == "transformed_model_predictor"
                    else "raw_descriptive"
                ),
                **{name: result[prefix + name] for name in (
                    "candidate_n", "candidate_mean", "candidate_sd", "candidate_min",
                    "candidate_q25", "candidate_median", "candidate_q75", "candidate_max",
                    "curator_n", "curator_mean", "audience_effective_n", "audience_mean",
                )},
            }
        )
    output = pd.DataFrame(rows)
    output["curator_minus_audience_sd"] = (
        output["curator_mean"] - output["audience_mean"]
    ) / output["candidate_sd"].replace(0, np.nan)
    return output


def _sample_flow(
    discussions: pd.DataFrame,
    scopes: tuple[str, ...],
    provenance: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    source = provenance.get("source", {})
    for scope in scopes:
        scoped = discussions[discussions["scope"] == scope]
        prior = provenance.get(scope, {})
        base = {"scope": scope, "primary_or_appendix": "primary" if scope == "all" else "appendix"}
        rows.extend(
            [
                {**base, "stage": "scraped_2025_collection", "n_articles": source.get("articles"), "n_comment_rows": source.get("comments"), "n_candidates": np.nan, "n_picks": np.nan},
                {**base, "stage": "published_feature_choice_set", "n_articles": prior.get("eligible_stories"), "n_comment_rows": np.nan, "n_candidates": prior.get("candidate_rows"), "n_picks": prior.get("sticky_comments")},
                {**base, "stage": "shared_preprocessed_choice_set", "n_articles": scoped["story_id"].nunique(), "n_comment_rows": np.nan, "n_candidates": scoped["candidate_rows"].sum(), "n_picks": scoped["n_picks"].sum()},
            ]
        )
        for role, label in (("development", "development"), ("paper2_test", "held_out_test")):
            partition = scoped[scoped["split_role"] == role]
            rows.append(
                {**base, "stage": label, "n_articles": partition["story_id"].nunique(), "n_comment_rows": np.nan, "n_candidates": partition["candidate_rows"].sum(), "n_picks": partition["n_picks"].sum()}
            )
    return pd.DataFrame(rows)


def _save_figures(
    discussions: pd.DataFrame,
    topic_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    output_root: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scope = "all" if "all" in set(discussions["scope"]) else discussions["scope"].iloc[0]
    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    def save(fig, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = figures_root / f"{stem}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)

    primary = discussions[discussions["scope"] == scope]
    monthly = primary.groupby("article_month")["story_id"].nunique().reindex(range(1, 13), fill_value=0)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(monthly.index, monthly.values, color="#356a8a")
    axis.set(xlabel="2025 month", ylabel="Eligible discussions", title=f"Monthly model-sample coverage ({scope})")
    axis.set_xticks(range(1, 13))
    save(fig, f"{scope}_monthly_coverage")

    overall_topics = topic_summary[
        (topic_summary["scope"] == scope) & (topic_summary["analysis_partition"] == "all_partitions")
    ].nlargest(15, "n_articles").sort_values("n_articles")
    fig, axis = plt.subplots(figsize=(9, max(4.5, 0.32 * len(overall_topics))))
    axis.barh(overall_topics["primary_topic"], overall_topics["n_articles"], color="#4b8b6f")
    axis.set(xlabel="Eligible discussions", title=f"Primary topics from section_1 ({scope})")
    save(fig, f"{scope}_topic_composition")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(primary["candidate_rows"], bins=40, color="#356a8a")
    axes[0].set(xlabel="Candidates per discussion", ylabel="Discussions")
    axes[1].hist(primary["n_picks"], bins=np.arange(0.5, primary["n_picks"].max() + 1.5), color="#d4744c")
    axes[1].set(xlabel="Curator picks per discussion", ylabel="Discussions")
    fig.suptitle(f"Discussion size and selection intensity ({scope})")
    save(fig, f"{scope}_discussion_size_and_picks")

    contrasts = feature_summary[
        (feature_summary["scope"] == scope)
        & (feature_summary["feature_source"] == "transformed_model_predictor")
    ].dropna(subset=["curator_minus_audience_sd"])
    contrasts = contrasts.assign(abs_contrast=contrasts["curator_minus_audience_sd"].abs()).nlargest(20, "abs_contrast").sort_values("curator_minus_audience_sd")
    fig, axis = plt.subplots(figsize=(9, max(5, 0.3 * len(contrasts))))
    colors = np.where(contrasts["curator_minus_audience_sd"] >= 0, "#4b8b6f", "#d4744c")
    axis.barh(contrasts["label"], contrasts["curator_minus_audience_sd"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Curator mean minus audience mean (candidate SDs)", title=f"Largest selected-feature contrasts ({scope})")
    save(fig, f"{scope}_selected_feature_contrasts")
    return outputs


def run_descriptive_analysis(
    *,
    model_data_root: Path = Path("model_output/selection_2025/model_data"),
    data_root: Path = Path("data/scrape_2025"),
    output_root: Path = Path("model_output/selection_2025/paper1/descriptives"),
    scopes: Iterable[str] = ("all",),
    year: int = 2025,
    threads: int = 4,
    make_figures: bool = True,
) -> dict:
    """Build the complete stage-5 artifact set and write its manifest last."""
    scopes = _normalise_scopes(scopes)
    output_root.mkdir(parents=True, exist_ok=True)
    split_path = model_data_root / "master_article_split.parquet"
    registry_path = model_data_root / "feature_manifest.json"
    provenance_path = model_data_root / "provenance_manifest.json"
    preprocessing_path = model_data_root / "preprocessing_manifest.json"
    for path in (split_path, registry_path, provenance_path):
        if not path.exists():
            raise FileNotFoundError(path)
    registry = json.loads(registry_path.read_text())
    provenance = json.loads(provenance_path.read_text())

    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, int(threads))}")
    try:
        topics = build_article_topics(data_root, year=year, connection=con)
        discussion_frames = []
        feature_frames = []
        for scope in scopes:
            choice_path = model_data_root / f"choice_set_{scope}.parquet"
            discussion_frames.append(
                _discussion_table(con, choice_path, split_path, topics, scope)
            )
            feature_frames.append(_feature_summary(con, choice_path, scope, registry))
    finally:
        con.close()

    discussions = pd.concat(discussion_frames, ignore_index=True)
    features = pd.concat(feature_frames, ignore_index=True)
    topic_audit = (
        topics.groupby(["primary_topic", "section_1", "section_2", "section_3"], dropna=False)
        .size().rename("n_articles").reset_index().sort_values("n_articles", ascending=False)
    )
    monthly = _summarise_discussions(discussions, ["scope", "analysis_partition", "article_month"])
    monthly_all = _summarise_discussions(
        discussions.assign(analysis_partition="all_partitions"),
        ["scope", "analysis_partition", "article_month"],
    )
    monthly = pd.concat([monthly, monthly_all], ignore_index=True)
    topic = _summarise_discussions(discussions, ["scope", "analysis_partition", "primary_topic"])
    topic_all = _summarise_discussions(
        discussions.assign(analysis_partition="all_partitions"),
        ["scope", "analysis_partition", "primary_topic"],
    )
    topic = pd.concat([topic, topic_all], ignore_index=True)
    sample_flow = _sample_flow(discussions, scopes, provenance)
    qa = provenance.get("qa", {})
    collection_qa = pd.DataFrame(
        [
            {
                "status": status,
                "n_articles": count,
                "qa_passed": qa.get("passed"),
                "nonterminal_stories": qa.get("nonterminal_stories"),
            }
            for status, count in qa.get("status_counts", {}).items()
        ],
        columns=["status", "n_articles", "qa_passed", "nonterminal_stories"],
    )

    table_paths = {
        "article_topics": output_root / "article_topics.parquet",
        "discussion_descriptives": output_root / "discussion_descriptives.parquet",
        "sample_flow": output_root / "sample_flow.csv",
        "collection_qa_status": output_root / "collection_qa_status.csv",
        "monthly_summary": output_root / "monthly_summary.csv",
        "topic_summary": output_root / "topic_summary.csv",
        "topic_hierarchy_audit": output_root / "topic_hierarchy_audit.csv",
        "feature_summary": output_root / "feature_summary.csv",
    }
    topics.to_parquet(table_paths["article_topics"], index=False)
    discussions.to_parquet(table_paths["discussion_descriptives"], index=False)
    sample_flow.to_csv(table_paths["sample_flow"], index=False)
    collection_qa.to_csv(table_paths["collection_qa_status"], index=False)
    monthly.to_csv(table_paths["monthly_summary"], index=False)
    topic.to_csv(table_paths["topic_summary"], index=False)
    topic_audit.to_csv(table_paths["topic_hierarchy_audit"], index=False)
    features.to_csv(table_paths["feature_summary"], index=False)
    figure_paths = _save_figures(discussions, topic, features, output_root) if make_figures else []

    preprocessing = json.loads(preprocessing_path.read_text()) if preprocessing_path.exists() else {}
    pre_outputs = preprocessing.get("outputs", {})
    input_records = {
        "article_dataset": {
            "path": str(data_root / "articles" / f"year={year}"),
            "dataset_fingerprint": provenance.get("source", {}).get("dataset_fingerprint"),
        },
        "feature_manifest": {"path": str(registry_path), "sha256": _sha256(registry_path)},
        "provenance_manifest": {"path": str(provenance_path), "sha256": _sha256(provenance_path)},
        "master_article_split": {
            "path": str(split_path),
            "sha256": pre_outputs.get("master_article_split.parquet", {}).get("sha256") or _sha256(split_path),
        },
    }
    for scope in scopes:
        path = model_data_root / f"choice_set_{scope}.parquet"
        input_records[f"choice_set_{scope}"] = {
            "path": str(path),
            "sha256": pre_outputs.get(path.name, {}).get("sha256") or _sha256(path),
        }

    outputs = {}
    row_counts = {
        "article_topics": len(topics),
        "discussion_descriptives": len(discussions),
        "sample_flow": len(sample_flow),
        "collection_qa_status": len(collection_qa),
        "monthly_summary": len(monthly),
        "topic_summary": len(topic),
        "topic_hierarchy_audit": len(topic_audit),
        "feature_summary": len(features),
    }
    for name, path in table_paths.items():
        outputs[name] = {"path": str(path), "sha256": _sha256(path), "rows": row_counts[name]}
    for path in figure_paths:
        outputs[f"figure:{path.name}"] = {"path": str(path), "sha256": _sha256(path)}

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "year": year,
        "scopes": list(scopes),
        "primary_scope": "all",
        "appendix_scope": "root",
        "split_display_mapping": {"paper2_test": "held_out_test"},
        "topic_mapping": {
            "version": TOPIC_MAPPING_VERSION,
            "primary": "trimmed section_1 without category collapsing",
            "missing": UNKNOWN_TOPIC,
            "audit_fields": ["section_1", "section_2", "section_3"],
        },
        "audience_tie_policy": "equal-weight mean across audience_selected_draw_01..10",
        "collection_caveat": "Vote totals are collection-time snapshots; only relative_votes survives the shared model contract.",
        "inputs": input_records,
        "outputs": outputs,
    }
    _write_json(manifest, output_root / "descriptive_manifest.json")
    return manifest
