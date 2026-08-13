from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import Manifest


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "validation requires DuckDB; install requirements-scraper.txt"
        ) from exc
    return duckdb


def _glob(root: Path, table: str, year: int) -> str:
    return str(root / table / f"year={year}" / "month=*" / "*.parquet").replace("'", "''")


def validate_row_invariants(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = {row.get("comment_id") for row in rows}
    if len(ids) != len(rows):
        errors.append("duplicate comment IDs")
    for row in rows:
        if int(row.get("votes_positive") or 0) < 0 or int(row.get("votes_negative") or 0) < 0:
            errors.append(f"negative reaction count for {row.get('comment_id')}")
        parent = row.get("parent_comment_id")
        if parent and parent not in ids:
            errors.append(f"missing parent {parent} for {row.get('comment_id')}")
        author_hash = row.get("author_hash")
        if author_hash and (not str(author_hash).startswith("author_") or len(str(author_hash)) != 71):
            errors.append(f"invalid author hash for {row.get('comment_id')}")
    return errors


def validate_dataset(root: Path, year: int, *, allow_incomplete: bool = False) -> dict[str, Any]:
    with Manifest(root / "crawl_manifest.sqlite3") as manifest:
        manifest_rows = manifest.rows(year)
        status_counts = manifest.status_counts(year)

    terminal = {"completed", "no_forum", "no_postings", "inaccessible", "failed"}
    nonterminal = sum(
        count for status, count in status_counts.items() if status not in terminal
    )
    monthly_manifest: dict[str, dict[str, int]] = {}
    for row in manifest_rows:
        month = f"{int(row['month']):02d}"
        bucket = monthly_manifest.setdefault(month, {"discovered": 0})
        bucket["discovered"] += 1
        status = str(row["status"])
        bucket[status] = bucket.get(status, 0) + 1
    terminal_count = len(manifest_rows) - nonterminal
    summary: dict[str, Any] = {
        "year": year,
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "discovered_stories": len(manifest_rows),
        "status_counts": status_counts,
        "nonterminal_stories": nonterminal,
        "terminal_stories": terminal_count,
        "terminal_coverage_rate": (
            terminal_count / len(manifest_rows) if manifest_rows else 0.0
        ),
        "monthly_manifest": monthly_manifest,
        "allow_incomplete": allow_incomplete,
    }

    comments_glob = _glob(root, "comments", year)
    forums_glob = _glob(root, "forums", year)
    articles_glob = _glob(root, "articles", year)
    has_comments = bool(list((root / "comments" / f"year={year}").glob("month=*/*.parquet")))
    has_forums = bool(list((root / "forums" / f"year={year}").glob("month=*/*.parquet")))
    has_articles = bool(list((root / "articles" / f"year={year}").glob("month=*/*.parquet")))

    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        if has_articles:
            article_metrics = connection.execute(
                f"""
                SELECT
                    COUNT(*),
                    COUNT(*) FILTER (WHERE published_at IS NULL),
                    COUNT(*) FILTER (WHERE title IS NULL OR trim(title) = ''),
                    COUNT(*) FILTER (WHERE body IS NULL OR trim(body) = ''),
                    COUNT(*) FILTER (WHERE section_1 IS NULL OR trim(section_1) = '')
                FROM read_parquet('{articles_glob}', hive_partitioning=false)
                """
            ).fetchone()
            summary["article_rows"] = article_metrics[0]
            summary["article_missingness"] = {
                "published_at": article_metrics[1],
                "title": article_metrics[2],
                "body": article_metrics[3],
                "section_1": article_metrics[4],
            }
            summary["publication_time_sources"] = {
                str(source or "missing"): count
                for source, count in connection.execute(
                    f"""
                    SELECT published_at_source, COUNT(*)
                    FROM read_parquet('{articles_glob}', hive_partitioning=false)
                    GROUP BY published_at_source
                    ORDER BY published_at_source
                    """
                ).fetchall()
            }
            summary["monthly_article_rows"] = {
                f"{int(month):02d}": count
                for month, count in connection.execute(
                    f"""
                    SELECT month, COUNT(*)
                    FROM read_parquet('{articles_glob}', hive_partitioning=false)
                    GROUP BY month ORDER BY month
                    """
                ).fetchall()
            }
        else:
            summary["article_rows"] = 0
            summary["article_missingness"] = {
                "published_at": 0,
                "title": 0,
                "body": 0,
                "section_1": 0,
            }
            summary["publication_time_sources"] = {}
            summary["monthly_article_rows"] = {}

        if has_forums:
            summary["forum_rows"] = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet('{forums_glob}', hive_partitioning=false)"
            ).fetchone()[0]
            summary["monthly_forum_rows"] = {
                f"{int(month):02d}": count
                for month, count in connection.execute(
                    f"""
                    SELECT month, COUNT(*)
                    FROM read_parquet('{forums_glob}', hive_partitioning=false)
                    GROUP BY month ORDER BY month
                    """
                ).fetchall()
            }
        else:
            summary["forum_rows"] = 0
            summary["monthly_forum_rows"] = {}
        if has_comments:
            metrics = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS rows,
                    COUNT(DISTINCT comment_id) AS distinct_ids,
                    COUNT(*) FILTER (WHERE votes_positive < 0 OR votes_negative < 0) AS negative_votes,
                    COUNT(*) FILTER (WHERE lifecycle_status <> 'Published') AS tombstones,
                    COUNT(*) FILTER (WHERE created_at IS NULL) AS missing_created,
                    COUNT(*) FILTER (WHERE text IS NULL OR trim(text) = '') AS missing_text,
                    COUNT(*) FILTER (WHERE author_hash IS NULL) AS missing_author,
                    COUNT(*) FILTER (
                        WHERE author_hash IS NOT NULL
                          AND NOT regexp_matches(author_hash, '^author_[0-9a-f]{{64}}$')
                    ) AS invalid_hashes
                FROM read_parquet('{comments_glob}', hive_partitioning=false)
                """
            ).fetchone()
            summary.update(
                {
                    "comment_rows": metrics[0],
                    "distinct_comment_ids": metrics[1],
                    "duplicate_comment_ids": metrics[0] - metrics[1],
                    "negative_reaction_rows": metrics[2],
                    "deletion_tombstone_rows": metrics[3],
                    "comment_missingness": {
                        "created_at": metrics[4],
                        "text": metrics[5],
                        "author_hash": metrics[6],
                    },
                    "invalid_author_hash_rows": metrics[7],
                }
            )
            summary["monthly_comment_rows"] = {
                f"{int(month):02d}": count
                for month, count in connection.execute(
                    f"""
                    SELECT month, COUNT(*)
                    FROM read_parquet('{comments_glob}', hive_partitioning=false)
                    GROUP BY month ORDER BY month
                    """
                ).fetchall()
            }
            column_names = {
                str(row[0]).lower()
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{comments_glob}', hive_partitioning=false)"
                ).fetchall()
            }
            forbidden_author_columns = {
                "author_id",
                "author_name",
                "community_identity_id",
                "community_name",
                "legacy_community_identity_id",
                "legacy_community_name",
            }
            summary["raw_author_identifier_columns"] = sorted(
                column_names & forbidden_author_columns
            )
            metadata_path = root / "collection_metadata.json"
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.exists()
                else {}
            )
            reply_depth = int(metadata.get("reply_query_depth") or 32)
            summary["reply_query_depth"] = reply_depth
            summary["reply_depth_limit_rows"] = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet('{comments_glob}', hive_partitioning=false) WHERE depth >= ?",
                [reply_depth],
            ).fetchone()[0]
            relationships = connection.execute(
                f"""
                WITH comments AS (
                    SELECT comment_id, parent_comment_id, root_comment_id, forum_id,
                           depth, is_root
                    FROM read_parquet('{comments_glob}', hive_partitioning=false)
                )
                SELECT
                    COUNT(*) FILTER (
                        WHERE child.parent_comment_id IS NOT NULL
                          AND parent.comment_id IS NULL
                    ) AS missing_parents,
                    COUNT(*) FILTER (
                        WHERE child.root_comment_id IS NULL
                           OR root_post.comment_id IS NULL
                    ) AS missing_roots,
                    COUNT(*) FILTER (
                        WHERE child.depth < 0
                           OR (
                               child.depth = 0 AND (
                                   child.parent_comment_id IS NOT NULL
                                   OR child.comment_id <> child.root_comment_id
                                   OR NOT child.is_root
                               )
                           )
                           OR (
                               child.depth > 0 AND (
                                   child.parent_comment_id IS NULL
                                   OR child.comment_id = child.root_comment_id
                                   OR child.is_root
                                   OR parent.depth <> child.depth - 1
                                   OR parent.root_comment_id <> child.root_comment_id
                               )
                           )
                    ) AS invalid_tree_rows
                FROM comments child
                LEFT JOIN comments parent
                  ON child.forum_id = parent.forum_id
                 AND child.parent_comment_id = parent.comment_id
                LEFT JOIN comments root_post
                  ON child.forum_id = root_post.forum_id
                 AND child.root_comment_id = root_post.comment_id
                 AND root_post.depth = 0
                """
            ).fetchone()
            summary["missing_parent_rows"] = relationships[0]
            summary["missing_root_rows"] = relationships[1]
            summary["invalid_tree_relationship_rows"] = relationships[2]
        else:
            summary.update(
                {
                    "comment_rows": 0,
                    "distinct_comment_ids": 0,
                    "duplicate_comment_ids": 0,
                    "negative_reaction_rows": 0,
                    "deletion_tombstone_rows": 0,
                    "comment_missingness": {
                        "created_at": 0,
                        "text": 0,
                        "author_hash": 0,
                    },
                    "invalid_author_hash_rows": 0,
                    "missing_parent_rows": 0,
                    "missing_root_rows": 0,
                    "invalid_tree_relationship_rows": 0,
                    "reply_depth_limit_rows": 0,
                    "monthly_comment_rows": {},
                    "raw_author_identifier_columns": [],
                }
            )
    finally:
        connection.close()

    completed_mismatches = [
        row["story_id"]
        for row in manifest_rows
        if row["status"] == "completed"
        and row.get("expected_count") != row.get("observed_count")
    ]
    summary["manifest_count_mismatches"] = len(completed_mismatches)
    summary["manifest_count_mismatch_examples"] = completed_mismatches[:20]
    if has_forums:
        qa_connection = _duckdb().connect()
        try:
            summary["forum_count_mismatches"] = qa_connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{forums_glob}', hive_partitioning=false)
                WHERE reported_posting_count <> reconciled_posting_count
                """
            ).fetchone()[0]
        finally:
            qa_connection.close()
    else:
        summary["forum_count_mismatches"] = 0
    summary["passed"] = all(
        summary[key] == 0
        for key in (
            "duplicate_comment_ids",
            "negative_reaction_rows",
            "invalid_author_hash_rows",
            "missing_parent_rows",
            "missing_root_rows",
            "invalid_tree_relationship_rows",
            "manifest_count_mismatches",
            "forum_count_mismatches",
            "reply_depth_limit_rows",
        )
    ) and not summary["raw_author_identifier_columns"] and (
        allow_incomplete or summary["nonterminal_stories"] == 0
    ) and status_counts.get("failed", 0) == 0

    qa_dir = root / "qa_summary" / f"year={year}"
    qa_dir.mkdir(parents=True, exist_ok=True)
    json_path = qa_dir / "summary.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, json_path)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("validation output requires PyArrow") from exc
    flat_summary = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in summary.items()
    }
    parquet_path = qa_dir / "summary.parquet"
    parquet_temp = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist([flat_summary]), parquet_temp, compression="zstd")
    os.replace(parquet_temp, parquet_path)
    return summary
