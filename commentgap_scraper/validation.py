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

    terminal = {
        "completed",
        "completed_with_count_discrepancy",
        "no_forum",
        "no_postings",
        "inaccessible",
        "failed",
    }
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
    forum_pages_glob = _glob(root, "forum_pages", year)
    has_comments = bool(list((root / "comments" / f"year={year}").glob("month=*/*.parquet")))
    has_forums = bool(list((root / "forums" / f"year={year}").glob("month=*/*.parquet")))
    has_articles = bool(list((root / "articles" / f"year={year}").glob("month=*/*.parquet")))
    has_forum_pages = bool(
        list((root / "forum_pages" / f"year={year}").glob("month=*/*.parquet"))
    )

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
            summary["article_body_missingness_by_section"] = {
                str(section or "<missing>"): {"articles": rows, "missing_body": missing}
                for section, rows, missing in connection.execute(
                    f"""
                    SELECT section_2, COUNT(*), COUNT(*) FILTER (
                        WHERE body IS NULL OR trim(body) = ''
                    )
                    FROM read_parquet('{articles_glob}', hive_partitioning=false)
                    GROUP BY section_2 ORDER BY COUNT(*) DESC, section_2
                    """
                ).fetchall()
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
            summary["article_body_missingness_by_section"] = {}
            summary["monthly_article_rows"] = {}

        if has_forums:
            summary["forum_rows"] = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet('{forums_glob}', hive_partitioning=false, union_by_name=true)"
            ).fetchone()[0]
            forum_columns = {
                str(row[0]).lower()
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{forums_glob}', hive_partitioning=false, union_by_name=true)"
                ).fetchall()
            }
            summary["forum_discrepancy_schema_available"] = "crawl_status" in forum_columns
            summary["monthly_forum_rows"] = {
                f"{int(month):02d}": count
                for month, count in connection.execute(
                    f"""
                    SELECT month, COUNT(*)
                    FROM read_parquet('{forums_glob}', hive_partitioning=false, union_by_name=true)
                    GROUP BY month ORDER BY month
                    """
                ).fetchall()
            }
        else:
            summary["forum_rows"] = 0
            forum_columns = set()
            summary["forum_discrepancy_schema_available"] = False
            summary["monthly_forum_rows"] = {}
        if has_comments:
            column_names = {
                str(row[0]).lower()
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{comments_glob}', hive_partitioning=false, union_by_name=true)"
                ).fetchall()
            }
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
                    COUNT(*) FILTER (WHERE is_sticky AND depth > 0) AS sticky_descendants,
                    COUNT(*) FILTER (
                        WHERE author_hash IS NOT NULL
                          AND NOT regexp_matches(author_hash, '^author_[0-9a-f]{{64}}$')
                    ) AS invalid_hashes
                FROM read_parquet('{comments_glob}', hive_partitioning=false, union_by_name=true)
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
                    "sticky_descendant_rows": metrics[7],
                    "invalid_author_hash_rows": metrics[8],
                }
            )
            content_rows = connection.execute(
                f"""
                SELECT
                    coalesce(lifecycle_status, '<NULL>') AS lifecycle_status,
                    COUNT(*) AS rows,
                    COUNT(*) FILTER (
                        WHERE (title IS NULL OR trim(title) = '')
                          AND (text IS NULL OR trim(text) = '')
                    ) AS missing_all_content,
                    COUNT(*) FILTER (
                        WHERE title IS NOT NULL AND trim(title) <> ''
                          AND (text IS NULL OR trim(text) = '')
                    ) AS title_only,
                    COUNT(*) FILTER (
                        WHERE (title IS NULL OR trim(title) = '')
                          AND text IS NOT NULL AND trim(text) <> ''
                    ) AS text_only,
                    COUNT(*) FILTER (
                        WHERE title IS NOT NULL AND trim(title) <> ''
                          AND text IS NOT NULL AND trim(text) <> ''
                    ) AS title_and_text
                FROM read_parquet(
                    '{comments_glob}', hive_partitioning=false, union_by_name=true
                )
                GROUP BY lifecycle_status
                ORDER BY lifecycle_status
                """
            ).fetchall()
            summary["comment_content_by_lifecycle"] = {
                str(row[0]): {
                    "rows": row[1],
                    "missing_all_content": row[2],
                    "title_only": row[3],
                    "text_only": row[4],
                    "title_and_text": row[5],
                }
                for row in content_rows
            }
            effective_expression = (
                "coalesce(nullif(trim(effective_text), ''), CASE WHEN coalesce(trim(title), '') <> '' OR coalesce(trim(text), '') <> '' THEN 'present' END)"
                if "effective_text" in column_names
                else "CASE WHEN coalesce(trim(title), '') <> '' OR coalesce(trim(text), '') <> '' THEN 'present' END"
            )
            summary["missing_effective_comment_text_rows"] = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    '{comments_glob}', hive_partitioning=false, union_by_name=true
                )
                WHERE {effective_expression} IS NULL OR trim({effective_expression}) = ''
                """
            ).fetchone()[0]
            summary["monthly_comment_rows"] = {
                f"{int(month):02d}": count
                for month, count in connection.execute(
                    f"""
                    SELECT month, COUNT(*)
                    FROM read_parquet('{comments_glob}', hive_partitioning=false, union_by_name=true)
                    GROUP BY month ORDER BY month
                    """
                ).fetchall()
            }
            summary["comment_created_year_counts"] = {
                str(created_year): count
                for created_year, count in connection.execute(
                    f"""
                    SELECT year(try_cast(created_at AS TIMESTAMP)), COUNT(*)
                    FROM read_parquet(
                        '{comments_glob}', hive_partitioning=false, union_by_name=true
                    )
                    GROUP BY 1 ORDER BY 1
                    """
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
                f"SELECT COUNT(*) FROM read_parquet('{comments_glob}', hive_partitioning=false, union_by_name=true) WHERE depth >= ?",
                [reply_depth],
            ).fetchone()[0]
            relationships = connection.execute(
                f"""
                WITH comments AS (
                    SELECT comment_id, parent_comment_id, root_comment_id, forum_id,
                           depth, is_root
                    FROM read_parquet('{comments_glob}', hive_partitioning=false, union_by_name=true)
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
            if has_forums:
                summary["comments_without_forum_rows"] = connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet('{comments_glob}', hive_partitioning=false, union_by_name=true) comments
                    LEFT JOIN read_parquet(
                        '{forums_glob}', hive_partitioning=false, union_by_name=true
                    ) forums
                      ON comments.story_id = forums.story_id
                     AND comments.forum_id = forums.forum_id
                    WHERE forums.forum_id IS NULL
                    """
                ).fetchone()[0]
            else:
                summary["comments_without_forum_rows"] = summary["comment_rows"]
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
                    "sticky_descendant_rows": 0,
                    "missing_effective_comment_text_rows": 0,
                    "comment_content_by_lifecycle": {},
                    "missing_parent_rows": 0,
                    "missing_root_rows": 0,
                    "invalid_tree_relationship_rows": 0,
                    "reply_depth_limit_rows": 0,
                    "monthly_comment_rows": {},
                    "comment_created_year_counts": {},
                    "raw_author_identifier_columns": [],
                    "comments_without_forum_rows": 0,
                }
            )

        if has_forum_pages:
            page_metrics = connection.execute(
                f"""
                WITH pages AS (
                    SELECT *,
                           row_number() OVER (
                               PARTITION BY story_id
                               ORDER BY page_index DESC
                           ) AS reverse_page
                    FROM read_parquet(
                        '{forum_pages_glob}', hive_partitioning=false, union_by_name=true
                    )
                    WHERE page_kind = 'threads'
                )
                SELECT
                    COUNT(*),
                    COUNT(DISTINCT story_id),
                    COUNT(*) FILTER (WHERE NOT cursor_progression_valid),
                    COUNT(*) FILTER (WHERE reverse_page = 1 AND has_next_page),
                    COUNT(*) FILTER (
                        WHERE request_cursor_hash IS NOT NULL
                          AND NOT regexp_matches(request_cursor_hash, '^[0-9a-f]{{64}}$')
                    ),
                    COUNT(*) FILTER (
                        WHERE next_cursor_hash IS NOT NULL
                          AND NOT regexp_matches(next_cursor_hash, '^[0-9a-f]{{64}}$')
                    )
                FROM pages
                """
            ).fetchone()
            summary["forum_page_rows"] = page_metrics[0]
            summary["forum_page_diagnostic_stories"] = page_metrics[1]
            summary["invalid_cursor_progression_pages"] = page_metrics[2]
            summary["incomplete_terminal_page_walks"] = page_metrics[3]
            summary["invalid_cursor_hash_rows"] = page_metrics[4] + page_metrics[5]
            if has_forums and "pagination_page_count" in forum_columns:
                summary["forum_page_aggregate_mismatches"] = connection.execute(
                    f"""
                    WITH page_totals AS (
                        SELECT
                            story_id,
                            COUNT(*) FILTER (WHERE page_kind = 'threads') AS page_count,
                            coalesce(SUM(root_edge_count) FILTER (
                                WHERE page_kind = 'threads'
                            ), 0) AS roots,
                            SUM(flattened_record_count) AS flattened
                        FROM read_parquet(
                            '{forum_pages_glob}', hive_partitioning=false, union_by_name=true
                        )
                        GROUP BY story_id
                    )
                    SELECT COUNT(*)
                    FROM read_parquet(
                        '{forums_glob}', hive_partitioning=false, union_by_name=true
                    ) forums
                    LEFT JOIN page_totals USING (story_id)
                    WHERE forums.pagination_page_count IS NOT NULL
                      AND forums.crawl_status IN (
                          'completed', 'completed_with_count_discrepancy'
                      )
                      AND (
                          page_totals.story_id IS NULL
                          OR forums.pagination_page_count <> page_totals.page_count
                          OR forums.root_edge_count <> page_totals.roots
                          OR forums.flattened_record_count <> page_totals.flattened
                      )
                    """
                ).fetchone()[0]
            else:
                summary["forum_page_aggregate_mismatches"] = 0
            if has_comments:
                sticky_metrics = connection.execute(
                    f"""
                    WITH expected AS (
                        SELECT story_id, SUM(root_edge_count) AS expected_sticky
                        FROM read_parquet(
                            '{forum_pages_glob}',
                            hive_partitioning=false,
                            union_by_name=true
                        )
                        WHERE page_kind = 'sticky'
                        GROUP BY story_id
                    ), actual AS (
                        SELECT story_id, COUNT(*) FILTER (WHERE is_sticky) AS observed_sticky
                        FROM read_parquet(
                            '{comments_glob}',
                            hive_partitioning=false,
                            union_by_name=true
                        )
                        GROUP BY story_id
                    )
                    SELECT
                        COUNT(*),
                        coalesce(SUM(expected.expected_sticky), 0),
                        coalesce(SUM(actual.observed_sticky), 0),
                        COUNT(*) FILTER (
                            WHERE coalesce(actual.observed_sticky, 0)
                               <> expected.expected_sticky
                        )
                    FROM expected
                    LEFT JOIN actual USING (story_id)
                    """
                ).fetchone()
                summary["sticky_reconciliation_stories"] = sticky_metrics[0]
                summary["api_sticky_record_count"] = sticky_metrics[1]
                summary["observed_sticky_record_count"] = sticky_metrics[2]
                summary["sticky_count_mismatch_stories"] = sticky_metrics[3]
            else:
                summary["sticky_reconciliation_stories"] = 0
                summary["api_sticky_record_count"] = 0
                summary["observed_sticky_record_count"] = 0
                summary["sticky_count_mismatch_stories"] = 0
        else:
            summary["forum_page_rows"] = 0
            summary["forum_page_diagnostic_stories"] = 0
            summary["invalid_cursor_progression_pages"] = 0
            summary["incomplete_terminal_page_walks"] = 0
            summary["invalid_cursor_hash_rows"] = 0
            summary["forum_page_aggregate_mismatches"] = 0
            summary["sticky_reconciliation_stories"] = 0
            summary["api_sticky_record_count"] = 0
            summary["observed_sticky_record_count"] = 0
            summary["sticky_count_mismatch_stories"] = 0
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
    discrepancy_status_errors = [
        row["story_id"]
        for row in manifest_rows
        if row["status"] == "completed_with_count_discrepancy"
        and row.get("expected_count") == row.get("observed_count")
    ]
    summary["invalid_discrepancy_status_rows"] = len(discrepancy_status_errors)
    summary["invalid_discrepancy_status_examples"] = discrepancy_status_errors[:20]
    if has_forums:
        qa_connection = _duckdb().connect()
        try:
            summary["forum_count_mismatches"] = qa_connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{forums_glob}', hive_partitioning=false, union_by_name=true)
                WHERE reported_posting_count <> reconciled_posting_count
                """
            ).fetchone()[0]
            if "crawl_status" in forum_columns:
                forum_discrepancy_metrics = qa_connection.execute(
                    f"""
                    SELECT
                        COUNT(*) FILTER (
                            WHERE reported_posting_count <> reconciled_posting_count
                              AND coalesce(crawl_status, 'completed')
                                  <> 'completed_with_count_discrepancy'
                        ),
                        COUNT(*) FILTER (
                            WHERE reported_posting_count = reconciled_posting_count
                              AND crawl_status = 'completed_with_count_discrepancy'
                        ),
                        COUNT(*) FILTER (
                            WHERE crawl_status = 'completed_with_count_discrepancy'
                              AND count_discrepancy_reproduced
                        ),
                        COUNT(*) FILTER (
                            WHERE crawl_status = 'completed_with_count_discrepancy'
                              AND NOT count_discrepancy_reproduced
                        ),
                        COUNT(*) FILTER (
                            WHERE crawl_status = 'completed_with_count_discrepancy'
                              AND count_discrepancy_reproduced IS NULL
                        )
                    FROM read_parquet(
                        '{forums_glob}', hive_partitioning=false, union_by_name=true
                    )
                    """
                ).fetchone()
                summary["unexpected_forum_count_mismatches"] = forum_discrepancy_metrics[0]
                summary["invalid_forum_discrepancy_status_rows"] = forum_discrepancy_metrics[1]
                summary["reproduced_count_discrepancy_forums"] = forum_discrepancy_metrics[2]
                summary["changed_count_discrepancy_on_retry_forums"] = forum_discrepancy_metrics[3]
                summary["count_discrepancy_forums_not_recrawled"] = forum_discrepancy_metrics[4]
            else:
                summary["unexpected_forum_count_mismatches"] = summary["forum_count_mismatches"]
                summary["invalid_forum_discrepancy_status_rows"] = 0
                summary["reproduced_count_discrepancy_forums"] = 0
                summary["changed_count_discrepancy_on_retry_forums"] = 0
                summary["count_discrepancy_forums_not_recrawled"] = 0
        finally:
            qa_connection.close()
    else:
        summary["forum_count_mismatches"] = 0
        summary["unexpected_forum_count_mismatches"] = 0
        summary["invalid_forum_discrepancy_status_rows"] = 0
        summary["reproduced_count_discrepancy_forums"] = 0
        summary["changed_count_discrepancy_on_retry_forums"] = 0
        summary["count_discrepancy_forums_not_recrawled"] = 0
    summary["passed"] = all(
        summary[key] == 0
        for key in (
            "duplicate_comment_ids",
            "negative_reaction_rows",
            "invalid_author_hash_rows",
            "sticky_count_mismatch_stories",
            "missing_parent_rows",
            "missing_root_rows",
            "invalid_tree_relationship_rows",
            "comments_without_forum_rows",
            "manifest_count_mismatches",
            "invalid_discrepancy_status_rows",
            "unexpected_forum_count_mismatches",
            "invalid_forum_discrepancy_status_rows",
            "invalid_cursor_progression_pages",
            "incomplete_terminal_page_walks",
            "invalid_cursor_hash_rows",
            "forum_page_aggregate_mismatches",
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
