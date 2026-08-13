from __future__ import annotations

from pathlib import Path


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "legacy export requires DuckDB; install requirements-scraper.txt"
        ) from exc
    return duckdb


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def export_legacy(root: Path, year: int) -> tuple[Path, Path]:
    comments = _quoted(root / "comments" / f"year={year}" / "month=*" / "*.parquet")
    articles = _quoted(root / "articles" / f"year={year}" / "month=*" / "*.parquet")
    destination = root / "legacy" / f"year={year}"
    destination.mkdir(parents=True, exist_ok=True)
    comment_output = destination / "comments_legacy.parquet"
    article_output = destination / "articles_legacy.parquet"

    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            COPY (
                SELECT
                    'content' || c.comment_id AS comment_id,
                    'content' || c.root_comment_id AS root_of_tree,
                    regexp_replace(a.canonical_url, '^https://www\\.derstandard\\.at', '') AS article,
                    c.author_hash AS user_names,
                    c.author_follower_count AS user_follower,
                    c.created_at AS timestamp,
                    c.title AS heading,
                    c.text,
                    CASE WHEN c.is_sticky THEN ' Angeheftet ·' ELSE '' END AS pinned,
                    c.is_root AS is_root_comment,
                    c.is_leaf AS is_leaf_comment,
                    c.depth AS level_in_tree,
                    c.votes_positive AS votes_pos,
                    c.votes_negative AS votes_neg,
                    c.comment_id AS id,
                    c.created_at AS timestamp_f,
                    CASE WHEN c.is_sticky THEN 1 ELSE 0 END AS pinned_f,
                    c.votes_positive - c.votes_negative AS votes_rel,
                    c.display_order AS order_all
                FROM read_parquet('{comments}', hive_partitioning=false) c
                JOIN read_parquet('{articles}', hive_partitioning=false) a USING (story_id)
                ORDER BY c.story_id, c.display_order
            ) TO '{_quoted(comment_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        connection.execute(
            f"""
            COPY (
                SELECT
                    title,
                    subtitle,
                    section_1 AS genre1,
                    section_2 AS genre2,
                    section_3 AS genre3,
                    published_at AS date,
                    body AS text,
                    regexp_replace(canonical_url, '^https://www\\.derstandard\\.at', '') AS url
                FROM read_parquet('{articles}', hive_partitioning=false)
                ORDER BY story_id
            ) TO '{_quoted(article_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        connection.close()
    return comment_output, article_output
