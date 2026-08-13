from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .crawler import utc_now
from .storage import COMMENT_COLUMNS, FORUM_COLUMNS, ParquetStore
from .transform import effective_text


def _parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("migration requires PyArrow") from exc
    return pq


def migrate_existing(root: Path, year: int) -> dict[str, int]:
    """Idempotently upgrade completed local outputs without network access."""
    store = ParquetStore(root)
    pq = _parquet()
    comment_files = sorted(
        (root / "comments" / f"year={year}").glob("month=*/*.parquet")
    )
    forum_files = sorted(
        (root / "forums" / f"year={year}").glob("month=*/*.parquet")
    )
    diagnostic_story_ids = {
        path.stem
        for path in (root / "forum_pages" / f"year={year}").glob(
            "month=*/*.parquet"
        )
    }
    changed_comments = 0
    changed_forums = 0
    sticky_descendants_cleared = 0

    for path in comment_files:
        rows = pq.read_table(path).to_pylist()
        changed = False
        for row in rows:
            combined = effective_text(row.get("title"), row.get("text"))
            if row.get("effective_text") != combined:
                row["effective_text"] = combined
                changed = True
            # The legacy scraper inherited a pinned root's state into every
            # reply. Current API evidence also shows that an individually
            # sticky posting can legitimately sit below depth zero in its
            # normal thread. Only clean historical files which lack retained
            # sticky-page diagnostics; never rewrite evidenced new records.
            if (
                path.stem not in diagnostic_story_ids
                and row.get("is_sticky")
                and int(row.get("depth") or 0) > 0
            ):
                row["is_sticky"] = False
                sticky_descendants_cleared += 1
                changed = True
        if changed or set(pq.read_schema(path).names) != set(COMMENT_COLUMNS):
            store._write_atomic(path, rows, COMMENT_COLUMNS)
            changed_comments += 1

    for path in forum_files:
        rows = pq.read_table(path).to_pylist()
        changed = False
        for row in rows:
            reported = int(row.get("reported_posting_count") or 0)
            unique = int(row.get("observed_unique_count") or 0)
            published = int(
                row.get("observed_published_count")
                if row.get("observed_published_count") is not None
                else row.get("reconciled_posting_count") or 0
            )
            deleted = int(
                row.get("observed_deleted_count")
                if row.get("observed_deleted_count") is not None
                else max(0, unique - published)
            )
            defaults: dict[str, Any] = {
                "observed_published_count": published,
                "observed_deleted_count": deleted,
                "reconciled_posting_count": published,
                "posting_count_difference": published - reported,
                "posting_count_discrepancy_absolute": abs(published - reported),
                "posting_count_discrepancy_pct": (
                    abs(published - reported) / reported * 100.0 if reported else 0.0
                ),
                "count_discrepancy_reproduced": row.get("count_discrepancy_reproduced"),
                # Historical outputs did not retain enough evidence to reconstruct these.
                "pagination_page_count": row.get("pagination_page_count"),
                "root_edge_count": row.get("root_edge_count"),
                "flattened_record_count": row.get("flattened_record_count"),
                "cursor_walk_complete": row.get("cursor_walk_complete"),
                "cursor_progression_valid": row.get("cursor_progression_valid"),
                "crawl_status": row.get("crawl_status") or "completed",
            }
            for key, value in defaults.items():
                if row.get(key) != value:
                    row[key] = value
                    changed = True
        if changed or set(pq.read_schema(path).names) != set(FORUM_COLUMNS):
            store._write_atomic(path, rows, FORUM_COLUMNS)
            changed_forums += 1

    metadata_path = root / "collection_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {"year": year}
    )
    metadata["schema_version"] = max(3, int(metadata.get("schema_version") or 1))
    metadata["last_offline_migration_at"] = utc_now()
    metadata["historical_page_diagnostics_reconstructable"] = False
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)
    return {
        "comment_files_scanned": len(comment_files),
        "comment_files_rewritten": changed_comments,
        "forum_files_scanned": len(forum_files),
        "forum_files_rewritten": changed_forums,
        "sticky_descendants_cleared": sticky_descendants_cleared,
    }
