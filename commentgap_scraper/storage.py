from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Any

from .transform import merge_comment_records


ARTICLE_COLUMNS = {
    "story_id": "string",
    "year": "int32",
    "month": "int8",
    "canonical_url": "string",
    "published_at": "string",
    "published_at_source": "string",
    "modified_at": "string",
    "sitemap_lastmod": "string",
    "title": "string",
    "subtitle": "string",
    "body": "string",
    "section_1": "string",
    "section_2": "string",
    "section_3": "string",
    "collected_at": "string",
}

FORUM_COLUMNS = {
    "story_id": "string",
    "year": "int32",
    "month": "int8",
    "forum_id": "string",
    "flags_json": "string",
    "metadata_json": "string",
    "reported_posting_count": "int64",
    "observed_unique_count": "int64",
    "observed_published_count": "int64",
    "observed_deleted_count": "int64",
    "reconciled_posting_count": "int64",
    "posting_count_difference": "int64",
    "posting_count_discrepancy_absolute": "int64",
    "posting_count_discrepancy_pct": "float64",
    "count_discrepancy_reproduced": "bool",
    "pagination_page_count": "int64",
    "root_edge_count": "int64",
    "flattened_record_count": "int64",
    "cursor_walk_complete": "bool",
    "cursor_progression_valid": "bool",
    "crawl_status": "string",
    "collected_at": "string",
}

COMMENT_COLUMNS = {
    "comment_id": "string",
    "legacy_posting_id": "string",
    "story_id": "string",
    "forum_id": "string",
    "year": "int32",
    "month": "int8",
    "parent_comment_id": "string",
    "root_comment_id": "string",
    "depth": "int32",
    "root_order": "int64",
    "preorder_position": "int64",
    "display_order": "int64",
    "is_root": "bool",
    "is_leaf": "bool",
    "created_at": "string",
    "title": "string",
    "text": "string",
    "effective_text": "string",
    "lifecycle_status": "string",
    "flags_json": "string",
    "is_sticky": "bool",
    "votes_positive": "int64",
    "votes_negative": "int64",
    "author_hash": "string",
    "author_follower_count": "int64",
    "collected_at": "string",
}

FORUM_PAGE_COLUMNS = {
    "story_id": "string",
    "forum_id": "string",
    "year": "int32",
    "month": "int8",
    "page_index": "int32",
    "page_kind": "string",
    "request_cursor_hash": "string",
    "next_cursor_hash": "string",
    "root_edge_count": "int64",
    "flattened_record_count": "int64",
    "has_next_page": "bool",
    "cursor_progression_valid": "bool",
    "collected_at": "string",
}


def _modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support is required; install requirements-scraper.txt"
        ) from exc
    return pa, pq


def _schema(columns: dict[str, str]):
    pa, _ = _modules()
    types = {
        "string": pa.string(),
        "int8": pa.int8(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
    }
    return pa.schema([pa.field(name, types[type_name]) for name, type_name in columns.items()])


class ParquetStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _partition_dir(self, table: str, year: int, month: int) -> Path:
        path = self.root / table / f"year={year}" / f"month={month:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_atomic(
        self,
        path: Path,
        records: list[dict[str, Any]],
        columns: dict[str, str],
    ) -> None:
        pa, pq = _modules()
        normalized = [{name: record.get(name) for name in columns} for record in records]
        table = pa.Table.from_pylist(normalized, schema=_schema(columns))
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        os.replace(temporary, path)

    def write_article(self, record: dict[str, Any]) -> Path:
        path = self._partition_dir("articles", record["year"], record["month"]) / f"{record['story_id']}.parquet"
        self._write_atomic(path, [record], ARTICLE_COLUMNS)
        return path

    def write_forum(self, record: dict[str, Any]) -> Path:
        path = self._partition_dir("forums", record["year"], record["month"]) / f"{record['story_id']}.parquet"
        self._write_atomic(path, [record], FORUM_COLUMNS)
        return path

    def remove_published_outputs(self, story_id: str, year: int, month: int) -> None:
        """Remove outputs left by an interrupted or legacy failed finalization."""
        for table in ("comments", "forums", "forum_pages"):
            path = (
                self.root
                / table
                / f"year={year}"
                / f"month={month:02d}"
                / f"{story_id}.parquet"
            )
            path.unlink(missing_ok=True)

    def _staging_dir(self, story_id: str) -> Path:
        path = self.root / ".staging" / story_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reset_staging(self, story_id: str) -> None:
        path = self.root / ".staging" / story_id
        if path.exists():
            shutil.rmtree(path)

    def write_comment_page(
        self, story_id: str, page_index: int, records: list[dict[str, Any]]
    ) -> Path:
        path = self._staging_dir(story_id) / f"page-{page_index:06d}.parquet"
        self._write_atomic(path, records, COMMENT_COLUMNS)
        return path

    def write_page_diagnostic(self, story_id: str, record: dict[str, Any]) -> Path:
        path = self._staging_dir(story_id) / f"diagnostic-{int(record['page_index']):06d}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    def staging_diagnostics(self, story_id: str) -> list[dict[str, Any]]:
        path = self.root / ".staging" / story_id
        if not path.exists():
            return []
        return [
            json.loads(item.read_text(encoding="utf-8"))
            for item in sorted(path.glob("diagnostic-*.json"))
        ]

    def staging_pages(self, story_id: str) -> list[Path]:
        path = self.root / ".staging" / story_id
        return sorted(path.glob("page-*.parquet")) if path.exists() else []

    def prune_staging_pages(self, story_id: str, keep_count: int) -> None:
        """Discard only pages written after the last committed manifest checkpoint."""
        for page in self.staging_pages(story_id)[keep_count:]:
            page.unlink()
        path = self.root / ".staging" / story_id
        for diagnostic in sorted(path.glob("diagnostic-*.json"))[keep_count:]:
            diagnostic.unlink()

    def staging_record_count(self, story_id: str) -> int:
        """Read Parquet metadata only, avoiding a full scan for progress reporting."""
        _, pq = _modules()
        return sum(pq.ParquetFile(page).metadata.num_rows for page in self.staging_pages(story_id))

    def prepare_comments(
        self, story_id: str, year: int, month: int
    ) -> tuple[Path, int, int, int]:
        _, pq = _modules()
        records: list[dict[str, Any]] = []
        for page in self.staging_pages(story_id):
            records.extend(pq.read_table(page).to_pylist())
        merged = merge_comment_records(records)
        for record in merged:
            record["year"] = year
            record["month"] = month
        path = self._staging_dir(story_id) / "final.parquet"
        self._write_atomic(path, merged, COMMENT_COLUMNS)
        published = sum(row.get("lifecycle_status") == "Published" for row in merged)
        deleted = sum(row.get("lifecycle_status") == "Deleted" for row in merged)
        return path, len(merged), published, deleted

    def publish_prepared_comments(self, story_id: str, year: int, month: int) -> Path:
        source = self.root / ".staging" / story_id / "final.parquet"
        if not source.exists():
            raise FileNotFoundError(f"prepared comments missing for story {story_id}")
        destination = self._partition_dir("comments", year, month) / f"{story_id}.parquet"
        os.replace(source, destination)
        diagnostics = self.staging_diagnostics(story_id)
        if diagnostics:
            diagnostics_path = (
                self._partition_dir("forum_pages", year, month) / f"{story_id}.parquet"
            )
            self._write_atomic(diagnostics_path, diagnostics, FORUM_PAGE_COLUMNS)
        self.reset_staging(story_id)
        return destination

    def finalize_comments(
        self, story_id: str, year: int, month: int
    ) -> tuple[Path, int, int, int]:
        _, unique, published, deleted = self.prepare_comments(story_id, year, month)
        path = self.publish_prepared_comments(story_id, year, month)
        return path, unique, published, deleted

    def export_manifest(
        self, rows: list[dict[str, Any]], year: int, month: int | None = None
    ) -> Path:
        if not rows:
            raise ValueError("cannot export an empty manifest")
        pa, pq = _modules()
        path = self.root / "crawl_manifest" / f"year={year}"
        if month is not None:
            path = path / f"month={month:02d}"
        path.mkdir(parents=True, exist_ok=True)
        destination = path / "manifest.parquet"
        table = pa.Table.from_pylist(rows)
        temporary = destination.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, destination)
        return destination
