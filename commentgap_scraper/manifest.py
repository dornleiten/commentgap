from __future__ import annotations

import sqlite3
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .parsing import DiscoveredStory


TERMINAL_STATUSES = (
    "completed",
    "completed_with_count_discrepancy",
    "no_forum",
    "no_postings",
    "inaccessible",
    "failed",
)


class Manifest:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS stories (
                story_id TEXT PRIMARY KEY,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                url TEXT NOT NULL,
                sitemap_lastmod TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                forum_id TEXT,
                expected_count INTEGER,
                observed_count INTEGER,
                next_cursor TEXT,
                page_index INTEGER NOT NULL DEFAULT 0,
                pagination_complete INTEGER NOT NULL DEFAULT 0,
                error_category TEXT,
                error_message TEXT,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS stories_year_status
                ON stories(year, status, month, story_id);
            """
        )
        columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(stories)")
        }
        if "pagination_complete" not in columns:
            self.connection.execute(
                "ALTER TABLE stories ADD COLUMN pagination_complete INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.commit()

    def upsert_discovered(self, stories: Iterable[DiscoveredStory]) -> int:
        rows = [
            (story.story_id, story.year, story.month, story.url, story.sitemap_lastmod)
            for story in stories
        ]
        self.connection.executemany(
            """
            INSERT INTO stories(story_id, year, month, url, sitemap_lastmod)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(story_id) DO UPDATE SET
                year=excluded.year,
                month=excluded.month,
                url=excluded.url,
                sitemap_lastmod=excluded.sitemap_lastmod,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)

    def stories_for_crawl(
        self,
        year: int,
        *,
        retry_failed: bool = False,
        only_failed: bool = False,
        limit: int | None = None,
        story_ids: list[str] | None = None,
        monthly_random: bool = False,
        selection_seed: int = 2025,
    ) -> list[dict[str, Any]]:
        statuses = ["failed", "in_progress"] if only_failed else ["pending", "in_progress"]
        if retry_failed and not only_failed:
            statuses.append("failed")
        placeholders = ",".join("?" for _ in statuses)
        params: list[Any] = [year, *statuses]
        query = f"SELECT * FROM stories WHERE year=? AND status IN ({placeholders})"
        if story_ids:
            query += f" AND story_id IN ({','.join('?' for _ in story_ids)})"
            params.extend(story_ids)
        if not monthly_random:
            query += " ORDER BY month, story_id"
        if limit is not None and not monthly_random:
            query += " LIMIT ?"
            params.append(limit)
        rows = [dict(row) for row in self.connection.execute(query, params)]
        if monthly_random:
            by_month: dict[int, list[dict[str, Any]]] = {}
            for row in rows:
                by_month.setdefault(int(row["month"]), []).append(row)
            for month, month_rows in by_month.items():
                random.Random(f"{selection_seed}:{month}").shuffle(month_rows)
            rows = [
                row
                for rank in range(max((len(values) for values in by_month.values()), default=0))
                for month in sorted(by_month)
                for row in by_month[month][rank : rank + 1]
            ]
            if limit is not None:
                rows = rows[:limit]
        return rows

    def mark_in_progress(self, story_id: str, started_at: str) -> None:
        self.connection.execute(
            """
            UPDATE stories SET status='in_progress', attempts=attempts+1,
                started_at=COALESCE(started_at, ?), finished_at=NULL,
                error_category=NULL, error_message=NULL, updated_at=CURRENT_TIMESTAMP
            WHERE story_id=?
            """,
            (started_at, story_id),
        )
        self.connection.commit()

    def save_progress(
        self,
        story_id: str,
        *,
        forum_id: str,
        expected_count: int,
        next_cursor: str | None,
        page_index: int,
        pagination_complete: bool = False,
    ) -> None:
        self.connection.execute(
            """
            UPDATE stories SET forum_id=?, expected_count=?, next_cursor=?, page_index=?,
                pagination_complete=?,
                updated_at=CURRENT_TIMESTAMP WHERE story_id=?
            """,
            (forum_id, expected_count, next_cursor, page_index, int(pagination_complete), story_id),
        )
        self.connection.commit()

    def mark_terminal(
        self,
        story_id: str,
        status: str,
        finished_at: str,
        *,
        forum_id: str | None = None,
        expected_count: int | None = None,
        observed_count: int | None = None,
        error_category: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal status: {status}")
        self.connection.execute(
            """
            UPDATE stories SET status=?, forum_id=COALESCE(?, forum_id),
                expected_count=COALESCE(?, expected_count), observed_count=?,
                next_cursor=NULL,
                pagination_complete=CASE
                    WHEN ? IN ('completed', 'completed_with_count_discrepancy') THEN 1
                    ELSE pagination_complete
                END,
                error_category=?, error_message=?,
                finished_at=?, updated_at=CURRENT_TIMESTAMP WHERE story_id=?
            """,
            (
                status,
                forum_id,
                expected_count,
                observed_count,
                status,
                error_category,
                (error_message or "")[:2000] or None,
                finished_at,
                story_id,
            ),
        )
        self.connection.commit()

    def reset_progress(self, story_id: str) -> None:
        self.connection.execute(
            """
            UPDATE stories SET forum_id=NULL, expected_count=NULL, observed_count=NULL,
                next_cursor=NULL, page_index=0, pagination_complete=0,
                updated_at=CURRENT_TIMESTAMP
            WHERE story_id=?
            """,
            (story_id,),
        )
        self.connection.commit()

    def rows(self, year: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM stories WHERE year=? ORDER BY month, story_id", (year,)
            )
        ]

    def story(self, story_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM stories WHERE story_id=?", (story_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def status_counts(self, year: int) -> dict[str, int]:
        return {
            row["status"]: row["count"]
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS count FROM stories WHERE year=? GROUP BY status",
                (year,),
            )
        }
