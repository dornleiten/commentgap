from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api import ForumApi, context_uri
from .api import GRAPHQL_ENDPOINT
from .config import ScrapeConfig
from .http import HttpClient, HttpFailure
from .manifest import Manifest
from .parsing import DiscoveredStory, parse_article, parse_sitemap
from .privacy import AuthorPseudonymizer
from .progress import crawl_progress_line, forum_progress_line
from .storage import ParquetStore
from .transform import flatten_postings


LOG = logging.getLogger("commentgap_scraper")
SITEMAP_TEMPLATE = "https://www.derstandard.at/sitemaps/sitemap-{year}-{month:02d}.xml"


class CountMismatch(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_http(config: ScrapeConfig) -> HttpClient:
    return HttpClient(
        user_agent=config.identifying_user_agent,
        interval_seconds=config.request_interval,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def discover(config: ScrapeConfig, *, http: HttpClient | None = None) -> dict[str, int]:
    client = http or build_http(config)
    by_id: dict[str, DiscoveredStory] = {}
    monthly_counts: dict[str, int] = {}
    for month in range(1, 13):
        url = SITEMAP_TEMPLATE.format(year=config.year, month=month)
        LOG.info("reading sitemap %s", url)
        stories = parse_sitemap(client.get_text(url), config.year, month)
        monthly_counts[f"{month:02d}"] = len(stories)
        for story in stories:
            by_id[story.story_id] = story
        LOG.info(
            "sitemap progress: %d/12 (%.1f%%) | month=%02d | month stories=%s | unique stories=%s",
            month,
            month / 12 * 100,
            month,
            f"{len(stories):,}",
            f"{len(by_id):,}",
        )

    with Manifest(config.manifest_path) as manifest:
        manifest.upsert_discovered(by_id.values())
        ParquetStore(config.output_dir).export_manifest(manifest.rows(config.year), config.year)
    summary = {"unique_stories": len(by_id), **monthly_counts}
    LOG.info("discovered %d unique stories", len(by_id))
    return summary


def _article_record(
    story: dict[str, Any], html: str, collected_at: str
) -> dict[str, Any]:
    record = parse_article(
        html, story["url"], story["story_id"], story.get("sitemap_lastmod")
    )
    record.update(
        {
            "year": int(story["year"]),
            "month": int(story["month"]),
            "collected_at": collected_at,
        }
    )
    return record


def _forum_record(
    story: dict[str, Any],
    info: dict[str, Any],
    observed_unique: int,
    reconciled: int,
    collected_at: str,
) -> dict[str, Any]:
    return {
        "story_id": story["story_id"],
        "year": int(story["year"]),
        "month": int(story["month"]),
        "forum_id": str(info["id"]),
        "flags_json": json.dumps(
            sorted(str(flag) for flag in (info.get("flags") or [])),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "metadata_json": json.dumps(
            info.get("metadata") or [], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "reported_posting_count": int(info.get("totalPostingCount") or 0),
        "observed_unique_count": observed_unique,
        "reconciled_posting_count": reconciled,
        "collected_at": collected_at,
    }


def _resume_is_consistent(store: ParquetStore, story: dict[str, Any], forum_id: str) -> bool:
    if story.get("forum_id") != forum_id or int(story.get("page_index") or 0) < 1:
        return False
    pages = store.staging_pages(story["story_id"])
    # page_index is the next page number; page zero contains sticky records.
    committed_count = int(story["page_index"])
    expected_names = {f"page-{index:06d}.parquet" for index in range(committed_count)}
    present_names = {page.name for page in pages[:committed_count]}
    if present_names != expected_names:
        return False
    if len(pages) > committed_count:
        # A crash can occur after an atomic page write but before the SQLite commit.
        # Repeat only that uncommitted request, not all previously checkpointed pages.
        store.prune_staging_pages(story["story_id"], committed_count)
    return True


def crawl_story(
    story: dict[str, Any],
    *,
    config: ScrapeConfig,
    http: HttpClient,
    api: ForumApi,
    manifest: Manifest,
    store: ParquetStore,
    pseudonymizer: AuthorPseudonymizer,
) -> str:
    story_id = story["story_id"]
    collected_at = utc_now()
    manifest.mark_in_progress(story_id, collected_at)

    try:
        html = http.get_text(story["url"])
        store.write_article(_article_record(story, html, collected_at))

        info = api.get_forum_info(context_uri(story_id))
        if not info or not info.get("id"):
            store.reset_staging(story_id)
            manifest.mark_terminal(story_id, "no_forum", utc_now(), observed_count=0)
            return "no_forum"

        forum_id = str(info["id"])
        expected = int(info.get("totalPostingCount") or 0)
        if expected == 0:
            store.reset_staging(story_id)
            store.write_forum(_forum_record(story, info, 0, 0, collected_at))
            manifest.mark_terminal(
                story_id,
                "no_postings",
                utc_now(),
                forum_id=forum_id,
                expected_count=0,
                observed_count=0,
            )
            return "no_postings"

        resume = _resume_is_consistent(store, story, forum_id)
        if resume:
            cursor = story.get("next_cursor")
            page_index = int(story["page_index"])
            pagination_complete = bool(story.get("pagination_complete"))
            staged_record_count = store.staging_record_count(story_id)
            LOG.info("resuming story %s at page %d", story_id, page_index)
        else:
            manifest.reset_progress(story_id)
            store.reset_staging(story_id)
            sticky_records = flatten_postings(
                info.get("stickyPostings") or [],
                story_id=story_id,
                forum_id=forum_id,
                pseudonymizer=pseudonymizer,
                collected_at=collected_at,
                page_index=0,
                is_sticky=True,
            )
            store.write_comment_page(story_id, 0, sticky_records)
            staged_record_count = len(sticky_records)
            cursor = None
            page_index = 1
            pagination_complete = False
            manifest.save_progress(
                story_id,
                forum_id=forum_id,
                expected_count=expected,
                next_cursor=cursor,
                page_index=page_index,
            )

        while not pagination_complete:
            page = api.get_threads_page(forum_id, cursor)
            edges = page.get("edges") or []
            roots = [edge.get("node") for edge in edges if isinstance(edge, dict) and edge.get("node")]
            records = flatten_postings(
                roots,
                story_id=story_id,
                forum_id=forum_id,
                pseudonymizer=pseudonymizer,
                collected_at=utc_now(),
                page_index=page_index,
            )
            store.write_comment_page(story_id, page_index, records)
            staged_record_count += len(records)
            page_info = page.get("pageInfo") or {}
            has_next = bool(page_info.get("hasNextPage"))
            next_cursor = page_info.get("nextCursor")
            if has_next and (not next_cursor or next_cursor == cursor):
                raise RuntimeError(f"pagination cursor did not advance for story {story_id}")
            cursor = next_cursor if has_next else None
            page_index += 1
            pagination_complete = not has_next
            manifest.save_progress(
                story_id,
                forum_id=forum_id,
                expected_count=expected,
                next_cursor=cursor,
                page_index=page_index,
                pagination_complete=pagination_complete,
            )
            completed_pages = page_index - 1
            if (
                pagination_complete
                or completed_pages % config.progress_every_pages == 0
            ):
                LOG.info(
                    "%s",
                    forum_progress_line(
                        story_id,
                        completed_pages,
                        staged_record_count,
                        expected,
                        complete=pagination_complete,
                    ),
                )

        _, observed_unique, reconciled = store.finalize_comments(
            story_id, int(story["year"]), int(story["month"])
        )
        if reconciled != expected:
            refreshed = api.get_forum_info(context_uri(story_id)) or info
            refreshed_expected = int(refreshed.get("totalPostingCount") or 0)
            info = refreshed
            expected = refreshed_expected
        if reconciled != expected:
            raise CountMismatch(
                f"story {story_id}: API reports {expected} postings but {reconciled} published records "
                f"({observed_unique} including deletion tombstones) were collected"
            )

        store.write_forum(
            _forum_record(story, info, observed_unique, reconciled, collected_at)
        )
        manifest.mark_terminal(
            story_id,
            "completed",
            utc_now(),
            forum_id=forum_id,
            expected_count=expected,
            observed_count=reconciled,
        )
        return "completed"
    except HttpFailure as exc:
        status = "inaccessible" if exc.status in {401, 403, 404, 410} else "failed"
        manifest.mark_terminal(
            story_id,
            status,
            utc_now(),
            error_category="http",
            error_message=str(exc),
        )
        LOG.error("story %s: %s", story_id, exc)
        return status
    except Exception as exc:
        manifest.mark_terminal(
            story_id,
            "failed",
            utc_now(),
            error_category=type(exc).__name__,
            error_message=str(exc),
        )
        LOG.exception("story %s failed", story_id)
        return "failed"


def crawl(
    config: ScrapeConfig,
    *,
    hash_key: str,
    limit: int | None = None,
    retry_failed: bool = False,
    story_ids: list[str] | None = None,
    monthly_round_robin: bool = False,
    http: HttpClient | None = None,
) -> dict[str, int]:
    client = http or build_http(config)
    api = ForumApi(client, reply_depth=config.reply_query_depth)
    pseudonymizer = AuthorPseudonymizer(hash_key)
    store = ParquetStore(config.output_dir)
    metadata_path = config.output_dir / "collection_metadata.json"
    existing_metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOG.warning("replacing unreadable collection metadata at %s", metadata_path)
    metadata_updated_at = utc_now()
    metadata = {
        "schema_version": 1,
        "year": config.year,
        "scope": "All comments visible at collection time on articles published in the selected year",
        "snapshot_warning": (
            "Votes, deletions, sticky state, and author follower counts reflect collection time, "
            "not the end of the publication year."
        ),
        "forum_endpoint": GRAPHQL_ENDPOINT,
        "reply_query_depth": config.reply_query_depth,
        "minimum_request_interval_seconds": config.request_interval,
        "collection_started_at": existing_metadata.get("collection_started_at")
        or metadata_updated_at,
        "metadata_updated_at": metadata_updated_at,
    }
    metadata_temp = metadata_path.with_suffix(".json.tmp")
    metadata_temp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(metadata_temp, metadata_path)
    results: dict[str, int] = {}
    crawl_started = time.monotonic()
    with Manifest(config.manifest_path) as manifest:
        stories = manifest.stories_for_crawl(
            config.year,
            retry_failed=retry_failed,
            limit=limit,
            story_ids=story_ids,
            monthly_round_robin=monthly_round_robin,
        )
        if not stories:
            LOG.warning("no pending stories; run discover first or pass --retry-failed")
        elif stories:
            LOG.info("%s", crawl_progress_line(0, len(stories), 0.0, results))
        for index, story in enumerate(stories, start=1):
            LOG.info("[%d/%d] story %s", index, len(stories), story["story_id"])
            status = crawl_story(
                story,
                config=config,
                http=client,
                api=api,
                manifest=manifest,
                store=store,
                pseudonymizer=pseudonymizer,
            )
            current = manifest.story(story["story_id"])
            if (
                status == "failed"
                and current is not None
                and current.get("error_category") == "CountMismatch"
            ):
                LOG.warning(
                    "story %s had a count mismatch; starting one clean reconciliation pass",
                    story["story_id"],
                )
                status = crawl_story(
                    current,
                    config=config,
                    http=client,
                    api=api,
                    manifest=manifest,
                    store=store,
                    pseudonymizer=pseudonymizer,
                )
            results[status] = results.get(status, 0) + 1
            LOG.info(
                "%s",
                crawl_progress_line(
                    index,
                    len(stories),
                    time.monotonic() - crawl_started,
                    results,
                ),
            )
        rows = manifest.rows(config.year)
        if rows:
            store.export_manifest(rows, config.year)
    return results
