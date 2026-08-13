from __future__ import annotations

import json
import hashlib
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
from .progress import crawl_progress_line, format_duration, forum_progress_line
from .storage import ParquetStore
from .transform import flatten_postings


LOG = logging.getLogger("commentgap_scraper")
SITEMAP_TEMPLATE = "https://www.derstandard.at/sitemaps/sitemap-{year}-{month:02d}.xml"
PILOT_SIZE_BANDS = (
    "no_forum",
    "no_postings",
    "1-49",
    "50-249",
    "250-999",
    "1000+",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cursor_hash(cursor: str | None) -> str | None:
    if not cursor:
        return None
    return hashlib.sha256(cursor.encode("utf-8")).hexdigest()


def _forum_size_band(info: dict[str, Any] | None) -> str:
    if not info or not info.get("id"):
        return "no_forum"
    count = int(info.get("totalPostingCount") or 0)
    if count == 0:
        return "no_postings"
    if count < 50:
        return "1-49"
    if count < 250:
        return "50-249"
    if count < 1000:
        return "250-999"
    return "1000+"


def select_stratified_pilot(
    candidates: list[dict[str, Any]],
    *,
    api: ForumApi,
    limit: int,
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Balance a reproducible candidate pool across month and forum-size bands."""
    assessed: list[dict[str, Any]] = []
    for index, story in enumerate(candidates, start=1):
        try:
            info = api.get_forum_info(context_uri(story["story_id"]))
        except HttpFailure as exc:
            LOG.warning("pilot preflight skipped story %s: %s", story["story_id"], exc)
            continue
        band = _forum_size_band(info)
        assessed.append(
            {
                "story": story,
                "story_id": story["story_id"],
                "month": int(story["month"]),
                "size_band": band,
                "reported_posting_count": (
                    int(info.get("totalPostingCount") or 0) if info else None
                ),
            }
        )
        if index == 1 or index % 25 == 0 or index == len(candidates):
            LOG.info(
                "pilot preflight: %d/%d (%.1f%%) forum sizes assessed",
                index,
                len(candidates),
                index / len(candidates) * 100,
            )

    month_targets = {
        month: sum(index % 12 + 1 == month for index in range(limit))
        for month in range(1, 13)
    }
    band_targets = {
        band: sum(PILOT_SIZE_BANDS[index % len(PILOT_SIZE_BANDS)] == band for index in range(limit))
        for band in PILOT_SIZE_BANDS
    }
    month_counts = {month: 0 for month in month_targets}
    band_counts = {band: 0 for band in band_targets}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        selected.append(item)
        selected_ids.add(item["story_id"])
        month_counts[item["month"]] += 1
        band_counts[item["size_band"]] += 1

    # First fill both dimensions; then fill any remaining monthly slots even
    # where rare bands were unavailable in the bounded pool.
    for item in assessed:
        if len(selected) >= limit:
            break
        if (
            month_counts[item["month"]] < month_targets[item["month"]]
            and band_counts[item["size_band"]] < band_targets[item["size_band"]]
        ):
            add(item)
    for item in assessed:
        if len(selected) >= limit:
            break
        if item["story_id"] not in selected_ids and (
            month_counts[item["month"]] < month_targets[item["month"]]
        ):
            add(item)
    for item in assessed:
        if len(selected) >= limit:
            break
        if item["story_id"] not in selected_ids:
            add(item)

    audit = {
        "generated_at": utc_now(),
        "selection": "stratified-pilot",
        "seed": seed,
        "candidate_pool_size": len(candidates),
        "requested_limit": limit,
        "selected_count": len(selected),
        "month_counts": month_counts,
        "size_band_counts": band_counts,
        "size_band_targets": band_targets,
        "stories": [
            {key: item[key] for key in ("story_id", "month", "size_band", "reported_posting_count")}
            for item in selected
        ],
    }
    path = output_dir / "pilot_selection.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return [item["story"] for item in selected]


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
    observed_published: int,
    observed_deleted: int,
    collected_at: str,
    *,
    crawl_status: str,
    discrepancy_reproduced: bool | None = None,
    page_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reported = int(info.get("totalPostingCount") or 0)
    difference = observed_published - reported
    diagnostics = page_diagnostics or []
    thread_pages = [row for row in diagnostics if row.get("page_kind") == "threads"]
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
        "reported_posting_count": reported,
        "observed_unique_count": observed_unique,
        "observed_published_count": observed_published,
        "observed_deleted_count": observed_deleted,
        "reconciled_posting_count": observed_published,
        "posting_count_difference": difference,
        "posting_count_discrepancy_absolute": abs(difference),
        "posting_count_discrepancy_pct": (
            abs(difference) / reported * 100.0 if reported else 0.0
        ),
        "count_discrepancy_reproduced": discrepancy_reproduced,
        "pagination_page_count": len(thread_pages),
        "root_edge_count": sum(int(row.get("root_edge_count") or 0) for row in thread_pages),
        "flattened_record_count": sum(
            int(row.get("flattened_record_count") or 0) for row in diagnostics
        ),
        "cursor_walk_complete": bool(thread_pages) and not bool(thread_pages[-1].get("has_next_page")),
        "cursor_progression_valid": all(
            bool(row.get("cursor_progression_valid")) for row in thread_pages
        ),
        "crawl_status": crawl_status,
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
    if story.get("status") in {"failed", "in_progress"}:
        # Older versions exposed mismatch rows before failure, and a process can
        # stop between file publication and the final manifest transaction.
        store.remove_published_outputs(
            story_id, int(story["year"]), int(story["month"])
        )

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
            store.write_forum(
                _forum_record(
                    story,
                    info,
                    0,
                    0,
                    0,
                    collected_at,
                    crawl_status="no_postings",
                    page_diagnostics=[],
                )
            )
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
            store.write_page_diagnostic(
                story_id,
                {
                    "story_id": story_id,
                    "forum_id": forum_id,
                    "year": int(story["year"]),
                    "month": int(story["month"]),
                    "page_index": 0,
                    "page_kind": "sticky",
                    "request_cursor_hash": None,
                    "next_cursor_hash": None,
                    "root_edge_count": len(info.get("stickyPostings") or []),
                    "flattened_record_count": len(sticky_records),
                    "has_next_page": None,
                    "cursor_progression_valid": True,
                    "collected_at": collected_at,
                },
            )
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
            request_cursor = cursor
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
            store.write_page_diagnostic(
                story_id,
                {
                    "story_id": story_id,
                    "forum_id": forum_id,
                    "year": int(story["year"]),
                    "month": int(story["month"]),
                    "page_index": page_index,
                    "page_kind": "threads",
                    "request_cursor_hash": _cursor_hash(request_cursor),
                    "next_cursor_hash": _cursor_hash(next_cursor),
                    "root_edge_count": len(edges),
                    "flattened_record_count": len(records),
                    "has_next_page": has_next,
                    "cursor_progression_valid": (
                        not has_next or bool(next_cursor and next_cursor != request_cursor)
                    ),
                    "collected_at": utc_now(),
                },
            )
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

        _, observed_unique, observed_published, observed_deleted = store.prepare_comments(
            story_id, int(story["year"]), int(story["month"])
        )
        if observed_published != expected:
            refreshed = api.get_forum_info(context_uri(story_id), refresh=True) or info
            refreshed_expected = int(refreshed.get("totalPostingCount") or 0)
            info = refreshed
            expected = refreshed_expected
        status = (
            "completed_with_count_discrepancy"
            if observed_published != expected
            else "completed"
        )
        if status == "completed_with_count_discrepancy":
            LOG.warning(
                "story %s completed with count discrepancy: API reports %d postings "
                "but %d published records (%d including %d deletion tombstones) "
                "were collected; retaining the complete cursor walk without recrawling",
                story_id,
                expected,
                observed_published,
                observed_unique,
                observed_deleted,
            )
        page_diagnostics = store.staging_diagnostics(story_id)
        store.publish_prepared_comments(
            story_id, int(story["year"]), int(story["month"])
        )
        store.write_forum(
            _forum_record(
                story,
                info,
                observed_unique,
                observed_published,
                observed_deleted,
                collected_at,
                crawl_status=status,
                # Full reconciliation recrawls were retired in scraper 0.3.2.
                # Null means the internally valid first walk was accepted
                # without repeating its article and comment-page requests.
                discrepancy_reproduced=None,
                page_diagnostics=page_diagnostics,
            )
        )
        manifest.mark_terminal(
            story_id,
            status,
            utc_now(),
            forum_id=forum_id,
            expected_count=expected,
            observed_count=observed_published,
        )
        return status
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
    only_failed: bool = False,
    story_ids: list[str] | None = None,
    monthly_random: bool = False,
    selection_seed: int = 2025,
    stratified_pilot: bool = False,
    pilot_candidate_pool: int = 500,
    http: HttpClient | None = None,
) -> dict[str, int]:
    if stratified_pilot and limit is None:
        raise ValueError("stratified-pilot selection requires --limit")
    if stratified_pilot and pilot_candidate_pool < limit:
        raise ValueError("--pilot-candidate-pool must be at least --limit")
    if only_failed and stratified_pilot:
        raise ValueError("--only-failed cannot be combined with stratified-pilot selection")
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
        **existing_metadata,
        "schema_version": 3,
        "year": config.year,
        "scope": "All comments visible at collection time on articles published in the selected year",
        "snapshot_warning": (
            "Votes, deletions, sticky state, and author follower counts reflect collection time, "
            "not the end of the publication year."
        ),
        "forum_endpoint": GRAPHQL_ENDPOINT,
        "reply_query_depth": config.reply_query_depth,
        "minimum_request_interval_seconds": config.request_interval,
        "count_discrepancy_policy": (
            "refresh forum counter, then retain a complete cursor walk without recrawling"
        ),
        "collection_started_at": existing_metadata.get("collection_started_at")
        or metadata_updated_at,
        "metadata_updated_at": metadata_updated_at,
        "last_selection": (
            "only-failed"
            if only_failed
            else "stratified-pilot"
            if stratified_pilot
            else "monthly-random" if monthly_random else "chronological"
        ),
        "last_selection_seed": selection_seed if (monthly_random or stratified_pilot) else None,
    }
    metadata_temp = metadata_path.with_suffix(".json.tmp")
    metadata_temp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(metadata_temp, metadata_path)
    results: dict[str, int] = {}
    final_status_counts: dict[str, int] = {}
    with Manifest(config.manifest_path) as manifest:
        requested_limit = limit
        candidate_limit = pilot_candidate_pool if stratified_pilot else limit
        stories = manifest.stories_for_crawl(
            config.year,
            retry_failed=retry_failed,
            only_failed=only_failed,
            limit=candidate_limit,
            story_ids=story_ids,
            monthly_random=monthly_random or stratified_pilot,
            selection_seed=selection_seed,
        )
        if stratified_pilot:
            preflight_started = time.monotonic()
            stories = select_stratified_pilot(
                stories,
                api=api,
                limit=requested_limit,
                seed=selection_seed,
                output_dir=config.output_dir,
            )
            LOG.info(
                "pilot preflight complete: %d stories selected | elapsed %s",
                len(stories),
                format_duration(time.monotonic() - preflight_started),
            )
        # Selection and forum-size preflight are setup work. Start the crawl
        # clock only after they finish so their fixed cost is not projected
        # across every remaining story by the rate and ETA calculations.
        crawl_started = time.monotonic()
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
        final_status_counts = manifest.status_counts(config.year)
        if rows:
            store.export_manifest(rows, config.year)
    metadata["last_crawl_finished_at"] = utc_now()
    metadata["last_crawl_results"] = results
    metadata["status_counts_after_last_crawl"] = final_status_counts
    unfinished = sum(
        count
        for status, count in final_status_counts.items()
        if status in {"pending", "in_progress"}
    )
    if unfinished == 0 and final_status_counts.get("failed", 0) == 0:
        metadata["collection_completed_at"] = metadata["last_crawl_finished_at"]
    metadata_temp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(metadata_temp, metadata_path)
    return results
