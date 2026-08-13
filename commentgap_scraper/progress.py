from __future__ import annotations

from collections.abc import Mapping


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, round(seconds))
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def crawl_progress_line(
    completed: int,
    total: int,
    elapsed_seconds: float,
    status_counts: Mapping[str, int],
) -> str:
    fraction = completed / total if total else 1.0
    rate_per_hour = completed / elapsed_seconds * 3_600 if elapsed_seconds > 0 else 0.0
    eta = (
        elapsed_seconds / completed * (total - completed)
        if completed > 0 and completed < total
        else 0.0 if completed >= total else None
    )
    statuses = ", ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items())
    ) or "none"
    return (
        f"crawl progress: {completed:,}/{total:,} ({fraction:.1%}) | "
        f"elapsed {format_duration(elapsed_seconds)} | ETA {format_duration(eta)} | "
        f"{rate_per_hour:.1f} stories/hour | {statuses}"
    )


def forum_progress_line(
    story_id: str,
    pages: int,
    staged_records: int,
    reported_postings: int,
    *,
    complete: bool,
) -> str:
    fraction = (
        min(staged_records, reported_postings) / reported_postings
        if reported_postings
        else 1.0
    )
    state = "complete" if complete else "paging"
    return (
        f"forum progress: story {story_id} | {pages:,} pages | "
        f"{staged_records:,}/{reported_postings:,} records ({fraction:.1%}) | {state}"
    )
