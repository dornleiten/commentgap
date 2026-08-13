from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .privacy import AuthorPseudonymizer


def _json_list(value: Any) -> str:
    items = value if isinstance(value, list) else []
    return json.dumps(sorted(str(item) for item in items), ensure_ascii=False, separators=(",", ":"))


def _reaction_value(posting: dict[str, Any], name: str) -> int:
    aggregated = ((posting.get("reactions") or {}).get("aggregated") or [])
    for reaction in aggregated:
        if str(reaction.get("name", "")).lower() == name:
            try:
                return max(0, int(reaction.get("value") or 0))
            except (TypeError, ValueError):
                return 0
    return 0


def effective_text(title: Any, text: Any) -> str | None:
    """Combine non-empty title and body with a line boundary for analysis."""
    parts = [str(value).strip() for value in (title, text) if value is not None and str(value).strip()]
    return "\n".join(parts) or None


def flatten_postings(
    roots: Iterable[dict[str, Any]],
    *,
    story_id: str,
    forum_id: str,
    pseudonymizer: AuthorPseudonymizer,
    collected_at: str,
    page_index: int,
    is_sticky: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    preorder = 0
    page_order_base = page_index * 1_000_000

    def visit(
        posting: dict[str, Any],
        *,
        parent_id: str | None,
        root_id: str,
        depth: int,
        root_order: int,
    ) -> None:
        nonlocal preorder
        comment_id = str(posting.get("id") or "")
        if not comment_id:
            return
        replies = [reply for reply in (posting.get("replies") or []) if isinstance(reply, dict)]
        author = posting.get("author") or {}
        legacy = posting.get("legacy") or {}
        title = posting.get("title")
        text = posting.get("text")
        records.append(
            {
                "comment_id": comment_id,
                "legacy_posting_id": str(legacy.get("postingId")) if legacy.get("postingId") is not None else None,
                "story_id": story_id,
                "forum_id": forum_id,
                "parent_comment_id": parent_id,
                "root_comment_id": root_id,
                "depth": depth,
                "root_order": root_order,
                "preorder_position": page_order_base + preorder,
                "display_order": None,
                "is_root": parent_id is None,
                "is_leaf": not replies,
                "created_at": ((posting.get("history") or {}).get("created")),
                "title": title,
                "text": text,
                "effective_text": effective_text(title, text),
                "lifecycle_status": posting.get("lifecycleStatus"),
                "flags_json": _json_list(posting.get("flags")),
                # The sticky API record is a root thread. Replies are ordinary
                # comments and must not inherit the root's pinned state.
                "is_sticky": is_sticky and depth == 0,
                "votes_positive": _reaction_value(posting, "positive"),
                "votes_negative": _reaction_value(posting, "negative"),
                "author_hash": pseudonymizer.pseudonymize(posting),
                "author_follower_count": max(0, int(author.get("followerCount") or 0)),
                "collected_at": collected_at,
            }
        )
        preorder += 1
        for reply in replies:
            visit(
                reply,
                parent_id=comment_id,
                root_id=root_id,
                depth=depth + 1,
                root_order=root_order,
            )

    for root_index, root in enumerate(roots):
        root_id = str(root.get("id") or "")
        if root_id:
            visit(
                root,
                parent_id=None,
                root_id=root_id,
                depth=0,
                root_order=page_index * 100_000 + root_index,
            )
    return records


def merge_comment_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate sticky copies and assign a contiguous visible order."""
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        comment_id = record["comment_id"]
        existing = by_id.get(comment_id)
        if existing is None:
            by_id[comment_id] = dict(record)
            continue
        sticky = bool(existing.get("is_sticky")) or bool(record.get("is_sticky"))
        # Page zero is the sticky-thread copy. Prefer every regular thread copy,
        # including descendants which correctly have is_sticky=False in both trees.
        existing_is_sticky_page = int(existing.get("root_order") or 0) < 100_000
        record_is_regular_page = int(record.get("root_order") or 0) >= 100_000
        if existing_is_sticky_page and record_is_regular_page:
            by_id[comment_id] = dict(record)
        by_id[comment_id]["is_sticky"] = sticky

    ordered = sorted(
        by_id.values(),
        key=lambda row: (
            row.get("root_order") if row.get("root_order") is not None else 2**63 - 1,
            row.get("preorder_position") if row.get("preorder_position") is not None else 2**63 - 1,
            row["comment_id"],
        ),
    )
    for index, row in enumerate(ordered, start=1):
        row["display_order"] = index
    return ordered
