"""Display labels for the legacy news-category vocabulary.

The stored section values are deliberately kept unchanged.  This module is
only for tables, figures, and other human-facing output.  Unknown categories
fall back to their stored value so that newly introduced sections are visible
until they receive an explicit translation.
"""

from __future__ import annotations

from typing import Any


# This is the explicit equivalent of the positional ``genre1_eng`` mapping in
# legacy/02B_exploratory-analysis.Rmd.  Keys are normalised below, so both
# ``/wissenschaft`` and ``wissenschaft`` are accepted.
LEGACY_NEWS_CATEGORY_LABELS: dict[str, str] = {
    "international": "international",
    "sport": "sports",
    "web": "digital",
    "kultur": "culture",
    "etat": "media industry",
    "panorama": "human interest",
    "inland": "domestic",
    "wirtschaft": "economy",
    "lifestyle": "lifestyle",
    "diskurs": "opinion",
    "wissenschaft": "science",
    "video": "video",
    "podcast": "podcast",
    "recht": "legal",
    "diestandard": "women's policy",
}


def _normalise_category(value: Any) -> str:
    return str(value).strip().lstrip("/").lower()


def translate_news_category(value: Any) -> Any:
    """Translate one legacy category, preserving missing and unknown values."""

    if value is None:
        return value
    try:
        if value != value:  # NaN-like values
            return value
    except Exception:
        pass
    return LEGACY_NEWS_CATEGORY_LABELS.get(_normalise_category(value), value)


__all__ = ["LEGACY_NEWS_CATEGORY_LABELS", "translate_news_category"]
