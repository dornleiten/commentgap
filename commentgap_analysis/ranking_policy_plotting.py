"""Plotting helpers for ranking-policy similarity analyses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .presentation_labels import REPLY_DISPLAY_MARKERS


def plot_policy_space(
    axis: Any,
    frame: pd.DataFrame,
    x: str,
    y: str,
    colour_column: str,
    colour_order: list[str],
    colour_palette: Mapping[str, str],
    *,
    reply_markers: Mapping[str, str] | None = None,
) -> None:
    """Plot policy-space points by ordering, reply mode, and pin state."""
    markers = REPLY_DISPLAY_MARKERS if reply_markers is None else reply_markers
    for ordering in colour_order:
        for reply_mode, marker in markers.items():
            for pinned in (False, True):
                panel = frame[
                    frame[colour_column].eq(ordering)
                    & frame["reply_mode"].eq(reply_mode)
                    & frame["pinned"].eq(pinned)
                ]
                if panel.empty:
                    continue
                colour = colour_palette[ordering]
                axis.scatter(
                    panel[x], panel[y], marker=marker, s=80, alpha=.85,
                    facecolors=colour if pinned else "none",
                    edgecolors=colour, linewidths=2,
                )
