"""Summary correlations used by the FORUM analysis notebook."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def correlation_row(
    frame: pd.DataFrame, x: str, y: str, group_name: str = "overall",
) -> dict[str, object]:
    """Return Pearson and Spearman correlations for one pair of columns."""
    pair = frame[[x, y]].dropna()
    valid = len(pair) >= 3 and pair[x].nunique() >= 2 and pair[y].nunique() >= 2
    return {
        "group": group_name,
        "n": len(pair),
        "pearson_r": pair[x].corr(pair[y]) if valid else np.nan,
        "spearman_rho": pair[x].corr(pair[y], method="spearman") if valid else np.nan,
    }


def correlations_by(
    frame: pd.DataFrame, x: str, y: str, group_cols: Sequence[str],
) -> pd.DataFrame:
    """Return correlations for each combination of grouping columns."""
    grouping = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    rows = [
        correlation_row(
            group, x, y,
            " | ".join(map(str, key if isinstance(key, tuple) else (key,))),
        )
        for key, group in frame.groupby(grouping, dropna=False)
    ]
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def calculate_forum_ndcg_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate FORUM/nDCG Pearson correlation for each depth and feature."""
    rows = [
        {
            'depth': depth,
            'feature_label': feature_label,
            'pearson_r': group['forum'].corr(group['ndcg']),
        }
        for (depth, feature_label), group in frame.groupby(['depth', 'feature_label'])
    ]
    return pd.DataFrame(rows)
