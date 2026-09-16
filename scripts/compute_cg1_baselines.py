#!/usr/bin/env python3
"""Compute deterministic ranking baselines for the CG1 held-out table.

The baselines rank every comment in each held-out discussion using a fixed
ordering and are evaluated with the same complete-discussion nDCG and
balanced macro-F1 utilities as the fitted models.  The audience target is
``audience_selected_draw_01``, the first deterministic vote-cutoff draw used
by the existing held-out model table.  Equal baseline keys are resolved by
ascending ``comment_id``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from commentgap_analysis.paper1_reporting import (
    _bootstrap_group_mean,
    balanced_macro_f1_at_k,
)
from commentgap_analysis.ranking import bootstrap_metric_summary, evaluate_rank_scores


BASELINES = {
    "Earliest-first": ("hours_since_article",),
    "Roots-first/earliest-first": ("is_reply", "hours_since_article"),
    "Length-based": ("word_count",),
}


def _ranked_scores(group: pd.DataFrame, label: str) -> np.ndarray:
    """Return scores whose descending order is the named baseline order."""
    if label == "Earliest-first":
        ordered = group.sort_values(
            ["hours_since_article", "comment_id"],
            ascending=[True, True],
            kind="mergesort",
        )
    elif label == "Roots-first/earliest-first":
        ordered = group.sort_values(
            ["is_reply", "hours_since_article", "comment_id"],
            ascending=[True, True, True],
            kind="mergesort",
        )
    elif label == "Length-based":
        ordered = group.sort_values(
            ["word_count", "comment_id"],
            ascending=[False, True],
            kind="mergesort",
        )
    else:
        raise ValueError(f"Unknown baseline: {label}")
    positions = pd.Series(-np.arange(len(ordered), dtype=float), index=ordered.index)
    return positions.reindex(group.index).to_numpy()


def _score_rows(
    choice: pd.DataFrame, label: str, selector: str
) -> tuple[pd.DataFrame, np.ndarray]:
    selected_column = (
        "audience_selected_draw_01" if selector == "audience" else "curator_selected"
    )
    rows = []
    scores = []
    for story_id, group in choice.groupby("story_id", sort=False):
        group = group.copy()
        group["selector"] = selector
        group["query_id"] = group["story_id"].astype(str) + ":" + selector
        group["selected"] = group[selected_column].astype(int)
        rows.append(
            group[["story_id", "comment_id", "selector", "query_id", "n_picks", "selected"]]
        )
        scores.append(_ranked_scores(group, label))
    frame = pd.concat(rows, ignore_index=True)
    score_array = np.concatenate(scores)
    frame["score"] = score_array
    return frame, score_array


def compute_baselines(
    choice_path: Path,
    split_path: Path,
    *,
    draws: int = 1_000,
    balance_draws: int = 100,
    seed: int = 20_260_813,
) -> tuple[pd.DataFrame, dict[str, object]]:
    choice = pd.read_parquet(choice_path)
    split = pd.read_parquet(split_path, columns=["story_id", "split_role"])
    split["story_id"] = split["story_id"].astype(str)
    held_out = set(split.loc[split["split_role"].eq("paper2_test"), "story_id"])
    choice["story_id"] = choice["story_id"].astype(str)
    choice = choice.loc[choice["story_id"].isin(held_out)].copy()
    if choice["story_id"].nunique() != 2_559:
        raise RuntimeError("CG1 baseline input does not contain 2,559 held-out discussions")
    if choice["hours_since_article"].isna().any():
        raise RuntimeError("Earliest-first baseline has missing posting times")
    if choice.duplicated(["story_id", "comment_id"]).any():
        raise RuntimeError("CG1 baseline input has duplicate comments")

    metric_frames = []
    balanced_frames = []
    for index, label in enumerate(BASELINES):
        selector_frames = []
        selector_scores = []
        for selector in ("audience", "curator"):
            frame, scores = _score_rows(choice, label, selector)
            selector_frames.append(frame)
            selector_scores.append(scores)
            evaluated = evaluate_rank_scores(frame, scores)
            evaluated["model"] = label
            metric_frames.append(evaluated)
        score_frame = pd.concat(selector_frames, ignore_index=True)
        balanced = balanced_macro_f1_at_k(
            score_frame,
            draws=balance_draws,
            seed=seed + 600_000 + index * 100_000,
        )
        balanced["model"] = label
        balanced_frames.append(balanced)

    article_metrics = pd.concat(metric_frames, ignore_index=True)
    ndcg_rows = []
    for label in BASELINES:
        summary = bootstrap_metric_summary(
            article_metrics.loc[article_metrics["model"].eq(label)],
            draws=draws,
            seed=seed,
        )
        summary = summary.loc[summary["metric"].eq("ndcg_at_k")].copy()
        summary["model"] = label
        summary["articles"] = 2_559
        summary["bootstrap_draws"] = draws
        summary["balance_draws"] = balance_draws
        ndcg_rows.append(summary)
    ndcg = pd.concat(ndcg_rows, ignore_index=True)

    balanced_articles = pd.concat(balanced_frames, ignore_index=True)
    balanced = _bootstrap_group_mean(
        balanced_articles,
        groups=["model", "selector"],
        value="balanced_macro_f1_at_k",
        draws=draws,
        seed=seed + 700_000,
        estimate_name="estimate",
    )
    balanced["metric"] = "balanced_macro_f1_at_k"
    balanced["articles"] = 2_559
    balanced["bootstrap_draws"] = draws
    balanced["balance_draws"] = balance_draws

    output = pd.concat(
        [
            ndcg[["model", "selector", "metric", "estimate", "conf_low", "conf_high", "articles", "bootstrap_draws", "balance_draws"]],
            balanced[["model", "selector", "metric", "estimate", "conf_low", "conf_high", "articles", "bootstrap_draws", "balance_draws"]],
        ],
        ignore_index=True,
    ).sort_values(["model", "selector", "metric"]).reset_index(drop=True)
    assumptions = {
        "held_out_discussions": 2_559,
        "held_out_comments": int(len(choice)),
        "audience_label": "audience_selected_draw_01 (first deterministic cutoff-tie draw)",
        "tie_break": "ascending comment_id after baseline keys",
        "earliest_first": "hours_since_article ascending",
        "roots_first_earliest_first": "is_reply ascending, then hours_since_article ascending",
        "length_based": "word_count descending",
        "ndcg": "complete discussion, equal-discussion mean, 1,000 article-bootstrap draws",
        "balanced_macro_f1": "all k selected comments plus k sampled non-selected comments, 100 draws per discussion-selector, then 1,000 article-bootstrap draws",
        "seed": seed,
    }
    return output, assumptions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--choice-set",
        type=Path,
        default=Path("model_output/selection_2025/model_data/choice_set_all.parquet"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("model_output/selection_2025/model_data/master_article_split.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_output/selection_2025/paper1/reporting/tables/cg1_baseline_performance.csv"),
    )
    parser.add_argument(
        "--assumptions-output",
        type=Path,
        default=Path("model_output/selection_2025/paper1/reporting/tables/cg1_baseline_assumptions.json"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=1_000)
    parser.add_argument("--balance-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_260_813)
    args = parser.parse_args()
    output, assumptions = compute_baselines(
        args.choice_set,
        args.split,
        draws=args.bootstrap_draws,
        balance_draws=args.balance_draws,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.assumptions_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.assumptions_output.write_text(json.dumps(assumptions, indent=2) + "\n")
    print(output.to_string(index=False))
    print(json.dumps(assumptions, sort_keys=True))


if __name__ == "__main__":
    main()
