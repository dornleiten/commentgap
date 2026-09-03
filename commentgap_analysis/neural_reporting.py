"""Extend the FORUM report with the three text-augmented rankers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _paired_differences(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    reference_model: str,
    candidate_model: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    keys = ["story_id", "selector"]
    left = reference[keys].sort_values(keys).reset_index(drop=True)
    right = candidate[keys].sort_values(keys).reset_index(drop=True)
    if left.duplicated().any() or right.duplicated().any() or not left.equals(right):
        raise ValueError(
            f"{reference_model} and {candidate_model} require identical unique test keys"
        )
    joined = reference.merge(candidate, on=keys, suffixes=("_reference", "_candidate"))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for selector, subset in joined.groupby("selector", sort=True):
        for metric in ("ndcg_at_k", "top_k_overlap", "jaccard"):
            difference = (
                subset[f"{metric}_candidate"] - subset[f"{metric}_reference"]
            ).to_numpy(dtype=float)
            bootstrap = difference[
                rng.integers(0, len(difference), size=(draws, len(difference)))
            ].mean(axis=1)
            rows.append(
                {
                    "reference_model": reference_model,
                    "candidate_model": candidate_model,
                    "selector": selector,
                    "metric": metric,
                    "estimate_candidate_minus_reference": float(difference.mean()),
                    "conf_low": float(np.quantile(bootstrap, 0.025)),
                    "conf_high": float(np.quantile(bootstrap, 0.975)),
                    "articles": len(difference),
                    "bootstrap_draws": draws,
                }
            )
    return pd.DataFrame(rows)


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".tex").write_text(
        frame.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}"),
        encoding="utf-8",
    )


def build_neural_reporting_outputs(
    xgb_root: Path,
    neural_roots: dict[str, Path],
    report_root: Path,
    *,
    bootstrap_draws: int = 1_000,
    seed: int = 20260813,
) -> dict[str, Any]:
    """Merge neural metrics into an already-built conditional-logit/XGBoost report."""
    xgb_root = Path(xgb_root)
    neural_roots = {str(name): Path(root) for name, root in neural_roots.items()}
    report_root = Path(report_root)
    table_root = report_root / "tables"
    figure_root = report_root / "figures"
    base_path = table_root / "model_test_performance.csv"
    tie_path = table_root / "model_test_tie_sensitivity.csv"
    if not base_path.exists() or not tie_path.exists():
        raise FileNotFoundError("Run build_reporting_outputs before neural reporting")
    performance_rows: list[pd.DataFrame] = []
    tie_rows: list[pd.DataFrame] = []
    difference_rows: list[pd.DataFrame] = []
    for scope_index, scope in enumerate(("root", "all")):
        reference = pd.read_parquet(xgb_root / scope / "test_article_metrics.parquet")
        for model_index, (model_name, root) in enumerate(neural_roots.items()):
            scope_root = root / scope
            manifest = json.loads((scope_root / "model_manifest.json").read_text())
            if manifest.get("reported_scores") != "sealed_paper2_test":
                raise ValueError(f"{model_name}/{scope} is not a sealed-test model")
            metrics = pd.read_parquet(scope_root / "test_metric_summary.parquet")
            metrics["scope"] = scope
            metrics["model"] = model_name
            performance_rows.append(metrics)
            ties = pd.read_parquet(scope_root / "test_tie_sensitivity_metrics.parquet")
            ties = ties.groupby(
                ["selector", "audience_tie_draw"], as_index=False
            )[["ndcg_at_k", "top_k_overlap", "jaccard"]].mean()
            ties["scope"] = scope
            ties["model"] = model_name
            tie_rows.append(ties)
            candidate = pd.read_parquet(scope_root / "test_article_metrics.parquet")
            difference = _paired_differences(
                reference,
                candidate,
                reference_model="XGBoost",
                candidate_model=model_name,
                draws=bootstrap_draws,
                seed=seed + scope_index * 100 + model_index,
            )
            difference["scope"] = scope
            difference_rows.append(difference)
    neural_performance = pd.concat(performance_rows, ignore_index=True)
    neural_ties = pd.concat(tie_rows, ignore_index=True)
    differences = pd.concat(difference_rows, ignore_index=True)
    combined_performance = pd.concat(
        [pd.read_csv(base_path), neural_performance], ignore_index=True
    )
    combined_ties = pd.concat([pd.read_csv(tie_path), neural_ties], ignore_index=True)
    _save_table(neural_performance, table_root / "neural_test_performance")
    _save_table(combined_performance, table_root / "all_model_test_performance")
    _save_table(differences, table_root / "xgb_vs_neural_test_differences")
    _save_table(combined_ties, table_root / "all_model_test_tie_sensitivity")

    plotting = combined_performance[
        combined_performance["metric"].isin(("ndcg_at_k", "top_k_overlap", "jaccard"))
    ].copy()
    plotting["series"] = (
        plotting["model"] + " / " + plotting["scope"] + " / " + plotting["selector"]
    )
    sns.set_theme(style="whitegrid", context="paper")
    fig, axis = plt.subplots(figsize=(12, 7))
    sns.pointplot(
        data=plotting,
        x="metric",
        y="estimate",
        hue="series",
        order=["top_k_overlap", "jaccard", "ndcg_at_k"],
        errorbar=None,
        dodge=0.55,
        ax=axis,
    )
    axis.set_xticks(range(3), ["Top-k overlap", "Jaccard", "nDCG@k"])
    axis.set_xlabel("")
    axis.set_ylabel("Article-weighted sealed-test estimate")
    axis.legend(title="Model / scope / selector", frameon=False, fontsize=7)
    fig.tight_layout()
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "all_model_test_performance.svg", bbox_inches="tight")
    fig.savefig(figure_root / "all_model_test_performance.pdf", bbox_inches="tight")
    plt.close(fig)
    return {
        "models": list(neural_roots),
        "tables": [
            "neural_test_performance.csv",
            "all_model_test_performance.csv",
            "xgb_vs_neural_test_differences.csv",
            "all_model_test_tie_sensitivity.csv",
        ],
        "figures": [
            "all_model_test_performance.svg",
            "all_model_test_performance.pdf",
        ],
    }
