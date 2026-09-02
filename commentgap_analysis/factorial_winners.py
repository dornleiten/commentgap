"""Freeze factorial winners using development cross-validation and nothing else."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "variant_id",
    "family",
    "scope",
    "fold",
    "macro_ndcg_at_k",
}


def active_factorial_runner_processes() -> list[dict[str, str | int]]:
    """Return live Linux processes whose command runs the factorial launcher."""
    processes: list[dict[str, str | int]] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return processes
    for process_root in proc_root.iterdir():
        if not process_root.name.isdigit() or int(process_root.name) == os.getpid():
            continue
        try:
            command = (process_root / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "scripts/run_ranker_factorial.py" in command:
            processes.append({"pid": int(process_root.name), "command": command})
    return sorted(processes, key=lambda item: int(item["pid"]))


def assert_factorial_idle() -> None:
    active = active_factorial_runner_processes()
    if active:
        pids = ", ".join(str(item["pid"]) for item in active)
        raise RuntimeError(
            "Refusing to freeze winners while run_ranker_factorial.py is active "
            f"(PID(s): {pids})"
        )


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def rank_development_cv_variants(
    frame: pd.DataFrame,
    *,
    scopes: Iterable[str] = ("all",),
    expected_folds: int = 5,
    draw_policies: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return complete-variant ranking, winners, and excluded variants.

    Held-out columns are neither required nor used. Duplicate variant/fold rows
    are treated as corruption; incomplete variants are reported and excluded.
    """
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Development CV table is missing columns: {missing}")
    scopes = tuple(dict.fromkeys(scopes))
    if not scopes or not set(scopes).issubset({"all", "root"}):
        raise ValueError("scopes must contain all and/or root")
    if expected_folds < 2:
        raise ValueError("expected_folds must be at least 2")
    requested_draw_policies = (
        None if draw_policies is None else tuple(dict.fromkeys(draw_policies))
    )
    if requested_draw_policies is not None:
        if not requested_draw_policies or not set(requested_draw_policies).issubset(
            {"draw1", "mean10"}
        ):
            raise ValueError("draw_policies must contain draw1 and/or mean10")
        if "draw_policy" not in frame.columns:
            raise ValueError("Development CV table is missing draw_policy")

    work = frame[frame["scope"].isin(scopes)].copy()
    if work.empty:
        raise ValueError(f"No development CV rows for requested scopes {scopes}")
    if requested_draw_policies is not None:
        work = work[work["draw_policy"].isin(requested_draw_policies)]
    keys = ["variant_id", "family", "scope"]
    work["fold"] = pd.to_numeric(work["fold"], errors="raise").astype(int)
    work["macro_ndcg_at_k"] = pd.to_numeric(
        work["macro_ndcg_at_k"], errors="raise"
    )
    if not np.isfinite(work["macro_ndcg_at_k"]).all():
        raise ValueError("Development CV macro nDCG contains non-finite values")
    if work.duplicated([*keys, "fold"]).any():
        examples = work.loc[
            work.duplicated([*keys, "fold"], keep=False), [*keys, "fold"]
        ].head().to_dict("records")
        raise ValueError(f"Duplicate variant/fold development results: {examples}")

    expected = set(range(expected_folds))
    eligibility_rows = []
    complete_keys = []
    for key, group in work.groupby(keys, sort=False, observed=True):
        observed = set(group["fold"].tolist())
        complete = observed == expected and len(group) == expected_folds
        eligibility_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "complete": complete,
                "observed_folds": ",".join(map(str, sorted(observed))),
                "missing_folds": ",".join(map(str, sorted(expected - observed))),
                "unexpected_folds": ",".join(map(str, sorted(observed - expected))),
            }
        )
        if complete:
            complete_keys.append(key)
    eligibility = pd.DataFrame(eligibility_rows)
    excluded = eligibility[~eligibility["complete"]].reset_index(drop=True)
    if not complete_keys:
        raise ValueError("No variants have all expected development folds")

    complete_index = pd.MultiIndex.from_tuples(complete_keys, names=keys)
    indexed = work.set_index(keys)
    complete = indexed[indexed.index.isin(complete_index)].reset_index()
    aggregations: dict[str, tuple[str, str]] = {
        "mean_macro_ndcg_at_k": ("macro_ndcg_at_k", "mean"),
        "sd_macro_ndcg_at_k": ("macro_ndcg_at_k", "std"),
        "min_fold_macro_ndcg_at_k": ("macro_ndcg_at_k", "min"),
        "completed_folds": ("fold", "nunique"),
    }
    for column in ("audience_ndcg_at_k", "curator_ndcg_at_k"):
        if column in complete.columns:
            aggregations[f"mean_{column}"] = (column, "mean")
    for column in (
        "feature_set",
        "draw_policy",
        "negative_sampling",
        "schedule",
        "heads",
        "network",
    ):
        if column in complete.columns:
            aggregations[column] = (column, "first")
    ranking = complete.groupby(keys, as_index=False, observed=True).agg(**aggregations)
    ranking = ranking.sort_values(
        ["family", "scope", "mean_macro_ndcg_at_k", "sd_macro_ndcg_at_k", "min_fold_macro_ndcg_at_k", "variant_id"],
        ascending=[True, True, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    # The factorial crosses model family with feature set.  When feature-set
    # metadata are present, select one winner within each family/scope/
    # feature-set cell rather than allowing the stronger feature set to hide
    # the other cell.  Keep the two-dimensional fallback for legacy fixtures
    # and pre-factorial result tables that do not carry feature_set.
    winner_groups = ["family", "scope"]
    if "feature_set" in ranking.columns:
        winner_groups.append("feature_set")
    ranking["development_cv_rank"] = (
        ranking.groupby(winner_groups, sort=False).cumcount() + 1
    )
    winners = ranking[ranking["development_cv_rank"] == 1].reset_index(drop=True)
    return ranking, winners, excluded


def freeze_development_cv_winners(
    *,
    development_cv_path: Path = Path(
        "model_output/selection_2025/factorial_rankers/development_cv_results.csv"
    ),
    experiment_variants_path: Path | None = None,
    output_root: Path = Path("model_output/selection_2025/paper1/factorial_winners"),
    scopes: Iterable[str] = ("all",),
    draw_policies: Iterable[str] | None = None,
    expected_folds: int = 5,
    require_idle: bool = True,
) -> dict:
    """Write the ranking and manifest without reading any held-out artifact."""
    if require_idle:
        assert_factorial_idle()
    if not development_cv_path.exists():
        raise FileNotFoundError(development_cv_path)
    experiment_variants_path = experiment_variants_path or (
        development_cv_path.parent / "experiment_variants.csv"
    )
    if not experiment_variants_path.exists():
        raise FileNotFoundError(experiment_variants_path)
    requested_scopes = tuple(dict.fromkeys(scopes))
    requested_draw_policies = (
        None if draw_policies is None else tuple(dict.fromkeys(draw_policies))
    )
    if requested_draw_policies is not None and (
        not requested_draw_policies
        or not set(requested_draw_policies).issubset({"draw1", "mean10"})
    ):
        raise ValueError("draw_policies must contain draw1 and/or mean10")
    plan = pd.read_csv(experiment_variants_path)
    plan_required = {"variant_id", "family", "scope", "status"}
    plan_missing = sorted(plan_required - set(plan.columns))
    if plan_missing:
        raise ValueError(f"Experiment variant table is missing columns: {plan_missing}")
    requested_plan = plan[plan["scope"].isin(requested_scopes)].copy()
    if requested_draw_policies is not None:
        if "draw_policy" not in requested_plan.columns:
            raise ValueError("Experiment variant table is missing draw_policy")
        requested_plan = requested_plan[
            requested_plan["draw_policy"].isin(requested_draw_policies)
        ]
    if requested_plan.empty:
        raise ValueError(f"No planned variants for requested scopes {requested_scopes}")
    incomplete_plan = requested_plan[requested_plan["status"] != "complete"]
    if not incomplete_plan.empty:
        examples = incomplete_plan[["variant_id", "scope", "status"]].head().to_dict("records")
        raise RuntimeError(
            "Factorial experiment is not complete for every requested variant/scope: "
            f"{examples}"
        )
    source = pd.read_csv(development_cv_path)
    ranking, winners, excluded = rank_development_cv_variants(
        source, scopes=requested_scopes, expected_folds=expected_folds, draw_policies=requested_draw_policies
    )
    planned_keys = set(
        requested_plan[["variant_id", "family", "scope"]].itertuples(index=False, name=None)
    )
    ranked_keys = set(
        ranking[["variant_id", "family", "scope"]].itertuples(index=False, name=None)
    )
    missing_results = sorted(planned_keys - ranked_keys)
    unexpected_results = sorted(ranked_keys - planned_keys)
    if missing_results or unexpected_results or not excluded.empty:
        raise RuntimeError(
            "Development CV results do not exactly cover the completed experiment plan; "
            f"missing={missing_results[:5]}, unexpected={unexpected_results[:5]}, "
            f"incomplete={excluded['variant_id'].head().tolist()}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    ranking_path = output_root / "development_cv_variant_ranking.csv"
    winners_path = output_root / "development_cv_winners.csv"
    excluded_path = output_root / "incomplete_variants_excluded.csv"
    ranking.to_csv(ranking_path, index=False)
    winners.to_csv(winners_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_source": "development_cv_results.csv only",
        "held_out_artifacts_read": False,
        "scopes": list(requested_scopes),
        "draw_policies": (
            list(requested_draw_policies) if requested_draw_policies is not None else None
        ),
        "selection_rationale": "Restricting reported winners to draw1 keeps the reported model-selection contract consistent across cells; mean10 remains available in the full development-CV results for sensitivity analysis." if requested_draw_policies == ("draw1",) else None,
        "primary_scope": "all",
        "appendix_scope": "root",
        "winner_dimensions": [
            "family",
            "scope",
            *(["feature_set"] if "feature_set" in winners.columns else []),
        ],
        "winner_levels": {
            column: sorted(winners[column].dropna().astype(str).unique().tolist())
            for column in ("family", "scope", "feature_set")
            if column in winners.columns
        },
        "expected_folds": expected_folds,
        "rule": [
            "descending mean macro nDCG@k",
            "ascending fold SD",
            "descending minimum fold macro nDCG@k",
            "lexicographic stable variant_id",
        ],
        "input": {
            "path": str(development_cv_path),
            "sha256": _sha256(development_cv_path),
            "rows": len(source),
        },
        "experiment_plan": {
            "path": str(experiment_variants_path),
            "sha256": _sha256(experiment_variants_path),
            "rows_requested": len(requested_plan),
            "all_requested_status_complete": True,
        },
        "winners": winners.to_dict("records"),
        "outputs": {
            "ranking": {"path": str(ranking_path), "sha256": _sha256(ranking_path), "rows": len(ranking)},
            "winners": {"path": str(winners_path), "sha256": _sha256(winners_path), "rows": len(winners)},
            "excluded": {"path": str(excluded_path), "sha256": _sha256(excluded_path), "rows": len(excluded)},
        },
    }
    manifest_path = output_root / "factorial_winner_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
