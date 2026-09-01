"""Article-grouped learning-to-rank workflow for the two candidate scopes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd


RANKER_SEED = 20260813


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}|{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def build_stacked_rank_data(
    choice_set: pd.DataFrame,
    features: list[str],
    *,
    audience_draw: int = 1,
    auxiliary_columns: list[str] | None = None,
) -> pd.DataFrame:
    label = f"audience_selected_draw_{audience_draw:02d}"
    auxiliary_columns = list(auxiliary_columns or [])
    overlap = set(features).intersection(auxiliary_columns)
    if overlap:
        raise ValueError(
            "Auxiliary stacked-ranker columns duplicate model features: "
            f"{sorted(overlap)}"
        )
    required = (
        {"story_id", "comment_id", "curator_selected", label, "n_picks"}
        | set(features)
        | set(auxiliary_columns)
    )
    missing = required - set(choice_set.columns)
    if missing:
        raise ValueError(f"Missing stacked-ranker columns: {sorted(missing)}")
    common = ["story_id", "comment_id", "n_picks"] + features + auxiliary_columns
    audience = choice_set[common].copy()
    audience["selector"] = "audience"
    audience["selector_code"] = 0
    audience["selected"] = choice_set[label].astype(int).to_numpy()
    curator = choice_set[common].copy()
    curator["selector"] = "curator"
    curator["selector_code"] = 1
    curator["selected"] = choice_set["curator_selected"].astype(int).to_numpy()
    stacked = pd.concat([audience, curator], ignore_index=True)
    stacked["query_id"] = stacked["story_id"].astype(str) + "::" + stacked["selector"]
    stacked = stacked.sort_values(["query_id", "comment_id"]).reset_index(drop=True)
    checks = stacked.groupby("query_id").agg(
        selected=("selected", "sum"), expected=("n_picks", "first"), rows=("comment_id", "size")
    )
    if not (checks["selected"] == checks["expected"]).all():
        raise ValueError("A stacked query does not contain exactly k selected comments")
    if ((checks["selected"] <= 0) | (checks["selected"] >= checks["rows"])).any():
        raise ValueError("A stacked query lacks outcome variation")
    return stacked


def _article_scope_summary(choice_set: pd.DataFrame, scope: str) -> pd.DataFrame:
    required = {"story_id", "article_month", "n_candidates", "n_picks"}
    if scope == "all":
        required.add("is_reply")
    missing = required - set(choice_set.columns)
    if missing:
        raise ValueError(f"Missing {scope} split columns: {sorted(missing)}")
    aggregations: dict[str, tuple[str, str]] = {
        "article_month": ("article_month", "first"),
        f"n_candidates_{scope}": ("n_candidates", "first"),
        f"n_picks_{scope}": ("n_picks", "first"),
    }
    if scope == "all":
        aggregations["reply_proportion"] = ("is_reply", "mean")
    summary = choice_set.groupby("story_id", as_index=False).agg(**aggregations)
    summary["story_id"] = summary["story_id"].astype(str)
    return summary


def article_split_balance(split: pd.DataFrame) -> pd.DataFrame:
    """Summarize prespecified development/test balance covariates."""
    required = {
        "split_role",
        "n_candidates_root",
        "n_candidates_all",
        "n_picks_root",
        "n_picks_all",
        "reply_proportion",
    }
    missing = required - set(split.columns)
    if missing:
        raise ValueError(f"Missing split-balance columns: {sorted(missing)}")
    roles = {"development", "paper2_test"}
    if set(split["split_role"].unique()) != roles:
        raise ValueError("Split must contain development and paper2_test articles")
    values = pd.DataFrame(
        {
            "split_role": split["split_role"],
            "log1p_n_candidates_root": np.log1p(split["n_candidates_root"].astype(float)),
            "log1p_n_candidates_all": np.log1p(split["n_candidates_all"].astype(float)),
            "n_picks_root": split["n_picks_root"].astype(float),
            "n_picks_all": split["n_picks_all"].astype(float),
            "reply_proportion": split["reply_proportion"].astype(float),
        }
    )
    rows: list[dict[str, Any]] = []
    for covariate in values.columns.drop("split_role"):
        development = values.loc[values["split_role"] == "development", covariate]
        test = values.loc[values["split_role"] == "paper2_test", covariate]
        pooled_sd = float(np.sqrt((development.var(ddof=1) + test.var(ddof=1)) / 2))
        mean_difference = float(development.mean() - test.mean())
        if pooled_sd == 0:
            smd = 0.0 if mean_difference == 0 else np.inf
        else:
            smd = mean_difference / pooled_sd
        rows.append(
            {
                "covariate": covariate,
                "development_mean": float(development.mean()),
                "paper2_test_mean": float(test.mean()),
                "standardized_mean_difference": float(smd),
                "abs_standardized_mean_difference": float(abs(smd)),
            }
        )
    return pd.DataFrame(rows)


def _assign_stratified_roles(
    article: pd.DataFrame,
    *,
    test_fraction: float,
    development_folds: int,
    seed: int,
) -> pd.DataFrame:
    assigned = article.copy()
    assigned["_assignment_hash"] = assigned["story_id"].map(
        lambda value: _stable_fraction(str(value), seed)
    )
    target_test = int(round(len(assigned) * test_fraction))
    allocations: dict[str, int] = {}
    allocation_rows: list[tuple[float, float, str]] = []
    allocated = 0
    for stratum, indices in assigned.groupby("split_stratum", sort=True).groups.items():
        exact = len(indices) * test_fraction
        base = int(np.floor(exact))
        allocations[str(stratum)] = base
        allocated += base
        allocation_rows.append(
            (
                exact - base,
                _stable_fraction(str(stratum), seed + 1),
                str(stratum),
            )
        )
    extras = target_test - allocated
    for _, _, stratum in sorted(allocation_rows, key=lambda row: (-row[0], row[1], row[2]))[
        :extras
    ]:
        allocations[stratum] += 1

    assigned["split_role"] = "development"
    for stratum, indices in assigned.groupby("split_stratum", sort=True).groups.items():
        ordered = assigned.loc[indices].sort_values(["_assignment_hash", "story_id"]).index
        assigned.loc[ordered[: allocations[str(stratum)]], "split_role"] = "paper2_test"

    assigned["development_fold"] = -1
    development = assigned[assigned["split_role"] == "development"]
    for stratum, indices in development.groupby("split_stratum", sort=True).groups.items():
        ordered = assigned.loc[indices].sort_values(["_assignment_hash", "story_id"]).index
        offset = int(_stable_fraction(str(stratum), seed + 2) * development_folds)
        assigned.loc[ordered, "development_fold"] = (
            np.arange(len(ordered)) + offset
        ) % development_folds
    assigned["assignment_seed"] = seed
    return assigned.drop(columns="_assignment_hash")


def assign_article_splits(
    root_choice_set: pd.DataFrame,
    all_choice_set: pd.DataFrame,
    *,
    seed: int = RANKER_SEED,
    test_fraction: float = 0.50,
    size_bins: int = 3,
    development_folds: int = 5,
    max_abs_smd: float = 0.05,
    max_attempts: int = 1_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one shared month-by-size split for both ranking scopes.

    Only stories present in both choice sets enter the primary Paper 2 sample.
    Assignment is rerandomized deterministically until all prespecified balance
    diagnostics meet max_abs_smd.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie between zero and one")
    if size_bins < 2:
        raise ValueError("size_bins must be at least two")
    if development_folds < 2:
        raise ValueError("development_folds must be at least two")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    root = _article_scope_summary(root_choice_set, "root")
    all_comments = _article_scope_summary(all_choice_set, "all")
    article = root.merge(
        all_comments,
        on="story_id",
        how="inner",
        suffixes=("_root", "_all"),
        validate="one_to_one",
    ).sort_values("story_id").reset_index(drop=True)
    if len(article) < development_folds * 2:
        raise ValueError("Too few common articles for development and Paper 2 test sets")
    if not (article["article_month_root"] == article["article_month_all"]).all():
        raise ValueError("Root and all choice sets disagree on article month")
    article["article_month"] = article.pop("article_month_root")
    article = article.drop(columns="article_month_all")

    # A shared size score gives equal influence to root and all-comment pool
    # sizes without crossing two separate size strata.
    article["joint_size_rank"] = (
        article["n_candidates_root"].rank(pct=True, method="average")
        + article["n_candidates_all"].rank(pct=True, method="average")
    ) / 2
    article["size_tercile"] = pd.qcut(
        article["joint_size_rank"].rank(method="first"),
        q=size_bins,
        labels=False,
        duplicates="drop",
    ).astype(int)
    article["split_stratum"] = (
        article["article_month"].astype(str)
        + "|size_"
        + article["size_tercile"].astype(str)
    )

    best_max_smd = np.inf
    for attempt in range(max_attempts):
        candidate = _assign_stratified_roles(
            article,
            test_fraction=test_fraction,
            development_folds=development_folds,
            seed=seed + attempt,
        )
        balance = article_split_balance(candidate)
        candidate_max_smd = float(balance["abs_standardized_mean_difference"].max())
        if candidate_max_smd < best_max_smd:
            best_max_smd = candidate_max_smd
        if candidate_max_smd <= max_abs_smd:
            candidate["assignment_attempt"] = attempt + 1
            balance["acceptance_threshold"] = max_abs_smd
            balance["accepted"] = True
            return candidate, balance
    raise ValueError(
        "Could not construct an acceptably balanced split after "
        f"{max_attempts} attempts; best maximum absolute SMD was {best_max_smd:.4f}"
    )


def _numeric_qid(frame: pd.DataFrame) -> np.ndarray:
    return pd.factorize(frame["query_id"], sort=True)[0].astype(np.int64)


def _xgb_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import xgboost as xgb

        info = xgb.build_info()
        if bool(info.get("USE_CUDA")):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def default_search_configs(
    *,
    count: int = 32,
    seed: int = RANKER_SEED,
) -> list[dict[str, Any]]:
    """Draw a reproducible broad random XGBoost search."""

    if count < 1:
        raise ValueError("Broad-search configuration count must be positive")
    rng = np.random.default_rng(seed)
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(configs) < count:
        reg_alpha = 0.0 if rng.random() < 0.30 else float(
            np.exp(rng.uniform(np.log(1e-3), np.log(3.0)))
        )
        gamma = 0.0 if rng.random() < 0.30 else float(
            np.exp(rng.uniform(np.log(1e-3), np.log(3.0)))
        )
        config = {
            "max_depth": int(rng.integers(3, 10)),
            "learning_rate": round(
                float(np.exp(rng.uniform(np.log(0.015), np.log(0.12)))), 6
            ),
            "min_child_weight": round(
                float(np.exp(rng.uniform(np.log(3.0), np.log(50.0)))), 6
            ),
            "subsample": round(float(rng.uniform(0.65, 1.0)), 6),
            "colsample_bytree": round(float(rng.uniform(0.55, 1.0)), 6),
            "reg_lambda": round(
                float(np.exp(rng.uniform(np.log(0.5), np.log(30.0)))), 6
            ),
            "reg_alpha": round(reg_alpha, 6),
            "gamma": round(gamma, 6),
        }
        fingerprint = json.dumps(config, sort_keys=True)
        if fingerprint not in seen:
            configs.append(config)
            seen.add(fingerprint)
    return configs


def refinement_search_configs(
    broad_history: pd.DataFrame,
    *,
    top_configs: int = 5,
) -> list[dict[str, Any]]:
    """Build a 27-point local grid around the strongest broad-CV region."""

    required = {"config_index", "macro_ndcg_at_k", "parameters"}
    missing = required - set(broad_history.columns)
    if missing:
        raise ValueError(f"Broad-search history is missing columns: {sorted(missing)}")
    if top_configs < 1:
        raise ValueError("refinement top_configs must be positive")
    summary = (
        broad_history.groupby(["config_index", "parameters"], as_index=False)
        .agg(mean_macro_ndcg_at_k=("macro_ndcg_at_k", "mean"))
        .sort_values(
            ["mean_macro_ndcg_at_k", "config_index"],
            ascending=[False, True],
        )
        .head(top_configs)
    )
    if summary.empty:
        raise ValueError("Broad-search history contains no completed configurations")
    strongest = [json.loads(value) for value in summary["parameters"]]

    def median(key: str) -> float:
        return float(np.median([float(config[key]) for config in strongest]))

    def geometric_median(key: str) -> float:
        values = np.asarray([float(config[key]) for config in strongest])
        return float(np.exp(np.median(np.log(values))))

    center_depth = int(round(median("max_depth")))
    center_learning_rate = geometric_median("learning_rate")
    center_child_weight = geometric_median("min_child_weight")
    held = {
        "subsample": round(median("subsample"), 6),
        "colsample_bytree": round(median("colsample_bytree"), 6),
        "reg_lambda": round(geometric_median("reg_lambda"), 6),
        "reg_alpha": round(median("reg_alpha"), 6),
        "gamma": round(median("gamma"), 6),
    }
    depths = (
        [3, 4, 5]
        if center_depth <= 3
        else [center_depth - 1, center_depth, min(10, center_depth + 1)]
    )
    learning_rates = [
        round(max(0.005, min(0.2, center_learning_rate * factor)), 6)
        for factor in (0.75, 1.0, 1.25)
    ]
    child_weights = [
        round(max(1.0, min(100.0, center_child_weight * factor)), 6)
        for factor in (0.75, 1.0, 1.25)
    ]
    configs = []
    for depth in depths:
        for learning_rate in learning_rates:
            for child_weight in child_weights:
                configs.append(
                    {
                        "max_depth": depth,
                        "learning_rate": learning_rate,
                        "min_child_weight": child_weight,
                        **held,
                    }
                )
    return configs


def select_best_configuration(
    history: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select the highest mean development-CV nDCG configuration."""

    required = {
        "search_stage",
        "config_index",
        "fold",
        "macro_ndcg_at_k",
        "best_iteration",
        "parameters",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Tuning history is missing columns: {sorted(missing)}")
    summary = (
        history.groupby(
            ["search_stage", "config_index", "parameters"],
            as_index=False,
        )
        .agg(
            mean_macro_ndcg_at_k=("macro_ndcg_at_k", "mean"),
            sd_macro_ndcg_at_k=("macro_ndcg_at_k", "std"),
            median_best_iteration=("best_iteration", "median"),
            completed_folds=("fold", "nunique"),
        )
        .sort_values(
            ["mean_macro_ndcg_at_k", "sd_macro_ndcg_at_k", "search_stage", "config_index"],
            ascending=[False, True, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    winner = summary.iloc[0]
    best = json.loads(winner["parameters"])
    best["n_estimators"] = max(50, int(winner["median_best_iteration"]) + 1)
    return best, summary


def make_ranker(
    parameters: dict[str, Any],
    *,
    device: str = "auto",
    seed: int = RANKER_SEED,
    early_stopping_rounds: int | None = 50,
):
    import xgboost as xgb

    parameters = dict(parameters)
    n_estimators = int(parameters.pop("n_estimators", 1_500))
    return xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        tree_method="hist",
        device=_xgb_device(device),
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=10,
        early_stopping_rounds=early_stopping_rounds,
        **parameters,
    )


def _fit_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    features: list[str],
    parameters: dict[str, Any],
    *,
    device: str,
    seed: int,
):
    train = train.sort_values(["query_id", "comment_id"]).reset_index(drop=True)
    validation = (
        validation.sort_values(["query_id", "comment_id"]).reset_index(drop=True)
        if validation is not None
        else None
    )
    model = make_ranker(
        parameters,
        device=device,
        seed=seed,
        early_stopping_rounds=50 if validation is not None else None,
    )
    kwargs: dict[str, Any] = {"qid": _numeric_qid(train), "verbose": False}
    if validation is not None:
        kwargs.update(
            eval_set=[(validation[features + ["selector_code"]], validation["selected"])],
            eval_qid=[_numeric_qid(validation)],
        )
    model.fit(train[features + ["selector_code"]], train["selected"], **kwargs)
    return model


def _predict_cpu(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Predict from a pandas frame without CUDA/CPU device-mismatch fallback."""
    model.set_params(device="cpu")
    return np.asarray(model.predict(frame))


def evaluate_rank_scores(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    if len(frame) != len(scores):
        raise ValueError("Score length does not match ranking frame")
    evaluated = frame[["story_id", "comment_id", "selector", "query_id", "n_picks", "selected"]].copy()
    evaluated["score"] = np.asarray(scores)
    rows: list[dict[str, Any]] = []
    for query_id, group in evaluated.groupby("query_id", sort=False):
        group = group.sort_values(["score", "comment_id"], ascending=[False, True]).reset_index(drop=True)
        k = int(group["n_picks"].iloc[0])
        top = group.iloc[:k]
        hits = int(top["selected"].sum())
        discounts = 1 / np.log2(np.arange(2, len(group) + 2))
        dcg = float((group["selected"].to_numpy()[:k] * discounts[:k]).sum())
        ideal = float(discounts[:k].sum())
        selected_ranks = np.flatnonzero(group["selected"].to_numpy()) + 1
        rows.append(
            {
                "query_id": query_id,
                "story_id": group["story_id"].iloc[0],
                "selector": group["selector"].iloc[0],
                "n_candidates": len(group),
                "n_picks": k,
                "top_k_overlap": hits / k,
                "jaccard": hits / (2 * k - hits),
                "ndcg_at_k": dcg / ideal if ideal else np.nan,
                "mean_selected_rank": float(selected_ranks.mean()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric_summary(
    article_metrics: pd.DataFrame,
    *,
    draws: int = 1_000,
    seed: int = RANKER_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ["top_k_overlap", "jaccard", "ndcg_at_k", "mean_selected_rank"]
    output: list[dict[str, Any]] = []
    for selector, subset in article_metrics.groupby("selector"):
        stories = subset["story_id"].drop_duplicates().to_numpy()
        point = subset[metrics].mean()
        samples = np.empty((draws, len(metrics)))
        for draw in range(draws):
            sampled = rng.choice(stories, size=len(stories), replace=True)
            weights = pd.Series(sampled).value_counts()
            merged = subset.merge(weights.rename("weight"), left_on="story_id", right_index=True)
            samples[draw] = [np.average(merged[name], weights=merged["weight"]) for name in metrics]
        for column, name in enumerate(metrics):
            output.append(
                {
                    "selector": selector,
                    "metric": name,
                    "estimate": float(point[name]),
                    "conf_low": float(np.quantile(samples[:, column], 0.025)),
                    "conf_high": float(np.quantile(samples[:, column], 0.975)),
                    "bootstrap_draws": draws,
                }
            )
    return pd.DataFrame(output)


def grouped_permutation_importance(
    model: Any,
    validation: pd.DataFrame,
    features: list[str],
    *,
    repeats: int = 3,
    seed: int = RANKER_SEED,
) -> pd.DataFrame:
    """Permute one feature within article-selector queries and score macro nDCG loss."""
    x_columns = features + ["selector_code"]
    baseline_scores = _predict_cpu(model, validation[x_columns])
    baseline = evaluate_rank_scores(validation, baseline_scores)["ndcg_at_k"].mean()
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for feature in features:
        for repeat in range(repeats):
            permuted = validation[x_columns].copy()
            for _, indices in validation.groupby("query_id", sort=False).groups.items():
                values = permuted.loc[indices, feature].to_numpy(copy=True)
                rng.shuffle(values)
                permuted.loc[indices, feature] = values
            score = _predict_cpu(model, permuted)
            macro = evaluate_rank_scores(validation, score)["ndcg_at_k"].mean()
            records.append(
                {
                    "feature": feature,
                    "repeat": repeat + 1,
                    "baseline_macro_ndcg_at_k": float(baseline),
                    "permuted_macro_ndcg_at_k": float(macro),
                    "importance": float(baseline - macro),
                }
            )
    return pd.DataFrame(records)


def tune_ranker(
    development_frame: pd.DataFrame,
    features: list[str],
    *,
    configs: list[dict[str, Any]] | None = None,
    folds: int = 5,
    fold_column: str = "development_fold",
    device: str = "auto",
    seed: int = RANKER_SEED,
    search_stage: str = "broad_random",
    history_path: Path | None = None,
    fold_feature_columns: dict[int, dict[str, str]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    configs = configs if configs is not None else default_search_configs(seed=seed)
    if not configs:
        raise ValueError("At least one tuning configuration is required")
    stories = sorted(development_frame["story_id"].astype(str).unique())
    if len(stories) < folds:
        raise ValueError("Too few development articles for grouped cross-validation")
    if fold_column in development_frame:
        article_folds = development_frame[["story_id", fold_column]].drop_duplicates()
        if article_folds["story_id"].duplicated().any():
            raise ValueError("A development article has multiple fold assignments")
        observed_folds = set(article_folds[fold_column].astype(int))
        if observed_folds != set(range(folds)):
            raise ValueError(
                f"Expected development folds {list(range(folds))}, got {sorted(observed_folds)}"
            )
        fold_by_story = dict(
            zip(article_folds["story_id"].astype(str), article_folds[fold_column].astype(int))
        )
    else:
        fold_by_story = {
            story: int(_stable_fraction(story, seed + 17) * folds) % folds for story in stories
        }
        for position, story in enumerate(stories[:folds]):
            fold_by_story[story] = position
    fold_source_columns = sorted({
        source
        for replacements in (fold_feature_columns or {}).values()
        for source in replacements.values()
    })
    tuning_columns = ["story_id"] + (
        [fold_column] if fold_column in development_frame else []
    ) + features + fold_source_columns
    missing_tuning_columns = set(tuning_columns) - set(development_frame.columns)
    if missing_tuning_columns:
        raise ValueError(f"Missing tuning-signature columns: {sorted(missing_tuning_columns)}")
    tuning_signature = hashlib.sha256(
        pd.util.hash_pandas_object(
            development_frame[tuning_columns], index=False
        ).to_numpy().tobytes()
    ).hexdigest()
    valid_keys = {
        (
            search_stage,
            config_index,
            fold,
            json.dumps(parameters, sort_keys=True),
            tuning_signature,
        )
        for config_index, parameters in enumerate(configs)
        for fold in range(folds)
    }
    records: list[dict[str, Any]] = []
    if history_path is not None and Path(history_path).exists():
        existing = pd.read_csv(history_path)
        if not existing.empty:
            records = [
                record
                for record in existing.to_dict("records")
                if (
                    str(record["search_stage"]),
                    int(record["config_index"]),
                    int(record["fold"]),
                    str(record["parameters"]),
                    str(record.get("tuning_signature", "")),
                )
                in valid_keys
            ]
    completed = {
        (
            str(record["search_stage"]),
            int(record["config_index"]),
            int(record["fold"]),
            str(record["parameters"]),
            str(record.get("tuning_signature", "")),
        )
        for record in records
    }
    for config_index, parameters in enumerate(configs):
        serialized = json.dumps(parameters, sort_keys=True)
        print(
            f"{search_stage}: configuration {config_index + 1}/{len(configs)} "
            f"({folds} development folds)",
            flush=True,
        )
        for fold in range(folds):
            key = (search_stage, config_index, fold, serialized, tuning_signature)
            if key in completed:
                continue
            validation_stories = {story for story, assigned in fold_by_story.items() if assigned == fold}
            train = development_frame[
                ~development_frame["story_id"].astype(str).isin(validation_stories)
            ].copy()
            validation = development_frame[
                development_frame["story_id"].astype(str).isin(validation_stories)
            ].copy()
            replacements = (fold_feature_columns or {}).get(fold, {})
            for feature, source_column in replacements.items():
                if feature not in features:
                    raise ValueError(f"Fold replacement targets unknown feature: {feature}")
                if source_column not in development_frame:
                    raise ValueError(f"Missing fold-specific feature column: {source_column}")
                train[feature] = train[source_column].to_numpy()
                validation[feature] = validation[source_column].to_numpy()
            model = _fit_ranker(
                train,
                validation,
                features,
                parameters,
                device=device,
                seed=seed + config_index * 10 + fold,
            )
            scores = _predict_cpu(model, validation[features + ["selector_code"]])
            metrics = evaluate_rank_scores(validation, scores)
            records.append(
                {
                    "search_stage": search_stage,
                    "config_index": config_index,
                    "fold": fold,
                    "macro_ndcg_at_k": float(metrics["ndcg_at_k"].mean()),
                    "best_iteration": int(getattr(model, "best_iteration", 0)),
                    "parameters": serialized,
                    "tuning_signature": tuning_signature,
                }
            )
            completed.add(key)
            if history_path is not None:
                Path(history_path).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(records).to_csv(history_path, index=False)
    history = pd.DataFrame(records)
    stage_history = history[history["search_stage"] == search_stage].copy()
    expected = len(configs) * folds
    if len(stage_history) != expected:
        raise ValueError(
            f"{search_stage} history has {len(stage_history)} rows; expected {expected}"
        )
    best, _ = select_best_configuration(stage_history)
    return best, stage_history


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd")
    os.replace(temporary, path)


def _frame_fingerprint(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    sort_by: list[str],
) -> str:
    ordered = frame[columns].sort_values(sort_by).reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    ).hexdigest()


def _cache_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_stage_valid(
    cache: dict[str, Any],
    stage: str,
    signature: str,
    artifacts: list[Path],
    *,
    force_recompute: bool,
) -> bool:
    recorded = cache.get("stages", {}).get(stage, {})
    artifact_hashes = recorded.get("artifact_sha256", {})
    return (
        not force_recompute
        and recorded.get("signature") == signature
        and set(artifact_hashes) == {path.name for path in artifacts}
        and all(
            path.exists() and artifact_hashes.get(path.name) == _file_sha256(path)
            for path in artifacts
        )
    )


def run_ranker_workflow(
    choice_set: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    *,
    scope: str,
    article_split: pd.DataFrame,
    device: str = "auto",
    seed: int = RANKER_SEED,
    bootstrap_draws: int = 1_000,
    development_folds: int = 5,
    configs: list[dict[str, Any]] | None = None,
    broad_search_count: int = 32,
    refinement_top_configs: int = 5,
    permutation_repeats: int = 3,
    force_recompute: bool = False,
    fold_feature_columns: dict[int, dict[str, str]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir) / scope
    output_dir.mkdir(parents=True, exist_ok=True)
    required_split_columns = {
        "story_id",
        "split_role",
        "development_fold",
        "split_stratum",
    }
    missing_split_columns = required_split_columns - set(article_split.columns)
    if missing_split_columns:
        raise ValueError(f"Missing article-split columns: {sorted(missing_split_columns)}")
    if scope not in {"root", "all"}:
        raise ValueError("scope must be 'root' or 'all'")

    split = article_split.copy()
    split["story_id"] = split["story_id"].astype(str)
    if split["story_id"].duplicated().any():
        raise ValueError("The shared article split contains duplicate stories")
    if set(split["split_role"].unique()) != {"development", "paper2_test"}:
        raise ValueError("The shared article split has invalid roles")
    if not (split.loc[split["split_role"] == "paper2_test", "development_fold"] == -1).all():
        raise ValueError("Paper 2 test articles must not have development folds")
    choice = choice_set.copy()
    choice["story_id"] = choice["story_id"].astype(str)
    split_stories = set(split["story_id"])
    available_stories = set(choice["story_id"])
    missing_stories = split_stories - available_stories
    if missing_stories:
        raise ValueError(
            f"{scope} choice set is missing {len(missing_stories)} shared-split stories"
        )
    excluded_articles = len(available_stories - split_stories)
    choice = choice[choice["story_id"].isin(split_stories)].copy()

    fold_source_columns = sorted({
        source
        for replacements in (fold_feature_columns or {}).values()
        for source in replacements.values()
    })
    fingerprint_columns = [
        "story_id",
        "comment_id",
        "curator_selected",
        "audience_selected_draw_01",
    ] + features + fold_source_columns
    choice_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(
            choice[fingerprint_columns], index=False
        ).to_numpy().tobytes()
    ).hexdigest()
    split_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(
            split.sort_values("story_id")[
                ["story_id", "split_role", "development_fold", "split_stratum"]
            ],
            index=False,
        ).to_numpy().tobytes()
    ).hexdigest()

    stacked = build_stacked_rank_data(
        choice,
        features,
        auxiliary_columns=fold_source_columns,
    )
    stacked = stacked.merge(
        split[["story_id", "split_role", "development_fold"]],
        on="story_id",
        validate="many_to_one",
    )
    development = stacked[stacked["split_role"] == "development"].copy()
    test = stacked[stacked["split_role"] == "paper2_test"].copy()
    if development.empty or test.empty:
        raise ValueError("Development and Paper 2 test data must both be non-empty")
    development_stories = set(development["story_id"])
    test_stories = set(test["story_id"])
    if development_stories & test_stories:
        raise ValueError("Development and Paper 2 test stories overlap")

    if configs is not None:
        best, history = tune_ranker(
            development,
            features,
            configs=configs,
            folds=development_folds,
            fold_column="development_fold",
            device=device,
            seed=seed,
            search_stage="provided",
            history_path=output_dir / "development_cv_provided.csv",
            fold_feature_columns=fold_feature_columns,
        )
        search_summary = select_best_configuration(history)[1]
        search_strategy = "provided_configurations"
    else:
        broad_configs = default_search_configs(count=broad_search_count, seed=seed)
        (output_dir / "development_broad_configurations.json").write_text(
            json.dumps(broad_configs, indent=2, sort_keys=True) + "\n"
        )
        _, broad_history = tune_ranker(
            development,
            features,
            configs=broad_configs,
            folds=development_folds,
            fold_column="development_fold",
            device=device,
            seed=seed,
            search_stage="broad_random",
            history_path=output_dir / "development_cv_broad_random.csv",
            fold_feature_columns=fold_feature_columns,
        )
        narrow_configs = refinement_search_configs(
            broad_history,
            top_configs=refinement_top_configs,
        )
        (output_dir / "development_narrow_configurations.json").write_text(
            json.dumps(narrow_configs, indent=2, sort_keys=True) + "\n"
        )
        _, narrow_history = tune_ranker(
            development,
            features,
            configs=narrow_configs,
            folds=development_folds,
            fold_column="development_fold",
            device=device,
            seed=seed + 100_000,
            search_stage="narrow_grid",
            history_path=output_dir / "development_cv_narrow_grid.csv",
            fold_feature_columns=fold_feature_columns,
        )
        history = pd.concat([broad_history, narrow_history], ignore_index=True)
        best, search_summary = select_best_configuration(history)
        search_strategy = "broad_random_then_narrow_grid"
    history.to_csv(output_dir / "development_cv_history.csv", index=False)
    search_summary.to_csv(output_dir / "development_cv_summary.csv", index=False)
    (output_dir / "best_parameters.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n"
    )

    import xgboost as xgb

    cache_path = output_dir / "workflow_cache.json"
    try:
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        cache = {}
    if cache.get("version") != 1:
        cache = {"version": 1, "stages": {}}
    cache.setdefault("stages", {})

    x_columns = features + ["selector_code"]
    development_fingerprint = _frame_fingerprint(
        development,
        ["query_id", "comment_id", "selector", "selected"] + x_columns,
        sort_by=["query_id", "comment_id"],
    )
    test_feature_fingerprint = _frame_fingerprint(
        test,
        ["query_id", "comment_id", "selector"] + x_columns,
        sort_by=["query_id", "comment_id"],
    )
    test_label_fingerprint = _frame_fingerprint(
        test,
        ["query_id", "comment_id", "selector", "n_picks", "selected"],
        sort_by=["query_id", "comment_id"],
    )
    draw_columns = sorted(
        column for column in choice.columns if column.startswith("audience_selected_draw_")
    )
    tie_source = choice[choice["story_id"].isin(test_stories)]
    tie_label_fingerprint = _frame_fingerprint(
        tie_source,
        ["story_id", "comment_id", "curator_selected", "n_picks"] + draw_columns,
        sort_by=["story_id", "comment_id"],
    )
    model_signature = _cache_signature(
        {
            "stage": "development_model",
            "scope": scope,
            "seed": seed,
            "device": _xgb_device(device),
            "features": x_columns,
            "best_parameters": best,
            "development_fingerprint": development_fingerprint,
            "split_fingerprint": split_fingerprint,
            "xgboost": xgb.__version__,
        }
    )
    predictions_signature = _cache_signature(
        {
            "stage": "test_predictions",
            "model_signature": model_signature,
            "test_feature_fingerprint": test_feature_fingerprint,
            "test_label_fingerprint": test_label_fingerprint,
        }
    )
    metrics_signature = _cache_signature(
        {
            "stage": "test_metrics",
            "predictions_signature": predictions_signature,
            "test_label_fingerprint": test_label_fingerprint,
            "bootstrap_draws": bootstrap_draws,
            "seed": seed,
        }
    )
    permutation_signature = _cache_signature(
        {
            "stage": "test_grouped_permutation",
            "model_signature": model_signature,
            "test_feature_fingerprint": test_feature_fingerprint,
            "test_label_fingerprint": test_label_fingerprint,
            "repeats": permutation_repeats,
            "seed": seed,
        }
    )
    shap_signature = _cache_signature(
        {
            "stage": "test_treeshap",
            "model_signature": model_signature,
            "test_feature_fingerprint": test_feature_fingerprint,
            "sample_rows": min(50_000, len(test)),
            "seed": seed,
            "xgboost": xgb.__version__,
        }
    )
    tie_signature = _cache_signature(
        {
            "stage": "test_tie_sensitivity",
            "predictions_signature": predictions_signature,
            "tie_label_fingerprint": tie_label_fingerprint,
            "draw_columns": draw_columns,
        }
    )

    model_path = output_dir / "development_model.json"
    scores_long_path = output_dir / "test_scores_long.parquet"
    scores_wide_path = output_dir / "test_scores_wide.parquet"
    article_metrics_path = output_dir / "test_article_metrics.parquet"
    metric_summary_path = output_dir / "test_metric_summary.parquet"
    permutation_path = output_dir / "test_grouped_permutation_importance.parquet"
    shap_path = output_dir / "test_treeshap_sample.parquet"
    tie_path = output_dir / "test_tie_sensitivity_metrics.parquet"
    stage_specifications = {
        "development_model": (model_signature, [model_path]),
        "test_predictions": (
            predictions_signature,
            [scores_long_path, scores_wide_path],
        ),
        "test_metrics": (
            metrics_signature,
            [article_metrics_path, metric_summary_path],
        ),
        "test_grouped_permutation": (
            permutation_signature,
            [permutation_path],
        ),
        "test_treeshap": (shap_signature, [shap_path]),
        "test_tie_sensitivity": (tie_signature, [tie_path]),
    }

    def record_stage(stage: str) -> None:
        signature, artifacts = stage_specifications[stage]
        cache["stages"][stage] = {
            "signature": signature,
            "artifacts": [path.name for path in artifacts],
            "artifact_sha256": {
                path.name: _file_sha256(path) for path in artifacts
            },
        }
        _write_json_atomic(cache, cache_path)

    # Adopt artifacts from the immediately preceding workflow version once,
    # but only after checking its fingerprints, parameters, settings, and shapes.
    legacy_manifest_path = output_dir / "model_manifest.json"
    if (
        not force_recompute
        and not cache["stages"]
        and legacy_manifest_path.exists()
    ):
        try:
            legacy = json.loads(legacy_manifest_path.read_text())
            legacy_ok = (
                legacy.get("scope") == scope
                and legacy.get("seed") == seed
                and legacy.get("features") == x_columns
                and legacy.get("split_fingerprint") == split_fingerprint
                and legacy.get("choice_set_fingerprint") == choice_fingerprint
                and legacy.get("best_parameters") == best
                and legacy.get("environment", {}).get("xgboost") == xgb.__version__
            )
            if legacy_ok:
                legacy_predictions = pd.read_parquet(scores_long_path)
                legacy_wide = pd.read_parquet(scores_wide_path)
                legacy_article_metrics = pd.read_parquet(article_metrics_path)
                legacy_metric_summary = pd.read_parquet(metric_summary_path)
                legacy_permutation = pd.read_parquet(permutation_path)
                legacy_shap = pd.read_parquet(shap_path)
                legacy_ties = pd.read_parquet(tie_path)
                legacy_ok = (
                    model_path.exists()
                    and len(legacy_predictions) == len(test)
                    and len(legacy_wide) * 2 == len(test)
                    and len(legacy_article_metrics) == 2 * len(test_stories)
                    and set(legacy_metric_summary["bootstrap_draws"]) == {bootstrap_draws}
                    and set(legacy_permutation["feature"]) == set(features)
                    and int(legacy_permutation["repeat"].max()) == permutation_repeats
                    and len(legacy_shap) == min(50_000, len(test))
                    and set(x_columns).issubset(legacy_shap.columns)
                    and int(legacy_ties["audience_tie_draw"].nunique()) == len(draw_columns)
                )
            if legacy_ok:
                for stage in stage_specifications:
                    signature, artifacts = stage_specifications[stage]
                    cache["stages"][stage] = {
                        "signature": signature,
                        "artifacts": [path.name for path in artifacts],
                        "artifact_sha256": {
                            path.name: _file_sha256(path) for path in artifacts
                        },
                    }
                _write_json_atomic(cache, cache_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    cache_upgraded = False
    for stage, (signature, artifacts) in stage_specifications.items():
        recorded = cache["stages"].get(stage, {})
        if (
            recorded.get("signature") == signature
            and not recorded.get("artifact_sha256")
            and all(path.exists() for path in artifacts)
        ):
            recorded["artifact_sha256"] = {
                path.name: _file_sha256(path) for path in artifacts
            }
            recorded["artifacts"] = [path.name for path in artifacts]
            cache["stages"][stage] = recorded
            cache_upgraded = True
    if cache_upgraded:
        _write_json_atomic(cache, cache_path)

    cache_report: dict[str, str] = {}

    model_reused = _cache_stage_valid(
        cache,
        "development_model",
        model_signature,
        [model_path],
        force_recompute=force_recompute,
    )
    if model_reused:
        try:
            model = make_ranker(
                best,
                device=device,
                seed=seed,
                early_stopping_rounds=None,
            )
            model.load_model(model_path)
        except (OSError, ValueError, xgb.core.XGBoostError):
            model_reused = False
    if not model_reused:
        model = _fit_ranker(
            development,
            None,
            features,
            best,
            device=device,
            seed=seed,
        )
        model.save_model(model_path)
        record_stage("development_model")
    cache_report["development_model"] = "reused" if model_reused else "computed"

    predictions_reused = _cache_stage_valid(
        cache,
        "test_predictions",
        predictions_signature,
        [scores_long_path, scores_wide_path],
        force_recompute=force_recompute,
    )
    predictions_base = test[
        ["story_id", "comment_id", "selector", "query_id", "n_picks", "selected"]
    ].copy()
    if predictions_reused:
        try:
            stored_scores = pd.read_parquet(scores_long_path)[
                ["story_id", "comment_id", "selector", "score"]
            ]
            predictions = predictions_base.merge(
                stored_scores,
                on=["story_id", "comment_id", "selector"],
                how="left",
                validate="one_to_one",
                sort=False,
            )
            if predictions["score"].isna().any() or len(predictions) != len(test):
                raise ValueError("Cached test scores do not cover the current test rows")
        except (OSError, ValueError, KeyError):
            predictions_reused = False
    if not predictions_reused:
        predictions = predictions_base
        predictions["score"] = _predict_cpu(model, test[x_columns])
        predictions["split_role"] = "paper2_test"
        predictions["score_source"] = "sealed_paper2_test"
        _atomic_parquet(predictions, scores_long_path)
        pivot = predictions.pivot(
            index=["story_id", "comment_id"],
            columns="selector",
            values="score",
        ).reset_index()
        pivot = pivot.rename(
            columns={
                "audience": "xgb_audience_score",
                "curator": "xgb_curator_score",
            }
        )
        pivot["split_role"] = "paper2_test"
        pivot["score_source"] = "sealed_paper2_test"
        _atomic_parquet(pivot, scores_wide_path)
        record_stage("test_predictions")
    else:
        predictions["split_role"] = "paper2_test"
        predictions["score_source"] = "sealed_paper2_test"
    scores = predictions["score"].to_numpy()
    cache_report["test_predictions"] = "reused" if predictions_reused else "computed"

    metrics_reused = _cache_stage_valid(
        cache,
        "test_metrics",
        metrics_signature,
        [article_metrics_path, metric_summary_path],
        force_recompute=force_recompute,
    )
    if metrics_reused:
        try:
            article_metrics = pd.read_parquet(article_metrics_path)
            metric_summary = pd.read_parquet(metric_summary_path)
        except (OSError, ValueError, KeyError):
            metrics_reused = False
    if not metrics_reused:
        article_metrics = evaluate_rank_scores(test, scores)
        article_metrics["split_role"] = "paper2_test"
        metric_summary = bootstrap_metric_summary(
            article_metrics,
            draws=bootstrap_draws,
            seed=seed,
        )
        _atomic_parquet(article_metrics, article_metrics_path)
        _atomic_parquet(metric_summary, metric_summary_path)
        record_stage("test_metrics")
    cache_report["test_metrics"] = "reused" if metrics_reused else "computed"

    permutation_reused = _cache_stage_valid(
        cache,
        "test_grouped_permutation",
        permutation_signature,
        [permutation_path],
        force_recompute=force_recompute,
    )
    if permutation_reused:
        try:
            permutation = pd.read_parquet(permutation_path)
        except (OSError, ValueError, KeyError):
            permutation_reused = False
    if not permutation_reused:
        permutation = grouped_permutation_importance(
            model,
            test,
            features,
            repeats=permutation_repeats,
            seed=seed,
        )
        permutation["split_role"] = "paper2_test"
        _atomic_parquet(permutation, permutation_path)
        record_stage("test_grouped_permutation")
    cache_report["test_grouped_permutation"] = (
        "reused" if permutation_reused else "computed"
    )

    shap_reused = _cache_stage_valid(
        cache,
        "test_treeshap",
        shap_signature,
        [shap_path],
        force_recompute=force_recompute,
    )
    if shap_reused:
        try:
            shap_values = pd.read_parquet(shap_path)
        except (OSError, ValueError, KeyError):
            shap_reused = False
    if not shap_reused:
        sample = test.assign(
            _hash=(
                test["story_id"].astype(str)
                + "::"
                + test["comment_id"].astype(str)
                + "::"
                + test["selector"].astype(str)
            ).map(lambda value: _stable_fraction(value, seed))
        )
        sample = sample.nsmallest(min(50_000, len(sample)), "_hash")
        model.set_params(device=_xgb_device(device))
        contributions = model.get_booster().predict(
            xgb.DMatrix(sample[x_columns]),
            pred_contribs=True,
        )
        shap_values = pd.DataFrame(
            contributions[:, :-1],
            columns=x_columns,
        )
        shap_values.insert(0, "selector", sample["selector"].to_numpy())
        shap_values.insert(0, "comment_id", sample["comment_id"].to_numpy())
        shap_values.insert(0, "story_id", sample["story_id"].to_numpy())
        shap_values["split_role"] = "paper2_test"
        _atomic_parquet(shap_values, shap_path)
        record_stage("test_treeshap")
    cache_report["test_treeshap"] = "reused" if shap_reused else "computed"

    tie_reused = _cache_stage_valid(
        cache,
        "test_tie_sensitivity",
        tie_signature,
        [tie_path],
        force_recompute=force_recompute,
    )
    if tie_reused:
        try:
            tie_metrics_frame = pd.read_parquet(tie_path)
        except (OSError, ValueError, KeyError):
            tie_reused = False
    if not tie_reused:
        tie_metrics: list[pd.DataFrame] = []
        for draw_number, _ in enumerate(draw_columns, start=1):
            draw_stacked = build_stacked_rank_data(
                tie_source,
                features,
                audience_draw=draw_number,
            )
            scored = draw_stacked.merge(
                predictions[["story_id", "comment_id", "selector", "score"]],
                on=["story_id", "comment_id", "selector"],
                validate="one_to_one",
            )
            metrics = evaluate_rank_scores(scored, scored["score"].to_numpy())
            metrics["audience_tie_draw"] = draw_number
            metrics["split_role"] = "paper2_test"
            tie_metrics.append(metrics)
        tie_metrics_frame = pd.concat(tie_metrics, ignore_index=True)
        _atomic_parquet(tie_metrics_frame, tie_path)
        record_stage("test_tie_sensitivity")
    cache_report["test_tie_sensitivity"] = "reused" if tie_reused else "computed"

    scope_split = split.copy()
    scope_split["n_candidates"] = scope_split[f"n_candidates_{scope}"]
    scope_split["n_picks"] = scope_split[f"n_picks_{scope}"]
    _atomic_parquet(scope_split, output_dir / "article_split.parquet")

    manifest = {
        "scope": scope,
        "seed": seed,
        "device": _xgb_device(device),
        "features": features + ["selector_code"],
        "objective": "rank:ndcg",
        "pair_method": "topk",
        "reported_scores": "sealed_paper2_test",
        "model_artifact": "development_model.json",
        "development_articles": len(development_stories),
        "paper2_test_articles": len(test_stories),
        "excluded_scope_only_articles": excluded_articles,
        "development_folds": development_folds,
        "hyperparameter_search": {
            "strategy": search_strategy,
            "broad_random_configurations": (
                broad_search_count if configs is None else 0
            ),
            "narrow_grid_configurations": (
                len(narrow_configs) if configs is None else 0
            ),
            "refinement_top_broad_configurations": (
                refinement_top_configs if configs is None else 0
            ),
            "selection_metric": "mean_macro_ndcg_at_k",
            "broad_distributions": {
                "max_depth": "discrete_uniform_[3,9]",
                "learning_rate": "log_uniform_[0.015,0.12]",
                "min_child_weight": "log_uniform_[3,50]",
                "subsample": "uniform_[0.65,1.0]",
                "colsample_bytree": "uniform_[0.55,1.0]",
                "reg_lambda": "log_uniform_[0.5,30]",
                "reg_alpha": "30%_zero_else_log_uniform_[0.001,3]",
                "gamma": "30%_zero_else_log_uniform_[0.001,3]",
            },
            "narrow_grid_axes": [
                "max_depth_center_plus_or_minus_1",
                "learning_rate_center_times_[0.75,1,1.25]",
                "min_child_weight_center_times_[0.75,1,1.25]",
            ],
            "test_used_for_selection": False,
        },
        "split_strategy": "shared_month_by_joint_candidate_size_tercile_50_50",
        "split_fingerprint": split_fingerprint,
        "best_parameters": best,
        "choice_set_fingerprint": choice_fingerprint,
        "cache": {
            "metadata_artifact": "workflow_cache.json",
            "force_recompute": force_recompute,
            "stage_status": cache_report,
            "development_fingerprint": development_fingerprint,
            "test_feature_fingerprint": test_feature_fingerprint,
            "test_label_fingerprint": test_label_fingerprint,
            "tie_label_fingerprint": tie_label_fingerprint,
        },
        "preprocessing_note": (
            "The article split is sealed for ranker fitting and selection. "
            "Length adjustments and novelty imputation were created upstream "
            "before this split; refit them on development data before claiming "
            "a fully sealed preprocessing pipeline."
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "xgboost": __import__("xgboost").__version__,
        },
    }
    (output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
