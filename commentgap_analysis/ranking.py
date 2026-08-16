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
) -> pd.DataFrame:
    label = f"audience_selected_draw_{audience_draw:02d}"
    required = {"story_id", "comment_id", "curator_selected", label, "n_picks"} | set(features)
    missing = required - set(choice_set.columns)
    if missing:
        raise ValueError(f"Missing stacked-ranker columns: {sorted(missing)}")
    common = ["story_id", "comment_id", "n_picks"] + features
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


def assign_article_splits(
    choice_set: pd.DataFrame,
    *,
    seed: int = RANKER_SEED,
    tuning_fraction: float = 0.15,
    outer_folds: int = 5,
) -> pd.DataFrame:
    if not 0 < tuning_fraction < 1:
        raise ValueError("tuning_fraction must lie between zero and one")
    split_source = choice_set.copy()
    split_source["sticky_reply"] = (
        split_source["curator_selected"].astype(bool)
        & split_source.get("is_reply", pd.Series(False, index=split_source.index)).astype(bool)
    )
    article = split_source.groupby("story_id", as_index=False).agg(
        article_month=("article_month", "first"),
        n_candidates=("n_candidates", "first"),
        n_picks=("n_picks", "first"),
        has_sticky_reply=("sticky_reply", "any"),
    )
    # The all-comment table does not retain is_reply for every workflow version;
    # size and k still provide deterministic, balanced strata.
    try:
        article["size_band"] = pd.qcut(
            article["n_candidates"].rank(method="first"), 4, labels=False, duplicates="drop"
        ).astype(int)
    except ValueError:
        article["size_band"] = 0
    article["pick_band"] = np.minimum(article["n_picks"].astype(int), 4)
    article["stratum"] = (
        article["article_month"].astype(str)
        + "|"
        + article["size_band"].astype(str)
        + "|"
        + article["pick_band"].astype(str)
        + "|"
        + article["has_sticky_reply"].astype(int).astype(str)
    )
    article["hash"] = article["story_id"].astype(str).map(lambda x: _stable_fraction(x, seed))
    article["is_tuning"] = False
    article["outer_fold"] = -1
    for _, indices in article.groupby("stratum", sort=True).groups.items():
        ordered = article.loc[indices].sort_values(["hash", "story_id"]).index.to_list()
        tuning_n = int(round(len(ordered) * tuning_fraction))
        if len(ordered) >= 4:
            tuning_n = max(1, tuning_n)
        tuning_n = min(tuning_n, max(0, len(ordered) - 1))
        tuning = ordered[:tuning_n]
        analysis = ordered[tuning_n:]
        article.loc[tuning, "is_tuning"] = True
        for position, idx in enumerate(analysis):
            offset = int(article.at[idx, "hash"] * outer_folds)
            article.at[idx, "outer_fold"] = (position + offset) % outer_folds
    target_tuning = max(3, int(round(len(article) * tuning_fraction)))
    target_tuning = min(target_tuning, max(0, len(article) - outer_folds))
    current_tuning = int(article["is_tuning"].sum())
    if current_tuning < target_tuning:
        add = (
            article.loc[~article["is_tuning"]]
            .sort_values(["hash", "story_id"])
            .head(target_tuning - current_tuning)
            .index
        )
        article.loc[add, "is_tuning"] = True
        article.loc[add, "outer_fold"] = -1
    elif current_tuning > target_tuning:
        remove = (
            article.loc[article["is_tuning"]]
            .sort_values(["hash", "story_id"], ascending=[False, True])
            .head(current_tuning - target_tuning)
            .index
        )
        article.loc[remove, "is_tuning"] = False
    # Rebalance analysis stories globally so every fold is populated and sizes
    # differ by at most one. Stratification remains encoded in the ordering.
    analysis_indices = article.index[~article["is_tuning"]].to_list()
    ordered = article.loc[analysis_indices].sort_values(["stratum", "hash", "story_id"]).index
    article.loc[ordered, "outer_fold"] = np.arange(len(ordered)) % outer_folds
    return article.drop(columns=["hash", "stratum", "has_sticky_reply"])


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


def default_search_configs() -> list[dict[str, Any]]:
    return [
        {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10},
        {"max_depth": 4, "learning_rate": 0.06, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 0.8, "reg_lambda": 1},
        {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 10},
        {"max_depth": 6, "learning_rate": 0.06, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 1},
        {"max_depth": 8, "learning_rate": 0.03, "min_child_weight": 20, "subsample": 1.0, "colsample_bytree": 0.7, "reg_lambda": 10},
        {"max_depth": 8, "learning_rate": 0.06, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10},
        {"max_depth": 5, "learning_rate": 0.04, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 5},
        {"max_depth": 7, "learning_rate": 0.04, "min_child_weight": 20, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 5},
    ]


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
    baseline_scores = model.predict(validation[x_columns])
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
            score = model.predict(permuted)
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
    tuning_frame: pd.DataFrame,
    features: list[str],
    *,
    configs: list[dict[str, Any]] | None = None,
    folds: int = 3,
    device: str = "auto",
    seed: int = RANKER_SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    configs = configs or default_search_configs()
    stories = sorted(tuning_frame["story_id"].astype(str).unique())
    if len(stories) < folds:
        raise ValueError("Too few tuning articles for grouped cross-validation")
    fold_by_story = {
        story: int(_stable_fraction(story, seed + 17) * folds) % folds for story in stories
    }
    # Guarantee every fold is populated.
    for position, story in enumerate(stories[:folds]):
        fold_by_story[story] = position
    records: list[dict[str, Any]] = []
    for config_index, parameters in enumerate(configs):
        for fold in range(folds):
            validation_stories = {story for story, assigned in fold_by_story.items() if assigned == fold}
            train = tuning_frame[~tuning_frame["story_id"].astype(str).isin(validation_stories)]
            validation = tuning_frame[tuning_frame["story_id"].astype(str).isin(validation_stories)]
            model = _fit_ranker(
                train,
                validation,
                features,
                parameters,
                device=device,
                seed=seed + config_index * 10 + fold,
            )
            scores = model.predict(validation[features + ["selector_code"]])
            metrics = evaluate_rank_scores(validation, scores)
            records.append(
                {
                    "config_index": config_index,
                    "fold": fold,
                    "macro_ndcg_at_k": float(metrics["ndcg_at_k"].mean()),
                    "best_iteration": int(getattr(model, "best_iteration", 0)),
                    "parameters": json.dumps(parameters, sort_keys=True),
                }
            )
    history = pd.DataFrame(records)
    scores = history.groupby("config_index")["macro_ndcg_at_k"].mean()
    best_index = int(scores.idxmax())
    best = dict(configs[best_index])
    best_iterations = history.loc[history["config_index"] == best_index, "best_iteration"]
    best["n_estimators"] = max(50, int(np.median(best_iterations)) + 1)
    return best, history


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd")
    os.replace(temporary, path)


def run_ranker_workflow(
    choice_set: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    *,
    scope: str,
    device: str = "auto",
    seed: int = RANKER_SEED,
    bootstrap_draws: int = 1_000,
) -> dict[str, Any]:
    output_dir = Path(output_dir) / scope
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_columns = [
        "story_id",
        "comment_id",
        "curator_selected",
        "audience_selected_draw_01",
    ] + features
    choice_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(
            choice_set[fingerprint_columns], index=False
        ).to_numpy().tobytes()
    ).hexdigest()
    splits = assign_article_splits(choice_set, seed=seed)
    stacked = build_stacked_rank_data(choice_set, features)
    stacked = stacked.merge(
        splits[["story_id", "is_tuning", "outer_fold"]], on="story_id", validate="many_to_one"
    )
    tuning = stacked[stacked["is_tuning"]].copy()
    analysis = stacked[~stacked["is_tuning"]].copy()
    if tuning["story_id"].nunique() < 3:
        raise ValueError("The tuning set contains fewer than three articles")
    best, history = tune_ranker(tuning, features, device=device, seed=seed)
    history.to_csv(output_dir / "tuning_history.csv", index=False)
    (output_dir / "best_parameters.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n"
    )
    oof_parts: list[pd.DataFrame] = []
    metrics_parts: list[pd.DataFrame] = []
    shap_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    for fold in sorted(analysis["outer_fold"].unique()):
        train = analysis[analysis["outer_fold"] != fold]
        validation = analysis[analysis["outer_fold"] == fold].copy()
        # Hyperparameters and tree count were frozen using only the separate
        # tuning articles. The OOF fold is never used for early stopping.
        model = _fit_ranker(
            train,
            None,
            features,
            best,
            device=device,
            seed=seed + int(fold),
        )
        model_path = output_dir / f"fold_{int(fold)}.json"
        model.save_model(model_path)
        x_validation = validation[features + ["selector_code"]]
        scores = model.predict(x_validation)
        predictions = validation[
            ["story_id", "comment_id", "selector", "query_id", "n_picks", "selected"]
        ].copy()
        predictions["score"] = scores
        predictions["outer_fold"] = int(fold)
        predictions["score_source"] = "article_grouped_oof"
        oof_parts.append(predictions)
        metrics = evaluate_rank_scores(validation, scores)
        metrics["outer_fold"] = int(fold)
        metrics_parts.append(metrics)
        importance = grouped_permutation_importance(
            model,
            validation,
            features,
            repeats=3,
            seed=seed + int(fold) * 100,
        )
        importance["outer_fold"] = int(fold)
        importance_parts.append(importance)
        # TreeSHAP on a deterministic cap keeps output tractable.
        sample = validation.assign(_hash=validation["comment_id"].astype(str).map(lambda x: _stable_fraction(x, seed + fold)))
        sample = sample.nsmallest(min(10_000, len(sample)), "_hash")
        import xgboost as xgb

        contributions = model.get_booster().predict(
            xgb.DMatrix(sample[features + ["selector_code"]]), pred_contribs=True
        )
        shap = pd.DataFrame(contributions[:, :-1], columns=features + ["selector_code"])
        shap.insert(0, "selector", sample["selector"].to_numpy())
        shap.insert(0, "comment_id", sample["comment_id"].to_numpy())
        shap.insert(0, "story_id", sample["story_id"].to_numpy())
        shap["outer_fold"] = int(fold)
        shap_parts.append(shap)
    oof = pd.concat(oof_parts, ignore_index=True)
    article_metrics = pd.concat(metrics_parts, ignore_index=True)
    metric_summary = bootstrap_metric_summary(
        article_metrics, draws=bootstrap_draws, seed=seed
    )
    shap_values = pd.concat(shap_parts, ignore_index=True)
    permutation = pd.concat(importance_parts, ignore_index=True)
    _atomic_parquet(splits, output_dir / "article_splits.parquet")
    _atomic_parquet(oof, output_dir / "oof_scores_long.parquet")
    _atomic_parquet(article_metrics, output_dir / "article_metrics.parquet")
    _atomic_parquet(metric_summary, output_dir / "metric_summary.parquet")
    _atomic_parquet(shap_values, output_dir / "oof_treeshap_sample.parquet")
    _atomic_parquet(permutation, output_dir / "grouped_permutation_importance.parquet")
    pivot = oof.pivot(
        index=["story_id", "comment_id", "outer_fold"], columns="selector", values="score"
    ).reset_index()
    pivot = pivot.rename(columns={"audience": "xgb_audience_score", "curator": "xgb_curator_score"})
    pivot["score_source"] = "article_grouped_oof"
    _atomic_parquet(pivot, output_dir / "oof_scores_wide.parquet")

    # Re-evaluate the fixed OOF scores against every deterministic audience tie
    # realization. Curator labels are unchanged but are retained for a complete
    # selector-by-draw diagnostic.
    tie_metrics: list[pd.DataFrame] = []
    draw_columns = sorted(
        column for column in choice_set.columns if column.startswith("audience_selected_draw_")
    )
    analysis_stories = set(analysis["story_id"].astype(str))
    for draw_number, _ in enumerate(draw_columns, start=1):
        draw_stacked = build_stacked_rank_data(
            choice_set[choice_set["story_id"].astype(str).isin(analysis_stories)],
            features,
            audience_draw=draw_number,
        )
        scored = draw_stacked.merge(
            oof[["story_id", "comment_id", "selector", "score"]],
            on=["story_id", "comment_id", "selector"],
            validate="one_to_one",
        )
        metrics = evaluate_rank_scores(scored, scored["score"].to_numpy())
        metrics["audience_tie_draw"] = draw_number
        tie_metrics.append(metrics)
    _atomic_parquet(
        pd.concat(tie_metrics, ignore_index=True),
        output_dir / "tie_sensitivity_metrics.parquet",
    )

    final_model = _fit_ranker(
        stacked,
        None,
        features,
        best,
        device=device,
        seed=seed,
    )
    final_model.save_model(output_dir / "final_deployable_model.json")
    manifest = {
        "scope": scope,
        "seed": seed,
        "device": _xgb_device(device),
        "features": features + ["selector_code"],
        "objective": "rank:ndcg",
        "pair_method": "topk",
        "reported_scores": "article_grouped_oof_non_tuning_articles",
        "tuning_articles": int(splits["is_tuning"].sum()),
        "analysis_articles": int((~splits["is_tuning"]).sum()),
        "best_parameters": best,
        "choice_set_fingerprint": choice_fingerprint,
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
