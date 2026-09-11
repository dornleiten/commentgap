"""Paper 1 tables and figures from frozen factorial winners and stage 7."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from commentgap_analysis.comment_gap import weighted_median
from commentgap_analysis.factorial_winners import assert_factorial_idle
from commentgap_analysis.presentation_labels import model_prefix, model_selector_label
from commentgap_analysis.paper1_plotting import (
    _largest_regression_coefficient_rows,
    plot_regression_selector_coefficients,
    plot_regression_selector_differences,
    plot_regression_vs_shap_gaps,
    plot_winner_permutation_importance,
    plot_winner_permutation_importance_gaps,
    plot_winner_shap_importance,
    plot_winner_shap_importance_gaps,
)


METRICS = ("ndcg_at_k", "top_k_overlap", "jaccard", "mean_selected_rank")
MODEL_ORDER = ("conditional_logit", "xgboost", "neural")
PERMUTATION_CACHE_VERSION = 1


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _normalise_scopes(scopes: Iterable[str]) -> tuple[str, ...]:
    resolved = tuple(dict.fromkeys(scopes))
    if not resolved or not set(resolved).issubset({"all", "root"}):
        raise ValueError("scopes must contain all and/or root")
    return resolved


def _require_hash(path: Path, expected: str | None, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if expected and actual != expected:
        raise RuntimeError(
            f"{label} changed after winner freezing: expected {expected}, got {actual}"
        )


def _winner_label(family: str, feature_set: str | None = None) -> str:
    """Return the short presentation label for a model pair."""
    return model_prefix(family, feature_set)


def model_implied_comment_gap(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the curator-top-k versus audience-rank gap by article.

    Curator-score ties at the kth boundary receive equal fractional selection
    weight. Audience-score ties receive equal midranks, making the result
    invariant to row and comment-id ordering.
    """
    required = {
        "story_id", "comment_id", "n_picks", "audience_score", "curator_score"
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Model scores are missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for story_id, group in scores.groupby("story_id", sort=False):
        group = group.copy()
        if group["comment_id"].duplicated().any():
            raise ValueError(f"Duplicate candidate comments for story {story_id}")
        audience = pd.to_numeric(group["audience_score"], errors="coerce")
        curator = pd.to_numeric(group["curator_score"], errors="coerce")
        if audience.isna().any() or curator.isna().any():
            raise ValueError(f"Non-finite model scores for story {story_id}")
        n_candidates = len(group)
        picks = pd.to_numeric(group["n_picks"], errors="raise").astype(int)
        if picks.nunique() != 1:
            raise ValueError(f"n_picks is not constant for story {story_id}")
        n_picks = int(picks.iloc[0])
        if not 0 < n_picks < n_candidates:
            raise ValueError(
                f"Invalid candidate/selection counts for story {story_id}: "
                f"N={n_candidates}, k={n_picks}"
            )

        audience_midrank = audience.rank(method="average", ascending=False)
        cutoff = float(curator.sort_values(ascending=False).iloc[n_picks - 1])
        above = curator > cutoff
        boundary = curator == cutoff
        remaining = n_picks - int(above.sum())
        boundary_count = int(boundary.sum())
        if remaining < 0 or remaining > boundary_count or boundary_count < 1:
            raise RuntimeError(f"Could not resolve curator cutoff tie for story {story_id}")
        selection_weight = above.astype(float)
        selection_weight.loc[boundary] = remaining / boundary_count
        if not np.isclose(selection_weight.sum(), n_picks):
            raise RuntimeError(f"Curator selection weights do not sum to k for {story_id}")

        mean_rank = float(np.average(audience_midrank, weights=selection_weight))
        best_mean_rank = (n_picks + 1) / 2
        gap = (mean_rank - best_mean_rank) / (n_candidates - n_picks)
        if gap < -1e-12 or gap > 1 + 1e-12:
            raise ValueError(f"Model-implied gap outside [0, 1] for {story_id}: {gap}")
        rows.append(
            {
                "story_id": str(story_id),
                "n_candidates": n_candidates,
                "n_picks": n_picks,
                "mean_curator_model_audience_rank": mean_rank,
                "best_possible_mean_rank": best_mean_rank,
                "model_implied_gap": float(np.clip(gap, 0, 1)),
                "curator_cutoff_tie_size": boundary_count,
                "curator_cutoff_fraction": remaining / boundary_count,
                "has_audience_score_tie": bool(audience.nunique() < n_candidates),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_group_mean(
    frame: pd.DataFrame,
    *,
    groups: list[str],
    value: str,
    draws: int,
    seed: int,
    estimate_name: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = groups[0] if len(groups) == 1 else groups
    for keys, subset in frame.groupby(grouper, dropna=False, observed=True):
        keys = (keys,) if len(groups) == 1 else tuple(keys)
        values = pd.to_numeric(subset[value], errors="raise").to_numpy(float)
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        bootstrap = values[indices].mean(axis=1)
        rows.append(
            {
                **dict(zip(groups, keys)),
                estimate_name: float(values.mean()),
                "conf_low": float(np.quantile(bootstrap, 0.025)),
                "conf_high": float(np.quantile(bootstrap, 0.975)),
                "n_articles": len(values),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(rows)


def summarize_model_implied_gaps(
    article_gaps: pd.DataFrame, *, draws: int, seed: int
) -> pd.DataFrame:
    groups = ["scope", "model_family", "model_id", "model_label"]
    ordinary = _bootstrap_group_mean(
        article_gaps,
        groups=groups,
        value="model_implied_gap",
        draws=draws,
        seed=seed,
        estimate_name="gap_mean",
    )
    rng = np.random.default_rng(seed + 1)
    rows: list[dict[str, Any]] = []
    for keys, subset in article_gaps.groupby(groups, dropna=False, observed=True):
        gaps = pd.to_numeric(subset["model_implied_gap"], errors="raise").to_numpy(float)
        weights = pd.to_numeric(subset["n_candidates"], errors="raise").to_numpy(float)
        if (weights <= 0).any():
            raise ValueError("Model-implied gap weights must be positive")
        indices = rng.integers(0, len(gaps), size=(draws, len(gaps)))
        boot_gaps = gaps[indices]
        boot_weights = weights[indices]
        bootstrap = (boot_gaps * boot_weights).sum(axis=1) / boot_weights.sum(axis=1)
        bootstrap_median = np.asarray(
            [
                weighted_median(sample_gaps, sample_weights)
                for sample_gaps, sample_weights in zip(
                    boot_gaps, boot_weights, strict=True
                )
            ]
        )
        rows.append(
            {
                **dict(zip(groups, tuple(keys))),
                "gap_comment_weighted_mean": float(np.average(gaps, weights=weights)),
                "weighted_conf_low": float(np.quantile(bootstrap, 0.025)),
                "weighted_conf_high": float(np.quantile(bootstrap, 0.975)),
                "gap_comment_weighted_median": weighted_median(gaps, weights),
                "weighted_median_conf_low": float(
                    np.quantile(bootstrap_median, 0.025)
                ),
                "weighted_median_conf_high": float(
                    np.quantile(bootstrap_median, 0.975)
                ),
                "candidate_comments": int(weights.sum()),
            }
        )
    return ordinary.merge(pd.DataFrame(rows), on=groups, validate="one_to_one")


def regression_feature_gaps(associations: pd.DataFrame) -> pd.DataFrame:
    """Expose the signed curator-minus-audience interaction as a feature gap."""
    required = {
        "scope", "term", "feature", "curator_minus_audience_log_odds",
        "difference_conf_low", "difference_conf_high",
    }
    missing = required - set(associations.columns)
    if missing:
        raise ValueError(f"Regression associations are missing: {sorted(missing)}")
    output = associations.copy()
    output["feature_gap_log_odds"] = output["curator_minus_audience_log_odds"]
    output["feature_gap_conf_low"] = output["difference_conf_low"]
    output["feature_gap_conf_high"] = output["difference_conf_high"]
    output["curator_to_audience_odds_ratio"] = np.exp(output["feature_gap_log_odds"])
    return output


def summarize_permutation_importance_gaps(
    article_importance: pd.DataFrame, *, draws: int, seed: int
) -> pd.DataFrame:
    """Summarize paired curator-minus-audience permutation losses by article."""
    required = {
        "story_id", "scope", "model_family", "model_id", "model_label",
        "feature", "repeat", "selector", "baseline_ndcg_at_k",
        "permuted_ndcg_at_k",
    }
    missing = required - set(article_importance.columns)
    if missing:
        raise ValueError(f"Permutation importance is missing: {sorted(missing)}")
    working = article_importance.copy()
    working["importance"] = (
        working["baseline_ndcg_at_k"] - working["permuted_ndcg_at_k"]
    )
    keys = [
        "story_id", "scope", "model_family", "model_id", "model_label",
        "feature", "repeat",
    ]
    wide = working.pivot(index=keys, columns="selector", values="importance").reset_index()
    if not {"audience", "curator"}.issubset(wide.columns):
        raise ValueError("Permutation importance must contain both selectors")
    article = wide.groupby(keys[:-1], as_index=False, observed=True).agg(
        audience_importance=("audience", "mean"),
        curator_importance=("curator", "mean"),
        permutation_repeats=("repeat", "nunique"),
    )
    article["permutation_importance_gap"] = (
        article["curator_importance"] - article["audience_importance"]
    )
    summary_keys = ["scope", "model_family", "model_id", "model_label", "feature"]
    gap = _bootstrap_group_mean(
        article,
        groups=summary_keys,
        value="permutation_importance_gap",
        draws=draws,
        seed=seed,
        estimate_name="permutation_importance_gap",
    )
    means = article.groupby(summary_keys, as_index=False, observed=True).agg(
        audience_importance=("audience_importance", "mean"),
        curator_importance=("curator_importance", "mean"),
        permutation_repeats=("permutation_repeats", "max"),
    )
    return gap.merge(means, on=summary_keys, validate="one_to_one")


def _selector_article_ndcg(scored: pd.DataFrame) -> pd.DataFrame:
    """Average audience tie draws before returning one row per selector/article."""
    draw_columns = [f"audience_selected_draw_{draw:02d}" for draw in range(1, 11)]
    required = {
        "story_id", "comment_id", "n_picks", "curator_selected",
        "audience_score", "curator_score", *draw_columns,
    }
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Scored candidates are missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for story_id, story in scored.groupby("story_id", sort=False):
        k = int(story["n_picks"].iloc[0])
        discounts = 1 / np.log2(np.arange(2, k + 2))
        ideal = float(discounts.sum())
        audience = story.sort_values(
            ["audience_score", "comment_id"], ascending=[False, True]
        ).iloc[:k]
        audience_values = [
            float((audience[column].to_numpy(dtype=float) * discounts).sum() / ideal)
            for column in draw_columns
        ]
        curator = story.sort_values(
            ["curator_score", "comment_id"], ascending=[False, True]
        ).iloc[:k]
        curator_value = float(
            (curator["curator_selected"].to_numpy(dtype=float) * discounts).sum()
            / ideal
        )
        rows.extend(
            [
                {
                    "story_id": str(story_id),
                    "selector": "audience",
                    "ndcg_at_k": float(np.mean(audience_values)),
                },
                {
                    "story_id": str(story_id),
                    "selector": "curator",
                    "ndcg_at_k": curator_value,
                },
            ]
        )
    return pd.DataFrame(rows)


def _permute_within_article(
    frame: pd.DataFrame, feature: str, rng: np.random.Generator
) -> pd.DataFrame:
    permuted = frame.copy()
    for indices in permuted.groupby("story_id", sort=False).groups.values():
        values = permuted.loc[indices, feature].to_numpy(copy=True)
        permuted.loc[indices, feature] = rng.permutation(values)
    return permuted


def selector_permutation_importance(
    frame: pd.DataFrame,
    features: list[str],
    predictor: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    repeats: int,
    seed: int,
    metadata: dict[str, str],
) -> pd.DataFrame:
    """Calculate paired within-article nDCG permutation losses."""
    if repeats < 1:
        raise ValueError("Permutation repeats must be positive")
    baseline = _selector_article_ndcg(predictor(frame)).rename(
        columns={"ndcg_at_k": "baseline_ndcg_at_k"}
    )
    rows: list[pd.DataFrame] = []
    for feature_index, feature in enumerate(features):
        if feature not in frame:
            raise ValueError(f"Permutation feature is missing: {feature}")
        for repeat in range(1, repeats + 1):
            rng = np.random.default_rng(seed + 10_000 * feature_index + repeat)
            permuted = _permute_within_article(frame, feature, rng)
            metric = _selector_article_ndcg(predictor(permuted)).rename(
                columns={"ndcg_at_k": "permuted_ndcg_at_k"}
            )
            joined = baseline.merge(
                metric,
                on=["story_id", "selector"],
                validate="one_to_one",
            )
            joined["feature"] = feature
            joined["repeat"] = repeat
            for key, value in metadata.items():
                joined[key] = value
            rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def _winner_permutation_components(
    *,
    winner: dict[str, Any],
    factorial_root: Path,
    model_data_root: Path,
    split_role: str = "paper2_test",
) -> tuple[pd.DataFrame, list[str], Callable[[pd.DataFrame], pd.DataFrame]]:
    """Load a frozen winner and return a split frame and score function."""
    from commentgap_analysis.factorial_rankers import (
        _embedding_cache,
        _scored_from_predictions,
        _xgb_arrays,
    )
    from commentgap_analysis.neural_ranking import (
        FeatureScaler,
        NeuralTrainingRecipe,
        _load_model_checkpoint,
        _load_scope_inputs,
        _make_model,
        _source_for_model,
        apply_fold_feature_columns,
        score_candidates,
    )

    family = str(winner["family"])
    scope = str(winner["scope"])
    variant_id = str(winner["variant_id"])
    root = Path(factorial_root) / variant_id / scope
    manifest = json.loads((root / "model_manifest.json").read_text())
    frame, features, _, _, _ = _load_scope_inputs(model_data_root, scope)
    frame = apply_fold_feature_columns(
        frame[frame["split_role"] == split_role].copy(), features, None
    )
    stories = sorted(frame["story_id"].astype(str).unique())

    if family == "xgboost":
        from commentgap_analysis.ranking import make_ranker

        parameters = json.loads((root / "best_parameters.json").read_text())
        model = make_ranker(
            {**parameters, "ndcg_exp_gain": False},
            device="cpu",
            seed=20260813,
            early_stopping_rounds=None,
        )
        model.load_model(root / "development_model.json")
        embedding_matrix = None
        if str(winner.get("feature_set")) == "metadata_bge":
            embedding_store = manifest.get("embedding_store")
            if not embedding_store:
                raise RuntimeError(f"{variant_id}/{scope} lacks its embedding store")
            full_frame, _, _, _, _ = _load_scope_inputs(model_data_root, scope)
            embedding_matrix = _embedding_cache(
                full_frame.reset_index(drop=True),
                scope=scope,
                embedding_store=Path(embedding_store),
                cache_root=Path(factorial_root) / "_cache" / "xgboost_bge",
                progress_every_stories=100,
                run_label=f"stage9-permutation-{variant_id}",
            )

        def predict_xgb(candidate_frame: pd.DataFrame) -> pd.DataFrame:
            arrays = _xgb_arrays(
                candidate_frame,
                features=features,
                draw_policy=str(winner["draw_policy"]),
                embedding_matrix=embedding_matrix,
            )
            predictions = np.asarray(model.predict(arrays[0]))
            return _scored_from_predictions(
                candidate_frame,
                row_index=arrays[3],
                selector=arrays[4],
                scores=predictions,
            )

        return frame, features, predict_xgb

    if family != "neural":
        raise ValueError(f"Unsupported permutation-importance family: {family}")
    recipe = NeuralTrainingRecipe(**manifest["recipe"])
    scaler = FeatureScaler.from_dict(
        json.loads((root / "development_scaler.json").read_text())
    )
    model = _make_model(len(features), recipe, "cpu")
    _load_model_checkpoint(model, root / "development_model.pt", recipe.approach)
    embedding_store = manifest.get("embedding_store")
    source = _source_for_model(
        recipe,
        embedding_store=Path(embedding_store) if embedding_store else None,
    )

    def predict_neural(candidate_frame: pd.DataFrame) -> pd.DataFrame:
        return score_candidates(
            model,
            candidate_frame,
            stories,
            source,
            scaler,
            recipe,
            device="cpu",
        )

    return frame, features, predict_neural


def _score_development_regression(
    *,
    model_data_root: Path,
    regression_root: Path,
    scope: str,
) -> pd.DataFrame:
    """Score the frozen Stage-7 regression fit on its development articles."""
    root = Path(regression_root) / scope
    coefficients = pd.read_csv(root / "model_coefficients.csv")
    scaling = pd.read_csv(root / "feature_scaling.csv").set_index("term")
    base_terms = [
        str(term)
        for term in coefficients.loc[
            ~coefficients["term"].astype(str).str.endswith(":curator"), "term"
        ]
    ]
    interaction = coefficients.set_index("term")["estimate"]
    columns = [
        "story_id", "comment_id", "n_picks", "curator_selected",
        *[f"audience_selected_draw_{draw:02d}" for draw in range(1, 11)],
        *base_terms,
    ]
    frame = pd.read_parquet(Path(model_data_root) / f"choice_set_{scope}.parquet", columns=columns)
    split = pd.read_parquet(Path(model_data_root) / "master_article_split.parquet")
    split["story_id"] = split["story_id"].astype(str)
    frame["story_id"] = frame["story_id"].astype(str)
    frame["comment_id"] = frame["comment_id"].astype(str)
    frame = frame.merge(
        split[["story_id", "split_role"]],
        on="story_id",
        how="inner",
        validate="many_to_one",
    )
    frame = frame.loc[frame["split_role"].eq("development")].copy()
    if frame.empty:
        raise ValueError(f"No development rows found for regression scope={scope}")

    design = frame[base_terms].astype(float).copy()
    for term in base_terms:
        if bool(scaling.loc[term, "standardized"]):
            design[term] = (
                design[term] - float(scaling.loc[term, "mean"])
            ) / float(scaling.loc[term, "sd"])
    beta = coefficients.set_index("term")["estimate"]
    audience_beta = beta.loc[base_terms].to_numpy(dtype=float)
    curator_beta = audience_beta.copy()
    for index, term in enumerate(base_terms):
        curator_beta[index] += float(interaction.get(f"{term}:curator", 0.0))
    matrix = design.to_numpy(dtype=float)
    frame["audience_score"] = matrix @ audience_beta
    frame["curator_score"] = matrix @ curator_beta
    return frame.drop(columns="split_role")


def score_development_model(
    *,
    model_family: str,
    model_data_root: Path,
    factorial_root: Path,
    regression_root: Path,
    winner: dict[str, Any] | None = None,
    scope: str = "all",
) -> pd.DataFrame:
    """Return frozen-model scores for every development candidate.

    The returned wide frame has the same score and selection columns as the
    held-out score artifacts, so the primary nDCG and balanced macro-F1
    diagnostics use identical definitions on development and test data.
    """
    if model_family == "conditional_logit":
        return _score_development_regression(
            model_data_root=model_data_root,
            regression_root=regression_root,
            scope=scope,
        )
    if winner is None:
        raise ValueError("ML development scoring requires a frozen winner record")
    frame, _, predictor = _winner_permutation_components(
        winner=winner,
        factorial_root=factorial_root,
        model_data_root=model_data_root,
        split_role="development",
    )
    return predictor(frame)


def _load_or_compute_winner_permutation(
    *,
    winner: dict[str, Any],
    factorial_root: Path,
    model_data_root: Path,
    cache_root: Path,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    variant_id = str(winner["variant_id"])
    family = str(winner["family"])
    scope = str(winner["scope"])
    model_root = Path(factorial_root) / variant_id / scope
    supplied = model_root / "test_selector_permutation_importance.parquet"
    if supplied.exists():
        supplied_frame = pd.read_parquet(supplied)
        supplied_frame["feature_set"] = winner.get("feature_set")
        supplied_frame["model_label"] = _winner_label(
            family, winner.get("feature_set")
        )
        return supplied_frame
    cache = Path(cache_root) / variant_id / scope
    cache.mkdir(parents=True, exist_ok=True)
    output_path = cache / "test_selector_permutation_importance.parquet"
    manifest_path = cache / "permutation_manifest.json"
    model_path = model_root / (
        "development_model.json" if family == "xgboost" else "development_model.pt"
    )
    choice_path = Path(model_data_root) / f"choice_set_{scope}.parquet"
    signature_payload = {
        "version": PERMUTATION_CACHE_VERSION,
        "variant_id": variant_id,
        "scope": scope,
        "repeats": repeats,
        "seed": seed,
        "model_sha256": _sha256(model_path),
        "choice_set_sha256": _sha256(choice_path),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    if output_path.exists() and manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached.get("signature") == signature:
            return pd.read_parquet(output_path)
    frame, features, predictor = _winner_permutation_components(
        winner=winner,
        factorial_root=factorial_root,
        model_data_root=model_data_root,
    )
    output = selector_permutation_importance(
        frame,
        features,
        predictor,
        repeats=repeats,
        seed=seed,
        metadata={
            "scope": scope,
            "model_family": family,
            "model_id": variant_id,
            "model_label": _winner_label(family, winner.get("feature_set")),
            "feature_set": winner.get("feature_set"),
        },
    )
    output.to_parquet(output_path, index=False)
    manifest_path.write_text(
        json.dumps({**signature_payload, "signature": signature}, indent=2, sort_keys=True)
        + "\n"
    )
    return output


def _load_winner_inputs(
    winner_manifest_path: Path,
    factorial_root: Path,
    scopes: tuple[str, ...],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if not winner_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {winner_manifest_path}; finish stage 8 before stage 9"
        )
    manifest = json.loads(winner_manifest_path.read_text())
    if manifest.get("held_out_artifacts_read") is not False:
        raise RuntimeError("Winner manifest does not certify development-only selection")
    development_cv_path = factorial_root / "development_cv_results.csv"
    experiment_path = factorial_root / "experiment_variants.csv"
    _require_hash(development_cv_path, manifest.get("input", {}).get("sha256"), "Development CV summary")
    _require_hash(experiment_path, manifest.get("experiment_plan", {}).get("sha256"), "Experiment plan")

    ranking_record = manifest.get("outputs", {}).get("ranking", {})
    ranking_path = Path(ranking_record.get("path", ""))
    _require_hash(ranking_path, ranking_record.get("sha256"), "Frozen CV ranking")
    ranking = pd.read_csv(ranking_path)
    winners = pd.DataFrame(manifest.get("winners", []))
    if winners.empty:
        raise RuntimeError("Winner manifest contains no winners")
    winners = winners[winners["scope"].isin(scopes)].copy()
    winner_dimensions = manifest.get("winner_dimensions", ["family", "scope"])
    if "feature_set" in winner_dimensions:
        feature_sets = tuple(manifest.get("winner_levels", {}).get("feature_set", ()))
        if not feature_sets:
            raise RuntimeError("Winner manifest declares feature-set winners without feature-set levels")
        expected = {
            (family, scope, feature_set)
            for family in ("xgboost", "neural")
            for scope in scopes
            for feature_set in feature_sets
        }
        key_columns = ["family", "scope", "feature_set"]
    else:
        expected = {(family, scope) for family in ("xgboost", "neural") for scope in scopes}
        key_columns = ["family", "scope"]
    observed = set(winners[key_columns].itertuples(index=False, name=None))
    return manifest, winners, ranking[ranking["scope"].isin(scopes)].copy()


def _read_dual_model_scores(root: Path, family: str) -> pd.DataFrame:
    """Read one candidate row with both selector scores and article k."""
    if family == "conditional_logit":
        path = root / "test_scores_long.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        long = pd.read_parquet(
            path,
            columns=["story_id", "comment_id", "n_picks", "selector", "score"],
        )
        if long.duplicated(["story_id", "comment_id", "selector"]).any():
            raise ValueError(f"Duplicate regression score keys in {path}")
        index = ["story_id", "comment_id", "n_picks"]
        wide = long.pivot(index=index, columns="selector", values="score").reset_index()
        missing = {"audience", "curator"} - set(wide.columns)
        if missing:
            raise ValueError(f"Regression scores lack selectors: {sorted(missing)}")
        return wide.rename(
            columns={"audience": "audience_score", "curator": "curator_score"}
        )
    path = root / "test_scores_wide.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    required = [
        "story_id", "comment_id", "n_picks", "audience_score", "curator_score"
    ]
    return pd.read_parquet(path, columns=required)


def _read_scored_model_rows(root: Path, family: str) -> pd.DataFrame:
    """Read held-out scores and labels in one long selector-specific frame."""
    if family == "conditional_logit":
        path = root / "test_scores_long.parquet"
        columns = [
            "story_id", "comment_id", "n_picks", "selector", "selected", "score"
        ]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=columns)
    else:
        path = root / "test_scores_wide.parquet"
        columns = [
            "story_id", "comment_id", "n_picks", "curator_selected",
            "audience_selected_draw_01", "audience_score", "curator_score",
        ]
        if not path.exists():
            raise FileNotFoundError(path)
        wide = pd.read_parquet(path, columns=columns)
        audience = wide[
            [
                "story_id", "comment_id", "n_picks",
                "audience_selected_draw_01", "audience_score",
            ]
        ].rename(
            columns={
                "audience_selected_draw_01": "selected",
                "audience_score": "score",
            }
        )
        audience["selector"] = "audience"
        curator = wide[
            ["story_id", "comment_id", "n_picks", "curator_selected", "curator_score"]
        ].rename(columns={"curator_selected": "selected", "curator_score": "score"})
        curator["selector"] = "curator"
        frame = pd.concat([audience, curator], ignore_index=True)
    required = {"story_id", "comment_id", "n_picks", "selector", "selected", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Held-out scores are missing columns: {sorted(missing)}")
    if frame.duplicated(["story_id", "comment_id", "selector"]).any():
        raise ValueError("Held-out scores contain duplicate story/comment/selector rows")
    frame["selected"] = frame["selected"].astype(int)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    return frame


def balanced_macro_f1_at_k(
    scores: pd.DataFrame,
    *,
    draws: int = 100,
    seed: int = 20260813,
) -> pd.DataFrame:
    """Estimate paper-style balanced macro-F1 after selecting the top k.

    Each article-selector query retains all selected comments and draws the
    same number of non-selected comments without replacement. The model then
    ranks this balanced candidate set and selects its top k. F1 is calculated
    for both classes and macro-averaged, then averaged over deterministic
    negative-sampling draws. This is a secondary, balanced diagnostic; it does
    not alter model fitting, winner selection, or the primary full-candidate
    ranking metrics.
    """
    required = {"story_id", "comment_id", "n_picks", "selector", "selected", "score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Scores are missing columns: {sorted(missing)}")
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (story_id, selector), group in scores.groupby(
        ["story_id", "selector"], sort=False
    ):
        group = group.copy()
        group["selected"] = group["selected"].astype(int)
        k_values = pd.to_numeric(group["n_picks"], errors="raise").astype(int).unique()
        if len(k_values) != 1:
            raise ValueError(f"n_picks is not constant for {story_id}/{selector}")
        k = int(k_values[0])
        positives = group[group["selected"] == 1]
        negatives = group[group["selected"] == 0]
        if len(positives) != k or k < 1 or len(negatives) < k:
            raise ValueError(
                f"Cannot balance {story_id}/{selector}: positives={len(positives)}, "
                f"negatives={len(negatives)}, k={k}"
            )
        draw_values: list[float] = []
        positive_scores = positives["score"].to_numpy(dtype=float)
        positive_ids = positives["comment_id"].astype(str).to_numpy()
        negative_scores = negatives["score"].to_numpy(dtype=float)
        negative_ids = negatives["comment_id"].astype(str).to_numpy()
        for _ in range(draws):
            sampled = rng.choice(len(negative_scores), size=k, replace=False)
            scores = np.concatenate([positive_scores, negative_scores[sampled]])
            comment_ids = np.concatenate([positive_ids, negative_ids[sampled]])
            labels = np.concatenate([np.ones(k, dtype=int), np.zeros(k, dtype=int)])
            order = np.lexsort((comment_ids, -scores))
            true_positives = int(labels[order[:k]].sum())
            # On a balanced 2k candidate set with exactly k predictions,
            # positive- and negative-class F1 are both true_positives / k.
            draw_values.append(true_positives / k)
        rows.append(
            {
                "story_id": str(story_id),
                "selector": selector,
                "n_candidates": len(group),
                "n_picks": k,
                "balanced_macro_f1_at_k": float(np.mean(draw_values)),
                "balance_draws": draws,
            }
        )
    return pd.DataFrame(rows)


def _paired_balanced_f1_differences(
    article_metrics: pd.DataFrame,
    *,
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap paired model differences for the balanced F1 diagnostic."""
    metric = "balanced_macro_f1_at_k"
    required = {"story_id", "scope", "selector", "model_family", "model_id", metric}
    missing = required - set(article_metrics.columns)
    if missing:
        raise ValueError(f"Balanced F1 rows are missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    family_order = {family: index for index, family in enumerate(MODEL_ORDER)}
    for (scope, selector), group in article_metrics.groupby(
        ["scope", "selector"], observed=True
    ):
        models = group[["model_family", "model_id"]].drop_duplicates()
        models["_order"] = models["model_family"].map(family_order).fillna(len(MODEL_ORDER))
        models = list(
            models.sort_values(["_order", "model_id"])[
                ["model_family", "model_id"]
            ].itertuples(index=False, name=None)
        )
        for (first_family, first_id), (second_family, second_id) in itertools.combinations(models, 2):
            if first_family == second_family:
                continue
            left = group[group["model_id"] == first_id][["story_id", metric]]
            right = group[group["model_id"] == second_id][["story_id", metric]]
            paired = left.merge(right, on="story_id", suffixes=("_first", "_second"), validate="one_to_one")
            if paired.empty:
                raise RuntimeError(f"No paired held-out articles for {first_id} vs {second_id}")
            indices = rng.integers(0, len(paired), size=(bootstrap_draws, len(paired)))
            delta = paired[f"{metric}_second"].to_numpy(float) - paired[f"{metric}_first"].to_numpy(float)
            bootstrap = delta[indices].mean(axis=1)
            rows.append(
                {
                    "scope": scope,
                    "selector": selector,
                    "comparison": f"{second_id}_minus_{first_id}",
                    "model_first": first_family,
                    "model_second": second_family,
                    "model_first_id": first_id,
                    "model_second_id": second_id,
                    "metric": metric,
                    "estimate": float(delta.mean()),
                    "conf_low": float(np.quantile(bootstrap, 0.025)),
                    "conf_high": float(np.quantile(bootstrap, 0.975)),
                    "n_articles": len(paired),
                    "bootstrap_draws": bootstrap_draws,
                }
            )
    return pd.DataFrame(rows)


def _model_gap_artifact(
    root: Path,
    *,
    family: str,
    model_id: str,
    model_label: str,
    scope: str,
) -> pd.DataFrame:
    return model_implied_comment_gap(_read_dual_model_scores(root, family)).assign(
        model_id=model_id,
        model_family=family,
        model_label=model_label,
        scope=scope,
        analysis_partition="held_out_test",
    )


def _add_mean_selected_rank_from_scores(
    article_metrics: pd.DataFrame, scores_path: Path
) -> pd.DataFrame:
    if "mean_selected_rank" in article_metrics:
        return article_metrics
    scores = pd.read_parquet(
        scores_path,
        columns=["story_id", "comment_id", "selector", "selected", "score"],
    )
    rows = []
    for (story_id, selector), group in scores.groupby(
        ["story_id", "selector"], sort=False
    ):
        ranked = group.sort_values(
            ["score", "comment_id"], ascending=[False, True]
        ).reset_index(drop=True)
        selected_ranks = np.flatnonzero(ranked["selected"].to_numpy(dtype=bool)) + 1
        rows.append(
            {
                "story_id": story_id,
                "selector": selector,
                "mean_selected_rank": float(selected_ranks.mean()),
            }
        )
    return article_metrics.merge(
        pd.DataFrame(rows),
        on=["story_id", "selector"],
        how="left",
        validate="one_to_one",
    )


def _read_model_artifacts(
    factorial_root: Path, winners: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_frames = []
    article_frames = []
    tie_frames = []
    specification_rows = []
    gap_frames = []
    for winner in winners.to_dict("records"):
        variant_id = str(winner["variant_id"])
        family = str(winner["family"])
        scope = str(winner["scope"])
        root = factorial_root / variant_id / scope
        required = (
            root / "test_metric_summary.parquet",
            root / "test_article_metrics.parquet",
            root / "test_tie_sensitivity_metrics.parquet",
            root / "model_manifest.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Winner artifacts are incomplete: {missing}")
        common = {
            "model_id": variant_id,
            "model_family": family,
            "model_label": _winner_label(family, winner.get("feature_set")),
            "feature_set": winner.get("feature_set"),
            "scope": scope,
            "analysis_partition": "held_out_test",
        }
        performance_frames.append(
            pd.read_parquet(required[0]).assign(**common)
        )
        article_frames.append(pd.read_parquet(required[1]).assign(**common))
        tie = pd.read_parquet(required[2])
        tie_metrics = [metric for metric in METRICS if metric in tie]
        tie_frames.append(
            tie.groupby(["selector", "audience_tie_draw"], as_index=False)[tie_metrics]
            .mean()
            .assign(**common)
        )
        model_manifest = json.loads(required[3].read_text())
        recipe = model_manifest.get("recipe") or model_manifest.get("best_parameters") or {}
        specification_rows.append(
            {
                **common,
                "feature_set": winner.get("feature_set"),
                "draw_policy": winner.get("draw_policy"),
                "negative_sampling": winner.get("negative_sampling"),
                "schedule": winner.get("schedule"),
                "heads": winner.get("heads"),
                "network": winner.get("network"),
                "n_features": len(model_manifest.get("features", [])),
                "development_articles": model_manifest.get("development_articles"),
                "held_out_articles": model_manifest.get("paper2_test_articles"),
                "specification_json": json.dumps(recipe, sort_keys=True),
                "model_manifest_sha256": _sha256(required[3]),
            }
        )
        gap_frames.append(
            _model_gap_artifact(
                root,
                family=family,
                model_id=variant_id,
                model_label=_winner_label(family, winner.get("feature_set")),
                scope=scope,
            )
        )
    return (
        pd.concat(performance_frames, ignore_index=True),
        pd.concat(article_frames, ignore_index=True),
        pd.concat(tie_frames, ignore_index=True),
        pd.DataFrame(specification_rows),
        pd.concat(gap_frames, ignore_index=True),
    )


def _read_regression_artifacts(
    regression_root: Path, scopes: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_frames = []
    article_frames = []
    tie_frames = []
    association_frames = []
    diagnostic_frames = []
    gap_frames = []
    for scope in scopes:
        root = regression_root / scope
        required = {
            "performance": root / "test_metric_summary.parquet",
            "article": root / "test_article_metrics.parquet",
            "tie": root / "test_tie_sensitivity_metrics.parquet",
            "associations": root / "selector_associations.csv",
            "diagnostics": root / "model_diagnostics.csv",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Stage-7 artifacts are incomplete for scope={scope}: {missing}"
            )
        common = {
            "model_id": "stage7_stacked_selection",
            "model_family": "conditional_logit",
            "model_label": _winner_label("conditional_logit"),
            "feature_set": None,
            "scope": scope,
            "analysis_partition": "held_out_test",
        }
        performance_frames.append(pd.read_parquet(required["performance"]).assign(**common))
        regression_article = _add_mean_selected_rank_from_scores(
            pd.read_parquet(required["article"]), root / "test_scores_long.parquet"
        )
        article_frames.append(regression_article.assign(**common))
        tie = pd.read_parquet(required["tie"])
        tie_metrics = [metric for metric in METRICS if metric in tie]
        tie_frames.append(
            tie.groupby(["selector", "audience_tie_draw"], as_index=False)[tie_metrics]
            .mean().assign(**common)
        )
        association_frames.append(pd.read_csv(required["associations"]).assign(scope=scope))
        diagnostic_frames.append(pd.read_csv(required["diagnostics"]).assign(scope=scope))
        gap_frames.append(
            _model_gap_artifact(
                root,
                family="conditional_logit",
                model_id="stage7_stacked_selection",
                model_label=_winner_label("conditional_logit"),
                scope=scope,
            )
        )
    return (
        pd.concat(performance_frames, ignore_index=True),
        pd.concat(article_frames, ignore_index=True),
        pd.concat(tie_frames, ignore_index=True),
        pd.concat(association_frames, ignore_index=True),
        pd.concat(diagnostic_frames, ignore_index=True),
        pd.concat(gap_frames, ignore_index=True),
    )


def _paired_model_differences(
    article_metrics: pd.DataFrame,
    *,
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame:
    required = {"story_id", "scope", "selector", "model_family", "model_id", *METRICS}
    missing = required - set(article_metrics.columns)
    if missing:
        raise ValueError(f"Article metrics are missing columns: {sorted(missing)}")
    rows = []
    rng = np.random.default_rng(seed)
    for (scope, selector), group in article_metrics.groupby(
        ["scope", "selector"], observed=True
    ):
        family_order = {family: index for index, family in enumerate(MODEL_ORDER)}
        models = group[["model_family", "model_id"]].drop_duplicates()
        models["_order"] = models["model_family"].map(family_order).fillna(len(MODEL_ORDER))
        models = models.sort_values(["_order", "model_id"])
        models = list(models[["model_family", "model_id"]].itertuples(index=False, name=None))
        for (first_family, first_id), (second_family, second_id) in itertools.combinations(models, 2):
            if first_family == second_family:
                continue
            left = group[group["model_id"] == first_id][["story_id", *METRICS]]
            right = group[group["model_id"] == second_id][["story_id", *METRICS]]
            paired = left.merge(right, on="story_id", suffixes=("_first", "_second"), validate="one_to_one")
            if paired.empty:
                raise RuntimeError(f"No paired held-out articles for {first_id} vs {second_id}")
            indices = rng.integers(0, len(paired), size=(bootstrap_draws, len(paired)))
            for metric in METRICS:
                delta = (
                    paired[f"{metric}_second"].to_numpy(float)
                    - paired[f"{metric}_first"].to_numpy(float)
                )
                boot = delta[indices].mean(axis=1)
                rows.append(
                    {
                        "scope": scope,
                        "selector": selector,
                        "comparison": f"{second_id}_minus_{first_id}",
                        "model_first": first_family,
                        "model_second": second_family,
                        "model_first_id": first_id,
                        "model_second_id": second_id,
                        "metric": metric,
                        "estimate": float(delta.mean()),
                        "conf_low": float(np.quantile(boot, 0.025)),
                        "conf_high": float(np.quantile(boot, 0.975)),
                        "n_articles": len(paired),
                        "bootstrap_draws": bootstrap_draws,
                    }
                )
    return pd.DataFrame(rows)


def _save_latex(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    available = [column for column in columns if column in frame.columns]
    path.write_text(
        frame[available].to_latex(index=False, float_format=lambda value: f"{value:.3f}")
    )


def _save_figures(
    ranking: pd.DataFrame,
    performance: pd.DataFrame,
    ties: pd.DataFrame,
    associations: pd.DataFrame,
    model_gap_summary: pd.DataFrame,
    permutation_gaps: pd.DataFrame,
    shap_summary: pd.DataFrame,
    output_root: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    def save(fig, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = figures_root / f"{stem}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)

    primary = ranking[ranking["scope"] == "all"] if "all" in set(ranking["scope"]) else ranking
    families = [family for family in ("xgboost", "neural") if family in set(primary["family"])]
    fig, axes = plt.subplots(1, len(families), figsize=(14, 6), squeeze=False)
    for axis, family in zip(axes.flat, families, strict=True):
        shown = primary[primary["family"] == family].nsmallest(10, "development_cv_rank").sort_values("mean_macro_ndcg_at_k")
        axis.barh(shown["variant_id"], shown["mean_macro_ndcg_at_k"], xerr=shown["sd_macro_ndcg_at_k"], color="#356a8a" if family == "xgboost" else "#4b8b6f")
        axis.set(title=f"{family.title()}: top development-CV variants", xlabel="Mean macro nDCG@k (± fold SD)")
    save(fig, "all_development_cv_top_variants")

    ndcg = performance[(performance["scope"] == "all") & (performance["metric"] == "ndcg_at_k")].copy()
    if ndcg.empty:
        ndcg = performance[performance["metric"] == "ndcg_at_k"].copy()
    ndcg["display"] = ndcg["model_label"]
    ndcg = ndcg.sort_values(["model_family", "selector"])
    fig, axis = plt.subplots(figsize=(6.6, max(4.5, 0.45 * len(ndcg))))
    y = np.arange(len(ndcg))
    axis.errorbar(ndcg["estimate"], y, xerr=[ndcg["estimate"] - ndcg["conf_low"], ndcg["conf_high"] - ndcg["estimate"]], fmt="o", color="#356a8a", capsize=3)
    axis.set_yticks(y, ndcg["display"])
    axis.set(xlabel="Held-out nDCG@k (95% article-bootstrap CI)", title="Frozen Paper 1 model performance")
    save(fig, "all_winner_held_out_ndcg")

    tie = ties[(ties["scope"] == "all") & (ties["selector"] == "audience")].copy()
    if tie.empty:
        tie = ties[ties["selector"] == "audience"].copy()
    fig, axis = plt.subplots(figsize=(6.6, 4.8))
    for label, group in tie.groupby("model_label", sort=False):
        axis.plot(group["audience_tie_draw"], group["ndcg_at_k"], marker="o", label=label)
    axis.set(xlabel="Audience tie draw", ylabel="Held-out nDCG@k", title="Audience-label tie sensitivity")
    axis.set_xticks(range(1, 11))
    axis.legend(frameon=False)
    save(fig, "all_winner_audience_tie_sensitivity")

    if not associations.empty:
        association_scope = associations[associations["scope"] == "all"].copy()
        if association_scope.empty:
            association_scope = associations.copy()
        for plotter, stem in (
            (plot_regression_selector_differences, "all_regression_selector_differences"),
            (plot_regression_selector_coefficients, "all_regression_selector_coefficients"),
        ):
            if plotter(association_scope, output_root, show=False) is not None:
                outputs.extend(figures_root / f"{stem}.{suffix}" for suffix in ("png", "pdf"))

    shown_gap = model_gap_summary[
        model_gap_summary["scope"] == "all"
    ].copy()
    if shown_gap.empty:
        shown_gap = model_gap_summary.copy()
    shown_gap = shown_gap.sort_values("gap_mean", ascending=False)
    fig, axis = plt.subplots(figsize=(6.6, max(4.2, 0.55 * len(shown_gap))))
    y = np.arange(len(shown_gap))
    axis.errorbar(
        shown_gap["gap_mean"],
        y - 0.18,
        xerr=[
            shown_gap["gap_mean"] - shown_gap["conf_low"],
            shown_gap["conf_high"] - shown_gap["gap_mean"],
        ],
        fmt="o",
        color="#356a8a",
        capsize=3,
        label="Equal article weight",
    )
    axis.errorbar(
        shown_gap["gap_comment_weighted_mean"],
        y,
        xerr=[
            shown_gap["gap_comment_weighted_mean"] - shown_gap["weighted_conf_low"],
            shown_gap["weighted_conf_high"] - shown_gap["gap_comment_weighted_mean"],
        ],
        fmt="s",
        color="#4b8b6f",
        capsize=3,
        label="Candidate-comment weight",
    )
    axis.errorbar(
        shown_gap["gap_comment_weighted_median"],
        y + 0.18,
        xerr=[
            shown_gap["gap_comment_weighted_median"]
            - shown_gap["weighted_median_conf_low"],
            shown_gap["weighted_median_conf_high"]
            - shown_gap["gap_comment_weighted_median"],
        ],
        fmt="D",
        color="#76528b",
        capsize=3,
        label="Candidate-comment-weighted median",
    )
    axis.axvline(0.5, color="#d4744c", linestyle="--", label="random-set expectation")
    axis.set_yticks(y, shown_gap["model_label"])
    axis.set(
        xlim=(0, 1),
        xlabel="Model-implied curator–audience comment gap",
        title="Held-out model-implied comment gaps",
    )
    axis.legend(frameon=False)
    save(fig, "all_model_implied_comment_gaps")

    primary_permutation = permutation_gaps[
        permutation_gaps["scope"] == "all"
    ].copy()
    if primary_permutation.empty:
        primary_permutation = permutation_gaps.copy()
    if not primary_permutation.empty:
        for plotter, stem in (
            (plot_winner_permutation_importance_gaps, "all_winner_permutation_importance_gaps"),
            (plot_winner_permutation_importance, "all_winner_permutation_importance"),
        ):
            if plotter(primary_permutation, output_root, show=False) is not None:
                outputs.extend(figures_root / f"{stem}.{suffix}" for suffix in ("png", "pdf"))

    primary_shap = shap_summary[shap_summary["scope"] == "all"].copy()
    if primary_shap.empty:
        primary_shap = shap_summary.copy()
    shap_columns = {"mean_shap_audience", "mean_shap_curator", "mean_shap_gap"}
    if not primary_shap.empty and shap_columns.issubset(primary_shap.columns):
        for plotter, stem in (
            (plot_winner_shap_importance, "all_winner_shap_importance"),
        ):
            if plotter(primary_shap, output_root, show=False) is not None:
                outputs.extend(figures_root / f"{stem}.{suffix}" for suffix in ("png", "pdf"))
        shap_gap_figures = plot_winner_shap_importance_gaps(
            primary_shap, output_root, show=False,
        )
        if shap_gap_figures:
            for stem in (
                "all_winner_shap_importance_gaps_xgb",
                "all_winner_shap_importance_gaps_nn",
            ):
                outputs.extend(
                    figures_root / f"{stem}.{suffix}"
                    for suffix in ("png", "pdf")
                )
        if plot_regression_vs_shap_gaps(
            association_scope,
            primary_shap,
            output_root,
            show=False,
        ) is not None:
            outputs.extend(
                figures_root / f"all_regression_vs_shap_gaps.{suffix}"
                for suffix in ("png", "pdf")
            )
    return outputs


def run_paper1_reporting(
    *,
    model_data_root: Path = Path("model_output/selection_2025/model_data"),
    factorial_root: Path = Path("model_output/selection_2025/factorial_rankers"),
    winner_root: Path = Path("model_output/selection_2025/paper1/factorial_winners"),
    regression_root: Path = Path("model_output/selection_2025/regression"),
    output_root: Path = Path("model_output/selection_2025/paper1/reporting"),
    scopes: Iterable[str] = ("all",),
    bootstrap_draws: int = 1000,
    permutation_repeats: int = 1,
    shap_test_rows: int = 50_000,
    shap_background_rows: int = 2_048,
    shap_nsamples: int = 100,
    shap_force_recompute: bool = False,
    shap_chunk_rows: int = 500,
    seed: int = 20260813,
    balanced_f1_draws: int = 100,
    make_figures: bool = True,
    make_latex_tables: bool = True,
    require_factorial_idle: bool = True,
) -> dict:
    """Create Stage 9 only after Stage 8 has frozen development-CV winners."""
    if require_factorial_idle:
        assert_factorial_idle()
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    if permutation_repeats < 1:
        raise ValueError("permutation_repeats must be positive")
    if balanced_f1_draws < 1:
        raise ValueError("balanced_f1_draws must be positive")
    scopes = _normalise_scopes(scopes)
    winner_manifest_path = winner_root / "factorial_winner_manifest.json"
    winner_manifest, winners, ranking = _load_winner_inputs(
        winner_manifest_path, factorial_root, scopes
    )
    ml_performance, ml_articles, ml_ties, specifications, ml_gaps = _read_model_artifacts(
        factorial_root, winners
    )
    reg_performance, reg_articles, reg_ties, associations, diagnostics, reg_gaps = _read_regression_artifacts(
        regression_root, scopes
    )
    performance = pd.concat([reg_performance, ml_performance], ignore_index=True)
    article_metrics = pd.concat([reg_articles, ml_articles], ignore_index=True)
    ties = pd.concat([reg_ties, ml_ties], ignore_index=True)
    balanced_article_frames = []
    for index, scope in enumerate(scopes):
        regression_balanced = balanced_macro_f1_at_k(
            _read_scored_model_rows(regression_root / scope, "conditional_logit"),
            draws=balanced_f1_draws,
            seed=seed + index * 100_000,
        )
        balanced_article_frames.append(
            regression_balanced.assign(
                model_id="stage7_stacked_selection",
                model_family="conditional_logit",
                model_label=_winner_label("conditional_logit"),
                feature_set=None,
                scope=scope,
                analysis_partition="held_out_test",
            )
        )
    for index, winner in enumerate(winners.to_dict("records"), start=1):
        scope = str(winner["scope"])
        balanced = balanced_macro_f1_at_k(
            _read_scored_model_rows(
                factorial_root / str(winner["variant_id"]) / scope,
                str(winner["family"]),
            ),
            draws=balanced_f1_draws,
            seed=seed + index * 100_000,
        )
        balanced_article_frames.append(
            balanced.assign(
                model_id=str(winner["variant_id"]),
                model_family=str(winner["family"]),
                model_label=_winner_label(
                    str(winner["family"]), winner.get("feature_set")
                ),
                feature_set=winner.get("feature_set"),
                scope=scope,
                analysis_partition="held_out_test",
            )
        )
    balanced_articles = pd.concat(balanced_article_frames, ignore_index=True)
    balanced_performance = _bootstrap_group_mean(
        balanced_articles,
        groups=["scope", "model_family", "model_id", "model_label", "selector"],
        value="balanced_macro_f1_at_k",
        draws=bootstrap_draws,
        seed=seed + 400_000,
        estimate_name="estimate",
    )
    balanced_performance["metric"] = "balanced_macro_f1_at_k"
    balanced_performance["model_pair_label"] = balanced_performance["model_label"]
    balanced_performance["model_label"] = balanced_performance.apply(
        lambda row: model_selector_label(
            row["model_family"],
            balanced_articles.loc[
                (balanced_articles["model_id"] == row["model_id"])
                & (balanced_articles["scope"] == row["scope"]),
                "feature_set",
            ].iloc[0],
            row["selector"],
        ),
        axis=1,
    )
    balanced_paired = _paired_balanced_f1_differences(
        balanced_articles, bootstrap_draws=bootstrap_draws, seed=seed + 500_000
    )
    # Keep the model-pair label for relationship summaries, but expose the
    # requested audience/editor label wherever a table or figure has a
    # selector-specific row.
    for frame in (performance, ties):
        frame["model_pair_label"] = frame["model_label"]
        frame["model_label"] = frame.apply(
            lambda row: model_selector_label(
                row["model_family"], row.get("feature_set"), row["selector"]
            ),
            axis=1,
        )
    paired = _paired_model_differences(
        article_metrics, bootstrap_draws=bootstrap_draws, seed=seed
    )
    model_gaps = pd.concat([reg_gaps, ml_gaps], ignore_index=True)
    model_gap_summary = summarize_model_implied_gaps(
        model_gaps, draws=bootstrap_draws, seed=seed + 100_000
    )
    feature_gaps = regression_feature_gaps(associations)
    permutation_article = pd.concat(
        [
            _load_or_compute_winner_permutation(
                winner=winner,
                factorial_root=factorial_root,
                model_data_root=model_data_root,
                cache_root=output_root / "cache" / "permutation_importance",
                repeats=permutation_repeats,
                seed=seed + index * 100_000,
            )
            for index, winner in enumerate(winners.to_dict("records"), start=1)
        ],
        ignore_index=True,
    )
    permutation_gaps = summarize_permutation_importance_gaps(
        permutation_article, draws=bootstrap_draws, seed=seed + 200_000
    )
    feature_manifest_path = Path(model_data_root) / "feature_manifest.json"
    feature_labels: dict[str, str] = {}
    if feature_manifest_path.exists():
        registry = json.loads(feature_manifest_path.read_text()).get("features", {})
        feature_labels = {
            name: str(details.get("label", name)) for name, details in registry.items()
        }
    permutation_gaps["feature_label"] = permutation_gaps["feature"].map(
        lambda value: feature_labels.get(str(value), str(value))
    )

    from commentgap_analysis.explanations import run_shap_explanations

    shap_result = run_shap_explanations(
        model_data_root=model_data_root,
        factorial_root=factorial_root,
        winners=winners,
        output_root=output_root,
        scopes=scopes,
        test_rows=shap_test_rows,
        background_rows=shap_background_rows,
        nsamples=shap_nsamples,
        seed=seed + 300_000,
        force_recompute=shap_force_recompute,
        chunk_rows=shap_chunk_rows,
    )
    shap_summary = shap_result["summary"]
    if feature_labels:
        shap_summary["feature_label"] = shap_summary["feature"].map(
            lambda value: feature_labels.get(str(value), str(value))
        )

    tables_root = output_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    table_frames = {
        "development_cv_variant_ranking.csv": ranking,
        "development_cv_winners.csv": winners,
        "winner_model_specifications.csv": specifications,
        "held_out_model_performance.csv": performance,
        "held_out_balanced_macro_f1.csv": balanced_performance,
        "held_out_paired_model_differences.csv": paired,
        "held_out_paired_balanced_macro_f1.csv": balanced_paired,
        "held_out_tie_sensitivity.csv": ties,
        "regression_selector_associations.csv": associations,
        "regression_feature_gaps.csv": feature_gaps,
        "held_out_model_implied_gap_summary.csv": model_gap_summary,
        "held_out_permutation_importance_gaps.csv": permutation_gaps,
        "held_out_shap_importance.csv": shap_summary,
        "regression_model_diagnostics.csv": diagnostics,
    }
    table_paths = {}
    for name, frame in table_frames.items():
        path = tables_root / name
        frame.to_csv(path, index=False)
        table_paths[name] = path
    parquet_paths = {
        "held_out_model_implied_article_gaps.parquet": tables_root
        / "held_out_model_implied_article_gaps.parquet",
        "held_out_balanced_macro_f1_articles.parquet": tables_root
        / "held_out_balanced_macro_f1_articles.parquet",
        "held_out_selector_permutation_importance.parquet": tables_root
        / "held_out_selector_permutation_importance.parquet",
        "held_out_shap_values.parquet": tables_root / "held_out_shap_values.parquet",
        "held_out_shap_importance.parquet": tables_root / "held_out_shap_importance.parquet",
    }
    model_gaps.to_parquet(
        parquet_paths["held_out_model_implied_article_gaps.parquet"], index=False
    )
    balanced_articles.to_parquet(
        parquet_paths["held_out_balanced_macro_f1_articles.parquet"], index=False
    )
    permutation_article.to_parquet(
        parquet_paths["held_out_selector_permutation_importance.parquet"], index=False
    )
    shap_result["values"].to_parquet(parquet_paths["held_out_shap_values.parquet"], index=False)
    shap_summary.to_parquet(parquet_paths["held_out_shap_importance.parquet"], index=False)
    latex_paths = {}
    if make_latex_tables:
        latex_paths = {
            "development_cv_winners.tex": tables_root / "development_cv_winners.tex",
            "held_out_ndcg.tex": tables_root / "held_out_ndcg.tex",
            "paired_ndcg_differences.tex": tables_root / "paired_ndcg_differences.tex",
            "model_implied_gaps.tex": tables_root / "model_implied_gaps.tex",
            "regression_feature_gaps.tex": tables_root / "regression_feature_gaps.tex",
            "permutation_importance_gaps.tex": tables_root / "permutation_importance_gaps.tex",
        }
        _save_latex(winners, latex_paths["development_cv_winners.tex"], ["scope", "family", "variant_id", "mean_macro_ndcg_at_k", "sd_macro_ndcg_at_k", "min_fold_macro_ndcg_at_k"])
        _save_latex(performance[performance["metric"] == "ndcg_at_k"], latex_paths["held_out_ndcg.tex"], ["scope", "model_label", "selector", "estimate", "conf_low", "conf_high", "bootstrap_draws"])
        _save_latex(paired[paired["metric"] == "ndcg_at_k"], latex_paths["paired_ndcg_differences.tex"], ["scope", "selector", "comparison", "estimate", "conf_low", "conf_high", "n_articles"])
        _save_latex(model_gap_summary, latex_paths["model_implied_gaps.tex"], ["scope", "model_label", "gap_mean", "conf_low", "conf_high", "gap_comment_weighted_mean", "weighted_conf_low", "weighted_conf_high", "gap_comment_weighted_median", "weighted_median_conf_low", "weighted_median_conf_high", "n_articles", "candidate_comments"])
        _save_latex(feature_gaps, latex_paths["regression_feature_gaps.tex"], ["scope", "feature", "feature_gap_log_odds", "feature_gap_conf_low", "feature_gap_conf_high", "curator_to_audience_odds_ratio"])
        _save_latex(permutation_gaps, latex_paths["permutation_importance_gaps.tex"], ["scope", "model_label", "feature_label", "audience_importance", "curator_importance", "permutation_importance_gap", "conf_low", "conf_high"])
    figure_paths = (
        _save_figures(
            ranking,
            performance,
            ties,
            associations,
            model_gap_summary,
            permutation_gaps,
            shap_summary,
            output_root,
        )
        if make_figures
        else []
    )

    outputs = {}
    for name, path in {**table_paths, **parquet_paths, **latex_paths}.items():
        outputs[f"table:{name}"] = {"path": str(path), "sha256": _sha256(path)}
    for path in figure_paths:
        outputs[f"figure:{path.name}"] = {"path": str(path), "sha256": _sha256(path)}
    manifest = {
        "version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scopes": list(scopes),
        "primary_scope": "all",
        "appendix_scope": "root",
        "winner_selection": "development CV only; held-out artifacts opened only after winner-manifest hash validation",
        "winner_manifest": {"path": str(winner_manifest_path), "sha256": _sha256(winner_manifest_path)},
        "development_cv_sha256": winner_manifest["input"]["sha256"],
        "bootstrap_draws": bootstrap_draws,
        "balanced_f1_draws": balanced_f1_draws,
        "permutation_repeats": permutation_repeats,
        "model_implied_gap": "predicted curator top-k ranked by predicted audience score; midranks and fractional curator cutoff ties; equal-article mean, candidate-comment-weighted mean, and candidate-comment-weighted median are reported",
        "comment_weighting": "sum(n_candidates * article_gap) / sum(n_candidates); confidence interval resamples articles and recomputes the weighted ratio",
        "comment_weighted_median": "first ordered article gap where cumulative n_candidates reaches at least 50%; confidence interval resamples articles and recomputes the weighted median",
        "permutation_importance_gap": "paired within-article nDCG@k loss, curator minus audience; audience metrics average ten tie draws",
        "balanced_macro_f1_at_k": "secondary held-out diagnostic; retain all selected comments and sample an equal number of non-selected comments within each article-selector query, rank by model score, select the top k, calculate macro-F1 across selected/non-selected classes, and average over articles; non-selected is the closest available analogue to the paper's zero-pick class",
        "permutation_features": "named tabular model features; frozen BGE vectors are held fixed rather than interpreted dimension by dimension",
        "shap": shap_result["manifest"],
        "outputs": outputs,
    }
    manifest_path = output_root / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
