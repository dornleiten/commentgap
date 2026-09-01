"""Resumable XGBoost/neural factorial ranking experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fnmatch import fnmatch
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .neural_ranking import (
    DEFAULT_BGE_MODEL_ID,
    DEFAULT_BGE_REVISION,
    EmbeddingStorySource,
    NeuralTrainingRecipe,
    _load_scope_inputs,
    apply_fold_feature_columns,
    default_recipe,
    discover_complete_embedding_store,
    evaluate_tie_draws,
    resolve_model_revision,
    run_neural_ranker_workflow,
    scores_to_long,
)
from .ranking import (
    bootstrap_metric_summary,
    default_search_configs,
    evaluate_rank_scores,
    make_ranker,
    refinement_search_configs,
    select_best_configuration,
)


FACTORIAL_VERSION = 2
EMBEDDING_CACHE_VERSION = 1
DRAW_COLUMNS = tuple(f"audience_selected_draw_{draw:02d}" for draw in range(1, 11))


@dataclass(frozen=True)
class FactorialVariant:
    variant_id: str
    family: str
    feature_set: str
    draw_policy: str
    negative_sampling: str = "na"
    schedule: str = "na"
    heads: str = "na"
    network: str = "na"

    def as_record(self) -> dict[str, str]:
        return asdict(self)


def factorial_variants() -> list[FactorialVariant]:
    variants = [
        FactorialVariant(
            variant_id=f"xgb__{features}__{draw}",
            family="xgboost",
            feature_set=features,
            draw_policy=draw,
        )
        for features, draw in product(
            ("metadata", "metadata_bge"), ("draw1", "mean10")
        )
    ]
    for features, draw, negatives, schedule, heads, network in product(
        ("metadata", "metadata_bge"),
        ("draw1", "mean10"),
        ("random4", "hard1_random3"),
        ("fixed", "plateau"),
        ("separate", "shared_residual"),
        ("base", "large"),
    ):
        variants.append(
            FactorialVariant(
                variant_id=(
                    f"nn__{features}__{draw}__{negatives}__{schedule}"
                    f"__{heads}__{network}"
                ),
                family="neural",
                feature_set=features,
                draw_policy=draw,
                negative_sampling=negatives,
                schedule=schedule,
                heads=heads,
                network=network,
            )
        )
    return variants


def select_variants(
    variants: Iterable[FactorialVariant],
    *,
    include: tuple[str, ...] = ("*",),
    exclude: tuple[str, ...] = (),
) -> list[FactorialVariant]:
    return [
        variant
        for variant in variants
        if any(fnmatch(variant.variant_id, pattern) for pattern in include)
        and not any(fnmatch(variant.variant_id, pattern) for pattern in exclude)
    ]


def neural_recipe(variant: FactorialVariant) -> NeuralTrainingRecipe:
    if variant.family != "neural":
        raise ValueError("Only neural variants have neural recipes")
    approach = "metadata_mlp" if variant.feature_set == "metadata" else "frozen_bge"
    base = default_recipe(approach)
    dimensions = (
        {"text_projection_dim": 256, "metadata_projection_dim": 128, "head_hidden_dim": 128}
        if variant.network == "base"
        else {"text_projection_dim": 512, "metadata_projection_dim": 256, "head_hidden_dim": 256}
    )
    return replace(
        base,
        ranking_loss="pairwise_logistic",
        audience_draw_policy=variant.draw_policy,
        negatives_per_positive=4,
        hard_negatives_per_positive=(
            0 if variant.negative_sampling == "random4" else 1
        ),
        learning_rate_schedule=(
            "fixed" if variant.schedule == "fixed" else "plateau"
        ),
        patience=3 if variant.schedule == "fixed" else 8,
        head_type=variant.heads,
        **dimensions,
    )


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _signature(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _macro_from_scored(
    scored: pd.DataFrame, draw_policy: str
) -> tuple[float, float, float]:
    draws = 1 if draw_policy == "draw1" else 10
    metrics = evaluate_tie_draws(scored, draws=draws)
    selector = metrics.groupby("selector", sort=False)["ndcg_at_k"].mean()
    audience = float(selector["audience"])
    curator = float(selector["curator"])
    return (audience + curator) / 2, audience, curator


def _embedding_cache(
    frame: pd.DataFrame,
    *,
    scope: str,
    embedding_store: Path,
    cache_root: Path,
    progress_every_stories: int,
    run_label: str,
) -> np.ndarray:
    cache_dir = Path(cache_root) / scope
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / "bge_embeddings.float16.npy"
    state_path = cache_dir / "embedding_cache_state.json"
    manifest_path = cache_dir / "embedding_cache_manifest.json"
    keys_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            frame[["story_id", "comment_id"]], index=False
        ).to_numpy().tobytes()
    ).hexdigest()
    payload = {
        "version": EMBEDDING_CACHE_VERSION,
        "scope": scope,
        "rows": len(frame),
        "columns": 1024,
        "dtype": "float16",
        "keys_sha256": keys_hash,
        "embedding_store": str(embedding_store),
        "embedding_manifest_sha256": _file_sha256(
            Path(embedding_store) / "embedding_manifest.json"
        ),
    }
    signature = _signature(payload)
    if manifest_path.exists() and matrix_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature") == signature:
            matrix = np.load(matrix_path, mmap_mode="r")
            if matrix.shape == (len(frame), 1024) and matrix.dtype == np.float16:
                return matrix

    stories = list(frame.groupby("story_id", sort=True).groups.items())
    completed = 0
    if state_path.exists() and matrix_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("signature") == signature:
            completed = int(state.get("completed_stories", 0))
    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="r+" if completed else "w+",
        dtype=np.float16,
        shape=(len(frame), 1024),
    )
    source = EmbeddingStorySource(embedding_store)
    started = time.monotonic()
    for position, (_, indices) in enumerate(
        stories[completed:], start=completed + 1
    ):
        story = frame.loc[indices].copy()
        matrix[np.asarray(indices, dtype=int)] = source.load(story).astype(
            np.float16, copy=False
        )
        if (
            position == len(stories)
            or position == 1
            or position % max(1, progress_every_stories) == 0
        ):
            matrix.flush()
            elapsed = max(time.monotonic() - started, 1e-9)
            done_now = position - completed
            eta = (len(stories) - position) * elapsed / max(done_now, 1)
            print(
                f"{run_label}/{scope} BGE cache: {position}/{len(stories)} stories "
                f"({100 * position / len(stories):.1f}%) ETA={eta / 60:.1f}m",
                flush=True,
            )
            _atomic_json(
                {
                    "signature": signature,
                    "completed_stories": position,
                    "total_stories": len(stories),
                },
                state_path,
            )
    matrix.flush()
    _atomic_json({"signature": signature, **payload}, manifest_path)
    return np.load(matrix_path, mmap_mode="r")


def _xgb_arrays(
    frame: pd.DataFrame,
    *,
    features: list[str],
    draw_policy: str,
    embedding_matrix: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["story_id", "comment_id"]).copy()
    original_rows = ordered.index.to_numpy(dtype=np.int64)
    ordered = ordered.reset_index(drop=True)
    row_parts: list[np.ndarray] = []
    selector_parts: list[np.ndarray] = []
    qid_parts: list[np.ndarray] = []
    for query_number, (_, indices) in enumerate(
        ordered.groupby("story_id", sort=True).groups.items()
    ):
        local = np.asarray(indices, dtype=np.int64)
        row_parts.extend((local, local))
        selector_parts.extend(
            (
                np.zeros(len(local), dtype=np.int8),
                np.ones(len(local), dtype=np.int8),
            )
        )
        qid_parts.extend(
            (
                np.full(len(local), query_number * 2, dtype=np.int32),
                np.full(len(local), query_number * 2 + 1, dtype=np.int32),
            )
        )
    row_index = np.concatenate(row_parts)
    selector = np.concatenate(selector_parts)
    qid = np.concatenate(qid_parts)
    metadata = ordered[features].to_numpy(dtype=np.float32)
    width = len(features) + (1024 if embedding_matrix is not None else 0) + 1
    x = np.empty((len(row_index), width), dtype=np.float32)
    x[:, : len(features)] = metadata[row_index]
    offset = len(features)
    if embedding_matrix is not None:
        aligned_embedding_rows = original_rows[row_index]
        x[:, offset : offset + 1024] = embedding_matrix[aligned_embedding_rows]
        offset += 1024
    x[:, offset] = selector
    audience = (
        ordered["audience_selected_draw_01"].to_numpy(dtype=np.int32)
        if draw_policy == "draw1"
        else ordered[list(DRAW_COLUMNS)].sum(axis=1).to_numpy(dtype=np.int32)
    )
    curator = ordered["curator_selected"].to_numpy(dtype=np.int32)
    labels = np.where(selector == 0, audience[row_index], curator[row_index])
    return x, labels, qid, row_index, selector


def _scored_from_predictions(
    frame: pd.DataFrame,
    *,
    row_index: np.ndarray,
    selector: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    ordered = frame.sort_values(["story_id", "comment_id"]).reset_index(drop=True)
    audience_scores = np.empty(len(ordered), dtype=float)
    curator_scores = np.empty(len(ordered), dtype=float)
    audience_rows = selector == 0
    curator_rows = ~audience_rows
    audience_scores[row_index[audience_rows]] = scores[audience_rows]
    curator_scores[row_index[curator_rows]] = scores[curator_rows]
    columns = [
        "story_id",
        "comment_id",
        "n_picks",
        "curator_selected",
        *DRAW_COLUMNS,
    ]
    scored = ordered[columns].copy()
    scored["audience_score"] = audience_scores
    scored["curator_score"] = curator_scores
    return scored


def _fit_xgb(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    parameters: dict[str, Any],
    *,
    device: str,
    seed: int,
    validation_arrays: (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    ),
    verbose_every: int,
    progress_label: str,
    early_stopping_rounds: int | None,
) -> Any:
    import xgboost as xgb

    class _LabeledProgress(xgb.callback.TrainingCallback):
        def after_iteration(
            self, model: Any, epoch: int, evals_log: dict[str, Any]
        ) -> bool:
            del model
            completed = epoch + 1
            total = int(parameters.get("n_estimators", 1_500))
            if completed != 1 and completed != total and completed % verbose_every:
                return False
            detail = ""
            if evals_log:
                dataset = next(reversed(evals_log))
                metric = next(reversed(evals_log[dataset]))
                value = evals_log[dataset][metric][-1]
                detail = f" {dataset}-{metric}={value:.5f}"
            print(
                f"{progress_label}: tree {completed}/{total}{detail}",
                flush=True,
            )
            return False

    x_train, y_train, qid_train, _, _ = train_arrays
    adjusted = {**parameters, "ndcg_exp_gain": False}
    if verbose_every > 0:
        adjusted["callbacks"] = [_LabeledProgress()]
    model = make_ranker(
        adjusted,
        device=device,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
    )
    kwargs: dict[str, Any] = {"qid": qid_train, "verbose": False}
    if validation_arrays is not None:
        x_valid, y_valid, qid_valid, _, _ = validation_arrays
        kwargs["eval_set"] = [(x_valid, y_valid)]
        kwargs["eval_qid"] = [qid_valid]
    model.fit(x_train, y_train, **kwargs)
    return model


def _write_xgb_test_outputs(
    scored: pd.DataFrame,
    output: Path,
    *,
    bootstrap_draws: int,
    seed: int,
) -> None:
    wide = scored.copy()
    wide["score_source"] = "sealed_paper2_test"
    wide.to_parquet(output / "test_scores_wide.parquet", index=False)
    long = scores_to_long(scored, audience_draw=1)
    long["score_source"] = "sealed_paper2_test"
    long.to_parquet(output / "test_scores_long.parquet", index=False)
    article = evaluate_rank_scores(long, long["score"].to_numpy())
    article.to_parquet(output / "test_article_metrics.parquet", index=False)
    summary = bootstrap_metric_summary(article, draws=bootstrap_draws, seed=seed)
    summary.to_parquet(output / "test_metric_summary.parquet", index=False)
    evaluate_tie_draws(scored).to_parquet(
        output / "test_tie_sensitivity_metrics.parquet", index=False
    )


def _tune_xgb_stage(
    variant: FactorialVariant,
    development: pd.DataFrame,
    *,
    features: list[str],
    fold_columns: dict[int, dict[str, str]],
    embedding_matrix: np.ndarray | None,
    configs: list[dict[str, Any]],
    stage: str,
    history_path: Path,
    device: str,
    seed: int,
    workflow_signature: str,
    progress_name: str,
    verbose_every: int,
    force_recompute: bool,
    folds: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one resumable configuration-by-fold LambdaRank search stage."""
    if not configs:
        raise ValueError(f"{stage} requires at least one configuration")
    tuning_signature = _signature(
        {
            "workflow_signature": workflow_signature,
            "stage": stage,
            "configs": configs,
            "seed": seed,
            "folds": folds,
        }
    )
    valid_keys = {
        (
            config_index,
            fold,
            json.dumps(parameters, sort_keys=True),
            tuning_signature,
        )
        for config_index, parameters in enumerate(configs)
        for fold in range(folds)
    }
    records_by_key: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    if history_path.exists() and not force_recompute:
        existing = pd.read_csv(history_path)
        for record in existing.to_dict("records"):
            key = (
                int(record["config_index"]),
                int(record["fold"]),
                str(record["parameters"]),
                str(record.get("tuning_signature", "")),
            )
            if key in valid_keys:
                records_by_key[key] = record

    total_fits = len(configs) * folds
    for fold in range(folds):
        missing = [
            config_index
            for config_index, parameters in enumerate(configs)
            if (
                config_index,
                fold,
                json.dumps(parameters, sort_keys=True),
                tuning_signature,
            )
            not in records_by_key
        ]
        if not missing:
            print(
                f"{progress_name}: {stage} fold {fold + 1}/{folds} "
                f"all {len(configs)} configurations reused",
                flush=True,
            )
            continue

        print(
            f"{progress_name}: {stage} fold {fold + 1}/{folds} "
            f"building arrays for {len(missing)} remaining configurations",
            flush=True,
        )
        fold_frame = apply_fold_feature_columns(
            development, features, fold_columns.get(fold)
        )
        validation_mask = fold_frame["development_fold"].astype(int) == fold
        train_frame = fold_frame[~validation_mask]
        validation_frame = fold_frame[validation_mask]
        train_arrays = _xgb_arrays(
            train_frame,
            features=features,
            draw_policy=variant.draw_policy,
            embedding_matrix=embedding_matrix,
        )
        validation_arrays = _xgb_arrays(
            validation_frame,
            features=features,
            draw_policy=variant.draw_policy,
            embedding_matrix=embedding_matrix,
        )

        for config_index in missing:
            parameters = configs[config_index]
            serialized = json.dumps(parameters, sort_keys=True)
            key = (config_index, fold, serialized, tuning_signature)
            completed_before = len(records_by_key)
            fit_number = completed_before + 1
            label = (
                f"{progress_name} {stage} fit {fit_number}/{total_fits} "
                f"config {config_index + 1}/{len(configs)} "
                f"fold {fold + 1}/{folds}"
            )
            print(
                f"{label}: fitting LambdaRank with up to "
                f"{int(parameters.get('n_estimators', 1_500))} trees",
                flush=True,
            )
            started = time.monotonic()
            model = _fit_xgb(
                train_arrays,
                parameters,
                device=device,
                seed=seed + config_index * 10 + fold,
                validation_arrays=validation_arrays,
                verbose_every=verbose_every,
                progress_label=label,
                early_stopping_rounds=50,
            )
            elapsed = time.monotonic() - started
            scores = np.asarray(model.predict(validation_arrays[0]))
            scored = _scored_from_predictions(
                validation_frame,
                row_index=validation_arrays[3],
                selector=validation_arrays[4],
                scores=scores,
            )
            macro, audience, curator = _macro_from_scored(
                scored, variant.draw_policy
            )
            try:
                best_iteration = int(model.best_iteration)
            except (AttributeError, TypeError):
                best_iteration = int(parameters.get("n_estimators", 1_500)) - 1
            try:
                early_stopping_score = float(model.best_score)
            except (AttributeError, TypeError):
                early_stopping_score = float("nan")
            records_by_key[key] = {
                "search_stage": stage,
                "config_index": config_index,
                "fold": fold,
                "macro_ndcg_at_k": macro,
                "audience_ndcg_at_k": audience,
                "curator_ndcg_at_k": curator,
                "best_iteration": best_iteration,
                "early_stopping_ndcg_at_10": early_stopping_score,
                "fit_seconds": elapsed,
                "parameters": serialized,
                "tuning_signature": tuning_signature,
                "draw_policy": variant.draw_policy,
            }
            history = pd.DataFrame(records_by_key.values()).sort_values(
                ["config_index", "fold"]
            )
            history.to_csv(history_path, index=False)
            print(
                f"{label}: complete macro_nDCG@k={macro:.4f} "
                f"best_iteration={best_iteration} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )
            del model, scores, scored

        del train_arrays, validation_arrays

    history = pd.DataFrame(records_by_key.values()).sort_values(
        ["config_index", "fold"]
    )
    if len(history) != total_fits:
        raise RuntimeError(
            f"{stage} has {len(history)}/{total_fits} completed configuration-fold fits"
        )
    best, _ = select_best_configuration(history)
    return best, history


def run_xgboost_variant(
    variant: FactorialVariant,
    *,
    model_data_root: Path,
    output_root: Path,
    embedding_root: Path,
    embedding_store: Path | None,
    device: str,
    bootstrap_draws: int,
    progress_every_stories: int,
    verbose_every: int,
    force_recompute: bool,
    scopes: tuple[str, ...],
    run_label: str | None = None,
    seed: int = 20260813,
    broad_search_count: int = 32,
    refinement_top_configs: int = 5,
    development_folds: int = 5,
) -> dict[str, Any]:
    """Tune, fit, and test one XGBoost factorial variant."""
    if variant.family != "xgboost":
        raise ValueError("run_xgboost_variant requires an XGBoost variant")
    progress_name = run_label or variant.variant_id
    if variant.feature_set == "metadata_bge" and embedding_store is None:
        revision = resolve_model_revision(DEFAULT_BGE_MODEL_ID, DEFAULT_BGE_REVISION)
        embedding_store, _ = discover_complete_embedding_store(
            embedding_root,
            model_id=DEFAULT_BGE_MODEL_ID,
            revision=revision,
        )

    results: dict[str, Any] = {}
    for scope_position, scope in enumerate(scopes, start=1):
        print(
            f"{progress_name}: scope {scope_position}/{len(scopes)} ({scope})",
            flush=True,
        )
        frame, features, _, fold_columns, _ = _load_scope_inputs(
            model_data_root, scope
        )
        frame = frame.reset_index(drop=True)
        output = Path(output_root) / variant.variant_id / scope
        output.mkdir(parents=True, exist_ok=True)
        broad_configs = default_search_configs(
            count=broad_search_count, seed=seed
        )
        signature_payload = {
            "version": FACTORIAL_VERSION,
            "variant": variant.as_record(),
            "scope": scope,
            "features": features,
            "search_strategy": "broad_random_then_narrow_grid",
            "broad_search_count": broad_search_count,
            "refinement_top_configs": refinement_top_configs,
            "development_folds": development_folds,
            "broad_configs": broad_configs,
            "seed": seed,
            "model_data_manifest": _file_sha256(
                Path(model_data_root) / "feature_manifest.json"
            ),
            "split": _file_sha256(
                Path(model_data_root) / "master_article_split.parquet"
            ),
            "embedding_store": str(embedding_store) if embedding_store else None,
        }
        workflow_signature = _signature(signature_payload)
        state_path = output / "workflow_state.json"
        complete_path = output / "model_manifest.json"
        required = (
            "development_broad_configurations.json",
            "development_narrow_configurations.json",
            "development_cv_broad_random.csv",
            "development_cv_narrow_grid.csv",
            "development_cv_history.csv",
            "development_cv_summary.csv",
            "development_cv_summary.json",
            "development_cv_metrics.csv",
            "development_cv_training_history.csv",
            "best_parameters.json",
            "development_training_summary.json",
            "development_model.json",
            "test_scores_wide.parquet",
            "test_scores_long.parquet",
            "test_article_metrics.parquet",
            "test_metric_summary.parquet",
            "test_tie_sensitivity_metrics.parquet",
        )
        if not force_recompute and complete_path.exists():
            manifest = json.loads(complete_path.read_text())
            if manifest.get("workflow_signature") == workflow_signature and all(
                (output / name).exists() for name in required
            ):
                print(f"{progress_name}/{scope}: tuned cache reused", flush=True)
                results[scope] = {"status": "reused", "output_root": str(output)}
                continue

        embedding_matrix = None
        if variant.feature_set == "metadata_bge":
            if embedding_store is None:
                raise RuntimeError("Frozen-BGE XGBoost variant lacks an embedding store")
            embedding_matrix = _embedding_cache(
                frame,
                scope=scope,
                embedding_store=embedding_store,
                cache_root=Path(output_root) / "_cache" / "xgboost_bge",
                progress_every_stories=progress_every_stories,
                run_label=progress_name,
            )

        development = frame[frame["split_role"] == "development"]
        test = frame[frame["split_role"] == "paper2_test"]
        _atomic_json(
            broad_configs, output / "development_broad_configurations.json"
        )
        _, broad_history = _tune_xgb_stage(
            variant,
            development,
            features=features,
            fold_columns=fold_columns,
            embedding_matrix=embedding_matrix,
            configs=broad_configs,
            stage="broad_random",
            history_path=output / "development_cv_broad_random.csv",
            device=device,
            seed=seed,
            workflow_signature=workflow_signature,
            progress_name=f"{progress_name}/{scope}",
            verbose_every=verbose_every,
            force_recompute=force_recompute,
            folds=development_folds,
        )
        narrow_configs = refinement_search_configs(
            broad_history, top_configs=refinement_top_configs
        )
        _atomic_json(
            narrow_configs, output / "development_narrow_configurations.json"
        )
        _, narrow_history = _tune_xgb_stage(
            variant,
            development,
            features=features,
            fold_columns=fold_columns,
            embedding_matrix=embedding_matrix,
            configs=narrow_configs,
            stage="narrow_grid",
            history_path=output / "development_cv_narrow_grid.csv",
            device=device,
            seed=seed + 100_000,
            workflow_signature=workflow_signature,
            progress_name=f"{progress_name}/{scope}",
            verbose_every=verbose_every,
            force_recompute=force_recompute,
            folds=development_folds,
        )
        history = pd.concat(
            [broad_history, narrow_history], ignore_index=True
        )
        best, search_summary = select_best_configuration(history)
        history.to_csv(output / "development_cv_history.csv", index=False)
        history.to_csv(
            output / "development_cv_training_history.csv", index=False
        )
        search_summary.to_csv(
            output / "development_cv_summary.csv", index=False
        )
        _atomic_json(best, output / "best_parameters.json")

        winner = search_summary.iloc[0]
        selected_cv = history[
            history["search_stage"].astype(str).eq(str(winner["search_stage"]))
            & history["config_index"].astype(int).eq(int(winner["config_index"]))
            & history["parameters"].astype(str).eq(str(winner["parameters"]))
        ].copy()
        if len(selected_cv) != development_folds:
            raise RuntimeError(
                f"Selected XGBoost configuration has {len(selected_cv)} "
                f"of {development_folds} CV folds"
            )
        selected_cv["selected_configuration"] = True
        selected_cv.sort_values("fold").to_csv(
            output / "development_cv_metrics.csv", index=False
        )
        cv_summary = {
            "search_stage": str(winner["search_stage"]),
            "config_index": int(winner["config_index"]),
            "mean_macro_ndcg_at_k": float(winner["mean_macro_ndcg_at_k"]),
            "sd_macro_ndcg_at_k": float(winner["sd_macro_ndcg_at_k"]),
            "mean_audience_ndcg_at_k": float(
                selected_cv["audience_ndcg_at_k"].mean()
            ),
            "mean_curator_ndcg_at_k": float(
                selected_cv["curator_ndcg_at_k"].mean()
            ),
            "median_best_iteration": float(winner["median_best_iteration"]),
            "completed_folds": int(winner["completed_folds"]),
            "best_parameters": best,
            "search_strategy": "broad_random_then_narrow_grid",
            "evaluated_configurations": int(
                history[["search_stage", "config_index"]]
                .drop_duplicates()
                .shape[0]
            ),
        }
        _atomic_json(cv_summary, output / "development_cv_summary.json")
        previous_state = (
            json.loads(state_path.read_text()) if state_path.exists() else {}
        )
        reusable_final = (
            not force_recompute
            and previous_state.get("workflow_signature") == workflow_signature
            and previous_state.get("phase") in {"test_inference", "complete"}
            and previous_state.get("best_parameters") == best
            and (output / "development_model.json").exists()
            and (output / "development_training_summary.json").exists()
        )
        if not reusable_final:
            _atomic_json(
                {
                    "workflow_signature": workflow_signature,
                    "phase": "tuning_complete",
                    "best_parameters": best,
                    "cv": cv_summary,
                },
                state_path,
            )
        print(
            f"{progress_name}/{scope}: tuning complete "
            f"best_CV_macro_nDCG@k={cv_summary['mean_macro_ndcg_at_k']:.4f} "
            f"({cv_summary['search_stage']} config "
            f"{cv_summary['config_index'] + 1})",
            flush=True,
        )

        model_path = output / "development_model.json"
        if reusable_final:
            model = make_ranker(
                {**best, "ndcg_exp_gain": False},
                device=device,
                seed=seed + 10_000,
                early_stopping_rounds=None,
            )
            model.load_model(model_path)
            fit_seconds = float(
                json.loads(
                    (output / "development_training_summary.json").read_text()
                )["fit_seconds"]
            )
            print(f"{progress_name}/{scope}: tuned final model reused", flush=True)
        else:
            print(
                f"{progress_name}/{scope}: final tuned development fit building arrays",
                flush=True,
            )
            final_frame = apply_fold_feature_columns(development, features, None)
            development_arrays = _xgb_arrays(
                final_frame,
                features=features,
                draw_policy=variant.draw_policy,
                embedding_matrix=embedding_matrix,
            )
            print(
                f"{progress_name}/{scope}: final tuned development fit "
                f"{best['n_estimators']} LambdaRank trees",
                flush=True,
            )
            started = time.monotonic()
            model = _fit_xgb(
                development_arrays,
                best,
                device=device,
                seed=seed + 10_000,
                validation_arrays=None,
                verbose_every=verbose_every,
                progress_label=(
                    f"{progress_name}/{scope} final tuned development fit"
                ),
                early_stopping_rounds=None,
            )
            fit_seconds = time.monotonic() - started
            model.save_model(model_path)
            _atomic_json(
                {
                    "fit_seconds": fit_seconds,
                    "n_development_candidates": len(development),
                    "n_estimators": int(best["n_estimators"]),
                    "parameters": best,
                    "selected_cv": cv_summary,
                },
                output / "development_training_summary.json",
            )
            _atomic_json(
                {
                    "workflow_signature": workflow_signature,
                    "phase": "test_inference",
                    "best_parameters": best,
                    "cv": cv_summary,
                },
                state_path,
            )
            del development_arrays

        print(f"{progress_name}/{scope}: sealed-test inference", flush=True)
        test_arrays = _xgb_arrays(
            test,
            features=features,
            draw_policy=variant.draw_policy,
            embedding_matrix=embedding_matrix,
        )
        test_scores = np.asarray(model.predict(test_arrays[0]))
        test_scored = _scored_from_predictions(
            test,
            row_index=test_arrays[3],
            selector=test_arrays[4],
            scores=test_scores,
        )
        _write_xgb_test_outputs(
            test_scored,
            output,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
        )
        manifest = {
            **signature_payload,
            "workflow_signature": workflow_signature,
            "status": "complete",
            "reported_scores": "sealed_paper2_test",
            "cv": cv_summary,
            "best_parameters": best,
            "training_fit_seconds": fit_seconds,
            "output_root": str(output),
            "ten_draw_training_contract": (
                "integer selection counts with linear-gain LambdaRank"
                if variant.draw_policy == "mean10"
                else "audience_selected_draw_01"
            ),
        }
        _atomic_json(manifest, complete_path)
        _atomic_json(
            {
                "workflow_signature": workflow_signature,
                "phase": "complete",
                "best_parameters": best,
                "cv": cv_summary,
            },
            state_path,
        )
        results[scope] = {
            "status": "computed",
            "output_root": str(output),
            "best_cv_macro_ndcg_at_k": cv_summary["mean_macro_ndcg_at_k"],
        }
        del model, test_arrays, test_scores, test_scored
    return results

def _variant_metadata(
    variant: FactorialVariant, scope: str
) -> dict[str, str]:
    return {**variant.as_record(), "scope": scope}


def _write_records_csv(
    rows: list[dict[str, Any]], path: Path, *, empty_columns: list[str]
) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=empty_columns)
    frame.to_csv(path, index=False)


def refresh_experiment_summaries(
    output_root: Path,
    variants: Iterable[FactorialVariant],
    *,
    scopes: tuple[str, ...],
    statuses: dict[str, str] | None = None,
) -> None:
    output_root = Path(output_root)
    manifest_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for variant in variants:
        for scope in scopes:
            root = output_root / variant.variant_id / scope
            metadata = _variant_metadata(variant, scope)
            manifest_path = root / "model_manifest.json"
            status = (
                statuses.get(variant.variant_id, "pending")
                if statuses is not None
                else "pending"
            )
            if manifest_path.exists():
                status = "complete"
            manifest_rows.append(
                {**metadata, "status": status, "output_root": str(root)}
            )
            cv_path = root / "development_cv_metrics.csv"
            if cv_path.exists():
                frame = pd.read_csv(cv_path)
                for key, value in metadata.items():
                    frame[key] = value
                cv_rows.extend(frame.to_dict("records"))
            train_path = root / "development_final_training_history.csv"
            if train_path.exists():
                frame = pd.read_csv(train_path)
                for key, value in metadata.items():
                    frame[key] = value
                training_rows.extend(frame.to_dict("records"))
            xgb_train_path = root / "development_training_summary.json"
            if xgb_train_path.exists():
                training_rows.append(
                    {**metadata, **json.loads(xgb_train_path.read_text())}
                )
            tie_path = root / "test_tie_sensitivity_metrics.parquet"
            if tie_path.exists():
                ties = pd.read_parquet(tie_path)
                grouped = (
                    ties.groupby(
                        ["selector", "audience_tie_draw"], as_index=False
                    )
                    .agg(
                        ndcg_at_k=("ndcg_at_k", "mean"),
                        top_k_overlap=("top_k_overlap", "mean"),
                        jaccard=("jaccard", "mean"),
                        mean_selected_rank=("mean_selected_rank", "mean"),
                    )
                )
                for key, value in metadata.items():
                    grouped[key] = value
                test_rows.extend(grouped.to_dict("records"))
    pd.DataFrame(manifest_rows).to_csv(
        output_root / "experiment_variants.csv", index=False
    )
    _write_records_csv(
        cv_rows,
        output_root / "development_cv_results.csv",
        empty_columns=["variant_id", "family", "scope", "fold", "macro_ndcg_at_k"],
    )
    _write_records_csv(
        training_rows,
        output_root / "development_training_results.csv",
        empty_columns=["variant_id", "family", "scope", "epoch", "training_loss"],
    )
    _write_records_csv(
        test_rows,
        output_root / "sealed_test_results.csv",
        empty_columns=["variant_id", "family", "scope", "selector", "audience_tie_draw", "ndcg_at_k"],
    )


def run_factorial_experiment(
    *,
    model_data_root: Path,
    data_root: Path,
    embedding_root: Path,
    embedding_store: Path | None,
    output_root: Path,
    neural_device: str,
    xgb_device: str,
    bootstrap_draws: int,
    progress_every_stories: int,
    xgb_verbose_every: int,
    training_mode: str,
    force_recompute: bool,
    resume: bool,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    scopes: tuple[str, ...],
    keep_going: bool,
) -> dict[str, str]:
    all_variants = factorial_variants()
    selected = select_variants(
        all_variants, include=include, exclude=exclude
    )
    selected_ids = {variant.variant_id for variant in selected}
    output_root = Path(output_root)
    statuses = {
        variant.variant_id: (
            "complete"
            if all(
                (output_root / variant.variant_id / scope / "model_manifest.json").exists()
                for scope in scopes
            )
            else ("pending" if variant.variant_id in selected_ids else "filtered")
        )
        for variant in all_variants
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        {
            "version": FACTORIAL_VERSION,
            "total_variants": len(all_variants),
            "selected_variants": len(selected),
            "include": include,
            "exclude": exclude,
            "scopes": scopes,
            "variants": [variant.as_record() for variant in all_variants],
        },
        Path(output_root) / "experiment_manifest.json",
    )
    selected_position = 0
    for overall_position, variant in enumerate(all_variants, start=1):
        if variant.variant_id not in selected_ids:
            filter_status = (
                "CACHED COMPLETE; FILTERED"
                if statuses[variant.variant_id] == "complete"
                else "FILTERED"
            )
            print(
                f"--- model {overall_position}/{len(all_variants)} "
                f"{filter_status}: "
                f"{variant.variant_id} ---",
                flush=True,
            )
            continue
        selected_position += 1
        print(
            f"\n=== model {overall_position}/{len(all_variants)} "
            f"(selected {selected_position}/{len(selected)}): "
            f"{variant.variant_id} ===",
            flush=True,
        )
        print(json.dumps(variant.as_record(), sort_keys=True), flush=True)
        statuses[variant.variant_id] = "running"
        model_progress = (
            f"model {overall_position}/{len(all_variants)} {variant.variant_id}"
        )
        refresh_experiment_summaries(
            output_root, all_variants, scopes=scopes, statuses=statuses
        )
        try:
            if variant.family == "neural":
                recipe = neural_recipe(variant)
                run_neural_ranker_workflow(
                    model_data_root,
                    data_root,
                    Path(output_root) / variant.variant_id,
                    approach=recipe.approach,
                    embedding_root=embedding_root,
                    embedding_store=embedding_store,
                    device=neural_device,
                    bootstrap_draws=bootstrap_draws,
                    force_recompute=force_recompute,
                    resume=resume,
                    progress_every_stories=progress_every_stories,
                    training_mode=training_mode,
                    recipe=recipe,
                    scopes=scopes,
                    run_label=model_progress,
                )
            else:
                run_xgboost_variant(
                    variant,
                    model_data_root=model_data_root,
                    output_root=output_root,
                    embedding_root=embedding_root,
                    embedding_store=embedding_store,
                    device=xgb_device,
                    bootstrap_draws=bootstrap_draws,
                    progress_every_stories=progress_every_stories,
                    verbose_every=xgb_verbose_every,
                    force_recompute=force_recompute,
                    scopes=scopes,
                    run_label=model_progress,
                )
            statuses[variant.variant_id] = "complete"
            print(
                f"=== model {overall_position}/{len(all_variants)} COMPLETE: "
                f"{variant.variant_id} ===",
                flush=True,
            )
        except Exception as error:
            statuses[variant.variant_id] = "failed"
            _atomic_json(
                {
                    "variant": variant.as_record(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                Path(output_root) / variant.variant_id / "failure.json",
            )
            print(
                f"=== model {overall_position}/{len(all_variants)} FAILED: "
                f"{variant.variant_id}: {error} ===",
                flush=True,
            )
            refresh_experiment_summaries(
                output_root, all_variants, scopes=scopes, statuses=statuses
            )
            if not keep_going:
                raise
        refresh_experiment_summaries(
            output_root, all_variants, scopes=scopes, statuses=statuses
        )
    print(
        f"\n=== factorial run finished: "
        f"{sum(value == 'complete' for value in statuses.values())}/"
        f"{len(all_variants)} complete, "
        f"{sum(value == 'filtered' for value in statuses.values())} filtered, "
        f"{sum(value == 'failed' for value in statuses.values())} failed ===",
        flush=True,
    )
    return statuses
