"""Model explanations for the frozen Paper 1 ranker winners.

This module deliberately runs after winner freezing.  Explanations are computed
on the sealed ``paper2_test`` comments with a development-only background set;
notebook 12 consumes the exported, story-aggregated summaries for comparison
with FORUM.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


SHAP_WORKFLOW_VERSION = 2
SELECTORS = ("audience", "curator")
TEXT_FEATURE = "text_bge"
SHAP_CHUNK_ROWS = 500


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _require_shap() -> Any:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on analysis environment
        raise RuntimeError(
            "SHAP calculation requires the optional analysis dependency; "
            "install requirements-analysis.txt"
        ) from exc
    return shap


def _deterministic_sample(frame: pd.DataFrame, limit: int, *, seed: int) -> pd.DataFrame:
    """Sample rows deterministically while retaining broad story coverage."""
    if limit < 1:
        raise ValueError("SHAP sample limits must be positive")
    ordered = frame.sort_values(["story_id", "comment_id"]).copy()
    if len(ordered) <= limit:
        return ordered

    # Keep one deterministic candidate from each story first. The remaining
    # slots are filled at evenly spaced positions so large stories do not crowd
    # the background or explanation sample.
    first_by_story = ordered.groupby("story_id", sort=True).head(1).index.to_numpy()
    if len(first_by_story) >= limit:
        positions = np.linspace(0, len(first_by_story) - 1, limit, dtype=int)
        return ordered.loc[first_by_story[positions]]

    remaining = ordered.drop(index=first_by_story)
    slots = limit - len(first_by_story)
    positions = np.linspace(0, len(remaining) - 1, slots, dtype=int)
    chosen = remaining.iloc[positions]
    selected = pd.concat([ordered.loc[first_by_story], chosen], axis=0)
    return selected.sort_values(["story_id", "comment_id"])


def _load_embedding_matrix(
    full_frame: pd.DataFrame,
    *,
    embedding_store: Path,
    factorial_root: Path,
    scope: str,
    run_label: str,
) -> np.ndarray:
    from commentgap_analysis.factorial_rankers import _embedding_cache

    return _embedding_cache(
        full_frame.reset_index(drop=True),
        scope=scope,
        embedding_store=Path(embedding_store),
        cache_root=Path(factorial_root) / "_cache" / "xgboost_bge",
        progress_every_stories=100,
        run_label=run_label,
    )


def _matrix_for_frame(
    frame: pd.DataFrame,
    features: list[str],
    *,
    embedding_matrix: np.ndarray | None,
    selector: int,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    ordered = frame.sort_values(["story_id", "comment_id"]).copy()
    metadata = ordered[features].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(metadata).all():
        raise ValueError("SHAP metadata input contains non-finite values")
    parts = [metadata]
    names = list(features)
    if embedding_matrix is not None:
        positions = ordered.index.to_numpy(dtype=np.int64)
        text = np.asarray(embedding_matrix[positions], dtype=np.float32)
        if text.shape != (len(ordered), 1024) or not np.isfinite(text).all():
            raise ValueError("SHAP BGE input must have shape (n, 1024) and be finite")
        parts.append(text)
        names.extend([f"bge_{index:04d}" for index in range(text.shape[1])])
    parts.append(np.full((len(ordered), 1), selector, dtype=np.float32))
    names.append("selector_code")
    return np.concatenate(parts, axis=1), names, ordered


def _collapse_feature_values(
    values: np.ndarray,
    *,
    features: list[str],
    has_text: bool,
    has_selector_code: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    expected_width = len(features) + (1024 if has_text else 0) + int(has_selector_code)
    if values.shape[1] != expected_width:
        raise ValueError(
            f"SHAP output has {values.shape[1]} columns; expected {expected_width}"
        )
    columns = [*features]
    groups = ["metadata"] * len(features)
    parts = [values[:, : len(features)]]
    offset = len(features)
    if has_text:
        parts.append(values[:, offset : offset + 1024].sum(axis=1, keepdims=True))
        columns.append(TEXT_FEATURE)
        groups.append("text")
        offset += 1024
    if has_selector_code:
        parts.append(values[:, offset : offset + 1])
        columns.append("selector_code")
        groups.append("selector")
    return np.concatenate(parts, axis=1), columns, groups


def _normalise_explainer_values(values: Any, *, rows: int) -> np.ndarray:
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("SHAP output unexpectedly contains multiple outputs")
        values = values[0]
    values = np.asarray(values)
    if values.shape[0] != rows:
        raise ValueError(f"SHAP output has {values.shape[0]} rows; expected {rows}")
    return values


def _shap_values_with_progress(
    explainer: Any,
    matrix: Any,
    *,
    label: str,
    seed: int,
    nsamples: int | None = None,
    chunk_rows: int = SHAP_CHUNK_ROWS,
) -> np.ndarray:
    """Calculate SHAP values in visible, deterministic chunks.

    SHAP's gradient implementation iterates over test rows internally and does
    not expose a progress callback. Chunking keeps the calculation observable
    during the multi-hour full run while also bounding the temporary arrays.
    """
    if chunk_rows < 1:
        raise ValueError("SHAP chunk_rows must be positive")
    total = len(matrix)
    if total < 1:
        raise ValueError("SHAP explanation input must not be empty")
    chunks: list[np.ndarray] = []
    started = time.perf_counter()
    for chunk_index, start in enumerate(range(0, total, chunk_rows)):
        stop = min(start + chunk_rows, total)
        kwargs: dict[str, Any] = {}
        if nsamples is not None:
            kwargs["nsamples"] = nsamples
            kwargs["rseed"] = seed + chunk_index
        values = _normalise_explainer_values(
            explainer.shap_values(matrix[start:stop], **kwargs),
            rows=stop - start,
        )
        chunks.append(values)
        elapsed = time.perf_counter() - started
        print(
            f"[SHAP] {label}: {stop:,}/{total:,} rows "
            f"({100 * stop / total:.1f}%), elapsed={elapsed / 60:.1f} min",
            flush=True,
        )
    return np.concatenate(chunks, axis=0)


def _expected_value(explainer: Any) -> float:
    value = getattr(explainer, "expected_value", np.nan)
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0]) if len(array) else float("nan")


def _records_from_values(
    keys: pd.DataFrame,
    values: np.ndarray,
    columns: list[str],
    groups: list[str],
    *,
    scope: str,
    family: str,
    model_id: str,
    feature_set: str,
    selector: str,
    expected_value: float,
) -> pd.DataFrame:
    if len(keys) != len(values):
        raise ValueError("SHAP keys and values have different row counts")
    output = keys[["story_id", "comment_id"]].copy()
    output["scope"] = scope
    output["model_family"] = family
    output["model_id"] = model_id
    output["feature_set"] = feature_set
    output["selector"] = selector
    output["split_role"] = "paper2_test"
    output["expected_value"] = expected_value
    for index, (feature, _group) in enumerate(zip(columns, groups, strict=True)):
        output[f"shap_{feature}"] = values[:, index]
    return output


def _winner_context(
    winner: dict[str, Any], *, model_data_root: Path, factorial_root: Path
) -> dict[str, Any]:
    from commentgap_analysis.neural_ranking import (
        _load_scope_inputs,
        apply_fold_feature_columns,
    )

    family = str(winner["family"])
    scope = str(winner["scope"])
    model_id = str(winner["variant_id"])
    model_root = Path(factorial_root) / model_id / scope
    manifest_path = model_root / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    full_frame, features, _, _, _ = _load_scope_inputs(model_data_root, scope)
    full_frame = apply_fold_feature_columns(full_frame, features, None)
    test = full_frame[full_frame["split_role"].eq("paper2_test")].copy()
    development = full_frame[full_frame["split_role"].eq("development")].copy()
    if test.empty or development.empty:
        raise ValueError(f"Winner {model_id}/{scope} lacks development or test rows")
    return {
        "winner": winner,
        "family": family,
        "scope": scope,
        "model_id": model_id,
        "feature_set": str(winner["feature_set"]),
        "model_root": model_root,
        "manifest": manifest,
        "full_frame": full_frame,
        "development": development,
        "test": test,
        "features": features,
    }


def _calculate_xgb(
    context: dict[str, Any],
    *,
    factorial_root: Path,
    test_rows: int,
    background_rows: int,
    seed: int,
    chunk_rows: int,
) -> pd.DataFrame:
    shap = _require_shap()
    from commentgap_analysis.ranking import make_ranker

    root = context["model_root"]
    parameters = json.loads((root / "best_parameters.json").read_text())
    model = make_ranker(
        {**parameters, "ndcg_exp_gain": False},
        device="cpu",
        seed=20260813,
        early_stopping_rounds=None,
    )
    model.load_model(root / "development_model.json")
    embedding_matrix = None
    if context["feature_set"] == "metadata_bge":
        embedding_store = context["manifest"].get("embedding_store")
        if not embedding_store:
            raise RuntimeError(f"{context['model_id']}/{context['scope']} lacks its embedding store")
        embedding_matrix = _load_embedding_matrix(
            context["full_frame"],
            embedding_store=Path(embedding_store),
            factorial_root=factorial_root,
            scope=context["scope"],
            run_label=f"stage9-shap-{context['model_id']}",
        )
    test = _deterministic_sample(context["test"], test_rows, seed=seed)
    background = _deterministic_sample(context["development"], background_rows, seed=seed + 1)
    output: list[pd.DataFrame] = []
    for selector_index, selector in enumerate(SELECTORS):
        test_matrix, _, ordered_test = _matrix_for_frame(
            test,
            context["features"],
            embedding_matrix=embedding_matrix,
            selector=selector_index,
        )
        background_matrix, _, _ = _matrix_for_frame(
            background,
            context["features"],
            embedding_matrix=embedding_matrix,
            selector=selector_index,
        )
        explainer = shap.TreeExplainer(
            model.get_booster(),
            data=background_matrix,
            feature_perturbation="interventional",
            model_output="raw",
        )
        values = _shap_values_with_progress(
            explainer,
            test_matrix,
            label=f"{context['model_id']} {context['scope']} {selector}",
            seed=seed + selector_index,
            chunk_rows=chunk_rows,
        )
        collapsed, columns, groups = _collapse_feature_values(
            values,
            features=context["features"],
            has_text=embedding_matrix is not None,
            has_selector_code=True,
        )
        output.append(
            _records_from_values(
                ordered_test,
                collapsed,
                columns,
                groups,
                scope=context["scope"],
                family=context["family"],
                model_id=context["model_id"],
                feature_set=context["feature_set"],
                selector=selector,
                expected_value=_expected_value(explainer),
            )
        )
    return pd.concat(output, ignore_index=True)


def _calculate_neural(
    context: dict[str, Any],
    *,
    factorial_root: Path,
    test_rows: int,
    background_rows: int,
    nsamples: int,
    seed: int,
    chunk_rows: int,
) -> pd.DataFrame:
    shap = _require_shap()
    import torch

    from commentgap_analysis.neural_ranking import (
        FeatureScaler,
        NeuralTrainingRecipe,
        _load_model_checkpoint,
        _make_model,
    )

    root = context["model_root"]
    recipe = NeuralTrainingRecipe(**context["manifest"]["recipe"])
    scaler = FeatureScaler.from_dict(
        json.loads((root / "development_scaler.json").read_text())
    )
    embedding_matrix = None
    if context["feature_set"] == "metadata_bge":
        embedding_store = context["manifest"].get("embedding_store")
        if not embedding_store:
            raise RuntimeError(f"{context['model_id']}/{context['scope']} lacks its embedding store")
        embedding_matrix = _load_embedding_matrix(
            context["full_frame"],
            embedding_store=Path(embedding_store),
            factorial_root=factorial_root,
            scope=context["scope"],
            run_label=f"stage9-shap-{context['model_id']}",
        )
    test = _deterministic_sample(context["test"], test_rows, seed=seed)
    background = _deterministic_sample(context["development"], background_rows, seed=seed + 1)

    def neural_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        ordered = frame.sort_values(["story_id", "comment_id"]).copy()
        metadata = scaler.transform(ordered)
        parts = [metadata]
        if embedding_matrix is not None:
            positions = ordered.index.to_numpy(dtype=np.int64)
            parts.append(np.asarray(embedding_matrix[positions], dtype=np.float32))
        return np.concatenate(parts, axis=1), ordered

    test_matrix, ordered_test = neural_matrix(test)
    background_matrix, _ = neural_matrix(background)
    metadata_width = len(context["features"])
    has_text = embedding_matrix is not None
    output: list[pd.DataFrame] = []
    torch.manual_seed(seed)
    model = _make_model(metadata_width, recipe, "cpu")
    _load_model_checkpoint(model, root / "development_model.pt", recipe.approach)
    model.eval()

    class FixedSelectorModel(torch.nn.Module):
        def __init__(self, ranker: Any, selector_index: int):
            super().__init__()
            self.ranker = ranker
            self.selector_index = selector_index

        def forward(self, values: Any) -> Any:
            metadata = values[:, :metadata_width]
            payload = values[:, metadata_width:] if has_text else None
            selector = torch.full(
                (values.shape[0],), self.selector_index, dtype=torch.long, device=values.device
            )
            # GradientExplainer's PyTorch backend indexes model outputs as
            # ``outputs[:, output_index]``.  The ranker API intentionally
            # returns scalar scores as ``(batch,)`` for training/inference,
            # so expose the same scalar as a single-output matrix here.
            return self.ranker.score(payload, metadata, selector).reshape(-1, 1)

    background_tensor = torch.as_tensor(background_matrix, dtype=torch.float32)
    test_tensor = torch.as_tensor(test_matrix, dtype=torch.float32)
    for selector_index, selector in enumerate(SELECTORS):
        wrapper = FixedSelectorModel(model, selector_index)
        wrapper.eval()
        explainer = shap.GradientExplainer(wrapper, background_tensor)
        values = _shap_values_with_progress(
            explainer,
            test_tensor,
            label=f"{context['model_id']} {context['scope']} {selector}",
            seed=seed + selector_index,
            nsamples=nsamples,
            chunk_rows=chunk_rows,
        )
        collapsed, columns, groups = _collapse_feature_values(
            values,
            features=context["features"],
            has_text=has_text,
            has_selector_code=False,
        )
        output.append(
            _records_from_values(
                ordered_test,
                collapsed,
                columns,
                groups,
                scope=context["scope"],
                family=context["family"],
                model_id=context["model_id"],
                feature_set=context["feature_set"],
                selector=selector,
                expected_value=_expected_value(explainer),
            )
        )
    return pd.concat(output, ignore_index=True)


def _story_feature_values(values: pd.DataFrame) -> pd.DataFrame:
    shap_columns = [
        column for column in values.columns
        if column.startswith("shap_") and column != "shap_expected_value"
    ]
    shap_columns = [column for column in shap_columns if column != "shap_selector_code"]
    if not shap_columns:
        raise ValueError("SHAP output has no substantive features")
    keys = ["story_id", "scope", "model_family", "model_id", "feature_set", "selector"]
    story_frames: list[pd.DataFrame] = []
    for column in shap_columns:
        feature = column.removeprefix("shap_")
        working = values[keys + ["comment_id", column]].rename(
            columns={column: "shap_value"}
        )
        working["feature"] = feature
        working["abs_shap_value"] = working["shap_value"].abs()
        story_frames.append(
            working.groupby(keys + ["feature"], as_index=False, observed=True).agg(
                mean_shap=("shap_value", "mean"),
                mean_abs_shap=("abs_shap_value", "mean"),
                n_comments=("comment_id", "nunique"),
            )
        )
    return pd.concat(story_frames, ignore_index=True)


def _story_aggregate(values: pd.DataFrame) -> pd.DataFrame:
    story = _story_feature_values(values)
    summary_keys = ["scope", "model_family", "model_id", "feature_set", "feature"]
    summary = story.groupby(summary_keys, as_index=False, observed=True).agg(
        mean_shap=("mean_shap", "mean"),
        mean_abs_shap=("mean_abs_shap", "mean"),
        n_stories=("story_id", "nunique"),
        sampled_comments=("n_comments", "sum"),
    )
    # Collapse each selector across stories before pivoting.  Pivoting the
    # story-level rows directly with ``first`` silently dropped every story
    # after the first one for each feature.
    selector_summary = story.groupby(
        summary_keys + ["selector"], as_index=False, observed=True
    ).agg(
        mean_shap=("mean_shap", "mean"),
        mean_abs_shap=("mean_abs_shap", "mean"),
    )
    wide = selector_summary.pivot_table(
        index=summary_keys,
        columns="selector",
        values=["mean_shap", "mean_abs_shap"],
        aggfunc="first",
    ).reset_index()
    wide.columns = [
        "_".join(str(part) for part in column if str(part) != "").rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in wide.columns
    ]
    summary = summary.merge(wide, on=summary_keys, how="left", validate="one_to_one")
    for column in ("mean_shap", "mean_abs_shap"):
        audience = f"{column}_audience"
        curator = f"{column}_curator"
        if audience not in summary or curator not in summary:
            raise ValueError("SHAP summary must contain both audience and curator selectors")
        summary[f"{column}_gap"] = summary[curator] - summary[audience]
    summary["aggregation"] = "mean over comments within story, then equal mean over stories"
    return summary.sort_values(summary_keys).reset_index(drop=True)


def summarize_shap_values(values: pd.DataFrame) -> pd.DataFrame:
    """Return the story-aggregated SHAP summary used by the reporting plots."""
    return _story_aggregate(values)


def summarize_shap_story_variability(
    values: pd.DataFrame,
    *,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
) -> pd.DataFrame:
    """Return per-story SHAP spread for the reporting plot error bars.

    The returned bounds are the central interval across story-level mean SHAP
    values.  For the selector gap, the difference is formed within each story
    before taking quantiles, preserving the paired editor--audience structure.
    """
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    story = _story_feature_values(values)
    variant_keys = ["scope", "model_family", "model_id", "feature_set"]
    records: list[dict[str, Any]] = []
    for variant, group in story.groupby(variant_keys, observed=True, sort=False):
        audience = group[group["selector"].eq("audience")].pivot(
            index="story_id", columns="feature", values="mean_shap"
        )
        curator = group[group["selector"].eq("curator")].pivot(
            index="story_id", columns="feature", values="mean_shap"
        )
        features = audience.columns.intersection(curator.columns).sort_values()
        stories = audience.index.intersection(curator.index)
        for feature in features:
            paired = pd.concat(
                [audience.loc[stories, feature], curator.loc[stories, feature]],
                axis=1,
                keys=["audience", "curator"],
            ).dropna()
            if paired.empty:
                continue
            gap = paired["curator"] - paired["audience"]
            records.append({
                **dict(zip(variant_keys, variant)),
                "feature": feature,
                "mean_shap_audience_story_q25": paired["audience"].quantile(lower_quantile),
                "mean_shap_audience_story_q75": paired["audience"].quantile(upper_quantile),
                "mean_shap_curator_story_q25": paired["curator"].quantile(lower_quantile),
                "mean_shap_curator_story_q75": paired["curator"].quantile(upper_quantile),
                "mean_shap_gap_story_q25": gap.quantile(lower_quantile),
                "mean_shap_gap_story_q75": gap.quantile(upper_quantile),
                "story_count": len(paired),
                "story_lower_quantile": lower_quantile,
                "story_upper_quantile": upper_quantile,
            })
    return pd.DataFrame(records)


def bootstrap_shap_summary(
    values: pd.DataFrame,
    *,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 20260911,
) -> pd.DataFrame:
    """Estimate paired story-level confidence intervals for mean SHAP gaps.

    Comments are first averaged within story, matching the main SHAP summary.
    Stories are then resampled with replacement within each model variant;
    audience and curator values use the same draws so the gap interval is paired.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    story = _story_feature_values(values)
    variant_keys = ["scope", "model_family", "model_id", "feature_set"]
    alpha = (1 - confidence_level) / 2
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for variant, group in story.groupby(variant_keys, observed=True, sort=False):
        audience = group[group["selector"].eq("audience")].pivot(
            index="story_id", columns="feature", values="mean_shap"
        )
        curator = group[group["selector"].eq("curator")].pivot(
            index="story_id", columns="feature", values="mean_shap"
        )
        features = audience.columns.intersection(curator.columns).sort_values()
        stories = audience.index.intersection(curator.index)
        if len(features) == 0 or len(stories) == 0:
            continue
        audience_matrix = audience.loc[stories, features].to_numpy(dtype=float)
        curator_matrix = curator.loc[stories, features].to_numpy(dtype=float)
        finite_features = np.isfinite(audience_matrix).all(axis=0) & np.isfinite(curator_matrix).all(axis=0)
        features = features[finite_features]
        audience_matrix = audience_matrix[:, finite_features]
        curator_matrix = curator_matrix[:, finite_features]
        if len(features) == 0:
            continue
        n_stories = len(stories)
        probabilities = np.full(n_stories, 1 / n_stories)
        draws = rng.multinomial(n_stories, probabilities, size=n_bootstrap)
        audience_boot = draws @ audience_matrix / n_stories
        curator_boot = draws @ curator_matrix / n_stories
        gap_boot = curator_boot - audience_boot
        audience_low, audience_high = np.quantile(
            audience_boot, [alpha, 1 - alpha], axis=0
        )
        curator_low, curator_high = np.quantile(
            curator_boot, [alpha, 1 - alpha], axis=0
        )
        gap_low, gap_high = np.quantile(gap_boot, [alpha, 1 - alpha], axis=0)
        for feature_index, feature in enumerate(features):
            records.append({
                **dict(zip(variant_keys, variant)),
                "feature": feature,
                "mean_shap_audience_conf_low": audience_low[feature_index],
                "mean_shap_audience_conf_high": audience_high[feature_index],
                "mean_shap_curator_conf_low": curator_low[feature_index],
                "mean_shap_curator_conf_high": curator_high[feature_index],
                "mean_shap_gap_conf_low": gap_low[feature_index],
                "mean_shap_gap_conf_high": gap_high[feature_index],
                "bootstrap_stories": n_stories,
                "bootstrap_draws": n_bootstrap,
                "confidence_level": confidence_level,
            })
    return pd.DataFrame(records)


def run_shap_explanations(
    *,
    model_data_root: Path,
    factorial_root: Path,
    winners: pd.DataFrame,
    output_root: Path,
    scopes: Iterable[str] = ("all",),
    test_rows: int = 50_000,
    background_rows: int = 2_048,
    nsamples: int = 100,
    seed: int = 20260813,
    force_recompute: bool = False,
    chunk_rows: int = SHAP_CHUNK_ROWS,
) -> dict[str, Any]:
    """Calculate cached SHAP values and story-aggregated winner summaries."""
    if test_rows < 1 or background_rows < 1 or nsamples < 1 or chunk_rows < 1:
        raise ValueError("SHAP row, nsample, and chunk limits must be positive")
    scopes = tuple(dict.fromkeys(str(scope) for scope in scopes))
    selected = winners[winners["scope"].astype(str).isin(scopes)].copy()
    if selected.empty:
        raise ValueError("No frozen winners match the requested SHAP scopes")
    cache_root = Path(output_root) / "cache" / "shap"
    cache_root.mkdir(parents=True, exist_ok=True)
    value_frames: list[pd.DataFrame] = []
    cache_records: list[dict[str, Any]] = []
    for index, winner in enumerate(selected.to_dict("records"), start=1):
        context = _winner_context(
            winner, model_data_root=Path(model_data_root), factorial_root=Path(factorial_root)
        )
        model_root = context["model_root"]
        winner_label = f"{context['family']} {context['model_id']} {context['scope']}"
        print(f"[SHAP] winner {index}/{len(selected)}: {winner_label}", flush=True)
        model_path = model_root / (
            "development_model.json" if context["family"] == "xgboost" else "development_model.pt"
        )
        choice_path = Path(model_data_root) / f"choice_set_{context['scope']}.parquet"
        signature_payload = {
            "version": SHAP_WORKFLOW_VERSION,
            "model_id": context["model_id"],
            "family": context["family"],
            "scope": context["scope"],
            "feature_set": context["feature_set"],
            "model_sha256": _sha256(model_path),
            "choice_set_sha256": _sha256(choice_path),
            "test_rows": test_rows,
            "background_rows": background_rows,
            "nsamples": nsamples,
            "seed": seed + index,
            "chunk_rows": chunk_rows,
        }
        signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
        cache_dir = cache_root / context["model_id"] / context["scope"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "test_shap_values.parquet"
        manifest_path = cache_dir / "shap_manifest.json"
        reusable = False
        if not force_recompute and cache_path.exists() and manifest_path.exists():
            cached_manifest = json.loads(manifest_path.read_text())
            reusable = cached_manifest.get("signature") == signature
        if reusable:
            values = pd.read_parquet(cache_path)
            print(f"[SHAP] reused cache: {winner_label}", flush=True)
        elif context["family"] == "xgboost":
            values = _calculate_xgb(
                context,
                factorial_root=Path(factorial_root),
                test_rows=test_rows,
                background_rows=background_rows,
                seed=seed + index,
                chunk_rows=chunk_rows,
            )
            values.to_parquet(cache_path, index=False)
            manifest_path.write_text(
                json.dumps({**signature_payload, "signature": signature}, indent=2, sort_keys=True) + "\n"
            )
            print(f"[SHAP] completed: {winner_label}", flush=True)
        else:
            values = _calculate_neural(
                context,
                factorial_root=Path(factorial_root),
                test_rows=test_rows,
                background_rows=background_rows,
                nsamples=nsamples,
                seed=seed + index,
                chunk_rows=chunk_rows,
            )
            values.to_parquet(cache_path, index=False)
            manifest_path.write_text(
                json.dumps({**signature_payload, "signature": signature}, indent=2, sort_keys=True) + "\n"
            )
            print(f"[SHAP] completed: {winner_label}", flush=True)
        value_frames.append(values)
        cache_records.append({**signature_payload, "signature": signature, "cache_path": str(cache_path), "reused": reusable})

    values = pd.concat(value_frames, ignore_index=True)
    summary = _story_aggregate(values)
    tables = Path(output_root) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "held_out_shap_values.parquet"
    summary_path = tables / "held_out_shap_importance.parquet"
    summary_csv_path = tables / "held_out_shap_importance.csv"
    values.to_parquet(values_path, index=False)
    summary.to_parquet(summary_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    manifest = {
        "version": SHAP_WORKFLOW_VERSION,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "scopes": list(scopes),
        "split_role": "paper2_test",
        "selection": "frozen development-CV winners",
        "background_split_role": "development",
        "test_rows_per_winner": test_rows,
        "background_rows_per_winner": background_rows,
        "neural_nsamples": nsamples,
        "aggregation": "mean over comments within story, then equal mean over stories",
        "values_layout": "one row per sampled comment x selector; SHAP values are shap_<feature> columns",
        "embedding_aggregation": "BGE SHAP dimensions are summed to text_bge",
        "outputs": {
            "values": {"path": str(values_path), "sha256": _sha256(values_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "summary_csv": {"path": str(summary_csv_path), "sha256": _sha256(summary_csv_path)},
        },
        "cache": cache_records,
    }
    manifest_path = Path(output_root) / "shap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"values": values, "summary": summary, "manifest": manifest}
