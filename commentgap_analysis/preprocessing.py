"""Shared split and predictor preprocessing for Paper 2 selection models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ranking import article_split_balance, assign_article_splits


PREPROCESSING_VERSION = 4
REPLY_COMPOSITION_SMOOTHING = 0.5
AUTHOR_RECEPTION_SMOOTHING = 0.5
ACTIVITY_SMOOTHING = 0.5
DEFAULT_SEED = 20260813


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        temporary,
        compression="zstd",
    )
    os.replace(temporary, path)


def validate_frozen_split(split: pd.DataFrame, *, development_folds: int = 5) -> pd.DataFrame:
    """Validate and return a canonical copy of the already-frozen article split."""
    required = {"story_id", "split_role", "development_fold", "split_stratum"}
    missing = required - set(split.columns)
    if missing:
        raise ValueError(f"Frozen split is missing columns: {sorted(missing)}")
    output = split.copy()
    output["story_id"] = output["story_id"].astype(str)
    if output["story_id"].duplicated().any():
        raise ValueError("Frozen split contains duplicate story IDs")
    if set(output["split_role"].unique()) != {"development", "paper2_test"}:
        raise ValueError("Frozen split must contain development and paper2_test roles")
    development = output[output["split_role"] == "development"]
    test = output[output["split_role"] == "paper2_test"]
    if set(development["development_fold"].astype(int)) != set(range(development_folds)):
        raise ValueError("Frozen split does not contain the expected development folds")
    if not (test["development_fold"].astype(int) == -1).all():
        raise ValueError("Paper 2 test articles must have development_fold = -1")
    return output.sort_values("story_id").reset_index(drop=True)


def add_reply_composition(
    frame: pd.DataFrame,
    *,
    smoothing: float = REPLY_COMPOSITION_SMOOTHING,
) -> pd.DataFrame:
    """Add prior replies and the smoothed reply-to-root log ratio."""
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Reply-composition smoothing must be positive")
    required = {"prior_comments", "prior_roots"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prior-activity columns: {sorted(missing)}")
    output = frame.copy()
    comments = pd.to_numeric(output["prior_comments"], errors="raise").to_numpy()
    roots = pd.to_numeric(output["prior_roots"], errors="raise").to_numpy()
    if not (np.isfinite(comments).all() and np.isfinite(roots).all()):
        raise ValueError("Prior-activity counts must be finite")
    if (comments < 0).any() or (roots < 0).any() or (roots > comments).any():
        raise ValueError("Prior activity must satisfy prior_comments >= prior_roots >= 0")
    replies = comments - roots
    output["prior_replies"] = replies.astype(np.int64)
    output["prior_reply_composition"] = np.log(
        (replies.astype(float) + smoothing) / (roots.astype(float) + smoothing)
    )
    return output


def add_author_reception(
    frame: pd.DataFrame,
    *,
    smoothing: float = AUTHOR_RECEPTION_SMOOTHING,
) -> pd.DataFrame:
    """Add smoothed log vote totals per prior 30-day author comment."""
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Author-reception smoothing must be positive")
    required = {
        "author_prior_30d_comments",
        "author_prior_30d_snapshot_upvotes",
        "author_prior_30d_snapshot_downvotes",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing author-history columns: {sorted(missing)}")
    output = frame.copy()
    comments = pd.to_numeric(
        output["author_prior_30d_comments"], errors="raise"
    ).to_numpy(dtype=float)
    upvotes = pd.to_numeric(
        output["author_prior_30d_snapshot_upvotes"], errors="raise"
    ).to_numpy(dtype=float)
    downvotes = pd.to_numeric(
        output["author_prior_30d_snapshot_downvotes"], errors="raise"
    ).to_numpy(dtype=float)
    if not (
        np.isfinite(comments).all()
        and np.isfinite(upvotes).all()
        and np.isfinite(downvotes).all()
    ):
        raise ValueError("Author-history counts must be finite")
    if (comments < 0).any() or (upvotes < 0).any() or (downvotes < 0).any():
        raise ValueError("Author-history counts must be non-negative")
    denominator = comments + smoothing
    output["author_prior_30d_upvote_reception"] = np.log(
        (upvotes + smoothing) / denominator
    )
    output["author_prior_30d_downvote_reception"] = np.log(
        (downvotes + smoothing) / denominator
    )
    return output


def add_activity_rates(
    frame: pd.DataFrame,
    *,
    smoothing: float = ACTIVITY_SMOOTHING,
) -> pd.DataFrame:
    """Add smoothed prior-comment pace relative to article age."""
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Activity smoothing must be positive")
    required = {"hours_since_article", "prior_comments"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing activity columns: {sorted(missing)}")
    output = frame.copy()
    hours = pd.to_numeric(output["hours_since_article"], errors="raise").to_numpy(
        dtype=float
    )
    comments = pd.to_numeric(output["prior_comments"], errors="raise").to_numpy(
        dtype=float
    )
    if not (
        np.isfinite(hours).all()
        and np.isfinite(comments).all()
    ):
        raise ValueError("Activity values must be finite")
    if (hours < 0).any() or (comments < 0).any():
        raise ValueError("Activity values must be non-negative")
    output["discussion_pace"] = np.log(
        (comments + smoothing) / (hours + smoothing)
    )
    return output


def reply_depth_centers(
    all_choice_set: pd.DataFrame,
    article_split: pd.DataFrame,
    *,
    development_folds: int = 5,
) -> dict[str, float]:
    """Fit full-development and fold-training means among reply candidates."""
    required = {"story_id", "is_reply", "log_depth"}
    missing = required - set(all_choice_set.columns)
    if missing:
        raise ValueError(f"All-comment choice set is missing: {sorted(missing)}")
    split = validate_frozen_split(article_split, development_folds=development_folds)
    values = all_choice_set[["story_id", "is_reply", "log_depth"]].copy()
    values["story_id"] = values["story_id"].astype(str)
    values = values.merge(
        split[["story_id", "split_role", "development_fold"]],
        on="story_id",
        how="inner",
        validate="many_to_one",
    )
    development = values[values["split_role"] == "development"]
    if development.empty or not set(development["development_fold"].astype(int)) == set(
        range(development_folds)
    ):
        raise ValueError("Cannot fit reply-depth centres without all development folds")

    def fit_mean(rows: pd.DataFrame, label: str) -> float:
        replies = pd.to_numeric(
            rows.loc[rows["is_reply"].astype(int) == 1, "log_depth"], errors="raise"
        ).to_numpy(dtype=float)
        if not len(replies) or not np.isfinite(replies).all():
            raise ValueError(f"No finite reply depths for {label}")
        return float(replies.mean())

    centers = {"full_development": fit_mean(development, "full development")}
    for fold in range(development_folds):
        centers[f"fold_{fold:02d}_training"] = fit_mean(
            development[development["development_fold"].astype(int) != fold],
            f"fold {fold} training",
        )
    return centers


def add_centered_reply_depths(
    frame: pd.DataFrame,
    centers: dict[str, float],
) -> pd.DataFrame:
    """Apply full-development and fold-training reply-depth centres."""
    required = {"is_reply", "log_depth"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing reply-depth columns: {sorted(missing)}")
    if "full_development" not in centers:
        raise ValueError("Missing full-development reply-depth centre")
    output = frame.copy()
    is_reply = output["is_reply"].astype(int).to_numpy(dtype=float)
    log_depth = pd.to_numeric(output["log_depth"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(log_depth).all():
        raise ValueError("Reply depths must be finite")
    output["reply_depth_centered"] = is_reply * (
        log_depth - float(centers["full_development"])
    )
    for name, center in sorted(centers.items()):
        if not name.startswith("fold_"):
            continue
        fold = int(name.split("_")[1])
        output[f"reply_depth_centered_fold_{fold:02d}"] = is_reply * (
            log_depth - float(center)
        )
    return output


def transformed_feature_manifest(source_manifest: dict[str, Any]) -> dict[str, Any]:
    """Create the shared model contract while retaining the source registry."""
    manifest = json.loads(json.dumps(source_manifest))
    manifest["shared_preprocessing_version"] = PREPROCESSING_VERSION
    features = manifest.setdefault("features", {})
    features["prior_reply_composition"] = {
        "label": "Earlier reply-to-root composition",
        "standardize": True,
        "deferred": False,
        "transformation": "log((prior_comments - prior_roots + 0.5) / (prior_roots + 0.5))",
    }
    features["reply_depth_centered"] = {
        "label": "Reply depth centred at the development-reply mean",
        "standardize": True,
        "deferred": False,
        "transformation": "is_reply * (log_depth - development_reply_mean_log_depth)",
    }
    features["author_prior_30d_upvote_reception"] = {
        "label": "Upvotes per prior author comment",
        "standardize": True,
        "deferred": False,
        "transformation": "log((author_prior_30d_snapshot_upvotes + 0.5) / (author_prior_30d_comments + 0.5))",
    }
    features["author_prior_30d_downvote_reception"] = {
        "label": "Downvotes per prior author comment",
        "standardize": True,
        "deferred": False,
        "transformation": "log((author_prior_30d_snapshot_downvotes + 0.5) / (author_prior_30d_comments + 0.5))",
    }
    features["discussion_pace"] = {
        "label": "Earlier comments per article-hour",
        "standardize": True,
        "deferred": False,
        "transformation": "log((prior_comments + 0.5) / (hours_since_article + 0.5))",
    }
    for scope in ("root", "all"):
        model_features = list(manifest["models"][scope]["features"])
        model_features = [
            "prior_reply_composition" if feature == "log_prior_roots" else feature
            for feature in model_features
        ]
        author_replacements = {
            "log_author_prior_30d_snapshot_upvotes":
                "author_prior_30d_upvote_reception",
            "log_author_prior_30d_snapshot_downvotes":
                "author_prior_30d_downvote_reception",
        }
        model_features = [
            author_replacements.get(feature, feature) for feature in model_features
        ]
        activity_replacements = {
            "log_prior_comments": "discussion_pace",
        }
        model_features = [
            activity_replacements.get(feature, feature) for feature in model_features
        ]
        model_features = [
            feature for feature in model_features
            if feature != "log_comments_prev_hour"
        ]
        if scope == "all":
            model_features = [
                "reply_depth_centered" if feature == "log_depth" else feature
                for feature in model_features
            ]
            model_features = [
                feature for feature in model_features
                if feature not in {
                    "log_branch_prior_comments", "log_branch_comments_prev_hour"
                }
            ]
        if len(model_features) != len(set(model_features)):
            raise ValueError(f"Duplicate transformed model features for {scope}")
        manifest["models"][scope]["features"] = model_features
    manifest["replaced_model_features"] = {
        "log_prior_roots": "prior_reply_composition",
        "log_depth": "reply_depth_centered",
        "log_author_prior_30d_snapshot_upvotes":
            "author_prior_30d_upvote_reception",
        "log_author_prior_30d_snapshot_downvotes":
            "author_prior_30d_downvote_reception",
        "log_prior_comments": "discussion_pace",
        "log_comments_prev_hour": None,
        "log_branch_prior_comments": None,
        "log_branch_comments_prev_hour": None,
    }
    return manifest


def prepare_shared_model_data(
    feature_root: Path,
    output_root: Path,
    *,
    frozen_split_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    development_folds: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize the frozen split and shared transformed choice sets."""
    import pyarrow.parquet as pq

    feature_root = Path(feature_root)
    output_root = Path(output_root)
    source_paths = {
        scope: feature_root / f"choice_set_{scope}.parquet" for scope in ("root", "all")
    }
    source_manifest_path = feature_root / "feature_manifest.json"
    source_provenance_path = feature_root / "provenance_manifest.json"
    missing_paths = [
        path
        for path in [*source_paths.values(), source_manifest_path, source_provenance_path]
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing feature artifacts: {missing_paths}")

    cache_manifest_path = output_root / "preprocessing_manifest.json"
    expected_outputs = [
        output_root / "master_article_split.parquet",
        output_root / "choice_set_root.parquet",
        output_root / "choice_set_all.parquet",
        output_root / "feature_manifest.json",
        output_root / "provenance_manifest.json",
        output_root / "preprocessing_parameters.json",
        output_root / "split_balance_diagnostics.csv",
    ]
    if cache_manifest_path.exists() and not force and all(
        path.exists() for path in expected_outputs
    ):
        cached = json.loads(cache_manifest_path.read_text())
        source_files = [
            *source_paths.values(), source_manifest_path, source_provenance_path
        ]
        current_sources = {path.name: _file_sha256(path) for path in source_files}
        recorded_sources = {
            name: record["sha256"]
            for name, record in cached.get("source_artifacts", {}).items()
        }
        frozen_matches = frozen_split_path is None or not Path(frozen_split_path).exists() or (
            cached.get("split_source_sha256") == _file_sha256(Path(frozen_split_path))
        )
        if (
            cached.get("version") == PREPROCESSING_VERSION
            and current_sources == recorded_sources
            and frozen_matches
):
            parameters = json.loads(
                (output_root / "preprocessing_parameters.json").read_text()
)
            return {
                "article_split": pq.read_table(
                    output_root / "master_article_split.parquet"
).to_pandas(),
                "split_balance": pd.read_csv(
                    output_root / "split_balance_diagnostics.csv"
) ,
                "reply_depth_centers": parameters["reply_depth_centers"],
                "feature_manifest": json.loads(
                    (output_root / "feature_manifest.json").read_text()
) ,
                "output_root": str(output_root),
                "rows": {
                    scope: pq.ParquetFile(
                        output_root / f"choice_set_{scope}.parquet"
).metadata.num_rows
                    for scope in ("root", "all")
                },
                "cache_hit": True,
            }
        raise RuntimeError(
            "Shared preprocessing inputs changed; set force=True to rebuild"
)

    source_manifest = json.loads(source_manifest_path.read_text())
    root = pq.read_table(source_paths["root"]).to_pandas()
    all_comments = pq.read_table(source_paths["all"]).to_pandas()
    for frame in (root, all_comments):
        frame["story_id"] = frame["story_id"].astype(str)

    if frozen_split_path is not None and Path(frozen_split_path).exists():
        frozen_split_path = Path(frozen_split_path)
        article_split = validate_frozen_split(
            pq.read_table(frozen_split_path).to_pandas(),
            development_folds=development_folds,
        )
        regenerated, _ = assign_article_splits(
            root,
            all_comments,
            seed=seed,
            test_fraction=0.50,
            size_bins=3,
            development_folds=development_folds,
            max_abs_smd=0.05,
        )
        columns = ["story_id", "split_role", "development_fold", "split_stratum"]
        regenerated = regenerated[columns].sort_values("story_id").reset_index(drop=True)
        existing = article_split[columns].sort_values("story_id").reset_index(drop=True)
        if not regenerated.equals(existing):
            raise ValueError(
                "Regenerated split differs from the frozen Paper 2 split; refusing to replace it"
            )
        split_source = str(frozen_split_path)
        split_source_sha256 = _file_sha256(frozen_split_path)
    else:
        article_split, _ = assign_article_splits(
            root,
            all_comments,
            seed=seed,
            test_fraction=0.50,
            size_bins=3,
            development_folds=development_folds,
            max_abs_smd=0.05,
        )
        article_split = validate_frozen_split(
            article_split, development_folds=development_folds
        )
        split_source = "created_by_shared_preprocessing"
        split_source_sha256 = None

    split_ids = set(article_split["story_id"])
    for scope, frame in (("root", root), ("all", all_comments)):
        missing = split_ids - set(frame["story_id"])
        if missing:
            raise ValueError(f"{scope} choice set is missing {len(missing)} frozen-split stories")

    centers = reply_depth_centers(
        all_comments, article_split, development_folds=development_folds
    )
    prepared = {
        "root": add_activity_rates(
            add_author_reception(add_reply_composition(root))
        ),
        "all": add_centered_reply_depths(
            add_activity_rates(
                add_author_reception(add_reply_composition(all_comments))
            ),
            centers,
        ),
    }
    for scope, frame in prepared.items():
        prepared[scope] = frame[frame["story_id"].isin(split_ids)].copy()

    transformed_manifest = transformed_feature_manifest(source_manifest)
    for scope, frame in prepared.items():
        missing = set(transformed_manifest["models"][scope]["features"]) - set(frame.columns)
        if missing:
            raise ValueError(f"Prepared {scope} choice set is missing: {sorted(missing)}")

    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        scope: output_root / f"choice_set_{scope}.parquet" for scope in prepared
    }
    if not force:
        existing = [path for path in [*output_paths.values(), output_root / "master_article_split.parquet"] if path.exists()]
        if existing:
            raise FileExistsError(
                "Shared model-data outputs already exist; set force=True to replace: "
                + ", ".join(map(str, existing))
            )
    _atomic_parquet(article_split, output_root / "master_article_split.parquet")
    for scope, frame in prepared.items():
        _atomic_parquet(frame, output_paths[scope])
    balance = article_split_balance(article_split)
    balance["acceptance_threshold"] = 0.05
    balance["accepted"] = balance["abs_standardized_mean_difference"] <= 0.05
    if not bool(balance["accepted"].all()):
        raise ValueError("Frozen split no longer satisfies its balance contract")
    balance.to_csv(output_root / "split_balance_diagnostics.csv", index=False)
    _atomic_json(transformed_manifest, output_root / "feature_manifest.json")
    _atomic_json(
        json.loads(source_provenance_path.read_text()),
        output_root / "provenance_manifest.json",
    )
    parameters = {
        "version": PREPROCESSING_VERSION,
        "reply_composition_smoothing": REPLY_COMPOSITION_SMOOTHING,
        "author_reception_smoothing": AUTHOR_RECEPTION_SMOOTHING,
        "reply_depth_centers": centers,
        "activity_smoothing": ACTIVITY_SMOOTHING,
        "fold_feature_columns": {
            str(fold): {
                "reply_depth_centered": f"reply_depth_centered_fold_{fold:02d}"
            }
            for fold in range(development_folds)
        },
    }
    _atomic_json(parameters, output_root / "preprocessing_parameters.json")
    manifest = {
        "version": PREPROCESSING_VERSION,
        "seed": seed,
        "development_folds": development_folds,
        "split_source": split_source,
        "split_source_sha256": split_source_sha256,
        "source_artifacts": {
            path.name: {"path": str(path), "sha256": _file_sha256(path)}
            for path in [
                *source_paths.values(), source_manifest_path, source_provenance_path
            ]
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": _file_sha256(path)}
            for path in [
                output_root / "master_article_split.parquet",
                *output_paths.values(),
                output_root / "feature_manifest.json",
                output_root / "provenance_manifest.json",
                output_root / "preprocessing_parameters.json",
            ]
        },
        "feature_replacements": transformed_manifest["replaced_model_features"],
        "known_upstream_pre_split_learned_features": [
            "lexdiv_length_adjusted",
            "reading_level_length_adjusted",
            "novelty_prior_roots_model / novelty_prior_all_model imputation",
        ],
        "paper2_split_status": "frozen and unchanged",
    }
    _atomic_json(manifest, output_root / "preprocessing_manifest.json")
    return {
        "article_split": article_split,
        "split_balance": balance,
        "reply_depth_centers": centers,
        "feature_manifest": transformed_manifest,
        "output_root": str(output_root),
        "rows": {scope: len(frame) for scope, frame in prepared.items()},
    }
