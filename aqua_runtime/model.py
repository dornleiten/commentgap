"""Legacy adapter-transformers inference implementation.

Import this module only from the isolated Python 3.10 environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .schema import (
    AQUA_ARTIFACT_FILES,
    AQUA_BASE_MODEL_ID,
    AQUA_BASE_MODEL_REVISION,
    AQUA_FEATURES,
    AQUA_PROBABILITY_STATUS,
    AquaFeature,
    composite_raw,
    expected_raw_column,
    label_column,
    logit_column,
    normalize_score,
    probability_column,
    sha256_file,
)


def plan_length_aware_batches(
    effective_lengths: np.ndarray,
    *,
    max_batch_size: int,
    max_batch_tokens: int,
    adaptive: bool,
) -> list[np.ndarray]:
    """Plan stable keyed batches, optionally sorting by capped token length."""
    lengths = np.asarray(effective_lengths, dtype=np.int64)
    if lengths.ndim != 1:
        raise ValueError("AQuA token lengths must be one-dimensional")
    if max_batch_size < 1 or max_batch_tokens < 1:
        raise ValueError("AQuA batch limits must be positive")
    if not len(lengths):
        return []
    if (lengths < 1).any():
        raise ValueError("AQuA token lengths must be positive")
    if not adaptive:
        indices = np.arange(len(lengths), dtype=np.int64)
        return [
            indices[start : start + max_batch_size]
            for start in range(0, len(indices), max_batch_size)
        ]

    order = np.argsort(lengths, kind="stable")
    batches: list[np.ndarray] = []
    current: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        proposed_size = len(current) + 1
        proposed_tokens = proposed_size * int(lengths[index])
        if current and (
            proposed_size > max_batch_size or proposed_tokens > max_batch_tokens
        ):
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
        current.append(index)
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


def load_artifact_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("base_model", {}).get("revision") != AQUA_BASE_MODEL_REVISION:
        raise RuntimeError("Artifact manifest has the wrong multilingual BERT revision")
    return manifest


def validate_adapter_artifacts(
    adapter_root: Path,
    artifact_manifest: Path,
    features: Sequence[AquaFeature] = AQUA_FEATURES,
) -> dict[str, dict[str, str]]:
    """Fail closed on missing, changed, or non-ordinal released adapters."""
    manifest = load_artifact_manifest(artifact_manifest)
    expected_files = manifest.get("adapter_files", {})
    validated: dict[str, dict[str, str]] = {}
    for feature in features:
        adapter_dir = Path(adapter_root) / feature.repository_adapter
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Missing AQuA adapter directory: {adapter_dir}")
        expected = expected_files.get(feature.repository_adapter)
        if set(expected or {}) != set(AQUA_ARTIFACT_FILES):
            raise RuntimeError(
                f"Incomplete checksum manifest for adapter {feature.repository_adapter}"
            )
        actual: dict[str, str] = {}
        for filename in AQUA_ARTIFACT_FILES:
            path = adapter_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing AQuA artifact: {path}")
            actual[filename] = sha256_file(path)
            if actual[filename] != expected[filename]:
                raise RuntimeError(f"Checksum mismatch for AQuA artifact: {path}")
        head = json.loads((adapter_dir / "head_config.json").read_text()).get("config", {})
        if head.get("num_labels") != 4:
            raise RuntimeError(f"Adapter {feature.repository_adapter} is not four-class")
        label2id = head.get("label2id")
        expected_labels = {f"LABEL_{value}": value for value in range(4)}
        if label2id != expected_labels:
            raise RuntimeError(
                f"Adapter {feature.repository_adapter} has unexpected ordinal labels: "
                f"{label2id!r}"
            )
        validated[feature.repository_adapter] = actual
    return validated


class AquaModel:
    """Released multilingual BERT adapters with parallel/sequential execution."""

    def __init__(
        self,
        *,
        adapter_root: Path,
        artifact_manifest: Path,
        device: str,
        execution_mode: str,
        max_length: int = 512,
        features: Sequence[AquaFeature] = AQUA_FEATURES,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("AQuA device must be explicitly 'cpu' or 'cuda'")
        if execution_mode not in {"parallel", "sequential"}:
            raise ValueError("execution_mode must be 'parallel' or 'sequential'")
        self.features = tuple(features)
        self.device = device
        self.execution_mode = execution_mode
        self.max_length = max_length
        self.artifact_hashes = validate_adapter_artifacts(
            adapter_root, artifact_manifest, self.features
        )

        import torch
        from transformers import AutoAdapterModel, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for AQuA, but PyTorch cannot see CUDA")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            AQUA_BASE_MODEL_ID, revision=AQUA_BASE_MODEL_REVISION
        )
        self._model = AutoAdapterModel.from_pretrained(
            AQUA_BASE_MODEL_ID, revision=AQUA_BASE_MODEL_REVISION
        )
        self._aliases: list[str] = []
        for index, feature in enumerate(self.features):
            alias = f"aqua_adapter_{index:02d}"
            self._model.load_adapter(
                str(Path(adapter_root) / feature.repository_adapter),
                load_as=alias,
                with_head=True,
                set_active=False,
                source="hf",
            )
            self._aliases.append(alias)
        if execution_mode == "parallel":
            from transformers.adapters.composition import Parallel

            self._model.active_adapters = Parallel(*self._aliases)
        self._model.to(device)
        self._model.eval()

    def _batch_logits(self, encoded: dict) -> list[np.ndarray]:
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if self.execution_mode == "parallel":
            with self._torch.inference_mode():
                outputs = self._model(**encoded)
            # AdapterHub's Parallel output is iterable over heads even though the
            # container also exposes a ``logits`` attribute. Match the released
            # AQuA script and always iterate it.
            output_list = list(outputs)
            if len(output_list) != len(self.features):
                raise RuntimeError(
                    f"Parallel AQuA returned {len(output_list)} heads for "
                    f"{len(self.features)} adapters"
                )
            return [output.logits.detach().cpu().numpy() for output in output_list]

        logits: list[np.ndarray] = []
        for alias in self._aliases:
            self._model.set_active_adapters(alias)
            with self._torch.inference_mode():
                output = self._model(**encoded)
            logits.append(output.logits.detach().cpu().numpy())
        return logits

    def _token_lengths(self, texts: list[str], chunk_size: int = 1024) -> np.ndarray:
        lengths: list[int] = []
        for start in range(0, len(texts), chunk_size):
            encoded = self._tokenizer(
                texts[start : start + chunk_size],
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            lengths.extend(len(values) for values in encoded["input_ids"])
        return np.asarray(lengths, dtype=np.int32)

    def predict(
        self,
        texts: list[str],
        batch_size: int,
        *,
        adaptive_batches: bool = False,
        max_batch_tokens: int = 2048,
        return_diagnostics: bool = False,
    ):
        token_counts = self._token_lengths(texts)
        truncated = token_counts > self.max_length
        effective_lengths = np.minimum(token_counts, self.max_length)
        batches = plan_length_aware_batches(
            effective_lengths,
            max_batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            adaptive=adaptive_batches,
        )
        planned_batches = [batch.copy() for batch in batches]
        all_logits = {
            feature.stem: np.empty((len(texts), 4), dtype=np.float32)
            for feature in self.features
        }
        completed_batches: list[np.ndarray] = []
        oom_backoffs = 0
        position = 0
        while position < len(batches):
            batch_indices = batches[position]
            batch_texts = [texts[int(index)] for index in batch_indices]
            encoded = self._tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            try:
                batch_logits = self._batch_logits(encoded)
            except RuntimeError as error:
                is_cuda_oom = self.device == "cuda" and "out of memory" in str(error).lower()
                if not is_cuda_oom or len(batch_indices) == 1:
                    raise
                del encoded
                self._torch.cuda.empty_cache()
                midpoint = len(batch_indices) // 2
                batches[position : position + 1] = [
                    batch_indices[:midpoint],
                    batch_indices[midpoint:],
                ]
                oom_backoffs += 1
                print(
                    "AQuA adaptive batching: CUDA OOM at "
                    f"rows={len(batch_indices)}; retrying as "
                    f"{midpoint}+{len(batch_indices) - midpoint}",
                    flush=True,
                )
                continue
            for feature, values in zip(self.features, batch_logits):
                if values.ndim != 2 or values.shape[1] != 4:
                    raise RuntimeError(
                        f"Adapter {feature.repository_adapter} returned shape {values.shape}"
                    )
                all_logits[feature.stem][batch_indices] = values
            completed_batches.append(batch_indices)
            position += 1

        def padded_tokens(items: list[np.ndarray]) -> int:
            return int(
                sum(
                    len(batch) * int(effective_lengths[batch].max())
                    for batch in items
                    if len(batch)
                )
            )

        executed_sizes = [len(batch) for batch in completed_batches]
        diagnostics = {
            "strategy": (
                "length_bucketed_token_budget" if adaptive_batches else "fixed_rows"
            ),
            "max_batch_size": batch_size,
            "max_batch_tokens": max_batch_tokens if adaptive_batches else None,
            "planned_batches": len(planned_batches),
            "executed_batches": len(completed_batches),
            "oom_backoffs": oom_backoffs,
            "minimum_executed_batch_size": min(executed_sizes, default=0),
            "maximum_executed_batch_size": max(executed_sizes, default=0),
            "mean_executed_batch_size": (
                float(np.mean(executed_sizes)) if executed_sizes else 0.0
            ),
            "effective_unpadded_tokens": int(effective_lengths.sum()),
            "planned_padded_tokens": padded_tokens(planned_batches),
            "executed_padded_tokens": padded_tokens(completed_batches),
        }
        result = (all_logits, token_counts, truncated)
        return (*result, diagnostics) if return_diagnostics else result


def predictions_to_frame(
    keys: pd.DataFrame,
    logits_by_stem: dict[str, np.ndarray],
    token_counts: np.ndarray,
    truncated: np.ndarray,
    *,
    build_signature: str,
    watermark: str,
    features: Sequence[AquaFeature] = AQUA_FEATURES,
) -> pd.DataFrame:
    columns: dict[str, object] = {}
    hard_values: list[np.ndarray] = []
    expected_values: list[np.ndarray] = []
    for feature in features:
        logits = np.asarray(logits_by_stem[feature.stem], dtype=np.float64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"Non-finite probabilities for {feature.repository_adapter}")
        labels = probabilities.argmax(axis=1).astype(np.uint8)
        expected = probabilities @ np.arange(4, dtype=np.float64)
        columns[label_column(feature.stem)] = labels
        for ordinal_class in range(4):
            columns[logit_column(feature.stem, ordinal_class)] = logits[:, ordinal_class]
            columns[probability_column(feature.stem, ordinal_class)] = probabilities[:, ordinal_class]
        columns[expected_raw_column(feature.stem)] = expected
        hard_values.append(labels)
        expected_values.append(expected)

    if tuple(features) == AQUA_FEATURES:
        hard_raw = composite_raw(np.column_stack(hard_values))
        expected_raw = composite_raw(np.column_stack(expected_values))
        columns["aqua_score_hard_raw"] = hard_raw
        columns["aqua_score_hard"] = normalize_score(hard_raw)
        columns["aqua_score_expected_raw_unscaled"] = expected_raw
        columns["aqua_score_expected_raw"] = normalize_score(expected_raw)
    columns["aqua_input_token_count"] = token_counts
    columns["aqua_input_truncated"] = truncated
    columns["aqua_runtime_status"] = "ok"
    columns["aqua_probability_status"] = AQUA_PROBABILITY_STATUS
    columns["aqua_build_signature"] = build_signature
    columns["aqua_watermark"] = watermark
    return pd.concat([keys.reset_index(drop=True), pd.DataFrame(columns)], axis=1)
