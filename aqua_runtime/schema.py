"""Immutable AQuA feature schema shared across the process boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


AQUA_SCHEMA_VERSION = 1
AQUA_UPSTREAM_REPOSITORY = "https://github.com/mabehrendt/AQuA"
AQUA_UPSTREAM_COMMIT = "637914dcd62491766ff478dc21632813780d005d"
AQUA_BASE_MODEL_ID = "bert-base-multilingual-cased"
AQUA_BASE_MODEL_REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
AQUA_SCORE_MIN = -1.66928295
AQUA_SCORE_MAX = 4.98926754
AQUA_SCORE_RANGE = AQUA_SCORE_MAX - AQUA_SCORE_MIN
AQUA_PROBABILITY_STATUS = "uncalibrated"
AQUA_ARTIFACT_FILES = (
    "adapter_config.json",
    "head_config.json",
    "pytorch_adapter.bin",
    "pytorch_model_head.bin",
)


@dataclass(frozen=True)
class AquaFeature:
    order: int
    repository_adapter: str
    stem: str
    description: str
    weight: float


AQUA_FEATURES = (
    AquaFeature(1, "relevance", "relevance", "Relevance to the discussion topic", 0.20908452),
    AquaFeature(2, "fact", "fact", "At least one factual claim", 0.18285757),
    AquaFeature(3, "opinion", "opinion", "Subjective statement or opinion", -0.11069402),
    AquaFeature(4, "justification", "justification", "A statement is justified", 0.29000763),
    AquaFeature(5, "solproposal", "solution_proposal", "A solution is proposed", 0.39535126),
    AquaFeature(6, "addknowledge", "additional_knowledge", "Additional knowledge is contributed", 0.14655912),
    AquaFeature(7, "question", "question", "Genuine, non-rhetorical question", -0.07331445),
    AquaFeature(8, "refusers", "referencing_users", "Reference to another user or community", -0.03768367),
    AquaFeature(9, "refmedium", "referencing_medium", "Reference to medium, editor, or moderator", 0.07019062),
    AquaFeature(10, "refcontents", "referencing_contents", "Reference to another comment's content", -0.02847408),
    AquaFeature(11, "refpersonal", "referencing_personal", "Personal reference to another user", 0.21126469),
    AquaFeature(12, "refformat", "referencing_format", "Reference to tone, spelling, or format", -0.02674237),
    AquaFeature(13, "address", "polite_address", "Greeting, farewell, or polite address", 0.01482095),
    AquaFeature(14, "respect", "respect", "Respect, appreciation, or gratitude", 0.00732909),
    AquaFeature(15, "screaming", "screaming", "Capitalization or punctuation suggesting shouting", -0.01900971),
    AquaFeature(16, "vulgar", "vulgarity", "Language inappropriate for civil discourse", -0.04995486),
    AquaFeature(17, "insult", "insult", "Insult or personal attack", -0.05884586),
    AquaFeature(18, "sarcasm", "sarcasm", "Devaluing, biting mockery", -0.15170863),
    AquaFeature(19, "discrimination", "discrimination", "Discriminatory language", 0.02934227),
    AquaFeature(20, "storytelling", "storytelling", "Personal story or experience", 0.10628146),
)


def feature_by_adapter(adapter: str) -> AquaFeature:
    for feature in AQUA_FEATURES:
        if feature.repository_adapter == adapter:
            return feature
    raise KeyError(adapter)


def label_column(stem: str) -> str:
    return f"aqua_{stem}_label"


def logit_column(stem: str, ordinal_class: int) -> str:
    return f"aqua_{stem}_logit_{ordinal_class}_raw"


def probability_column(stem: str, ordinal_class: int) -> str:
    return f"aqua_{stem}_prob_{ordinal_class}_raw"


def expected_raw_column(stem: str) -> str:
    return f"aqua_{stem}_expected_raw"


def expected_alias_column(stem: str) -> str:
    return f"aqua_{stem}_expected"


def normalize_score(raw_score: np.ndarray | float) -> np.ndarray:
    return 5.0 * (np.asarray(raw_score, dtype=np.float64) - AQUA_SCORE_MIN) / AQUA_SCORE_RANGE


def composite_raw(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != len(AQUA_FEATURES):
        raise ValueError(f"Expected {len(AQUA_FEATURES)} AQuA values, got {array.shape}")
    weights = np.asarray([feature.weight for feature in AQUA_FEATURES], dtype=np.float64)
    return array @ weights


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_feature_columns(features: Iterable[AquaFeature] = AQUA_FEATURES) -> list[str]:
    columns: list[str] = []
    for feature in features:
        columns.append(label_column(feature.stem))
        columns.extend(logit_column(feature.stem, value) for value in range(4))
        columns.extend(probability_column(feature.stem, value) for value in range(4))
        columns.append(expected_raw_column(feature.stem))
    return columns


def downstream_feature_columns() -> list[str]:
    columns: list[str] = []
    for feature in AQUA_FEATURES:
        columns.extend((label_column(feature.stem), expected_alias_column(feature.stem)))
    return columns + [
        "aqua_score_hard",
        "aqua_score_expected",
        "aqua_input_token_count",
        "aqua_input_truncated",
        "aqua_runtime_status",
        "aqua_probability_status",
    ]


_derived_min = 3.0 * sum(min(0.0, feature.weight) for feature in AQUA_FEATURES)
_derived_max = 3.0 * sum(max(0.0, feature.weight) for feature in AQUA_FEATURES)
if not np.isclose(_derived_min, AQUA_SCORE_MIN) or not np.isclose(
    _derived_max, AQUA_SCORE_MAX
):
    raise RuntimeError("Published AQuA extrema do not match the canonical weights")
