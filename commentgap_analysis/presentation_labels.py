"""Short labels used only in publication-facing tables and figures."""

from __future__ import annotations

from typing import Mapping


MODEL_PREFIXES: Mapping[tuple[str, str | None], str] = {
    ("conditional_logit", None): "Reg",
    ("xgboost", "metadata"): "XGB",
    ("xgboost", "metadata_bge"): "XGB-T",
    ("neural", "metadata"): "NN",
    ("neural", "metadata_bge"): "NN-T",
}

# Canonical order and labels for publication-facing outcome displays.
OUTCOME_DISPLAY_ORDER: tuple[str, ...] = (
    "article_similarity_top3",
    "semantic_novelty_knn5",
    "log_author_prior_30d_comments",
    "author_prior_30d_reception_balance",
    "toxicity_probability",
    "sentiment_positive",
    "sentiment_negative",
    "cttr",
    "smog_de",
    "aqua_score_expected",
)

OUTCOME_DISPLAY_LABELS: Mapping[str, str] = {
    "article_similarity_top3": "Comment-Article similarity",
    "semantic_novelty_knn5": "Comment novelty",
    "log_author_prior_30d_comments": "Author incumbency",
    "author_prior_30d_reception_balance": "Author’s prior reception",
    "toxicity_probability": "Toxicity",
    "sentiment_positive": "Positive sentiment",
    "sentiment_negative": "Negative sentiment",
    "cttr": "Lexical diversity",
    "smog_de": "Reading difficulty",
    "aqua_score_expected": "AQuA deliberative quality",
}

REPLY_DISPLAY_MARKERS: Mapping[str, str] = {
    "loose": "o",
    "trees": "^",
    "hidden": "X",
}

ORDERING_LABELS: Mapping[str, str] = {
    "regression_editor": "Reg: Editor",
    "regression_audience": "Reg: Audience",
    "xgb_metadata_editor": "XGB: Editor",
    "xgb_metadata_audience": "XGB: Audience",
    "xgb_metadata_text_editor": "XGB-T: Editor",
    "xgb_metadata_text_audience": "XGB-T: Audience",
    "neural_metadata_editor": "NN: Editor",
    "neural_metadata_audience": "NN: Audience",
    "neural_metadata_text_editor": "NN-T: Editor",
    "neural_metadata_text_audience": "NN-T: Audience",
}

# Canonical order for publication-facing displays that include every policy,
# including the explicit random reference ordering.
ORDERING_DISPLAY_ORDER: tuple[str, ...] = (
    "chronological",
    "reverse_chronological",
    "relative_votes",
    "upvotes",
    "random",
    "regression_audience",
    "regression_editor",
    "xgb_metadata_audience",
    "xgb_metadata_editor",
    "xgb_metadata_text_audience",
    "xgb_metadata_text_editor",
    "neural_metadata_audience",
    "neural_metadata_editor",
    "neural_metadata_text_audience",
    "neural_metadata_text_editor",
)

ORDERING_DISPLAY_LABELS: Mapping[str, str] = {
    "chronological": "Chronological",
    "reverse_chronological": "Rev. chron.",
    "relative_votes": "Relative votes",
    "upvotes": "Upvotes",
    "random": "Random",
    "regression_audience": "Reg: Audience",
    "regression_editor": "Reg: Editor",
    "xgb_metadata_audience": "XGB: Audience",
    "xgb_metadata_editor": "XGB: Editor",
    "xgb_metadata_text_audience": "XGB-T: Audience",
    "xgb_metadata_text_editor": "XGB-T: Editor",
    "neural_metadata_audience": "NN: Audience",
    "neural_metadata_editor": "NN: Editor",
    "neural_metadata_text_audience": "NN-T: Audience",
    "neural_metadata_text_editor": "NN-T: Editor",
}

# Canonical colours for plots with an ordering legend. Model variants share a
# hue family: XGB is blue, XGB-T blue-teal, NN red, and NN-T red-plum.
ORDERING_DISPLAY_COLORS: Mapping[str, str] = {
    "chronological": "#084081",
    "reverse_chronological": "#2C7FB8",
    "relative_votes": "#1B7837",
    "upvotes": "#5AAE61",
    "random": "#222222",
    "regression_audience": "#A63603",
    "regression_editor": "#E08214",
    "xgb_metadata_audience": "#1F4E79",
    "xgb_metadata_editor": "#5B9BD5",
    "xgb_metadata_text_audience": "#176B87",
    "xgb_metadata_text_editor": "#6FB7C9",
    "neural_metadata_audience": "#8F1D2C",
    "neural_metadata_editor": "#D66A5E",
    "neural_metadata_text_audience": "#8E3A63",
    "neural_metadata_text_editor": "#D28AB0",
}


def model_prefix(family: str, feature_set: str | None = None) -> str:
    """Return the short display name for a model pair."""
    family = str(family)
    feature_value = None if feature_set is None else str(feature_set)
    if feature_value is not None and feature_value.lower() == "nan":
        feature_value = None
    key = (family, None if family == "conditional_logit" else feature_value)
    return MODEL_PREFIXES.get(key, family)


def model_selector_label(
    family: str, feature_set: str | None, selector: str
) -> str:
    """Return the short display name for one audience/editor model."""
    role = {"audience": "Audience", "curator": "Editor", "editor": "Editor"}.get(
        str(selector), str(selector).title()
    )
    return f"{model_prefix(family, feature_set)}: {role}"


def ordering_label(ordering: str) -> str:
    """Return the canonical short label for an ordering policy."""
    return ORDERING_DISPLAY_LABELS.get(str(ordering), str(ordering))
