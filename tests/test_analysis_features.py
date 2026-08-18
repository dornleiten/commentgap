import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from commentgap_analysis.features import (
    ROOT_MODEL_FEATURES,
    FeatureBuildConfig,
    article_similarity_top3,
    assign_audience_labels,
    compute_author_history,
    compute_discussion_history,
    temporal_novelty,
    validate_approximate_novelty,
    validate_qa_summary,
    vienna_period,
)
from commentgap_analysis.nlp import (
    DEFAULT_SENTIMENT_MODEL_ID,
    DEFAULT_SENTIMENT_MODEL_REVISION,
    DEFAULT_TOXICITY_MODEL_ID,
    DEFAULT_TOXICITY_MODEL_REVISION,
    TextDetoxToxicityEncoder,
    XLMTwitterSentimentEncoder,
    resolve_hf_model_revision,
    select_torch_device,
)


class AnalysisFeatureTests(unittest.TestCase):
    def test_sentiment_special_tokens_support_transformers_5_tokenizer(self):
        encoder = XLMTwitterSentimentEncoder.__new__(XLMTwitterSentimentEncoder)
        encoder._tokenizer = SimpleNamespace(cls_token_id=101, sep_token_id=102)
        self.assertEqual(encoder._add_special_tokens([7, 8]), [101, 7, 8, 102])

        encoder._tokenizer = SimpleNamespace(
            build_inputs_with_special_tokens=lambda values: [11, *values, 12],
            cls_token_id=None,
            sep_token_id=None,
        )
        self.assertEqual(encoder._add_special_tokens([7, 8]), [11, 7, 8, 12])

    def test_new_nlp_defaults_are_pinned_and_toxicity_is_a_model_feature(self):
        config = FeatureBuildConfig(inference_mode=False)
        self.assertEqual(config.sentiment_model_id, DEFAULT_SENTIMENT_MODEL_ID)
        self.assertEqual(config.sentiment_revision, DEFAULT_SENTIMENT_MODEL_REVISION)
        self.assertEqual(config.toxicity_model_id, DEFAULT_TOXICITY_MODEL_ID)
        self.assertEqual(config.toxicity_revision, DEFAULT_TOXICITY_MODEL_REVISION)
        self.assertIn("toxicity_probability", ROOT_MODEL_FEATURES)

    def test_xlmt_sentiment_uses_cardiff_label_order_and_weighted_chunks(self):
        encoder = XLMTwitterSentimentEncoder.__new__(XLMTwitterSentimentEncoder)
        encoder._label_indices = {"positive": 2, "negative": 0, "neutral": 1}
        encoder._predict_chunks = lambda texts, batch_size: (
            np.asarray(
                [
                    [0.6, 0.3, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.2, 0.5, 0.3],
                ]
            ),
            np.asarray([0, 0, 1]),
            np.asarray([1.0, 3.0, 2.0]),
        )
        actual = encoder.predict(["first", "second"], batch_size=2)
        np.testing.assert_allclose(actual[0], [0.55, 0.225, 0.225])
        np.testing.assert_allclose(actual[1], [0.3, 0.2, 0.5])

    def test_textdetox_returns_max_and_weighted_mean_toxicity(self):
        encoder = TextDetoxToxicityEncoder.__new__(TextDetoxToxicityEncoder)
        encoder._toxic_index = 1
        encoder._predict_chunks = lambda texts, batch_size: (
            np.asarray([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]),
            np.asarray([0, 0, 1]),
            np.asarray([1.0, 3.0, 2.0]),
        )
        actual = encoder.predict(["first", "second"], batch_size=2)
        np.testing.assert_allclose(actual[0], [0.8, 0.625])
        np.testing.assert_allclose(actual[1], [0.4, 0.4])

    def test_hugging_face_revisions_are_resolved_to_immutable_commits(self):
        commit = "a" * 40
        with patch(
            "huggingface_hub.HfApi.model_info",
            return_value=SimpleNamespace(sha=commit),
        ) as model_info:
            self.assertEqual(resolve_hf_model_revision("org/model"), commit)
            model_info.assert_called_once_with("org/model", revision=None)
        with patch("huggingface_hub.HfApi.model_info") as model_info:
            self.assertEqual(resolve_hf_model_revision("org/model", commit.upper()), commit)
            model_info.assert_not_called()
        with patch(
            "huggingface_hub.HfApi.model_info",
            return_value=SimpleNamespace(sha="not-a-commit"),
        ):
            with self.assertRaisesRegex(RuntimeError, "no valid immutable commit"):
                resolve_hf_model_revision("org/model", "main")

    def test_explicit_accelerator_request_is_validated(self):
        unavailable = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            ),
        )
        with patch.dict("sys.modules", {"torch": unavailable}):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                select_torch_device("cuda")
            with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
                select_torch_device("mps")
        available = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            ),
        )
        with patch.dict("sys.modules", {"torch": available}):
            self.assertEqual(select_torch_device("cuda"), "cuda")
        self.assertEqual(select_torch_device("cpu"), "cpu")

    def test_strict_discussion_and_branch_history(self):
        frame = pd.DataFrame(
            [
                ("r1", "s", None, "r1", True, "2025-03-01T10:00:00Z", "a"),
                ("c1", "s", "r1", "r1", False, "2025-03-01T10:30:00Z", "b"),
                ("r2", "s", None, "r2", True, "2025-03-01T11:00:00Z", "a"),
                ("c2", "s", "r1", "r1", False, "2025-03-01T11:00:00Z", "a"),
            ],
            columns=[
                "comment_id",
                "story_id",
                "parent_comment_id",
                "root_comment_id",
                "is_root",
                "created_at",
                "author_hash",
            ],
        )
        result = compute_discussion_history(frame).set_index("comment_id")
        self.assertEqual(result.at["c1", "prior_roots"], 1)
        self.assertEqual(result.at["c1", "branch_prior_comments"], 1)
        self.assertEqual(result.at["r2", "prior_comments"], 2)
        self.assertEqual(result.at["c2", "prior_comments"], 2)
        self.assertEqual(result.at["c2", "branch_prior_comments"], 2)
        self.assertEqual(result.at["r2", "branch_prior_comments"], 0)
        self.assertEqual(result.at["r2", "author_prior_comments_story"], 1)
        self.assertEqual(result.at["c2", "author_prior_comments_story"], 1)

    def test_author_history_excludes_focal_story_and_timestamp_ties(self):
        frame = pd.DataFrame(
            [
                ("a1", "A", "2025-01-01T10:00:00Z", "u", 5, 1),
                ("a2", "A", "2025-01-02T10:00:00Z", "u", 7, 2),
                ("b1", "B", "2025-01-03T10:00:00Z", "u", 0, 0),
                ("c1", "C", "2025-01-03T10:00:00Z", "u", 0, 0),
                ("late", "D", "2025-02-15T10:00:00Z", "u", 0, 0),
            ],
            columns=[
                "comment_id",
                "story_id",
                "created_at",
                "author_hash",
                "votes_positive",
                "votes_negative",
            ],
        )
        result = compute_author_history(frame).set_index("comment_id")
        self.assertEqual(result.at["a2", "author_prior_30d_comments"], 0)
        self.assertEqual(result.at["b1", "author_prior_30d_comments"], 2)
        self.assertEqual(result.at["b1", "author_prior_30d_snapshot_upvotes"], 12)
        self.assertEqual(result.at["b1", "author_prior_30d_snapshot_downvotes"], 3)
        self.assertEqual(result.at["c1", "author_prior_30d_comments"], 2)
        self.assertEqual(result.at["late", "author_prior_30d_comments"], 0)

    def test_top_k_ties_are_exact_and_reproducible(self):
        frame = pd.DataFrame(
            {
                "story_id": ["s"] * 5,
                "comment_id": list("abcde"),
                "is_sticky": [True, True, False, False, False],
                "relative_votes": [10, 8, 8, 8, 1],
            }
        )
        first, diagnostics = assign_audience_labels(frame, draws=10, seed=9)
        second, _ = assign_audience_labels(frame.sample(frac=1), draws=10, seed=9)
        for draw in range(1, 11):
            column = f"audience_selected_draw_{draw:02d}"
            self.assertEqual(int(first[column].sum()), 2)
            left = first.set_index("comment_id")[column].sort_index()
            right = second.set_index("comment_id")[column].sort_index()
            pd.testing.assert_series_equal(left, right)
        self.assertTrue(bool(diagnostics.at[0, "ambiguous_cutoff"]))

    def test_similarity_and_strict_temporal_novelty(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        passages = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        similarity = article_similarity_top3(vectors, passages)
        np.testing.assert_allclose(similarity, [0.5, 0.5, 0.5])
        timestamps = pd.to_datetime(
            ["2025-01-01T10:00Z", "2025-01-01T10:00Z", "2025-01-01T11:00Z"], utc=True
        )
        novelty = temporal_novelty(
            vectors, pd.Series(timestamps), np.ones(3, dtype=bool), exact_threshold=10
        )
        self.assertTrue(np.isnan(novelty[0]))
        self.assertTrue(np.isnan(novelty[1]))
        self.assertAlmostEqual(float(novelty[2]), 1.0)
        validation = validate_approximate_novelty(
            vectors,
            pd.Series(timestamps),
            np.ones(3, dtype=bool),
            novelty,
            seed=1,
        )
        self.assertEqual(validation["recall_within_1e_3"], 1.0)

    def test_vienna_periods_and_qa_gate(self):
        self.assertEqual(vienna_period(pd.Timestamp("2025-07-01T01:00:00Z")), "overnight")
        self.assertEqual(vienna_period(pd.Timestamp("2025-07-01T08:00:00Z")), "weekday_work")
        self.assertEqual(vienna_period(pd.Timestamp("2025-07-05T10:00:00Z")), "weekend_day_evening")
        summary = {"passed": True, "nonterminal_stories": 1, "status_counts": {}}
        validate_qa_summary(summary, allow_incomplete=True)
        with self.assertRaises(ValueError):
            validate_qa_summary(summary, allow_incomplete=False)


if __name__ == "__main__":
    unittest.main()
