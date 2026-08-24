import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from commentgap_analysis.features import (
    AQUA_EXPECTED_MODEL_FEATURES,
    ROOT_MODEL_FEATURES,
    FeatureBuildConfig,
    _feature_registry,
    primary_model_features,
    _adapter_implementation_signature,
    _build_choice_sets_bounded,
    _identity_signature,
    _invalid_sticky_story_sets_from_frame,
    _length_residual,
    _local_text_identity,
    _make_choice_set,
    article_similarity_top3,
    assign_audience_labels,
    compute_author_history,
    compute_discussion_history,
    compute_local_text_features,
    temporal_novelty,
    validate_approximate_novelty,
    validate_local_text_features,
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
    def test_bounded_choice_assembly_matches_in_memory_implementation(self):
        rows: list[dict[str, object]] = []
        raw_rows: list[dict[str, object]] = []
        for story_number, story_id in enumerate(("s1", "s2"), start=1):
            for position in range(12):
                comment_id = f"{story_id}-c{position:02d}"
                is_root = position < 6
                row: dict[str, object] = {
                    "story_id": story_id,
                    "comment_id": comment_id,
                    "article_year": 2025,
                    "article_month": 2,
                    "is_root": is_root,
                    "is_sticky": position == 0,
                    "published_at_source": "page",
                    "invalid_posting_time": False,
                    "votes_positive": 20 - position,
                    "votes_negative": position % 2,
                    "vienna_period": "weekday_work",
                    "word_count": 10 + position,
                    "cttr": 0.3 + (story_number * 0.01) + position * 0.005,
                    "smog_de": 4.0 + story_number * 0.1 + position * 0.03,
                    "sentiment_positive": 0.4,
                    "sentiment_negative": 0.2,
                    "sentiment_neutral": 0.4,
                    "toxicity_probability": 0.1,
                    "toxicity_mean_probability": 0.05,
                    "log_words": np.log1p(10 + position),
                    "url_present": 0,
                    "article_similarity_top3": 0.5,
                    "novelty_prior_roots": np.nan if position == 0 else 0.2 + position * 0.01,
                    "novelty_prior_all": np.nan if position == 0 else 0.3 + position * 0.01,
                }
                for column in (
                    "hours_since_article",
                    "prior_roots",
                    "prior_comments",
                    "comments_prev_hour",
                    "author_prior_30d_comments",
                    "author_prior_30d_snapshot_upvotes",
                    "author_prior_30d_snapshot_downvotes",
                    "author_prior_comments_story",
                    "depth",
                    "branch_prior_comments",
                    "branch_comments_prev_hour",
                ):
                    row[column] = position
                rows.append(row)
                raw_rows.append(
                    {
                        "story_id": story_id,
                        "comment_id": comment_id,
                        "is_root": is_root,
                        "is_sticky": position == 0,
                        "lifecycle_status": "Published",
                        "effective_text": f"Text {comment_id}",
                        "created_at": pd.Timestamp("2025-02-01T10:00:00Z")
                        + pd.Timedelta(minutes=position),
                    }
                )
        scalar = pd.DataFrame(rows)
        raw = pd.DataFrame(raw_rows)
        expected = scalar.copy()
        expected["lexdiv_length_adjusted"], _ = _length_residual(expected, "cttr")
        expected["reading_level_length_adjusted"], _ = _length_residual(
            expected, "smog_de"
        )
        config = FeatureBuildConfig(
            inference_mode=False,
            tie_draws=2,
            exclude_january_without_lookback=False,
        )
        expected_root, _, expected_root_summary = _make_choice_set(
            expected, raw, scope="root", config=config
        )
        expected_all, _, expected_all_summary = _make_choice_set(
            expected, raw, scope="all", config=config
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scalar_root = root / "scalar" / "year=2025"
            comments_root = root / "data" / "comments" / "year=2025" / "month=02"
            comments_root.mkdir(parents=True)
            raw.to_parquet(comments_root / "comments.parquet", index=False)
            for story_id, story in scalar.groupby("story_id", sort=True):
                destination = scalar_root / "month=02" / f"{story_id}.parquet"
                destination.parent.mkdir(parents=True, exist_ok=True)
                story.to_parquet(destination, index=False)
            actual_root_summary, actual_all_summary, metadata = (
                _build_choice_sets_bounded(
                    checkpoint_root=scalar_root,
                    data_root=root / "data",
                    output_root=root / "output",
                    config=config,
                    invalid_stories=_invalid_sticky_story_sets_from_frame(raw),
                )
            )
            actual_root = pd.read_parquet(root / "output" / "choice_set_root.parquet")
            actual_all = pd.read_parquet(root / "output" / "choice_set_all.parquet")

        pd.testing.assert_frame_equal(
            actual_root.reset_index(drop=True),
            expected_root.reset_index(drop=True),
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
        pd.testing.assert_frame_equal(
            actual_all.reset_index(drop=True),
            expected_all.reset_index(drop=True),
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertEqual(actual_root_summary, expected_root_summary)
        self.assertEqual(actual_all_summary, expected_all_summary)
        self.assertEqual([item["outcome"] for item in metadata], ["cttr", "smog_de"])

    def test_feature_family_cache_identities_are_isolated(self):
        class SentimentAdapter:
            pass

        class ToxicityAdapterV1:
            pass

        class ToxicityAdapterV2:
            changed = True

        sentiment = _adapter_implementation_signature(SentimentAdapter())
        toxicity_v1 = _adapter_implementation_signature(ToxicityAdapterV1())
        toxicity_v2 = _adapter_implementation_signature(ToxicityAdapterV2())
        self.assertNotEqual(toxicity_v1, toxicity_v2)
        history = _identity_signature({"data": "same", "semantics": 1})
        first = {
            "history": history,
            "sentiment": sentiment,
            "toxicity": toxicity_v1,
        }
        second = {**first, "toxicity": toxicity_v2}
        self.assertEqual(first["history"], second["history"])
        self.assertEqual(first["sentiment"], second["sentiment"])
        self.assertNotEqual(_identity_signature(first), _identity_signature(second))

    def test_local_text_cache_is_independent_of_aqua_and_lookback(self):
        fingerprint = "f" * 64
        baseline = _local_text_identity(
            FeatureBuildConfig(inference_mode=False), fingerprint
        )
        with_external_stores = _local_text_identity(
            FeatureBuildConfig(
                inference_mode=False,
                aqua_store="aqua",
                lookback_root="lookback",
            ),
            fingerprint,
        )
        self.assertEqual(baseline, with_external_stores)

    def test_local_text_checkpoint_round_trip_and_validation(self):
        source = pd.DataFrame(
            {
                "story_id": ["s1", "s1"],
                "comment_id": ["c1", "c2"],
                "effective_text": [
                    "Ein einfacher Beitrag.",
                    "Mehr Wörter und https://example.org als Verweis!",
                ],
            }
        )
        checkpoint = compute_local_text_features(source)
        validated = validate_local_text_features(checkpoint, source)
        self.assertEqual(validated["comment_id"].tolist(), ["c1", "c2"])
        self.assertEqual(validated["url_present"].tolist(), [0, 1])
        self.assertEqual(
            validated["log_words"].tolist(),
            np.log1p(validated["word_count"]).tolist(),
        )
        self.assertNotIn("effective_text", checkpoint)

        corrupted = checkpoint.copy()
        corrupted.at[0, "effective_text_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "effective_text_hash mismatch"):
            validate_local_text_features(corrupted, source)

        corrupted = checkpoint.copy()
        corrupted.at[0, "log_words"] += 0.1
        with self.assertRaisesRegex(ValueError, "log_words does not recompute"):
            validate_local_text_features(corrupted, source)

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

    def test_classifier_chunks_long_text_before_checked_model_input(self):
        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": list(range(565))}

        encoder = XLMTwitterSentimentEncoder.__new__(XLMTwitterSentimentEncoder)
        encoder.chunk_tokens = 510
        encoder._tokenizer = FakeTokenizer()
        encoder._maximum_input_tokens = 512
        encoder._sequence_diagnostics = {
            "texts": 0,
            "texts_requiring_chunking": 0,
            "chunks": 0,
            "maximum_content_tokens_observed": 0,
            "maximum_model_input_tokens_observed": 0,
            "model_input_limit": 512,
            "content_tokens_per_chunk": 510,
        }
        chunks = encoder._chunks("long text")
        self.assertEqual([len(chunk) for chunk in chunks], [510, 55])
        self.assertFalse(encoder._tokenizer.kwargs["verbose"])
        self.assertEqual(encoder.sequence_diagnostics()["texts_requiring_chunking"], 1)

    def test_new_nlp_defaults_are_pinned_and_toxicity_is_a_model_feature(self):
        config = FeatureBuildConfig(inference_mode=False)
        self.assertEqual(config.sentiment_model_id, DEFAULT_SENTIMENT_MODEL_ID)
        self.assertEqual(config.sentiment_revision, DEFAULT_SENTIMENT_MODEL_REVISION)
        self.assertEqual(config.toxicity_model_id, DEFAULT_TOXICITY_MODEL_ID)
        self.assertEqual(config.toxicity_revision, DEFAULT_TOXICITY_MODEL_REVISION)
        self.assertIn("toxicity_probability", ROOT_MODEL_FEATURES)

    def test_aqua_expected_dimensions_enter_aqua_backed_primary_models(self):
        self.assertEqual(len(AQUA_EXPECTED_MODEL_FEATURES), 20)
        self.assertNotIn("aqua_score_expected", AQUA_EXPECTED_MODEL_FEATURES)
        for scope, baseline_size in (("root", 20), ("all", 24)):
            baseline = primary_model_features(scope, aqua_available=False)
            primary = primary_model_features(scope, aqua_available=True)
            self.assertEqual(len(baseline), baseline_size)
            self.assertEqual(len(primary), baseline_size + 20)
            self.assertEqual(primary[-20:], AQUA_EXPECTED_MODEL_FEATURES)

        registry = _feature_registry(aqua_available=True)
        self.assertEqual(registry["version"], 4)
        for scope in ("root", "all"):
            self.assertEqual(
                registry["models"][scope]["features"][-20:],
                AQUA_EXPECTED_MODEL_FEATURES,
            )
        for feature in AQUA_EXPECTED_MODEL_FEATURES:
            metadata = registry["features"][feature]
            self.assertTrue(metadata["primary_model_feature"])
            self.assertFalse(metadata["descriptive_only"])
            self.assertTrue(metadata["standardize"])

    def test_production_nlp_cache_signatures_survive_diagnostic_code_changes(self):
        from commentgap_analysis.features import _adapter_implementation_signature

        sentiment = XLMTwitterSentimentEncoder.__new__(XLMTwitterSentimentEncoder)
        toxicity = TextDetoxToxicityEncoder.__new__(TextDetoxToxicityEncoder)
        self.assertEqual(
            _adapter_implementation_signature(sentiment),
            "d596848be71b609ff495690da3cce50f902d760a712cfa8423d7fa0d0c2a8c4d",
        )
        self.assertEqual(
            _adapter_implementation_signature(toxicity),
            "6b8e118aa2d8185297f3ab3c3ec4953d5dfb05b3fefe3a8ecf25c94f54d8a8e7",
        )

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
