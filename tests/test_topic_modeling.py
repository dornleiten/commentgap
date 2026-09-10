import unittest

import numpy as np
import pandas as pd

from commentgap_analysis.topic_modeling import (
    aggregate_topic_distributions,
    prepare_documents,
    TopicModelBundle,
    TopicModelConfig,
)
from commentgap_analysis.topic_metrics import (
    build_topic_agenda_rarefaction_analysis,
    classify_projection,
    compute_topic_metrics,
    effective_topic_count,
    hellinger_distance,
    hellinger_projection,
    jensen_shannon_distance,
    cosine_similarity,
    shannon_entropy,
    topicwise_gap_ratios,
)
from commentgap_analysis.topic_policy import (
    _policy_contrast_values,
    _rank_attention_weights,
    aggregate_topic_policy_draw_distributions,
    aggregate_topic_policy_draw_metrics,
    build_policy_reference_target_metrics,
    build_topic_policy_exposure_coverage_analysis,
    exact_hellinger_oracle_order,
    hellinger_oracle_order,
    paired_policy_contrasts,
    ranked_topic_trajectory,
    weighted_ranked_topic_distributions,
)
from commentgap_analysis.topic_plotting import (
    plot_article_relative_votes_marginal_effects,
)


def _memberships() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "story_id": ["s1", "s1", "s1", "s1"],
            "comment_id": ["c1", "c2", "c3", "c4"],
            "doc_type": ["comment"] * 4,
            "topic_000": [1.0, 0.0, 0.8, 0.2],
            "topic_001": [0.0, 1.0, 0.2, 0.8],
        }
    )


class TopicModelBundleTests(unittest.TestCase):
    def test_soft_memberships_are_retained(self):
        class FakeBERTopic:
            def transform(self, texts, embeddings=None):
                return np.array([0, 1]), np.array([[0.8, 0.2], [0.1, 0.9]])

        bundle = TopicModelBundle(
            config=TopicModelConfig(calculate_probabilities=True),
            vectorizer=None,
            model=FakeBERTopic(),
            topic_terms=(("first",), ("second",)),
            topic_ids=(0, 1),
        )
        documents = pd.DataFrame({
            "doc_id": ["d1", "d2"],
            "story_id": ["s1", "s1"],
            "doc_type": ["comment", "comment"],
            "text": ["first", "second"],
        })
        result = bundle.transform(documents)
        np.testing.assert_allclose(
            result[["topic_000", "topic_001"]].to_numpy(),
            [[0.8, 0.2], [0.1, 0.9]],
        )


class ReferenceTargetMetricTests(unittest.TestCase):
    def test_random_reference_averages_metrics_before_subtraction(self):
        visible = pd.DataFrame({
            "story_id": ["s1"] * 4,
            "policy_id": ["random__loose__unpinned"] * 2 + [
                "relative_votes__loose__unpinned", "chronological__loose__unpinned",
            ],
            "draw": [1, 2, 1, 1],
            "topic_000": [1.0, 0.0, 0.5, 0.5],
            "topic_001": [0.0, 1.0, 0.5, 0.5],
        })
        article = pd.DataFrame({
            "story_id": ["s1"], "topic_000": [0.5], "topic_001": [0.5],
        })
        metrics = visible[["story_id", "policy_id"]].drop_duplicates().assign(
            ordering=["random", "relative_votes", "chronological"],
            reply_mode="loose", pinned=False,
        )
        result = build_policy_reference_target_metrics(
            visible, article, metrics,
            topic_columns=["topic_000", "topic_001"], chunk_size=1,
        )
        random = result[result.policy_id.eq("random__loose__unpinned")]
        np.testing.assert_allclose(random[["js_raw_target_gain", "cosine_raw_target_gain"]], 0)
        expected_distance = jensen_shannon_distance([1, 0], [0.5, 0.5])
        expected_similarity = cosine_similarity([1, 0], [0.5, 0.5])
        np.testing.assert_allclose(random.js_distance_random_to_target, expected_distance)
        np.testing.assert_allclose(random.cosine_similarity_random_to_target, expected_similarity)
        policy = result[result.policy_id.eq("chronological__loose__unpinned")]
        np.testing.assert_allclose(policy.js_raw_target_gain, expected_distance)
        np.testing.assert_allclose(policy.cosine_raw_target_gain, 1 - expected_similarity)
        self.assertTrue(random.n_draws.eq(2).all())

    def test_reference_gain_plot_can_use_normalized_target_gap(self):
        import tempfile
        from pathlib import Path

        rows = []
        for policy_id, ordering, js_article, js_votes, cosine_article, cosine_votes in [
            ("random__loose__unpinned", "random", 0.0, 0.0, 0.0, 0.0),
            ("chronological__loose__unpinned", "chronological", 0.4, 0.2, 0.2, 0.1),
        ]:
            rows.extend([
                {
                    "story_id": "s1", "policy_id": policy_id, "ordering": ordering,
                    "reply_mode": "loose", "pinned": False, "target": "article",
                    "js_raw_target_gain": js_article,
                    "js_distance_random_to_target": 0.8,
                    "cosine_raw_target_gain": cosine_article,
                    "cosine_similarity_random_to_target": 0.5,
                },
                {
                    "story_id": "s1", "policy_id": policy_id, "ordering": ordering,
                    "reply_mode": "loose", "pinned": False, "target": "relative_votes",
                    "js_raw_target_gain": js_votes,
                    "js_distance_random_to_target": 0.4,
                    "cosine_raw_target_gain": cosine_votes,
                    "cosine_similarity_random_to_target": 0.8,
                },
            ])
        with tempfile.TemporaryDirectory() as directory:
            result = plot_article_relative_votes_marginal_effects(
                pd.DataFrame(rows),
                contrast_order=[("ordering_vs_random", "chronological")],
                output_root=Path(directory),
                ordering_labels={"chronological": "Chronological"},
                reply_labels={"loose": "Loose"},
                normalize_gains=True,
            )
        js = result.loc[result.metric.eq("js_raw_target_gain")].iloc[0]
        cosine = result.loc[result.metric.eq("cosine_raw_target_gain")].iloc[0]
        self.assertAlmostEqual(js["mean"], 0.4 / 0.8 - 0.2 / 0.4)
        self.assertAlmostEqual(cosine["mean"], 0.2 / 0.5 - 0.1 / 0.2)
        self.assertTrue((result.metric_label.str.contains("normalized", case=False)).all())

    def test_normalized_gain_plot_explains_stale_reference_artifact(self):
        import tempfile
        from pathlib import Path

        stale = pd.DataFrame({
            "story_id": ["s1", "s1"],
            "policy_id": ["random__loose__unpinned"] * 2,
            "ordering": ["random"] * 2,
            "reply_mode": ["loose"] * 2,
            "pinned": [False] * 2,
            "target": ["article", "relative_votes"],
            "js_raw_target_gain": [0.0, 0.0],
            "js_distance_random_to_target": [0.1, 0.1],
            "cosine_raw_target_gain": [0.0, 0.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Rerun notebook 15"):
                plot_article_relative_votes_marginal_effects(
                    stale,
                    contrast_order=[],
                    output_root=Path(directory),
                    ordering_labels={},
                    reply_labels={},
                    normalize_gains=True,
                )

    def test_reference_targets_expose_raw_gains_without_normalized_progress(self):
        visible = pd.DataFrame({
            "story_id": ["s1", "s1", "s1"],
            "policy_id": [
                "random__loose__unpinned",
                "relative_votes__loose__unpinned",
                "chronological__loose__unpinned",
            ],
            "topic_000": [0.5, 1.0, 0.75],
            "topic_001": [0.5, 0.0, 0.25],
        })
        article = pd.DataFrame({
            "story_id": ["s1"],
            "topic_000": [1.0],
            "topic_001": [0.0],
        })
        metrics = pd.DataFrame({
            "story_id": ["s1"] * 3,
            "policy_id": visible["policy_id"],
            "ordering": ["random", "relative_votes", "chronological"],
            "reply_mode": ["loose"] * 3,
            "pinned": [False] * 3,
        })
        result = build_policy_reference_target_metrics(
            visible,
            article,
            metrics,
            topic_columns=["topic_000", "topic_001"],
        )
        policy_article = result[
            (result["policy_id"] == "chronological__loose__unpinned")
            & (result["target"] == "article")
        ].iloc[0]
        expected_js_gain = (
            jensen_shannon_distance([0.5, 0.5], [1.0, 0.0])
            - jensen_shannon_distance([0.75, 0.25], [1.0, 0.0])
        )
        self.assertAlmostEqual(policy_article["js_raw_target_gain"], expected_js_gain)
        self.assertNotIn("progress_toward_target", result.columns)
        self.assertNotIn("cosine_progress_to_target", result.columns)


class DistributionMetricTests(unittest.TestCase):
    def test_entropy_and_effective_topic_count(self):
        self.assertAlmostEqual(shannon_entropy([0.5, 0.5]), np.log(2))
        self.assertAlmostEqual(effective_topic_count([0.5, 0.5]), 2.0)
        self.assertAlmostEqual(shannon_entropy([1.0, 0.0]), 0.0)

    def test_js_distance_is_symmetric_and_zero_for_identity(self):
        self.assertAlmostEqual(jensen_shannon_distance([1, 0], [1, 0]), 0.0)
        self.assertAlmostEqual(
            jensen_shannon_distance([1, 0], [0, 1]),
            jensen_shannon_distance([0, 1], [1, 0]),
        )

    def test_js_distance_uses_natural_log_scale(self):
        self.assertAlmostEqual(
            jensen_shannon_distance([1.0, 0.0], [0.0, 1.0]),
            np.sqrt(np.log(2.0)),
        )

    def test_raw_cosine_is_ordinary_proportion_cosine(self):
        left = np.array([0.8, 0.2])
        right = np.array([0.6, 0.4])
        expected = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
        self.assertAlmostEqual(cosine_similarity(left, right), expected)
        self.assertNotAlmostEqual(
            cosine_similarity(left, right),
            float(np.dot(np.sqrt(left), np.sqrt(right))),
        )

    def test_hellinger_distance_has_conventional_unit_scale(self):
        self.assertAlmostEqual(hellinger_distance([1, 0], [1, 0]), 0.0)
        self.assertAlmostEqual(hellinger_distance([1, 0], [0, 1]), 1.0)

    def test_hellinger_oracle_improves_article_visible_distance(self):
        article = [1.0, 0.0]
        comments = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        result = hellinger_oracle_order(article, comments, max_iterations=5)
        self.assertEqual(result["order"].tolist()[0], 1)
        self.assertLessEqual(result["hellinger_distance"], result["initial_hellinger_distance"])

    def test_hellinger_oracle_multistart_is_reproducible_and_nonworsening(self):
        article = np.array([0.7, 0.2, 0.1])
        comments = np.array([
            [0.9, 0.1, 0.0],
            [0.05, 0.9, 0.05],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
            [0.1, 0.2, 0.7],
        ])
        single = hellinger_oracle_order(
            article,
            comments,
            n_starts=1,
            random_state=2025,
            objective="cosine_similarity",
        )
        first = hellinger_oracle_order(
            article,
            comments,
            n_starts=8,
            random_state=2025,
            n_perturbations=2,
            perturbation_fraction=0.5,
            objective="cosine_similarity",
        )
        second = hellinger_oracle_order(
            article,
            comments,
            n_starts=8,
            random_state=2025,
            n_perturbations=2,
            perturbation_fraction=0.5,
            objective="cosine_similarity",
        )
        np.testing.assert_array_equal(first["order"], second["order"])
        self.assertEqual(first["best_start"], second["best_start"])
        self.assertEqual(first["total_iterations"], second["total_iterations"])
        self.assertLessEqual(first["hellinger_distance"], single["hellinger_distance"] + 1e-12)
        self.assertEqual(first["n_starts"], 8)
        self.assertEqual(first["n_perturbations"], 2)
        self.assertEqual(first["accepted_perturbations"], second["accepted_perturbations"])

    def test_exact_oracle_validates_heuristic_on_small_thread(self):
        article = np.array([0.7, 0.2, 0.1])
        comments = np.array([
            [0.9, 0.1, 0.0],
            [0.05, 0.9, 0.05],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
        ])
        exact = exact_hellinger_oracle_order(
            article,
            comments,
            objective="cosine_similarity",
            max_valid_comments=8,
        )
        heuristic = hellinger_oracle_order(
            article,
            comments,
            objective="cosine_similarity",
            n_starts=4,
            n_perturbations=2,
            n_local_moves=2,
            random_state=2025,
        )
        self.assertEqual(exact["n_permutations"], 24)
        self.assertLessEqual(
            exact["hellinger_distance"],
            heuristic["hellinger_distance"] + 1e-12,
        )

    def test_exact_oracle_can_place_invalid_comments_between_valid_comments(self):
        article = np.array([1.0, 0.0])
        comments = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        result = exact_hellinger_oracle_order(
            article,
            comments,
            valid_mask=[True, True, False],
            max_valid_comments=3,
            allow_invalid_placement=True,
        )
        self.assertEqual(result["n_permutations"], 6)
        self.assertEqual(result["n_comments"], 3)
        self.assertNotEqual(result["order"].tolist()[-1], 2)
        heuristic = hellinger_oracle_order(
            article,
            comments,
            valid_mask=[True, True, False],
            n_starts=30,
            n_perturbations=2,
            random_state=2,
        )
        self.assertLessEqual(
            heuristic["hellinger_distance"], result["hellinger_distance"] + 1e-12
        )
        self.assertTrue(heuristic["allow_invalid_placement"])
        fixed = exact_hellinger_oracle_order(
            article,
            comments,
            valid_mask=[True, True, False],
            max_valid_comments=2,
            allow_invalid_placement=False,
        )
        self.assertEqual(fixed["n_permutations"], 2)

    def test_oracle_can_optimize_jensen_shannon_distance(self):
        article = np.array([0.7, 0.2, 0.1])
        comments = np.array([
            [0.9, 0.1, 0.0],
            [0.05, 0.9, 0.05],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
        ])
        result = hellinger_oracle_order(
            article,
            comments,
            objective="jensen_shannon_distance",
            n_starts=4,
            n_perturbations=2,
            n_local_moves=2,
            random_state=2025,
        )
        self.assertEqual(result["objective"], "jensen_shannon_distance")
        self.assertAlmostEqual(
            result["objective_value"],
            jensen_shannon_distance(article, result["distribution"]),
        )

    def test_hellinger_oracle_rejects_nonpositive_start_count(self):
        with self.assertRaises(ValueError):
            hellinger_oracle_order(
                [1.0, 0.0],
                np.array([[1.0, 0.0]]),
                n_starts=0,
            )
        with self.assertRaises(ValueError):
            hellinger_oracle_order(
                [1.0, 0.0],
                np.array([[1.0, 0.0]]),
                n_perturbations=1,
                perturbation_fraction=0.0,
            )

    def test_projection_detects_overshoot(self):
        result = hellinger_projection([0.6, 0.4], [0.3, 0.7], [0.8, 0.2])
        self.assertGreater(result["hellinger_alpha"], 1.0)
        self.assertEqual(classify_projection(result["hellinger_alpha"]), "overshoot")
        self.assertGreaterEqual(result["hellinger_residual"], 0.0)

    def test_topicwise_small_gaps_are_not_ratioed(self):
        result = topicwise_gap_ratios([0.51, 0.49], [0.50, 0.50], [0.60, 0.40])
        self.assertTrue(result.loc[0, "small_baseline_gap"])
        self.assertTrue(np.isnan(result.loc[0, "alpha_topic"]))


class TopicTableTests(unittest.TestCase):
    def test_aggregate_and_ranked_trajectory(self):
        memberships = _memberships()
        complete = aggregate_topic_distributions(memberships)
        self.assertEqual(complete.loc[0, "n_documents"], 4)
        self.assertAlmostEqual(complete.loc[0, "topic_000"] + complete.loc[0, "topic_001"], 1.0)

        rankings = pd.DataFrame(
            {
                "story_id": ["s1"] * 4,
                "policy_id": ["policy"] * 4,
                "comment_id": ["c1", "c3", "c2", "c4"],
                "rank": [1, 2, 3, 4],
            }
        )
        trajectory = ranked_topic_trajectory(memberships, rankings)
        self.assertEqual(len(trajectory), 4)
        self.assertAlmostEqual(trajectory.iloc[0]["topic_000"], 1.0)
        self.assertAlmostEqual(trajectory.iloc[-1]["topic_000"], complete.iloc[0]["topic_000"])

    def test_inverse_rank_distribution_uses_full_ranking(self):
        memberships = _memberships()
        comments = pd.DataFrame({
            "story_id": ["s1"] * 4,
            "comment_id": ["c1", "c2", "c3", "c4"],
            "score": [4.0, 3.0, 2.0, 1.0],
        })
        visible = weighted_ranked_topic_distributions(
            memberships, comments, {"score": "score"}, random_draws=1
        )
        weights = 1.0 / np.arange(1, 5, dtype=float)
        values = memberships[["topic_000", "topic_001"]].to_numpy()
        expected = (values * weights[:, None]).sum(axis=0)
        expected /= expected.sum()
        self.assertEqual(len(visible), 1)
        np.testing.assert_allclose(
            visible[["topic_000", "topic_001"]].to_numpy()[0], expected
        )
        self.assertEqual(visible.loc[0, "n_documents"], 4)

    def test_metrics_report_toward_and_concentration_measures(self):
        article = pd.DataFrame({"story_id": ["s1"], "topic_000": [0.6], "topic_001": [0.4]})
        discussion = pd.DataFrame({"story_id": ["s1"], "topic_000": [0.3], "topic_001": [0.7]})
        visible = pd.DataFrame(
            {"story_id": ["s1"], "policy_id": ["policy"], "depth": [10], "topic_000": [0.5], "topic_001": [0.5]}
        )
        metrics = compute_topic_metrics(article, discussion, visible)
        self.assertGreater(metrics.loc[0, "alignment_gain"], 0.0)
        self.assertGreater(metrics.loc[0, "cosine_progress"], 0.0)
        self.assertIn(metrics.loc[0, "projection_class"], {"partial_convergence", "matches_article"})
        expected_ratio = np.exp(shannon_entropy([0.5, 0.5]) - shannon_entropy([0.3, 0.7]))
        self.assertAlmostEqual(metrics.loc[0, "visible_discussion_effective_topic_ratio"], expected_ratio)

    def test_paired_policy_contrast_averages_repeated_controls(self):
        metrics = pd.DataFrame(
            {
                "story_id": ["s1", "s1", "s2", "s2", "s1", "s2"],
                "policy_id": ["random", "random", "random", "random", "policy", "policy"],
                "depth": [1, 1, 1, 1, 1, 1],
                "alignment_gain": [0.1, 0.3, 0.2, 0.2, 0.5, 0.7],
            }
        )
        result = paired_policy_contrasts(metrics, metric="alignment_gain", bootstrap_draws=20)
        self.assertEqual(result.loc[0, "n_stories"], 2)
        self.assertAlmostEqual(result.loc[0, "estimate"], 0.4)

    def test_interface_contrasts_pair_within_ordering_and_interface(self):
        frame = pd.DataFrame({
            "story_id": ["s1"] * 5,
            "ordering": ["random", "chronological", "chronological", "chronological", "chronological"],
            "reply_mode": ["loose", "loose", "trees", "loose", "trees"],
            "pinned": [False, False, False, True, True],
            "metric": [0.5, 0.4, 0.6, 0.7, 0.8],
        })
        trees = _policy_contrast_values(frame, "reply_vs_loose", "trees", "metric")
        pinned = _policy_contrast_values(frame, "pinned_vs_unpinned", "pinned", "metric")
        self.assertAlmostEqual(trees.loc[0, "difference"], 0.15)
        self.assertAlmostEqual(pinned.loc[0, "difference"], 0.25)

    def test_ordering_contrast_averages_reply_pin_variants(self):
        frame = pd.DataFrame({
            "story_id": ["s1"] * 12,
            "ordering": ["random"] * 6 + ["chronological"] * 6,
            "reply_mode": ["loose", "trees", "hidden"] * 4,
            "pinned": [False, False, False, True, True, True] * 2,
            "metric": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9],
        })
        result = _policy_contrast_values(
            frame, "ordering_vs_random", "chronological", "metric"
        )
        self.assertAlmostEqual(result.loc[0, "treatment"], 0.8)
        self.assertAlmostEqual(result.loc[0, "control"], 0.2)
        self.assertAlmostEqual(result.loc[0, "difference"], 0.6)

    def test_exposure_coverage_summary_validates_and_relates_alignment(self):
        metrics = pd.DataFrame({
            "story_id": ["s1", "s2"],
            "policy_id": ["p1", "p1"],
            "ordering": ["score", "score"],
            "reply_mode": ["loose", "loose"],
            "pinned": [False, False],
            "assigned_exposure_coverage": [0.25, 0.75],
            "alignment_gain": [0.1, 0.3],
            "article_visible_js_distance": [0.5, 0.3],
        })
        result = build_topic_policy_exposure_coverage_analysis(metrics)
        self.assertAlmostEqual(
            result["summary"].loc[0, "assigned_exposure_coverage_mean"], 0.5
        )
        self.assertGreater(
            result["correlations"].loc[
                result["correlations"]["level"].eq("story_policy"),
                "pearson_coverage_alignment_gain",
            ].iloc[0],
            0.99,
        )

    def test_draw_metrics_are_averaged_after_metric_calculation(self):
        visible_draws = pd.DataFrame({
            "story_id": ["s1", "s1"],
            "policy_id": ["p1", "p1"],
            "ordering": ["score", "score"],
            "reply_mode": ["loose", "loose"],
            "pinned": [False, False],
            "draw": [1, 2],
            "topic_000": [1.0, 0.0],
            "topic_001": [0.0, 1.0],
        })
        visible = aggregate_topic_policy_draw_distributions(visible_draws)
        self.assertAlmostEqual(visible.loc[0, "topic_000"], 0.5)
        metrics_draws = pd.DataFrame({
            "story_id": ["s1", "s1"],
            "policy_id": ["p1", "p1"],
            "ordering": ["score", "score"],
            "reply_mode": ["loose", "loose"],
            "pinned": [False, False],
            "draw": [1, 2],
            "alignment_gain": [0.1, 0.3],
            "visible_entropy": [0.0, 0.7],
        })
        metrics = aggregate_topic_policy_draw_metrics(metrics_draws)
        self.assertEqual(metrics.loc[0, "n_draws"], 2)
        self.assertAlmostEqual(metrics.loc[0, "alignment_gain"], 0.2)

    def test_rank_attention_weights_support_exponential_decay(self):
        inverse = _rank_attention_weights(3, rank_weight_mode="inverse_power")
        exponential = _rank_attention_weights(
            3, rank_weight_mode="exponential", rank_decay=np.log(2)
        )
        np.testing.assert_allclose(inverse, [1.0, 0.5, 1 / 3])
        np.testing.assert_allclose(exponential, [1.0, 0.5, 0.25])
        with self.assertRaises(ValueError):
            _rank_attention_weights(3, rank_weight_mode="exponential", rank_decay=0)

    def test_rarefaction_matches_article_and_discussion_sample_sizes(self):
        memberships = pd.DataFrame({
            "story_id": ["s1"] * 6,
            "doc_type": ["article", "article", "comment", "comment", "comment", "comment"],
            "valid_topic": [True, True, True, True, True, True],
            "topic_000": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "topic_001": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        })
        result = build_topic_agenda_rarefaction_analysis(memberships, n_draws=3, seed=7)
        self.assertEqual(result["story"]["sample_size"].unique().tolist(), [2])
        self.assertEqual(len(result["story"]), 3)
        self.assertEqual(result["summary"]["n_stories"].iloc[0], 1)

    def test_prepare_documents_uses_effective_text(self):
        articles = pd.DataFrame({"story_id": ["s1"], "title": ["News"], "subtitle": [None], "body": ["Article text"]})
        comments = pd.DataFrame({"story_id": ["s1"], "comment_id": ["c1"], "effective_text": ["Comment text"]})
        documents = prepare_documents(articles, comments)
        self.assertEqual(set(documents["doc_type"]), {"article", "comment"})
        self.assertEqual(documents.loc[documents["doc_type"].eq("article"), "text"].iloc[0], "News\n\nArticle text")


if __name__ == "__main__":
    unittest.main()
