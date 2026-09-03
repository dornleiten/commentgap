import unittest

import numpy as np
import pandas as pd

from commentgap_analysis.forum_scores import (
    ALL_ORDERINGS,
    ALL_OUTCOMES,
    CHOICE_COLUMNS,
    FORUM_OUTCOMES,
    PREDICTIVE_SCORE_COLUMNS,
    PolicySpec,
    REGRESSION_FEATURE_COLUMNS,
    bootstrap_policy_decomposition,
    forum_score,
    make_policy_order,
    ndcg_score,
    policy_design_matrix,
    static_knn_novelty,
    policy_specs,
    score_story_policies,
    static_nearest_neighbor_novelty,
)


class ForumScoresTests(unittest.TestCase):
    def test_forum_outcomes_are_exactly_the_selected_outcomes(self):
        self.assertEqual(len(REGRESSION_FEATURE_COLUMNS), 41)
        self.assertTrue(
            set(REGRESSION_FEATURE_COLUMNS).issubset(CHOICE_COLUMNS)
        )
        self.assertEqual(FORUM_OUTCOMES, ALL_OUTCOMES)
        self.assertNotIn("lexdiv_length_adjusted", ALL_OUTCOMES)
        self.assertNotIn("reading_level_length_adjusted", ALL_OUTCOMES)

    def test_policy_scoring_rejects_nonpositive_worker_count(self):
        from commentgap_analysis.forum_scores import run_policy_scoring

        with self.assertRaisesRegex(ValueError, "workers must be positive"):
            run_policy_scoring(
                analysis_path="missing-analysis.parquet",
                workers=0,
            )

    def _story(self) -> pd.DataFrame:
        n_comments = 11
        roots = {0: "c0", 6: "c6"}
        root_ids = ["c0"] * 6 + ["c6"] * 5
        frame = pd.DataFrame(
            {
                "story_id": ["s1"] * n_comments,
                "comment_id": [f"c{index}" for index in range(n_comments)],
                "is_root": [index in roots for index in range(n_comments)],
                "root_comment_id": root_ids,
                "parent_comment_id": [
                    None,
                    "c0",
                    "c0",
                    "c1",
                    "c0",
                    "c4",
                    None,
                    "c6",
                    "c6",
                    "c8",
                    "c6",
                ],
                "created_at": pd.date_range(
                    "2025-01-01", periods=n_comments, freq="min", tz="UTC"
                ),
                "preorder_position": np.arange(n_comments),
                "display_order": np.arange(n_comments),
                "root_order": [0] * 6 + [1] * 5,
                "is_sticky": [
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                "relative_votes": np.arange(n_comments, dtype=float),
                "votes_positive": np.arange(n_comments, dtype=float) + 10,
                "votes_negative": np.repeat(10.0, n_comments),
                "test_outcome": np.linspace(-1, 1, n_comments),
            }
        )
        for position, column in enumerate(PREDICTIVE_SCORE_COLUMNS):
            frame[column] = np.arange(n_comments, dtype=float) + position / 100
        return frame

    def test_factorial_has_random_grounding_controls(self):
        specs = policy_specs()
        self.assertEqual(len(specs), 90)
        self.assertEqual(sum(spec.deployable for spec in specs), 84)
        self.assertEqual(len(ALL_ORDERINGS), 15)
        random = [spec for spec in specs if spec.ordering == "random"]
        self.assertEqual(len(random), 6)
        self.assertTrue(all(not spec.deployable for spec in random))

    def test_structure_respecting_pinning(self):
        story = self._story()
        loose = make_policy_order(
            story, PolicySpec("relative_votes", "loose", True)
        )
        trees = make_policy_order(
            story, PolicySpec("relative_votes", "trees", True)
        )
        hidden = make_policy_order(
            story, PolicySpec("relative_votes", "hidden", True)
        )
        self.assertEqual(loose[:2].tolist(), [1, 6])
        self.assertEqual(trees.tolist(), [6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5])
        self.assertEqual(hidden.tolist(), [6, 0])
        self.assertEqual(sorted(trees.tolist()), list(range(len(story))))

    def test_forum_anchors_and_affine_invariance(self):
        values = np.asarray([0.0, 1.0, 2.0, 3.0])
        best = np.argsort(-values)
        worst = np.argsort(values)
        self.assertAlmostEqual(forum_score(values, best, depth=3), 1.0)
        self.assertAlmostEqual(forum_score(values, worst, depth=3), -1.0)
        transformed = 7.0 + 3.5 * values
        self.assertAlmostEqual(
            forum_score(values, [2, 0, 3, 1], depth=3),
            forum_score(transformed, [2, 0, 3, 1], depth=3),
        )

    def test_hidden_forum_interpolates_to_full_length(self):
        values = np.asarray([4.0, 3.0, 2.0, 1.0])
        score = forum_score(values, [0, 2], depth=3, hidden=True)
        self.assertTrue(np.isfinite(score))
        with self.assertRaisesRegex(ValueError, "every comment"):
            forum_score(values, [0, 2], depth=3, hidden=False)

    def test_direct_ndcg_prefers_ideal_permutation(self):
        values = np.asarray([-2.0, 0.0, 1.0, 4.0])
        self.assertAlmostEqual(ndcg_score(values, [3, 2, 1, 0], depth=3), 1.0)
        self.assertLess(
            ndcg_score(values, [0, 1, 2, 3], depth=3),
            ndcg_score(values, [3, 2, 1, 0], depth=3),
        )

    def test_static_novelty_uses_nearest_other_comment(self):
        vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        novelty, metadata = static_nearest_neighbor_novelty(vectors)
        np.testing.assert_allclose(novelty, [0.0, 0.0, 1.0], atol=1e-7)
        self.assertEqual(metadata["method"], "exact_blockwise_cosine")

    def test_static_knn_novelty_averages_five_nearest_comments(self):
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        novelty, metadata = static_knn_novelty(vectors)
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        distances = 1.0 - normalized @ normalized.T
        np.fill_diagonal(distances, np.inf)
        expected = np.sort(distances, axis=1)[:, :5].mean(axis=1)
        np.testing.assert_allclose(novelty, expected)
        self.assertEqual(metadata["effective_neighbors"], 5)

    def test_policy_scoring_provides_ndcg_for_all_complete_policy_orders(self):
        scores = score_story_policies(
            self._story(),
            outcomes=["test_outcome"],
            tie_draws=1,
            random_draws=3,
        )
        self.assertEqual(len(scores), 180)
        self.assertTrue(scores["ndcg"].notna().all())
        self.assertTrue(scores["reply_mode"].eq("hidden").any())
        self.assertEqual(
            set(scores.loc[scores["ordering"].eq("random"), "ordering_draws"]),
            {3},
        )

    def test_deleted_ancestors_form_an_induced_visible_forest(self):
        story = self._story().iloc[1:].reset_index(drop=True)
        tree = make_policy_order(
            story, PolicySpec("relative_votes", "trees", False)
        )
        hidden = make_policy_order(
            story, PolicySpec("relative_votes", "hidden", False)
        )
        self.assertEqual(sorted(tree.tolist()), list(range(len(story))))
        self.assertEqual(len(hidden), 4)
        self.assertEqual(
            set(story.iloc[hidden]["comment_id"]),
            {"c1", "c2", "c4", "c6"},
        )

    def test_vectorized_policy_score_matches_scalar_functions(self):
        story = self._story()
        scores = score_story_policies(
            story,
            outcomes=["test_outcome"],
            tie_draws=1,
            random_draws=2,
        )
        row = scores[
            scores["ordering"].eq("relative_votes")
            & scores["reply_mode"].eq("loose")
            & ~scores["pinned"]
            & scores["depth"].eq("full")
        ].iloc[0]
        order = make_policy_order(
            story, PolicySpec("relative_votes", "loose", False), draw=1
        )
        values = story["test_outcome"].to_numpy(float)
        self.assertAlmostEqual(
            row["forum"],
            forum_score(values, order, depth=len(story) - 1),
        )
        self.assertAlmostEqual(
            row["ndcg"],
            ndcg_score(values, order, depth=len(story) - 1),
        )

    def test_random_loose_unpinned_is_design_reference(self):
        metadata = pd.DataFrame(
            [
                {
                    "policy_id": spec.policy_id,
                    "ordering": spec.ordering,
                    "reply_mode": spec.reply_mode,
                    "pinned": spec.pinned,
                    "deployable": spec.deployable,
                }
                for spec in policy_specs()
            ]
        )
        design, names = policy_design_matrix(metadata)
        self.assertEqual(design.shape, (90, 62))
        self.assertEqual(np.linalg.matrix_rank(design), 62)
        reference = metadata[
            metadata["ordering"].eq("random")
            & metadata["reply_mode"].eq("loose")
            & ~metadata["pinned"]
        ].index.item()
        np.testing.assert_array_equal(
            design[reference], np.r_[1.0, np.zeros(61)]
        )
        self.assertEqual(names[0], "intercept[random,loose,unpinned]")


if __name__ == "__main__":
    unittest.main()
