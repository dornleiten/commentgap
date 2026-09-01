import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from commentgap_analysis.preprocessing import (
    add_author_reception,
    add_activity_rates,
    add_centered_reply_depths,
    add_reply_composition,
    reply_depth_centers,
    transformed_feature_manifest,
    validate_frozen_split,
)


class SharedPreprocessingTests(unittest.TestCase):
    def split(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "story_id": ["d0", "d1", "d2", "d3", "d4", "t"],
                "split_role": ["development"] * 5 + ["paper2_test"],
                "development_fold": [0, 1, 2, 3, 4, -1],
                "split_stratum": ["s"] * 6,
            }
        )

    def test_reply_composition_uses_smoothed_reply_root_ratio(self) -> None:
        frame = pd.DataFrame(
            {"prior_comments": [0, 3, 4], "prior_roots": [0, 1, 4]}
        )
        result = add_reply_composition(frame)
        self.assertEqual(result["prior_replies"].tolist(), [0, 2, 0])
        expected = np.log(np.array([0.5 / 0.5, 2.5 / 1.5, 0.5 / 4.5]))
        np.testing.assert_allclose(result["prior_reply_composition"], expected)

    def test_reply_composition_rejects_impossible_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "prior_comments >= prior_roots"):
            add_reply_composition(
                pd.DataFrame({"prior_comments": [1], "prior_roots": [2]})
            )

    def test_author_reception_uses_smoothed_vote_totals_per_comment(self) -> None:
        frame = pd.DataFrame(
            {
                "author_prior_30d_comments": [0, 2, 4],
                "author_prior_30d_snapshot_upvotes": [0, 5, 1],
                "author_prior_30d_snapshot_downvotes": [0, 1, 3],
            }
        )
        result = add_author_reception(frame)
        np.testing.assert_allclose(
            result["author_prior_30d_upvote_reception"],
            np.log(np.array([0.5 / 0.5, 5.5 / 2.5, 1.5 / 4.5])),
        )
        np.testing.assert_allclose(
            result["author_prior_30d_downvote_reception"],
            np.log(np.array([0.5 / 0.5, 1.5 / 2.5, 3.5 / 4.5])),
        )

    def test_author_reception_rejects_negative_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            add_author_reception(pd.DataFrame({
                "author_prior_30d_comments": [1],
                "author_prior_30d_snapshot_upvotes": [-1],
                "author_prior_30d_snapshot_downvotes": [0],
            }))

    def test_activity_rates_use_smoothed_discussion_pace(self) -> None:
        frame = pd.DataFrame({
            "hours_since_article": [0, 3, 9],
            "prior_comments": [0, 7, 19],
            "comments_prev_hour": [0, 2, 4],
        })
        result = add_activity_rates(frame)
        np.testing.assert_allclose(
            result["discussion_pace"],
            np.log(np.array([0.5 / 0.5, 7.5 / 3.5, 19.5 / 9.5])),
        )

    def test_reply_depth_centres_exclude_test_and_each_validation_fold(self) -> None:
        rows = []
        for fold in range(5):
            rows.append(
                {
                    "story_id": f"d{fold}",
                    "is_reply": 1,
                    "log_depth": float(fold + 1),
                }
            )
        rows.append({"story_id": "t", "is_reply": 1, "log_depth": 1000.0})
        centers = reply_depth_centers(pd.DataFrame(rows), self.split())
        self.assertAlmostEqual(centers["full_development"], 3.0)
        self.assertAlmostEqual(centers["fold_00_training"], 3.5)
        self.assertAlmostEqual(centers["fold_04_training"], 2.5)

    def test_centered_depth_is_zero_for_roots(self) -> None:
        frame = pd.DataFrame(
            {"is_reply": [0, 1, 1], "log_depth": [0.0, 2.0, 4.0]}
        )
        result = add_centered_reply_depths(
            frame,
            {"full_development": 3.0, "fold_00_training": 2.5},
        )
        np.testing.assert_allclose(result["reply_depth_centered"], [0, -1, 1])
        np.testing.assert_allclose(
            result["reply_depth_centered_fold_00"], [0, -0.5, 1.5]
        )

    def test_manifest_replaces_only_the_model_contract(self) -> None:
        source = {
            "features": {
                "log_prior_roots": {"standardize": True},
                "log_prior_comments": {"standardize": True},
                "log_comments_prev_hour": {"standardize": True},
                "log_branch_prior_comments": {"standardize": True},
                "log_branch_comments_prev_hour": {"standardize": True},
                "is_reply": {"standardize": False},
                "log_depth": {"standardize": True},
                "log_author_prior_30d_snapshot_upvotes": {"standardize": True},
                "log_author_prior_30d_snapshot_downvotes": {
                    "standardize": True
                },
            },
            "models": {
                "root": {"features": [
                    "log_prior_roots",
                    "log_prior_comments",
                    "log_comments_prev_hour",
                    "log_author_prior_30d_snapshot_upvotes",
                    "log_author_prior_30d_snapshot_downvotes",
                ]},
                "all": {
                    "features": [
                        "log_prior_roots",
                        "log_prior_comments",
                        "log_comments_prev_hour",
                        "log_branch_prior_comments",
                        "log_branch_comments_prev_hour",
                        "log_author_prior_30d_snapshot_upvotes",
                        "log_author_prior_30d_snapshot_downvotes",
                        "is_reply",
                        "log_depth",
                    ]
                },
            },
        }
        result = transformed_feature_manifest(source)
        self.assertEqual(
            result["models"]["root"]["features"],
            [
                "prior_reply_composition",
                "discussion_pace",
                "author_prior_30d_upvote_reception",
                "author_prior_30d_downvote_reception",
            ],
        )
        self.assertEqual(
            result["models"]["all"]["features"],
            [
                "prior_reply_composition",
                "discussion_pace",
                "author_prior_30d_upvote_reception",
                "author_prior_30d_downvote_reception",
                "is_reply",
                "reply_depth_centered",
            ],
        )

    def test_split_validation_refuses_test_fold_assignment(self) -> None:
        split = self.split()
        split.loc[split["split_role"] == "paper2_test", "development_fold"] = 0
        with self.assertRaisesRegex(ValueError, "development_fold = -1"):
            validate_frozen_split(split)


if __name__ == "__main__":
    unittest.main()
