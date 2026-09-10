import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from commentgap_analysis.explanations import (
    _collapse_feature_values,
    _deterministic_sample,
    _story_aggregate,
    run_shap_explanations,
)


class ExplanationTests(unittest.TestCase):
    def test_deterministic_sample_retains_story_coverage(self):
        frame = pd.DataFrame(
            {
                "story_id": ["b", "a", "c", "b", "a", "c"],
                "comment_id": ["b2", "a2", "c2", "b1", "a1", "c1"],
            }
        )
        first = _deterministic_sample(frame, 4, seed=11)
        second = _deterministic_sample(frame.sample(frac=1, random_state=3), 4, seed=11)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["story_id"]), {"a", "b", "c"})
        self.assertEqual(len(first), 4)

    def test_collapse_feature_values_sums_embedding_dimensions(self):
        values = np.zeros((2, 2 + 1024 + 1), dtype=float)
        values[:, :2] = [[1.0, -2.0], [3.0, 4.0]]
        values[:, 2 : 2 + 1024] = 0.5
        values[:, -1] = [0.0, 1.0]
        collapsed, columns, groups = _collapse_feature_values(
            values, features=["feature_a", "feature_b"], has_text=True, has_selector_code=True
        )
        self.assertEqual(columns, ["feature_a", "feature_b", "text_bge", "selector_code"])
        self.assertEqual(groups, ["metadata", "metadata", "text", "selector"])
        np.testing.assert_allclose(collapsed[0], [1.0, -2.0, 512.0, 0.0])
        np.testing.assert_allclose(collapsed[1], [3.0, 4.0, 512.0, 1.0])

    def test_story_aggregate_averages_comments_within_story(self):
        rows = []
        for index, (audience_value, curator_value) in enumerate(
            zip((1.0, 3.0), (2.0, 4.0), strict=True)
        ):
            for selector, value in (("audience", audience_value), ("curator", curator_value)):
                rows.append(
                    {
                        "story_id": "s1",
                        "comment_id": f"s1-{index}",
                        "scope": "all",
                        "model_family": "xgboost",
                        "model_id": "winner",
                        "feature_set": "metadata",
                        "selector": selector,
                        "shap_feature_a": value,
                        "shap_selector_code": 0.0,
                    }
                )
        summary = _story_aggregate(pd.DataFrame(rows))
        result = summary.iloc[0]
        self.assertEqual(result["feature"], "feature_a")
        self.assertAlmostEqual(result["mean_shap_audience"], 2.0)
        self.assertAlmostEqual(result["mean_shap_curator"], 3.0)
        self.assertAlmostEqual(result["mean_shap_gap"], 1.0)
        self.assertEqual(result["n_stories"], 1)
        self.assertEqual(result["sampled_comments"], 4)

    def test_story_aggregate_weights_unequal_stories_equally(self):
        rows = []
        for story_id, audience, curator in (
            ("short", (1.0,), (3.0,)),
            ("long", (5.0, 5.0, 5.0), (9.0, 9.0, 9.0)),
        ):
            for index, (audience_value, curator_value) in enumerate(
                zip(audience, curator, strict=True)
            ):
                for selector, value in (("audience", audience_value), ("curator", curator_value)):
                    rows.append(
                        {
                            "story_id": story_id,
                            "comment_id": f"{story_id}-{index}",
                            "scope": "all",
                            "model_family": "xgboost",
                            "model_id": "winner",
                            "feature_set": "metadata",
                            "selector": selector,
                            "shap_feature_a": value,
                        }
                    )
        values = pd.DataFrame(rows)
        summary = _story_aggregate(values).iloc[0]
        self.assertAlmostEqual(summary["mean_shap_audience"], 3.0)
        self.assertAlmostEqual(summary["mean_shap_curator"], 6.0)
        self.assertAlmostEqual(summary["mean_abs_shap_audience"], 3.0)
        self.assertAlmostEqual(summary["mean_abs_shap_curator"], 6.0)
        self.assertAlmostEqual(summary["mean_shap_gap"], 3.0)
        self.assertAlmostEqual(summary["mean_abs_shap_gap"], 3.0)
        self.assertEqual(summary["n_stories"], 2)
        self.assertEqual(summary["sampled_comments"], 8)

        shuffled = _story_aggregate(values.sample(frac=1, random_state=23))
        pd.testing.assert_frame_equal(_story_aggregate(values), shuffled)

    def test_cache_reuse_and_model_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "model"
            model_root.mkdir()
            model_path = model_root / "development_model.json"
            model_path.write_bytes(b"model-v1")
            choice_set = root / "choice_set_all.parquet"
            pd.DataFrame({"story_id": ["s1"]}).to_parquet(choice_set, index=False)
            context = {
                "family": "xgboost",
                "scope": "all",
                "model_id": "winner",
                "feature_set": "metadata",
                "model_root": model_root,
                "manifest": {},
            }
            values = pd.DataFrame(
                {
                    "story_id": ["s1", "s1"],
                    "comment_id": ["c1", "c1"],
                    "scope": ["all", "all"],
                    "model_family": ["xgboost", "xgboost"],
                    "model_id": ["winner", "winner"],
                    "feature_set": ["metadata", "metadata"],
                    "selector": ["audience", "curator"],
                    "shap_feature_a": [1.0, 2.0],
                }
            )
            winners = pd.DataFrame(
                {
                    "scope": ["all"],
                    "variant_id": ["winner"],
                    "family": ["xgboost"],
                    "feature_set": ["metadata"],
                }
            )
            with (
                patch("commentgap_analysis.explanations._winner_context", return_value=context),
                patch("commentgap_analysis.explanations._calculate_xgb", return_value=values) as calculate,
            ):
                run_shap_explanations(
                    model_data_root=root,
                    factorial_root=root,
                    winners=winners,
                    output_root=root / "output",
                    test_rows=2,
                    background_rows=2,
                    seed=7,
                )
                self.assertEqual(calculate.call_count, 1)
                run_shap_explanations(
                    model_data_root=root,
                    factorial_root=root,
                    winners=winners,
                    output_root=root / "output",
                    test_rows=2,
                    background_rows=2,
                    seed=7,
                )
                self.assertEqual(calculate.call_count, 1)
                model_path.write_bytes(b"model-v2")
                run_shap_explanations(
                    model_data_root=root,
                    factorial_root=root,
                    winners=winners,
                    output_root=root / "output",
                    test_rows=2,
                    background_rows=2,
                    seed=7,
                )
                self.assertEqual(calculate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
