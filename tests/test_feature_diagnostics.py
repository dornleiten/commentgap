from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

from commentgap_analysis.feature_diagnostics import (
    expected_range,
    feature_group,
    full_distribution_summary,
    invariant_checks,
    ks_statistic,
    quote_identifier,
    within_story_variation,
)


class FeatureDiagnosticsTests(unittest.TestCase):
    def test_pure_helpers_preserve_diagnostic_rules(self):
        self.assertEqual(quote_identifier('feature"name'), '"feature""name"')
        self.assertEqual(expected_range("aqua_quality_expected"), (0.0, 3.0))
        self.assertEqual(expected_range("article_similarity_top3"), (-1.0, 1.0))
        model_features = {"root": ["log_author_comments", "is_reply", "other"]}
        self.assertEqual(feature_group("log_author_comments", model_features), "author_history")
        self.assertEqual(feature_group("is_reply", model_features), "reply_structure")
        self.assertEqual(ks_statistic(np.array([0.0, 1.0]), np.array([1.0, 2.0])), 0.5)

    def test_sql_diagnostics_run_from_explicit_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "choice_set.parquet"
            pd.DataFrame(
                {
                    "story_id": ["story", "story", "other"],
                    "comment_id": ["a", "b", "c"],
                    "n_candidates": [2, 2, 1],
                    "n_picks": [1, 1, 0],
                    "aqua_quality_expected": [0.0, 3.0, 1.0],
                    "is_reply": [0, 1, 1],
                }
            ).to_parquet(path)
            connection = duckdb.connect()
            try:
                paths = {"root": path}
                features = {"root": ["aqua_quality_expected", "is_reply"]}
                registry = {
                    "features": {
                        "aqua_quality_expected": {"label": "Quality"},
                        "is_reply": {"label": "Reply"},
                    }
                }
                summary = full_distribution_summary(
                    "root",
                    connection=connection,
                    choice_paths=paths,
                    model_features=features,
                    registry=registry,
                )
                self.assertEqual(summary["rows"].tolist(), [3, 3])
                self.assertFalse(summary["range_violation"].any())

                invariants = invariant_checks(
                    "root",
                    connection=connection,
                    choice_paths=paths,
                    model_features=features,
                )
                self.assertEqual(invariants["failures"].tolist(), [0, 1, 0])

                variation = within_story_variation(
                    "root",
                    connection=connection,
                    choice_paths=paths,
                    model_features=features,
                )
                self.assertEqual(variation["stories"].tolist(), [2, 2])
                self.assertEqual(variation["stories_with_variation"].tolist(), [1, 1])
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
