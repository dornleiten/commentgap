import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from commentgap_analysis.reporting import build_reporting_outputs


class AnalysisReportingTests(unittest.TestCase):
    def test_reporting_builds_vector_figures_and_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = root / "features"
            regressions = root / "regression"
            rankers = root / "xgboost"
            output = root / "reporting"
            features.mkdir()
            manifest = {
                "models": {"root": {"features": ["feature"]}, "all": {"features": ["feature"]}},
                "features": {"feature": {"label": "Feature", "standardize": True}},
            }
            (features / "feature_manifest.json").write_text(json.dumps(manifest))
            (features / "provenance_manifest.json").write_text(
                json.dumps({"watermark": "PILOT_NOT_FOR_INFERENCE"})
            )
            for scope in ("root", "all"):
                choice = pd.DataFrame(
                    {
                        "story_id": ["s", "s"],
                        "comment_id": ["a", "b"],
                        "n_candidates": [2, 2],
                        "n_picks": [1, 1],
                        "curator_selected": [True, False],
                        "feature": [0.0, 1.0],
                    }
                )
                choice.to_parquet(features / f"choice_set_{scope}.parquet", index=False)
                regression_scope = regressions / scope
                regression_scope.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "term": ["feature"],
                        "feature": ["Feature"],
                        "audience_log_odds": [0.1],
                        "audience_conf_low": [-0.1],
                        "audience_conf_high": [0.3],
                        "curator_log_odds": [0.4],
                        "curator_conf_low": [0.1],
                        "curator_conf_high": [0.7],
                        "curator_minus_audience_log_odds": [0.3],
                        "difference_conf_low": [0.0],
                        "difference_conf_high": [0.6],
                    }
                ).to_csv(regression_scope / "selector_associations.csv", index=False)
                pd.DataFrame({"term": ["feature"], "curator_minus_audience_pp": [1.0]}).to_csv(
                    regression_scope / "probability_contrasts.csv", index=False
                )
                ranker_scope = rankers / scope
                ranker_scope.mkdir(parents=True)
                estimates = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
                pd.DataFrame(
                    {
                        "selector": ["audience", "curator"] * 3,
                        "metric": ["ndcg_at_k", "ndcg_at_k", "top_k_overlap", "top_k_overlap", "jaccard", "jaccard"],
                        "estimate": estimates,
                        "conf_low": [value - 0.05 for value in estimates],
                        "conf_high": [value + 0.05 for value in estimates],
                    }
                ).to_parquet(ranker_scope / "metric_summary.parquet", index=False)
                pd.DataFrame(
                    {"feature": ["feature"], "repeat": [1], "importance": [0.1], "outer_fold": [0]}
                ).to_parquet(ranker_scope / "grouped_permutation_importance.parquet", index=False)
                pd.DataFrame(
                    {
                        "story_id": ["s", "s"],
                        "comment_id": ["a", "b"],
                        "selector": ["audience", "curator"],
                        "outer_fold": [0, 0],
                        "feature": [0.2, -0.3],
                        "selector_code": [0.0, 0.0],
                    }
                ).to_parquet(ranker_scope / "oof_treeshap_sample.parquet", index=False)
                pd.DataFrame(
                    {
                        "scope": [scope],
                        "story_id": ["s"],
                        "selector": ["audience"],
                        "audience_tie_draw": [1],
                        "ndcg_at_k": [0.8],
                        "top_k_overlap": [0.6],
                        "jaccard": [0.4],
                    }
                ).to_parquet(ranker_scope / "tie_sensitivity_metrics.parquet", index=False)
                pd.DataFrame(
                    {
                        "story_id": ["s"],
                        "is_tuning": [False],
                        "outer_fold": [0],
                        "n_candidates": [2],
                    }
                ).to_parquet(ranker_scope / "article_splits.parquet", index=False)
            report = build_reporting_outputs(features, regressions, rankers, output)
            self.assertEqual(report["watermark"], "PILOT_NOT_FOR_INFERENCE")
            self.assertTrue((output / "figures/regression_preference_differences.svg").exists())
            self.assertTrue((output / "figures/xgb_oof_performance.pdf").exists())
            self.assertTrue((output / "tables/sample_accounting.csv").exists())


if __name__ == "__main__":
    unittest.main()
