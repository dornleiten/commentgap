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
                regression_article_metrics = pd.DataFrame(
                    {
                        "story_id": ["s", "s", "t", "t"],
                        "selector": ["audience", "curator", "audience", "curator"],
                        "ndcg_at_k": [0.60, 0.50, 0.70, 0.60],
                        "top_k_overlap": [0.40, 0.30, 0.50, 0.40],
                        "jaccard": [0.30, 0.20, 0.40, 0.30],
                    }
                )
                regression_article_metrics.to_parquet(
                    regression_scope / "test_article_metrics.parquet", index=False
                )
                pd.DataFrame(
                    {
                        "selector": ["audience", "curator"] * 3,
                        "metric": [
                            "ndcg_at_k", "ndcg_at_k", "top_k_overlap",
                            "top_k_overlap", "jaccard", "jaccard"
                        ],
                        "estimate": [0.65, 0.55, 0.45, 0.35, 0.35, 0.25],
                        "conf_low": [0.60, 0.50, 0.40, 0.30, 0.30, 0.20],
                        "conf_high": [0.70, 0.60, 0.50, 0.40, 0.40, 0.30],
                        "articles": [2] * 6,
                        "bootstrap_draws": [1000] * 6,
                    }
                ).to_parquet(
                    regression_scope / "test_metric_summary.parquet", index=False
                )
                pd.DataFrame(
                    {
                        "selector": ["audience", "curator"] * 3,
                        "metric": [
                            "ndcg_at_k", "ndcg_at_k", "top_k_overlap",
                            "top_k_overlap", "jaccard", "jaccard"
                        ],
                        "estimate": [0.25] * 6,
                        "conf_low": [0.20] * 6,
                        "conf_high": [0.30] * 6,
                        "articles": [2] * 6,
                        "bootstrap_draws": [1000] * 6,
                    }
                ).to_parquet(
                    regression_scope / "test_chance_metric_summary.parquet",
                    index=False,
                )
                pd.DataFrame(
                    {
                        "selector": ["audience", "curator"] * 3,
                        "metric": [
                            "ndcg_at_k", "ndcg_at_k", "top_k_overlap",
                            "top_k_overlap", "jaccard", "jaccard"
                        ],
                        "model_estimate": [0.50] * 6,
                        "chance_estimate": [0.25] * 6,
                        "estimate_above_chance": [0.25] * 6,
                        "conf_low": [0.20] * 6,
                        "conf_high": [0.30] * 6,
                        "articles": [2] * 6,
                        "bootstrap_draws": [1000] * 6,
                    }
                ).to_parquet(
                    regression_scope / "test_above_chance_summary.parquet",
                    index=False,
                )
                pd.DataFrame(
                    {"statistic": ["iterations"], "value": [4]}
                ).to_csv(regression_scope / "model_diagnostics.csv", index=False)
                regression_article_metrics.assign(
                    audience_tie_draw=1, split_role="paper2_test"
                ).query("selector == 'audience'").to_parquet(
                    regression_scope / "test_tie_sensitivity_metrics.parquet",
                    index=False,
                )
                ranker_scope = rankers / scope
                ranker_scope.mkdir(parents=True)
                xgb_article_metrics = regression_article_metrics.copy()
                for metric in ("ndcg_at_k", "top_k_overlap", "jaccard"):
                    xgb_article_metrics[metric] += 0.10
                xgb_article_metrics.to_parquet(
                    ranker_scope / "test_article_metrics.parquet", index=False
                )
                estimates = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
                pd.DataFrame(
                    {
                        "selector": ["audience", "curator"] * 3,
                        "metric": ["ndcg_at_k", "ndcg_at_k", "top_k_overlap", "top_k_overlap", "jaccard", "jaccard"],
                        "estimate": estimates,
                        "conf_low": [value - 0.05 for value in estimates],
                        "conf_high": [value + 0.05 for value in estimates],
                    }
                ).to_parquet(ranker_scope / "test_metric_summary.parquet", index=False)
                pd.DataFrame(
                    {
                        "feature": ["feature"],
                        "repeat": [1],
                        "importance": [0.1],
                        "split_role": ["paper2_test"],
                    }
                ).to_parquet(
                    ranker_scope / "test_grouped_permutation_importance.parquet",
                    index=False,
                )
                pd.DataFrame(
                    {
                        "story_id": ["s", "s"],
                        "comment_id": ["a", "b"],
                        "selector": ["audience", "curator"],
                        "split_role": ["paper2_test", "paper2_test"],
                        "feature": [0.2, -0.3],
                        "selector_code": [0.0, 0.0],
                    }
                ).to_parquet(ranker_scope / "test_treeshap_sample.parquet", index=False)
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
                ).to_parquet(
                    ranker_scope / "test_tie_sensitivity_metrics.parquet", index=False
                )
                pd.DataFrame(
                    {
                        "story_id": ["s"],
                        "split_role": ["paper2_test"],
                        "development_fold": [-1],
                        "n_candidates": [2],
                    }
                ).to_parquet(ranker_scope / "article_split.parquet", index=False)
            pd.DataFrame(
                {
                    "covariate": ["log1p_n_candidates_root"],
                    "development_mean": [1.0],
                    "paper2_test_mean": [1.0],
                    "standardized_mean_difference": [0.0],
                    "abs_standardized_mean_difference": [0.0],
                    "acceptance_threshold": [0.05],
                    "accepted": [True],
                }
            ).to_csv(features / "split_balance_diagnostics.csv", index=False)
            report = build_reporting_outputs(features, regressions, rankers, output)
            self.assertEqual(report["watermark"], "PILOT_NOT_FOR_INFERENCE")
            self.assertTrue((output / "figures/regression_preference_differences.svg").exists())
            self.assertTrue((output / "figures/xgb_test_performance.pdf").exists())
            self.assertTrue(
                (output / "figures/regression_vs_xgb_test_performance.pdf").exists()
            )
            self.assertTrue((output / "tables/sample_accounting.csv").exists())
            self.assertTrue((output / "tables/held_out_covariate_balance.csv").exists())
            self.assertTrue(
                (output / "tables/regression_model_diagnostics.csv").exists()
            )
            self.assertTrue(
                (output / "tables/model_test_tie_sensitivity.csv").exists()
            )
            self.assertTrue(
                (output / "tables/regression_test_chance_baseline.csv").exists()
            )
            self.assertTrue(
                (output / "tables/regression_test_above_chance.csv").exists()
            )
            differences = pd.read_csv(
                output / "tables/regression_vs_xgb_test_differences.csv"
            )
            self.assertTrue(
                (differences["estimate_xgboost_minus_regression"].round(8) == 0.1).all()
            )


if __name__ == "__main__":
    unittest.main()
