import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from commentgap_analysis.neural_reporting import build_neural_reporting_outputs


class NeuralReportingTests(unittest.TestCase):
    def test_neural_reporting_combines_models_and_paired_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xgb = root / "xgb"
            neural = root / "neural"
            report = root / "report"
            (report / "tables").mkdir(parents=True)
            base_summary = pd.DataFrame(
                {
                    "selector": ["audience"],
                    "metric": ["ndcg_at_k"],
                    "estimate": [0.5],
                    "conf_low": [0.4],
                    "conf_high": [0.6],
                    "scope": ["root"],
                    "model": ["XGBoost"],
                }
            )
            base_summary.to_csv(report / "tables/model_test_performance.csv", index=False)
            pd.DataFrame(
                {
                    "selector": ["audience"],
                    "audience_tie_draw": [1],
                    "ndcg_at_k": [0.5],
                    "top_k_overlap": [0.5],
                    "jaccard": [0.5],
                    "scope": ["root"],
                    "model": ["XGBoost"],
                }
            ).to_csv(report / "tables/model_test_tie_sensitivity.csv", index=False)
            for scope in ("root", "all"):
                xgb_scope = xgb / scope
                neural_scope = neural / scope
                xgb_scope.mkdir(parents=True)
                neural_scope.mkdir(parents=True)
                article = pd.DataFrame(
                    {
                        "story_id": ["s", "s"],
                        "selector": ["audience", "curator"],
                        "ndcg_at_k": [0.5, 0.5],
                        "top_k_overlap": [0.4, 0.4],
                        "jaccard": [0.3, 0.3],
                    }
                )
                article.to_parquet(xgb_scope / "test_article_metrics.parquet", index=False)
                improved = article.copy()
                for metric in ("ndcg_at_k", "top_k_overlap", "jaccard"):
                    improved[metric] += 0.1
                improved.to_parquet(
                    neural_scope / "test_article_metrics.parquet", index=False
                )
                pd.DataFrame(
                    {
                        "selector": ["audience", "curator"] * 3,
                        "metric": ["ndcg_at_k", "ndcg_at_k", "top_k_overlap", "top_k_overlap", "jaccard", "jaccard"],
                        "estimate": [0.6] * 6,
                        "conf_low": [0.5] * 6,
                        "conf_high": [0.7] * 6,
                    }
                ).to_parquet(neural_scope / "test_metric_summary.parquet", index=False)
                improved.assign(audience_tie_draw=1).to_parquet(
                    neural_scope / "test_tie_sensitivity_metrics.parquet", index=False
                )
                (neural_scope / "model_manifest.json").write_text(
                    json.dumps({"reported_scores": "sealed_paper2_test"})
                )
            result = build_neural_reporting_outputs(
                xgb, {"Frozen BGE-M3": neural}, report, bootstrap_draws=10
            )
            self.assertIn("Frozen BGE-M3", result["models"])
            differences = pd.read_csv(
                report / "tables/xgb_vs_neural_test_differences.csv"
            )
            self.assertTrue(
                (differences["estimate_candidate_minus_reference"].round(8) == 0.1).all()
            )
            self.assertTrue((report / "figures/all_model_test_performance.pdf").exists())


if __name__ == "__main__":
    unittest.main()
