import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from commentgap_analysis.factorial_winners import freeze_development_cv_winners
from commentgap_analysis.paper1_plotting import _largest_regression_coefficient_rows
from commentgap_analysis.paper1_reporting import (
    _paired_model_differences,
    model_implied_comment_gap,
    run_paper1_reporting,
    summarize_model_implied_gaps,
    summarize_permutation_importance_gaps,
)


METRICS = ("ndcg_at_k", "top_k_overlap", "jaccard", "mean_selected_rank")


class Paper1ReportingTests(unittest.TestCase):
    def test_paired_differences_support_multiple_winners_per_family(self):
        models = (("conditional_logit", "reg"), ("xgboost", "xgb-a"), ("xgboost", "xgb-b"), ("neural", "nn-a"), ("neural", "nn-b"))
        frame = pd.DataFrame([
            {"story_id": story, "scope": "all", "selector": "audience", "model_family": family, "model_id": model, **{metric: index / 10 for metric in METRICS}}
            for index, (family, model) in enumerate(models) for story in ("s1", "s2")
        ])
        result = _paired_model_differences(frame, bootstrap_draws=100, seed=1)
        self.assertEqual((len(result), result["comparison"].nunique()), (32, 8))

    def _write_metric_artifacts(self, root: Path, offset: float) -> None:
        root.mkdir(parents=True)
        summary_rows = []
        article_rows = []
        tie_rows = []
        for selector_index, selector in enumerate(("audience", "curator")):
            for metric_index, metric in enumerate(METRICS):
                estimate = offset + selector_index * 0.01 + metric_index * 0.02
                summary_rows.append(
                    {
                        "selector": selector,
                        "metric": metric,
                        "estimate": estimate,
                        "conf_low": estimate - 0.01,
                        "conf_high": estimate + 0.01,
                        "bootstrap_draws": 100,
                    }
                )
            for story_index, story_id in enumerate(("s1", "s2", "s3")):
                values = {
                    "ndcg_at_k": offset + story_index * 0.01,
                    "top_k_overlap": offset + 0.1 + story_index * 0.01,
                    "jaccard": offset + 0.05 + story_index * 0.01,
                    "mean_selected_rank": 5.0 - offset + story_index,
                }
                article_rows.append(
                    {
                        "query_id": f"{story_id}::{selector}",
                        "story_id": story_id,
                        "selector": selector,
                        "n_candidates": 10,
                        "n_picks": 2,
                        **values,
                    }
                )
                for draw in range(1, 11):
                    tie_rows.append(
                        {
                            "query_id": f"{story_id}::{selector}",
                            "story_id": story_id,
                            "selector": selector,
                            "n_candidates": 10,
                            "n_picks": 2,
                            "audience_tie_draw": draw,
                            **{key: value + draw / 10000 for key, value in values.items()},
                        }
                    )
        pd.DataFrame(summary_rows).to_parquet(root / "test_metric_summary.parquet", index=False)
        pd.DataFrame(article_rows).to_parquet(root / "test_article_metrics.parquet", index=False)
        pd.DataFrame(tie_rows).to_parquet(root / "test_tie_sensitivity_metrics.parquet", index=False)
        score_rows = []
        for story_index, story_id in enumerate(("s1", "s2", "s3")):
            for comment_index in range(4):
                score_rows.append(
                    {
                        "story_id": story_id,
                        "comment_id": f"c{comment_index}",
                        "n_picks": 2,
                        "curator_selected": comment_index in (0, 2),
                        **{
                            f"audience_selected_draw_{draw:02d}": comment_index < 2
                            for draw in range(1, 11)
                        },
                        "audience_score": 4 - comment_index + offset,
                        "curator_score": [4, 1, 3, 2][comment_index] + offset,
                    }
                )
        wide_scores = pd.DataFrame(score_rows)
        wide_scores.to_parquet(root / "test_scores_wide.parquet", index=False)
        long_scores = []
        for selector in ("audience", "curator"):
            for row in score_rows:
                long_scores.append(
                    {
                        "story_id": row["story_id"],
                        "comment_id": row["comment_id"],
                        "n_picks": row["n_picks"],
                        "selector": selector,
                        "selected": int(
                            row["audience_selected_draw_01"]
                            if selector == "audience"
                            else row["curator_selected"]
                        ),
                        "score": row[f"{selector}_score"],
                    }
                )
        pd.DataFrame(long_scores).to_parquet(root / "test_scores_long.parquet", index=False)

    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        factorial_root = root / "factorial"
        factorial_root.mkdir()
        cv_rows = []
        plan_rows = []
        for family, prefix in (("xgboost", "xgb"), ("neural", "nn")):
            for variant_index, variant in enumerate((f"{prefix}-winner", f"{prefix}-loser")):
                plan_rows.append(
                    {
                        "variant_id": variant,
                        "family": family,
                        "feature_set": "metadata",
                        "draw_policy": "mean10",
                        "negative_sampling": "na" if family == "xgboost" else "random4",
                        "schedule": "na" if family == "xgboost" else "fixed",
                        "heads": "na" if family == "xgboost" else "separate",
                        "network": "na" if family == "xgboost" else "base",
                        "scope": "all",
                        "status": "complete",
                    }
                )
                for fold in range(5):
                    score = 0.7 - variant_index * 0.1 + fold / 1000
                    cv_rows.append(
                        {
                            "variant_id": variant,
                            "family": family,
                            "scope": "all",
                            "fold": fold,
                            "macro_ndcg_at_k": score,
                            "audience_ndcg_at_k": score - 0.01,
                            "curator_ndcg_at_k": score + 0.01,
                            "feature_set": "metadata",
                            "draw_policy": "mean10",
                            "negative_sampling": "na" if family == "xgboost" else "random4",
                            "schedule": "na" if family == "xgboost" else "fixed",
                            "heads": "na" if family == "xgboost" else "separate",
                            "network": "na" if family == "xgboost" else "base",
                        }
                    )
        pd.DataFrame(cv_rows).to_csv(factorial_root / "development_cv_results.csv", index=False)
        pd.DataFrame(plan_rows).to_csv(factorial_root / "experiment_variants.csv", index=False)
        winner_root = root / "winners"
        freeze_development_cv_winners(
            development_cv_path=factorial_root / "development_cv_results.csv",
            experiment_variants_path=factorial_root / "experiment_variants.csv",
            output_root=winner_root,
            scopes=("all",),
            require_idle=False,
        )

        for variant, offset, family in (
            ("xgb-winner", 0.3, "xgboost"),
            ("nn-winner", 0.4, "neural"),
        ):
            model_root = factorial_root / variant / "all"
            self._write_metric_artifacts(model_root, offset)
            manifest = {
                "features": ["feature_a", "feature_b"],
                "development_articles": 3,
                "paper2_test_articles": 3,
                "best_parameters": {"depth": 4} if family == "xgboost" else None,
                "recipe": {"network": "base"} if family == "neural" else None,
            }
            (model_root / "model_manifest.json").write_text(json.dumps(manifest))
            permutation_rows = []
            for feature_index, feature in enumerate(("feature_a", "feature_b")):
                for story_id in ("s1", "s2", "s3"):
                    for selector in ("audience", "curator"):
                        permutation_rows.append(
                            {
                                "story_id": story_id,
                                "selector": selector,
                                "baseline_ndcg_at_k": 0.8,
                                "permuted_ndcg_at_k": 0.8
                                - 0.02 * (feature_index + 1)
                                - (0.01 if selector == "curator" else 0),
                                "feature": feature,
                                "repeat": 1,
                                "scope": "all",
                                "model_family": family,
                                "model_id": variant,
                                "model_label": (
                                    "XGBoost winner" if family == "xgboost" else "Neural winner"
                                ),
                            }
                        )
            pd.DataFrame(permutation_rows).to_parquet(
                model_root / "test_selector_permutation_importance.parquet",
                index=False,
            )

        regression_root = root / "regression"
        self._write_metric_artifacts(regression_root / "all", 0.2)
        associations = pd.DataFrame(
            {
                "term": ["feature_a", "feature_b"],
                "feature": ["Feature A", "Feature B"],
                "curator_minus_audience_log_odds": [0.2, -0.1],
                "difference_conf_low": [0.1, -0.2],
                "difference_conf_high": [0.3, 0.0],
            }
        )
        associations.to_csv(regression_root / "all" / "selector_associations.csv", index=False)
        pd.DataFrame({"statistic": ["converged"], "value": [True]}).to_csv(
            regression_root / "all" / "model_diagnostics.csv", index=False
        )
        return factorial_root, winner_root, regression_root

    def test_reporting_uses_only_frozen_winners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factorial_root, winner_root, regression_root = self._fixture(root)
            output_root = root / "reporting"
            shap_result = {
                "values": pd.DataFrame(
                    {
                        "story_id": ["s1"],
                        "comment_id": ["c1"],
                        "scope": ["all"],
                        "model_family": ["xgboost"],
                        "model_id": ["xgb-winner"],
                        "feature_set": ["metadata"],
                        "selector": ["audience"],
                        "shap_feature_a": [0.0],
                    }
                ),
                "summary": pd.DataFrame(columns=["scope", "feature"]),
                "manifest": {"workflow_version": 2, "cached": False},
            }
            with patch(
                "commentgap_analysis.explanations.run_shap_explanations",
                return_value=shap_result,
            ):
                manifest = run_paper1_reporting(
                    factorial_root=factorial_root,
                    winner_root=winner_root,
                    regression_root=regression_root,
                    output_root=output_root,
                    scopes=("all",),
                    bootstrap_draws=100,
                    make_figures=True,
                    require_factorial_idle=False,
                )
            winners = pd.read_csv(output_root / "tables" / "development_cv_winners.csv")
            self.assertEqual(set(winners["variant_id"]), {"xgb-winner", "nn-winner"})
            performance = pd.read_csv(output_root / "tables" / "held_out_model_performance.csv")
            self.assertEqual(set(performance["model_family"]), {"conditional_logit", "xgboost", "neural"})
            self.assertEqual(
                set(performance["model_label"]),
                {"Reg: Audience", "Reg: Editor", "XGB: Audience", "XGB: Editor", "NN: Audience", "NN: Editor"},
            )
            paired = pd.read_csv(output_root / "tables" / "held_out_paired_model_differences.csv")
            self.assertEqual(len(paired), 24)
            self.assertTrue((output_root / "figures" / "all_winner_held_out_ndcg.png").exists())
            self.assertTrue((output_root / "figures" / "all_model_implied_comment_gaps.png").exists())
            self.assertTrue((output_root / "tables" / "regression_feature_gaps.csv").exists())
            self.assertTrue((output_root / "tables" / "held_out_permutation_importance_gaps.csv").exists())
            self.assertIn("table:held_out_ndcg.tex", manifest["outputs"])

    def test_largest_coefficients_use_all_association_rows(self):
        associations = pd.DataFrame(
            {
                "feature": ["large_shared", "large_difference", "small"],
                "audience_log_odds": [5.0, 0.1, 0.2],
                "curator_log_odds": [5.1, 4.0, 0.3],
                "audience_conf_low": [4.9, 0.0, 0.1],
                "audience_conf_high": [5.1, 0.2, 0.3],
                "curator_conf_low": [5.0, 3.9, 0.2],
                "curator_conf_high": [5.2, 4.1, 0.4],
                "curator_minus_audience_log_odds": [0.1, 3.9, 0.1],
            }
        )
        selected = _largest_regression_coefficient_rows(associations, limit=1)
        self.assertEqual(selected["feature"].tolist(), ["large_shared"])

    def test_model_implied_gap_uses_midranks_and_fractional_cutoff_ties(self):
        scores = pd.DataFrame(
            {
                "story_id": ["s"] * 4,
                "comment_id": ["a", "b", "c", "d"],
                "n_picks": [2] * 4,
                "audience_score": [4.0, 3.0, 2.0, 1.0],
                "curator_score": [2.0, 1.0, 1.0, 1.0],
            }
        )
        result = model_implied_comment_gap(scores).iloc[0]
        self.assertAlmostEqual(result["mean_curator_model_audience_rank"], 2.0)
        self.assertAlmostEqual(result["model_implied_gap"], 0.25)
        self.assertEqual(result["curator_cutoff_tie_size"], 3)

    def test_model_gap_summary_weights_by_candidate_comments(self):
        article = pd.DataFrame(
            {
                "story_id": ["small", "large"],
                "scope": ["all", "all"],
                "model_family": ["xgboost", "xgboost"],
                "model_id": ["xgb", "xgb"],
                "model_label": ["XGBoost winner", "XGBoost winner"],
                "n_candidates": [2, 8],
                "model_implied_gap": [0.0, 1.0],
            }
        )
        result = summarize_model_implied_gaps(article, draws=100, seed=1).iloc[0]
        self.assertEqual(result["gap_mean"], 0.5)
        self.assertEqual(result["gap_comment_weighted_mean"], 0.8)
        self.assertEqual(result["gap_comment_weighted_median"], 1.0)
        self.assertEqual(result["candidate_comments"], 10)
        self.assertLessEqual(result["weighted_conf_low"], 0.8)
        self.assertGreaterEqual(result["weighted_conf_high"], 0.8)
        self.assertLessEqual(result["weighted_median_conf_low"], 1.0)
        self.assertGreaterEqual(result["weighted_median_conf_high"], 1.0)

    def test_permutation_gap_is_curator_minus_audience_loss(self):
        rows = []
        for story_id in ("s1", "s2"):
            for selector, permuted in (("audience", 0.7), ("curator", 0.6)):
                rows.append(
                    {
                        "story_id": story_id,
                        "scope": "all",
                        "model_family": "xgboost",
                        "model_id": "xgb",
                        "model_label": "XGBoost winner",
                        "feature": "f",
                        "repeat": 1,
                        "selector": selector,
                        "baseline_ndcg_at_k": 0.8,
                        "permuted_ndcg_at_k": permuted,
                    }
                )
        result = summarize_permutation_importance_gaps(
            pd.DataFrame(rows), draws=100, seed=1
        ).iloc[0]
        self.assertAlmostEqual(result["audience_importance"], 0.1)
        self.assertAlmostEqual(result["curator_importance"], 0.2)
        self.assertAlmostEqual(result["permutation_importance_gap"], 0.1)

    def test_reporting_rejects_changed_cv_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factorial_root, winner_root, regression_root = self._fixture(root)
            cv_path = factorial_root / "development_cv_results.csv"
            changed = pd.read_csv(cv_path)
            changed.loc[0, "macro_ndcg_at_k"] += 0.01
            changed.to_csv(cv_path, index=False)
            with self.assertRaisesRegex(RuntimeError, "changed after winner freezing"):
                run_paper1_reporting(
                    factorial_root=factorial_root,
                    winner_root=winner_root,
                    regression_root=regression_root,
                    output_root=root / "reporting",
                    scopes=("all",),
                    bootstrap_draws=100,
                    make_figures=False,
                    require_factorial_idle=False,
                )


if __name__ == "__main__":
    unittest.main()
