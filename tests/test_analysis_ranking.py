from pathlib import Path
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from commentgap_analysis.ranking import (
    article_split_balance,
    assign_article_splits,
    build_stacked_rank_data,
    default_search_configs,
    evaluate_rank_scores,
    make_ranker,
    refinement_search_configs,
    run_ranker_workflow,
    select_best_configuration,
    tune_ranker,
)


def fixture_choice_set(stories=20):
    rows = []
    for story in range(stories):
        for comment in range(4):
            rows.append(
                {
                    "story_id": f"s{story:02d}",
                    "comment_id": f"s{story:02d}c{comment}",
                    "article_month": story % 12 + 1,
                    "n_candidates": 4,
                    "n_picks": 1,
                    "curator_selected": comment == 0,
                    "audience_selected_draw_01": comment == 1,
                    "is_reply": comment >= 2,
                    "feature": float(comment),
                }
            )
    return pd.DataFrame(rows)


def fixture_scope_choice_sets(stories=40):
    root = fixture_choice_set(stories)
    all_rows = []
    for story in range(stories):
        candidates = 6 + story % 3
        for comment in range(candidates):
            all_rows.append(
                {
                    "story_id": f"s{story:02d}",
                    "comment_id": f"s{story:02d}a{comment}",
                    "article_month": story % 12 + 1,
                    "n_candidates": candidates,
                    "n_picks": 1 + story % 2,
                    "curator_selected": comment < 1 + story % 2,
                    "audience_selected_draw_01": comment < 1 + story % 2,
                    "is_reply": comment >= 4,
                    "feature": float(comment),
                }
            )
    return root, pd.DataFrame(all_rows)


class AnalysisRankingTests(unittest.TestCase):
    def test_stacking_has_two_valid_queries_per_article(self):
        choice = fixture_choice_set(3)
        stacked = build_stacked_rank_data(choice, ["feature"])
        self.assertEqual(stacked["query_id"].nunique(), 6)
        checks = stacked.groupby("query_id")["selected"].sum()
        self.assertTrue((checks == 1).all())

    def test_stacking_preserves_auxiliary_fold_features_without_modelling_them(self):
        choice = fixture_choice_set(3)
        choice["feature_fold_00"] = choice["feature"] - 10.0
        stacked = build_stacked_rank_data(
            choice,
            ["feature"],
            auxiliary_columns=["feature_fold_00"],
        )
        self.assertIn("feature_fold_00", stacked.columns)
        expected = choice.set_index("comment_id")["feature_fold_00"]
        for row in stacked.itertuples():
            self.assertEqual(row.feature_fold_00, expected.loc[row.comment_id])


    def test_shared_article_split_is_exact_balanced_and_deterministic(self):
        root, all_comments = fixture_scope_choice_sets(72)
        splits, balance = assign_article_splits(root, all_comments)
        repeated, repeated_balance = assign_article_splits(root, all_comments)

        self.assertFalse(splits["story_id"].duplicated().any())
        self.assertEqual(splits["split_role"].value_counts().to_dict(), {
            "development": 36,
            "paper2_test": 36,
        })
        development = set(splits.loc[splits["split_role"] == "development", "story_id"])
        test = set(splits.loc[splits["split_role"] == "paper2_test", "story_id"])
        self.assertFalse(development & test)
        self.assertEqual(
            set(splits.loc[splits["split_role"] == "development", "development_fold"]),
            set(range(5)),
        )
        self.assertTrue(
            (
                pd.crosstab(splits["split_stratum"], splits["split_role"])
                .diff(axis=1)
                .iloc[:, -1]
                .abs()
                <= 1
            ).all()
        )
        self.assertTrue(balance["accepted"].all())
        self.assertTrue((balance["abs_standardized_mean_difference"] <= 0.05).all())
        pd.testing.assert_frame_equal(splits, repeated)
        pd.testing.assert_frame_equal(balance, repeated_balance)

    def test_shared_article_split_uses_only_common_stories(self):
        root, all_comments = fixture_scope_choice_sets(72)
        extra = all_comments[all_comments["story_id"] == "s00"].copy()
        extra["story_id"] = "all-only"
        extra["comment_id"] = "all-only-" + extra["comment_id"]
        all_comments = pd.concat([all_comments, extra], ignore_index=True)

        splits, _ = assign_article_splits(root, all_comments)

        self.assertEqual(set(splits["story_id"]), set(root["story_id"]))
        self.assertNotIn("all-only", set(splits["story_id"]))
        diagnostics = article_split_balance(splits)
        self.assertEqual(
            set(diagnostics["covariate"]),
            {
                "log1p_n_candidates_root",
                "log1p_n_candidates_all",
                "n_picks_root",
                "n_picks_all",
                "reply_proportion",
            },
        )

    def test_article_level_ranking_metrics(self):
        choice = fixture_choice_set(2)
        stacked = build_stacked_rank_data(choice, ["feature"])
        scores = np.where(stacked["selected"].eq(1), 10.0, 0.0)
        metrics = evaluate_rank_scores(stacked, scores)
        np.testing.assert_allclose(metrics["top_k_overlap"], 1)
        np.testing.assert_allclose(metrics["jaccard"], 1)
        np.testing.assert_allclose(metrics["ndcg_at_k"], 1)

    def test_sealed_test_workflow_trains_only_on_development_articles(self):
        root, all_comments = fixture_scope_choice_sets(24)
        splits, _ = assign_article_splits(
            root,
            all_comments,
            development_folds=2,
            max_abs_smd=1.0,
        )
        parameters = {
            "max_depth": 2,
            "learning_rate": 0.2,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1,
            "n_estimators": 20,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = run_ranker_workflow(
                root,
                ["feature"],
                output,
                scope="root",
                article_split=splits,
                device="cpu",
                bootstrap_draws=10,
                development_folds=2,
                configs=[parameters],
                permutation_repeats=1,
            )
            scores = pd.read_parquet(output / "root/test_scores_long.parquet")
            expected_test = set(
                splits.loc[splits["split_role"] == "paper2_test", "story_id"]
            )

            self.assertEqual(manifest["reported_scores"], "sealed_paper2_test")
            self.assertEqual(set(scores["story_id"]), expected_test)
            self.assertTrue((scores["score_source"] == "sealed_paper2_test").all())
            self.assertTrue((output / "root/development_model.json").exists())
            self.assertFalse((output / "root/final_deployable_model.json").exists())
            self.assertTrue((output / "root/workflow_cache.json").exists())
            self.assertTrue(
                all(
                    status == "computed"
                    for status in manifest["cache"]["stage_status"].values()
                )
            )

            model_mtime = (output / "root/development_model.json").stat().st_mtime_ns
            scores_mtime = (output / "root/test_scores_long.parquet").stat().st_mtime_ns
            resumed = run_ranker_workflow(
                root,
                ["feature"],
                output,
                scope="root",
                article_split=splits,
                device="cpu",
                bootstrap_draws=10,
                development_folds=2,
                configs=[parameters],
                permutation_repeats=1,
            )
            self.assertTrue(
                all(
                    status == "reused"
                    for status in resumed["cache"]["stage_status"].values()
                )
            )
            self.assertEqual(
                (output / "root/development_model.json").stat().st_mtime_ns,
                model_mtime,
            )
            self.assertEqual(
                (output / "root/test_scores_long.parquet").stat().st_mtime_ns,
                scores_mtime,
            )

            changed_bootstrap = run_ranker_workflow(
                root,
                ["feature"],
                output,
                scope="root",
                article_split=splits,
                device="cpu",
                bootstrap_draws=11,
                development_folds=2,
                configs=[parameters],
                permutation_repeats=1,
            )
            changed_status = changed_bootstrap["cache"]["stage_status"]
            self.assertEqual(changed_status["test_metrics"], "computed")
            self.assertTrue(
                all(
                    changed_status[stage] == "reused"
                    for stage in changed_status
                    if stage != "test_metrics"
                )
            )

            forced = run_ranker_workflow(
                root,
                ["feature"],
                output,
                scope="root",
                article_split=splits,
                device="cpu",
                bootstrap_draws=11,
                development_folds=2,
                configs=[parameters],
                permutation_repeats=1,
                force_recompute=True,
            )
            self.assertTrue(
                all(
                    status == "computed"
                    for status in forced["cache"]["stage_status"].values()
                )
            )

    def test_tuning_uses_fold_training_feature_columns(self):
        rows = []
        for fold in range(2):
            for story_position in range(2):
                story = f"f{fold}s{story_position}"
                for comment in range(2):
                    rows.append(
                        {
                            "story_id": story,
                            "comment_id": f"{story}c{comment}",
                            "query_id": f"{story}::audience",
                            "selector": "audience",
                            "selector_code": 0,
                            "selected": int(comment == 0),
                            "n_picks": 1,
                            "development_fold": fold,
                            "feature": 999.0,
                            "feature_fold_00": 10.0,
                            "feature_fold_01": 20.0,
                        }
)
        development = pd.DataFrame(rows)
        observed = []

        def fake_fit(train, validation, features, parameters, **kwargs):
            observed.append(
                (set(train["feature"]), set(validation["feature"]))
)
            return SimpleNamespace(best_iteration=0)

        with patch("commentgap_analysis.ranking._fit_ranker", side_effect=fake_fit), patch(
            "commentgap_analysis.ranking._predict_cpu",
            side_effect=lambda model, frame: np.arange(len(frame), dtype=float),
):
            tune_ranker(
                development,
                ["feature"],
                configs=[{}],
                folds=2,
                device="cpu",
                fold_feature_columns={
                    0: {"feature": "feature_fold_00"},
                    1: {"feature": "feature_fold_01"},
                },
)
        self.assertEqual(observed, [({10.0}, {10.0}), ({20.0}, {20.0})])

    def test_two_stage_search_spaces_are_reproducible_and_bounded(self):
        broad = default_search_configs(count=32, seed=123)
        repeated = default_search_configs(count=32, seed=123)
        self.assertEqual(broad, repeated)
        self.assertEqual(len(broad), 32)
        self.assertEqual(len({json.dumps(config, sort_keys=True) for config in broad}), 32)
        for config in broad:
            self.assertGreaterEqual(config["max_depth"], 3)
            self.assertLessEqual(config["max_depth"], 9)
            self.assertGreaterEqual(config["learning_rate"], 0.015)
            self.assertLessEqual(config["learning_rate"], 0.12)
            self.assertGreaterEqual(config["min_child_weight"], 3)
            self.assertLessEqual(config["min_child_weight"], 50)
            self.assertGreaterEqual(config["subsample"], 0.65)
            self.assertLessEqual(config["subsample"], 1.0)
            self.assertGreaterEqual(config["colsample_bytree"], 0.55)
            self.assertLessEqual(config["colsample_bytree"], 1.0)

        history_rows = []
        for config_index, config in enumerate(broad[:6]):
            for fold in range(5):
                history_rows.append(
                    {
                        "search_stage": "broad_random",
                        "config_index": config_index,
                        "fold": fold,
                        "macro_ndcg_at_k": 0.70 + config_index / 100 + fold / 10_000,
                        "best_iteration": 100 + config_index + fold,
                        "parameters": json.dumps(config, sort_keys=True),
                    }
                )
        history = pd.DataFrame(history_rows)
        narrow = refinement_search_configs(history, top_configs=5)
        self.assertEqual(len(narrow), 27)
        self.assertEqual(
            len({json.dumps(config, sort_keys=True) for config in narrow}),
            27,
        )
        self.assertEqual(narrow, refinement_search_configs(history, top_configs=5))

        best, summary = select_best_configuration(history)
        self.assertEqual(best["max_depth"], broad[5]["max_depth"])
        self.assertEqual(best["n_estimators"], 108)
        self.assertEqual(len(summary), 6)

    def test_cpu_xgboost_ranker_smoke(self):
        choice = fixture_choice_set(8)
        stacked = build_stacked_rank_data(choice, ["feature"])
        model = make_ranker(
            {
                "max_depth": 2,
                "learning_rate": 0.2,
                "min_child_weight": 1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1,
                "n_estimators": 20,
            },
            device="cpu",
            early_stopping_rounds=None,
        )
        qid = pd.factorize(stacked["query_id"], sort=True)[0]
        model.fit(
            stacked[["feature", "selector_code"]],
            stacked["selected"],
            qid=qid,
            verbose=False,
        )
        scores = model.predict(stacked[["feature", "selector_code"]])
        self.assertEqual(len(scores), len(stacked))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
