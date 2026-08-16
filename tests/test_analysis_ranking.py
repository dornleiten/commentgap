import unittest

import numpy as np
import pandas as pd

from commentgap_analysis.ranking import (
    assign_article_splits,
    build_stacked_rank_data,
    evaluate_rank_scores,
    make_ranker,
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
                    "feature": float(comment),
                }
            )
    return pd.DataFrame(rows)


class AnalysisRankingTests(unittest.TestCase):
    def test_stacking_has_two_valid_queries_per_article(self):
        choice = fixture_choice_set(3)
        stacked = build_stacked_rank_data(choice, ["feature"])
        self.assertEqual(stacked["query_id"].nunique(), 6)
        checks = stacked.groupby("query_id")["selected"].sum()
        self.assertTrue((checks == 1).all())

    def test_article_splits_do_not_overlap(self):
        choice = fixture_choice_set(40)
        splits = assign_article_splits(choice)
        self.assertFalse(splits["story_id"].duplicated().any())
        tuning = set(splits.loc[splits["is_tuning"], "story_id"])
        analysis = set(splits.loc[~splits["is_tuning"], "story_id"])
        self.assertFalse(tuning & analysis)
        self.assertEqual(set(splits.loc[~splits["is_tuning"], "outer_fold"]), set(range(5)))

    def test_article_level_ranking_metrics(self):
        choice = fixture_choice_set(2)
        stacked = build_stacked_rank_data(choice, ["feature"])
        scores = np.where(stacked["selected"].eq(1), 10.0, 0.0)
        metrics = evaluate_rank_scores(stacked, scores)
        np.testing.assert_allclose(metrics["top_k_overlap"], 1)
        np.testing.assert_allclose(metrics["jaccard"], 1)
        np.testing.assert_allclose(metrics["ndcg_at_k"], 1)

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
