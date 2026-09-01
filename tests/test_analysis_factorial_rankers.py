from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from commentgap_analysis.factorial_rankers import (
    FactorialVariant,
    _tune_xgb_stage,
    _xgb_arrays,
    factorial_variants,
    neural_recipe,
    select_variants,
)
from commentgap_analysis.neural_ranking import (
    MetadataOnlyRanker,
    _train_epoch,
    fit_feature_scaler,
)


class FactorialRankerTests(unittest.TestCase):
    def test_grid_has_four_xgboost_and_sixty_four_neural_variants(self):
        variants = factorial_variants()
        self.assertEqual(len(variants), 68)
        self.assertEqual(len({variant.variant_id for variant in variants}), 68)
        self.assertEqual(
            sum(variant.family == "xgboost" for variant in variants), 4
        )
        self.assertEqual(
            sum(variant.family == "neural" for variant in variants), 64
        )

    def test_glob_filters_are_composable(self):
        selected = select_variants(
            factorial_variants(),
            include=("nn__metadata__draw1__*",),
            exclude=("*__large", "*__plateau__*"),
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue(
            all(variant.network == "base" for variant in selected)
        )
        self.assertTrue(
            all(variant.schedule == "fixed" for variant in selected)
        )

    def test_neural_recipe_encodes_all_requested_axes(self):
        variant = FactorialVariant(
            variant_id="test",
            family="neural",
            feature_set="metadata_bge",
            draw_policy="draw1",
            negative_sampling="hard1_random3",
            schedule="plateau",
            heads="shared_residual",
            network="large",
        )
        recipe = neural_recipe(variant)
        self.assertEqual(recipe.approach, "frozen_bge")
        self.assertEqual(recipe.ranking_loss, "pairwise_logistic")
        self.assertEqual(recipe.audience_draw_policy, "draw1")
        self.assertEqual(recipe.hard_negatives_per_positive, 1)
        self.assertEqual(recipe.patience, 8)
        self.assertEqual(recipe.learning_rate_schedule, "plateau")
        self.assertEqual(recipe.head_type, "shared_residual")
        self.assertEqual(recipe.text_projection_dim, 512)
        self.assertEqual(recipe.metadata_projection_dim, 256)
        self.assertEqual(recipe.head_hidden_dim, 256)

    def test_xgboost_mean10_uses_integer_selection_counts(self):
        rows = []
        for story in ("a", "b"):
            for comment in range(3):
                row = {
                    "story_id": story,
                    "comment_id": f"{story}{comment}",
                    "feature": float(comment),
                    "curator_selected": int(comment == 0),
                }
                for draw in range(1, 11):
                    row[f"audience_selected_draw_{draw:02d}"] = int(
                        comment == draw % 3
                    )
                rows.append(row)
        frame = pd.DataFrame(rows)
        x, labels, qid, row_index, selector = _xgb_arrays(
            frame,
            features=["feature"],
            draw_policy="mean10",
            embedding_matrix=None,
        )
        self.assertEqual(x.shape, (12, 2))
        self.assertTrue(np.all(np.diff(qid) >= 0))
        self.assertEqual(len(np.unique(qid)), 4)
        audience_labels = labels[selector == 0]
        expected = (
            frame.sort_values(["story_id", "comment_id"])
            [[f"audience_selected_draw_{draw:02d}" for draw in range(1, 11)]]
            .sum(axis=1)
            .to_numpy()
        )
        np.testing.assert_array_equal(audience_labels, expected)
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))
        self.assertEqual(len(row_index), len(labels))

    def test_xgboost_tuning_stage_selects_and_reuses_all_fold_fits(self):
        rows = []
        for story in range(4):
            for comment in range(3):
                row = {
                    "story_id": f"s{story}",
                    "comment_id": f"s{story}c{comment}",
                    "development_fold": story % 2,
                    "n_picks": 1,
                    "feature": float(comment),
                    "curator_selected": int(comment == 0),
                }
                row.update(
                    {
                        f"audience_selected_draw_{draw:02d}": int(comment == 1)
                        for draw in range(1, 11)
                    }
                )
                rows.append(row)
        development = pd.DataFrame(rows)
        variant = FactorialVariant(
            variant_id="xgb-test",
            family="xgboost",
            feature_set="metadata",
            draw_policy="draw1",
        )
        configs = [
            {
                "n_estimators": 3,
                "max_depth": depth,
                "learning_rate": 0.2,
                "min_child_weight": 1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
            }
            for depth in (2, 3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.csv"
            best, history = _tune_xgb_stage(
                variant,
                development,
                features=["feature"],
                fold_columns={},
                embedding_matrix=None,
                configs=configs,
                stage="test_stage",
                history_path=history_path,
                device="cpu",
                seed=7,
                workflow_signature="workflow",
                progress_name="test",
                verbose_every=0,
                force_recompute=False,
                folds=2,
            )
            self.assertEqual(len(history), 4)
            self.assertEqual(history["fold"].nunique(), 2)
            self.assertEqual(history["config_index"].nunique(), 2)
            self.assertGreaterEqual(best["n_estimators"], 1)
            with patch(
                "commentgap_analysis.factorial_rankers._fit_xgb",
                side_effect=AssertionError("completed fits should be reused"),
            ):
                _, reused = _tune_xgb_stage(
                    variant,
                    development,
                    features=["feature"],
                    fold_columns={},
                    embedding_matrix=None,
                    configs=configs,
                    stage="test_stage",
                    history_path=history_path,
                    device="cpu",
                    seed=7,
                    workflow_signature="workflow",
                    progress_name="test",
                    verbose_every=0,
                    force_recompute=False,
                    folds=2,
                )
            self.assertEqual(len(reused), 4)

    def test_pairwise_hard_negative_recipe_completes_an_epoch(self):
        import torch

        variant = FactorialVariant(
            variant_id="test",
            family="neural",
            feature_set="metadata",
            draw_policy="draw1",
            negative_sampling="hard1_random3",
            schedule="fixed",
            heads="separate",
            network="base",
        )
        recipe = neural_recipe(variant)
        frame = pd.DataFrame(
            {
                "story_id": ["s"] * 5,
                "comment_id": [f"c{index}" for index in range(5)],
                "feature": np.arange(5, dtype=float),
                "curator_selected": [1, 0, 0, 0, 0],
                **{
                    f"audience_selected_draw_{draw:02d}": [0, 1, 0, 0, 0]
                    for draw in range(1, 11)
                },
            }
        )
        model = MetadataOnlyRanker(1, recipe)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.head_learning_rate)
        scaler = fit_feature_scaler(
            frame, ["feature"], {"feature": {"standardize": True}}
        )
        loss = _train_epoch(
            model,
            frame,
            ["s"],
            None,
            scaler,
            optimizer,
            recipe,
            epoch=1,
            device="cpu",
        )
        self.assertTrue(np.isfinite(loss))

    def test_shared_residual_metadata_head_scores_both_selectors(self):
        import torch

        variant = FactorialVariant(
            variant_id="test",
            family="neural",
            feature_set="metadata",
            draw_policy="draw1",
            negative_sampling="random4",
            schedule="fixed",
            heads="shared_residual",
            network="base",
        )
        model = MetadataOnlyRanker(3, neural_recipe(variant))
        scores = model.score_both(None, torch.zeros((4, 3)))
        self.assertEqual(tuple(scores.shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
