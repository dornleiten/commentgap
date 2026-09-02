import tempfile
import unittest
from pathlib import Path

import pandas as pd

from commentgap_analysis.factorial_winners import (
    freeze_development_cv_winners,
    rank_development_cv_variants,
)


class FactorialWinnerTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        for family in ("xgboost", "neural"):
            for variant, values in (
                ("stable", [0.6, 0.6, 0.6, 0.6, 0.6]),
                ("variable", [0.8, 0.4, 0.8, 0.4, 0.6]),
                ("lower", [0.5, 0.5, 0.5, 0.5, 0.5]),
            ):
                for fold, value in enumerate(values):
                    rows.append(
                        {
                            "variant_id": f"{family}-{variant}",
                            "family": family,
                            "scope": "all",
                            "fold": fold,
                            "macro_ndcg_at_k": value,
                            "audience_ndcg_at_k": value + 0.01,
                            "curator_ndcg_at_k": value - 0.01,
                        }
                    )
            rows.append(
                {
                    "variant_id": f"{family}-incomplete",
                    "family": family,
                    "scope": "all",
                    "fold": 0,
                    "macro_ndcg_at_k": 0.99,
                }
            )
        return pd.DataFrame(rows)

    def test_complete_folds_and_variance_tie_break(self):
        ranking, winners, excluded = rank_development_cv_variants(self._frame())
        self.assertEqual(set(winners["variant_id"]), {"xgboost-stable", "neural-stable"})
        self.assertEqual(set(excluded["variant_id"]), {"xgboost-incomplete", "neural-incomplete"})
        self.assertTrue((ranking["completed_folds"] == 5).all())


    def test_winners_are_selected_within_feature_set(self):
        frame = self._frame()
        frame["feature_set"] = frame["variant_id"].map(
            lambda value: "metadata" if str(value).endswith("-stable") else "metadata_bge"
        )
        _, winners, _ = rank_development_cv_variants(frame)
        self.assertEqual(len(winners), 4)
        self.assertEqual(
            set(winners[["family", "feature_set"]].itertuples(index=False, name=None)),
            {
                ("xgboost", "metadata"),
                ("xgboost", "metadata_bge"),
                ("neural", "metadata"),
                ("neural", "metadata_bge"),
            },
        )

    def test_draw_policy_filter_excludes_other_policies(self):
        frame = self._frame()
        frame["draw_policy"] = frame["variant_id"].map(
            lambda value: "draw1" if str(value).endswith("-stable") else "mean10"
        )
        ranking, winners, _ = rank_development_cv_variants(
            frame, draw_policies=("draw1",)
        )
        self.assertTrue(ranking["draw_policy"].eq("draw1").all())
        self.assertEqual(set(winners["variant_id"]), {"xgboost-stable", "neural-stable"})
    def test_manifest_records_development_only_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "development_cv_results.csv"
            complete = self._frame()[
                ~self._frame()["variant_id"].str.endswith("-incomplete")
            ]
            complete.to_csv(source, index=False)
            plan = complete[["variant_id", "family", "scope"]].drop_duplicates()
            plan["status"] = "complete"
            plan.to_csv(root / "experiment_variants.csv", index=False)
            manifest = freeze_development_cv_winners(
                development_cv_path=source,
                output_root=root / "winners",
                require_idle=False,
            )
            self.assertFalse(manifest["held_out_artifacts_read"])
            self.assertEqual(len(manifest["winners"]), 2)
            self.assertTrue((root / "winners" / "factorial_winner_manifest.json").exists())

    def test_incomplete_experiment_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "development_cv_results.csv"
            complete = self._frame()[
                ~self._frame()["variant_id"].str.endswith("-incomplete")
            ]
            complete.to_csv(source, index=False)
            plan = complete[["variant_id", "family", "scope"]].drop_duplicates()
            plan["status"] = "complete"
            plan.loc[plan.index[0], "status"] = "running"
            plan.to_csv(root / "experiment_variants.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                freeze_development_cv_winners(
                    development_cv_path=source,
                    output_root=root / "winners",
                    require_idle=False,
                )

    def test_duplicate_fold_is_rejected(self):
        frame = self._frame()
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate variant/fold"):
            rank_development_cv_variants(duplicate)


if __name__ == "__main__":
    unittest.main()
