import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from commentgap_analysis.comment_gap import (
    _summary,
    normalised_selection_rank_gap,
    run_comment_gap_analysis,
    weighted_median,
)
from commentgap_analysis.paper1_descriptives import run_descriptive_analysis


class CommentGapMetricTests(unittest.TestCase):
    def test_metric_endpoints_random_expectation_and_ties(self):
        scores = [10, 9, 8, 7]
        self.assertEqual(normalised_selection_rank_gap(scores, [1, 1, 0, 0]), 0)
        self.assertEqual(normalised_selection_rank_gap(scores, [0, 0, 1, 1]), 1)
        self.assertEqual(normalised_selection_rank_gap(scores, [1, 0, 0, 1]), 0.5)
        self.assertEqual(
            normalised_selection_rank_gap([1, 1, 1, 1], [1, 0, 1, 0]),
            0.5,
        )
        self.assertTrue(
            np.isnan(normalised_selection_rank_gap(scores, [1, 1, 1, 1]))
        )

    def test_metric_rejects_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            normalised_selection_rank_gap([1, 2], [True])

    def test_weighted_median_uses_first_half_weight_crossing(self):
        self.assertEqual(weighted_median([0, 1], [2, 8]), 1)
        self.assertEqual(weighted_median([0, 1], [1, 1]), 0)

    def test_summary_weights_gap_by_candidate_comments(self):
        frame = pd.DataFrame(
            {
                "scope": ["all", "all"],
                "analysis_partition": ["all_partitions", "all_partitions"],
                "story_id": ["small", "large"],
                "gap_score": [0.0, 1.0],
                "curator_audience_overlap_mean": [1.0, 0.0],
                "curator_audience_overlap_sd": [0.0, 0.0],
                "n_candidates": [2, 8],
                "n_picks": [1, 1],
                "has_any_vote_tie": [False, False],
            }
        )
        result = _summary(frame, ["scope", "analysis_partition"]).iloc[0]
        self.assertEqual(result["gap_mean"], 0.5)
        self.assertEqual(result["gap_comment_weighted_mean"], 0.8)
        self.assertEqual(result["gap_comment_weighted_median"], 1.0)
        self.assertEqual(result["candidate_comments"], 10)


class Paper1PipelineFixtureTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        data_root = root / "scrape_2025"
        article_root = data_root / "articles" / "year=2025" / "month=01"
        article_root.mkdir(parents=True)
        for story_id, section_1, section_2 in (
            ("s1", "/inland", "National"),
            ("s2", "", "Audit-only child"),
        ):
            pd.DataFrame(
                {
                    "story_id": [story_id],
                    "year": [2025],
                    "month": [1],
                    "section_1": [section_1],
                    "section_2": [section_2],
                    "section_3": [None],
                }
            ).to_parquet(article_root / f"{story_id}.parquet", index=False)

        model_root = root / "model_data"
        model_root.mkdir()
        split = pd.DataFrame(
            {
                "story_id": ["s1", "s2"],
                "split_role": ["development", "paper2_test"],
                "development_fold": [0, -1],
            }
        )
        split.to_parquet(model_root / "master_article_split.parquet", index=False)

        rows = []
        for story_id, curator in (
            ("s1", [True, True, False, False]),
            ("s2", [False, False, True, True]),
        ):
            for position, (score, selected) in enumerate(
                zip([4, 3, 2, 1], curator, strict=True)
            ):
                row = {
                    "story_id": story_id,
                    "comment_id": f"{story_id}-c{position}",
                    "article_month": 1,
                    "candidate_scope": "all",
                    "n_candidates": 4,
                    "n_picks": 2,
                    "curator_selected": selected,
                    "relative_votes": score,
                    "log_words": float(position),
                    "word_count": position + 1,
                    "is_reply": int(position > 0),
                }
                for draw in range(1, 11):
                    row[f"audience_selected_draw_{draw:02d}"] = position < 2
                rows.append(row)
        choice_all = pd.DataFrame(rows)
        choice_all.to_parquet(model_root / "choice_set_all.parquet", index=False)
        choice_root = choice_all.drop(columns="is_reply").copy()
        choice_root["candidate_scope"] = "root"
        choice_root.to_parquet(model_root / "choice_set_root.parquet", index=False)

        manifest = {
            "version": 4,
            "features": {
                "log_words": {"label": "Log words"},
                "is_reply": {"label": "Reply indicator"},
            },
            "models": {
                "all": {"features": ["log_words", "is_reply"]},
                "root": {"features": ["log_words"]},
            },
        }
        (model_root / "feature_manifest.json").write_text(json.dumps(manifest))
        provenance = {
            "source": {"articles": 2, "comments": 8, "dataset_fingerprint": "fixture"},
            "all": {"eligible_stories": 2, "candidate_rows": 8, "sticky_comments": 4},
            "root": {"eligible_stories": 2, "candidate_rows": 8, "sticky_comments": 4},
        }
        (model_root / "provenance_manifest.json").write_text(json.dumps(provenance))
        (model_root / "preprocessing_manifest.json").write_text(json.dumps({"outputs": {}}))
        return data_root, model_root

    def test_descriptives_and_gap_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, model_root = self._write_fixture(root)
            descriptives_root = root / "descriptives"
            descriptive_manifest = run_descriptive_analysis(
                model_data_root=model_root,
                data_root=data_root,
                output_root=descriptives_root,
                scopes=("all", "root"),
                threads=1,
                make_figures=True,
            )
            self.assertEqual(descriptive_manifest["primary_scope"], "all")
            topics = pd.read_parquet(descriptives_root / "article_topics.parquet")
            self.assertEqual(
                topics.set_index("story_id").loc["s2", "primary_topic"],
                "Unknown/other",
            )
            discussions = pd.read_parquet(
                descriptives_root / "discussion_descriptives.parquet"
            )
            self.assertEqual(len(discussions), 4)
            self.assertEqual(
                discussions.loc[discussions["scope"] == "all", "reply_share"].tolist(),
                [0.75, 0.75],
            )
            self.assertTrue(
                discussions.loc[discussions["scope"] == "root", "reply_share"].isna().all()
            )
            self.assertTrue(
                (descriptives_root / "figures" / "all_topic_composition.png").exists()
            )
            topic_summary = pd.read_csv(descriptives_root / "topic_summary.csv")
            self.assertEqual(
                set(topic_summary["primary_topic_label"]),
                {"domestic", "Unknown/other"},
            )

            gap_root = root / "comment_gap"
            gap_manifest = run_comment_gap_analysis(
                model_data_root=model_root,
                descriptives_root=descriptives_root,
                output_root=gap_root,
                scopes=("all", "root"),
                threads=1,
                make_figures=True,
            )
            self.assertEqual(gap_manifest["primary_scope"], "all")
            scores = pd.read_parquet(gap_root / "article_gap_scores.parquet")
            all_scores = scores[scores["scope"] == "all"].set_index("story_id")
            self.assertEqual(all_scores.loc["s1", "gap_score"], 0)
            self.assertEqual(all_scores.loc["s2", "gap_score"], 1)
            self.assertEqual(all_scores.loc["s1", "curator_audience_overlap_mean"], 1)
            self.assertEqual(all_scores.loc["s2", "curator_audience_overlap_mean"], 0)
            self.assertEqual(all_scores.loc["s2", "analysis_partition"], "held_out_test")
            self.assertTrue(
                (gap_root / "figures" / "all_comment_gap_distribution.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
