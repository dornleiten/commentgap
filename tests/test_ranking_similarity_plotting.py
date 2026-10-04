from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from commentgap_analysis.ranking_similarity_plotting import plot_umap_clusters


class RankingSimilarityPlottingTests(unittest.TestCase):
    def _inputs(self):
        policy_ids = [f"policy-{index}" for index in range(8)]
        matrices = {
            depth: pd.DataFrame(
                {"feature-a": range(8), "feature-b": range(8, 16)},
                index=policy_ids,
            )
            for depth in ("top10", "full")
        }
        membership = pd.DataFrame([
            {
                "policy_id": policy_id,
                "depth": depth,
                "umap_cluster": 0,
                "ordering": "chronological",
                "reply_mode": "loose",
                "pinned": False,
            }
            for depth in ("top10", "full")
            for policy_id in reversed(policy_ids)
        ])
        coordinates = {
            depth: pd.DataFrame(
                {
                    "umap1": [float(index) for index in range(8)],
                    "umap2": [float((index * index) % 7) for index in range(8)],
                },
                index=policy_ids,
            ).iloc[::-1]
            for depth in ("top10", "full")
        }
        return matrices, membership, coordinates, policy_ids

    def _plot(self, matrices, membership, coordinates, output):
        return plot_umap_clusters(
            matrices,
            membership,
            ("top10", "full"),
            ("chronological",),
            {"chronological": "#336699"},
            {"chronological": "Chronological"},
            output,
            reply_markers={"loose": "o"},
            label_clusters=False,
            show=False,
            display_embeddings=coordinates,
        )

    def test_saved_coordinates_skip_umap_fit_and_align_on_policy_ids(self):
        matrices, membership, coordinates, policy_ids = self._inputs()
        captured = []

        def capture_policy_space(axis, frame, x_column, y_column, **kwargs):
            captured.append(frame[["policy_id", x_column, y_column]].copy())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "restored.png"
            with patch("umap.UMAP", side_effect=AssertionError("saved display coordinates must bypass UMAP")), patch(
                "commentgap_analysis.ranking_similarity_plotting.plot_policy_space",
                side_effect=capture_policy_space,
            ):
                figure = self._plot(matrices, membership, coordinates, output)
            self.assertTrue(output.is_file())
            self.assertEqual(len(captured), 2)
            for depth_index, frame in enumerate(captured):
                self.assertEqual(frame.policy_id.tolist(), policy_ids)
                expected = coordinates[("top10", "full")[depth_index]].loc[policy_ids]
                self.assertEqual(frame.umap1.tolist(), expected.umap1.tolist())
                self.assertEqual(frame.umap2.tolist(), expected.umap2.tolist())
            import matplotlib.pyplot as plt
            plt.close(figure)

    def test_saved_coordinates_reject_missing_policy_ids(self):
        matrices, membership, coordinates, _ = self._inputs()
        coordinates["full"] = coordinates["full"].iloc[:-1]
        with patch("umap.UMAP", side_effect=AssertionError("saved display coordinates must bypass UMAP")):
            with self.assertRaisesRegex(ValueError, "Saved full coordinates do not match"):
                self._plot(matrices, membership, coordinates, Path("unused.png"))


if __name__ == "__main__":
    unittest.main()
