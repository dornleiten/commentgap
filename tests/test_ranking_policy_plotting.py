import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from commentgap_analysis.ranking_policy_plotting import plot_policy_space


class RankingPolicyPlottingTests(unittest.TestCase):
    def test_plot_policy_space_preserves_policy_subgroup_encoding(self):
        frame = pd.DataFrame(
            {
                "ordering": ["a", "a", "b", "b"],
                "reply_mode": ["loose", "trees", "loose", "trees"],
                "pinned": [False, True, False, True],
                "x": [1, 2, 3, 4],
                "y": [4, 3, 2, 1],
            }
        )
        figure, axis = plt.subplots()
        try:
            plot_policy_space(
                axis,
                frame,
                "x",
                "y",
                colour_column="ordering",
                colour_order=["a", "b"],
                colour_palette={"a": "#111111", "b": "#222222"},
                reply_markers={"loose": "o", "trees": "s"},
            )
            self.assertEqual(len(axis.collections), 4)
            self.assertEqual(
                [len(collection.get_offsets()) for collection in axis.collections],
                [1, 1, 1, 1],
            )
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
