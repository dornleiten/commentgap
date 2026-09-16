import tempfile
import unittest
from pathlib import Path

import pandas as pd

from commentgap_analysis.forum_plotting import plot_forum_calculation
from commentgap_analysis.paper1_plotting import plot_regression_selector_differences
from commentgap_analysis.plotting import save_display_figure


class PlottingOutputTests(unittest.TestCase):
    def test_display_only_does_not_write_or_change_batch_artifacts(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close("all")

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "report"
            saved = save_display_figure(
                plt.subplots()[0], output_root, "example", show=False
            )
            png = output_root / "figures" / "example.png"
            pdf = output_root / "figures" / "example.pdf"
            before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (png, pdf)}

            display_figure = save_display_figure(
                plt.subplots()[0], None, "example", show=False
            )
            self.assertEqual(
                before,
                {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (png, pdf)},
            )
            self.assertFalse((output_root / "figures" / "example.svg").exists())
            self.assertEqual(plt.get_fignums(), [])
            plt.close(saved)
            plt.close(display_figure)

    def test_module_plot_can_render_without_output_root(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frame = pd.DataFrame(
            {
                "scope": ["all"],
                "feature": ["log_words"],
                "curator_minus_audience_log_odds": [0.5],
                "difference_conf_low": [0.2],
                "difference_conf_high": [0.8],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "report"
            figure = plot_regression_selector_differences(
                frame, output_root=None, show=False
            )
            self.assertFalse(output_root.exists())
            self.assertEqual(len(figure.axes), 1)
            plt.close(figure)

    def test_forum_calculation_plot_writes_requested_formats(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        values = np.linspace(0.1, 0.9, 10)
        policy = np.cumsum(values[::-1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figure = plot_forum_calculation(
                values,
                policy,
                rank_i=5,
                output_paths=(root / "feature_score.pdf", root / "feature_score.svg"),
                show=False,
            )
            self.assertEqual(len(figure.axes), 1)
            self.assertTrue((root / "feature_score.pdf").exists())
            self.assertTrue((root / "feature_score.svg").exists())
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
