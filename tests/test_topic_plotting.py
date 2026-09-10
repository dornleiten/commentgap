from pathlib import Path
import json
import tempfile
import unittest

try:
    import matplotlib
except ModuleNotFoundError:  # pragma: no cover - depends on the test environment
    matplotlib = None
else:
    matplotlib.use("Agg")
try:
    import pandas as pd
    from commentgap_analysis.topic_plotting import (
        SAMPLE_MARKERS,
        build_topic_search_plot_data,
        plot_topic_search_metrics,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on the test environment
    pd = None
    SAMPLE_MARKERS = {}
    build_topic_search_plot_data = None
    plot_topic_search_metrics = None


@unittest.skipUnless(pd is not None, "Pandas is not installed")
class TopicPlottingTests(unittest.TestCase):
    def _inputs(self):
        setups = [
            ("articles_comments", 5, 5, 1, 0.20, 0.80),
            ("articles_comments", 10, 10, 3, 0.50, 0.50),
            ("articles_comments", 15, 10, 5, 0.80, 0.20),
        ]
        report_rows = []
        seed_rows = []
        for fit_corpus, mcs, nn, ms, article_ami, comment_ami in setups:
            for doc_type in ("article_passage", "comment"):
                report_rows.append({
                    "fit_corpus": fit_corpus,
                    "min_cluster_size": mcs,
                    "n_neighbors": nn,
                    "min_samples": ms,
                    "doc_type": doc_type,
                    "ami_common_pooled_median": article_ami if doc_type == "article_passage" else comment_ami,
                })
                for seed in (2025, 2026, 2027):
                    seed_rows.append({
                        "fit_corpus": fit_corpus,
                        "min_cluster_size": mcs,
                        "n_neighbors": nn,
                        "min_samples": ms,
                        "doc_type": doc_type,
                        "raw_coverage_pct": (article_ami * 100 if doc_type == "article_passage" else comment_ami * 100),
                        "seed": seed,
                    })
        return pd.DataFrame(report_rows), pd.DataFrame(seed_rows)

    def test_build_data_marks_the_maximising_frontier(self):
        report, seeds = self._inputs()
        plot_data = build_topic_search_plot_data(report, seeds)
        self.assertEqual(plot_data.pareto_frontier.tolist(), [True, True, True])
        self.assertEqual(plot_data.configuration.iloc[0], "articles_comments_mcs5_nn5_ms1")

    @unittest.skipUnless(matplotlib is not None, "Matplotlib is not installed")
    def test_renderer_writes_two_ami_figures_with_fixed_bounds_and_shape_outlines(self):
        report, seeds = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for setup in (
                "articles_comments_mcs5_nn5_ms1",
                "articles_comments_mcs10_nn10_ms3",
                "articles_comments_mcs15_nn10_ms5",
            ):
                assignments = pd.DataFrame({
                    "raw_seed_2025": [0, 0, 1],
                    "raw_seed_2026": [0, 0, 1],
                    "raw_seed_2027": [0, 0, 1],
                })
                (root / setup).mkdir()
                assignments.to_parquet(root / setup / "assignments.parquet", index=False)
            plot_data, frontier = plot_topic_search_metrics(
                report,
                seeds,
                root,
                [2025, 2026, 2027],
                final_configuration={
                    "hdbscan_min_cluster_size": 15,
                    "umap_n_neighbors": 10,
                    "hdbscan_min_samples": 5,
                },
                final_fit_corpus="articles_comments",
                show=False,
            )
            self.assertEqual(len(frontier), 3)
            self.assertEqual(sorted(plot_data.min_samples.unique()), [1, 3, 5])
            for stem in (
                "raw_pooled_common_ami_vs_article_coverage_pareto",
                "raw_pooled_common_ami_vs_comment_coverage_pareto",
            ):
                self.assertTrue((root / f"{stem}.png").exists())
                self.assertTrue((root / f"{stem}.pdf").exists())

    def test_notebook_delegates_plotting_to_module(self):
        notebook = json.loads(Path("14_topic_model_fit.ipynb").read_text())
        source = "".join(notebook["cells"][8]["source"])
        self.assertIn("plot_topic_search_metrics", source)
        self.assertNotIn("def ", source)
        self.assertNotIn("matplotlib", source)
        self.assertNotIn("plt.", source)
        self.assertNotIn("Line2D", source)
        self.assertEqual(
            {value: SAMPLE_MARKERS[value] for value in (1, 3, 5, 10, 15)},
            {1: "o", 3: "^", 5: "s", 10: "D", 15: "P"},
        )
