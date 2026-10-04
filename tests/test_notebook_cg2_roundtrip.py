"""Regression tests for the saved-input roundtrips in CG2 notebooks 10 and 13."""

import json
import hashlib
import tempfile
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from commentgap_analysis.forum_analysis_cli import resolve_forum_canonical_path
from commentgap_analysis.ranking_similarity_plotting import (
    configure_similarity_plot_style, joined_unique_values,
)
import seaborn as sns
from IPython.display import Markdown
from commentgap_analysis.paths import ProjectPaths


ROOT = Path(__file__).resolve().parents[1]


def notebook_cell(notebook_name, cell_index):
    notebook = json.loads((ROOT / notebook_name).read_text())
    return "".join(notebook["cells"][cell_index]["source"])


class NotebookCG2RoundtripTests(unittest.TestCase):
    @mock.patch.dict("os.environ", {"COMMENTGAP_MODE": "recompute"})
    def test_notebook10_reads_relative_and_legacy_absolute_pipeline_paths(self):
        cell = notebook_cell("10_calculate_forum_scores.ipynb", 6)
        for canonical_format in ("relative", "absolute"):
            with self.subTest(canonical_format=canonical_format), tempfile.TemporaryDirectory() as directory:
                repo_root = Path(directory)
                forum_root = repo_root / "artifacts/frozen/CG2/forum"
                for relative_path in (
                    "ranker_handoff/ranker_handoff_manifest.json",
                    "analysis_comments.parquet",
                    "policy_scores/policy_scores.parquet",
                    "inference/inference_manifest.json",
                    "reporting/reporting_manifest.json",
                ):
                    path = forum_root / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}")

                target = "analysis_comments.parquet"
                canonical_path = (
                    f"CG2/forum/{target}"
                    if canonical_format == "relative"
                    else f"{forum_root}/{target}"
                )
                record = {
                    "pipeline": {
                        "stage_status": {"build": "reused"},
                        "targets": {
                            "build": [{
                                "original_path": f"/old/output/{target}",
                                "canonical_path": canonical_path,
                            }],
                        },
                    },
                }
                (forum_root / "pipeline_result.json").write_text(json.dumps(record))
                displayed = []
                namespace = {
                    "mode": "recompute",
                    "REPO_ROOT": repo_root,
                    "FORUM_ANALYSIS_ROOT": forum_root,
                    "Path": Path,
                    "json": json,
                    "pd": pd,
                    "display": displayed.append,
                    "unavailable": lambda *args: self.fail(f"Unexpected missing input: {args}"),
                    "resolve_forum_canonical_path": resolve_forum_canonical_path,
                }

                exec(compile(cell, "10_calculate_forum_scores.ipynb#cell6", "exec"), namespace)

                expected = forum_root / target
                self.assertEqual(namespace["pipeline"]["targets"]["build"], [expected])
                self.assertEqual(
                    namespace["pipeline_target_path_mapping"]["build"][0]["canonical_path"],
                    expected,
                )
                self.assertEqual(namespace["pipeline"]["stage_status"], {"build": "reused"})
                self.assertTrue(displayed)

    @mock.patch.dict("os.environ", {"COMMENTGAP_MODE": "fresh"})
    def test_notebook13_saves_and_restores_keyed_umap_coordinates_without_refitting(self):
        fresh_cell = notebook_cell("13_ranking_algorithm_similarity.ipynb", 8)
        save_cell = notebook_cell("13_ranking_algorithm_similarity.ipynb", 9)
        policy_ids = [f"policy-{index}" for index in range(6)]
        depths = ("top10", "full")
        matrices = {
            depth: pd.DataFrame(
                {"feature_a": np.arange(6.0), "feature_b": np.arange(6.0)[::-1]},
                index=policy_ids,
            )
            for depth in depths
        }
        policy_metadata = pd.DataFrame({
            "policy_id": policy_ids,
            "ordering": ["chronological"] * len(policy_ids),
            "reply_mode": ["loose"] * len(policy_ids),
            "pinned": [False] * len(policy_ids),
            "deployable": [True] * len(policy_ids),
        })

        class FakeUMAP:
            fit_count = 0

            def __init__(self, n_components, **kwargs):
                self.n_components = n_components

            def fit_transform(self, frame):
                type(self).fit_count += 1
                return np.arange(len(frame) * self.n_components, dtype=float).reshape(
                    len(frame), self.n_components
                )

        class FakeHDBSCAN:
            def __init__(self, **kwargs):
                pass

            def fit_predict(self, values):
                return np.array([0, 0, 0, 1, 1, 1])

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            fresh_plot_calls = []

            def fresh_plot(*args, **kwargs):
                fresh_plot_calls.append(kwargs["display_embeddings"])
                Path(args[6]).write_bytes(b"synthetic plot")

            fresh_namespace = {
                "mode": "fresh",
                "umap": types.SimpleNamespace(UMAP=FakeUMAP),
                "HDBSCAN": FakeHDBSCAN,
                "pd": pd,
                "np": np,
                "matrices": matrices,
                "policy_metadata": policy_metadata,
                "DEPTHS": depths,
                "REPLY_DISPLAY_MARKERS": {"loose": "o"},
                "OUTPUT_ROOT": output_root,
                "ORDERING_DISPLAY_ORDER": ["chronological"],
                "ORDERING_DISPLAY_COLORS": {"chronological": "#000000"},
                "ORDERING_DISPLAY_LABELS": {"chronological": "Chronological"},
                "display": lambda value: None,
                "plot_umap_clusters": fresh_plot,
                "joined_unique_values": joined_unique_values,
                "configure_similarity_plot_style": configure_similarity_plot_style,
                "sns": sns,
                "plt": plt,
            }
            FakeUMAP.fit_count = 0
            from commentgap_analysis import ranking_similarity_plotting
            with mock.patch.object(ranking_similarity_plotting, "plot_umap_clusters", side_effect=fresh_plot):
                exec(compile(fresh_cell, "13_ranking_algorithm_similarity.ipynb#cell8-fresh", "exec"), fresh_namespace)
                fresh_namespace["feature_means"] = pd.DataFrame({"value": [1]})
                exec(compile(save_cell, "13_ranking_algorithm_similarity.ipynb#cell9-fresh", "exec"), fresh_namespace)

            self.assertEqual(FakeUMAP.fit_count, 4)  # 10D and 2D once for each depth.
            self.assertEqual(len(fresh_plot_calls), 1)
            saved_10d = pd.read_parquet(output_root / "clustering_coordinates.parquet")
            saved_2d = pd.read_parquet(output_root / "display_coordinates.parquet")
            self.assertEqual(len(saved_10d), 12)
            self.assertEqual(sum(column.startswith("component_") for column in saved_10d), 10)
            self.assertEqual(len(saved_2d), 12)
            self.assertTrue({"policy_id", "depth", "umap1", "umap2"}.issubset(saved_2d.columns))

            from commentgap_analysis import paths
            frozen_plot_calls = []

            def frozen_plot(*args, **kwargs):
                frozen_plot_calls.append(kwargs["display_embeddings"])
                for depth in depths:
                    self.assertEqual(kwargs["display_embeddings"][depth].loc[policy_ids].shape, (6, 2))
                Path(args[6]).write_bytes(b"synthetic plot")

            fake_paths = types.SimpleNamespace(configured=lambda key: output_root)
            with (
                mock.patch.object(ranking_similarity_plotting, "plot_umap_clusters", side_effect=frozen_plot),
                mock.patch.object(paths.ProjectPaths, "load", return_value=fake_paths),
            ):
                frozen_namespace = {
                    "mode": "recompute",
                    "Path": Path,
                    "pd": pd,
                    "np": np,
                    "matrices": matrices,
                    "policy_metadata": policy_metadata,
                    "DEPTHS": depths,
                    "REPLY_DISPLAY_MARKERS": {"loose": "o"},
                    "OUTPUT_ROOT": output_root,
                    "REPO_ROOT": Path(directory),
                    "ORDERING_DISPLAY_ORDER": ["chronological"],
                    "ORDERING_DISPLAY_COLORS": {"chronological": "#000000"},
                    "ORDERING_DISPLAY_LABELS": {"chronological": "Chronological"},
                    "display": lambda value: None,
                    "sns": sns,
                    "plt": plt,
                    "Markdown": Markdown,
                    "ProjectPaths": ProjectPaths,
                    "TemporaryDirectory": TemporaryDirectory,
                    "hashlib": hashlib,
                    "plot_umap_clusters": ranking_similarity_plotting.plot_umap_clusters,
                    "joined_unique_values": joined_unique_values,
                    "configure_similarity_plot_style": configure_similarity_plot_style,
                }
                with mock.patch.dict("os.environ", {"COMMENTGAP_MODE": "recompute"}):
                    exec(compile(fresh_cell, "13_ranking_similarity.ipynb#cell8-frozen", "exec"), frozen_namespace)
                    exec(compile(save_cell, "13_ranking_similarity.ipynb#cell9-frozen", "exec"), frozen_namespace)

            self.assertEqual(len(frozen_plot_calls), 1)
            self.assertEqual(FakeUMAP.fit_count, 4)  # Frozen replay did not fit UMAP.
            self.assertEqual(set(frozen_namespace["clustering_embeddings"]), set(depths))
            expected_last_depth = (
                saved_10d.loc[saved_10d["depth"].eq(depths[-1])]
                .set_index("policy_id")
                .loc[policy_ids, [f"component_{i}" for i in range(10)]]
                .to_numpy()
            )
            np.testing.assert_array_equal(frozen_namespace["cluster_embedding"], expected_last_depth)


if __name__ == "__main__":
    unittest.main()
