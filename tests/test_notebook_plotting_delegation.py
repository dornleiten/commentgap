import ast
import json
from pathlib import Path
import unittest


PLOT_METHODS = {
    "add_patch", "axhline", "axvline", "bar", "figure", "heatmap", "hist",
    "plot", "regplot", "savefig", "scatter", "show", "subplots",
}


class NotebookPlottingDelegationTests(unittest.TestCase):
    def test_selected_notebooks_have_no_direct_plot_construction(self):
        for filename in (
            "04_feature_distribution_diagnostics.ipynb",
            "12_forum_correlations.ipynb",
            "13_ranking_algorithm_similarity.ipynb",
            "15_topic_agenda_calculations.ipynb",
        ):
            notebook = json.loads(Path(filename).read_text())
            for cell_number, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell.get("source", [])))
                direct_calls = []
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    if isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    elif isinstance(node.func, ast.Name):
                        name = node.func.id
                    else:
                        name = ""
                    if name in PLOT_METHODS:
                        direct_calls.append((name, node.lineno))
                self.assertEqual(
                    direct_calls,
                    [],
                    f"{filename} cell {cell_number} still plots inline: {direct_calls}",
                )

    def test_notebooks_call_the_extracted_plot_helpers(self):
        expected = {
            "04_feature_distribution_diagnostics.ipynb": {
                "plot_aqua_expected_distributions",
                "plot_spearman_correlation_heatmap",
                "plot_selection_contrasts",
            },
            "12_forum_correlations.ipynb": {
                "plot_top10_full_forum",
                "plot_forum_ndcg",
                "plot_regression_forum_coefficients",
                "plot_ml_forum_shap",
            },
            "13_ranking_algorithm_similarity.ipynb": {"plot_umap_clusters"},
            "15_topic_agenda_calculations.ipynb": {"plot_vote_attention_curve"},
        }
        for filename, names in expected.items():
            source = Path(filename).read_text()
            for name in names:
                self.assertIn(name, source)


if __name__ == "__main__":
    unittest.main()
