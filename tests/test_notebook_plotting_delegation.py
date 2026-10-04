import ast
import json
from pathlib import Path
import unittest


PLOT_METHODS = {
    "add_patch", "axhline", "axvline", "bar", "figure", "heatmap", "hist",
    "plot", "regplot", "savefig", "scatter", "show", "subplots",
}


class NotebookPlottingDelegationTests(unittest.TestCase):
    def test_notebooks_have_no_inline_functions_or_plot_construction(self):
        for notebook_path in sorted(Path.cwd().glob("[0-1][0-9]_*.ipynb")):
            notebook = json.loads(notebook_path.read_text())
            for cell_number, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell.get("source", [])))
                functions = [node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                             else 'lambda' for node in ast.walk(tree)
                             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))]
                self.assertEqual(
                    functions, [],
                    f"{notebook_path.name} cell {cell_number} defines functions: {functions}",
                )
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
                    f"{notebook_path.name} cell {cell_number} still plots inline: {direct_calls}",
                )

    def test_notebooks_call_the_extracted_plot_helpers(self):
        expected = {
            "04_feature_distribution_diagnostics.ipynb": {
                "plot_aqua_distributions_from_bins",
                "plot_distribution_grid_from_bins",
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
