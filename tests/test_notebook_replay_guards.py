from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


# These entry points fit/score models or regenerate scientific products. Frozen
# paths must load their saved return values, not invoke them with cache flags.
PRODUCERS = {
    "prepare_shared_model_data", "run_descriptive_analysis",
    "run_comment_gap_analysis", "freeze_development_cv_winners",
    "run_paper1_reporting", "run_forum_analysis_pipeline",
    "freeze_ranker_handoff", "score_development_model",
    "fit_transform", "fit_predict", "run_policy_inference",
    "fit_vote_attention_models", "diagnostic_fit", "run_oracle_benchmark",
    "compute_draw_metrics_in_batches", "compute_topic_metrics",
    "build_topic_agenda_baseline_analysis", "build_topic_agenda_rarefaction_analysis",
    "build_topic_policy_concentration_analysis", "build_topic_policy_exposure_coverage_analysis",
    "build_policy_reference_target_metrics", "_bootstrap_group_mean",
}


def frozen_branch(test):
    """Evaluate notebook mode checks as recompute, leaving other conditions unknown."""
    text = ast.unparse(test)
    if "mode" not in text:
        return None
    try:
        return bool(eval(compile(ast.Expression(test), "<mode>", "eval"),
                         {"mode": "recompute"}))
    except (NameError, TypeError, AttributeError):
        return None


def frozen_calls(node):
    if isinstance(node, ast.If):
        selected = frozen_branch(node.test)
        children = node.body if selected is True else node.orelse if selected is False else (*node.body, *node.orelse)
    else:
        children = ast.iter_child_nodes(node)
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        yield name
    for child in children:
        yield from frozen_calls(child)


class NotebookFrozenPathTests(unittest.TestCase):
    def test_frozen_paths_load_objects_and_regenerate_plots(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted(root.glob("[0-9][0-9]_*.ipynb"))
        self.assertEqual(len(paths), 15)
        for path in paths:
            notebook = json.loads(path.read_text())
            cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]
                     if cell["cell_type"] == "code"]
            source = "\n".join(cells)
            self.assertIn('mode = "replay"', cells[0], path.name)
            self.assertNotIn("execution_mode()", source, path.name)
            self.assertNotIn("display_frozen_cell", source, path.name)
            self.assertNotIn("initialize_frozen_notebook", source, path.name)
            self.assertNotIn("frozen-cells-", source, path.name)
            for cell in cells:
                calls = set(frozen_calls(ast.parse(cell)))
                self.assertFalse(calls & PRODUCERS, (path.name, calls & PRODUCERS))
                self.assertNotIn("Image", calls, f"{path.name} displays a raster rather than a plot")
