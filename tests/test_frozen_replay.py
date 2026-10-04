import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from commentgap_analysis.paths import ExecutionContext, PathContractError
from commentgap_analysis.replay import replay_cg1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenReplayTests(unittest.TestCase):
    @patch.dict(os.environ, {"COMMENTGAP_MODE": "recompute"})
    def test_cg1_replay_uses_saved_products_without_mutating_them(self):
        context = ExecutionContext.from_values(mode="frozen", run_id="frozen-replay-test")
        source = context.paths.read_root("frozen_cg1_reporting")
        if not source.is_dir():
            self.skipTest("Frozen reporting requires the external artifact bundle")
        watched = [
            source / "tables" / "held_out_model_performance.csv",
            source / "tables" / "cg1_baseline_performance.csv",
            source / "tables" / "held_out_shap_importance.csv",
        ]
        before = {path: _sha256(path) for path in watched}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rendered" / "CG1"
            manifest = replay_cg1(context, output_root=output)
            self.assertTrue((output / "reporting" / "figures" / "all_winner_held_out_ndcg.pdf").is_file())
            self.assertTrue((output / "reporting" / "tables" / "cg1_test_performance_with_baselines.csv").is_file())
            self.assertEqual(len(manifest["reporting"]["generated_figures"]), 22)
            submitted = context.paths.root / "submissions/CG1/tables/test_model_performance.tex"
            regenerated = output / "reporting/tables/cg1_test_performance_with_baselines.tex"
            self.assertEqual(submitted.read_text().strip(), regenerated.read_text().strip())
        self.assertEqual(before, {path: _sha256(path) for path in watched})

    def test_replay_rejects_fresh_execution_context(self):
        context = ExecutionContext.from_values(mode="fresh", run_id="new-scientific-run")
        with self.assertRaises(PathContractError):
            replay_cg1(context)


if __name__ == "__main__":
    unittest.main()
