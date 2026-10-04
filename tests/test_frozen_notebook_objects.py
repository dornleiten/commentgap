"""Exercise actual notebook loading code against small saved scientific products."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock
from unittest import mock

import pandas as pd
import pyarrow.parquet as pq
from commentgap_analysis.frozen_inputs import unavailable


class FrozenNotebookObjectTests(unittest.TestCase):
    def preprocessing_cell(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "03_shared_model_preprocessing.ipynb").read_text())
        return "".join(next(cell["source"] for cell in notebook["cells"]
                            if cell.get("id") == "2f8c7703"))

    @mock.patch.dict(os.environ, {"COMMENTGAP_MODE": "recompute"})
    def test_preprocessing_restores_typed_result_without_calling_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = pd.DataFrame({"story_id": ["a", "b"], "split_role": ["development", "paper2_test"]})
            balance = pd.DataFrame({"split_role": ["development"], "articles": [1]})
            split.to_parquet(root / "master_article_split.parquet", index=False)
            balance.to_csv(root / "split_balance_diagnostics.csv", index=False)
            (root / "preprocessing_parameters.json").write_text(json.dumps({"reply_depth_centers": {"all": 2.5}}))
            registry = {"models": {"all": {"features": ["reply_depth_centered"]}}}
            (root / "feature_manifest.json").write_text(json.dumps(registry))
            for scope in ("root", "all"):
                split.to_parquet(root / f"choice_set_{scope}.parquet", index=False)
            producer = Mock(side_effect=AssertionError("Frozen mode must not preprocess"))
            namespace = {"mode": "recompute", "MODEL_DATA_ROOT": root,
                         "json": json, "pd": pd, "pq": pq, "unavailable": unavailable,
                         "prepare_shared_model_data": producer}
            exec(self.preprocessing_cell(), namespace)
            result = namespace["result"]
            pd.testing.assert_frame_equal(result["article_split"], split)
            pd.testing.assert_frame_equal(result["split_balance"], balance)
            self.assertEqual(result["feature_manifest"], registry)
            self.assertEqual(result["reply_depth_centers"], {"all": 2.5})
            self.assertEqual(result["rows"], {"root": 2, "all": 2})
            producer.assert_not_called()

    @mock.patch.dict(os.environ, {"COMMENTGAP_MODE": "recompute"})
    def test_missing_preprocessing_input_names_unrestorable_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = {"mode": "recompute", "MODEL_DATA_ROOT": Path(directory),
                         "json": json, "pd": pd, "unavailable": unavailable}
            output = io.StringIO()
            with redirect_stdout(output):
                exec(self.preprocessing_cell(), namespace)
            self.assertIn("result", output.getvalue())
            self.assertIn("master_article_split.parquet", output.getvalue())
            self.assertNotIn("result", namespace)

    def diagnostics_cells(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "04_feature_distribution_diagnostics.ipynb").read_text())
        cells = {cell.get("id"): "".join(cell["source"]) for cell in notebook["cells"]}
        return cells["cg1-04-000"], cells["aab0a3c3"]

    @mock.patch.dict(os.environ, {"COMMENTGAP_MODE": "fresh"})
    def test_diagnostics_samples_are_saved_reused_on_resume_and_loaded_frozen(self):
        setup_cell, sample_cell = self.diagnostics_cells()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            feature_root = output_root / "explicit-features"
            model_root = output_root / "explicit-model-data"
            diagnostics_root = output_root / "explicit-diagnostics"
            config = output_root / "config"
            config.mkdir()
            (config / "paths.json").write_text(json.dumps({
                "schema_version": 1,
                "paths": {"outputs": "outputs", "features": "configured/features",
                          "model_data": "configured/model_data",
                          "frozen_cg1_feature_diagnostics": "configured/diagnostics"},
            }))
            with mock.patch.dict("os.environ", {
                "COMMENTGAP_PROJECT_ROOT": str(output_root),
                "COMMENTGAP_MODE": "recompute",
                "COMMENTGAP_FEATURE_ROOT": str(feature_root),
                "COMMENTGAP_MODEL_DATA_ROOT": str(model_root),
                "COMMENTGAP_FEATURE_DIAGNOSTIC_ROOT": str(diagnostics_root),
            }, clear=True):
                setup_namespace = {}
                exec(setup_cell, setup_namespace)
                self.assertEqual(Path(os.environ["COMMENTGAP_FEATURE_ROOT"]), feature_root)
                self.assertEqual(Path(os.environ["COMMENTGAP_MODEL_DATA_ROOT"]), model_root)
                self.assertEqual(Path(os.environ["COMMENTGAP_FEATURE_DIAGNOSTIC_ROOT"]), diagnostics_root)

            sample_rows = pd.DataFrame({
                "story_id": ["story-a", "story-b"],
                "comment_id": ["comment-a", "comment-b"],
                "n_candidates": [2, 3],
                "n_picks": [1, 1],
                "curator_selected": [True, False],
                "audience_selected_draw_01": [False, True],
                "feature": [0.25, 0.75],
            })
            class Cursor:
                def fetchdf(self):
                    return sample_rows.copy()

            connection = Mock()
            connection.execute.return_value = Cursor()
            parquet_schema = SimpleNamespace(schema_arrow=SimpleNamespace(names=list(sample_rows.columns)))
            pq = SimpleNamespace(ParquetFile=Mock(return_value=parquet_schema))
            namespace = {
                "mode": "fresh",
                "OUTPUT_ROOT": output_root,
                "choice_paths": {"root": output_root / "root.parquet", "all": output_root / "all.parquet"},
                "model_features": {"root": ["feature"], "all": ["feature"]},
                "pq": pq,
                "quote_identifier": lambda name: name,
                "SAMPLE_ROWS": 100,
                "SEED": 42,
                "connection": connection,
                "pd": pd,
                "display": Mock(),
            }
            exec(sample_cell, namespace)
            self.assertEqual(connection.execute.call_count, 2)
            self.assertTrue((output_root / "sample_root.parquet").is_file())
            self.assertTrue((output_root / "sample_all.parquet").is_file())
            self.assertTrue((output_root / "sample_accounting.csv").is_file())
            expected_samples = namespace["samples"]

            resumed_connection = Mock(side_effect=AssertionError("Resume must reuse the saved sample"))
            namespace.update({"connection": resumed_connection})
            exec(sample_cell, namespace)
            resumed_connection.execute.assert_not_called()
            for scope in ("root", "all"):
                pd.testing.assert_frame_equal(namespace["samples"][scope], expected_samples[scope])

            frozen_namespace = {
                "mode": "recompute",
                "SAVED_DIAGNOSTICS_ROOT": output_root,
                "pd": pd,
                "display": Mock(),
            }
            with mock.patch.dict(os.environ, {"COMMENTGAP_MODE": "recompute"}):
                exec(sample_cell, frozen_namespace)
            self.assertEqual(set(frozen_namespace["samples"]), {"root", "all"})
            for scope in ("root", "all"):
                pd.testing.assert_frame_equal(frozen_namespace["samples"][scope], expected_samples[scope])
