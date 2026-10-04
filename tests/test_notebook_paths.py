from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from commentgap_analysis.notebook_paths import configure_notebook_stage
from commentgap_analysis.paths import ExecutionContext, PathContractError


class NotebookPathTests(unittest.TestCase):
    def test_fresh_stage_uses_run_root_for_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "paths.json").write_text(json.dumps({
                "schema_version": 1, "paths": {"outputs": "outputs"}
            }))
            with mock.patch.dict(os.environ, {
                "COMMENTGAP_PROJECT_ROOT": str(root),
                "COMMENTGAP_MODE": "fresh",
                "COMMENTGAP_RUN_ID": "fixture-run",
            }, clear=False):
                output = configure_notebook_stage("05")
                configured_root = os.environ["COMMENTGAP_DESCRIPTIVES_ROOT"]
            self.assertEqual(output, root / "outputs/fixture-run/CG1/descriptives")
            self.assertTrue(output.parent.parent.is_dir())
            self.assertEqual(configured_root, str(output))
            self.assertNotIn("model_output", str(output))

    def test_sequential_stages_record_actual_inputs_and_changed_resume_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "paths.json").write_text(json.dumps({
                "schema_version": 1,
                "paths": {"outputs": "outputs", "raw_scrape": "data/raw/scrape", "features": "data/derived/features"},
            }))
            env = {
                "COMMENTGAP_PROJECT_ROOT": str(root),
                "COMMENTGAP_MODE": "fresh",
                "COMMENTGAP_RUN_ID": "sequential",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                stage_two = configure_notebook_stage("02")
                # A producer cell materialises the first stage's output before
                # the next notebook starts.
                stage_two.mkdir(parents=True, exist_ok=True)
                source = stage_two / "fixture.txt"
                source.write_text("before")
                stage_three = configure_notebook_stage("03")
                self.assertEqual(stage_three, root / "outputs/sequential/shared/model_data")
                manifest = json.loads((root / "outputs/sequential/run.json").read_text())
                self.assertIn("notebook/02", manifest["stages"])
                self.assertIn("notebook/03", manifest["stages"])
                recorded = manifest["stages"]["notebook/03"]
                self.assertIn(str(stage_two), recorded["inputs"])
                self.assertRegex(recorded["input_hashes"][str(stage_two)], r"^[0-9a-f]{64}$")
                source.write_text("after")
                os.environ["COMMENTGAP_MODE"] = "resume"
                with self.assertRaisesRegex(PathContractError, "identity differs"):
                    configure_notebook_stage("03")

    def test_notebook_orchestration_does_not_claim_cli_output_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "paths.json").write_text(json.dumps({
                "schema_version": 1, "paths": {"outputs": "outputs"}
            }))
            with mock.patch.dict(os.environ, {
                "COMMENTGAP_PROJECT_ROOT": str(root),
                "COMMENTGAP_MODE": "fresh",
                "COMMENTGAP_RUN_ID": "subprocess",
            }, clear=True):
                configure_notebook_stage("02")
                cli = ExecutionContext.from_values()
                self.assertEqual(
                    cli.output_root("shared/features"),
                    root / "outputs/subprocess/shared/features",
                )

    def test_stage08_writes_winners_and_reads_factorial_rankers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "paths.json").write_text(json.dumps({
                "schema_version": 1,
                "paths": {"outputs": "outputs", "frozen_cg1_factorial_rankers": "artifacts/frozen/CG1/rankers/factorial"},
            }))
            with mock.patch.dict(os.environ, {
                "COMMENTGAP_PROJECT_ROOT": str(root),
                "COMMENTGAP_MODE": "fresh",
                "COMMENTGAP_RUN_ID": "winners",
            }, clear=True):
                rankers = root / "outputs/winners/CG1/rankers/factorial"
                rankers.mkdir(parents=True)
                (rankers / "experiment_variants.csv").write_text("fixture")
                output = configure_notebook_stage("08")
                self.assertEqual(output, root / "outputs/winners/CG1/winners")
                self.assertEqual(
                    os.environ["COMMENTGAP_FACTORIAL_ROOT"], str(rankers)
                )
                self.assertEqual(
                    os.environ["COMMENTGAP_FACTORIAL_WINNER_ROOT"], str(output)
                )

    def test_stage09_routes_xgboost_cache_into_fresh_and_resumed_run_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            input_paths = {
                "model_data": "fixture/model_data",
                "frozen_cg1_factorial_rankers": "fixture/factorial",
                "frozen_cg1_winners": "fixture/winners",
                "frozen_cg1_regression": "fixture/regression",
            }
            for relative in input_paths.values():
                (root / relative).mkdir(parents=True)
            (root / "config" / "paths.json").write_text(json.dumps({
                "schema_version": 1,
                "paths": {"outputs": "outputs", **input_paths},
            }))
            env = {
                "COMMENTGAP_PROJECT_ROOT": str(root),
                "COMMENTGAP_MODE": "fresh",
                "COMMENTGAP_RUN_ID": "paper1-reporting",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                output = configure_notebook_stage("09")
                expected = output / "cache" / "xgboost_bge"
                self.assertEqual(Path(os.environ["COMMENTGAP_XGBOOST_CACHE_ROOT"]), expected)
                os.environ["COMMENTGAP_MODE"] = "resume"
                resumed_output = configure_notebook_stage("09")
                self.assertEqual(resumed_output, output)
                self.assertEqual(
                    Path(os.environ["COMMENTGAP_XGBOOST_CACHE_ROOT"]), expected
                )

    def test_stage11_reads_forum_products_and_writes_separate_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "paths.json").write_text(json.dumps({
                "schema_version": 1,
                "paths": {
                    "outputs": "outputs",
                    "model_data": "fixture/model_data",
                    "frozen_cg1_factorial_rankers": "fixture/factorial",
                    "frozen_cg1_winners": "fixture/winners",
                    "frozen_cg1_regression": "fixture/regression",
                    "raw_scrape": "fixture/raw",
                    "embeddings": "fixture/embeddings",
                },
            }))
            env = {
                "COMMENTGAP_PROJECT_ROOT": str(root),
                "COMMENTGAP_MODE": "fresh",
                "COMMENTGAP_RUN_ID": "forum-routing",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                forum_root = configure_notebook_stage("10")
                comments = forum_root / "analysis_comments.parquet"
                scores = forum_root / "policy_scores/policy_scores.parquet"
                comments.parent.mkdir(parents=True)
                scores.parent.mkdir(parents=True)
                comments.write_bytes(b"comments")
                scores.write_bytes(b"scores")

                inference_root = configure_notebook_stage("11")
                manifest = json.loads((root / "outputs/forum-routing/run.json").read_text())
                inputs = manifest["stages"]["notebook/11"]["inputs"]
                self.assertEqual(inference_root, root / "outputs/forum-routing/CG2/forum/inference")
                self.assertEqual(Path(os.environ["COMMENTGAP_FORUM_ANALYSIS_ROOT"]), forum_root)
                self.assertEqual(Path(os.environ["COMMENTGAP_FORUM_INFERENCE_ROOT"]), inference_root)
                self.assertEqual(Path(os.environ["COMMENTGAP_FORUM_REPORTING_ROOT"]), forum_root / "reporting")
                self.assertIn(str(comments), inputs)
                self.assertIn(str(scores), inputs)

                inference_root.mkdir(parents=True, exist_ok=True)
                (inference_root / "inference_manifest.json").write_text("stage output")
                os.environ["COMMENTGAP_MODE"] = "resume"
                resumed = configure_notebook_stage("11")
                self.assertEqual(resumed, inference_root)

    def test_stage13_hashes_nested_policy_score_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "paths.json").write_text(json.dumps({
                "schema_version": 1, "paths": {"outputs": "outputs"}
            }))
            score_path = root / "outputs/ranking/CG2/forum/policy_scores/policy_scores.parquet"
            score_path.parent.mkdir(parents=True)
            score_path.write_bytes(b"policy scores")
            with mock.patch.dict(os.environ, {
                "COMMENTGAP_PROJECT_ROOT": str(root),
                "COMMENTGAP_MODE": "fresh",
                "COMMENTGAP_RUN_ID": "ranking",
                "COMMENTGAP_FORUM_SCORES_PATH": str(score_path),
            }, clear=True):
                output = configure_notebook_stage("13")
                manifest = json.loads((root / "outputs/ranking/run.json").read_text())
                recorded = manifest["stages"]["notebook/13"]
                self.assertEqual(output, root / "outputs/ranking/CG2/ranking_similarity")
                self.assertEqual(recorded["inputs"], [str(score_path)])
                self.assertRegex(recorded["input_hashes"][str(score_path)], r"^[0-9a-f]{64}$")

    def test_stage13_frozen_fallback_uses_nested_policy_scores_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "paths.json").write_text(json.dumps({
                "schema_version": 1,
                "paths": {
                    "outputs": "outputs",
                    "frozen_cg2_forum": "artifacts/frozen/CG2/forum",
                },
            }))
            score_path = root / "artifacts/frozen/CG2/forum/policy_scores/policy_scores.parquet"
            score_path.parent.mkdir(parents=True)
            score_path.write_bytes(b"frozen policy scores")
            with mock.patch.dict(os.environ, {
                "COMMENTGAP_PROJECT_ROOT": str(root),
                "COMMENTGAP_MODE": "fresh",
                "COMMENTGAP_RUN_ID": "ranking-frozen-source",
            }, clear=True):
                configure_notebook_stage("13")
                manifest = json.loads((root / "outputs/ranking-frozen-source/run.json").read_text())
                self.assertEqual(manifest["stages"]["notebook/13"]["inputs"], [str(score_path)])

    def test_stage16_keeps_calculation_inputs_separate_from_reporting_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "paths.json").write_text(json.dumps({
                "schema_version": 1, "paths": {"outputs": "outputs"}
            }))
            run_root = root / "outputs/topics"
            input_root = run_root / "CG2/topics/analysis"
            input_root.mkdir(parents=True)
            (input_root / "fixture.csv").write_text("calculation input")
            with mock.patch.dict(os.environ, {
                "COMMENTGAP_PROJECT_ROOT": str(root),
                "COMMENTGAP_MODE": "fresh",
                "COMMENTGAP_RUN_ID": "topics",
                "COMMENTGAP_TOPIC_INPUT_ROOT": str(input_root),
                "COMMENTGAP_TOPIC_RUNS_ROOT": str(run_root / "CG2/topics/runs"),
                "COMMENTGAP_ANALYSIS_COMMENTS_PATH": str(root / "comments.parquet"),
            }, clear=True):
                output = configure_notebook_stage("16")
                manifest = json.loads((run_root / "run.json").read_text())
                inputs = manifest["stages"]["notebook/16"]["inputs"]
                self.assertEqual(output, run_root / "CG2/topics/reporting")
                self.assertEqual(Path(os.environ["COMMENTGAP_TOPIC_INPUT_ROOT"]), input_root)
                self.assertEqual(Path(os.environ["COMMENTGAP_TOPIC_ANALYSIS_ROOT"]), output)
                self.assertIn(str(input_root), inputs)


if __name__ == "__main__":
    unittest.main()
