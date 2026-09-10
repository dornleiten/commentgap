"""Current caches, explicit historical reuse, and fresh oracle execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from commentgap_analysis.topic_artifacts import (
    begin_topic_run, publish_topic_run, read_cached_parquet, read_existing_run_parquet,
    reusable_artifacts, topic_calculation_config, validate_topic_run, write_signature,
)
from commentgap_analysis.topic_policy import read_existing_oracle, run_oracle_benchmark


class ArtifactReuseTests(unittest.TestCase):
    def test_completion_manifest_requires_full_inventory_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.csv"
            optional = root / "source_manifest.json"
            result.write_text("value\n1\n")
            token = begin_topic_run(root, {"inputs": {"version": 1}})
            published = publish_topic_run(
                root, run_token=token, provenance={"inputs": {"version": 1}},
                required_artifacts={"result": result},
                optional_artifacts={"source": optional},
            )
            self.assertEqual(published["status"], "complete")
            validated = validate_topic_run(
                root, expected_provenance={"inputs": {"version": 1}},
                required_artifacts={"result": result},
                optional_artifacts={"source": optional},
            )
            self.assertEqual(validated["status"], "complete")
            result.write_text("value\n2\n")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_topic_run(
                    root, expected_provenance={"inputs": {"version": 99}},
                    required_artifacts={"result": result},
                    optional_artifacts={"source": optional},
                    allow_older_results=True,
                )

    def test_interrupted_run_cannot_be_read_with_older_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.csv"
            result.write_text("value\n1\n")
            begin_topic_run(root, {"inputs": {"version": 1}})
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                validate_topic_run(
                    root, required_artifacts={"result": result},
                    allow_older_results=True, allow_legacy_missing=True,
                )

    def test_publish_missing_required_output_leaves_in_progress_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = begin_topic_run(root, {})
            with self.assertRaisesRegex(RuntimeError, "missing artifacts"):
                publish_topic_run(
                    root, run_token=token, provenance={},
                    required_artifacts={"missing": root / "missing.csv"},
                )
            self.assertEqual(
                json.loads((root / "topic_agenda_completion_manifest.json").read_text())["status"],
                "in_progress",
            )

    def test_older_complete_run_allows_provenance_difference_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.csv"
            result.write_text("value\n1\n")
            token = begin_topic_run(root, {"inputs": {"version": 1}})
            publish_topic_run(
                root, run_token=token, provenance={"inputs": {"version": 1}},
                required_artifacts={"result": result},
            )
            self.assertEqual(
                validate_topic_run(
                    root, expected_provenance={"inputs": {"version": 2}},
                    required_artifacts={"result": result}, allow_older_results=True,
                )["status"],
                "complete",
            )
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                validate_topic_run(
                    root, expected_provenance={"inputs": {"version": 2}},
                    required_artifacts={"result": result},
                )

    def test_strict_validation_rejects_historical_reuse_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.csv"
            result.write_text("value\n1\n")
            token = begin_topic_run(root, {"run_id": "run"})
            publish_topic_run(
                root, run_token=token,
                provenance={
                    "run_id": "run", "inputs": {"version": 1},
                    "reuse": {"allow_older_run_results": True},
                },
                required_artifacts={"result": result},
            )
            with self.assertRaisesRegex(RuntimeError, "older calculations"):
                validate_topic_run(
                    root, expected_provenance={"run_id": "run", "inputs": {"version": 1}},
                    required_artifacts={"result": result},
                )
            self.assertEqual(
                validate_topic_run(
                    root, expected_provenance={"run_id": "run", "inputs": {"version": 1}},
                    required_artifacts={"result": result}, allow_older_results=True,
                )["status"],
                "complete",
            )

    def test_publish_requires_matching_run_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.csv"
            result.write_text("value\n1\n")
            begin_topic_run(root, {})
            with self.assertRaisesRegex(RuntimeError, "different run"):
                publish_topic_run(
                    root, run_token="wrong", provenance={},
                    required_artifacts={"result": result},
                )

    def test_calculation_config_preserves_actual_oracle_kwargs(self):
        config = topic_calculation_config(
            score_policies={"random": None}, policy_specs=(),
            oracle_kwargs={"max_iterations": 99, "custom_setting": "edited"},
        )
        self.assertEqual(config["oracle"], {"max_iterations": 99, "custom_setting": "edited"})

    def test_calculation_config_canonicalizes_family_order(self):
        kwargs = dict(score_policies={"random": None}, policy_specs=())
        first = topic_calculation_config(
            analysis_models=("spline", "power_law"),
            distance_measures=("cosine", "jensen_shannon"), **kwargs,
        )
        second = topic_calculation_config(
            analysis_models=("power_law", "spline"),
            distance_measures=("jensen_shannon", "cosine"), **kwargs,
        )
        self.assertEqual(first, second)

    def test_legacy_override_requires_basic_file_and_schema_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.parquet"
            pd.DataFrame({"story_id": ["s"]}).to_parquet(result)
            self.assertEqual(
                validate_topic_run(
                    root, required_artifacts={"result": result},
                    schema_checks={"result": ("story_id",)},
                    allow_older_results=True, allow_legacy_missing=True,
                )["status"],
                "legacy_complete",
            )
            with self.assertRaisesRegex(RuntimeError, "invalid artifact"):
                validate_topic_run(
                    root, required_artifacts={"result": result},
                    schema_checks={"result": ("missing",)},
                    allow_older_results=True, allow_legacy_missing=True,
                )

    def test_stale_cache_requires_explicit_override_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.parquet'
            manifest = Path(directory) / 'manifest.json'
            frame = pd.DataFrame({'story_id': ['s']})
            frame.to_parquet(path)
            write_signature(manifest, {'version': 1})
            kwargs = dict(manifest_path=manifest, signature={'version': 2})
            self.assertIsNone(read_cached_parquet(path, **kwargs))
            self.assertIsNone(read_existing_run_parquet(path))
            self.assertFalse(reusable_artifacts([path], **kwargs))
            self.assertTrue(reusable_artifacts([path], allow_older_results=True, **kwargs))
            self.assertFalse(reusable_artifacts([path], allow_older_results=True, force_recompute=True, **kwargs))
            self.assertFalse(reusable_artifacts([path, path.with_name('missing')], allow_older_results=True, **kwargs))
            pd.testing.assert_frame_equal(read_existing_run_parquet(path, reuse_existing=True), frame)
            self.assertIsNone(read_existing_run_parquet(path, reuse_existing=True, required_columns=['missing']))
            self.assertEqual(json.loads(manifest.read_text()), {'version': 1})
            write_signature(manifest, {'version': 2})
            pd.testing.assert_frame_equal(read_cached_parquet(path, **kwargs), frame)
            self.assertTrue(reusable_artifacts([path], **kwargs))

    def test_oracle_computes_then_validates_cache(self):
        memberships = pd.DataFrame({
            'story_id': ['s', 's'], 'comment_id': ['a', 'b'],
            'doc_type': ['comment', 'comment'], 'valid_topic': [True, True],
            'topic_000': [1., 0.], 'topic_001': [0., 1.],
        })
        comments = memberships[['story_id', 'comment_id']]
        article = pd.DataFrame({'story_id': ['s'], 'topic_000': [1.], 'topic_001': [0.]})
        discussion = pd.DataFrame({'story_id': ['s'], 'topic_000': [.5], 'topic_001': [.5]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {key + '_path': root / (key + ('.parquet' if key in ('visible', 'metrics') else '.csv'))
                     for key in ('visible', 'metrics', 'summary', 'skipped', 'exact_validation')}
            kwargs = dict(policy_id='oracle', ordering='oracle', objective='cosine_similarity',
                          structured_start_policies=(), n_starts=1, max_iterations=1,
                          n_perturbations=0, **paths)
            first = run_oracle_benchmark(memberships, comments, article, discussion,
                                         ['topic_000', 'topic_001'], **kwargs)
            self.assertEqual(len(first['metrics']), 1)
            self.assertIsNone(read_existing_oracle(paths, topic_columns=['topic_000', 'topic_001']))
            self.assertIsNotNone(read_existing_oracle(paths, topic_columns=['topic_000', 'topic_001'], reuse_existing=True))
            with patch('commentgap_analysis.topic_policy.hellinger_oracle_order', side_effect=AssertionError('cache missed')):
                cached = run_oracle_benchmark(memberships, comments, article, discussion,
                                             ['topic_000', 'topic_001'], **kwargs)
                pd.testing.assert_frame_equal(first['metrics'], cached['metrics'])
                changed = article.assign(topic_000=0., topic_001=1.)
                with self.assertRaisesRegex(AssertionError, 'cache missed'):
                    run_oracle_benchmark(memberships, comments, changed, discussion,
                                         ['topic_000', 'topic_001'], **kwargs)
            with patch('commentgap_analysis.topic_policy.compute_topic_metrics', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    run_oracle_benchmark(memberships, comments, article, discussion,
                                         ['topic_000', 'topic_001'], force_recompute=True, **kwargs)
            self.assertFalse(paths['visible_path'].with_suffix('.manifest.json').exists())
