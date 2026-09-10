from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from commentgap_analysis.topic_modeling import TopicModelConfig
from commentgap_analysis.topic_runs import (
    complete_run, list_runs, prepare_run, resolve_run, select_run,
)


class TopicRunTests(unittest.TestCase):
    def test_setup_isolation_reuse_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setup = dict(configuration=asdict(TopicModelConfig()), inputs={'version': 1})
            first, manifest = prepare_run(root, setup)
            self.assertEqual(manifest['status'], 'running')
            self.assertEqual(prepare_run(root, setup)[0], first)
            with self.assertRaises(ValueError):
                select_run(root, first.name)
            complete_run(first)
            with self.assertRaises(FileNotFoundError):
                select_run(root, first.name)
            for name in ['topic_model.joblib', 'topic_model_manifest.json',
                         'topic_model_run_metadata.json', 'document_topic_memberships.parquet',
                         'article_topic_coverage.parquet']:
                (first / name).touch()
            select_run(root, first.name)
            self.assertEqual(resolve_run(root), first)
            for key, value in [('random_state', 2026), ('hdbscan_min_cluster_size', 15),
                               ('umap_n_neighbors', 5), ('calculate_probabilities', True)]:
                changed = json.loads(json.dumps(setup))
                changed['configuration'][key] = value
                second, _ = prepare_run(root, changed)
                self.assertNotEqual(first, second)
                self.assertEqual(resolve_run(root), first)
            changed = dict(setup, inputs={'version': 2})
            self.assertNotEqual(prepare_run(root, changed)[0], first)
            self.assertEqual(len(list_runs(root)), 6)

    def test_missing_selection_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_run(Path(directory))
            for value in ['../outside', '.', '/tmp/run']:
                with self.assertRaises(ValueError):
                    resolve_run(Path(directory), value)
