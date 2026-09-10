import json
from pathlib import Path
import tempfile
import unittest

from commentgap_analysis.topic_modeling import TopicModelConfig
from scripts.run_topic_model_fit import _parser, _write_progress


class TopicMetadataTests(unittest.TestCase):
    def test_metadata_writer_round_trips_json(self):
        metadata = {"counts": {"training_articles": 3}, "configuration": {"seed": 2025}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topic_model_run_metadata.json"
            _write_progress(path, metadata)
            self.assertEqual(json.loads(path.read_text()), metadata)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_probability_calculation_is_explicit_and_opt_in(self):
        self.assertFalse(TopicModelConfig().calculate_probabilities)
        self.assertTrue(TopicModelConfig(calculate_probabilities=True).calculate_probabilities)
        args = _parser().parse_args(["--calculate-probabilities"])
        self.assertTrue(args.calculate_probabilities)

    def test_configuration_file_controls_fit_and_probability_mode(self):
        from scripts.run_topic_model_fit import _model_config
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps(dict(hdbscan_min_cluster_size=15,
                                                  umap_n_neighbors=5, random_state=2027, hdbscan_min_samples=1,
                                                  calculate_probabilities=True,
                                                  ngram_range=[1, 2])))
            args = _parser().parse_args(['--model-config', str(config_path), '--runs-root', directory])
            config = _model_config(args)
            self.assertEqual(config.hdbscan_min_cluster_size, 15)
            self.assertEqual(config.umap_n_neighbors, 5)
            self.assertEqual(config.random_state, 2027)
            self.assertEqual(config.hdbscan_min_samples, 1)
            self.assertTrue(config.calculate_probabilities)
            self.assertEqual(config.ngram_range, (1, 2))

    def test_exact_fit_sample_rejects_test_stories(self):
        import pandas as pd
        from scripts.run_topic_model_fit import _load_fit_comment_documents
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / 'split.parquet'
            sample = root / 'sample.parquet'
            pd.DataFrame({'story_id': ['dev', 'test'], 'split_role': ['development', 'paper2_test']}).to_parquet(split)
            data = pd.DataFrame({'story_id': ['dev'], 'comment_id': ['1'],
                                 'doc_id': ['comment:dev:1'], 'doc_type': ['comment'], 'text': ['sample']})
            data.to_parquet(sample)
            self.assertEqual(_load_fit_comment_documents(sample, split).doc_id.tolist(), ['comment:dev:1'])
            data['story_id'] = 'test'; data['doc_id'] = 'comment:test:1'; data.to_parquet(sample)
            with self.assertRaisesRegex(ValueError, 'development'):
                _load_fit_comment_documents(sample, split)
