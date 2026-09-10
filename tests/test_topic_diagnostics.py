from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from commentgap_analysis.topic_diagnostics import (
    corpus_signature, coverage_metrics, diagnostic_fit, load_search_artifacts,
    search_cache_matches, stability_pairs,
)
from commentgap_analysis.topic_diagnostics import pooled_stability_pairs
from commentgap_analysis.topic_modeling import TopicModelConfig


class StabilityTests(unittest.TestCase):
    def frame(self, a, b):
        return pd.DataFrame(dict(doc_id=[str(i) for i in range(len(a))],
                                 doc_type='article_passage', a=a, b=b))

    def test_label_permutations_are_identical(self):
        row = stability_pairs(self.frame([0, 0, 1, 1, -1], [5, 5, 3, 3, -1]), ['a', 'b']).iloc[0]
        for metric in ['ami_all', 'ami_common', 'assigned_jaccard']:
            self.assertAlmostEqual(row[metric], 1)
        self.assertEqual(row.common_coverage_pct, 80)

    def test_changing_outliers_is_visible_despite_stable_common_subset(self):
        row = stability_pairs(self.frame([0, 0, 1, 1, -1, 0], [3, 3, 7, 7, 7, -1]), ['a', 'b']).iloc[0]
        self.assertAlmostEqual(row.ami_common, 1)
        self.assertAlmostEqual(row.assigned_jaccard, 4 / 6)
        self.assertLess(row.ami_all, 1)

    def test_all_outliers_and_single_cluster_are_not_perfect_stability(self):
        for labels in [[-1] * 4, [0] * 4]:
            row = stability_pairs(self.frame(labels, labels), ['a', 'b']).iloc[0]
            self.assertTrue(np.isnan(row.ami_common))
            self.assertTrue(np.isnan(row.ami_all))

    def test_missing_or_duplicate_documents_rejected(self):
        frame = self.frame([0, 1], [0, np.nan])
        with self.assertRaises(ValueError):
            stability_pairs(frame, ['a', 'b'])
        frame = self.frame([0, 1], [0, 1]); frame['doc_id'] = 'same'
        with self.assertRaises(ValueError):
            stability_pairs(frame, ['a', 'b'])

    def test_coverage_excludes_outliers(self):
        metrics = coverage_metrics([-1, 2, 2, 3])
        self.assertEqual(metrics['substantive_topics'], 2)
        self.assertEqual(metrics['assigned_documents'], 3)
        self.assertEqual(metrics['coverage_pct'], 75)

    def test_search_artifact_loader_can_select_a_complete_older_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'old-search'
            root.mkdir()
            configuration = {'fit_role': 'old-role', 'corpus_sha256': 'old-signature'}
            (root / 'configuration.json').write_text(json.dumps(configuration))
            for filename in ['seeds.csv', 'pairs.csv', 'pooled_pairs.csv', 'summary.csv']:
                pd.DataFrame({'value': [1], 'ari_common': [0.5]}).to_csv(root / filename, index=False)

            self.assertTrue(search_cache_matches(root, configuration))
            self.assertFalse(search_cache_matches(root, {'fit_role': 'current-role'}))
            loaded = load_search_artifacts(root)
            self.assertEqual(loaded['search_root'], root)
            self.assertEqual(loaded['cache_configuration'], configuration)
            self.assertEqual(loaded['search_report'].value.tolist(), [1])
            self.assertNotIn('ari_common', loaded['search_report'])

    def test_diagnostic_cache_reuses_only_matching_seed_and_data(self):
        documents = pd.DataFrame(dict(doc_id=['a', 'b', 'c', 'd'], story_id=['s'] * 4,
                                      doc_type='article', text=['text'] * 4))
        corpus = {('development', 'article_passage'): (documents, np.eye(4, dtype=np.float32))}
        signature = corpus_signature(corpus)
        bundle = SimpleNamespace(model=SimpleNamespace(topics_=[0, 0, 1, -1],
                                 hdbscan_model=SimpleNamespace(labels_=[0, 0, 1, -1])), topic_ids=(0, 1))
        with tempfile.TemporaryDirectory() as directory, \
             patch('commentgap_analysis.topic_modeling.fit_topic_model', return_value=bundle) as fit, \
             patch('commentgap_analysis.topic_modeling.topic_model_quality_table', return_value=pd.DataFrame({'topic': [0, 1]})):
            config = TopicModelConfig()
            for _ in range(2):
                summary, labels = diagnostic_fit(corpus, config, Path(directory), signature)
                self.assertEqual(summary.iloc[0].assigned_documents, 3)
                self.assertEqual(labels.topic_assignment.tolist(), [0, 0, 1, -1])
            self.assertEqual(fit.call_count, 1)
            diagnostic_fit(corpus, replace(config, random_state=2026), Path(directory), signature)
            self.assertEqual(fit.call_count, 2)
        documents.loc[0, 'text'] = 'changed'
        self.assertNotEqual(signature, corpus_signature(corpus))

    def test_loader_excludes_test_and_unusable_comments_by_default(self):
        from commentgap_analysis.topic_diagnostics import load_diagnostic_corpus
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'articles').mkdir(); (root / 'comments').mkdir()
            pd.DataFrame(dict(story_id=['dev', 'test'], year=[2025, 2025],
                              title=['development', 'held out'], subtitle=['', ''], body=['body', 'body']
                              )).to_parquet(root / 'articles' / 'data.parquet')
            pd.DataFrame(dict(story_id=['dev', 'dev', 'test'], comment_id=['1', '2', '3'],
                              year=[2025] * 3, lifecycle_status=['published'] * 3,
                              effective_text=['usable', '', 'test'], text=['usable', '', 'test']
                              )).to_parquet(root / 'comments' / 'data.parquet')
            split_path = root / 'split.parquet'
            pd.DataFrame(dict(story_id=['dev', 'test'], split_role=['development', 'paper2_test']
                              )).to_parquet(split_path)
            def passages(articles):
                return articles[['story_id']].assign(doc_id='article:' + articles.story_id,
                                                      doc_type='article', text=articles.title)
            with patch('commentgap_analysis.topic_modeling.prepare_article_passages', side_effect=passages), \
                 patch('commentgap_analysis.topic_modeling.load_precomputed_embeddings',
                       side_effect=lambda docs, _: np.ones((len(docs), 3), dtype=np.float32)):
                corpus = load_diagnostic_corpus(root, split_path, root, 2025)
                self.assertEqual(set(corpus), {('development', 'article_passage'), ('development', 'comment')})
                for documents, _ in corpus.values():
                    self.assertEqual(documents.story_id.tolist(), ['dev'])
                self.assertEqual(corpus[('development', 'comment')][0].comment_id.tolist(), ['1'])

    def test_union_stability_ignores_shared_outliers_but_keeps_assignment_changes(self):
        from sklearn.metrics import adjusted_mutual_info_score
        a, b = [0, 0, 1, 1, -1, 0, -1], [3, 3, 7, 7, 7, -1, -1]
        row = stability_pairs(self.frame(a, b), ['a', 'b']).iloc[0]
        self.assertAlmostEqual(row.ami_union, adjusted_mutual_info_score(a[:-1], b[:-1]))
        self.assertLess(row.ami_union, row.ami_common)

    def test_pooled_stability_combines_article_and_comment_rows(self):
        assignments = pd.DataFrame({
            'doc_id': ['a1', 'a2', 'c1', 'c2'],
            'doc_type': ['article_passage', 'article_passage', 'comment', 'comment'],
            'a': [0, 0, 1, -1],
            'b': [5, 5, 3, 3],
        })
        row = pooled_stability_pairs(assignments, ['a', 'b']).iloc[0]
        self.assertEqual(row.doc_type, 'article_and_comment')
        self.assertEqual(row.documents, 4)
        self.assertEqual(row.common_documents, 3)
        self.assertTrue(np.isfinite(row.ami_all))

    def test_min_samples_validation(self):
        self.assertIsNone(TopicModelConfig().hdbscan_min_samples)
        self.assertEqual(TopicModelConfig(hdbscan_min_samples=1).hdbscan_min_samples, 1)
        with self.assertRaises(ValueError):
            TopicModelConfig(hdbscan_min_samples=0)

    def test_mixed_fit_uses_exact_sample_and_aligns_labels_and_embeddings(self):
        from unittest.mock import Mock
        articles = pd.DataFrame(dict(doc_id=['article:b', 'article:a'], story_id=['b', 'a'],
                                     doc_type='article', text=['b', 'a']))
        comments = pd.DataFrame(dict(doc_id=['comment:b:2', 'comment:a:1'], story_id=['b', 'a'],
                                     comment_id=['2', '1'], doc_type='comment', text=['two', 'one']))
        corpus = {('development', 'article_passage'): (articles, np.array([[2.], [1.]])),
                  ('development', 'comment'): (comments, np.array([[4.], [3.]]))}
        transform = Mock(return_value=pd.DataFrame({'topic_assignment': [-1, -1]}))
        def fit_model(documents, config, embeddings):
            # Vector values identify source rows even after sorting the mixed corpus.
            expected = {'article:a': 1, 'article:b': 2, 'comment:a:1': 3, 'comment:b:2': 4}
            self.assertEqual(embeddings[:, 0].tolist(), [expected[d] for d in documents.doc_id])
            return SimpleNamespace(model=SimpleNamespace(topics_=embeddings[:, 0].astype(int),
                                   hdbscan_model=SimpleNamespace(labels_=[0] * len(documents))),
                                   topic_ids=(1, 2, 3, 4), transform=transform)
        with tempfile.TemporaryDirectory() as directory, \
             patch('commentgap_analysis.topic_modeling.fit_topic_model', side_effect=fit_model) as fit, \
             patch('commentgap_analysis.topic_modeling.topic_model_quality_table', return_value=pd.DataFrame({'topic': [1]})):
            root = Path(directory)
            summary, labels = diagnostic_fit(corpus, TopicModelConfig(), root, 'same', fit_corpus='articles_comments')
            self.assertEqual(fit.call_args.args[0].doc_id.tolist(), sorted(articles.doc_id.tolist() + comments.doc_id.tolist()))
            self.assertEqual(labels.topic_assignment.tolist(), [2, 1, 4, 3])
            self.assertTrue(summary.label_source.eq('fit').all())
            self.assertTrue(summary.training_documents.eq(4).all())
            transform.assert_not_called()
            diagnostic_fit(corpus, TopicModelConfig(), root, 'same', fit_corpus='articles_comments')
            self.assertEqual(fit.call_count, 1)
            article_summary, _ = diagnostic_fit(corpus, TopicModelConfig(), root, 'same', fit_corpus='articles')
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(len(fit.call_args.args[0]), 2)
            self.assertEqual(article_summary.loc[article_summary.doc_type.eq('comment'), 'label_source'].iloc[0], 'transform')
