"""Paper 2 sampling and real BERTopic outlier-assignment contract."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from commentgap_analysis.topic_modeling import TopicModelConfig, TopicModelBundle, topic_stopwords
from commentgap_analysis.topic_diagnostics import load_diagnostic_corpus
from scripts.run_topic_model_fit import _load_fit_comment_documents


class Paper2Tests(unittest.TestCase):
    def test_real_reduce_outliers_retains_rejections_and_original_labels(self):
        from bertopic import BERTopic
        model = BERTopic()
        model.topic_sizes_ = {-1: 1, 0: 2, 1: 2}
        model.topic_embeddings_ = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        bundle = TopicModelBundle(TopicModelConfig(outlier_threshold=0.5), None, model,
                                  (('alpha',), ('beta',)), (0, 1))
        docs = pd.DataFrame(dict(doc_id=['a', 'b', 'c'], story_id='s', doc_type='comment', text='text'))
        vectors = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        with patch.object(model, 'transform', return_value=([-1, -1, 1], np.array([0., 0., 1.]))):
            result = bundle.transform(docs, embeddings=vectors)
        self.assertEqual(result.topic_assignment.tolist(), [0, -1, 1])
        self.assertEqual(result.raw_topic_assignment.tolist(), [-1, -1, 1])
        self.assertEqual(result.outlier_reassigned.tolist(), [True, False, False])
        np.testing.assert_equal(result[['topic_000', 'topic_001']].to_numpy(), [[1, 0], [0, 0], [0, 1]])
        # Prediction may produce noise even when the fitted model had none.
        model.topic_sizes_ = {0: 2, 1: 2}
        model.topic_embeddings_ = model.topic_embeddings_[1:]
        self.assertEqual(bundle.reduce_outlier_labels(docs.text.tolist(), [-1, -1, 1], vectors).tolist(), [0, -1, 1])
        self.assertEqual(bundle.reduce_outlier_labels(['x'], [0], vectors[:1]).tolist(), [0])

    def test_stopwords_removed_before_ngram_representation(self):
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(stop_words=topic_stopwords(), strip_accents='unicode', ngram_range=(1, 2))
        terms = vectorizer.fit(['und für über the and solar energie']).get_feature_names_out().tolist()
        self.assertEqual(terms, ['energie', 'solar', 'solar energie'])

    def test_paper2_sample_is_equal_sized_eligible_and_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'articles').mkdir(); (root / 'comments').mkdir()
            pd.DataFrame(dict(story_id=['dev', 'a', 'b'], year=2025, title='title', subtitle='', body='body')).to_parquet(root / 'articles/data.parquet')
            comments = pd.DataFrame([dict(story_id=story, comment_id=str(i), year=2025,
                effective_text='comment', text='comment', lifecycle_status='published')
                for story, count in [('dev', 5), ('a', 10), ('b', 3)] for i in range(count)])
            comments.to_parquet(root / 'comments/data.parquet')
            split = root / 'split.parquet'
            pd.DataFrame(dict(story_id=['dev', 'a', 'b'], split_role=['development', 'paper2_test', 'paper2_test'])).to_parquet(split)
            eligible = root / 'eligible.parquet'
            comments.loc[comments.story_id.ne('dev') & comments.comment_id.ne('0'), ['story_id', 'comment_id']].to_parquet(eligible)
            def passages(articles):
                return pd.DataFrame([dict(doc_id=f'article:{story}:{i}', story_id=story, doc_type='article', text='text')
                    for story in articles.story_id for i in range(2)])
            with patch('commentgap_analysis.topic_modeling.prepare_article_passages', side_effect=passages), patch(
                'commentgap_analysis.topic_modeling.load_precomputed_embeddings', side_effect=lambda docs, _: np.ones((len(docs), 3))):
                corpus = load_diagnostic_corpus(root, split, root, 2025, fit_role='paper2_test', analysis_comments_path=eligible)
            sample = corpus[('paper2_test', 'comment')][0]
            self.assertEqual(len(sample), len(corpus[('paper2_test', 'article_passage')][0]))
            self.assertEqual(sample.groupby('story_id').size().to_dict(), {'a': 2, 'b': 2})
            self.assertNotIn('0', sample.comment_id.tolist())
            sample_path = root / 'sample.parquet'; sample.to_parquet(sample_path)
            self.assertEqual(len(_load_fit_comment_documents(sample_path, split, 'paper2_test')), 4)
            with self.assertRaises(ValueError):
                _load_fit_comment_documents(sample_path, split, 'development')
