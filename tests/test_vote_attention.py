import unittest

import numpy as np
import pandas as pd

from commentgap_analysis.vote_attention import fit_vote_attention


def story(story_id, n, power=0.7):
    ranks = np.arange(1, n + 1)
    return pd.DataFrame({
        'story_id': story_id, 'comment_id': [f'{story_id}-{r}' for r in ranks],
        'is_root': True, 'root_comment_id': [f'{story_id}-{r}' for r in ranks],
        'parent_comment_id': None,
        'created_at': pd.date_range('2025-01-01', periods=n, freq='h')[::-1],
        'preorder_position': ranks, 'display_order': ranks, 'is_sticky': False,
        'votes_positive': 1000 * ranks.astype(float) ** (-power), 'votes_negative': 500 * ranks.astype(float) ** (-power),
    })


class VoteAttentionTests(unittest.TestCase):
    def test_recovers_power_with_unequal_lengths_and_excludes_test(self):
        comments = pd.concat([story('a', 11), story('b', 40), story('test', 30, 5)])
        split = pd.DataFrame({'story_id': ['a', 'b', 'test'], 'split_role': ['development', 'development', 'paper2_test']})
        curve, fit = fit_vote_attention(comments, split)
        self.assertAlmostEqual(fit['rank_weight_power'], 0.7, places=4)
        self.assertEqual(fit['n_stories'], 2)
        self.assertAlmostEqual(curve.vote_fraction.sum(), 1)
        self.assertAlmostEqual(curve.fitted_vote_fraction.sum(), 1)
        np.testing.assert_allclose(curve.vote_fraction, curve.fitted_vote_fraction, rtol=1e-5)
        comments.loc[comments.story_id.eq('test'), 'votes_positive'] = np.nan
        self.assertEqual(fit, fit_vote_attention(comments, split)[1])

    def test_equal_story_averaging_is_invariant_to_story_vote_scale(self):
        comments = pd.concat([story('a', 11), story('b', 40)])
        split = pd.DataFrame({'story_id': ['a', 'b'], 'split_role': ['development', 'development']})
        baseline, baseline_fit = fit_vote_attention(comments, split)
        comments.loc[comments.story_id.eq('b'), ['votes_positive', 'votes_negative']] *= 1000
        scaled, scaled_fit = fit_vote_attention(comments, split)
        np.testing.assert_allclose(baseline.vote_fraction, scaled.vote_fraction)
        self.assertAlmostEqual(baseline_fit['rank_weight_power'], scaled_fit['rank_weight_power'])

    def test_pinning_and_tree_order(self):
        comments = story('a', 3)
        comments.loc[2, 'is_sticky'] = True
        comments.loc[1, ['is_root', 'parent_comment_id', 'root_comment_id']] = [False, 'a-3', 'a-3']
        comments['preorder_position'] = [1, 3, 2]
        comments['votes_positive'] = [1, 2, 3]
        comments['votes_negative'] = [4, 5, 6]
        split = pd.DataFrame({'story_id': ['a'], 'split_role': ['development']})
        curve, _ = fit_vote_attention(comments, split, min_comments=1)
        self.assertEqual(curve.total_votes.tolist(), [9, 5])
        self.assertAlmostEqual(curve.vote_fraction.sum(), 1)

    def test_invalid_and_zero_votes(self):
        comments = story('a', 11)
        split = pd.DataFrame({'story_id': ['a'], 'split_role': ['development']})
        for value in (-1, np.nan, np.inf):
            comments.loc[0, 'votes_negative'] = value
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                fit_vote_attention(comments, split)
        comments[['votes_positive', 'votes_negative']] = 0
        with self.assertRaisesRegex(ValueError, 'with votes'):
            fit_vote_attention(comments, split)


class FittedAttentionModelTests(unittest.TestCase):
    def _data(self, shape='exponential'):
        frames = []
        for i in range(6):
            frame = story(str(i), 30 + 10 * i)
            ranks = np.arange(1, len(frame) + 1)
            weights = (np.exp(-0.06 * (ranks - 1)) if shape == 'exponential' else
                       0.2 + 8 * np.exp(-((np.log(ranks) - np.log(3)) / 0.4) ** 2))
            frame['votes_positive'] = 1000 * weights
            frame['votes_negative'] = 100 * weights
            frames.append(frame)
        split = pd.DataFrame({'story_id': list(map(str, range(6))),
                              'split_role': 'development', 'development_fold': np.arange(6) % 3})
        return pd.concat(frames, ignore_index=True), split

    def test_exponential_recovery_and_development_only_nested_cv(self):
        from commentgap_analysis.vote_attention import fit_vote_attention_models
        comments, split = self._data()
        curve, fit, comparison, cv, tuning = fit_vote_attention_models(comments, split, smoothing_grid=(1e-4,))
        self.assertAlmostEqual(fit['models']['exponential']['parameter'], 0.06, places=5)
        self.assertEqual(fit['primary_model'], 'spline')
        self.assertEqual(set(comparison.model), {'spline', 'power_law', 'exponential'})
        self.assertEqual(len(cv), 9)
        self.assertTrue((comparison.n_validation_stories == 6).all())
        for model in comparison.model:
            self.assertAlmostEqual(curve[f'{model}_fitted_vote_fraction'].sum(), 1.0)
        losses = comparison.set_index('model').cv_cross_entropy
        self.assertLess(losses.exponential, losses.power_law)
        for row in tuning.itertuples():
            if row.outer_fold != 'final':
                self.assertNotEqual(row.outer_fold, row.validation_fold)
        test = story('test', 100)
        test['votes_positive'] = np.nan
        contaminated, _fit, _comparison, _, _ = fit_vote_attention_models(
            pd.concat([comments, test]),
            pd.concat([split, pd.DataFrame({'story_id': ['test'], 'split_role': ['paper2_test'], 'development_fold': [-1]})]),
            smoothing_grid=(1e-4,))
        pd.testing.assert_frame_equal(curve, contaminated)

    def test_spline_captures_nonmonotone_peak(self):
        from commentgap_analysis.vote_attention import fit_vote_attention_models
        comments, split = self._data('peak')
        curve, fit, comparison, _, _ = fit_vote_attention_models(comments, split, smoothing_grid=(1e-5,))
        weights = np.asarray(fit['models']['spline']['rank_weight_values'])
        self.assertGreater(weights[2], weights[0])
        self.assertLess(comparison.set_index('model').cv_cross_entropy.spline,
                        comparison.set_index('model').cv_cross_entropy.power_law)
        self.assertTrue(np.isfinite(weights).all() and (weights > 0).all())

    def test_empirical_weights_and_nonmonotone_oracle(self):
        from commentgap_analysis.topic_policy import (
            _rank_attention_weights, hellinger_oracle_order, exact_hellinger_oracle_order)
        kwargs = {'rank_weight_mode': 'empirical', 'rank_weight_values': [0.1, 1.0, 0.2]}
        np.testing.assert_allclose(_rank_attention_weights(5, **kwargs), [0.1, 1, 0.2, 0.2, 0.2])
        article = [1.0, 0.0]
        comments = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        for objective in ('cosine_similarity', 'jensen_shannon_distance'):
            heuristic = hellinger_oracle_order(article, comments, objective=objective, **kwargs)
            exact = exact_hellinger_oracle_order(article, comments, objective=objective, **kwargs)
            self.assertEqual(int(heuristic['order'][1]), 0)
            np.testing.assert_allclose(heuristic['distribution'], exact['distribution'])
        for values in ([], [0, 1], [np.nan], [-1]):
            with self.assertRaisesRegex(ValueError, 'positive vector'):
                _rank_attention_weights(3, rank_weight_mode='empirical', rank_weight_values=values)
