"""Development-only voting-activity proxy for rank attention."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import BSpline

from .forum_scores import PolicySpec, make_policy_order


def _prepare_vote_attention(comments, article_split, *, min_comments=11):
    """Fit p(rank | story length) proportional to rank**(-power).

    Each eligible story first contributes its within-story vote fractions;
    those fractions are then averaged equally across stories. The likelihood
    conditions on each story's visible length, so ranks absent from short
    stories contribute zero to the across-story curve.
    Pinned descendants are excluded from ranks and vote totals.
    Zero-vote stories have no within-story fraction and are excluded from the
    fit; their count is reported in the metadata.
    """
    split = article_split[['story_id', 'split_role']].copy()
    split['story_id'] = split.story_id.astype(str)
    if split.story_id.duplicated().any():
        raise ValueError('Duplicate split story IDs')
    development = set(split.loc[split.split_role.eq('development'), 'story_id'])
    frame = comments.copy()
    frame['story_id'] = frame.story_id.astype(str)
    frame = frame.loc[frame.story_id.isin(development)].copy()
    if frame.duplicated(['story_id', 'comment_id']).any():
        raise ValueError('Duplicate development comment keys')
    frame = frame.loc[frame.groupby('story_id').story_id.transform('size').ge(min_comments)]
    votes = frame[['votes_positive', 'votes_negative']].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(votes.to_numpy()).all() or (votes < 0).any().any():
        raise ValueError('Vote counts must be finite and nonnegative')
    frame['total_votes'] = votes.sum(axis=1)
    frame['relative_votes'] = votes.votes_positive - votes.votes_negative
    rows, lengths, totals, story_ids = [], [], [], []
    n_visible_comments = 0
    n_zero_vote_stories = 0
    spec = PolicySpec('reverse_chronological', 'trees', True)
    for story_id, story in frame.groupby('story_id', sort=True):
        story = story.reset_index(drop=True)
        order = make_policy_order(story, spec)
        ranked = story.iloc[order]
        n_visible_comments += len(ranked)
        story_total_votes = float(ranked.total_votes.sum())
        if story_total_votes <= 0:
            n_zero_vote_stories += 1
            continue
        within_story_fraction = ranked.total_votes.to_numpy(dtype=float) / story_total_votes
        rows.append(pd.DataFrame({
            'rank': np.arange(1, len(ranked) + 1),
            'upvotes': ranked.votes_positive.to_numpy(dtype=float),
            'downvotes': ranked.votes_negative.to_numpy(dtype=float),
            'within_story_vote_fraction': within_story_fraction,
        }))
        story_ids.append(str(story_id))
        lengths.append(len(ranked))
        totals.append(story_total_votes)
    if not rows:
        raise ValueError('No eligible development stories with votes')
    observations = pd.concat(rows, ignore_index=True)
    n_stories = len(lengths)
    curve = observations.groupby('rank').agg(
        upvotes=('upvotes', 'sum'), downvotes=('downvotes', 'sum'),
        n_stories_at_rank=('upvotes', 'size'),
        within_story_vote_fraction=('within_story_vote_fraction', 'sum'),
    ).reset_index()
    curve['total_votes'] = curve.upvotes + curve.downvotes
    curve['vote_fraction'] = curve.within_story_vote_fraction / n_stories
    curve['mean_story_vote_fraction'] = curve.vote_fraction
    metadata = {
        'split_role': 'development', 'policy_id': spec.policy_id,
        'vote_measure': 'votes_positive + votes_negative',
        'fit': 'conditional multinomial power law; equal-story weighting',
        'aggregation': 'mean within-story vote fraction; absent ranks contribute zero',
        'n_stories': n_stories, 'n_comments': int(sum(lengths)),
        'n_input_comments': int(len(frame)),
        'n_hidden_comments': int(len(frame) - n_visible_comments),
        'visibility': 'pinned comments hide all descendants; visible votes only',
        'n_zero_vote_stories': n_zero_vote_stories,
        'total_votes': float(sum(totals)), 'min_comments': min_comments,
        'max_development_rank': int(max(lengths)),
    }
    fractions = [row.within_story_vote_fraction.to_numpy() for row in rows]
    return curve, metadata, fractions, np.asarray(lengths), story_ids


def _statistics(fractions, lengths, indices, n_ranks):
    if not len(indices):
        raise ValueError('Each training and validation partition must contain stories')
    observed = np.zeros(n_ranks)
    for index in indices:
        observed[:lengths[index]] += fractions[index]
    histogram = np.bincount(lengths[indices], minlength=n_ranks + 1)[1:].astype(float)
    return observed / len(indices), histogram / len(indices)


def _cross_entropy(log_weights, stats, *, gradient=False):
    """Equal-story conditional multinomial loss, stable under arbitrary scale."""
    observed, lengths = stats
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    normalizers = np.cumsum(weights)
    loss = float(lengths @ np.log(normalizers) - observed @ shifted)
    if not gradient:
        return loss
    predicted = weights * np.cumsum((lengths / normalizers)[::-1])[::-1]
    return loss, predicted - observed


def _fit_parametric(stats, mode):
    ranks = np.arange(1, len(stats[0]) + 1, dtype=float)
    x = np.log(ranks) if mode == 'power_law' else ranks - 1
    # Strictly positive parameters remain compatible with the ranking engine.
    bounds = (1e-8, 10.0 if mode == 'power_law' else 1.0)
    result = minimize_scalar(lambda value: _cross_entropy(-value * x, stats),
                             bounds=bounds, method='bounded', options={'xatol': 1e-10})
    if not result.success:
        raise RuntimeError(f'{mode} fit failed: {result.message}')
    return -float(result.x) * x, {
        'parameter': float(result.x), 'bounds': list(bounds),
        'fit_at_boundary': bool(min(result.x - bounds[0], bounds[1] - result.x) < 1e-6),
    }


def fit_vote_attention(comments, article_split, *, min_comments=11):
    """Compatibility entry point for the equal-story fitted power law."""
    curve, metadata, fractions, lengths, _ = _prepare_vote_attention(
        comments, article_split, min_comments=min_comments)
    stats = _statistics(fractions, lengths, np.arange(len(lengths)), len(curve))
    log_weights, fit = _fit_parametric(stats, 'power_law')
    _, gradient = _cross_entropy(log_weights, stats, gradient=True)
    curve['fitted_vote_fraction'] = gradient + stats[0]
    curve['relative_rank_weight'] = np.exp(log_weights)
    metadata.update(rank_weight_power=fit['parameter'], power_bounds=fit['bounds'],
                    fit_at_boundary=fit['fit_at_boundary'])
    return curve, metadata


# Fixed before looking at held-out stories: denser knots resolve the early peak.
SPLINE_KNOT_RANKS = (1, 2, 3, 4, 5, 6, 8, 12, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10000)
SMOOTHING_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _spline_basis(n_ranks):
    knots = np.log(SPLINE_KNOT_RANKS)
    if n_ranks > SPLINE_KNOT_RANKS[-1]:
        raise ValueError('Development ranks exceed the predeclared spline domain')
    t = np.r_[np.repeat(knots[0], 3), knots, np.repeat(knots[-1], 3)]
    basis = BSpline.design_matrix(np.log(np.arange(1, n_ranks + 1)), t, 3).toarray()
    # Fix the first coefficient at zero: a common intercept cancels in p(r|L).
    penalty = np.diff(np.eye(basis.shape[1]), n=2, axis=0)[:, 1:]
    return basis[:, 1:], penalty


def _fit_spline(stats, basis, penalty, smoothing):
    def objective(coefficients):
        loss, grad = _cross_entropy(basis @ coefficients, stats, gradient=True)
        roughness = penalty @ coefficients
        return (loss + smoothing * float(roughness @ roughness),
                basis.T @ grad + 2 * smoothing * penalty.T @ roughness)
    result = minimize(objective, np.zeros(basis.shape[1]), jac=True, method='L-BFGS-B',
                      bounds=[(-30, 30)] * basis.shape[1],
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-7})
    if not result.success:
        raise RuntimeError(f'Spline fit failed: {result.message}')
    return basis @ result.x, result.x


def fit_vote_attention_models(comments, article_split, *, min_comments=11,
                              smoothing_grid=SMOOTHING_GRID):
    """Compare three models by nested CV over frozen development story folds.

    Spline smoothing is selected inside each outer training fold; outer losses
    therefore evaluate the selection procedure without reuse of validation
    outcomes. Final smoothing is selected by CV over all development folds.
    No Paper 2 rows enter preparation, fitting, or hyperparameter selection.
    """
    curve, metadata, fractions, lengths, story_ids = _prepare_vote_attention(
        comments, article_split, min_comments=min_comments)
    if 'development_fold' not in article_split:
        raise ValueError('Frozen development_fold is required for model comparison')
    split = article_split.copy()
    split['story_id'] = split.story_id.astype(str)
    folds = pd.to_numeric(split.set_index('story_id').loc[story_ids, 'development_fold'], errors='raise').to_numpy()
    unique_folds = np.unique(folds)
    if len(unique_folds) < 3 or not np.isfinite(folds).all() or (folds < 0).any():
        raise ValueError('At least three nonnegative development folds are required')
    if not len(smoothing_grid) or any(not np.isfinite(x) or x <= 0 for x in smoothing_grid):
        raise ValueError('Smoothing strengths must be finite and positive')
    basis, penalty = _spline_basis(len(curve))
    all_indices = np.arange(len(lengths))
    statistics = lambda indices: _statistics(fractions, lengths, indices, len(curve))
    cv_rows, tuning_rows = [], []

    def select_smoothing(indices, outer_fold):
        for smoothing in smoothing_grid:
            for validation_fold in np.unique(folds[indices]):
                train = indices[folds[indices] != validation_fold]
                validation = indices[folds[indices] == validation_fold]
                log_weights, _ = _fit_spline(statistics(train), basis, penalty, smoothing)
                tuning_rows.append({'outer_fold': outer_fold, 'validation_fold': int(validation_fold),
                                    'smoothing': float(smoothing), 'n_stories': len(validation),
                                    'cross_entropy': _cross_entropy(log_weights, statistics(validation))})
        rows = pd.DataFrame([row for row in tuning_rows if row['outer_fold'] == outer_fold])
        rows['weighted_loss'] = rows.cross_entropy * rows.n_stories
        scores = rows.groupby('smoothing').weighted_loss.sum() / rows.groupby('smoothing').n_stories.sum()
        return float(scores.idxmin())

    for fold in unique_folds:
        train = all_indices[folds != fold]
        validation = all_indices[folds == fold]
        train_stats, validation_stats = statistics(train), statistics(validation)
        smoothing = select_smoothing(train, int(fold))
        spline, _ = _fit_spline(train_stats, basis, penalty, smoothing)
        for model in ('spline', 'power_law', 'exponential'):
            log_weights = spline if model == 'spline' else _fit_parametric(train_stats, model)[0]
            cv_rows.append({'model': model, 'fold': int(fold), 'n_stories': len(validation),
                            'cross_entropy': _cross_entropy(log_weights, validation_stats),
                            'smoothing': smoothing if model == 'spline' else None})
        print(f'Vote attention: completed outer development fold {int(fold)}', flush=True)

    smoothing = select_smoothing(all_indices, 'final')
    stats = statistics(all_indices)
    spline, coefficients = _fit_spline(stats, basis, penalty, smoothing)
    models = {}
    for model in ('spline', 'power_law', 'exponential'):
        if model == 'spline':
            log_weights = spline
            config = {'rank_weight_mode': 'empirical', 'rank_weight_power': 1.0, 'rank_decay': 0.1,
                      'rank_weight_values': np.exp(log_weights - log_weights.max()).tolist(),
                      'smoothing': smoothing, 'coefficients': coefficients.tolist(),
                      'knot_ranks': list(SPLINE_KNOT_RANKS),
                      'extrapolation': 'constant last development rank weight'}
        else:
            log_weights, fit = _fit_parametric(stats, model)
            config = {'rank_weight_mode': 'inverse_power' if model == 'power_law' else 'exponential',
                      'rank_weight_power': fit['parameter'] if model == 'power_law' else 1.0,
                      'rank_decay': fit['parameter'] if model == 'exponential' else 0.1,
                      'rank_weight_values': None, **fit}
        _, gradient = _cross_entropy(log_weights, stats, gradient=True)
        curve[f'{model}_fitted_vote_fraction'] = stats[0] + gradient
        curve[f'{model}_rank_weight'] = np.exp(log_weights - log_weights.max())
        config['training_cross_entropy'] = _cross_entropy(log_weights, stats)
        models[model] = config
    cv = pd.DataFrame(cv_rows)
    comparison = []
    for model, group in cv.groupby('model', sort=False):
        comparison.append({'model': model, 'role': 'primary' if model == 'spline' else 'sensitivity',
                           'cv_cross_entropy': float(np.average(group.cross_entropy, weights=group.n_stories)),
                           'training_cross_entropy': models[model]['training_cross_entropy'],
                           'n_validation_stories': int(group.n_stories.sum())})
    comparison = pd.DataFrame(comparison)
    baseline = comparison.loc[comparison.model.eq('power_law'), 'cv_cross_entropy'].iloc[0]
    comparison['cv_delta_vs_power_law'] = comparison.cv_cross_entropy - baseline
    metadata.update(fit='conditional multinomial; nested development CV; equal-story weighting',
                    primary_model='spline', models=models, smoothing_grid=list(smoothing_grid),
                    comparison_metric='mean per-story cross entropy (nats); lower is better',
                    extrapolation='spline holds last weight constant beyond development support; parametric families continue')
    return curve, metadata, comparison, cv, pd.DataFrame(tuning_rows)
