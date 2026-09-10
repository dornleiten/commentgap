"""Clustering diagnostics on an explicitly selected corpus."""
import json
from itertools import combinations
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


_SEARCH_ARTIFACT_FILES = (
    'configuration.json',
    'seeds.csv',
    'pairs.csv',
    'pooled_pairs.csv',
    'summary.csv',
)


def search_cache_matches(path, expected_cache: Mapping[str, object]) -> bool:
    """Return whether a completed search cache matches its expected metadata.

    This checks only the cache metadata and the presence of all search-level
    outputs.  It deliberately does not inspect the current corpus or load any
    embeddings, so it is safe to use while selecting an existing search run.
    """
    path = Path(path)
    if any(not (path / filename).exists() for filename in _SEARCH_ARTIFACT_FILES):
        return False
    try:
        cached = json.loads((path / 'configuration.json').read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return all(cached.get(key) == value for key, value in expected_cache.items())


def _without_ari_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove legacy ARI metrics from persisted search tables."""
    def is_ari(column: object) -> bool:
        name = str(column).lower()
        return name.startswith('ari_') or name.endswith('_ari') or '_ari_' in name

    return frame.loc[:, [not is_ari(column) for column in frame.columns]]


def load_search_artifacts(search_root, *, diagnostic_root=None) -> dict[str, object]:
    """Load the persisted outputs of a completed topic-model search.

    ``search_root`` may be selected explicitly, which allows an older
    completed run to be analysed even when its metadata no longer matches the
    current search grid or code hashes.  Completeness is still checked by
    requiring every search-level artifact written by the search cell.
    """
    search_root = Path(search_root)
    missing = [filename for filename in _SEARCH_ARTIFACT_FILES
               if not (search_root / filename).exists()]
    if missing:
        raise FileNotFoundError(
            f'Incomplete topic search at {search_root}; missing: {", ".join(missing)}'
        )

    cache_configuration = json.loads((search_root / 'configuration.json').read_text())
    diagnostic_signature = cache_configuration.get('corpus_sha256')
    fit_comment_documents_path = None
    if diagnostic_root is not None and diagnostic_signature:
        candidate = (Path(diagnostic_root) / 'corpora' / diagnostic_signature /
                     'fit_comment_documents.parquet')
        if candidate.exists():
            fit_comment_documents_path = candidate

    return {
        'search_root': search_root,
        'cache_configuration': cache_configuration,
        'diagnostic_signature': diagnostic_signature,
        'fit_comment_documents_path': fit_comment_documents_path,
        'search_seeds': _without_ari_columns(pd.read_csv(search_root / 'seeds.csv')),
        'search_pairs': _without_ari_columns(pd.read_csv(search_root / 'pairs.csv')),
        'search_pooled_pairs': _without_ari_columns(pd.read_csv(search_root / 'pooled_pairs.csv')),
        'search_report': _without_ari_columns(pd.read_csv(search_root / 'summary.csv')),
    }


def coverage_metrics(labels) -> dict:
    labels = np.asarray(labels)
    valid = labels != -1
    return dict(documents=len(labels), assigned_documents=int(valid.sum()),
                outlier_documents=int((~valid).sum()),
                coverage_pct=float(100 * valid.mean()) if len(labels) else np.nan,
                substantive_topics=len(set(labels) - {-1}))


def stability_pairs(assignments: pd.DataFrame, label_columns: list[str]) -> pd.DataFrame:
    """Pairwise seed agreement on aligned rows, split by document type.

    Outlier-inclusive scores treat -1 as one cluster. Common-substantive scores
    condition on assignment in both runs and must be read with their coverage.
    Degenerate partitions are reported as NaN rather than perfect stability.
    """
    from sklearn.metrics import adjusted_mutual_info_score

    if assignments['doc_id'].duplicated().any():
        raise ValueError('Assignments must have unique document IDs')
    if len(label_columns) < 2 or len(set(label_columns)) != len(label_columns):
        raise ValueError('Provide at least two distinct seed label columns')
    if assignments[label_columns].isna().any().any():
        raise ValueError('All runs must label the same documents')
    rows = []
    for doc_type, group in assignments.groupby('doc_type'):
        for first, second in combinations(label_columns, 2):
            a, b = (group[c].to_numpy(dtype=int) for c in (first, second))
            both = (a != -1) & (b != -1)
            either = (a != -1) | (b != -1)
            row = dict(doc_type=doc_type, first=first, second=second, documents=len(a),
                       first_coverage_pct=100 * (a != -1).mean(),
                       second_coverage_pct=100 * (b != -1).mean(),
                       common_documents=int(both.sum()), common_coverage_pct=100 * both.mean(),
                       assignment_agreement_pct=100 * ((a != -1) == (b != -1)).mean(),
                       assigned_jaccard=float(both.sum() / either.sum()) if either.any() else np.nan)
            for scope, mask in [('all', np.ones(len(a), dtype=bool)), ('common', both), ('union', either)]:
                x, y = a[mask], b[mask]
                usable = len(x) >= 2 and len(set(x)) >= 2 and len(set(y)) >= 2
                row[f'ami_{scope}'] = adjusted_mutual_info_score(x, y) if usable else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_stability_pairs(assignments: pd.DataFrame, label_columns: list[str]) -> pd.DataFrame:
    """Calculate seed-pair stability after pooling article and comment rows.

    This uses the same coverage, common-document, and union definitions as
    :func:`stability_pairs`, but reports one row per seed pair across both
    document types.
    """
    from sklearn.metrics import adjusted_mutual_info_score

    if assignments['doc_id'].duplicated().any():
        raise ValueError('Assignments must have unique document IDs')
    if len(label_columns) < 2 or len(set(label_columns)) != len(label_columns):
        raise ValueError('Provide at least two distinct seed label columns')
    if assignments[label_columns].isna().any().any():
        raise ValueError('All runs must label the same documents')

    rows = []
    for first, second in combinations(label_columns, 2):
        a, b = (assignments[c].to_numpy(dtype=int) for c in (first, second))
        both = (a != -1) & (b != -1)
        either = (a != -1) | (b != -1)
        row = dict(
            doc_type='article_and_comment', first=first, second=second,
            documents=len(a),
            first_coverage_pct=100 * (a != -1).mean(),
            second_coverage_pct=100 * (b != -1).mean(),
            common_documents=int(both.sum()),
            common_coverage_pct=100 * both.mean(),
            assignment_agreement_pct=100 * ((a != -1) == (b != -1)).mean(),
            assigned_jaccard=float(both.sum() / either.sum()) if either.any() else np.nan,
        )
        for scope, mask in [('all', np.ones(len(a), dtype=bool)), ('common', both), ('union', either)]:
            x, y = a[mask], b[mask]
            usable = len(x) >= 2 and len(set(x)) >= 2 and len(set(y)) >= 2
            row[f'ami_{scope}'] = adjusted_mutual_info_score(x, y) if usable else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_diagnostic_corpus(data_root, split_path, embedding_store, year, *, include_test=False, fit_role="development", analysis_comments_path=None):
    """Load articles and a deterministic size-matched comment sample per split."""
    import duckdb
    from .topic_modeling import prepare_article_passages, prepare_documents, load_precomputed_embeddings

    corpus = {}
    if fit_role not in {'development', 'paper2_test'}:
        raise ValueError('Unknown fit_role')
    roles = ['development', 'paper2_test'] if include_test else [fit_role]
    with duckdb.connect() as con:
        for role in roles:
            print(f'Loading {role} articles and sampled comments ...', flush=True)
            articles = con.execute(
                "SELECT CAST(story_id AS VARCHAR) AS story_id, title, subtitle, body "
                "FROM read_parquet(?, hive_partitioning=true, union_by_name=true) "
                "WHERE year = ? AND CAST(story_id AS VARCHAR) IN "
                "(SELECT CAST(story_id AS VARCHAR) FROM read_parquet(?) WHERE split_role = ?) "
                "ORDER BY CAST(story_id AS VARCHAR)",
                [str(data_root / 'articles/**/*.parquet'), year, str(split_path), role],
            ).df()
            passages = prepare_article_passages(articles).sort_values('doc_id').reset_index(drop=True)
            eligible_filter = ""
            eligible_params = []
            if analysis_comments_path is not None:
                eligible_filter = (
                    "AND (CAST(story_id AS VARCHAR), CAST(comment_id AS VARCHAR)) IN "
                    "(SELECT CAST(story_id AS VARCHAR), CAST(comment_id AS VARCHAR) FROM read_parquet(?)) "
                )
                eligible_params = [str(analysis_comments_path)]
            comments = con.execute(
                "SELECT CAST(story_id AS VARCHAR) AS story_id, CAST(comment_id AS VARCHAR) AS comment_id, "
                "effective_text, text FROM read_parquet(?, hive_partitioning=true, union_by_name=true) "
                "WHERE year = ? AND lower(COALESCE(lifecycle_status, 'published')) = 'published' "
                "AND length(trim(COALESCE(effective_text, ''))) > 0 "
                "AND CAST(story_id AS VARCHAR) IN "
                "(SELECT CAST(story_id AS VARCHAR) FROM read_parquet(?) WHERE split_role = ?) "
                + eligible_filter +
                "ORDER BY row_number() OVER (PARTITION BY story_id ORDER BY md5(CAST(comment_id AS VARCHAR)), CAST(comment_id AS VARCHAR)), "
                "md5(CAST(story_id AS VARCHAR) || ':' || CAST(comment_id AS VARCHAR)), "
                "CAST(story_id AS VARCHAR), CAST(comment_id AS VARCHAR) LIMIT ?",
                [str(data_root / 'comments/**/*.parquet'), year, str(split_path), role, *eligible_params, len(passages)],
            ).df()
            comments = prepare_documents(articles.iloc[:0], comments).sort_values('doc_id').reset_index(drop=True)
            if len(comments) != len(passages):
                raise ValueError(f'{role}: too few usable comments for a size-matched sample')
            for kind, documents in [('article_passage', passages), ('comment', comments)]:
                if documents.empty or not documents.doc_id.is_unique:
                    raise ValueError(f'{role}/{kind}: need nonempty documents with unique IDs')
                print(f'Loading embeddings: {role}/{kind}, {len(documents):,} documents', flush=True)
                embeddings = load_precomputed_embeddings(documents, embedding_store)
                if not np.isfinite(embeddings).all():
                    raise ValueError(f'Non-finite embeddings in {role}/{kind}; repair before fitting')
                corpus[(role, kind)] = (documents, embeddings)
    return corpus


def corpus_signature(corpus) -> str:
    """Hash actual ordered document content and embeddings once per loaded corpus."""
    import hashlib
    digest = hashlib.sha256()
    for key, (documents, embeddings) in sorted(corpus.items()):
        digest.update(repr(key).encode())
        digest.update(pd.util.hash_pandas_object(documents, index=False).values.tobytes())
        digest.update(str(embeddings.shape).encode())
        digest.update(str(embeddings.dtype).encode())
        digest.update(memoryview(np.ascontiguousarray(embeddings)).cast('B'))
    return digest.hexdigest()


def diagnostic_fit(corpus, config, output_root, signature, *, batch_size=10_000,
                   fit_corpus="articles", fit_role="development"):
    """Cache one seed fit's diagnostics; completed fits can be reused by stability."""
    from dataclasses import asdict
    import gc
    import hashlib
    import importlib.metadata
    import json
    from pathlib import Path
    from time import perf_counter
    from .topic_modeling import fit_topic_model, topic_model_quality_table
    from .topic_runs import setup_id, write_json

    if fit_corpus not in {'articles', 'articles_comments'}:
        raise ValueError('fit_corpus must be articles or articles_comments')
    setup = dict(configuration=asdict(config), corpus_sha256=signature, batch_size=batch_size,
                 fit_corpus=fit_corpus, fit_role=fit_role,
                 code_hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ('topic_diagnostics.py', 'topic_modeling.py')},
                 versions={name: importlib.metadata.version(name) for name in
                           ('bertopic', 'hdbscan', 'umap-learn', 'scikit-learn', 'numpy', 'pandas')})
    run_id = (f'{fit_corpus}_mcs{config.hdbscan_min_cluster_size}_nn{config.umap_n_neighbors}'
              f'_ms{config.hdbscan_min_samples}_seed{config.random_state}_{setup_id(setup)}')
    root = output_root / run_id
    manifest = root / 'diagnostic.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('completed'):
        print(f'Reusing {run_id}', flush=True)
        return pd.read_csv(root / 'summary.csv'), pd.read_parquet(root / 'assignments.parquet')
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    print(f'Fitting {run_id} ...', flush=True)
    training, embeddings = corpus[(fit_role, 'article_passage')]
    if fit_corpus == 'articles_comments':
        comments, comment_embeddings = corpus[(fit_role, 'comment')]
        training = pd.concat([training, comments], ignore_index=True, sort=False)
        embeddings = np.concatenate([embeddings, comment_embeddings], axis=0)
        order = np.argsort(training.doc_id.to_numpy(), kind='stable')
        training = training.iloc[order].reset_index(drop=True)
        embeddings = embeddings[order]
    if not training.doc_id.is_unique:
        raise ValueError('Training document IDs must be unique')
    bundle = fit_topic_model(training, config, embeddings=embeddings)
    fit_seconds = perf_counter() - started
    print(f'Fit complete in {fit_seconds / 60:.1f} min', flush=True)
    raw_fitted = np.asarray(bundle.model.topics_, dtype=int)
    reduced_fitted = (bundle.reduce_outlier_labels(training.text.tolist(), raw_fitted, embeddings)
                      if config.outlier_threshold is not None else raw_fitted)
    fitted_labels = pd.Series(reduced_fitted, index=training.doc_id)
    raw_fitted_labels = pd.Series(raw_fitted, index=training.doc_id)
    rows, assignments = [], []
    for (role, kind), (documents, vectors) in corpus.items():
        in_fit = role == fit_role and (kind == 'article_passage' or fit_corpus == 'articles_comments')
        if in_fit:
            labels = fitted_labels.loc[documents.doc_id].to_numpy(dtype=int)
            raw_labels = raw_fitted_labels.loc[documents.doc_id].to_numpy(dtype=int)
        else:
            labels = np.empty(len(documents), dtype=int)
            raw_labels = np.empty(len(documents), dtype=int)
            transform_started = perf_counter()
            for start in range(0, len(documents), batch_size):
                stop = min(start + batch_size, len(documents))
                # Fail visibly on numerical errors: do not count transform failures as outliers.
                transformed = bundle.transform(documents.iloc[start:stop], embeddings=vectors[start:stop])
                labels[start:stop] = transformed.topic_assignment.to_numpy(dtype=int)
                raw_labels[start:stop] = transformed.get('raw_topic_assignment', transformed.topic_assignment).to_numpy(dtype=int)
                elapsed = perf_counter() - transform_started
                print(f'{role}/{kind}: {stop:,}/{len(documents):,}; '
                      f'elapsed {elapsed / 60:.1f} min; ETA {elapsed * (len(documents) - stop) / stop / 60:.1f} min', flush=True)
        if labels.shape != (len(documents),):
            raise ValueError('Label/document alignment mismatch')
        rows.append(dict(run_id=run_id, split_role=role, doc_type=kind,
                         fit_corpus=fit_corpus, label_source="fit" if in_fit else "transform",
                         training_documents=len(training),
                         min_cluster_size=config.hdbscan_min_cluster_size,
                         n_neighbors=config.umap_n_neighbors, min_samples=config.hdbscan_min_samples,
                         seed=config.random_state,
                         raw_hdbscan_clusters=len(set(bundle.model.hdbscan_model.labels_) - {-1}),
                         fitted_substantive_topics=len(bundle.topic_ids), fit_seconds=fit_seconds,
                         raw_coverage_pct=coverage_metrics(raw_labels)["coverage_pct"],
                         reassigned_documents=int((labels != raw_labels).sum()),
                         **coverage_metrics(labels)))
        assignments.append(documents[['doc_id', 'story_id']].assign(
            split_role=role, doc_type=kind, topic_assignment=labels,
            raw_topic_assignment=raw_labels, outlier_reassigned=labels != raw_labels))
    summary = pd.DataFrame(rows)
    labels_frame = pd.concat(assignments, ignore_index=True)
    summary.to_csv(root / 'summary.csv', index=False)
    labels_frame.to_parquet(root / 'assignments.parquet', index=False)
    topic_model_quality_table(bundle).to_csv(root / 'topic_terms.csv', index=False)
    write_json(manifest, dict(completed=True, setup=setup))
    del bundle
    gc.collect()
    return summary, labels_frame


__all__ = [
    "search_cache_matches",
    "load_search_artifacts",
    "coverage_metrics",
    "stability_pairs",
    "pooled_stability_pairs",
    "load_diagnostic_corpus",
    "corpus_signature",
    "diagnostic_fit",
]
