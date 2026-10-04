"""Verify resumed reporting loads the notebook's persisted diagnostic objects."""
import json
import os
from pathlib import Path
import tempfile
import textwrap
from unittest.mock import patch, Mock

import pandas as pd


def test_resume_uses_saved_development_objects_without_scoring():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / '09_model_tables_plots.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells']
                  if 'DEVELOPMENT_PRODUCTS = TABLES.parent' in ''.join(c['source']))
    start = source.index('    DEVELOPMENT_PRODUCTS = TABLES.parent')
    stop = source.index('    training_table = pd.DataFrame([', start)
    code = textwrap.dedent(source[start:stop])
    with tempfile.TemporaryDirectory() as directory:
        tables = Path(directory) / 'tables'
        products = tables.parent / 'development_scores'
        products.mkdir()
        frame = pd.DataFrame({'story_id': ['one'], 'audience_score': [1.0]})
        for label in ('Reg', 'XGB', 'XGB-T', 'NN', 'NN-T'):
            frame.to_parquet(products / f'development_scores_{label}.parquet', index=False)
        articles = pd.DataFrame({'Model': ['NN-T'], 'ndcg_at_k': [0.8]})
        metrics = pd.DataFrame({'estimate': [0.8], 'conf_low': [0.7]})
        articles.to_parquet(products / 'training_articles.parquet', index=False)
        metrics.to_parquet(products / 'training_metrics.parquet', index=False)
        producer = Mock(side_effect=AssertionError('Resume must not score saved products'))
        namespace = {'os': os, 'pd': pd, 'TABLES': tables, 'score_development_model': producer}
        with patch.dict(os.environ, {'COMMENTGAP_MODE': 'resume'}):
            exec(code, namespace)
        producer.assert_not_called()
        pd.testing.assert_frame_equal(namespace['training_articles'], articles)
        pd.testing.assert_frame_equal(namespace['training_metrics'], metrics)
        pd.testing.assert_frame_equal(namespace['wide'], frame)
        assert set(namespace['training_wide_scores']) == {'Reg','XGB','XGB-T','NN','NN-T'}
