"""Regression coverage for the notebook's cached seed alignment/export."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


class NotebookAlignmentTests(unittest.TestCase):
    def test_three_seed_diagnostics_have_unique_columns_and_round_trip(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / '14_topic_model_fit.ipynb').read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if '        seed_summaries, seed_assignments = [], []\n' in ''.join(cell['source']))
        # Execute the actual notebook alignment block without launching fits.
        import textwrap
        block = textwrap.dedent(source[source.index('        seed_summaries, seed_assignments = [], []'):source.index('        label_columns =')])
        seeds = [2025, 2026, 2027]
        seed_results = {}
        for seed in seeds:
            assignments = pd.DataFrame({
                'doc_id': ['a', 'b'], 'story_id': ['s', 's'],
                'doc_type': ['article_passage', 'comment'], 'split_role': 'paper2_test',
                'topic_assignment': [0, seed], 'raw_topic_assignment': [0, -1],
                'outlier_reassigned': [False, True],
            })
            if seed == 2026:
                assignments = assignments.iloc[::-1]
            seed_results[seed] = (pd.DataFrame({'split_role': ['paper2_test']}), assignments)
        namespace = dict(pd=pd, STABILITY_SEEDS=seeds, seed_results=seed_results, FIT_ROLE='paper2_test')
        exec(compile(ast.parse(block), '<notebook seed alignment>', 'exec'), namespace)
        aligned = namespace['aligned']
        self.assertTrue(aligned.columns.is_unique)
        for seed in seeds:
            row = aligned.set_index('doc_id').loc['b']
            self.assertEqual(row[f'seed_{seed}'], seed)
            self.assertEqual(row[f'raw_seed_{seed}'], -1)
            self.assertTrue(row[f'outlier_reassigned_seed_{seed}'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'assignments.parquet'
            aligned.to_parquet(path, index=False)
            pd.testing.assert_frame_equal(pd.read_parquet(path), aligned)
