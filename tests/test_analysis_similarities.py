import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from commentgap_analysis.embeddings import _multi_year_fingerprint
from commentgap_analysis.features import build_analysis_features
from commentgap_analysis.similarities import (
    SimilarityBuildConfig,
    _cosine_matrix,
    _story_similarity,
    build_similarity_store,
    required_temporal_novelties,
)


def _vectors(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float16)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float16()), values.shape[1]
    )


class AnalysisSimilarityTests(unittest.TestCase):
    def test_exact_discussion_reuses_one_matrix_for_both_novelty_scopes(self):
        vectors = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        timestamps = pd.Series(
            pd.to_datetime(
                ["2025-01-01T10:00Z", "2025-01-01T11:00Z", "2025-01-01T12:00Z"],
                utc=True,
            )
        )
        with patch(
            "commentgap_analysis.similarities._cosine_matrix",
            wraps=_cosine_matrix,
        ) as cosine_matrix:
            all_values, root_values, diagnostics, methods = required_temporal_novelties(
                vectors,
                timestamps,
                np.asarray([True, False, True]),
                exact_threshold=5_000,
                seed=1,
                validation_sample=10,
            )
        cosine_matrix.assert_called_once()
        np.testing.assert_allclose(all_values[1:], [0.0, 1.0])
        self.assertTrue(np.isnan(root_values[0]))
        self.assertTrue(np.isnan(root_values[1]))
        self.assertAlmostEqual(float(root_values[2]), 1.0)
        self.assertEqual(methods, {"all": "exact_shared_matrix", "root": "exact_shared_matrix"})
        self.assertEqual(diagnostics, [])

    def test_story_output_contains_only_required_scalar_similarities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comment_path = root / "comments.parquet"
            passage_path = root / "passages.parquet"
            source_path = root / "source.parquet"
            comments = pa.table(
                {
                    "story_id": ["s", "s", "s"],
                    "comment_id": ["c1", "c2", "c3"],
                    "created_at": pa.array(
                        pd.to_datetime(
                            ["2025-01-01T10:00Z", "2025-01-01T11:00Z", "2025-01-01T12:00Z"],
                            utc=True,
                        )
                    ),
                    "source_text_sha256": ["a", "b", "c"],
                    "embedding": _vectors(np.asarray([[1, 0], [1, 0], [0, 1]])),
                }
            )
            passages = pa.table(
                {
                    "story_id": ["s", "s"],
                    "passage_id": ["s:title", "s:body:0000"],
                    "passage_kind": ["title", "body"],
                    "passage_index": pa.array([0, 0], type=pa.int32()),
                    "source_text_sha256": ["d", "e"],
                    "embedding": _vectors(np.asarray([[1, 0], [0, 1]])),
                }
            )
            source = pa.table(
                {
                    "story_id": ["s", "s", "s"],
                    "comment_id": ["c1", "c2", "c3"],
                    "is_root": [True, False, True],
                }
            )
            pq.write_table(comments, comment_path)
            pq.write_table(passages, passage_path)
            pq.write_table(source, source_path)
            output, diagnostics, summary = _story_similarity(
                comment_path,
                passage_path,
                source_path,
                exact_threshold=5_000,
                seed=1,
                validation_sample=10,
            )
        self.assertEqual(
            list(output.columns),
            [
                "story_id",
                "comment_id",
                "article_similarity_top3",
                "novelty_prior_all",
                "novelty_prior_roots",
            ],
        )
        self.assertEqual(len(output), 3)
        self.assertEqual(diagnostics, [])
        self.assertEqual(summary["methods"]["all"], "exact_shared_matrix")

    def test_hnsw_branch_is_validated_against_exact_prior_neighbours(self):
        try:
            import hnswlib  # noqa: F401
        except ImportError:
            self.skipTest("hnswlib is optional outside requirements-analysis.txt")
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(300, 64)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        timestamps = pd.Series(
            pd.date_range("2025-01-01", periods=len(vectors), freq="min", tz="UTC")
        )
        is_root = np.zeros(len(vectors), dtype=bool)
        is_root[::3] = True
        all_values, root_values, diagnostics, methods = required_temporal_novelties(
            vectors,
            timestamps,
            is_root,
            exact_threshold=100,
            seed=5,
            validation_sample=100,
        )
        self.assertEqual(methods, {"all": "hnsw", "root": "exact_root_matrix"})
        self.assertGreaterEqual(diagnostics[0]["recall_within_1e_3"], 0.95)
        self.assertTrue(np.isfinite(all_values[1:]).all())
        self.assertTrue(np.isfinite(root_values[3::3]).all())

    def test_feature_builder_has_no_embedding_adapter_or_encode_call(self):
        self.assertNotIn("embedder", inspect.signature(build_analysis_features).parameters)
        self.assertNotIn("embedder.encode", inspect.getsource(build_analysis_features))

    def test_resumable_store_build_uses_keyed_embedding_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding_store = root / "embedding-build"
            data_root = root / "data"
            output_root = root / "similarities"
            comment_path = embedding_store / "comments/year=2025/month=01/s.parquet"
            passage_path = embedding_store / "article_passages/year=2025/month=01/s.parquet"
            source_path = data_root / "comments/year=2025/month=01/s.parquet"
            comment_path.parent.mkdir(parents=True)
            passage_path.parent.mkdir(parents=True)
            source_path.parent.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "story_id": ["s", "s"],
                        "comment_id": ["c1", "c2"],
                        "created_at": pa.array(
                            pd.to_datetime(["2025-01-01T10:00Z", "2025-01-01T11:00Z"], utc=True)
                        ),
                        "source_text_sha256": ["a", "b"],
                        "embedding": _vectors(np.asarray([[1, 0], [0, 1]])),
                    }
                ),
                comment_path,
            )
            pq.write_table(
                pa.table(
                    {
                        "story_id": ["s"],
                        "passage_id": ["s:title"],
                        "passage_kind": ["title"],
                        "passage_index": pa.array([0], type=pa.int32()),
                        "source_text_sha256": ["c"],
                        "embedding": _vectors(np.asarray([[1, 0]])),
                    }
                ),
                passage_path,
            )
            pq.write_table(
                pa.table(
                    {
                        "story_id": ["s", "s"],
                        "comment_id": ["c1", "c2"],
                        "is_root": [True, False],
                    }
                ),
                source_path,
            )
            qa_path = data_root / "qa_summary/year=2025/summary.json"
            qa_path.parent.mkdir(parents=True)
            qa_path.write_text('{"passed": true, "nonterminal_stories": 0, "status_counts": {}}')
            dataset_fingerprint, _ = _multi_year_fingerprint(data_root, (2025,))
            (embedding_store / "embedding_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "watermark": "COMPLETE_SOURCE",
                        "build_signature": "b" * 64,
                        "dataset_fingerprint": dataset_fingerprint,
                        "model": {
                            "model_id": "fixture/model",
                            "resolved_revision": "r" * 40,
                        },
                        "source": {"years": [2025]},
                    }
                )
            )
            config = SimilarityBuildConfig(
                data_root=data_root,
                embedding_store=embedding_store,
                output_root=output_root,
                years=(2025,),
                model_id="fixture/model",
                revision="r" * 40,
                progress_every_stories=1,
            )
            first = build_similarity_store(config)
            second = build_similarity_store(config)
            scalar_path = Path(first["run_root"]) / "scalars/year=2025/month=01/s.parquet"
            scalar = pq.read_table(scalar_path)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["storage"]["rows"], 2)
        self.assertEqual(second["execution"]["skipped_story_checkpoints"], 1)
        self.assertEqual(
            scalar.schema.names,
            [
                "story_id",
                "comment_id",
                "article_similarity_top3",
                "novelty_prior_all",
                "novelty_prior_roots",
            ],
        )


if __name__ == "__main__":
    unittest.main()
