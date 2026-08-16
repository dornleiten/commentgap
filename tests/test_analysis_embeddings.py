import json
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from commentgap_analysis.embeddings import (
    AdaptiveBatchEncoder,
    EmbeddingBuildConfig,
    build_embedding_store,
    build_token_length_diagnostics,
)


class FakeEmbedder:
    resolved_revision = "fixture-revision"

    def __init__(self, model_id: str = "fixture/model") -> None:
        self.model_id = model_id
        self.calls = 0

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.calls += 1
        rows = []
        for text in texts:
            vector = np.array(
                [len(text), text.count("a") + 1, text.count("e") + 1, 1],
                dtype=np.float32,
            )
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)

    def token_lengths(self, texts: list[str], batch_size: int) -> np.ndarray:
        return np.asarray([len(text.split()) + 2 for text in texts], dtype=np.int32)


class CapacityLimitedEmbedder:
    model_id = "fixture/capacity"

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.attempts: list[int] = []

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.attempts.append(batch_size)
        if batch_size > self.capacity:
            raise RuntimeError("CUDA out of memory")
        return np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2)


class EmbeddingStoreTests(unittest.TestCase):
    def test_adaptive_batching_backs_off_and_cautiously_recovers(self):
        embedder = CapacityLimitedEmbedder(capacity=2)
        adaptive = AdaptiveBatchEncoder(
            embedder,
            device="cuda",
            initial_batch_size=8,
            maximum_batch_size=8,
            minimum_batch_size=1,
            growth_successes=2,
            adaptive=True,
        )
        adaptive.encode(["a", "b", "c"])
        self.assertEqual(embedder.attempts, [8, 4, 2])
        adaptive.encode(["a", "b", "c"])
        self.assertEqual(adaptive.current_batch_size, 4)
        adaptive.encode(["a", "b", "c"])
        self.assertEqual(adaptive.current_batch_size, 2)
        self.assertEqual(
            [event["event"] for event in adaptive.history],
            [
                "decrease_after_oom",
                "decrease_after_oom",
                "increase_after_successes",
                "decrease_after_oom",
            ],
        )

    def _fixture(self, root: Path) -> Path:
        data = root / "data"
        comments_path = data / "comments/year=2025/month=01/comments.parquet"
        articles_path = data / "articles/year=2025/month=01/articles.parquet"
        comments_path.parent.mkdir(parents=True)
        articles_path.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "comment_id": "c2",
                        "story_id": "s1",
                        "created_at": "2025-01-01T11:00:00Z",
                        "effective_text": "Zweiter Kommentar",
                        "lifecycle_status": "Published",
                        "year": 2025,
                        "month": 1,
                    },
                    {
                        "comment_id": "c1",
                        "story_id": "s1",
                        "created_at": "2025-01-01T10:00:00Z",
                        "effective_text": "Erster Kommentar",
                        "lifecycle_status": "Published",
                        "year": 2025,
                        "month": 1,
                    },
                    {
                        "comment_id": "deleted",
                        "story_id": "s1",
                        "created_at": "2025-01-01T09:00:00Z",
                        "effective_text": "",
                        "lifecycle_status": "Deleted",
                        "year": 2025,
                        "month": 1,
                    },
                ]
            ),
            comments_path,
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "story_id": "s1",
                        "title": "Titel",
                        "subtitle": "Untertitel",
                        "body": "Absatz eins.\n\nAbsatz zwei.",
                        "year": 2025,
                        "month": 1,
                    }
                ]
            ),
            articles_path,
        )
        qa_path = data / "qa_summary/year=2025/summary.json"
        qa_path.parent.mkdir(parents=True)
        qa_path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "nonterminal_stories": 0,
                    "status_counts": {"completed": 1},
                }
            )
        )
        return data

    def _add_2024_fixture(self, data: Path) -> None:
        comments_path = data / "comments/year=2024/month=12/comments.parquet"
        articles_path = data / "articles/year=2024/month=12/articles.parquet"
        comments_path.parent.mkdir(parents=True)
        articles_path.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "comment_id": "c2024",
                        "story_id": "s2024",
                        "created_at": "2024-12-15T11:00:00Z",
                        "effective_text": "Kommentar aus 2024",
                        "lifecycle_status": "Published",
                        "year": 2024,
                        "month": 12,
                    }
                ]
            ),
            comments_path,
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "story_id": "s2024",
                        "title": "Titel 2024",
                        "subtitle": "Untertitel 2024",
                        "body": "Ein Absatz.",
                        "year": 2024,
                        "month": 12,
                    }
                ]
            ),
            articles_path,
        )
        qa_path = data / "qa_summary/year=2024/month=12/summary.json"
        qa_path.parent.mkdir(parents=True)
        qa_path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "nonterminal_stories": 0,
                    "status_counts": {"completed": 1},
                }
            )
        )

    def test_keyed_vector_store_is_resumable_and_model_versioned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._fixture(root)
            output = root / "embeddings"
            config = EmbeddingBuildConfig(
                data_root=data,
                output_root=output,
                model_id="fixture/model",
            )
            first_embedder = FakeEmbedder()
            output_log = io.StringIO()
            with redirect_stdout(output_log):
                first = build_embedding_store(config, embedder=first_embedder)
            self.assertIn("requested_device=auto resolved_device=", output_log.getvalue())
            self.assertIn("Embedding progress: 1/1 articles", output_log.getvalue())
            self.assertEqual(first["embedding_dimension"], 4)
            self.assertEqual(first["storage"]["comments"]["rows"], 2)
            self.assertEqual(first["storage"]["article_passages"]["rows"], 4)
            self.assertEqual(first["watermark"], "COMPLETE_SOURCE")
            self.assertEqual(first_embedder.calls, 2)

            run_root = Path(first["run_root"])
            comments = ds.dataset(run_root / "comments", format="parquet").to_table()
            self.assertEqual(comments.column("comment_id").to_pylist(), ["c1", "c2"])
            self.assertTrue(pa.types.is_fixed_size_list(comments.schema.field("embedding").type))
            self.assertNotIn("effective_text", comments.schema.names)
            self.assertTrue((run_root / "embedding_manifest.json").exists())

            resumed_embedder = FakeEmbedder()
            resumed = build_embedding_store(config, embedder=resumed_embedder)
            self.assertEqual(resumed["run_root"], first["run_root"])
            self.assertEqual(resumed_embedder.calls, 0)

            alternate = build_embedding_store(
                EmbeddingBuildConfig(
                    data_root=data,
                    output_root=output,
                    model_id="fixture/alternative",
                    storage_dtype="float16",
                ),
                embedder=FakeEmbedder("fixture/alternative"),
            )
            self.assertNotEqual(alternate["run_root"], first["run_root"])
            self.assertIn("float16", alternate["storage"]["vector_type"])

    def test_adapter_model_must_match_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_embedding_store(
                    EmbeddingBuildConfig(
                        data_root=data,
                        output_root=root / "embeddings",
                        model_id="expected/model",
                    ),
                    embedder=FakeEmbedder("different/model"),
                )

    def test_default_discovers_annual_and_month_scoped_years(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._fixture(root)
            self._add_2024_fixture(data)
            result = build_embedding_store(
                EmbeddingBuildConfig(
                    data_root=data,
                    output_root=root / "embeddings",
                    model_id="fixture/model",
                ),
                embedder=FakeEmbedder(),
            )
            self.assertEqual(result["source"]["years"], [2024, 2025])
            self.assertEqual(result["source"]["eligible_comments_by_year"], {"2024": 1, "2025": 2})
            run_root = Path(result["run_root"])
            self.assertTrue(
                (run_root / "comments/year=2024/month=12/s2024.parquet").exists()
            )
            self.assertTrue(
                (run_root / "comments/year=2025/month=01/s1.parquet").exists()
            )

    def test_token_diagnostics_report_quantiles_and_key_only_truncations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._fixture(root)
            result = build_token_length_diagnostics(
                EmbeddingBuildConfig(
                    data_root=data,
                    output_root=root / "embeddings",
                    model_id="fixture/model",
                    max_length=3,
                ),
                inspector=FakeEmbedder(),
            )
            self.assertEqual(result["comments"]["count"], 2)
            self.assertEqual(result["comments"]["over_max_length"], 2)
            self.assertIsNotNone(result["comments"]["p95_tokens"])
            truncated = pq.read_table(result["truncated_records_path"])
            self.assertIn("source_text_sha256", truncated.schema.names)
            self.assertNotIn("text", truncated.schema.names)


if __name__ == "__main__":
    unittest.main()
