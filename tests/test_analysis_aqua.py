import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from aqua_runtime.model import predictions_to_frame, validate_adapter_artifacts
from aqua_runtime.schema import (
    AQUA_FEATURES,
    AQUA_SCORE_MAX,
    AQUA_SCORE_MIN,
    composite_raw,
    downstream_feature_columns,
    expected_alias_column,
    label_column,
    sha256_file,
)
from commentgap_analysis.aqua import (
    AquaBuildConfig,
    build_aqua_store,
    build_runtime_command,
    load_aqua_for_candidates,
    validate_aqua_frame,
)
from commentgap_analysis.aqua_parity import verify_parity
from commentgap_analysis.features import FeatureBuildConfig, _make_choice_set


def _fixture_output_for_keys(keys, build_signature="b" * 64, watermark="PRODUCTION"):
    rows = len(keys)
    logits = {}
    for index, feature in enumerate(AQUA_FEATURES):
        low = np.asarray([4.0 + index / 100, 3.0, 2.0, 1.0])
        high = np.asarray([1.0, 2.0, 3.0, 4.0 + index / 100])
        logits[feature.stem] = np.vstack(
            [low if row % 2 == 0 else high for row in range(rows)]
        )
    return predictions_to_frame(
        keys,
        logits,
        np.asarray([5 if row % 2 == 0 else 600 for row in range(rows)]),
        np.asarray([row % 2 == 1 for row in range(rows)]),
        build_signature=build_signature,
        watermark=watermark,
    )


def _fixture_output(build_signature="b" * 64, watermark="PRODUCTION"):
    keys = pd.DataFrame(
        {
            "story_id": ["s1", "s1"],
            "comment_id": ["c1", "c2"],
            "effective_text_hash": ["h1", "h2"],
        }
    )
    return _fixture_output_for_keys(keys, build_signature, watermark)


class AquaAnalysisTests(unittest.TestCase):
    def test_canonical_schema_and_published_extrema_are_locked(self):
        self.assertEqual(len(AQUA_FEATURES), 20)
        self.assertEqual([feature.order for feature in AQUA_FEATURES], list(range(1, 21)))
        self.assertEqual(AQUA_FEATURES[4].repository_adapter, "solproposal")
        self.assertEqual(AQUA_FEATURES[4].stem, "solution_proposal")
        minimum_labels = np.asarray(
            [[3 if feature.weight < 0 else 0 for feature in AQUA_FEATURES]]
        )
        maximum_labels = np.asarray(
            [[3 if feature.weight > 0 else 0 for feature in AQUA_FEATURES]]
        )
        self.assertAlmostEqual(float(composite_raw(minimum_labels)[0]), AQUA_SCORE_MIN)
        self.assertAlmostEqual(float(composite_raw(maximum_labels)[0]), AQUA_SCORE_MAX)

    def test_artifact_manifest_freezes_all_released_files(self):
        manifest_path = Path("aqua_runtime/artifacts.json")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["upstream"]["commit"], "637914dcd62491766ff478dc21632813780d005d")
        self.assertEqual(set(manifest["adapter_files"]), {feature.repository_adapter for feature in AQUA_FEATURES})
        self.assertTrue(
            all(
                set(files)
                == {
                    "adapter_config.json",
                    "head_config.json",
                    "pytorch_adapter.bin",
                    "pytorch_model_head.bin",
                }
                for files in manifest["adapter_files"].values()
            )
        )
        self.assertEqual(sum(map(len, manifest["adapter_files"].values())), 80)
        self.assertEqual(manifest["parity"]["status"], "pending")

    def test_adapter_artifacts_fail_closed_on_checksum_or_head_change(self):
        feature = AQUA_FEATURES[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / feature.repository_adapter
            adapter.mkdir()
            files = {
                "adapter_config.json": json.dumps({"config": {}}),
                "head_config.json": json.dumps(
                    {
                        "config": {
                            "num_labels": 4,
                            "label2id": {f"LABEL_{value}": value for value in range(4)},
                        }
                    }
                ),
                "pytorch_adapter.bin": "adapter",
                "pytorch_model_head.bin": "head",
            }
            for filename, content in files.items():
                (adapter / filename).write_text(content)
            manifest = {
                "base_model": {
                    "revision": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
                },
                "adapter_files": {
                    feature.repository_adapter: {
                        filename: sha256_file(adapter / filename) for filename in files
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            validated = validate_adapter_artifacts(
                root, manifest_path, features=(feature,)
            )
            self.assertIn(feature.repository_adapter, validated)
            (adapter / "pytorch_adapter.bin").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                validate_adapter_artifacts(root, manifest_path, features=(feature,))

    def test_raw_probabilities_labels_expected_values_and_scores_validate(self):
        output = _fixture_output()
        expected = output[["story_id", "comment_id", "effective_text_hash"]].copy()
        summary = validate_aqua_frame(
            output,
            expected=expected,
            build_signature="b" * 64,
            watermark="PRODUCTION",
        )
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["truncated_rows"], 1)
        self.assertTrue(output["aqua_score_hard"].between(0, 5).all())
        self.assertTrue(output["aqua_score_expected_raw"].between(0, 5).all())
        for feature in AQUA_FEATURES:
            self.assertTrue(output[label_column(feature.stem)].isin(range(4)).all())

    def test_validator_rejects_probability_and_text_hash_corruption(self):
        output = _fixture_output()
        corrupted = output.copy()
        corrupted.at[0, "aqua_relevance_prob_0_raw"] += 0.1
        with self.assertRaisesRegex(ValueError, "probability sums"):
            validate_aqua_frame(corrupted)

        corrupted = output.copy()
        corrupted.at[0, "aqua_relevance_prob_0_raw"] -= 0.01
        corrupted.at[0, "aqua_relevance_prob_1_raw"] += 0.01
        with self.assertRaisesRegex(ValueError, "probabilities from logits"):
            validate_aqua_frame(corrupted)

        expected = output[["story_id", "comment_id", "effective_text_hash"]].copy()
        expected.at[0, "effective_text_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "effective_text_hash mismatch"):
            validate_aqua_frame(output, expected=expected)

    def test_legacy_subprocess_command_is_explicit_and_keyed(self):
        config = AquaBuildConfig(
            runtime_python=Path("/isolated/python"),
            adapter_root=Path("/adapters"),
            artifact_manifest=Path("/artifacts.json"),
            requirements_lock=Path("/requirements.txt"),
            device="cuda",
        )
        command = build_runtime_command(
            config,
            input_path=Path("/tmp/input.parquet"),
            output_path=Path("/tmp/output.parquet"),
            build_signature="s" * 64,
            watermark="PRODUCTION",
        )
        self.assertEqual(command[:3], ["/isolated/python", "-m", "aqua_runtime.cli"])
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertIn("--artifact-manifest", command)
        self.assertIn("--build-signature", command)
        jobs_command = build_runtime_command(
            config,
            build_signature="s" * 64,
            watermark="PRODUCTION",
            job_manifest=Path("/tmp/jobs.json"),
            summary_output=Path("/tmp/summary.json"),
        )
        self.assertIn("--job-manifest", jobs_command)
        self.assertNotIn("--input", jobs_command)

    def test_parity_verifier_compares_upstream_hard_outputs_and_runtime_logits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _fixture_output()
            upstream = runtime[["story_id", "comment_id"]].copy()
            for feature in AQUA_FEATURES:
                upstream[f"{feature.repository_adapter}_ad"] = runtime[
                    label_column(feature.stem)
                ]
            upstream["score"] = runtime["aqua_score_hard"]
            upstream_path = root / "upstream.tsv"
            runtime_path = root / "runtime.parquet"
            repeat_path = root / "repeat.parquet"
            upstream.to_csv(upstream_path, sep="\t", index=False)
            runtime.to_parquet(runtime_path, index=False)
            runtime.to_parquet(repeat_path, index=False)
            report = verify_parity(
                upstream_path, runtime_path, repeat_output=repeat_path
            )
            self.assertEqual(report["status"], "verified")
            self.assertTrue(
                all(value["exact"] for value in report["adapter_results"].values())
            )

    def test_completed_store_merges_aliases_and_rejects_pilot_for_inference(self):
        from aqua_runtime.schema import text_hash

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_root = root / "build=fixture"
            shard = build_root / "year=2025" / "month=01" / "s1.parquet"
            shard.parent.mkdir(parents=True)
            texts = ["Hallo", "Eine zweite Aussage"]
            output = _fixture_output()
            output["effective_text_hash"] = [text_hash(text) for text in texts]
            output.to_parquet(shard, index=False)
            validation_path = build_root / "aqua_validation.json"
            validation_path.write_text(json.dumps({"rows": 2, "error_rows": 0}))
            manifest = {
                "status": "complete",
                "schema_version": 1,
                "build_signature": "b" * 64,
                "build_root": str(build_root),
                "watermark": "PRODUCTION",
                "source_dataset_fingerprint": "fingerprint",
                "probability_status": "uncalibrated",
                "validation_path": str(validation_path),
            }
            (root / "aqua_manifest.json").write_text(json.dumps(manifest))
            candidates = pd.DataFrame(
                {
                    "story_id": ["s1", "s1"],
                    "comment_id": ["c1", "c2"],
                    "effective_text": texts,
                }
            )
            compact, loaded_manifest = load_aqua_for_candidates(
                root,
                candidates,
                year=2025,
                source_fingerprint="fingerprint",
                require_production=True,
            )
            self.assertEqual(loaded_manifest["build_signature"], "b" * 64)
            self.assertIn(expected_alias_column("justification"), compact)
            self.assertIn("aqua_score_expected", compact)

            manifest["watermark"] = "PILOT_NOT_FOR_INFERENCE"
            (root / "aqua_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "production AQuA store"):
                load_aqua_for_candidates(
                    root,
                    candidates,
                    year=2025,
                    source_fingerprint="fingerprint",
                    require_production=True,
                )

    def test_store_build_batches_jobs_in_one_process_and_resumes_checkpoints(self):
        from aqua_runtime.schema import text_hash

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            comment_path = (
                data_root / "comments" / "year=2025" / "month=01" / "part.parquet"
            )
            comment_path.parent.mkdir(parents=True)
            comments = pd.DataFrame(
                {
                    "story_id": ["s1", "s2"],
                    "comment_id": ["c1", "c2"],
                    "effective_text": ["Ein Beitrag", "Noch ein Beitrag"],
                    "lifecycle_status": ["Published", "Published"],
                    "created_at": pd.to_datetime(
                        ["2025-01-02T10:00:00Z", "2025-01-03T11:00:00Z"]
                    ),
                    "year": [2025, 2025],
                    "month": [1, 1],
                }
            )
            comments.to_parquet(comment_path, index=False)
            qa_path = data_root / "qa_summary" / "year=2025" / "summary.json"
            qa_path.parent.mkdir(parents=True)
            qa_path.write_text(
                json.dumps(
                    {"passed": True, "nonterminal_stories": 0, "status_counts": {}}
                )
            )
            runtime_python = root / "python"
            runtime_python.write_text("fixture")
            config = AquaBuildConfig(
                data_root=data_root,
                output_root=root / "aqua",
                runtime_python=runtime_python,
                adapter_root=root / "adapters",
                artifact_manifest=Path("aqua_runtime/artifacts.json").resolve(),
                requirements_lock=Path("requirements-aqua-legacy.txt").resolve(),
                require_parity=False,
            )
            calls = []

            def fake_runtime(command, **kwargs):
                calls.append((command, kwargs))
                jobs_path = Path(command[command.index("--job-manifest") + 1])
                build_signature = command[command.index("--build-signature") + 1]
                watermark = command[command.index("--watermark") + 1]
                jobs = json.loads(jobs_path.read_text())["jobs"]
                for job in jobs:
                    source = pd.read_parquet(job["input"])
                    keys = source[
                        ["story_id", "comment_id", "effective_text_hash"]
                    ].copy()
                    self.assertEqual(
                        keys["effective_text_hash"].tolist(),
                        source["effective_text"].map(text_hash).tolist(),
                    )
                    output = _fixture_output_for_keys(
                        keys,
                        build_signature=build_signature,
                        watermark=watermark,
                    )
                    destination = Path(job["output"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    output.to_parquet(destination, index=False)
                summary_path = Path(command[command.index("--summary-output") + 1])
                summary_path.write_text(
                    json.dumps(
                        {
                            "device": "cpu",
                            "execution_mode": "parallel",
                            "rows": len(comments),
                            "shards": len(jobs),
                            "elapsed_seconds": 1.0,
                            "peak_memory_mib": 100.0,
                            "python": "3.10.14",
                            "platform": "test",
                            "packages": {"adapter-transformers": "3.2.1"},
                        }
                    )
                )
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            first = build_aqua_store(config, subprocess_runner=fake_runtime)
            self.assertEqual(len(calls), 1)
            self.assertEqual(first["new_story_checkpoints"], 2)
            self.assertEqual(first["skipped_story_checkpoints"], 0)
            self.assertEqual(first["watermark"], "PILOT_NOT_FOR_INFERENCE")
            self.assertEqual(len(first["adapter_artifact_hashes"]), 20)
            self.assertEqual(len(first["runtime_processes"]), 1)

            second = build_aqua_store(config, subprocess_runner=fake_runtime)
            self.assertEqual(len(calls), 1)
            self.assertEqual(second["new_story_checkpoints"], 0)
            self.assertEqual(second["skipped_story_checkpoints"], 2)

    def test_aqua_features_propagate_to_root_and_all_choice_sets(self):
        features = pd.DataFrame(
            {
                "story_id": ["s1"] * 3,
                "comment_id": ["c1", "c2", "c3"],
                "article_year": [2025] * 3,
                "article_month": [2] * 3,
                "is_root": [True, True, False],
                "is_sticky": [True, False, False],
                "published_at_source": ["page"] * 3,
                "invalid_posting_time": [False] * 3,
                "votes_positive": [5, 2, 1],
                "votes_negative": [0, 0, 0],
                "vienna_period": ["weekday_work"] * 3,
                "word_count": [10, 12, 8],
                "cttr": [0.5] * 3,
                "smog_de": [6.0] * 3,
                "sentiment_positive": [0.4] * 3,
                "sentiment_negative": [0.2] * 3,
                "sentiment_neutral": [0.4] * 3,
                "toxicity_probability": [0.1] * 3,
                "toxicity_mean_probability": [0.05] * 3,
                "log_words": [2.0] * 3,
                "lexdiv_length_adjusted": [0.0] * 3,
                "reading_level_length_adjusted": [0.0] * 3,
                "url_present": [0] * 3,
                "article_similarity_top3": [0.5] * 3,
                "novelty_prior_roots": [0.2, 0.3, 0.4],
                "novelty_prior_all": [0.2, 0.3, 0.4],
            }
        )
        for column in (
            "hours_since_article",
            "prior_roots",
            "prior_comments",
            "comments_prev_hour",
            "author_prior_30d_comments",
            "author_prior_30d_snapshot_upvotes",
            "author_prior_30d_snapshot_downvotes",
            "author_prior_comments_story",
            "depth",
            "branch_prior_comments",
            "branch_comments_prev_hour",
        ):
            features[column] = [1, 2, 3]
        for feature in AQUA_FEATURES:
            features[label_column(feature.stem)] = [0, 1, 2]
            features[expected_alias_column(feature.stem)] = [0.25, 1.25, 2.25]
        features["aqua_score_hard"] = [1.0, 2.0, 3.0]
        features["aqua_score_expected"] = [1.1, 2.1, 3.1]
        features["aqua_input_token_count"] = [10, 11, 12]
        features["aqua_input_truncated"] = [False, False, False]
        features["aqua_runtime_status"] = ["ok"] * 3
        features["aqua_probability_status"] = ["uncalibrated"] * 3
        raw = features[["story_id", "comment_id", "is_root", "is_sticky"]].copy()
        raw["lifecycle_status"] = "Published"
        raw["effective_text"] = ["eins", "zwei", "drei"]
        raw["created_at"] = pd.to_datetime(
            ["2025-02-01T10:00:00Z", "2025-02-01T11:00:00Z", "2025-02-01T12:00:00Z"]
        )
        config = FeatureBuildConfig(
            inference_mode=False,
            tie_draws=2,
            exclude_january_without_lookback=False,
        )
        root, _, _ = _make_choice_set(features, raw, scope="root", config=config)
        all_comments, _, _ = _make_choice_set(
            features, raw, scope="all", config=config
        )
        self.assertEqual(root["comment_id"].tolist(), ["c1", "c2"])
        self.assertEqual(all_comments["comment_id"].tolist(), ["c1", "c2", "c3"])
        for column in downstream_feature_columns():
            self.assertIn(column, root)
            self.assertIn(column, all_comments)


if __name__ == "__main__":
    unittest.main()
