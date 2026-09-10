import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from aqua_runtime.model import predictions_to_frame
from aqua_runtime.schema import AQUA_FEATURES, sha256_file
from commentgap_analysis.aqua import AquaBuildConfig, build_aqua_store
from commentgap_analysis.aqua_promotion import (
    AquaPromotionConfig,
    promote_aqua_store,
)


def _fixture_output(keys, build_signature, watermark):
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
        np.asarray([5 + row for row in range(rows)]),
        np.zeros(rows, dtype=bool),
        build_signature=build_signature,
        watermark=watermark,
    )


class AquaPromotionTests(unittest.TestCase):
    def test_promotion_requires_parity_then_preserves_predictions_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            comment_path = (
                data_root / "comments" / "year=2025" / "month=01" / "part.parquet"
            )
            comment_path.parent.mkdir(parents=True)
            comments = pd.DataFrame(
                {
                    "story_id": ["s1", "s1"],
                    "comment_id": ["c1", "c2"],
                    "effective_text": ["Ein Beitrag", "Noch ein Beitrag"],
                    "lifecycle_status": ["Published", "Published"],
                    "created_at": pd.to_datetime(
                        ["2025-01-02T10:00:00Z", "2025-01-02T11:00:00Z"]
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

            artifact_manifest = root / "artifacts.json"
            shutil.copyfile("aqua_runtime/artifacts.json", artifact_manifest)
            artifacts = json.loads(artifact_manifest.read_text())
            artifacts["parity"] = {"status": "pending"}
            artifact_manifest.write_text(json.dumps(artifacts))
            runtime_python = root / "python"
            runtime_python.write_text("fixture")
            output_root = root / "aqua"
            build_config = AquaBuildConfig(
                data_root=data_root,
                output_root=output_root,
                runtime_python=runtime_python,
                adapter_root=root / "adapters",
                artifact_manifest=artifact_manifest,
                requirements_lock=Path("requirements-aqua-legacy.txt").resolve(),
                require_parity=False,
            )

            def fake_runtime(command, **kwargs):
                jobs_path = Path(command[command.index("--job-manifest") + 1])
                summary_path = Path(command[command.index("--summary-output") + 1])
                signature = command[command.index("--build-signature") + 1]
                watermark = command[command.index("--watermark") + 1]
                jobs = json.loads(jobs_path.read_text())["jobs"]
                for job in jobs:
                    source = pd.read_parquet(job["input"])
                    destination = Path(job["output"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _fixture_output(
                        source[["story_id", "comment_id", "effective_text_hash"]],
                        signature,
                        watermark,
                    ).to_parquet(destination, index=False)
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

            pilot = build_aqua_store(build_config, subprocess_runner=fake_runtime)
            pilot_shard = (
                Path(pilot["build_root"]) / "year=2025" / "month=01" / "s1.parquet"
            )
            pilot_frame = pd.read_parquet(pilot_shard)
            self.assertEqual(pilot["watermark"], "PILOT_NOT_FOR_INFERENCE")

            promotion_config = AquaPromotionConfig(
                source_store=output_root,
                output_root=output_root,
                data_root=data_root,
                artifact_manifest=artifact_manifest,
            )
            with self.assertRaisesRegex(RuntimeError, "requires verified parity"):
                promote_aqua_store(promotion_config)
            self.assertEqual(
                json.loads((output_root / "aqua_manifest.json").read_text())[
                    "watermark"
                ],
                "PILOT_NOT_FOR_INFERENCE",
            )

            parity_report = root / "parity_report.json"
            parity_report.write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "upstream_commit": pilot["upstream"]["commit"],
                        "rows": 2,
                        "adapter_results": {
                            feature.repository_adapter: {
                                "rows": 2,
                                "matching_hard_labels": 2,
                                "exact": True,
                            }
                            for feature in AQUA_FEATURES
                        },
                        "maximum_hard_score_difference": 0.0,
                        "comparisons": {
                            name: {
                                "hard_labels_exact": True,
                                "logits_within_tolerance": True,
                                "logit_absolute_tolerance": 2e-6,
                                "logit_relative_tolerance": 1e-6,
                                "maximum_absolute_logit_difference": 0.0,
                                "sha256": "a" * 64,
                            }
                            for name in ("repeat_cpu", "sequential")
                        },
                        "contains_raw_comment_text": False,
                    },
                    sort_keys=True,
                )
            )
            artifacts = json.loads(artifact_manifest.read_text())
            artifacts["parity"] = {
                "status": "verified",
                "fixture": str(parity_report),
                "fixture_sha256": sha256_file(parity_report),
            }
            artifact_manifest.write_text(json.dumps(artifacts))

            production = promote_aqua_store(promotion_config)
            self.assertEqual(production["watermark"], "PRODUCTION")
            self.assertNotEqual(production["build_signature"], pilot["build_signature"])
            self.assertFalse(production["promotion"]["numeric_predictions_recomputed"])
            self.assertEqual(production["new_story_checkpoints"], 1)
            production_shard = (
                Path(production["build_root"])
                / "year=2025"
                / "month=01"
                / "s1.parquet"
            )
            production_frame = pd.read_parquet(production_shard)
            pd.testing.assert_frame_equal(
                pilot_frame.drop(
                    columns=["aqua_build_signature", "aqua_watermark"]
                ),
                production_frame.drop(
                    columns=["aqua_build_signature", "aqua_watermark"]
                ),
                check_exact=True,
            )
            self.assertTrue(production_frame["aqua_watermark"].eq("PRODUCTION").all())

            # Point the source option back to the retained pilot build to exercise
            # checkpoint resume after the root manifest now selects production.
            (output_root / "aqua_manifest.json").write_text(
                (Path(pilot["build_root"]) / "aqua_manifest.json").read_text()
            )
            resumed = promote_aqua_store(promotion_config)
            self.assertEqual(resumed["new_story_checkpoints"], 0)
            self.assertEqual(resumed["skipped_story_checkpoints"], 1)


if __name__ == "__main__":
    unittest.main()
