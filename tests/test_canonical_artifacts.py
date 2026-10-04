from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.canonical_artifacts import build, digest, promote, verify


class CanonicalArtifactsTests(unittest.TestCase):
    def test_build_merges_reconstruction_and_promotion_preserves_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "provenance").mkdir()
            frozen = root / "artifacts/frozen"
            earlier = root / "artifacts/reconstructed/20260930"
            later = root / "artifacts/reconstructed/20261003"
            for path in (frozen, earlier, later):
                path.mkdir(parents=True)
            original = frozen / "CG1/descriptives/table.csv"
            original.parent.mkdir(parents=True)
            original.write_text("historical\n")
            recovered = later / "CG1/descriptives/table.csv"
            recovered.parent.mkdir(parents=True)
            recovered.write_text("accepted recovery\n")
            supplemental = earlier / "CG1/development_scores/scores.csv"
            supplemental.parent.mkdir(parents=True)
            supplemental.write_text("scores\n")

            manifest = build(root)
            canonical = root / "artifacts/canonical/CG1/descriptives/table.csv"
            archive = root / "artifacts/archive/reconstructed/20261003/CG1/descriptives/table.csv"
            self.assertEqual(canonical.read_text(), "accepted recovery\n")
            self.assertEqual((root / "artifacts/canonical/CG1/development_scores/scores.csv").read_text(), "scores\n")
            self.assertEqual(len(manifest["different_name_collisions"]), 1)
            self.assertFalse(canonical.stat().st_mode & 0o222)
            self.assertTrue(verify(root)["archive_roots_present"])
            archived_hash = digest(archive)

            run = root / "outputs/new-run"
            produced = run / "CG1/descriptives/table.csv"
            produced.parent.mkdir(parents=True)
            produced.write_text("promoted\n")
            (run / "run.json").write_text(json.dumps({"stages": {"notebook/05": {}}}))
            self.assertEqual(promote(root, "new-run", "CG1/descriptives")["promoted_files"], 1)
            self.assertEqual(canonical.read_text(), "promoted\n")
            self.assertEqual(digest(archive), archived_hash)
            self.assertNotEqual(canonical.stat().st_ino, archive.stat().st_ino)
            self.assertEqual(verify(root)["promoted_files"], 1)
            self.assertEqual(len(json.loads((root / "provenance/canonical-run.json").read_text())["promotion_history"]), 1)


if __name__ == "__main__":
    unittest.main()
