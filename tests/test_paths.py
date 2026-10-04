from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from commentgap_analysis.paths import (
    ExecutionContext,
    PathContractError,
    ProjectPaths,
    discover_repository_root,
    resolve_artifact_path,
)


def _project(root: Path) -> Path:
    (root / "config").mkdir(parents=True)
    (root / "config" / "paths.json").write_text(json.dumps({
        "schema_version": 1,
        "paths": {
            "outputs": "outputs",
            "raw_scrape": {"path": "data/raw/scrape_2025", "legacy": ["data/scrape_2025"]},
        },
    }))
    return root


class ProjectPathTests(unittest.TestCase):
    def test_discovers_root_from_non_root_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            nested = root / "work" / "nested"
            nested.mkdir(parents=True)
            self.assertEqual(discover_repository_root(nested), root.resolve())

    def test_unrelated_cwd_falls_back_to_imported_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            unrelated = Path(temporary)
            self.assertTrue((discover_repository_root(unrelated) / "config/paths.json").is_file())

    def test_read_precedence_and_relocation_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            legacy = root / "data" / "scrape_2025"
            legacy.mkdir(parents=True)
            paths = ProjectPaths.load(repo_root=root)
            self.assertEqual(paths.read_root("raw_scrape"), legacy.resolve())
            with mock.patch.dict(os.environ, {"CG_TEST_ROOT": "environment"}):
                self.assertEqual(
                    paths.read_root("raw_scrape", env_var="CG_TEST_ROOT", cwd=root),
                    (root / "environment").resolve(),
                )
            self.assertEqual(
                paths.read_root("raw_scrape", explicit="argument", cwd=root),
                (root / "argument").resolve(),
            )

    def test_fresh_requires_run_id_and_output_is_new_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            with self.assertRaisesRegex(PathContractError, "requires --run-id"):
                ExecutionContext.from_values(mode="fresh", repo_root=root)
            context = ExecutionContext.from_values(mode="fresh", run_id="trial-1", repo_root=root)
            self.assertEqual(context.output_root("shared/features"), root / "outputs/trial-1/shared/features")
            with self.assertRaisesRegex(PathContractError, "already exists"):
                ExecutionContext.from_values(mode="fresh", run_id="trial-1", repo_root=root).output_root("shared/features")

    def test_fresh_stages_share_run_identity_but_record_separate_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            first_input = root / "first.txt"
            second_input = root / "second.txt"
            first_input.write_text("one")
            second_input.write_text("two")
            first = ExecutionContext.from_values(mode="fresh", run_id="trial", repo_root=root)
            first.read_path(first_input)
            first.output_root("shared/features")
            second = ExecutionContext.from_values(mode="fresh", run_id="trial", repo_root=root)
            second.read_path(second_input)
            second.output_root("shared/model_data")
            manifest = json.loads((root / "outputs/trial/run.json").read_text())
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(set(manifest["stages"]), {"shared/features", "shared/model_data"})

    def test_guard_blocks_protected_and_symlink_aliases_but_not_temp_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            protected = root / "data"
            protected.mkdir()
            alias = root / "alias"
            alias.symlink_to(protected, target_is_directory=True)
            paths = ProjectPaths.load(repo_root=root)
            for target in (protected / "new.csv", alias / "new.csv", root / "model_output/x"):
                with self.assertRaises(PathContractError):
                    paths.require_writable(target)
            self.assertEqual(paths.require_writable(Path(temporary) / "outside"), Path(temporary) / "outside")

    def test_frozen_rejects_computational_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            context = ExecutionContext.from_values(mode="frozen", repo_root=root)
            with self.assertRaisesRegex(PathContractError, "Saved-product modes"):
                context.output_root("shared/embeddings")

    def test_frozen_stages_presentation_in_a_new_output_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            context = ExecutionContext.from_values(
                mode="frozen", run_id="replay-1", repo_root=root
            )
            self.assertEqual(
                context.staging_output("rendered/CG1"),
                root / "outputs/replay-1/rendered/CG1",
            )

    def test_context_rejects_an_output_alias_of_an_external_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            external = Path(temporary) / "external-input"
            external.mkdir()
            context = ExecutionContext.from_values(mode="fresh", run_id="trial", repo_root=root)
            context.read_root("raw_scrape", explicit=external)
            with self.assertRaises(PathContractError):
                context.output_root("shared/test", explicit=external / "outputs")

    def test_resume_rejects_changed_scientific_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            with mock.patch.dict(os.environ, {"COMMENTGAP_MODEL_SEED": "123"}):
                ExecutionContext.from_values(mode="fresh", run_id="run", repo_root=root).prepare_run(stage="fit")
            with mock.patch.dict(os.environ, {"COMMENTGAP_MODEL_SEED": "456"}):
                with self.assertRaisesRegex(PathContractError, "identity differs"):
                    ExecutionContext.from_values(mode="resume", run_id="run", repo_root=root).prepare_run(stage="fit")

    def test_resume_checks_cli_options_while_allowing_mode_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            command = str(root / "scripts/fit.py")
            with mock.patch("sys.argv", [command, "--mode", "fresh", "--seed", "123"]):
                ExecutionContext.from_values(mode="fresh", run_id="run", repo_root=root).prepare_run(stage="fit")
            with mock.patch("sys.argv", [command, "--mode", "resume", "--seed", "123"]):
                ExecutionContext.from_values(mode="resume", run_id="run", repo_root=root).prepare_run(stage="fit")
            with mock.patch("sys.argv", [command, "--mode", "resume", "--seed", "456"]):
                with self.assertRaisesRegex(PathContractError, "identity differs"):
                    ExecutionContext.from_values(mode="resume", run_id="run", repo_root=root).prepare_run(stage="fit")

    def test_fresh_reader_prefers_materialized_run_and_preserves_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            retained = root / "data/raw/scrape_2025"
            retained.mkdir(parents=True)
            produced = root / "outputs/run/shared/raw_scrape"
            produced.mkdir(parents=True)
            context = ExecutionContext.from_values(mode="fresh", run_id="run", repo_root=root)
            self.assertEqual(context.read_root("raw_scrape"), produced)
            self.assertEqual(context.read_root("raw_scrape", explicit=retained), retained)
            frozen = ExecutionContext.from_values(mode="frozen", run_id="run", repo_root=root)
            self.assertEqual(frozen.read_root("raw_scrape"), retained)

    def test_resume_requires_exact_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            fresh = ExecutionContext.from_values(mode="fresh", run_id="run", repo_root=root)
            fresh.prepare_run(contract={"input_hashes": {"fixture": "one"}})
            resumed = ExecutionContext.from_values(mode="resume", run_id="run", repo_root=root)
            resumed.prepare_run(contract={"input_hashes": {"fixture": "one"}})
            changed = ExecutionContext.from_values(mode="resume", run_id="run", repo_root=root)
            with self.assertRaisesRegex(PathContractError, "identity differs"):
                changed.prepare_run(contract={"input_hashes": {"fixture": "two"}})

    def test_resume_rejects_changed_input_bytes_with_same_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            source = root / "input.txt"
            source.write_text("before")
            fresh = ExecutionContext.from_values(mode="fresh", run_id="run", repo_root=root)
            fresh.read_path(source)
            fresh.prepare_run()
            source.write_text("after")
            resumed = ExecutionContext.from_values(mode="resume", run_id="run", repo_root=root)
            resumed.read_path(source)
            with self.assertRaisesRegex(PathContractError, "identity differs"):
                resumed.prepare_run()

    def test_relocation_reader_prefers_exact_then_longest_prefix_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            (root / "provenance").mkdir()
            destination = root / "artifacts" / "frozen" / "one.txt"
            destination.parent.mkdir(parents=True)
            destination.write_text("retained")
            (root / "provenance" / "relocation.json").write_text(json.dumps({
                "schema_version": 1,
                "files": {"old/special.txt": "artifacts/frozen/one.txt"},
                "prefixes": {"old": "artifacts/frozen", "old/deeper": "artifacts/frozen"},
            }))
            paths = ProjectPaths.load(repo_root=root)
            self.assertEqual(resolve_artifact_path("old/special.txt", paths=paths), destination)
            self.assertEqual(
                resolve_artifact_path("old/deeper/one.txt", paths=paths), destination
            )
            with self.assertRaises(PathContractError):
                resolve_artifact_path("old/special.txt", paths=paths, expected_sha256="0" * 64)

    def test_relocation_mapping_wins_even_when_old_path_still_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary) / "repo")
            old = root / "old" / "value.txt"
            old.parent.mkdir()
            old.write_text("stale")
            destination = root / "artifacts" / "frozen" / "value.txt"
            destination.parent.mkdir(parents=True)
            destination.write_text("retained")
            (root / "provenance").mkdir()
            (root / "provenance" / "relocation.json").write_text(json.dumps({
                "schema_version": 1,
                "files": {str(old): "artifacts/frozen/value.txt"},
                "prefixes": {},
            }))
            paths = ProjectPaths.load(repo_root=root)
            self.assertEqual(resolve_artifact_path(old, paths=paths), destination)


if __name__ == "__main__":
    unittest.main()
