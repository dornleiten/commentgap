"""Repository-rooted paths and safe execution destinations.

This module is deliberately dependency-free so command-line entry points,
notebooks, and small maintenance tools all use the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable, Mapping



def _execution_configuration(root: Path) -> dict[str, object]:
    """Capture scientific options without persisting environment secret values."""
    import hashlib
    import sys
    operational = {"COMMENTGAP_MODE", "COMMENTGAP_RUN_ID", "COMMENTGAP_PYTHON",
                   "COMMENTGAP_FROZEN_REPLAY", "COMMENTGAP_STAGE_PREPARED"}
    environment = {
        name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in sorted(os.environ.items())
        if name.startswith("COMMENTGAP_") and name not in operational
        and not name.endswith(("_ROOT", "_PATH", "_STORE", "_OUTPUT", "_DIR"))
    }
    executable = Path(sys.argv[0]).resolve()
    is_cli = (executable.parent == root / "scripts" or
              executable.parent == root / "commentgap_analysis" or
              executable.name.startswith("commentgap-"))
    arguments = []
    if is_cli and executable.name != "path_contract.py":
        skip = False
        for argument in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if argument == "--mode":
                skip = True
                continue
            if argument.startswith("--mode="):
                continue
            arguments.append(argument)
    return {"environment_sha256": environment,
            "cli_arguments_sha256": hashlib.sha256(json.dumps(arguments).encode()).hexdigest()}


RUN_INPUT_AREAS = {
    "raw_scrape": "shared/raw_scrape",
    "aqua": "shared/aqua", "embeddings": "shared/embeddings",
    "similarities": "shared/similarities", "features": "shared/features",
    "model_data": "shared/model_data",
    "frozen_cg1_feature_diagnostics": "CG1/feature_diagnostics",
    "frozen_cg1_regression": "CG1/regression",
    "frozen_cg1_sensitivity": "CG1/sensitivity",
    "frozen_cg1_factorial_rankers": "CG1/rankers/factorial",
    "frozen_cg1_neural_rankers": "CG1/rankers/neural",
    "frozen_cg1_descriptives": "CG1/descriptives",
    "frozen_cg1_comment_gap": "CG1/comment_gap",
    "frozen_cg1_winners": "CG1/winners", "frozen_cg1_reporting": "CG1/reporting",
    "frozen_cg2_forum": "CG2/forum",
    "frozen_cg2_topic_source": "CG2/topic_source",
    "frozen_cg2_ranking_similarity": "CG2/ranking_similarity",
    "frozen_cg2_topics_runs": "CG2/topics/runs",
    "frozen_cg2_topics_diagnostics": "CG2/topics/diagnostics",
}

CONFIG_RELATIVE_PATH = Path("config") / "paths.json"
MODES = frozenset({"frozen", "replay", "recompute", "fresh", "resume"})
PROTECTED_RELATIVE_ROOTS = (
    Path("data"),
    Path("artifacts") / "canonical",
    Path("artifacts") / "archive",
    Path("artifacts") / "frozen",
    Path("artifacts") / "reconstructed",
    Path("submissions"),
    Path("provenance"),
    # Until relocation has completed, historical model output is input-only.
    Path("model_output"),
    Path("quarantine"),
)


class PathContractError(ValueError):
    """A path or execution-mode request violates the repository contract."""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_with_missing(path: Path) -> Path:
    """Resolve symlinks in every existing ancestor without requiring a target."""
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    missing: list[str] = []
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    resolved = ancestor.resolve(strict=ancestor.exists())
    for part in reversed(missing):
        resolved /= part
    return resolved


def discover_repository_root(start: Path | str | None = None) -> Path:
    """Find the repository by its tracked path configuration, never by CWD alone."""
    configured_root = os.environ.get("COMMENTGAP_PROJECT_ROOT")
    if configured_root:
        candidate = resolve_user_path(configured_root, cwd=start)
        if (candidate / CONFIG_RELATIVE_PATH).is_file():
            return candidate
        raise PathContractError(
            f"COMMENTGAP_PROJECT_ROOT does not contain {CONFIG_RELATIVE_PATH}: {candidate}"
        )
    current = Path.cwd() if start is None else Path(start)
    current = _resolve_with_missing(current)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / CONFIG_RELATIVE_PATH).is_file():
            return candidate
    # Installed/editable entry points may start in an unrelated caller directory.
    module_root = Path(__file__).resolve().parents[1]
    if (module_root / CONFIG_RELATIVE_PATH).is_file():
        return module_root
    raise PathContractError(f"Could not find {CONFIG_RELATIVE_PATH} above {current}; pass --repo-root.")


def resolve_user_path(value: Path | str, *, cwd: Path | str | None = None) -> Path:
    """Resolve a CLI/function/environment override relative to its caller CWD."""
    value = Path(value).expanduser()
    if not value.is_absolute():
        value = (Path.cwd() if cwd is None else Path(cwd)) / value
    return _resolve_with_missing(value)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    entries: Mapping[str, object]

    @classmethod
    def load(
        cls,
        *,
        repo_root: Path | str | None = None,
        config_path: Path | str | None = None,
        cwd: Path | str | None = None,
    ) -> "ProjectPaths":
        root = (
            resolve_user_path(repo_root, cwd=cwd)
            if repo_root is not None
            else discover_repository_root(cwd)
        )
        config = (
            resolve_user_path(config_path, cwd=cwd)
            if config_path is not None
            else root / CONFIG_RELATIVE_PATH
        )
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PathContractError(f"Missing path configuration: {config}") from exc
        except json.JSONDecodeError as exc:
            raise PathContractError(f"Invalid JSON in path configuration {config}: {exc}") from exc
        if payload.get("schema_version") != 1 or not isinstance(payload.get("paths"), dict):
            raise PathContractError(f"Unsupported path configuration: {config}")
        return cls(root=_resolve_with_missing(root), entries=payload["paths"])

    def _entry_paths(self, key: str) -> tuple[Path, tuple[Path, ...]]:
        try:
            entry = self.entries[key]
        except KeyError as exc:
            raise PathContractError(f"Unknown configured path key: {key}") from exc
        if isinstance(entry, str):
            return self.root / entry, ()
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise PathContractError(f"Invalid path entry for {key!r}")
        legacy = entry.get("legacy", [])
        if not isinstance(legacy, list) or not all(isinstance(value, str) for value in legacy):
            raise PathContractError(f"Invalid legacy paths for {key!r}")
        return self.root / entry["path"], tuple(self.root / value for value in legacy)

    def configured(self, key: str) -> Path:
        """Return the future layout path, irrespective of current relocation state."""
        primary, _ = self._entry_paths(key)
        return _resolve_with_missing(primary)

    def read_root(
        self,
        key: str,
        *,
        explicit: Path | str | None = None,
        env_var: str | None = None,
        cwd: Path | str | None = None,
        mode: str | None = None,
    ) -> Path:
        """Resolve explicit > environment > configured existing > legacy existing."""
        if explicit is not None:
            return self._resolve_read_override(explicit, cwd=cwd)
        if env_var and os.environ.get(env_var):
            return self._resolve_read_override(os.environ[env_var], cwd=cwd)
        primary, legacy = self._entry_paths(key)
        environment_mode = os.environ.get("COMMENTGAP_MODE", "").lower()
        requested_mode = (mode or environment_mode).lower()
        # The R Markdown saved-product branch keeps its historical internal
        # mode name, while its external mode selects the private archive.
        if requested_mode == "frozen" and environment_mode == "recompute":
            requested_mode = "recompute"
        use_private = requested_mode == "recompute" or (
            requested_mode in {"fresh", "resume"}
            and key.startswith(("frozen_", "reconstruction_"))
        )
        if use_private:
            canonical = self.root / "artifacts/canonical"
            try:
                relative = primary.relative_to(canonical)
            except ValueError:
                pass
            else:
                private = self.root / "artifacts/archive/recompute" / relative
                if not private.exists():
                    raise FileNotFoundError(f"Private recompute artifact unavailable: {private}")
                return _resolve_with_missing(private)
        for candidate in (primary, *legacy):
            if candidate.exists():
                return _resolve_with_missing(candidate)
        # A relocation registry may point an old configured directory at its
        # retained destination.  This matters after a move because manifests
        # and old configuration intentionally keep their original paths.
        configured_entries = self.entries[key]
        if isinstance(configured_entries, str):
            names = (configured_entries,)
        else:
            names = (configured_entries["path"], *configured_entries.get("legacy", []))
        for name in names:
            try:
                return resolve_artifact_path(name, paths=self)
            except (FileNotFoundError, PathContractError):
                continue
        return _resolve_with_missing(primary)

    def _resolve_read_override(self, value: Path | str, *, cwd: Path | str | None) -> Path:
        candidate = resolve_user_path(value, cwd=cwd)
        if _is_quarantine_path(candidate):
            raise PathContractError(f"Refusing quarantined input path: {candidate}")
        try:
            return resolve_artifact_path(candidate, paths=self)
        except FileNotFoundError:
            return candidate

    def protected_roots(self) -> tuple[Path, ...]:
        return tuple(_resolve_with_missing(self.root / relative) for relative in PROTECTED_RELATIVE_ROOTS)

    def require_writable(
        self,
        destination: Path | str,
        *,
        cwd: Path | str | None = None,
        additional_protected: tuple[Path | str, ...] = (),
    ) -> Path:
        destination = resolve_user_path(destination, cwd=cwd)
        protected_roots = [*self.protected_roots()]
        protected_roots.extend(_resolve_with_missing(path) for path in additional_protected)
        for protected in protected_roots:
            if _is_relative_to(destination, protected):
                raise PathContractError(
                    f"Refusing to write protected input path {destination} (under {protected})"
                )
        return destination


def require_writable_destination(
    destination: Path | str,
    *,
    repo_root: Path | str | None = None,
    cwd: Path | str | None = None,
) -> Path:
    """Guard a public library writer, including callers that bypass a CLI."""
    return ProjectPaths.load(repo_root=repo_root, cwd=cwd).require_writable(destination, cwd=cwd)


def resolve_artifact_path(
    recorded: Path | str,
    *,
    paths: ProjectPaths | None = None,
    anchors: tuple[Path | str, ...] = (),
    expected_sha256: str | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve an immutable path recorded in a historical manifest.

    The relocation registry applies exact file mappings, followed by its
    longest matching directory prefix.  Existing paths are used only when no
    registered mapping applies, so an old path that happens to remain present
    cannot shadow the recorded relocation destination.
    The registry is read-only and this function deliberately never substitutes
    a newest run or a quarantined source.
    """
    paths = paths or ProjectPaths.load()
    raw = Path(recorded).expanduser()
    if _is_quarantine_path(raw):
        raise PathContractError(f"Refusing quarantined artifact: {recorded}")
    candidates: list[Path] = [raw] if raw.is_absolute() else []
    if not raw.is_absolute():
        candidates.extend(Path(anchor) / raw for anchor in anchors)
        candidates.append(paths.root / raw)
    registry_path = paths.root / "provenance" / "relocation.json"
    registry = _load_relocation_registry(registry_path)
    mapping_candidates = [str(raw), *(str(candidate) for candidate in candidates)]
    for source in mapping_candidates:
        target = registry["files"].get(source)
        if target is not None:
            return _verify_artifact(_registry_target(paths, target), expected_sha256, require_exists)
    for source in mapping_candidates:
        source_paths = [Path(source)]
        source_path = source_paths[0]
        if source_path.is_absolute():
            try:
                source_paths.append(source_path.relative_to(paths.root))
            except ValueError:
                pass
        matches: list[tuple[int, str, str]] = []
        for prefix, target in registry["prefixes"].items():
            prefix_path = Path(prefix)
            for candidate_path in source_paths:
                try:
                    suffix = candidate_path.relative_to(prefix_path)
                except ValueError:
                    continue
                matches.append((len(prefix_path.parts), target, str(suffix)))
        if matches:
            _, target, suffix = max(matches, key=lambda item: item[0])
            return _verify_artifact(
                _registry_target(paths, target) / suffix, expected_sha256, require_exists
            )
    for candidate in candidates:
        if candidate.exists():
            return _verify_artifact(_resolve_with_missing(candidate), expected_sha256)
    if require_exists:
        raise FileNotFoundError(f"Recorded artifact cannot be resolved: {recorded}")
    return _resolve_with_missing(candidates[-1])


def _load_relocation_registry(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {"files": {}, "prefixes": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PathContractError(f"Invalid relocation registry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PathContractError(f"Invalid relocation registry {path}")
    result: dict[str, dict[str, str]] = {"files": {}, "prefixes": {}}
    for key in result:
        section = payload.get(key, {})
        if not isinstance(section, dict) or not all(
            isinstance(source, str) and isinstance(target, str)
            for source, target in section.items()
        ):
            raise PathContractError(f"Invalid {key} mappings in {path}")
        result[key] = dict(section)
    # A row-oriented registry is convenient for inventory tooling.
    mappings = payload.get("mappings", [])
    if mappings:
        if not isinstance(mappings, list):
            raise PathContractError(f"Invalid mappings in {path}")
        for row in mappings:
            if not isinstance(row, dict) or not isinstance(row.get("source"), str) or not isinstance(row.get("destination"), str):
                raise PathContractError(f"Invalid mapping row in {path}")
            kind = "prefixes" if row.get("kind") == "prefix" else "files"
            result[kind][row["source"]] = row["destination"]
    return result


def _registry_target(paths: ProjectPaths, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    return _resolve_with_missing(candidate if candidate.is_absolute() else paths.root / candidate)


def _verify_artifact(path: Path, expected_sha256: str | None, require_exists: bool = True) -> Path:
    for parent in path.parents:
        if parent.name == "quarantine":
            raise PathContractError(f"Refusing quarantined artifact: {path}")
    if require_exists and not path.exists():
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        observed = _path_digest(path)
        if observed != expected_sha256:
            raise PathContractError(f"Artifact hash differs from manifest for {path}")
    return path


def _is_quarantine_path(path: Path) -> bool:
    return any(part == "quarantine" for part in path.parts)


@dataclass
class ExecutionContext:
    paths: ProjectPaths
    mode: str
    run_id: str
    cwd: Path
    read_inputs: list[Path]
    _prepared: bool = False
    _prepared_stages: set[str] | None = None

    @classmethod
    def from_values(
        cls,
        *,
        mode: str | None = None,
        run_id: str | None = None,
        repo_root: Path | str | None = None,
        cwd: Path | str | None = None,
    ) -> "ExecutionContext":
        caller_cwd = Path.cwd() if cwd is None else Path(cwd)
        selected_mode = (mode or os.getenv("COMMENTGAP_MODE", "replay")).lower()
        if selected_mode not in MODES:
            raise PathContractError("COMMENTGAP_MODE/--mode must be replay, recompute, fresh, or resume")
        selected_run_id = run_id or os.getenv("COMMENTGAP_RUN_ID")
        if selected_mode in {"fresh", "resume"} and not selected_run_id:
            raise PathContractError(f"{selected_mode} mode requires --run-id or COMMENTGAP_RUN_ID")
        if selected_run_id is None:
            selected_run_id = "frozen-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if Path(selected_run_id).name != selected_run_id or selected_run_id in {"", ".", ".."}:
            raise PathContractError("run ID must be a single safe path component")
        return cls(
            paths=ProjectPaths.load(repo_root=repo_root, cwd=caller_cwd),
            mode=selected_mode,
            run_id=selected_run_id,
            cwd=caller_cwd,
            read_inputs=[],
            _prepared_stages=set(),
        )

    def read_root(
        self,
        key: str,
        *,
        explicit: Path | str | None = None,
        env_var: str | None = None,
    ) -> Path:
        if self.mode in {"fresh", "resume"} and explicit is None and not (env_var and os.environ.get(env_var)):
            area = RUN_INPUT_AREAS.get(key)
            if area is not None:
                candidate = self.run_path(area)
                if candidate.exists():
                    explicit = candidate
        resolved = self.paths.read_root(key, explicit=explicit, env_var=env_var,
                                        cwd=self.cwd, mode=self.mode)
        if resolved not in self.read_inputs:
            self.read_inputs.append(resolved)
        return resolved

    def read_path(self, value: Path | str) -> Path:
        """Record a concrete immutable input path supplied by a caller."""
        resolved = self.paths._resolve_read_override(value, cwd=self.cwd)
        if resolved not in self.read_inputs:
            self.read_inputs.append(resolved)
        return resolved

    def run_path(self, area: str) -> Path:
        """Return a path in this run for a later stage to consume or write."""
        return self.paths.configured("outputs") / self.run_id / area

    def run_input(self, area: str) -> Path:
        """Return and record a path produced by an earlier stage in this run."""
        return self.read_path(self.run_path(area))

    def _identity(self, contract: Mapping[str, object] | None = None) -> dict[str, object]:
        config_path = self.paths.root / CONFIG_RELATIVE_PATH
        config_sha256 = _sha256(config_path)
        code_identity = _code_identity(self.paths.root)
        input_hashes = {
            str(path): _path_digest(path) for path in sorted(self.read_inputs, key=str)
        }
        return {
            "schema_version": 1,
            "config_sha256": config_sha256,
            "code_identity": code_identity,
            "inputs": [str(path) for path in self.read_inputs],
            "input_hashes": input_hashes,
            "contract": dict(contract or {}),
            "execution_configuration": _execution_configuration(self.paths.root),
        }

    def prepare_run(
        self,
        *,
        contract: Mapping[str, object] | None = None,
        stage: str | None = None,
    ) -> Path:
        """Create or validate a run identity, optionally scoped to one stage."""
        run_root = self.paths.require_writable(
            self.paths.configured("outputs") / self.run_id,
            cwd=self.cwd,
            additional_protected=tuple(self.read_inputs),
        )
        identity = self._identity(contract)
        manifest = run_root / "run.json"
        if stage is not None:
            if self._prepared_stages is None:
                self._prepared_stages = set()
            if stage in self._prepared_stages:
                return run_root
            common = {
                "schema_version": 2,
                "config_sha256": identity["config_sha256"],
                "code_identity": identity["code_identity"],
            }
            stage_identity = {
                "inputs": identity["inputs"],
                "input_hashes": identity["input_hashes"],
                "contract": identity["contract"],
                "execution_configuration": identity["execution_configuration"],
            }
            if self.mode == "fresh":
                if manifest.exists():
                    try:
                        recorded = json.loads(manifest.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        raise PathContractError(f"Invalid run manifest: {manifest}") from exc
                    if recorded.get("schema_version") != 2:
                        raise PathContractError(
                            f"Fresh run has a legacy manifest; use a new --run-id: {manifest}"
                        )
                    if recorded.get("common") != common:
                        raise PathContractError("Run code/config identity changed")
                    stages = recorded.setdefault("stages", {})
                    if stage in stages:
                        raise PathContractError(f"Fresh stage already exists: {stage}")
                    stages[stage] = stage_identity
                    _atomic_json_write(manifest, recorded)
                else:
                    run_root.mkdir(parents=True, exist_ok=True)
                    _atomic_json_write(
                        manifest,
                        {"schema_version": 2, "common": common, "stages": {stage: stage_identity}},
                    )
            elif self.mode == "resume":
                if not manifest.is_file():
                    raise PathContractError(f"Resume run has no run manifest: {manifest}")
                try:
                    recorded = json.loads(manifest.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise PathContractError(f"Invalid resume manifest: {manifest}") from exc
                if recorded.get("schema_version") != 2:
                    raise PathContractError("Resume run has a legacy manifest; cannot resume a staged run")
                if recorded.get("common") != common:
                    raise PathContractError("Resume code/config identity differs")
                stages = recorded.get("stages", {})
                if stage not in stages:
                    raise PathContractError(f"Resume run has no stage manifest: {stage}")
                if stages[stage] != stage_identity:
                    raise PathContractError("Resume stage identity differs (inputs or contract changed)")
            self._prepared_stages.add(stage)
            self._prepared = True
            return run_root
        if self.mode == "fresh":
            if not self._prepared and run_root.exists():
                raise PathContractError(
                    f"Fresh run ID already exists: {run_root}; choose a new --run-id."
                )
            run_root.mkdir(parents=True, exist_ok=True)
            if not self._prepared:
                _atomic_json_write(manifest, identity)
            elif getattr(self, "_prepared_identity", None) != identity:
                raise PathContractError("Run identity changed after preparation")
        elif self.mode == "resume":
            if not manifest.is_file():
                raise PathContractError(f"Resume run has no run manifest: {manifest}")
            try:
                recorded = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise PathContractError(f"Invalid resume manifest: {manifest}") from exc
            if recorded != identity:
                raise PathContractError(
                    "Resume identity differs from the existing run (configuration, code, inputs, or contract changed)."
                )
        self._prepared_identity = identity
        self._prepared = True
        return run_root

    def output_root(
        self,
        area: str,
        *,
        explicit: Path | str | None = None,
        env_var: str | None = None,
        contract: Mapping[str, object] | None = None,
    ) -> Path:
        if self.mode in {"frozen", "replay", "recompute"}:
            raise PathContractError(
                "Saved-product modes cannot run a computational producer; choose --mode fresh or resume."
            )
        run_root = self.prepare_run(contract=contract, stage=area)
        expected = run_root / area
        if explicit is not None:
            candidate = resolve_user_path(explicit, cwd=self.cwd)
        elif env_var and os.environ.get(env_var):
            candidate = resolve_user_path(os.environ[env_var], cwd=self.cwd)
        else:
            candidate = expected
        if candidate != _resolve_with_missing(expected):
            raise PathContractError(
                f"CLI output must remain under this run: {expected}; direct library APIs may use guarded fixture paths."
            )
        return self.paths.require_writable(
            candidate, cwd=self.cwd, additional_protected=tuple(self.read_inputs)
        )

    def staging_output(
        self,
        area: str,
        *,
        explicit: Path | str | None = None,
        env_var: str | None = None,
    ) -> Path:
        """Return a guarded presentation/staging root in every execution mode."""
        run_root = self.paths.configured("outputs") / self.run_id
        expected = run_root / area
        if explicit is not None:
            candidate = resolve_user_path(explicit, cwd=self.cwd)
        elif env_var and os.environ.get(env_var):
            candidate = resolve_user_path(os.environ[env_var], cwd=self.cwd)
        else:
            candidate = expected
        if candidate != _resolve_with_missing(expected):
            raise PathContractError(f"CLI staging output must remain under this run: {expected}")
        return self.paths.require_writable(
            candidate, cwd=self.cwd, additional_protected=tuple(self.read_inputs)
        )


def add_execution_arguments(parser: object) -> None:
    """Add the common CLI controls without importing argparse at module import time."""
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--mode", choices=tuple(sorted(MODES)), default=None)
    parser.add_argument("--run-id", default=None)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    """Return a content digest for a file or deterministic directory tree."""
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        return "missing"
    digest = __import__("hashlib").sha256()
    for child in sorted(path.rglob("*"), key=lambda item: str(item.relative_to(path))):
        relative = str(child.relative_to(path))
        if child.is_symlink():
            payload = f"L:{relative}:{os.readlink(child)}\n".encode()
            digest.update(payload)
        elif child.is_file():
            digest.update(f"F:{relative}:".encode())
            digest.update(bytes.fromhex(_sha256(child)))
    return digest.hexdigest()


def _code_identity(root: Path) -> str:
    """Hash current source/configuration bytes, including uncommitted files."""
    suffixes = {".py", ".R", ".Rmd", ".ipynb", ".json", ".toml", ".yaml", ".yml"}
    excluded = {".git", ".venv", ".pytest_cache", "__pycache__", "outputs", "data", "artifacts", "model_output"}
    files: set[Path] = set()
    for base in (root / "commentgap_analysis", root / "commentgap_scraper", root / "aqua_runtime", root / "scripts"):
        if base.is_dir():
            files.update(path for path in base.rglob("*") if path.is_file() and path.suffix in suffixes)
    files.update(path for path in root.iterdir() if path.is_file() and path.suffix in suffixes)
    digest = __import__("hashlib").sha256()
    for path in sorted(files):
        if any(part in excluded for part in path.relative_to(root).parts):
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
