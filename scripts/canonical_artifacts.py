#!/usr/bin/env python3
"""Build, verify, or explicitly promote the local canonical artifact run.

The initial build uses hard links so the 82 GB historical bundle does not need
another 82 GB of disk. Shared files are made read-only. Promotions copy into a
temporary file and replace it atomically, leaving the archived inode intact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path("artifacts/canonical")
MANIFEST = Path("provenance/canonical-run.json")
PRIVATE_CANONICAL = Path("artifacts/archive/recompute")
PRIVATE_MANIFEST = Path("provenance/private-canonical-run.json")
SOURCES = (
    ("frozen", Path("artifacts/frozen")),
    ("reconstructed-20260930", Path("artifacts/reconstructed/20260930")),
    ("reconstructed-20261003", Path("artifacts/reconstructed/20261003")),
)
AREAS = {
    "shared/raw_scrape", "shared/features", "shared/model_data",
    "shared/embeddings", "shared/similarities", "shared/aqua",
    "CG1/feature_diagnostics", "CG1/descriptives", "CG1/comment_gap",
    "CG1/rankers/factorial", "CG1/rankers/neural", "CG1/winners",
    "CG1/regression", "CG1/sensitivity", "CG1/reporting",
    "CG2/forum", "CG2/ranking_similarity", "CG2/topics/runs",
    "CG2/forum/inference", "CG2/forum/reporting", "CG2/forum/policy_scores",
    "CG2/topics",
    "CG2/topics/diagnostics", "CG2/topics/analysis", "CG2/topics/reporting",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Artifact source contains a symlink: {path}")
        if path.is_file():
            yield path


def save_manifest(root: Path, payload: dict, path: Path = MANIFEST) -> None:
    target = root / path
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def build(root: Path) -> dict:
    canonical = root / CANONICAL
    staging = root / "artifacts/.canonical-building"
    if canonical.exists() or staging.exists():
        raise FileExistsError(f"Canonical or staging tree already exists: {canonical}, {staging}")
    if (root / MANIFEST).exists():
        raise FileExistsError(f"Canonical manifest already exists: {root / MANIFEST}")
    source_roots = [(name, root / path) for name, path in SOURCES]
    for name, source in source_roots:
        if not source.is_dir() or source.is_symlink():
            raise FileNotFoundError(f"Expected original {name} directory: {source}")
    staging.mkdir()
    counts: dict[str, int] = {}
    replacements: list[dict[str, str]] = []
    try:
        for name, source in source_roots:
            count = 0
            for item in files(source):
                relative = item.relative_to(source)
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    old_hash, new_hash = digest(destination), digest(item)
                    if old_hash != new_hash:
                        replacements.append({"path": str(relative), "source": name,
                                             "replaced_sha256": old_hash, "sha256": new_hash})
                    destination.unlink()
                os.link(item, destination)
                count += 1
            counts[name] = count
        # Read-only shared inodes prevent accidental in-place edits of the
        # archive through the canonical tree. Promotion uses os.replace.
        for item in files(staging):
            item.chmod(stat.S_IMODE(item.stat().st_mode) & ~0o222)
        os.rename(staging, canonical)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    archive = root / "artifacts/archive"
    archive.mkdir(exist_ok=True)
    for name, source in (("frozen", root / "artifacts/frozen"),
                         ("reconstructed", root / "artifacts/reconstructed")):
        destination = archive / name
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(source, destination)
        source.symlink_to(Path("archive") / name, target_is_directory=True)

    manifest = {
        "schema_version": 1,
        "canonical_root": str(CANONICAL),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_bundle": "provenance/artifact-bundle.json",
        "baseline_file_manifest": "provenance/artifact-files.csv.gz",
        "archive_roots": ["artifacts/archive/frozen", "artifacts/archive/reconstructed"],
        "source_precedence": [name for name, _ in SOURCES],
        "source_file_counts": counts,
        "different_name_collisions": replacements,
        "promotion_history": [],
        "storage": "Original and canonical files initially share read-only inodes; promotions replace canonical files atomically.",
    }
    save_manifest(root, manifest)
    return manifest


def verify_public(root: Path) -> dict:
    public = root / CANONICAL
    manifest = json.loads((public / "replay_manifest.json").read_text())
    expected = manifest["products"]
    actual = {str(path.relative_to(public)): path for path in files(public)
              if path.name not in {"replay_manifest.json", ".gitignore"}}
    if set(actual) != set(expected):
        raise ValueError(f"Public replay inventory differs: missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}")
    for relative, record in expected.items():
        path = actual[relative]
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Public replay product changed: {relative}")
        if path.stat().st_size >= 50 * 1024 * 1024:
            raise ValueError(f"Public replay product exceeds 50 MiB: {relative}")
    return {"public_files": len(actual), "public_bytes": sum(path.stat().st_size for path in actual.values())}


def verify(root: Path, *, private: bool = False) -> dict:
    is_public = (root / CANONICAL / "replay_manifest.json").is_file() and not private
    if is_public:
        return verify_public(root)
    manifest = json.loads((root / (PRIVATE_MANIFEST if private else MANIFEST)).read_text())
    canonical = root / (PRIVATE_CANONICAL if private else CANONICAL)
    if not canonical.is_dir():
        raise FileNotFoundError(canonical)
    archive = root / "artifacts/archive"
    layers = (archive / "frozen", archive / "reconstructed/20260930",
              archive / "reconstructed/20261003")
    expected: dict[Path, Path] = {}
    for layer in layers:
        if not layer.is_dir():
            raise FileNotFoundError(layer)
        for item in files(layer):
            expected[item.relative_to(layer)] = item
    promoted: dict[Path, str] = {}
    for event in manifest["promotion_history"]:
        for item in event["files"]:
            promoted[Path(item["path"])] = item["sha256"]
    actual = {item.relative_to(canonical): item for item in files(canonical)}
    if set(actual) != set(expected) | set(promoted):
        raise ValueError(f"Canonical file inventory differs: missing={len((set(expected) | set(promoted)) - set(actual))}, extra={len(set(actual) - (set(expected) | set(promoted)))}")
    for relative, source in expected.items():
        if relative not in promoted and not actual[relative].samefile(source):
            raise ValueError(f"Canonical file no longer shares its archived source: {relative}")
    for relative, hash_ in promoted.items():
        if digest(actual[relative]) != hash_:
            raise ValueError(f"Promoted file checksum differs: {relative}")
    return {"canonical_files": len(actual), "source_files": sum(manifest["source_file_counts"].values()),
            "promoted_files": len(promoted), "archive_roots_present": True}


def promote(root: Path, run_id: str, area: str) -> dict:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run ID must be one safe path component")
    if area not in AREAS:
        raise ValueError(f"Unrecognised promotion area: {area}")
    run = root / "outputs" / run_id
    source = run / area
    if not (run / "run.json").is_file() or not source.is_dir():
        raise FileNotFoundError(f"A recorded run and produced area are required: {source}")
    recorded = json.loads((run / "run.json").read_text())
    if not isinstance(recorded.get("stages"), dict):
        raise ValueError("Promotion requires a staged run manifest")
    is_public = (root / CANONICAL / "replay_manifest.json").is_file()
    manifest_path = root / (PRIVATE_MANIFEST if is_public else MANIFEST)
    manifest = json.loads(manifest_path.read_text())
    target_base = root / (PRIVATE_CANONICAL if is_public else CANONICAL)
    target = target_base / area
    entries = []
    for item in files(source):
        relative = item.relative_to(source)
        destination = target / relative
        if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents if parent != root):
            raise ValueError(f"Promotion destination traverses a symlink: {destination}")
        entries.append((item, destination, digest(item)))
    if not entries:
        raise ValueError(f"No files to promote: {source}")
    for item, destination, expected_hash in entries:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.promoting-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            shutil.copyfile(item, temporary)
            if digest(temporary) != expected_hash:
                raise ValueError(f"Promotion copy changed: {item}")
            temporary.chmod(stat.S_IMODE(item.stat().st_mode) & ~0o222)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    event = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "area": area,
        "run_manifest_sha256": digest(run / "run.json"),
        "files": [{"path": str(destination.relative_to(target_base)), "sha256": hash_}
                  for _, destination, hash_ in entries],
    }
    manifest["promotion_history"].append(event)
    save_manifest(root, manifest, PRIVATE_MANIFEST if is_public else MANIFEST)
    return {"promoted_files": len(entries), "area": area, "run_id": run_id,
            "destination": str(target_base),
            "public_replay_requires_rebuild": is_public}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("verify")
    sub.add_parser("verify-private")
    promotion = sub.add_parser("promote")
    promotion.add_argument("--run-id", required=True)
    promotion.add_argument("--area", required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    try:
        if args.command == "build":
            result = build(root)
        elif args.command == "verify":
            result = verify(root)
        elif args.command == "verify-private":
            result = verify(root, private=True)
        else:
            result = promote(root, args.run_id, args.area)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"canonical artifacts: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
