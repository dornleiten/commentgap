"""Verify compact public inputs before notebooks read them directly."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _root() -> Path:
    explicit = os.environ.get("COMMENTGAP_PUBLIC_ARTIFACT_ROOT")
    return Path(explicit).resolve() if explicit else Path(__file__).resolve().parents[1] / "artifacts/canonical"


def _hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_public_stage(stage: str, *, root: Path | None = None) -> None:
    """Check published files against their manifest without loading their data."""
    root = (root or _root()).resolve()
    manifest_path = root / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported replay manifest: {manifest_path}")
    try:
        paths = manifest["stages"][stage]
    except KeyError as exc:
        raise KeyError(f"No public replay products for notebook {stage}") from exc
    for relative in paths:
        candidate = root / relative
        if candidate.is_symlink():
            raise FileNotFoundError(f"Public replay product is a symlink: {candidate}")
        path = candidate.resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(f"Public replay product missing: {path}")
        expected = manifest["products"][relative]
        if path.stat().st_size != expected["bytes"] or _hash(path) != expected["sha256"]:
            raise ValueError(f"Public replay product differs from manifest: {path}")
        if path.suffix not in {".csv", ".parquet", ".json"}:
            raise ValueError(f"Unsupported replay product: {path}")
