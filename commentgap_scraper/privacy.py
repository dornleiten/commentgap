from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PSEUDONYMIZATION_SCHEME = "hmac-sha256-author-identity-v1"
_FINGERPRINT_MESSAGE = b"commentgap-dataset-hash-key-fingerprint-v1"


def _key_bytes(key: str | bytes) -> bytes:
    value = key.encode("utf-8") if isinstance(key, str) else key
    if len(value) < 16:
        raise ValueError("COMMENTGAP_HASH_KEY must contain at least 16 bytes")
    return value


def hash_key_fingerprint(key: str | bytes) -> str:
    """Return a non-secret equality marker without retaining the HMAC key."""
    return hmac.new(_key_bytes(key), _FINGERPRINT_MESSAGE, hashlib.sha256).hexdigest()


def dataset_has_comments(root: Path) -> bool:
    return any((root / "comments").glob("year=*/month=*/*.parquet"))


def hash_key_registration_required(root: Path) -> bool:
    return dataset_has_comments(root) and not (root / "privacy_metadata.json").exists()


def ensure_hash_key_compatible(
    root: Path,
    key: str | bytes,
    *,
    confirm_existing: bool = False,
) -> Path:
    """Reject silent key changes and register a key fingerprint atomically.

    Older datasets have no fingerprint. Because they intentionally contain no
    raw author identifiers, their key cannot be verified retrospectively from
    Parquet alone; adopting one therefore requires an explicit confirmation.
    """
    fingerprint = hash_key_fingerprint(key)
    path = root / "privacy_metadata.json"
    if path.exists():
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read hash-key fingerprint metadata at {path}") from exc
        recorded = metadata.get("hash_key_fingerprint")
        if not isinstance(recorded, str) or not recorded:
            raise ValueError(f"hash-key fingerprint metadata is invalid at {path}")
        if not hmac.compare_digest(recorded, fingerprint):
            raise ValueError(
                "COMMENTGAP_HASH_KEY does not match the key registered for this dataset; "
                "no data was collected"
            )
        if metadata.get("pseudonymization_scheme") != PSEUDONYMIZATION_SCHEME:
            raise ValueError(
                "the dataset uses a different author pseudonymization scheme; "
                "refusing to mix incompatible hashes"
            )
        return path

    existing_comment_files = dataset_has_comments(root)
    if existing_comment_files and not confirm_existing:
        raise ValueError(
            "this existing dataset predates hash-key fingerprints. Re-run once with "
            "--confirm-existing-hash-key after setting COMMENTGAP_HASH_KEY to the exact "
            "original 2025 key"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    registered_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    metadata = {
        "schema_version": 1,
        "pseudonymization_scheme": PSEUDONYMIZATION_SCHEME,
        "hash_key_fingerprint": fingerprint,
        "registered_at": registered_at,
        "adopted_for_preexisting_comments": existing_comment_files,
        "warning": "This fingerprint verifies key continuity; it is not the hash key.",
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


class AuthorPseudonymizer:
    def __init__(self, key: str | bytes):
        self._key = _key_bytes(key)

    def pseudonymize_candidates(self, posting: dict[str, Any]) -> tuple[str, ...]:
        author = posting.get("author") or {}
        legacy = posting.get("legacy") or {}
        candidates = (
            ("author-id", author.get("id")),
            ("legacy-community-id", legacy.get("communityIdentityId")),
            ("legacy-community-name", legacy.get("communityName")),
            ("author-name", author.get("name")),
        )
        values: list[str] = []
        for namespace, value in candidates:
            if value is not None and str(value).strip():
                digest = hmac.new(
                    self._key,
                    f"{namespace}:{value}".encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                values.append(f"author_{digest}")
        return tuple(values)

    def pseudonymize(self, posting: dict[str, Any]) -> str | None:
        candidates = self.pseudonymize_candidates(posting)
        return candidates[0] if candidates else None
