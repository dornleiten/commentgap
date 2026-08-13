from __future__ import annotations

import hashlib
import hmac
from typing import Any


class AuthorPseudonymizer:
    def __init__(self, key: str | bytes):
        key_bytes = key.encode("utf-8") if isinstance(key, str) else key
        if len(key_bytes) < 16:
            raise ValueError("COMMENTGAP_HASH_KEY must contain at least 16 bytes")
        self._key = key_bytes

    def pseudonymize(self, posting: dict[str, Any]) -> str | None:
        author = posting.get("author") or {}
        legacy = posting.get("legacy") or {}
        candidates = (
            ("author-id", author.get("id")),
            ("legacy-community-id", legacy.get("communityIdentityId")),
            ("legacy-community-name", legacy.get("communityName")),
            ("author-name", author.get("name")),
        )
        for namespace, value in candidates:
            if value is not None and str(value).strip():
                digest = hmac.new(
                    self._key,
                    f"{namespace}:{value}".encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                return f"author_{digest}"
        return None
