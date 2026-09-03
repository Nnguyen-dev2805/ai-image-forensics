"""Deterministic cache-key construction for expensive reusable outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

__all__ = ["cache_key"]


def cache_key(parts: Mapping[str, str]) -> str:
    """Return a deterministic SHA-256 hex digest for a string mapping.

    Keys are sorted and the mapping is serialized as canonical JSON before
    hashing, so insertion order never affects the result and values containing
    separators cannot create ambiguous encodings.
    """
    if not parts:
        raise ValueError("cache_key requires a non-empty mapping of parts")

    for key, value in parts.items():
        if not isinstance(key, str):
            raise TypeError(f"cache_key keys must be strings, got {type(key).__name__}: {key!r}")
        if not isinstance(value, str):
            raise TypeError(
                f"cache_key values must be strings, got {type(value).__name__} for key {key!r}"
            )

    canonical = json.dumps(
        dict(sorted(parts.items())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
