"""Checkpoint existence and optional SHA-256 integrity validation."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

__all__ = ["validate_checkpoint"]

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


def validate_checkpoint(checkpoint_path: Path, checkpoint_sha256: str | None) -> bool:
    """Validate checkpoint existence/integrity.

    Returns True when a configured checksum was actually verified, False when
    checksum verification was skipped (no real SHA-256 configured).

    Raises:
        FileNotFoundError: checkpoint missing or not a regular file.
        ValueError: checkpoint exists but the configured checksum mismatches
            (hard integrity failure; the adapter must never defer this), or the
            configured checksum is not a valid 64-character hex SHA-256 (a typo
            must fail loudly instead of silently losing integrity protection).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"NPR checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"NPR checkpoint is not a regular file: {checkpoint_path}")

    if checkpoint_sha256 is None:
        logger.warning(
            "Checkpoint checksum verification skipped: no SHA-256 configured for %s",
            checkpoint_path,
        )
        return False

    configured = checkpoint_sha256.strip()
    if not _SHA256_RE.fullmatch(configured):
        raise ValueError(
            f"Invalid checkpoint_sha256 for {checkpoint_path}: "
            f"{checkpoint_sha256!r} is not a 64-character hex SHA-256 digest"
        )

    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest.lower() != configured.lower():
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {checkpoint_path}: "
            f"expected {configured.lower()}, got {digest}"
        )
    logger.info("Checkpoint SHA-256 verified: %s", checkpoint_path)
    return True
