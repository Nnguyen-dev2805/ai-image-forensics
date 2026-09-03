"""Shared run-artifact primitives: run directories, environment, and status."""

from __future__ import annotations

import json
import logging
import platform
import re
import sys
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

__all__ = ["RunStatus", "create_run_dir", "write_environment", "write_status"]

logger = logging.getLogger(__name__)

_MAX_DIR_ATTEMPTS = 5

_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "pydantic",
    "PyYAML",
    "scikit-learn",
    "Pillow",
    "pytest",
    "ruff",
)

_UNSAFE_CHARS = re.compile(r"[^a-z0-9._-]+")
_REPEATED_DASH = re.compile(r"-{2,}")


class RunStatus(BaseModel):
    """Metadata record for a finished, failed, or deferred baseline run."""

    baseline: str
    status: Literal["completed", "failed", "deferred"]
    reason: str | None = None
    command: list[str]
    started_at: str
    ended_at: str

    @field_validator("started_at", "ended_at")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        """Enforce the documented UTC ISO-8601 timestamp contract."""
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"must be a UTC ISO-8601 timestamp, got: {value!r}") from exc
        offset = parsed.utcoffset()
        if offset is None:
            raise ValueError(f"must include a UTC offset or Z suffix, got: {value!r}")
        if offset != timedelta(0):
            raise ValueError(f"must be a UTC timestamp, got: {value!r}")
        return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(value: str, field_name: str) -> str:
    lowered = value.strip().lower()
    replaced = _UNSAFE_CHARS.sub("-", lowered)
    collapsed = _REPEATED_DASH.sub("-", replaced)
    stripped = collapsed.strip("-_.")
    if not stripped:
        raise ValueError(
            f"{field_name} must contain at least one path-safe character "
            f"after normalization, got: {value!r}"
        )
    return stripped


def create_run_dir(
    output_root: Path,
    baseline: str,
    run_name: str | None = None,
) -> Path:
    """Create a new exclusive run directory under ``output_root``.

    The directory is named ``<timestamp>_<baseline>[_<run_name>]`` with a UTC
    timestamp that includes microseconds. Existing directories are never
    reused or overwritten; on collision a fresh timestamp is tried a bounded
    number of times before ``FileExistsError`` is raised.
    """
    if run_name is not None and not run_name.strip():
        raise ValueError(
            f"run_name must not be empty or whitespace-only when provided, got: {run_name!r}"
        )

    baseline_slug = _slugify(baseline, "baseline")
    run_name_slug = _slugify(run_name, "run_name") if run_name is not None else None

    last_error: FileExistsError | None = None

    for attempt in range(1, _MAX_DIR_ATTEMPTS + 1):
        stamp = _utc_now().strftime("%Y%m%dT%H%M%S%f") + "Z"
        parts = [stamp, baseline_slug]
        if run_name_slug is not None:
            parts.append(run_name_slug)
        candidate = output_root / "_".join(parts)
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            last_error = exc
            logger.warning(
                "Run directory %s already exists, retrying (%d/%d)",
                candidate,
                attempt,
                _MAX_DIR_ATTEMPTS,
            )
            continue
        logger.info("Created run directory %s", candidate)
        return candidate

    logger.error(
        "Could not create a unique run directory under %s after %d attempts",
        output_root,
        _MAX_DIR_ATTEMPTS,
    )
    raise FileExistsError(
        f"Could not create a unique run directory under {output_root} after "
        f"{_MAX_DIR_ATTEMPTS} attempts: {last_error}"
    )


def _safe_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_environment(path: Path) -> None:
    """Write portable runtime-environment metadata as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    packages: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        version = _safe_version(name)
        if version is not None:
            packages[name] = version

    payload = {
        "captured_at": _utc_now().isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "command": list(sys.argv),
        "packages": packages,
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2) + "\n")

    logger.debug("Wrote environment metadata to %s", path)


def write_status(path: Path, status: RunStatus) -> None:
    """Serialize a ``RunStatus`` to UTF-8 JSON with indent=2 and a newline."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(status.model_dump(), indent=2) + "\n")

    logger.debug("Wrote run status to %s", path)
