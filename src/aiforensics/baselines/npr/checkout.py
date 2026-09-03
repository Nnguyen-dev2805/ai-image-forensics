"""External NPR checkout lifecycle management.

The checkout lives under ``paths.external_root / "NPR-DeepfakeDetection"`` and is
runtime state ignored by Git. All destructive Git commands are forbidden: an
existing dirty or foreign checkout fails instead of being repaired.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from aiforensics.baselines.npr.errors import NPRConfigError, NPRDeferredError

__all__ = ["OFFICIAL_NPR_REPO_URL", "CheckoutInfo", "ensure_npr_checkout"]

logger = logging.getLogger(__name__)

OFFICIAL_NPR_REPO_URL = "https://github.com/chuangchuangtan/NPR-DeepfakeDetection"
REPO_DIR_NAME = "NPR-DeepfakeDetection"


@dataclass(frozen=True)
class CheckoutInfo:
    """Verified state of the external NPR checkout."""

    repo_dir: Path
    resolved_commit: str
    action: str  # one of "reused", "cloned", "fetched", "checked_out"


def normalize_repo_url(repo_url: str) -> str:
    """Normalize a repository URL and require it to be the official NPR repo."""
    cleaned = repo_url.strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[: -len(".git")]
    if cleaned != OFFICIAL_NPR_REPO_URL:
        raise NPRConfigError(
            f"Unsupported NPR repo_url: {repo_url!r}. Only the official repository "
            f"is accepted: {OFFICIAL_NPR_REPO_URL}"
        )
    return cleaned


def resolve_repo_dir(external_root: Path) -> Path:
    """Return the deterministic checkout location under ``external_root``."""
    return Path(external_root) / REPO_DIR_NAME


def _run_git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise NPRConfigError(
            f"git {' '.join(args[:1])} failed in {repo_dir}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git_version() -> bool:
    """Return True when a git client is invocable (no network use)."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _verify_existing_checkout(
    repo_dir: Path,
    repo_commit: str,
    *,
    allow_deferred: bool,
) -> tuple[str, str]:
    """Verify an existing checkout and return (resolved_commit, action)."""
    if not (repo_dir / ".git").exists():
        raise NPRConfigError(f"Existing directory is not a Git checkout: {repo_dir}")

    remote = _run_git(repo_dir, "remote", "get-url", "origin")
    if normalize_repo_url(remote) != OFFICIAL_NPR_REPO_URL:
        raise NPRConfigError(
            f"Existing checkout points at another repository: {remote}. "
            f"Expected the official NPR repository."
        )

    status = _run_git(repo_dir, "status", "--porcelain")
    if status:
        raise NPRConfigError(
            f"Existing NPR checkout has uncommitted changes; refusing to use or "
            f"overwrite it: {repo_dir}"
        )

    head = _run_git(repo_dir, "rev-parse", "HEAD")
    if head != repo_commit:
        # A clean checkout at another commit may fetch and check out the pin.
        fetched = _fetch_commit(repo_dir, repo_commit, allow_deferred=allow_deferred)
        if not fetched:
            raise NPRConfigError(
                f"Configured commit {repo_commit} could not be resolved in {repo_dir} "
                f"after a successful fetch."
            )
        _run_git(repo_dir, "checkout", "--detach", repo_commit)
        return repo_commit, "checked_out"

    return head, "reused"


def _fetch_commit(repo_dir: Path, repo_commit: str, *, allow_deferred: bool) -> bool:
    """Fetch and verify the configured commit.

    Repository-access failures raise a defer signal only when ``allow_deferred``
    is set; otherwise they surface as configuration/runtime errors so the run
    records ``failed`` instead of ``deferred``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "fetch", "origin", repo_commit],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        if allow_deferred:
            raise NPRDeferredError(f"Network access unavailable for git fetch: {exc}") from exc
        raise NPRConfigError(f"git fetch failed to start: {exc}") from exc
    if result.returncode != 0:
        message = f"Could not fetch configured commit {repo_commit}: {result.stderr.strip()}"
        if allow_deferred:
            raise NPRDeferredError(message)
        raise NPRConfigError(message)
    resolved = _run_git(repo_dir, "rev-parse", "--verify", "--quiet", f"{repo_commit}^{{commit}}")
    if resolved != repo_commit:
        raise NPRConfigError(
            f"Fetched object does not resolve to the configured commit: "
            f"expected {repo_commit}, got {resolved!r}"
        )
    return True


def ensure_npr_checkout(
    *,
    repo_dir: Path,
    repo_url: str,
    repo_commit: str,
    allow_deferred: bool,
) -> CheckoutInfo:
    """Ensure a verified official NPR checkout at ``repo_commit``.

    Raises:
        NPRConfigError: invalid URL, foreign/dirty/corrupt checkout, a commit
            that cannot be resolved after repository access succeeds, or a
            repository-access failure while ``allow_deferred`` is False.
        NPRDeferredError: network unavailability while cloning/fetching, only
            when ``allow_deferred`` is True (defer signal surfaced by adapter).
    """
    normalized_url = normalize_repo_url(repo_url)

    if repo_dir.exists():
        resolved_commit, action = _verify_existing_checkout(
            repo_dir, repo_commit, allow_deferred=allow_deferred
        )
        logger.info(
            "NPR checkout %s: action=%s resolved_commit=%s", repo_dir, action, resolved_commit
        )
        return CheckoutInfo(repo_dir=repo_dir, resolved_commit=resolved_commit, action=action)

    if not _git_version():
        if allow_deferred:
            raise NPRDeferredError("git client unavailable for NPR checkout")
        raise NPRConfigError("git client unavailable for NPR checkout")

    # git clone fails when the destination's parent directory does not exist
    # (e.g. external_root never created); create it instead of misreporting a
    # network failure.
    try:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NPRConfigError(f"Could not create NPR checkout parent directory: {exc}") from exc

    try:
        result = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                normalized_url,
                str(repo_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        if allow_deferred:
            raise NPRDeferredError(f"Network access unavailable for git clone: {exc}") from exc
        raise NPRConfigError(f"git clone failed to start: {exc}") from exc

    if result.returncode != 0:
        if allow_deferred:
            raise NPRDeferredError(
                f"Could not clone NPR repository (network unavailable?): {result.stderr.strip()}"
            )
        raise NPRConfigError(f"git clone failed: {result.stderr.strip()}")

    _run_git(repo_dir, "checkout", "--detach", repo_commit)
    resolved_commit = _run_git(repo_dir, "rev-parse", "HEAD")
    if resolved_commit != repo_commit:
        raise NPRConfigError(
            f"Cloned checkout HEAD {resolved_commit} does not match configured commit {repo_commit}"
        )
    logger.info("NPR checkout cloned: repo_dir=%s commit=%s", repo_dir, resolved_commit)
    return CheckoutInfo(repo_dir=repo_dir, resolved_commit=resolved_commit, action="cloned")
