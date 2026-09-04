"""Run scope: the experiment identity that binds artifacts to one config.

A run directory only belongs to the current experiment when the evaluation
setup it was produced under still matches the current config. ``RunScope``
captures that setup (project phase, data root, which datasets are enabled,
which manifests they resolve to, and the exact evaluation sample ids) and
reduces it to one ``scope_id`` digest.

Consumers use the digest to keep unrelated history out of the current
experiment: ``assisted_qwen`` refuses foreign CLIP predictions, ``evaluate``
skips foreign runs, and ``report`` never selects them. The scope is computed
from config plus manifests only -- never from model weights, seeds, or
prediction contents -- so re-running the same experiment reproduces the same
digest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from aiforensics.cache.keys import cache_key
from aiforensics.config.models import AppConfig
from aiforensics.data.selection import (
    EvaluationSelection,
    selected_evaluation_manifests,
)

__all__ = [
    "SCOPE_FILENAME",
    "SCOPE_VERSION",
    "RunScope",
    "compute_run_scope",
    "read_run_scope",
    "scope_matches",
    "write_run_scope",
]

logger = logging.getLogger(__name__)

SCOPE_FILENAME = "run_scope.json"

# Bump when the scope payload changes shape so old artifacts stop matching
# instead of silently comparing against a different definition.
SCOPE_VERSION = "1"


class RunScope(BaseModel):
    """Fingerprint of the evaluation setup a run was produced under."""

    scope_version: str = SCOPE_VERSION
    scope_id: str = Field(min_length=1)
    phase: str
    data_root: str
    datasets: dict[str, str]
    sample_id_count: int = Field(ge=0)
    sample_ids_digest: str = Field(min_length=1)


def _dataset_scope_parts(config: AppConfig) -> dict[str, str]:
    """Describe each dataset slice: disabled, or enabled with its manifest.

    A disabled dataset contributes only ``disabled`` so its stale manifest path
    cannot change the digest, while an enabled one pins the manifest it selects.
    """
    datasets = config.datasets
    tiny = datasets.tiny_genimage
    unseen = datasets.genimage_unseen
    synth = datasets.synthbuster
    return {
        "tiny_genimage": (f"enabled:{tiny.dev_manifest}" if tiny.enabled else "disabled"),
        "genimage_unseen": (f"enabled:{unseen.manifest}" if unseen.enabled else "disabled"),
        "synthbuster": (f"enabled:{synth.manifest}" if synth.enabled else "disabled"),
    }


def compute_run_scope(
    config: AppConfig,
    *,
    selection: EvaluationSelection | None = None,
) -> RunScope:
    """Compute the current config's run scope.

    ``selection`` may be passed by callers that already resolved evaluation
    records, so manifests are not parsed twice. When omitted, evaluation
    manifests are selected here; a config whose enabled manifests are all
    missing yields an empty sample-id set rather than an error, because scope
    computation must stay usable while a run is still failing or deferring.
    """
    if selection is None:
        selection = selected_evaluation_manifests(config, strict=False)

    sample_ids = sorted(selection.sample_ids)
    sample_ids_digest = cache_key({"sample_ids": json.dumps(sample_ids, separators=(",", ":"))})
    datasets = _dataset_scope_parts(config)

    scope_id = cache_key(
        {
            "scope_version": SCOPE_VERSION,
            "phase": config.project.phase,
            "data_root": str(config.paths.data_root),
            "datasets": json.dumps(datasets, sort_keys=True, separators=(",", ":")),
            "sample_ids_digest": sample_ids_digest,
        }
    )

    return RunScope(
        scope_version=SCOPE_VERSION,
        scope_id=scope_id,
        phase=config.project.phase,
        data_root=str(config.paths.data_root),
        datasets=datasets,
        sample_id_count=len(sample_ids),
        sample_ids_digest=sample_ids_digest,
    )


def write_run_scope(path: Path, scope: RunScope) -> None:
    """Serialize a ``RunScope`` as UTF-8 JSON with indent=2 and a newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(scope.model_dump(), indent=2, sort_keys=True) + "\n")
    logger.debug("Wrote run scope to %s", path)


def read_run_scope(path: Path) -> RunScope | None:
    """Read a ``run_scope.json``, returning ``None`` when absent or unreadable.

    Unreadable or invalid scope files are treated as "no scope" rather than
    errors: run directories written before scopes existed, or partially
    written by an interrupted run, must not break discovery.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Could not read run scope %s: %s", path, exc)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed run scope %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Run scope %s must contain a JSON object", path)
        return None

    try:
        return RunScope(**payload)
    except Exception as exc:
        logger.warning("Invalid run scope %s: %s", path, exc)
        return None


def scope_matches(run_dir: Path, expected: RunScope) -> bool:
    """Report whether ``run_dir`` was produced under the expected scope.

    A run directory without a readable ``run_scope.json`` never matches: an
    unlabelled artifact cannot be proven to belong to the current experiment.
    """
    found = read_run_scope(run_dir / SCOPE_FILENAME)
    if found is None:
        return False
    return found.scope_version == expected.scope_version and found.scope_id == expected.scope_id
