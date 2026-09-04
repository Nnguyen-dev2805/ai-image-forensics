"""Single source of truth for which evaluation manifests a run may use.

Every baseline evaluates the same slice, so the decision of which dataset
manifests are in scope belongs in one place. A dataset contributes records only
when its config slice is ``enabled`` and its manifest file exists: a disabled
slice is ignored even when its manifest is still on disk, so a config that
disables ``tiny_genimage`` for an external-only experiment cannot be silently
re-mixed with tiny samples.

Selection is pure with respect to the filesystem beyond reading manifests: it
loads, orders, and reports, but never writes, downloads, or mutates config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import ManifestError, ManifestRecord, load_manifest

__all__ = [
    "EvaluationSelection",
    "SelectedManifest",
    "selected_evaluation_manifests",
    "training_manifest_path",
]

logger = logging.getLogger(__name__)

# Fixed dataset order; every baseline must see identical record ordering.
_TINY_LABEL = "tiny_genimage"
_EXTERNAL_LABELS = ("genimage_unseen", "synthbuster")


@dataclass(frozen=True)
class SelectedManifest:
    """One dataset manifest that was enabled, present, and loaded."""

    label: str
    path: Path
    record_count: int


@dataclass(frozen=True)
class EvaluationSelection:
    """Evaluation records plus the provenance of how they were selected."""

    records: tuple[ManifestRecord, ...]
    manifests: tuple[SelectedManifest, ...]
    warnings: tuple[str, ...]

    @property
    def sample_ids(self) -> set[str]:
        return {record.sample_id for record in self.records}


def _candidate_manifests(config: AppConfig) -> list[tuple[str, Path]]:
    """Return enabled (label, manifest) pairs in fixed dataset order."""
    datasets = config.datasets
    candidates: list[tuple[str, Path]] = []

    if datasets.tiny_genimage.enabled:
        candidates.append((_TINY_LABEL, datasets.tiny_genimage.dev_manifest))
    if datasets.genimage_unseen.enabled:
        candidates.append((_EXTERNAL_LABELS[0], datasets.genimage_unseen.manifest))
    if datasets.synthbuster.enabled:
        candidates.append((_EXTERNAL_LABELS[1], datasets.synthbuster.manifest))

    return candidates


def selected_evaluation_manifests(
    config: AppConfig,
    *,
    strict: bool = True,
) -> EvaluationSelection:
    """Load evaluation records from every enabled dataset manifest that exists.

    An enabled manifest that is missing produces a warning and is skipped; an
    enabled manifest that exists but is invalid raises ``ManifestError`` rather
    than being silently dropped. With ``strict`` (the default), selecting no
    records at all raises ``ManifestError``; callers that need to report their
    own failure message, or that compute a scope for a run that is about to
    fail, pass ``strict=False`` and inspect ``records``.
    """
    records: list[ManifestRecord] = []
    manifests: list[SelectedManifest] = []
    warnings: list[str] = []

    for label, path in _candidate_manifests(config):
        if not path.exists():
            message = f"{label} manifest missing, continuing without it: {path}"
            warnings.append(message)
            logger.warning("%s manifest missing, continuing without it: %s", label, path)
            continue
        loaded = load_manifest(path, data_root=config.paths.data_root)
        records.extend(loaded)
        manifests.append(SelectedManifest(label=label, path=path, record_count=len(loaded)))

    if strict and not records:
        raise ManifestError("No valid evaluation manifests found.")

    return EvaluationSelection(
        records=tuple(records),
        manifests=tuple(manifests),
        warnings=tuple(warnings),
    )


def training_manifest_path(config: AppConfig) -> Path | None:
    """Return the enabled training manifest path, or ``None`` when disabled.

    Only ``tiny_genimage`` carries a training split in Phase A/B; when that
    slice is disabled there is no training data, and trainable baselines must
    say so instead of training on a manifest the config excluded.
    """
    tiny = config.datasets.tiny_genimage
    return tiny.train_manifest if tiny.enabled else None
