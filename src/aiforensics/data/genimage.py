"""Build Phase A/B manifests from a GenImage-layout dataset directory.

GenImage-style releases (including Tiny-GenImage) lay images out as::

    <data_root>/<generator>/<split>/ai/*.<ext>
    <data_root>/<generator>/<split>/nature/*.<ext>

``ai`` means the image was generated, ``nature`` means it is a real photograph,
and the on-disk ``train``/``val`` directories are the dataset's own splits. This
module turns that tree into the CSV manifests described in
``docs/schemas/manifest.md`` so every baseline keeps reading manifests rather
than walking dataset-specific directory layouts.

Two properties matter more than convenience here:

Leakage control. A generator listed as held-out must never contribute a training
row, and one image must never appear in two splits. Assignment is therefore
resolved by explicit precedence (train > dev > external) and deduplicated by
content checksum across the whole build, not per manifest.

Determinism. Files are discovered in sorted order and subsampling is a stable
stride over that order, so the same directory tree and config always produce
byte-identical manifests without depending on a random seed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import ManifestError, ManifestRecord, compute_sha256

__all__ = [
    "IMAGE_EXTENSIONS",
    "BuiltManifest",
    "ManifestBuildResult",
    "build_genimage_manifests",
    "discover_generator_dirs",
]

logger = logging.getLogger(__name__)

# Pillow-readable still-image extensions seen in GenImage-style releases.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
)

_LABEL_DIRS: dict[str, Literal["real", "fake"]] = {"ai": "fake", "nature": "real"}

# Dataset-native split directory -> manifest split value. GenImage-style
# releases ship train/ and val/; anything else is reported rather than ignored.
_SPLIT_DIRS: dict[str, str] = {"train": "train", "val": "dev"}

_SplitName = Literal["train", "dev", "external"]


@dataclass(frozen=True)
class BuiltManifest:
    """One manifest that was written, plus the counts needed to sanity-check it."""

    label: str
    path: Path
    split: str
    record_count: int
    real_count: int
    fake_count: int
    generators: tuple[str, ...]
    # File-extension counts per label. A real/fake split that also splits by
    # container format lets a detector learn the format instead of generation
    # artifacts, so the build surfaces it instead of hiding it.
    extensions_by_label: dict[str, dict[str, int]]

    def format_skew(self) -> str | None:
        """Describe a real/fake format imbalance, or ``None`` when balanced.

        Returns a message when the dominant extension differs between labels, or
        when one label is single-format and the other is not; both patterns make
        container format a usable shortcut.
        """
        real = self.extensions_by_label.get("real", {})
        fake = self.extensions_by_label.get("fake", {})
        if not real or not fake:
            return None
        real_top = max(real, key=lambda ext: (real[ext], ext))
        fake_top = max(fake, key=lambda ext: (fake[ext], ext))
        if real_top != fake_top or set(real) != set(fake):
            return (
                f"{self.label} split={self.split}: real={_fmt_counts(real)} "
                f"vs fake={_fmt_counts(fake)}; container format may act as a "
                f"shortcut feature"
            )
        return None


def _fmt_counts(counts: dict[str, int]) -> str:
    return "{" + ", ".join(f"{ext}:{counts[ext]}" for ext in sorted(counts)) + "}"


@dataclass(frozen=True)
class ManifestBuildResult:
    """Everything a caller needs to report on a build."""

    manifests: tuple[BuiltManifest, ...]
    duplicate_checksums_skipped: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _Candidate:
    """One image discovered on disk, before split assignment and dedupe."""

    generator: str
    split_dir: str
    label: Literal["real", "fake"]
    path: Path


def discover_generator_dirs(data_root: Path) -> list[str]:
    """List generator directory names under ``data_root``, sorted.

    A generator directory is any immediate subdirectory that contains at least
    one recognized split directory, which keeps unrelated folders (archives,
    notes, checkpoints) out of the result.
    """
    if not data_root.is_dir():
        return []
    found: list[str] = []
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        if any((entry / split_dir).is_dir() for split_dir in _SPLIT_DIRS):
            found.append(entry.name)
    return found


def _iter_images(directory: Path) -> list[Path]:
    """Return recognized image files directly under ``directory``, sorted."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _collect_candidates(
    data_root: Path,
    generators: list[str],
    warnings: list[str],
    *,
    split_dirs: tuple[str, ...] | None = None,
) -> list[_Candidate]:
    """Walk ``<generator>/<split>/{ai,nature}`` for the requested generators."""
    candidates: list[_Candidate] = []
    active_splits = split_dirs if split_dirs is not None else tuple(_SPLIT_DIRS.keys())

    for generator in generators:
        generator_dir = data_root / generator
        if not generator_dir.is_dir():
            raise ManifestError(
                f"Configured generator directory does not exist: {generator_dir}. "
                f"Available generators under {data_root}: "
                f"{discover_generator_dirs(data_root) or 'none'}"
            )

        found_any = False
        for split_dir in active_splits:
            split_path = generator_dir / split_dir
            if not split_path.is_dir():
                continue
            for label_dir, label in _LABEL_DIRS.items():
                images = _iter_images(split_path / label_dir)
                if not images:
                    continue
                found_any = True
                for path in images:
                    candidates.append(
                        _Candidate(
                            generator=generator,
                            split_dir=split_dir,
                            label=label,
                            path=path,
                        )
                    )

        if not found_any:
            warnings.append(
                f"Generator {generator!r} has no readable images under "
                f"{sorted(_LABEL_DIRS)} in {sorted(active_splits)}"
            )

    return candidates


def _subsample(
    candidates: list[_Candidate], max_images: int, balance_labels: bool
) -> list[_Candidate]:
    """Cap candidates per generator, deterministically and label-balanced.

    ``max_images`` is a **per-generator** cap, not a total. Cross-generator
    evaluation compares per-source metrics, so every held-out generator needs a
    comparable sample size; a pooled cap would let one large generator dominate
    and leave another with too few rows to interpret.

    Selection is a fixed stride over the sorted discovery order rather than a
    random sample, so the same tree and config always yield byte-identical
    manifests without depending on a seed. With ``balance_labels`` the cap is
    split evenly between real and fake, which rounds an odd cap down to an even
    total rather than silently favouring one label.
    """
    if max_images <= 0:
        return candidates

    by_generator: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_generator.setdefault(candidate.generator, []).append(candidate)

    keep: set[int] = set()
    positions = {id(c): i for i, c in enumerate(candidates)}

    for group in by_generator.values():
        if len(group) <= max_images:
            keep.update(positions[id(c)] for c in group)
            continue
        if balance_labels:
            per_label = max_images // 2
            chosen = _stride_select(
                [c for c in group if c.label == "real"], per_label
            ) + _stride_select([c for c in group if c.label == "fake"], per_label)
        else:
            chosen = _stride_select(group, max_images)
        keep.update(positions[id(c)] for c in chosen)

    # Emit in discovery order so manifest rows stay grouped by directory.
    return [c for i, c in enumerate(candidates) if i in keep]


def _stride_select(items: list[_Candidate], keep: int) -> list[_Candidate]:
    """Keep ``keep`` items spread evenly across ``items``, preserving order.

    A stride rather than a head slice: dataset directories are often ordered by
    class or capture batch, so taking the first N would sample one corner of the
    generator instead of the whole thing.
    """
    if keep <= 0:
        return []
    if len(items) <= keep:
        return list(items)
    step = len(items) / keep
    return [items[int(i * step)] for i in range(keep)]


def _sample_id(candidate: _Candidate) -> str:
    """Build a stable, globally unique sample id from on-disk coordinates.

    The id encodes the **source** directory split (``train``/``val``), not the
    manifest split. A held-out generator contributes both of its on-disk splits
    to one ``external`` manifest, and GenImage reuses the same filenames in
    ``train`` and ``val``; keying on the manifest split would make those two
    files collide and silently drop half the evaluation set.
    """
    return (
        f"genimage/{candidate.split_dir}/{candidate.generator}/"
        f"{candidate.label}/{candidate.path.stem}"
    )


def _to_records(
    candidates: list[_Candidate],
    split: str,
    data_root: Path,
    seen_checksums: dict[str, str],
    duplicate_counter: list[int],
    warnings: list[str],
) -> list[ManifestRecord]:
    """Checksum candidates and drop content already claimed by another split."""
    from aiforensics.progress import progress_iter

    records: list[ManifestRecord] = []
    seen_ids: set[str] = set()

    for candidate in progress_iter(f"checksum {split}", candidates, log_every=500):
        try:
            checksum = compute_sha256(candidate.path)
        except OSError as exc:
            warnings.append(f"Skipped unreadable image {candidate.path}: {exc}")
            continue

        if checksum in seen_checksums:
            duplicate_counter[0] += 1
            logger.debug(
                "Skipping duplicate image content %s (already used by %s)",
                candidate.path,
                seen_checksums[checksum],
            )
            continue

        sample_id = _sample_id(candidate)
        if sample_id in seen_ids:
            warnings.append(f"Skipped colliding sample_id {sample_id} for {candidate.path}")
            continue

        try:
            relative = candidate.path.relative_to(data_root)
        except ValueError:
            relative = candidate.path

        seen_checksums[checksum] = sample_id
        seen_ids.add(sample_id)
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                path=relative,
                label=candidate.label,
                source=candidate.generator,
                split=split,  # type: ignore[arg-type]
                checksum=checksum,
                dataset="genimage",
                generator=candidate.generator,
            )
        )

    return records


def _validate_roles(config: AppConfig) -> None:
    """Reject a config whose generator lists would leak training data."""
    tiny = config.datasets.tiny_genimage
    unseen = config.datasets.genimage_unseen
    if not (tiny.enabled and unseen.enabled):
        return
    overlap = sorted(set(tiny.generators) & set(unseen.generators))
    if overlap:
        raise ManifestError(
            f"Generators appear in both tiny_genimage and genimage_unseen: {overlap}. "
            f"A held-out generator must not also be in-distribution."
        )


def build_genimage_manifests(config: AppConfig) -> ManifestBuildResult:
    """Build every enabled manifest from the GenImage-layout ``data_root``.

    ``tiny_genimage`` contributes train and dev manifests from its own
    ``train``/``val`` directories; ``genimage_unseen`` contributes one external
    manifest from all of its splits, because held-out generators are evaluated
    as a whole rather than split further.

    Raises:
        ManifestError: ``data_root`` is missing, a configured generator
            directory is absent, generator roles overlap, or an enabled dataset
            produced no records.
    """
    data_root = config.paths.data_root
    if not data_root.is_dir():
        raise ManifestError(
            f"data_root does not exist or is not a directory: {data_root}. "
            f"Point paths.data_root at the GenImage-layout dataset root."
        )

    _validate_roles(config)

    warnings: list[str] = []
    duplicate_counter = [0]
    # Checksum -> owning sample_id, shared across manifests so one image can
    # only ever belong to a single split.
    seen_checksums: dict[str, str] = {}
    built: list[BuiltManifest] = []

    tiny = config.datasets.tiny_genimage
    unseen = config.datasets.genimage_unseen

    if tiny.enabled:
        if not tiny.generators:
            raise ManifestError(
                "datasets.tiny_genimage.enabled is true but no generators are configured; "
                f"available generators under {data_root}: "
                f"{discover_generator_dirs(data_root) or 'none'}"
            )
        tiny_candidates = _collect_candidates(data_root, list(tiny.generators), warnings)

        # Split roles come from _SPLIT_DIRS so the directory-to-split mapping
        # lives in exactly one place.
        for source_dir, role in _SPLIT_DIRS.items():
            subset = [c for c in tiny_candidates if c.split_dir == source_dir]
            subset = _subsample(subset, tiny.max_images, tiny.balance_labels)
            records = _to_records(
                subset, role, data_root, seen_checksums, duplicate_counter, warnings
            )
            if not records:
                raise ManifestError(
                    f"tiny_genimage produced no {role} records from generators "
                    f"{list(tiny.generators)} under {data_root} "
                    f"(expected images in {source_dir}/{{{','.join(sorted(_LABEL_DIRS))}}})"
                )
            target = tiny.train_manifest if role == "train" else tiny.dev_manifest
            built.append(_write(records, target, role, label="tiny_genimage"))

    if unseen.enabled:
        if not unseen.generators:
            raise ManifestError(
                "datasets.genimage_unseen.enabled is true but no generators are configured; "
                f"available generators under {data_root}: "
                f"{discover_generator_dirs(data_root) or 'none'}"
            )
        allowed_splits = (unseen.source_split,) if unseen.source_split else None
        unseen_candidates = _collect_candidates(
            data_root, list(unseen.generators), warnings, split_dirs=allowed_splits
        )
        unseen_candidates = _subsample(unseen_candidates, unseen.max_images, unseen.balance_labels)
        records = _to_records(
            unseen_candidates, unseen.split, data_root, seen_checksums, duplicate_counter, warnings
        )
        if not records:
            raise ManifestError(
                f"genimage_unseen produced no records from generators "
                f"{list(unseen.generators)} under {data_root}"
            )
        built.append(_write(records, unseen.manifest, unseen.split, label="genimage_unseen"))

    for message in warnings:
        logger.warning("%s", message)

    return ManifestBuildResult(
        manifests=tuple(built),
        duplicate_checksums_skipped=duplicate_counter[0],
        warnings=tuple(warnings),
    )


def _write(
    records: list[ManifestRecord],
    target: Path,
    split: str,
    *,
    label: str,
) -> BuiltManifest:
    from aiforensics.data.manifest import write_manifest

    write_manifest(records, target)

    extensions_by_label: dict[str, dict[str, int]] = {}
    for record in records:
        per_label = extensions_by_label.setdefault(record.label, {})
        extension = record.path.suffix.lower()
        per_label[extension] = per_label.get(extension, 0) + 1

    return BuiltManifest(
        label=label,
        path=target,
        split=split,
        record_count=len(records),
        real_count=sum(1 for r in records if r.label == "real"),
        fake_count=sum(1 for r in records if r.label == "fake"),
        generators=tuple(sorted({r.source for r in records})),
        extensions_by_label=extensions_by_label,
    )
