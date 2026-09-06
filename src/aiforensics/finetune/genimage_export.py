"""GenImage subset selection and GCS export planning for qwen_ft fine-tuning.

Selection is deterministic: candidate files are sorted before a seeded
``random.Random`` sample, so the same data root, generator lists, sizes, and
seed always produce the same export plan.
"""

from __future__ import annotations

import csv
import random
import shutil
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from aiforensics.data.manifest import compute_sha256
from aiforensics.finetune.qwen_data import write_qwen_label_jsonl

__all__ = [
    "ALL_GENERATORS",
    "ExportPlan",
    "ExportRecord",
    "PROTOCOLS",
    "ProtocolSpec",
    "SEEN_GENERATORS",
    "UNSEEN_GENERATORS",
    "build_export_plan",
    "find_genimage_data_root",
    "write_export_archive",
    "write_export_manifests",
    "write_relative_export_manifests",
]

# GenImage maps the ``ai`` directory to fake and ``nature`` to real images.
LABEL_DIRS: dict[str, Literal["real", "fake"]] = {"ai": "fake", "nature": "real"}
# GenImage ``val`` becomes the manifest ``external`` split.
SPLITS: dict[str, Literal["train", "external"]] = {"train": "train", "val": "external"}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

MANIFEST_COLUMNS = [
    "sample_id",
    "path",
    "label",
    "source",
    "split",
    "checksum",
    "dataset",
    "generator",
]

# Generator sets from the design spec's Protocol A / Protocol B sections.
SEEN_GENERATORS: tuple[str, ...] = (
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0508_adm",
    "imagenet_glide",
)
UNSEEN_GENERATORS: tuple[str, ...] = (
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0424_wukong",
    "imagenet_midjourney",
)
ALL_GENERATORS: tuple[str, ...] = (
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0508_adm",
    "imagenet_glide",
    "imagenet_midjourney",
)


@dataclass(frozen=True)
class ProtocolSpec:
    """Which generators a protocol trains on and evaluates against."""

    name: str
    train_generators: tuple[str, ...]
    eval_generators: tuple[str, ...]


PROTOCOLS: dict[str, ProtocolSpec] = {
    "protocol-a-small": ProtocolSpec(
        name="protocol-a-small",
        train_generators=ALL_GENERATORS,
        eval_generators=ALL_GENERATORS,
    ),
    "protocol-b-seen-unseen": ProtocolSpec(
        name="protocol-b-seen-unseen",
        train_generators=SEEN_GENERATORS,
        eval_generators=UNSEEN_GENERATORS,
    ),
}


@dataclass(frozen=True)
class ExportRecord:
    """One selected image: its local source, GCS destination, and manifest row."""

    sample_id: str
    source_path: Path
    relative_gcs_path: str
    label: Literal["real", "fake"]
    source: str
    split: Literal["train", "external"]
    checksum: str
    # GenImage directory name (``ai``/``nature``) the image came from; the
    # Modal archive layout keeps it in the relative path.
    label_dir: str = ""
    is_full: bool = False


@dataclass(frozen=True)
class ExportPlan:
    train_records: list[ExportRecord]
    eval_records: list[ExportRecord]
    eval_full_records: list[ExportRecord] = field(default_factory=list)
    protocol: str = ""


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _looks_like_genimage_root(path: Path) -> bool:
    return any(
        (path / generator / "train").is_dir() and (path / generator / "val").is_dir()
        for generator in ALL_GENERATORS
    )


def find_genimage_data_root(search_root: Path) -> Path:
    """Find the GenImage root inside an extracted raw dataset archive."""
    search_root = Path(search_root)
    if _looks_like_genimage_root(search_root):
        return search_root

    for candidate in sorted(path for path in search_root.rglob("*") if path.is_dir()):
        if _looks_like_genimage_root(candidate):
            return candidate

    raise ValueError(f"Could not find GenImage data root under {search_root}")


def _sample_label_dir(
    *,
    data_root: Path,
    generator: str,
    split_dir: str,
    label_dir: str,
    per_label: int,
    seed: int,
    is_full: bool = False,
) -> list[ExportRecord]:
    label = LABEL_DIRS[label_dir]
    manifest_split = SPLITS[split_dir]
    if split_dir == "train":
        gcs_dir_prefix = "train"
    elif is_full:
        gcs_dir_prefix = "eval-full"
    else:
        gcs_dir_prefix = "eval-small"
    candidates = _list_images(data_root / generator / split_dir / label_dir)

    if len(candidates) < per_label:
        raise ValueError(
            f"Not enough images for generator={generator} split={split_dir} "
            f"label_dir={label_dir}: requested {per_label}, available {len(candidates)}"
        )

    chosen = sorted(random.Random(seed).sample(candidates, per_label))
    records: list[ExportRecord] = []
    for image_path in chosen:
        sample_id = f"genimage/{manifest_split}/{generator}/{label_dir}/{image_path.stem}"
        records.append(
            ExportRecord(
                sample_id=sample_id,
                source_path=image_path,
                relative_gcs_path=f"{gcs_dir_prefix}/{generator}/{label_dir}/{image_path.name}",
                label=label,
                source=generator,
                split=manifest_split,
                checksum=compute_sha256(image_path),
                label_dir=label_dir,
                is_full=is_full,
            )
        )
    return records


def build_export_plan(
    *,
    data_root: Path,
    train_generators: Sequence[str] | None = None,
    eval_generators: Sequence[str] | None = None,
    train_per_label: int,
    eval_per_label: int,
    eval_full_per_label: int = 0,
    seed: int,
    protocol: str | None = None,
) -> ExportPlan:
    """Select a label-balanced per-generator subset for training and evaluation.

    Either pass explicit generator lists, or pass ``protocol`` to resolve the
    generator sets from the design spec; giving both is an error because the
    intent would be ambiguous.
    """
    if protocol is not None and (train_generators is not None or eval_generators is not None):
        raise ValueError("Pass either protocol or explicit generator lists, not both")
    if protocol is not None:
        spec = PROTOCOLS.get(protocol)
        if spec is None:
            raise ValueError(f"unknown protocol {protocol!r}; expected one of {sorted(PROTOCOLS)}")
        train_generators = spec.train_generators
        eval_generators = spec.eval_generators
    if train_generators is None or eval_generators is None:
        raise ValueError("train_generators and eval_generators are required without protocol")

    data_root = Path(data_root)
    if not data_root.is_dir():
        raise ValueError(f"data_root does not exist or is not a directory: {data_root}")
    if train_per_label < 0:
        raise ValueError("train_per_label must not be negative")
    if eval_per_label < 0:
        raise ValueError("eval_per_label must not be negative")
    if eval_full_per_label < 0:
        raise ValueError("eval_full_per_label must not be negative")

    train_records: list[ExportRecord] = []
    for generator in train_generators:
        for label_dir in LABEL_DIRS:
            train_records.extend(
                _sample_label_dir(
                    data_root=data_root,
                    generator=generator,
                    split_dir="train",
                    label_dir=label_dir,
                    per_label=train_per_label,
                    seed=seed,
                )
            )

    eval_records: list[ExportRecord] = []
    for generator in eval_generators:
        for label_dir in LABEL_DIRS:
            eval_records.extend(
                _sample_label_dir(
                    data_root=data_root,
                    generator=generator,
                    split_dir="val",
                    label_dir=label_dir,
                    per_label=eval_per_label,
                    seed=seed,
                )
            )

    eval_full_records: list[ExportRecord] = []
    if eval_full_per_label > 0:
        for generator in eval_generators:
            for label_dir in LABEL_DIRS:
                eval_full_records.extend(
                    _sample_label_dir(
                        data_root=data_root,
                        generator=generator,
                        split_dir="val",
                        label_dir=label_dir,
                        per_label=eval_full_per_label,
                        seed=seed,
                        is_full=True,
                    )
                )

    train_records.sort(key=lambda record: record.sample_id)
    eval_records.sort(key=lambda record: record.sample_id)
    eval_full_records.sort(key=lambda record: record.sample_id)
    return ExportPlan(
        train_records=train_records,
        eval_records=eval_records,
        eval_full_records=eval_full_records,
        protocol=protocol or "",
    )


def write_export_manifests(
    plan: ExportPlan,
    manifest_dir: Path,
    *,
    gcs_base_uri: str,
) -> dict[str, Path]:
    """Write train/eval manifests whose ``path`` column holds GCS destination URIs."""
    if not gcs_base_uri.startswith("gs://"):
        raise ValueError(f"gcs_base_uri must be a gs:// URI, got: {gcs_base_uri}")

    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    def resolve_gcs_path(record: ExportRecord) -> str:
        return f"{gcs_base_uri.rstrip('/')}/{record.relative_gcs_path}"

    outputs = {
        "train": _write_manifest_csv(
            plan.train_records, manifest_dir / "train.csv", resolve_gcs_path
        ),
        "eval_small": _write_manifest_csv(
            plan.eval_records, manifest_dir / "eval_small.csv", resolve_gcs_path
        ),
    }
    if plan.eval_full_records:
        outputs["eval_full"] = _write_manifest_csv(
            plan.eval_full_records, manifest_dir / "eval_full.csv", resolve_gcs_path
        )
    return outputs


def _relative_export_path(record: ExportRecord) -> str:
    """Protocol-relative archive path, e.g. ``train/<generator>/ai/<name>``."""
    if record.split == "external":
        split_dir = "eval-full" if record.is_full else "eval-small"
    else:
        split_dir = "train"
    return f"{split_dir}/{record.source}/{record.label_dir}/{record.source_path.name}"


def write_relative_export_manifests(plan: ExportPlan, manifest_dir: Path) -> dict[str, Path]:
    """Write train/eval manifests with paths relative to the protocol directory.

    The Modal Volume unpacks the export zip under ``/vol/data``, so images are
    addressed relative to ``/vol/data/<protocol>`` instead of a GCS prefix.
    """
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train": _write_manifest_csv(
            plan.train_records, manifest_dir / "train.csv", _relative_export_path
        ),
        "eval_small": _write_manifest_csv(
            plan.eval_records, manifest_dir / "eval_small.csv", _relative_export_path
        ),
    }
    if plan.eval_full_records:
        outputs["eval_full"] = _write_manifest_csv(
            plan.eval_full_records, manifest_dir / "eval_full.csv", _relative_export_path
        )
    return outputs


def write_export_archive(plan: ExportPlan, archive_path: Path, staging_dir: Path) -> Path:
    """Stage the selected subset and zip it for the Kaggle -> Modal handoff.

    Zip layout: ``<protocol>/train|eval-small|eval-full/<generator>/<label_dir>/<file>``,
    ``<protocol>/manifests/{train,eval_small,eval_full}.csv``, and
    ``<protocol>/qwen_train.jsonl`` with protocol-relative image paths.
    """
    protocol_root = Path(staging_dir) / plan.protocol
    if protocol_root.exists():
        shutil.rmtree(protocol_root)
    protocol_root.mkdir(parents=True, exist_ok=True)

    for record in [*plan.train_records, *plan.eval_records, *plan.eval_full_records]:
        destination = protocol_root / _relative_export_path(record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(record.source_path, destination)

    manifests = write_relative_export_manifests(plan, protocol_root / "manifests")
    write_qwen_label_jsonl(manifests["train"], protocol_root / "qwen_train.jsonl")

    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(protocol_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging_dir))
    return archive_path


def _write_manifest_csv(
    records: list[ExportRecord],
    path: Path,
    resolve_path: Callable[[ExportRecord], str],
) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "path": resolve_path(record),
                    "label": record.label,
                    "source": record.source,
                    "split": record.split,
                    "checksum": record.checksum,
                    "dataset": "genimage",
                    "generator": record.source,
                }
            )
    return path
