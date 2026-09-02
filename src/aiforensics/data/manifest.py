import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from aiforensics.config.models import AppConfig


class ManifestError(ValueError):
    """Exception raised for manifest structural errors (e.g. missing columns, invalid CSV)."""

    pass


class ManifestValidationResult(BaseModel):
    is_valid: bool
    total_records: int
    records_by_label: dict[str, int]
    records_by_split: dict[str, int]
    records_by_source: dict[str, int]
    duplicate_sample_ids: list[str]
    duplicate_checksums: dict[str, list[str]]
    missing_files: list[str]
    checksum_mismatches: list[str]
    errors: list[str]
    warnings: list[str]


class ManifestRecord(BaseModel):
    sample_id: str = Field(min_length=1)
    path: Path
    label: Literal["real", "fake"]
    source: str
    split: Literal["train", "dev", "test", "external", "smoke"]
    checksum: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    # Optional fields
    dataset: str | None = None
    generator: str | None = None
    width: int | None = None
    height: int | None = None
    mime_type: str | None = None
    license: str | None = None
    notes: str | None = None


def compute_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found or is a directory: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        # Read the file in 64kb chunks
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_manifest(path: Path | str, *, data_root: Path | None = None) -> list[ManifestRecord]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ManifestError(f"Manifest file missing: {manifest_path}")

    records = []
    required_cols = {"sample_id", "path", "label", "source", "split", "checksum"}

    # Base path for relative path resolution
    base_dir = data_root if data_root is not None else manifest_path.parent

    try:
        with open(manifest_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ManifestError(f"Manifest file is empty or missing headers: {manifest_path}")

            headers = set(reader.fieldnames)
            missing_cols = required_cols - headers
            if missing_cols:
                raise ManifestError(f"Missing required columns: {sorted(missing_cols)}")

            for i, row in enumerate(reader, start=2):
                try:
                    # Remove empty strings so they fall back to Pydantic defaults (None)
                    for k in list(row.keys()):
                        if row[k] == "":
                            del row[k]

                    # Resolve relative image path against base_dir and make absolute
                    p = Path(row["path"])
                    if not p.is_absolute():
                        row["path"] = str((base_dir / p).resolve())
                    records.append(ManifestRecord(**row))
                except Exception as e:
                    raise ManifestError(f"Row {i} is invalid: {e}") from e

    except csv.Error as e:
        raise ManifestError(f"Failed to parse CSV: {e}") from e

    return records


def write_manifest(records: list[ManifestRecord], path: Path | str) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fields = list(ManifestRecord.model_fields.keys())

    # Required columns order per spec
    required_keys = ["sample_id", "path", "label", "source", "split", "checksum"]
    optional_keys = [k for k in fields if k not in required_keys]
    header_order = required_keys + optional_keys

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        writer.writeheader()

        for record in records:
            # We must output the exact string value stored in record.path
            # Pydantic's model_dump treats Path as strings or we can serialize it
            row = record.model_dump(exclude_none=True)
            # Ensure path is a string exactly as it was provided (or just str(path))
            # The spec says write_manifest() writes path values exactly as stored.
            row["path"] = str(record.path)
            writer.writerow(row)


def validate_manifest(records: list[ManifestRecord]) -> ManifestValidationResult:
    if not records:
        return ManifestValidationResult(
            is_valid=False,
            total_records=0,
            records_by_label={},
            records_by_split={},
            records_by_source={},
            duplicate_sample_ids=[],
            duplicate_checksums={},
            missing_files=[],
            checksum_mismatches=[],
            errors=["Manifest contains no records."],
            warnings=[],
        )

    records_by_label: dict[str, int] = defaultdict(int)
    records_by_split: dict[str, int] = defaultdict(int)
    records_by_source: dict[str, int] = defaultdict(int)

    duplicate_sample_ids: list[str] = []
    duplicate_checksums: dict[str, list[str]] = defaultdict(list)
    missing_files: list[str] = []
    checksum_mismatches: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    seen_ids = set()
    checksum_to_ids: dict[str, list[str]] = defaultdict(list)

    # Labels per split checker
    labels_per_split: dict[str, set] = defaultdict(set)

    for rec in records:
        # Update metrics
        records_by_label[rec.label] += 1
        records_by_split[rec.split] += 1
        records_by_source[rec.source] += 1

        labels_per_split[rec.split].add(rec.label)

        # 1. Uniqueness of sample_id
        if rec.sample_id in seen_ids:
            duplicate_sample_ids.append(rec.sample_id)
            errors.append(f"Duplicate sample_id found: {rec.sample_id}")
        else:
            seen_ids.add(rec.sample_id)

        # Add to checksum tracker
        checksum_to_ids[rec.checksum].append(rec.sample_id)

        # 2. File exists and 3. Checksum verification
        if not rec.path.is_file():
            missing_files.append(str(rec.path))
            errors.append(f"File missing or not a file: {rec.path}")
        else:
            try:
                actual_checksum = compute_sha256(rec.path)
                if actual_checksum != rec.checksum:
                    checksum_mismatches.append(str(rec.path))
                    errors.append(
                        f"Checksum mismatch for {rec.path}: "
                        f"expected {rec.checksum}, got {actual_checksum}"
                    )
            except Exception as e:
                errors.append(f"Failed to read file to compute checksum for {rec.path}: {e}")

    # Process duplicates in checksums
    for chk, ids in checksum_to_ids.items():
        if len(ids) > 1:
            duplicate_checksums[chk] = ids
            errors.append(f"Duplicate checksum {chk} found in records: {ids}")

    # Split label balance validation
    for split, count in records_by_split.items():
        if count >= 2:
            if len(labels_per_split[split]) < 2:
                split_labels = list(labels_per_split[split])
                errors.append(
                    f"Split '{split}' has {count} records but contains only "
                    f"label(s): {split_labels}. Must contain both 'real' and 'fake'."
                )

    is_valid = len(errors) == 0

    return ManifestValidationResult(
        is_valid=is_valid,
        total_records=len(records),
        records_by_label=dict(records_by_label),
        records_by_split=dict(records_by_split),
        records_by_source=dict(records_by_source),
        duplicate_sample_ids=duplicate_sample_ids,
        duplicate_checksums=dict(duplicate_checksums),
        missing_files=missing_files,
        checksum_mismatches=checksum_mismatches,
        errors=errors,
        warnings=warnings,
    )


def prepare_smoke_manifest(config: AppConfig) -> ManifestValidationResult:
    """Create deterministic tiny smoke fixtures and write smoke manifests."""
    from PIL import Image

    data_root = config.paths.data_root
    manifest_root = config.paths.manifest_root

    data_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    # 1. Create simple deterministic colors for PNG images
    # We will use Red, Green, Blue, Yellow (or just random stable configs)
    images_config = [
        ("real_0001", "real_0001.png", "real", (255, 0, 0)),
        ("real_0002", "real_0002.png", "real", (0, 255, 0)),
        ("fake_0001", "fake_0001.png", "fake", (0, 0, 255)),
        ("fake_0002", "fake_0002.png", "fake", (255, 255, 0)),
    ]

    records = []

    for sample_id, filename, label, color in images_config:
        img_path = data_root / filename
        if not img_path.exists():
            img = Image.new("RGB", (4, 4), color=color)
            img.save(img_path)

        # For smoke paths, paths in csv files should be relative to
        # config.paths.data_root. We store the relative path in the model for
        # writing; load_manifest resolves it against data_root for validation.
        csum = compute_sha256(img_path)

        split = "train" if "0001" in sample_id else "dev"
        record_path = Path(filename)

        record = ManifestRecord(
            sample_id=f"smoke/{split}/{sample_id}",
            path=record_path,
            label=label,
            source="smoke",
            split=split,
            checksum=csum,
        )
        records.append(record)

    train_records = [r for r in records if r.split == "train"]
    dev_records = [r for r in records if r.split == "dev"]

    write_manifest(train_records, manifest_root / "smoke_train.csv")
    write_manifest(dev_records, manifest_root / "smoke_dev.csv")

    # Before validation, load the manifests so relative paths resolve
    # against data_root and checksum validation sees absolute paths.

    validate_train = load_manifest(manifest_root / "smoke_train.csv", data_root=data_root)
    validate_dev = load_manifest(manifest_root / "smoke_dev.csv", data_root=data_root)

    combined = validate_train + validate_dev
    return validate_manifest(combined)
