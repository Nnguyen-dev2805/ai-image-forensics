import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aiforensics.config import load_config
from aiforensics.data.manifest import (
    ManifestError,
    ManifestRecord,
    compute_sha256,
    load_manifest,
    prepare_smoke_manifest,
    validate_manifest,
    write_manifest,
)


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "test_image.png"
    # Create a small dummy file
    img_path.write_bytes(b"dummy image data")
    return img_path


def test_compute_sha256(tmp_image: Path):
    # known SHA-256 for "dummy image data"
    import hashlib

    expected_hex = hashlib.sha256(b"dummy image data").hexdigest()

    actual_hex = compute_sha256(tmp_image)
    assert actual_hex == expected_hex


def test_write_and_load_roundtrip(tmp_path: Path, tmp_image: Path):
    csum = compute_sha256(tmp_image)
    records = [
        ManifestRecord(
            sample_id="file1",
            path=tmp_image,
            label="real",
            source="test_source",
            split="train",
            checksum=csum,
            dataset="test_ds",
        ),
        ManifestRecord(
            sample_id="file2",
            path=tmp_path / "another_image.png",
            label="fake",
            source="test_source",
            split="dev",
            checksum=csum,
        ),
    ]
    manifest_csv = tmp_path / "manifest.csv"

    write_manifest(records, manifest_csv)

    loaded = load_manifest(manifest_csv)
    assert len(loaded) == 2
    assert loaded[0].sample_id == "file1"
    assert loaded[0].label == "real"
    assert loaded[0].dataset == "test_ds"
    assert str(loaded[0].path) == str(tmp_image)

    assert loaded[1].sample_id == "file2"
    assert loaded[1].dataset is None


def test_load_manifest_missing_columns(tmp_path: Path):
    manifest_csv = tmp_path / "bad_manifest.csv"
    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "path", "label"])  # Missing source, split, checksum
        writer.writerow(["file1", "img.png", "real"])

    with pytest.raises(ManifestError, match="Missing required columns"):
        load_manifest(manifest_csv)


def test_validate_manifest_duplicate_sample_id(tmp_path: Path, tmp_image: Path):
    csum = compute_sha256(tmp_image)
    record1 = ManifestRecord(
        sample_id="dup_id", path=tmp_image, label="real", source="s", split="train", checksum=csum
    )
    record2 = ManifestRecord(
        sample_id="dup_id", path=tmp_image, label="fake", source="s", split="train", checksum=csum
    )

    result = validate_manifest([record1, record2])
    assert not result.is_valid
    assert "dup_id" in result.duplicate_sample_ids
    assert any("Duplicate sample_id" in err for err in result.errors)


def test_validate_manifest_duplicate_checksum(tmp_path: Path, tmp_image: Path):
    csum = compute_sha256(tmp_image)
    # create another file with same content or just use same file for
    # different logical samples to trigger checksum duplicate
    # the unique constraint usually implies images are unique, though here
    # duplicate files have same checksum
    img2 = tmp_path / "img2.png"
    img2.write_bytes(b"dummy image data")

    record1 = ManifestRecord(
        sample_id="id1", path=tmp_image, label="real", source="s", split="train", checksum=csum
    )
    record2 = ManifestRecord(
        sample_id="id2", path=img2, label="fake", source="s", split="train", checksum=csum
    )

    result = validate_manifest([record1, record2])
    assert not result.is_valid
    assert csum in result.duplicate_checksums
    assert any("Duplicate checksum" in err for err in result.errors)


def test_validate_manifest_checksum_mismatch(tmp_path: Path, tmp_image: Path):
    # valid file, wrong checksum
    bad_csum = "0" * 64
    # To pass split check, add another record
    img2 = tmp_path / "img2.png"
    img2.write_bytes(b"valid data")
    csum2 = compute_sha256(img2)

    record1 = ManifestRecord(
        sample_id="id1", path=tmp_image, label="real", source="s", split="train", checksum=bad_csum
    )
    record2 = ManifestRecord(
        sample_id="id2", path=img2, label="fake", source="s", split="train", checksum=csum2
    )

    result = validate_manifest([record1, record2])
    assert not result.is_valid
    assert str(tmp_image) in result.checksum_mismatches
    assert any("Checksum mismatch" in err for err in result.errors)


def test_validate_manifest_missing_files(tmp_path: Path):
    csum = "a" * 64
    missing_path = tmp_path / "does_not_exist.png"
    csum2 = "b" * 64
    missing_path2 = tmp_path / "does_not_exist2.png"

    record1 = ManifestRecord(
        sample_id="id1", path=missing_path, label="real", source="s", split="train", checksum=csum
    )
    record2 = ManifestRecord(
        sample_id="id2", path=missing_path2, label="fake", source="s", split="train", checksum=csum2
    )

    result = validate_manifest([record1, record2])
    assert not result.is_valid
    assert str(missing_path) in result.missing_files
    assert any("not a file" in err for err in result.errors)


def test_validate_manifest_split_label_balance(tmp_path: Path, tmp_image: Path):
    # two real records in train split
    csum = compute_sha256(tmp_image)
    img2 = tmp_path / "img2.png"
    img2.write_bytes(b"img2")
    csum2 = compute_sha256(img2)

    record1 = ManifestRecord(
        sample_id="id1", path=tmp_image, label="real", source="s", split="train", checksum=csum
    )
    record2 = ManifestRecord(
        sample_id="id2", path=img2, label="real", source="s", split="train", checksum=csum2
    )

    result = validate_manifest([record1, record2])
    assert not result.is_valid
    assert any("Must contain both 'real' and 'fake'" in err for err in result.errors)


def test_prepare_smoke_manifest():
    # Use real config phase_ab_smoke.yaml
    config = load_config("configs/phase_ab_smoke.yaml")

    # The spec allows the smoke prepare test to write to the configured smoke paths.
    result = prepare_smoke_manifest(config)
    assert result.is_valid
    assert result.total_records == 4

    # Check that it actually generated
    assert (config.paths.data_root / "real_0001.png").is_file()
    assert (config.paths.manifest_root / "smoke_train.csv").is_file()
    assert (config.paths.manifest_root / "smoke_dev.csv").is_file()


def test_cli_prepare_smoke_writes_json():
    # This invokes the actual CLI
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aiforensics.cli.main",
            "prepare",
            "--config",
            "configs/phase_ab_smoke.yaml",
        ],
        capture_output=True,
        text=True,
    )

    # It should succeed
    assert result.returncode == 0
    assert "[prepare]" in result.stdout
    assert "valid=True" in result.stdout

    config = load_config("configs/phase_ab_smoke.yaml")
    summary_path = config.paths.output_root / "manifest_validation.json"

    assert summary_path.is_file()
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)
        assert data["is_valid"] is True
        assert data["total_records"] == 4
