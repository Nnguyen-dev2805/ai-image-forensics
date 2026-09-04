"""Tests for the GenImage-layout manifest builder.

The builder is the boundary between a dataset's on-disk directory layout and the
manifest contract every baseline reads, so these tests pin the mapping rules and
the two properties that make results trustworthy: no image crosses splits, and
the same tree always produces the same manifests.

All tests build tiny PNG/JPEG images under ``tmp_path``. CPU-only, network-free,
model-free.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from aiforensics.config.load import load_config
from aiforensics.config.models import AppConfig
from aiforensics.data.genimage import (
    build_genimage_manifests,
    discover_generator_dirs,
)
from aiforensics.data.manifest import ManifestError, compute_sha256, load_manifest

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase_ab_smoke.yaml"


def _write_image(path: Path, index: int, *, suffix: str = ".png") -> None:
    """Write one deterministic image whose content is unique per ``index``.

    Uniqueness matters: the builder deduplicates by content checksum, so a
    fixture helper that repeats pixel data would trigger real dedupe and make
    unrelated tests look wrong. Colors are drawn from a hash of ``index`` so the
    spread is wide enough that distinct indices survive JPEG quantization.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(index).encode()).digest()
    base = (digest[0], digest[1], digest[2])
    accent = (digest[3], digest[4], digest[5])

    image = Image.new("RGB", (8, 8), color=base)
    for offset in range(4):
        image.putpixel((offset, offset), accent)

    target = path.with_suffix(suffix)
    if suffix in {".jpg", ".jpeg"}:
        image.save(target, format="JPEG", quality=95)
    else:
        image.save(target, format="PNG")


def _make_tree(
    root: Path,
    generator: str,
    *,
    train_ai: int = 0,
    train_nature: int = 0,
    val_ai: int = 0,
    val_nature: int = 0,
    suffix: str = ".png",
    start: int = 0,
) -> None:
    """Create ``<generator>/{train,val}/{ai,nature}`` with the requested counts."""
    index = start
    for split_dir, label_dir, count in (
        ("train", "ai", train_ai),
        ("train", "nature", train_nature),
        ("val", "ai", val_ai),
        ("val", "nature", val_nature),
    ):
        for i in range(count):
            _write_image(
                root / generator / split_dir / label_dir / f"img_{i:04d}",
                index,
                suffix=suffix,
            )
            index += 1


def _config(
    tmp_path: Path,
    *,
    tiny_generators: list[str],
    unseen_generators: list[str],
    tiny_max: int = 0,
    unseen_max: int = 0,
    tiny_enabled: bool = True,
    unseen_enabled: bool = True,
    balance_labels: bool = True,
) -> AppConfig:
    config = load_config(SMOKE_CONFIG)
    config.project.phase = "phase_ab"
    config.paths.data_root = tmp_path / "data"
    config.paths.manifest_root = tmp_path / "manifests"
    config.paths.output_root = tmp_path / "outputs"

    tiny = config.datasets.tiny_genimage
    tiny.enabled = tiny_enabled
    tiny.generators = tiny_generators
    tiny.max_images = tiny_max
    tiny.balance_labels = balance_labels
    tiny.train_manifest = tmp_path / "manifests" / "train.csv"
    tiny.dev_manifest = tmp_path / "manifests" / "dev.csv"

    unseen = config.datasets.genimage_unseen
    unseen.enabled = unseen_enabled
    unseen.generators = unseen_generators
    unseen.max_images = unseen_max
    unseen.balance_labels = balance_labels
    unseen.split = "external"
    unseen.manifest = tmp_path / "manifests" / "external.csv"

    config.datasets.synthbuster.enabled = False
    return config


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


class TestDiscoverGeneratorDirs:
    def test_lists_generator_dirs_sorted(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "gen_b", train_ai=1)
        _make_tree(root, "gen_a", val_nature=1)
        assert discover_generator_dirs(root) == ["gen_a", "gen_b"]

    def test_ignores_dirs_without_split_subdirs(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "real_gen", train_ai=1)
        (root / "archives").mkdir(parents=True)
        (root / "notes").mkdir(parents=True)
        assert discover_generator_dirs(root) == ["real_gen"]

    def test_missing_root_returns_empty(self, tmp_path):
        assert discover_generator_dirs(tmp_path / "absent") == []


# ---------------------------------------------------------------------------
# label and split mapping
# ---------------------------------------------------------------------------


class TestLabelAndSplitMapping:
    def test_ai_maps_to_fake_and_nature_to_real(self, tmp_path):
        """Assert against the on-disk directory, not the derived sample_id.

        ``sample_id`` is built from the label, so comparing the two would still
        pass if the label mapping were inverted.
        """
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=2, train_nature=2, val_ai=2, val_nature=2)
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        build_genimage_manifests(config)
        train = load_manifest(config.datasets.tiny_genimage.train_manifest, data_root=root)

        assert len(train) == 4
        for record in train:
            expected = "fake" if record.path.parent.name == "ai" else "real"
            assert record.label == expected, f"{record.path} labelled {record.label}"
        assert {r.path.parent.name for r in train} == {"ai", "nature"}

    def test_val_directory_becomes_dev_split(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        build_genimage_manifests(config)
        train = load_manifest(config.datasets.tiny_genimage.train_manifest, data_root=root)
        dev = load_manifest(config.datasets.tiny_genimage.dev_manifest, data_root=root)

        assert {r.split for r in train} == {"train"}
        assert {r.split for r in dev} == {"dev"}

    def test_held_out_generators_land_in_external_split(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        _make_tree(root, "glide", train_ai=1, train_nature=1, val_ai=1, val_nature=1, start=100)
        config = _config(tmp_path, tiny_generators=["sdv5"], unseen_generators=["glide"])

        build_genimage_manifests(config)
        external = load_manifest(config.datasets.genimage_unseen.manifest, data_root=root)

        assert {r.split for r in external} == {"external"}
        assert {r.source for r in external} == {"glide"}
        # Both on-disk splits of a held-out generator feed evaluation.
        assert len(external) == 4

    def test_source_and_generator_record_the_generator_dir(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "imagenet_ai_0424_sdv5", train_ai=1, train_nature=1)
        config = _config(
            tmp_path,
            tiny_generators=["imagenet_ai_0424_sdv5"],
            unseen_generators=[],
            unseen_enabled=False,
        )
        # dev is required, so give the generator a val split too.
        _make_tree(root, "imagenet_ai_0424_sdv5", val_ai=1, val_nature=1, start=50)

        build_genimage_manifests(config)
        train = load_manifest(config.datasets.tiny_genimage.train_manifest, data_root=root)
        assert {r.source for r in train} == {"imagenet_ai_0424_sdv5"}
        assert {r.generator for r in train} == {"imagenet_ai_0424_sdv5"}
        assert {r.dataset for r in train} == {"genimage"}


# ---------------------------------------------------------------------------
# image formats
# ---------------------------------------------------------------------------


class TestImageFormats:
    def test_png_and_jpeg_are_both_collected(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=2, train_nature=2, suffix=".png")
        _make_tree(root, "sdv5", val_ai=2, val_nature=2, suffix=".jpg", start=20)
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        build_genimage_manifests(config)
        dev = load_manifest(config.datasets.tiny_genimage.dev_manifest, data_root=root)
        assert len(dev) == 4
        assert {r.path.suffix.lower() for r in dev} == {".jpg"}

    def test_unrecognized_extensions_are_ignored(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        (root / "sdv5" / "train" / "ai" / "notes.txt").write_text("not an image")
        (root / "sdv5" / "train" / "ai" / "archive.zip").write_bytes(b"PK\x03\x04")
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        build_genimage_manifests(config)
        train = load_manifest(config.datasets.tiny_genimage.train_manifest, data_root=root)
        assert len(train) == 2

    def test_format_skew_is_reported_when_labels_differ(self, tmp_path):
        """PNG-only real vs JPEG-only fake makes format a usable shortcut."""
        root = tmp_path / "data"
        for i in range(3):
            _write_image(root / "sdv5" / "train" / "nature" / f"r{i}", i, suffix=".png")
            _write_image(root / "sdv5" / "train" / "ai" / f"f{i}", i + 10, suffix=".jpg")
            _write_image(root / "sdv5" / "val" / "nature" / f"vr{i}", i + 20, suffix=".png")
            _write_image(root / "sdv5" / "val" / "ai" / f"vf{i}", i + 30, suffix=".jpg")
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        result = build_genimage_manifests(config)
        train = next(m for m in result.manifests if m.split == "train")
        skew = train.format_skew()
        assert skew is not None
        assert ".png" in skew and ".jpg" in skew

    def test_no_format_skew_when_labels_match(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=2, train_nature=2, val_ai=2, val_nature=2)
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        result = build_genimage_manifests(config)
        for manifest in result.manifests:
            assert manifest.format_skew() is None


# ---------------------------------------------------------------------------
# leakage control
# ---------------------------------------------------------------------------


class TestLeakageControl:
    def test_overlapping_generator_roles_are_rejected(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        config = _config(tmp_path, tiny_generators=["sdv5"], unseen_generators=["sdv5"])

        with pytest.raises(ManifestError, match="both tiny_genimage and genimage_unseen"):
            build_genimage_manifests(config)

    def test_duplicate_content_is_claimed_by_train_not_external(self, tmp_path):
        """A real image shared between generators must not be evaluated after training."""
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        _make_tree(root, "glide", train_ai=1, val_ai=1, start=200)

        # Same bytes as the sdv5 training real image, as GenImage reuses ImageNet.
        shared = root / "sdv5" / "train" / "nature" / "img_0000.png"
        duplicate = root / "glide" / "train" / "nature" / "img_0000.png"
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        duplicate.write_bytes(shared.read_bytes())

        config = _config(tmp_path, tiny_generators=["sdv5"], unseen_generators=["glide"])
        result = build_genimage_manifests(config)

        train = load_manifest(config.datasets.tiny_genimage.train_manifest, data_root=root)
        external = load_manifest(config.datasets.genimage_unseen.manifest, data_root=root)
        shared_checksum = compute_sha256(shared)

        assert shared_checksum in {r.checksum for r in train}
        assert shared_checksum not in {r.checksum for r in external}
        assert result.duplicate_checksums_skipped == 1

    def test_duplicate_content_within_train_and_dev_prefers_train(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        train_real = root / "sdv5" / "train" / "nature" / "img_0000.png"
        val_real = root / "sdv5" / "val" / "nature" / "img_0000.png"
        val_real.write_bytes(train_real.read_bytes())

        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )
        result = build_genimage_manifests(config)

        train = load_manifest(config.datasets.tiny_genimage.train_manifest, data_root=root)
        dev = load_manifest(config.datasets.tiny_genimage.dev_manifest, data_root=root)
        checksum = compute_sha256(train_real)

        assert checksum in {r.checksum for r in train}
        assert checksum not in {r.checksum for r in dev}
        assert result.duplicate_checksums_skipped == 1

    def test_no_checksum_appears_in_two_manifests(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=3, train_nature=3, val_ai=3, val_nature=3)
        _make_tree(root, "glide", train_ai=3, train_nature=3, start=300)
        config = _config(tmp_path, tiny_generators=["sdv5"], unseen_generators=["glide"])

        build_genimage_manifests(config)
        seen: set[str] = set()
        for manifest_path in (
            config.datasets.tiny_genimage.train_manifest,
            config.datasets.tiny_genimage.dev_manifest,
            config.datasets.genimage_unseen.manifest,
        ):
            for record in load_manifest(manifest_path, data_root=root):
                assert record.checksum not in seen, f"leaked across splits: {record.sample_id}"
                seen.add(record.checksum)


# ---------------------------------------------------------------------------
# per-generator capping
# ---------------------------------------------------------------------------


class TestSubsampling:
    def test_cap_is_per_generator_not_pooled(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        _make_tree(root, "big", train_ai=20, train_nature=20, start=100)
        _make_tree(root, "small", train_ai=20, train_nature=20, start=500)
        config = _config(
            tmp_path,
            tiny_generators=["sdv5"],
            unseen_generators=["big", "small"],
            unseen_max=10,
        )

        build_genimage_manifests(config)
        external = load_manifest(config.datasets.genimage_unseen.manifest, data_root=root)

        per_source: dict[str, int] = {}
        for record in external:
            per_source[record.source] = per_source.get(record.source, 0) + 1
        assert per_source == {"big": 10, "small": 10}

    def test_cap_balances_labels(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        _make_tree(root, "glide", train_ai=30, train_nature=10, start=100)
        config = _config(
            tmp_path,
            tiny_generators=["sdv5"],
            unseen_generators=["glide"],
            unseen_max=8,
        )

        build_genimage_manifests(config)
        external = load_manifest(config.datasets.genimage_unseen.manifest, data_root=root)
        assert sum(1 for r in external if r.label == "real") == 4
        assert sum(1 for r in external if r.label == "fake") == 4

    def test_zero_cap_keeps_everything(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=5, train_nature=5, val_ai=2, val_nature=2)
        config = _config(
            tmp_path,
            tiny_generators=["sdv5"],
            unseen_generators=[],
            unseen_enabled=False,
            tiny_max=0,
        )

        build_genimage_manifests(config)
        train = load_manifest(config.datasets.tiny_genimage.train_manifest, data_root=root)
        assert len(train) == 10

    def test_cap_larger_than_available_keeps_everything(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=2, train_nature=2, val_ai=2, val_nature=2)
        config = _config(
            tmp_path,
            tiny_generators=["sdv5"],
            unseen_generators=[],
            unseen_enabled=False,
            tiny_max=1000,
        )

        build_genimage_manifests(config)
        train = load_manifest(config.datasets.tiny_genimage.train_manifest, data_root=root)
        assert len(train) == 4

    def test_unbalanced_cap_uses_full_budget(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        _make_tree(root, "glide", train_ai=30, train_nature=10, start=100)
        config = _config(
            tmp_path,
            tiny_generators=["sdv5"],
            unseen_generators=["glide"],
            unseen_max=9,
            balance_labels=False,
        )

        build_genimage_manifests(config)
        external = load_manifest(config.datasets.genimage_unseen.manifest, data_root=root)
        assert len(external) == 9

    def test_cap_spreads_across_the_generator_not_just_the_head(self, tmp_path):
        """A head slice would sample one corner of a class-ordered directory."""
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        # 20 fakes named img_0000..img_0019; a head slice keeps only 0000-0004.
        _make_tree(root, "glide", train_ai=20, train_nature=20, start=100)
        config = _config(
            tmp_path,
            tiny_generators=["sdv5"],
            unseen_generators=["glide"],
            unseen_max=10,
        )

        build_genimage_manifests(config)
        external = load_manifest(config.datasets.genimage_unseen.manifest, data_root=root)
        fake_stems = sorted(r.path.stem for r in external if r.label == "fake")

        assert len(fake_stems) == 5
        # The selection must reach the tail of the directory listing.
        assert fake_stems[-1] > "img_0010", fake_stems


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_build_is_byte_identical(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=6, train_nature=6, val_ai=4, val_nature=4)
        _make_tree(root, "glide", train_ai=15, train_nature=15, start=200)
        config = _config(
            tmp_path,
            tiny_generators=["sdv5"],
            unseen_generators=["glide"],
            tiny_max=6,
            unseen_max=8,
        )

        build_genimage_manifests(config)
        first = {
            path: path.read_bytes()
            for path in (
                config.datasets.tiny_genimage.train_manifest,
                config.datasets.tiny_genimage.dev_manifest,
                config.datasets.genimage_unseen.manifest,
            )
        }

        build_genimage_manifests(config)
        for path, content in first.items():
            assert path.read_bytes() == content


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_missing_data_root_fails(self, tmp_path):
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )
        with pytest.raises(ManifestError, match="data_root does not exist"):
            build_genimage_manifests(config)

    def test_missing_generator_dir_lists_available(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        config = _config(
            tmp_path, tiny_generators=["absent"], unseen_generators=[], unseen_enabled=False
        )

        with pytest.raises(ManifestError) as exc:
            build_genimage_manifests(config)
        assert "absent" in str(exc.value)
        assert "sdv5" in str(exc.value)

    def test_enabled_dataset_without_generators_fails(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1)
        config = _config(tmp_path, tiny_generators=[], unseen_generators=[], unseen_enabled=False)

        with pytest.raises(ManifestError, match="no generators are configured"):
            build_genimage_manifests(config)

    def test_generator_without_val_split_fails_for_dev(self, tmp_path):
        """A dev manifest is required, so a train-only generator cannot be in-distribution."""
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=2, train_nature=2)
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        with pytest.raises(ManifestError, match="no dev records"):
            build_genimage_manifests(config)

    def test_empty_generator_dir_reports_warning_and_fails(self, tmp_path):
        root = tmp_path / "data"
        (root / "empty" / "train").mkdir(parents=True)
        config = _config(
            tmp_path, tiny_generators=["empty"], unseen_generators=[], unseen_enabled=False
        )

        with pytest.raises(ManifestError, match="no train records"):
            build_genimage_manifests(config)

    def test_disabled_datasets_write_nothing(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        config = _config(
            tmp_path,
            tiny_generators=["sdv5"],
            unseen_generators=["sdv5"],
            tiny_enabled=False,
            unseen_enabled=False,
        )

        result = build_genimage_manifests(config)
        assert result.manifests == ()
        assert not config.datasets.tiny_genimage.train_manifest.exists()


# ---------------------------------------------------------------------------
# manifest contract
# ---------------------------------------------------------------------------


class TestManifestContract:
    def test_written_manifests_pass_validation(self, tmp_path):
        from aiforensics.data.manifest import validate_manifest

        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=3, train_nature=3, val_ai=2, val_nature=2)
        _make_tree(root, "glide", train_ai=2, train_nature=2, start=100)
        config = _config(tmp_path, tiny_generators=["sdv5"], unseen_generators=["glide"])

        build_genimage_manifests(config)
        records = []
        for path in (
            config.datasets.tiny_genimage.train_manifest,
            config.datasets.tiny_genimage.dev_manifest,
            config.datasets.genimage_unseen.manifest,
        ):
            records.extend(load_manifest(path, data_root=root))

        result = validate_manifest(records)
        assert result.is_valid, result.errors

    def test_paths_are_relative_to_data_root(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=1, train_nature=1, val_ai=1, val_nature=1)
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        build_genimage_manifests(config)
        raw = config.datasets.tiny_genimage.train_manifest.read_text(encoding="utf-8")
        assert "sdv5/train/" in raw
        assert str(root) not in raw

    def test_checksums_match_image_bytes(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=2, train_nature=2, val_ai=1, val_nature=1)
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        build_genimage_manifests(config)
        for record in load_manifest(config.datasets.tiny_genimage.train_manifest, data_root=root):
            assert record.checksum == compute_sha256(record.path)

    def test_build_result_counts_match_manifests(self, tmp_path):
        root = tmp_path / "data"
        _make_tree(root, "sdv5", train_ai=3, train_nature=2, val_ai=1, val_nature=1)
        config = _config(
            tmp_path, tiny_generators=["sdv5"], unseen_generators=[], unseen_enabled=False
        )

        result = build_genimage_manifests(config)
        train = next(m for m in result.manifests if m.split == "train")
        assert train.record_count == 5
        assert train.fake_count == 3
        assert train.real_count == 2
        assert train.generators == ("sdv5",)
