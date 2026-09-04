"""Tests for the shared evaluation-manifest selection contract.

Selection is the single source of truth for which dataset slices a baseline may
evaluate, so these tests pin the ``enabled`` semantics that every adapter now
inherits. CPU-only, network-free, and model-free.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aiforensics.config.load import load_config
from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import ManifestError
from aiforensics.data.selection import (
    selected_evaluation_manifests,
    training_manifest_path,
)

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase_ab_smoke.yaml"


def _config(tmp_path: Path) -> AppConfig:
    config = load_config(SMOKE_CONFIG)
    config.paths.data_root = tmp_path
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "train.csv"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "dev.csv"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen.csv"
    config.datasets.synthbuster.manifest = tmp_path / "synthbuster.csv"
    config.datasets.genimage_unseen.enabled = False
    config.datasets.synthbuster.enabled = False
    return config


def _write_manifest(path: Path, tmp_path: Path, sample_ids: list[str]) -> None:
    lines = ["sample_id,path,label,source,split,checksum"]
    for sample_id in sample_ids:
        content = f"img-{sample_id}".encode()
        filename = f"{sample_id}.png"
        (tmp_path / filename).write_bytes(content)
        checksum = hashlib.sha256(content).hexdigest()
        lines.append(f"{sample_id},{filename},real,smoke,dev,{checksum}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestEnabledFlags:
    def test_enabled_tiny_is_selected(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        selection = selected_evaluation_manifests(config)
        assert [r.sample_id for r in selection.records] == ["a"]
        assert [m.label for m in selection.manifests] == ["tiny_genimage"]

    def test_disabled_tiny_is_ignored_even_when_manifest_exists(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        _write_manifest(config.datasets.genimage_unseen.manifest, tmp_path, ["b"])
        config.datasets.genimage_unseen.enabled = True
        config.datasets.tiny_genimage.enabled = False

        selection = selected_evaluation_manifests(config)
        assert [r.sample_id for r in selection.records] == ["b"]

    def test_disabled_external_is_ignored_even_when_manifest_exists(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        _write_manifest(config.datasets.synthbuster.manifest, tmp_path, ["b"])
        config.datasets.synthbuster.enabled = False

        selection = selected_evaluation_manifests(config)
        assert [r.sample_id for r in selection.records] == ["a"]

    def test_dataset_order_is_fixed(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["t"])
        _write_manifest(config.datasets.genimage_unseen.manifest, tmp_path, ["u"])
        _write_manifest(config.datasets.synthbuster.manifest, tmp_path, ["s"])
        config.datasets.genimage_unseen.enabled = True
        config.datasets.synthbuster.enabled = True

        selection = selected_evaluation_manifests(config)
        assert [r.sample_id for r in selection.records] == ["t", "u", "s"]
        assert [m.label for m in selection.manifests] == [
            "tiny_genimage",
            "genimage_unseen",
            "synthbuster",
        ]


class TestMissingAndInvalid:
    def test_missing_enabled_manifest_warns_and_continues(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.genimage_unseen.manifest, tmp_path, ["u"])
        config.datasets.genimage_unseen.enabled = True

        selection = selected_evaluation_manifests(config)
        assert [r.sample_id for r in selection.records] == ["u"]
        assert any("tiny_genimage manifest missing" in w for w in selection.warnings)

    def test_invalid_existing_manifest_raises(self, tmp_path):
        config = _config(tmp_path)
        config.datasets.tiny_genimage.dev_manifest.write_text("not,a,manifest\n1,2\n")
        with pytest.raises(ManifestError):
            selected_evaluation_manifests(config)

    def test_strict_raises_when_nothing_selected(self, tmp_path):
        config = _config(tmp_path)
        with pytest.raises(ManifestError, match="No valid evaluation manifests found"):
            selected_evaluation_manifests(config)

    def test_non_strict_returns_empty_when_nothing_selected(self, tmp_path):
        config = _config(tmp_path)
        selection = selected_evaluation_manifests(config, strict=False)
        assert selection.records == ()
        assert selection.sample_ids == set()

    def test_all_disabled_selects_nothing(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        config.datasets.tiny_genimage.enabled = False

        selection = selected_evaluation_manifests(config, strict=False)
        assert selection.records == ()
        assert selection.warnings == ()


class TestTrainingManifest:
    def test_enabled_tiny_exposes_training_manifest(self, tmp_path):
        config = _config(tmp_path)
        assert training_manifest_path(config) == config.datasets.tiny_genimage.train_manifest

    def test_disabled_tiny_has_no_training_manifest(self, tmp_path):
        config = _config(tmp_path)
        config.datasets.tiny_genimage.enabled = False
        assert training_manifest_path(config) is None


class TestSampleIds:
    def test_sample_ids_reflect_selected_records(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a", "b"])
        selection = selected_evaluation_manifests(config)
        assert selection.sample_ids == {"a", "b"}
