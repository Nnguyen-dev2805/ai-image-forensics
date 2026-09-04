"""Tests for run scope: the experiment identity stamped into run artifacts.

Scope binds a run directory to the config that produced it, so these tests pin
what changes the digest, what does not, and how unreadable scope files behave.
CPU-only, network-free, and model-free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aiforensics.config.load import load_config
from aiforensics.config.models import AppConfig
from aiforensics.runs.scope import (
    SCOPE_FILENAME,
    SCOPE_VERSION,
    RunScope,
    compute_run_scope,
    read_run_scope,
    scope_matches,
    write_run_scope,
)

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase_ab_smoke.yaml"


def _config(tmp_path: Path) -> AppConfig:
    config = load_config(SMOKE_CONFIG)
    config.paths.data_root = tmp_path
    config.paths.output_root = tmp_path / "outputs"
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
        lines.append(f"{sample_id},{filename},real,smoke,dev,{hashlib.sha256(content).hexdigest()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestScopeStability:
    def test_same_config_yields_same_scope_id(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a", "b"])
        assert compute_run_scope(config).scope_id == compute_run_scope(config).scope_id

    def test_seed_change_does_not_change_scope(self, tmp_path):
        """Seeds are a baseline knob, not an evaluation-slice change."""
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        before = compute_run_scope(config).scope_id
        config.baselines.clip_probe.seeds = [1, 2, 3]
        config.runtime.seed = 999
        assert compute_run_scope(config).scope_id == before

    def test_model_id_change_does_not_change_scope(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        before = compute_run_scope(config).scope_id
        config.baselines.qwen_vl.model_id = "some/other-model"
        assert compute_run_scope(config).scope_id == before


class TestScopeSensitivity:
    def test_enabling_a_dataset_changes_scope(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        before = compute_run_scope(config).scope_id
        config.datasets.genimage_unseen.enabled = True
        assert compute_run_scope(config).scope_id != before

    def test_disabling_tiny_changes_scope(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        before = compute_run_scope(config).scope_id
        config.datasets.tiny_genimage.enabled = False
        assert compute_run_scope(config).scope_id != before

    def test_manifest_content_change_changes_scope(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        before = compute_run_scope(config).scope_id
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a", "b"])
        assert compute_run_scope(config).scope_id != before

    def test_phase_change_changes_scope(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        before = compute_run_scope(config).scope_id
        config.project.phase = "phase_ab"
        assert compute_run_scope(config).scope_id != before

    def test_data_root_change_changes_scope(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        before = compute_run_scope(config).scope_id
        config.paths.data_root = tmp_path / "elsewhere"
        assert compute_run_scope(config).scope_id != before

    def test_disabled_dataset_manifest_path_does_not_change_scope(self, tmp_path):
        """A disabled slice contributes nothing, so its stale path is inert."""
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        before = compute_run_scope(config).scope_id
        config.datasets.synthbuster.manifest = tmp_path / "totally_different.csv"
        assert compute_run_scope(config).scope_id == before


class TestScopeIo:
    def test_write_then_read_roundtrip(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        scope = compute_run_scope(config)
        path = tmp_path / "run" / SCOPE_FILENAME
        write_run_scope(path, scope)
        assert read_run_scope(path) == scope

    def test_written_file_ends_with_newline(self, tmp_path):
        config = _config(tmp_path)
        path = tmp_path / "run" / SCOPE_FILENAME
        write_run_scope(path, compute_run_scope(config))
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_read_missing_file_returns_none(self, tmp_path):
        assert read_run_scope(tmp_path / "absent.json") is None

    def test_read_malformed_json_returns_none(self, tmp_path):
        path = tmp_path / SCOPE_FILENAME
        path.write_text("{not json", encoding="utf-8")
        assert read_run_scope(path) is None

    def test_read_non_object_returns_none(self, tmp_path):
        path = tmp_path / SCOPE_FILENAME
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert read_run_scope(path) is None

    def test_read_invalid_payload_returns_none(self, tmp_path):
        path = tmp_path / SCOPE_FILENAME
        path.write_text(json.dumps({"scope_id": ""}), encoding="utf-8")
        assert read_run_scope(path) is None


class TestScopeMatching:
    def test_matching_scope_matches(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        scope = compute_run_scope(config)
        run_dir = tmp_path / "run"
        write_run_scope(run_dir / SCOPE_FILENAME, scope)
        assert scope_matches(run_dir, scope)

    def test_missing_scope_file_never_matches(self, tmp_path):
        config = _config(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert not scope_matches(run_dir, compute_run_scope(config))

    def test_different_scope_id_does_not_match(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        run_dir = tmp_path / "run"
        write_run_scope(run_dir / SCOPE_FILENAME, compute_run_scope(config))

        other = _config(tmp_path)
        other.datasets.genimage_unseen.enabled = True
        assert not scope_matches(run_dir, compute_run_scope(other))

    def test_different_scope_version_does_not_match(self, tmp_path):
        config = _config(tmp_path)
        scope = compute_run_scope(config)
        run_dir = tmp_path / "run"
        write_run_scope(run_dir / SCOPE_FILENAME, scope)

        future = RunScope(
            scope_version="999",
            scope_id=scope.scope_id,
            phase=scope.phase,
            data_root=scope.data_root,
            datasets=scope.datasets,
            sample_id_count=scope.sample_id_count,
            sample_ids_digest=scope.sample_ids_digest,
        )
        assert not scope_matches(run_dir, future)


class TestScopePayload:
    def test_scope_records_sample_count_and_version(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a", "b", "c"])
        scope = compute_run_scope(config)
        assert scope.sample_id_count == 3
        assert scope.scope_version == SCOPE_VERSION

    def test_scope_records_dataset_states(self, tmp_path):
        config = _config(tmp_path)
        _write_manifest(config.datasets.tiny_genimage.dev_manifest, tmp_path, ["a"])
        scope = compute_run_scope(config)
        assert scope.datasets["genimage_unseen"] == "disabled"
        assert scope.datasets["tiny_genimage"].startswith("enabled:")

    def test_missing_manifests_still_yield_a_scope(self, tmp_path):
        """Scope must be computable for a run that is about to fail or defer."""
        config = _config(tmp_path)
        scope = compute_run_scope(config)
        assert scope.sample_id_count == 0
        assert scope.scope_id
