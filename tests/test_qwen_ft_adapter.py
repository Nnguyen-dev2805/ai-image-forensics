"""Tests for the qwen_ft inference adapter using a patched model runner."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from PIL import Image

from aiforensics.baselines.qwen_ft import QwenFTAdapter
from aiforensics.config import load_config
from aiforensics.schemas.predictions import load_predictions


@pytest.fixture()
def adapter_config(tmp_path: Path) -> Path:
    """Build a minimal enabled-qwen_ft config over two local fixture images."""
    images_dir = tmp_path / "images"
    manifest_rows = ["sample_id,path,label,source,split,checksum"]
    for name, label, color in [
        ("fake_0001", "fake", (0, 0, 255)),
        ("real_0001", "real", (255, 0, 0)),
    ]:
        image_path = images_dir / f"{name}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=color).save(image_path)
        checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
        manifest_rows.append(
            f"genimage/dev/test-generator/{name},{image_path},{label},test-generator,dev,{checksum}"
        )
    dev_manifest = tmp_path / "manifests" / "eval_small.csv"
    dev_manifest.parent.mkdir(parents=True, exist_ok=True)
    dev_manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    config = {
        "project": {
            "name": "qwen-ft-adapter-test",
            "phase": "qwen_ft_adapter_test",
            "description": "adapter test config",
        },
        "paths": {
            "data_root": str(tmp_path / "data"),
            "manifest_root": str(tmp_path / "manifests"),
            "cache_root": str(tmp_path / ".cache"),
            "output_root": str(tmp_path / "outputs"),
            "external_root": str(tmp_path / "external"),
        },
        "runtime": {
            "python": "3.10",
            "seed": 70,
            "device": "cpu",
            "batch_size": 1,
            "num_workers": 0,
            "fail_fast": False,
        },
        "datasets": {
            "tiny_genimage": {
                "enabled": True,
                "source": "test-fixtures",
                "use_original_split": False,
                "train_manifest": str(dev_manifest),
                "dev_manifest": str(dev_manifest),
                "generators": [],
                "max_images": 0,
                "balance_labels": True,
            },
            "genimage_unseen": {
                "enabled": False,
                "generators": [],
                "max_images": 0,
                "balance_labels": True,
                "split": "external",
                "manifest": str(tmp_path / "manifests" / "external.csv"),
            },
            "synthbuster": {
                "enabled": False,
                "max_images": 0,
                "balance_labels": True,
                "split": "external",
                "manifest": str(tmp_path / "manifests" / "synthbuster.csv"),
            },
        },
        "baselines": {
            "clip_probe": {
                "enabled": False,
                "model_family": "synthetic",
                "model_name": "smoke-embedding",
                "pretrained": "none",
                "classifier": "logistic_regression",
                "seeds": [70],
                "cache_embeddings": False,
            },
            "qwen_vl": {
                "enabled": False,
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "prompt_id": "qwen_json_v1",
                "temperature": 0.0,
                "max_new_tokens": 32,
                "cache_outputs": False,
                "allow_deferred": True,
            },
            "assisted_qwen": {
                "enabled": False,
                "base_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "prompt_id": "assisted_qwen_json_v1",
                "assistant_source": "clip_probe",
                "include_classifier_pred": True,
                "include_fake_probability": True,
                "temperature": 0.0,
                "max_new_tokens": 32,
                "cache_outputs": False,
                "allow_deferred": True,
            },
            "npr": {
                "enabled": False,
                "repo_url": "https://github.com/example/NPR",
                "repo_commit": None,
                "checkpoint_path": "external/NPR.pth",
                "checkpoint_sha256": None,
                "batch_size": 1,
                "allow_deferred": True,
            },
            "qwen_ft": {
                "enabled": True,
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "adapter_uri": "gs://aiforensics-qwen-ft-579187260419/checkpoints/test/final_adapter",
                "prompt_id": "qwen_ft_label_json_v1",
                "temperature": 0.0,
                "max_new_tokens": 32,
                "cache_outputs": False,
                "allow_deferred": True,
                "dtype": "float16",
                "output_fields": ["label"],
            },
        },
        "evaluation": {
            "labels": {"negative": "real", "positive": "fake"},
            "metrics": ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc"],
            "group_by": ["source", "split"],
        },
        "report": {
            "filename": "adapter_test_report.md",
            "include_failure_notes": True,
            "include_explanations_sample": False,
            "explanation_sample_size": 0,
        },
    }
    config_path = tmp_path / "qwen_ft_adapter_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    # load_config resolves paths relative to the nearest repo root; mark the
    # tmp tree as one, like the CLI smoke tests do.
    (tmp_path / "pyproject.toml").touch()
    return config_path


def test_qwen_ft_adapter_writes_label_only_predictions(
    tmp_path: Path, adapter_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(adapter_config)
    adapter = QwenFTAdapter()

    monkeypatch.setattr(adapter, "_get_qwen_device", lambda config: "cpu")
    monkeypatch.setattr(
        adapter, "_load_model", lambda config, device: ("fake-model", "fake-processor")
    )
    monkeypatch.setattr(adapter, "_generate_one_image", lambda *args, **kwargs: '{"label":"fake"}')

    run_dir = tmp_path / "run"
    result = adapter.run(config=cfg, output_dir=run_dir, run_id="test_qwen_ft_run")

    assert result.status == "completed"
    assert result.prediction_path is not None
    preds = load_predictions(result.prediction_path)
    assert len(preds) == 2
    assert all(p.model_name == "qwen_ft" for p in preds)
    assert all(p.label_pred == "fake" for p in preds)
    assert all(p.score_fake is None for p in preds)
    assert all(p.parse_status == "parsed" for p in preds)
    assert all(p.prompt_id == "qwen_ft_label_json_v1" for p in preds)
    assert all(p.explanation == "" for p in preds)


def test_qwen_ft_adapter_defers_when_disabled(tmp_path: Path, adapter_config: Path) -> None:
    cfg = load_config(adapter_config)
    cfg.baselines.qwen_ft.enabled = False

    result = QwenFTAdapter().run(config=cfg, output_dir=tmp_path / "run", run_id="test_disabled")

    assert result.status == "deferred"
    assert "disabled" in (result.reason or "")


def test_qwen_ft_adapter_defers_without_adapter_uri(tmp_path: Path, adapter_config: Path) -> None:
    cfg = load_config(adapter_config)
    cfg.baselines.qwen_ft.adapter_uri = ""

    result = QwenFTAdapter().run(config=cfg, output_dir=tmp_path / "run", run_id="test_no_adapter")

    assert result.status == "deferred"
    assert "adapter_uri" in (result.reason or "")


def test_qwen_ft_adapter_failed_parse_becomes_unknown_label(
    tmp_path: Path, adapter_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(adapter_config)
    adapter = QwenFTAdapter()

    monkeypatch.setattr(adapter, "_get_qwen_device", lambda config: "cpu")
    monkeypatch.setattr(
        adapter, "_load_model", lambda config, device: ("fake-model", "fake-processor")
    )
    monkeypatch.setattr(adapter, "_generate_one_image", lambda *args, **kwargs: "garbage output")

    result = adapter.run(config=cfg, output_dir=tmp_path / "run", run_id="test_bad_parse")

    assert result.status == "completed"
    preds = load_predictions(result.prediction_path)
    assert all(p.label_pred == "unknown" for p in preds)
    assert all(p.parse_status == "failed" for p in preds)


def test_qwen_ft_adapter_resolves_relative_manifest_path_against_data_root(tmp_path: Path) -> None:
    adapter = QwenFTAdapter()
    image = tmp_path / "data" / "eval-small" / "gen" / "ai" / "0.JPEG"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")

    resolved = adapter._resolve_image_path(Path("eval-small/gen/ai/0.JPEG"), tmp_path / "data")

    assert resolved == image


def test_compute_adapter_fingerprint_changes_on_adapter_modification(tmp_path: Path) -> None:
    from aiforensics.baselines.qwen_ft.adapter import compute_adapter_fingerprint

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config_file = adapter_dir / "adapter_config.json"
    weights_file = adapter_dir / "adapter_model.safetensors"

    config_file.write_text('{"r": 8}', encoding="utf-8")
    weights_file.write_bytes(b"initial-weights")

    fp1 = compute_adapter_fingerprint(str(adapter_dir))

    # Overwrite weights file with different content
    weights_file.write_bytes(b"retrained-weights-with-new-params")
    fp2 = compute_adapter_fingerprint(str(adapter_dir))

    assert fp1 != fp2


def test_qwen_ft_adapter_cache_key_includes_adapter_fingerprint(
    tmp_path: Path, adapter_config: Path
) -> None:
    from aiforensics.data.manifest import ManifestRecord

    cfg = load_config(adapter_config)
    adapter = QwenFTAdapter()

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights-v1")

    cfg.baselines.qwen_ft.adapter_uri = str(adapter_dir)
    record = ManifestRecord(
        sample_id="test_01",
        path=str(tmp_path / "images" / "fake_0001.png"),
        label="fake",
        source="test",
        split="dev",
        checksum="a" * 64,
    )

    key1 = adapter._get_cache_key(record, cfg)

    # Simulate retrain into the exact same folder
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights-v2-retrained")
    key2 = adapter._get_cache_key(record, cfg)

    assert key1 != key2


def test_qwen_ft_adapter_cache_key_includes_pixels(tmp_path: Path, adapter_config: Path) -> None:
    from aiforensics.data.manifest import ManifestRecord

    cfg = load_config(adapter_config)
    adapter = QwenFTAdapter()

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights-v1")

    cfg.baselines.qwen_ft.adapter_uri = str(adapter_dir)
    record = ManifestRecord(
        sample_id="test_01",
        path=str(tmp_path / "images" / "fake_0001.png"),
        label="fake",
        source="test",
        split="dev",
        checksum="a" * 64,
    )

    cfg.baselines.qwen_ft.min_pixels = 50176
    cfg.baselines.qwen_ft.max_pixels = 200704
    key1 = adapter._get_cache_key(record, cfg)

    cfg.baselines.qwen_ft.max_pixels = 100352
    key2 = adapter._get_cache_key(record, cfg)

    assert key1 != key2


def test_qwen_ft_config_validates_min_max_pixels() -> None:
    from pydantic import ValidationError

    from aiforensics.config.models import QwenFTConfig

    with pytest.raises(ValidationError, match="min_pixels .* must be <= max_pixels"):
        QwenFTConfig(min_pixels=300000, max_pixels=200000)


def test_qwen_ft_load_model_passes_pixel_bounds(
    tmp_path: Path, adapter_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    cfg = load_config(adapter_config)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights-v1")
    cfg.baselines.qwen_ft.adapter_uri = str(adapter_dir)
    cfg.baselines.qwen_ft.min_pixels = 50176
    cfg.baselines.qwen_ft.max_pixels = 200704

    adapter = QwenFTAdapter()

    captured_kwargs = {}

    def fake_load_model(model_id, device, allow_deferred, exception_cls, **kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock(), MagicMock()

    monkeypatch.setattr("aiforensics.baselines.qwen_vl.runtime.load_model", fake_load_model)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, package=None: MagicMock())

    import sys

    fake_peft = MagicMock()
    fake_peft.__spec__ = MagicMock()
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    adapter._load_model(cfg, device="cpu")

    assert captured_kwargs.get("min_pixels") == 50176
    assert captured_kwargs.get("max_pixels") == 200704
