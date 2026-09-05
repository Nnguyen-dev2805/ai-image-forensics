import hashlib
import json
from pathlib import Path

import pytest

from aiforensics.config.load import load_config
from aiforensics.data.manifest import ManifestRecord

REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_FAKE = '{"label": "fake", "confidence": 0.75, "evidence": "synthetic texture"}'
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _record(tmp_path: Path, sample_id: str = "1") -> ManifestRecord:
    image_path = tmp_path / f"{sample_id}.jpg"
    image_path.write_bytes(b"")
    return ManifestRecord(
        sample_id=sample_id,
        path=image_path,
        label="real",
        source="smoke",
        split="dev",
        checksum=EMPTY_SHA256,
        dataset="smoke",
    )


def test_quick_config_enables_qwen_vertex_provider() -> None:
    config = load_config("configs/phase_ab_vertex_quick.yaml")

    assert config.baselines.clip_probe.seeds == [70]
    assert config.datasets.tiny_genimage.max_images <= 80
    assert config.datasets.genimage_unseen.max_images <= 20
    assert config.baselines.qwen_vl.provider == "vertex_openai"
    assert config.baselines.assisted_qwen.provider == "vertex_openai"
    assert config.baselines.qwen_vl.vertex_endpoint_domain.endswith(".prediction.vertexai.goog")
    assert config.baselines.qwen_vl.vertex_model_id == "qwen2_5-vl-7b-instruct-1788570383931"


def test_vertex_extra_includes_google_auth_requests_transport_dependency() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    vertex_section = pyproject_text.split("vertex = [", maxsplit=1)[1].split("]", maxsplit=1)[0]

    assert '"openai"' in vertex_section
    assert '"google-auth"' in vertex_section
    assert '"requests"' in vertex_section


def test_vertex_base_url_uses_dedicated_endpoint_domain() -> None:
    from aiforensics.baselines.qwen_vl.vertex_openai import build_vertex_base_url

    assert build_vertex_base_url(
        project_id="579187260419",
        location="asia-southeast1",
        endpoint_id="mg-endpoint-ccc259a2-c268-4ca6-8713-3bcdaaaf5909",
        endpoint_domain=(
            "mg-endpoint-ccc259a2-c268-4ca6-8713-3bcdaaaf5909."
            "asia-southeast1-635507464424.prediction.vertexai.goog"
        ),
    ) == (
        "https://mg-endpoint-ccc259a2-c268-4ca6-8713-3bcdaaaf5909."
        "asia-southeast1-635507464424.prediction.vertexai.goog/v1/projects/"
        "579187260419/locations/asia-southeast1/endpoints/"
        "mg-endpoint-ccc259a2-c268-4ca6-8713-3bcdaaaf5909"
    )


def test_vertex_base_url_rejects_generic_aiplatform_domain() -> None:
    from aiforensics.baselines.qwen_vl.vertex_openai import build_vertex_base_url

    with pytest.raises(ValueError, match="Dedicated Endpoint"):
        build_vertex_base_url(
            project_id="579187260419",
            location="asia-southeast1",
            endpoint_id="mg-endpoint-ccc259a2-c268-4ca6-8713-3bcdaaaf5909",
            endpoint_domain="https://aiplatform.googleapis.com",
        )


def test_qwen_vertex_inference_writes_predictions_without_local_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.schemas.predictions import load_predictions

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.cache_root = tmp_path / "cache"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.provider = "vertex_openai"
    config.baselines.qwen_vl.cache_outputs = False
    config.baselines.qwen_vl.vertex_model_id = "vertex-model"

    adapter = QwenVLAdapter()
    monkeypatch.setattr(adapter, "_load_manifests", lambda cfg: [_record(tmp_path)])
    monkeypatch.setattr(
        adapter,
        "_load_model",
        lambda *args, **kwargs: pytest.fail("Vertex provider must not load local model"),
    )
    monkeypatch.setattr(adapter, "_create_vertex_client", lambda cfg: object())
    monkeypatch.setattr(adapter, "_generate_one_image_vertex", lambda *args, **kwargs: RAW_FAKE)

    result = adapter.run(config=config, output_dir=tmp_path / "run", run_id="run")

    assert result.status == "completed"
    predictions = load_predictions(tmp_path / "run" / "predictions.jsonl")
    assert len(predictions) == 1
    assert predictions[0].model_name == "qwen_vl"
    assert predictions[0].label_pred == "fake"
    assert predictions[0].score_fake == 0.75
    assert predictions[0].raw_output == RAW_FAKE


def test_assisted_qwen_vertex_inference_writes_predictions_without_local_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aiforensics.baselines.assisted_qwen.adapter import AssistedInput, AssistedQwenAdapter
    from aiforensics.schemas.predictions import load_predictions

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.cache_root = tmp_path / "cache"
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.provider = "vertex_openai"
    config.baselines.assisted_qwen.cache_outputs = False
    config.baselines.assisted_qwen.vertex_model_id = "vertex-model"

    adapter = AssistedQwenAdapter()
    monkeypatch.setattr(adapter, "_load_manifests", lambda cfg: [_record(tmp_path)])
    monkeypatch.setattr(
        adapter,
        "_discover_assistant_inputs",
        lambda cfg: {
            "1": AssistedInput(
                sample_id="1",
                classifier_pred="fake",
                fake_probability=0.8,
                source_prediction_files=(),
            )
        },
    )
    monkeypatch.setattr(
        "aiforensics.baselines.assisted_qwen.adapter.load_model",
        lambda *args, **kwargs: pytest.fail("Vertex provider must not load local model"),
    )
    monkeypatch.setattr(adapter, "_create_vertex_client", lambda cfg: object())
    monkeypatch.setattr(adapter, "_generate_one_image_vertex", lambda *args, **kwargs: RAW_FAKE)

    result = adapter.run(config=config, output_dir=tmp_path / "run", run_id="run")

    assert result.status == "completed"
    predictions = load_predictions(tmp_path / "run" / "predictions.jsonl")
    assert len(predictions) == 1
    assert predictions[0].model_name == "assisted_qwen"
    assert predictions[0].label_pred == "fake"


def test_vertex_cache_key_changes_with_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.baselines.qwen_vl.cache import write_qwen_cache
    from aiforensics.schemas.predictions import load_predictions

    record = _record(tmp_path)
    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.cache_root = tmp_path / "cache"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.provider = "vertex_openai"
    config.baselines.qwen_vl.cache_outputs = True
    config.baselines.qwen_vl.vertex_model_id = "vertex-model"
    config.baselines.qwen_vl.vertex_endpoint_domain = "endpoint-a.prediction.vertexai.goog"

    adapter = QwenVLAdapter()
    monkeypatch.setattr(adapter, "_load_manifests", lambda cfg: [record])
    monkeypatch.setattr(adapter, "_create_vertex_client", lambda cfg: object())
    monkeypatch.setattr(adapter, "_generate_one_image_vertex", lambda *args, **kwargs: RAW_FAKE)

    result_a = adapter.run(config=config, output_dir=tmp_path / "run-a", run_id="run-a")
    assert result_a.status == "completed"
    cache_files_a = sorted((config.paths.cache_root / "qwen_vl" / "raw_outputs").glob("*.json"))
    assert len(cache_files_a) == 1

    write_qwen_cache(
        cache_files_a[0],
        record.sample_id,
        '{"label": "real", "confidence": 0.99, "evidence": "wrong endpoint cache"}',
    )
    config.baselines.qwen_vl.vertex_endpoint_domain = "endpoint-b.prediction.vertexai.goog"

    result_b = adapter.run(config=config, output_dir=tmp_path / "run-b", run_id="run-b")

    assert result_b.status == "completed"
    predictions = load_predictions(tmp_path / "run-b" / "predictions.jsonl")
    assert predictions[0].label_pred == "fake"
    cache_payloads = [
        json.loads(path.read_text(encoding="utf-8"))["raw_output"]
        for path in sorted((config.paths.cache_root / "qwen_vl" / "raw_outputs").glob("*.json"))
    ]
    assert RAW_FAKE in cache_payloads
    assert len(cache_payloads) == 2
