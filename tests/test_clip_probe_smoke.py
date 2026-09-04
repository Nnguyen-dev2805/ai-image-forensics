import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from aiforensics.baselines.base import RunResult
from aiforensics.baselines.clip_probe.adapter import ClipProbeAdapter
from aiforensics.cli.main import main
from aiforensics.evaluation.metrics import discover_prediction_files
from aiforensics.schemas.predictions import load_predictions, validate_predictions

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "phase_ab_smoke.yaml"


def write_tmp_smoke_config(tmp_path: Path, **overrides: object) -> Path:
    (tmp_path / "pyproject.toml").touch()
    data = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))

    # Absolute paths for data so they resolve correctly
    data["paths"]["output_root"] = str(tmp_path / "outputs")
    data["paths"]["cache_root"] = str(tmp_path / "cache")
    data["paths"]["data_root"] = str(REPO_ROOT / data["paths"]["data_root"])
    data["paths"]["manifest_root"] = str(REPO_ROOT / data["paths"]["manifest_root"])

    data["datasets"]["tiny_genimage"]["train_manifest"] = str(
        REPO_ROOT / data["datasets"]["tiny_genimage"]["train_manifest"]
    )
    data["datasets"]["tiny_genimage"]["dev_manifest"] = str(
        REPO_ROOT / data["datasets"]["tiny_genimage"]["dev_manifest"]
    )
    data["datasets"]["genimage_unseen"]["manifest"] = str(
        REPO_ROOT / data["datasets"]["genimage_unseen"]["manifest"]
    )
    data["datasets"]["synthbuster"]["manifest"] = str(
        REPO_ROOT / data["datasets"]["synthbuster"]["manifest"]
    )

    for dotted_key, value in overrides.items():
        target = data
        parts = dotted_key.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value

    config_path = tmp_path / "phase_ab_smoke.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def test_run_result_can_represent_completed_run_with_paths(tmp_path):
    rr = RunResult(
        baseline="clip_probe",
        run_id="run1",
        status="completed",
        output_dir=tmp_path,
        prediction_path=tmp_path / "predictions.jsonl",
        log_path=tmp_path / "logs.txt",
        environment_path=tmp_path / "environment.json",
        status_path=tmp_path / "status.json",
    )
    assert rr.status == "completed"
    assert rr.prediction_path is not None


def test_clip_probe_adapter_name():
    assert ClipProbeAdapter.name == "clip_probe"


def test_smoke_embedding_extraction_deterministic(tmp_path):
    from aiforensics.baselines.clip_probe.adapter import _smoke_image_embedding

    img_path = REPO_ROOT / "tests" / "fixtures" / "smoke_data" / "real_0001.png"
    emb1 = _smoke_image_embedding(img_path)
    emb2 = _smoke_image_embedding(img_path)
    np.testing.assert_array_equal(emb1, emb2)


def test_smoke_embedding_extraction_changes_for_different_images():
    from aiforensics.baselines.clip_probe.adapter import _smoke_image_embedding

    img1 = REPO_ROOT / "tests" / "fixtures" / "smoke_data" / "real_0001.png"
    img2 = REPO_ROOT / "tests" / "fixtures" / "smoke_data" / "fake_0001.png"
    emb1 = _smoke_image_embedding(img1)
    emb2 = _smoke_image_embedding(img2)
    assert not np.array_equal(emb1, emb2)


def test_adapter_rejects_seed_none(tmp_path):
    from aiforensics.config.load import load_config

    config = load_config(SMOKE_CONFIG)
    adapter = ClipProbeAdapter()
    with pytest.raises(ValueError, match="Seed is required"):
        adapter.run(config=config, output_dir=tmp_path, run_id="x", seed=None)


def test_adapter_runs_on_smoke_config(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)
    exit_code = main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    assert exit_code == 0


def test_smoke_adapter_writes_predictions(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])

    out_dir = tmp_path / "outputs"
    run_dir = list(out_dir.glob("*_clip_probe_seed70"))[0]
    pred_path = run_dir / "predictions.jsonl"
    assert pred_path.exists()


def test_smoke_predictions_load_through_schema(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    preds = load_predictions(run_dir / "predictions.jsonl")
    assert len(preds) == 2


def test_smoke_predictions_pass_validation(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    preds = load_predictions(run_dir / "predictions.jsonl")
    val_res = validate_predictions(preds, require_mllm_fields=True)
    assert val_res.is_valid


def test_smoke_predictions_attributes(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    preds = load_predictions(run_dir / "predictions.jsonl")

    for p in preds:
        assert p.model_name == "clip_probe"
        assert p.run_id == run_dir.name
        assert p.parse_status == "not_applicable"
        assert 0.0 <= getattr(p, "score_fake", 0.0) <= 1.0


def test_smoke_run_writes_all_files(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]

    assert (run_dir / "status.json").exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"

    assert (run_dir / "environment.json").exists()
    assert (run_dir / "logs.txt").exists()
    assert (run_dir / "config.yaml").exists()


def test_cli_smoke_run_creates_exactly_one_run_dir(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    out_dirs = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))
    assert len(out_dirs) == 1


def test_discover_prediction_files_finds_smoke_prediction(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    out_root = tmp_path / "outputs"
    found = discover_prediction_files(out_root)
    assert len(found) == 1
    assert found[0].name == "predictions.jsonl"


def test_evaluate_prediction_file(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    out_root = tmp_path / "outputs"
    found = discover_prediction_files(out_root)[0]

    from aiforensics.evaluation.metrics import evaluate_prediction_file

    evaluate_prediction_file(found)
    assert (found.parent / "metrics.json").exists()
    assert (found.parent / "metrics_by_source.csv").exists()


def test_multiple_seeds(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path, baselines__clip_probe__seeds=[70, 71])
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    out_dirs = list((tmp_path / "outputs").glob("*_clip_probe_seed*"))
    assert len(out_dirs) == 2


def test_disabling_clip_probe(tmp_path):
    config_path = write_tmp_smoke_config(tmp_path, baselines__clip_probe__enabled=False)
    exit_code = main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    assert exit_code == 0
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe"))[0]
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "deferred"


def test_training_manifest_only_one_class(tmp_path):
    bad_train_csv = tmp_path / "bad_train.csv"
    bad_train_csv.write_text(
        "sample_id,path,label,source,split,checksum\nsmoke/train/real_0001,tests/fixtures/smoke_data/real_0001.png,real,smoke,train,b6d767d2f8ed5d21a44b0e5886680cb9\n"
    )
    config_path = write_tmp_smoke_config(
        tmp_path, datasets__tiny_genimage__train_manifest=str(bad_train_csv)
    )

    exit_code = main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    assert exit_code == 1
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed"


def test_disabled_tiny_genimage_fails_clip_probe_explicitly(tmp_path):
    """CLIP needs the tiny training split; disabling tiny must fail, not train anyway."""
    config_path = write_tmp_smoke_config(
        tmp_path,
        datasets__tiny_genimage__enabled=False,
        datasets__genimage_unseen__enabled=True,
    )

    exit_code = main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    assert exit_code == 1
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert "datasets.tiny_genimage.enabled is false" in status["reason"]
    assert not (run_dir / "predictions.jsonl").exists()


def test_smoke_path_does_not_require_open_clip_or_torch(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "open_clip", None)
    from aiforensics.baselines.clip_probe.adapter import ClipProbeAdapter

    assert ClipProbeAdapter.name == "clip_probe"


def test_regression_optional_dataset_enabled_false(tmp_path):
    # Optional dataset manifest exists but is disabled in config
    unseen_csv = tmp_path / "unseen.csv"
    unseen_csv.write_text(
        "sample_id,path,label,source,split,checksum\n"
        "smoke/unseen/fake_0001,tests/fixtures/smoke_data/fake_0001.png,"
        "fake,smoke,test,3f80c6be812bbccdfc7b0d778fb8c24c\n"
    )
    config_path = write_tmp_smoke_config(
        tmp_path,
        datasets__genimage_unseen__enabled=False,
        datasets__genimage_unseen__manifest=str(unseen_csv),
    )
    main(["run", "--baseline", "clip_probe", "--config", str(config_path)])

    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    preds = load_predictions(run_dir / "predictions.jsonl")
    # Smoke config dev has 2 records. Since unseen is disabled,
    # total predictions should be 2, not 3.
    assert len(preds) == 2


def test_regression_openclip_setup_failure_deferred(monkeypatch, tmp_path):
    config_path = write_tmp_smoke_config(tmp_path, baselines__clip_probe__model_family="openclip")
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "open_clip", None)

    exit_code = main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    assert exit_code == 0
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "deferred"


def test_regression_sample_embedding_failure_failed(monkeypatch, tmp_path):
    config_path = write_tmp_smoke_config(tmp_path, baselines__clip_probe__model_family="openclip")
    import sys
    import types

    import numpy as np

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: type(
        "context", (), {"__enter__": lambda self: None, "__exit__": lambda self, *args: None}
    )()

    class DummyTensor(np.ndarray):
        def to(self, *args, **kwargs):
            return self

    def _make_dummy(shape):
        arr = np.ones(shape, dtype=np.float32).view(DummyTensor)
        return arr

    torch.ones = _make_dummy
    torch.stack = lambda tensors: np.stack(tensors).view(DummyTensor)
    torch.Tensor = np.ndarray

    class DummyModel:
        def eval(self):
            pass

        def encode_image(self, batch):
            raise RuntimeError("Simulated inference failure")

    def _mock_create_model(*args, **kwargs):
        return DummyModel(), None, lambda img: torch.ones((3, 224, 224))

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        type("obj", (object,), {"create_model_and_transforms": _mock_create_model}),
    )

    import aiforensics.baselines.clip_probe.adapter

    def _mock_img_open(*args, **kwargs):
        class DummyImg:
            def convert(self, mode):
                return "dummy_image"

        return DummyImg()

    monkeypatch.setattr(aiforensics.baselines.clip_probe.adapter.Image, "open", _mock_img_open)

    exit_code = main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    assert exit_code == 1
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    import json

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed"


def test_regression_openclip_setup_failure_deferred_model_error(monkeypatch, tmp_path):
    config_path = write_tmp_smoke_config(tmp_path, baselines__clip_probe__model_family="openclip")
    import sys
    import types

    torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)

    def _mock_create_model_fail(*args, **kwargs):
        raise RuntimeError("Model weights not found")

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        type("obj", (object,), {"create_model_and_transforms": _mock_create_model_fail}),
    )

    exit_code = main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    assert exit_code == 0
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    import json

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "deferred"


def test_regression_prediction_validation_fail_no_artifact(monkeypatch, tmp_path):
    config_path = write_tmp_smoke_config(tmp_path)

    def _mock_validate_predictions(*args, **kwargs):
        from aiforensics.schemas.predictions import PredictionValidationResult

        return PredictionValidationResult(
            is_valid=False, errors=["simulated failure"], total_records=0
        )

    import aiforensics.baselines.clip_probe.adapter

    monkeypatch.setattr(
        aiforensics.baselines.clip_probe.adapter,
        "validate_predictions",
        _mock_validate_predictions,
    )

    exit_code = main(["run", "--baseline", "clip_probe", "--config", str(config_path)])
    assert exit_code == 1
    run_dir = list((tmp_path / "outputs").glob("*_clip_probe_seed70"))[0]
    assert not (run_dir / "predictions.jsonl").exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed"


def test_regression_cache_order_maintained(monkeypatch, tmp_path):
    # Simulate a scenario where first record is a cache miss and second is a hit.
    config_path = write_tmp_smoke_config(
        tmp_path,
        baselines__clip_probe__model_family="openclip",
        baselines__clip_probe__cache_embeddings=True,
        runtime__device="cpu",
    )

    import sys
    import types

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: type(
        "context", (), {"__enter__": lambda self: None, "__exit__": lambda self, *args: None}
    )()

    class DummyTensor(np.ndarray):
        def to(self, *args, **kwargs):
            return self

        def norm(self, *args, **kwargs):
            return np.ones((self.shape[0], 1)) * np.sqrt(512)

        def cpu(self, *args, **kwargs):
            return self

        def numpy(self, *args, **kwargs):
            return self

    def _make_dummy(shape):
        arr = np.ones(shape, dtype=np.float32).view(DummyTensor)
        return arr

    torch.ones = _make_dummy
    torch.stack = lambda tensors: np.stack(tensors).view(DummyTensor)
    torch.Tensor = np.ndarray

    class DummyModel:
        def eval(self):
            pass

        def encode_image(self, batch):
            return torch.ones((batch.shape[0], 512))

    def _mock_create_model(*args, **kwargs):
        return DummyModel(), None, lambda img: torch.ones((3, 224, 224))

    import aiforensics.baselines.clip_probe.adapter

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        type("obj", (object,), {"create_model_and_transforms": _mock_create_model}),
    )

    original_load = np.load
    original_exists = Path.exists

    original_cache_key = aiforensics.baselines.clip_probe.adapter.cache_key

    def _mock_cache_key(data):
        if data["sample_checksum"] == "b" * 64:
            return "dummy_hit_key"
        return original_cache_key(data)

    monkeypatch.setattr(aiforensics.baselines.clip_probe.adapter, "cache_key", _mock_cache_key)

    def _mock_exists(path_obj):
        if "dummy_hit_key" in str(path_obj):
            return True
        return original_exists(path_obj)

    def _mock_load(file_obj, *args, **kwargs):
        if "dummy_hit_key" in str(file_obj):
            return np.ones(512, dtype=np.float32) * 99.0
        return original_load(file_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", _mock_exists)
    monkeypatch.setattr(np, "load", _mock_load)

    # Mock Image.open
    def _mock_img_open(*args, **kwargs):
        class DummyImg:
            def convert(self, mode):
                return "dummy_image"

        return DummyImg()

    monkeypatch.setattr(aiforensics.baselines.clip_probe.adapter.Image, "open", _mock_img_open)

    # Mock np.save
    monkeypatch.setattr(np, "save", lambda *args, **kwargs: None)

    from aiforensics.config.load import load_config
    from aiforensics.data.manifest import ManifestRecord

    config = load_config(config_path)
    adapter = aiforensics.baselines.clip_probe.adapter.ClipProbeAdapter()

    records = [
        ManifestRecord(
            sample_id="miss_1",
            path=Path("miss1.png"),
            label="real",
            source="test",
            split="dev",
            checksum="a" * 64,
        ),
        ManifestRecord(
            sample_id="hit_1",
            path=Path("hit1.png"),
            label="real",
            source="test",
            split="dev",
            checksum="b" * 64,
        ),
        ManifestRecord(
            sample_id="miss_2",
            path=Path("miss2.png"),
            label="real",
            source="test",
            split="dev",
            checksum="c" * 64,
        ),
    ]

    embeddings = adapter._get_openclip_embeddings(records, config)

    assert embeddings.shape == (3, 512)
    assert np.all(embeddings[1] == 99.0)
    assert np.allclose(embeddings[0], 1.0 / np.sqrt(512))
    assert np.allclose(embeddings[2], 1.0 / np.sqrt(512))
