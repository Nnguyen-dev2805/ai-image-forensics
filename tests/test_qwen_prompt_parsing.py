import pytest

from aiforensics.baselines.qwen_vl.parsing import parse_qwen_output


def test_valid_compact_json():
    # 6. valid compact JSON parses with parse_status="parsed".
    # 8. label=fake, confidence=0.9 maps to score_fake=0.9.
    res = parse_qwen_output('{"label":"fake","confidence":0.9,"evidence":"x"}')
    assert res.parse_status == "parsed"
    assert res.label_pred == "fake"
    assert res.score_fake == 0.9
    assert res.explanation == "x"


def test_valid_pretty_json():
    # 7. valid pretty JSON parses with parse_status="parsed".
    # 9. label=real, confidence=0.9 maps to score_fake=0.1.
    raw = """{
  "label": "real",
  "confidence": 0.9,
  "evidence": "looks real"
}"""
    res = parse_qwen_output(raw)
    assert res.parse_status == "parsed"
    assert res.label_pred == "real"
    assert res.score_fake == pytest.approx(0.1)
    assert res.explanation == "looks real"


def test_fenced_json_recovery():
    # 10. fenced JSON is recovered with parse_status="recovered".
    raw = """Here is your output:
```json
{"label":"fake","confidence":0.8,"evidence":"y"}
```"""
    res = parse_qwen_output(raw)
    assert res.parse_status == "recovered"
    assert res.label_pred == "fake"
    assert res.score_fake == 0.8
    assert res.explanation == "y"


def test_prose_json_recovery():
    # 11. one JSON object surrounded by prose is recovered.
    raw = """I think it is fake.
{"label": "fake", "confidence": 0.99, "evidence": "z"}
That is my final answer."""
    res = parse_qwen_output(raw)
    assert res.parse_status == "recovered"
    assert res.label_pred == "fake"
    assert res.score_fake == 0.99
    assert res.explanation == "z"


def test_label_whitespace_case_recovery():
    # 12. label case/whitespace-only normalization is recovered.
    raw = '{"label":" Fake ","confidence":0.9,"evidence":"x"}'
    res = parse_qwen_output(raw)
    assert res.parse_status == "recovered"
    assert res.label_pred == "fake"
    assert res.score_fake == 0.9
    assert res.explanation == "x"


def test_malformed_json_fails():
    # 13. malformed JSON returns unknown, None, empty explanation, and parse_status="failed".
    raw = '{"label":"fake","confidence":0.9,"evidence":"x"'
    res = parse_qwen_output(raw)
    assert res.parse_status == "failed"
    assert res.label_pred == "unknown"
    assert res.score_fake is None
    assert res.explanation == ""


def test_missing_label_fails():
    # 14. missing label fails parsing.
    raw = '{"confidence":0.9,"evidence":"x"}'
    res = parse_qwen_output(raw)
    assert res.parse_status == "failed"


def test_missing_confidence_fails():
    # 15. missing confidence fails parsing.
    raw = '{"label":"fake","evidence":"x"}'
    res = parse_qwen_output(raw)
    assert res.parse_status == "failed"


def test_missing_evidence_fails():
    # 16. missing/empty evidence fails parsing.
    raw = '{"label":"fake","confidence":0.9}'
    res = parse_qwen_output(raw)
    assert res.parse_status == "failed"

    raw2 = '{"label":"fake","confidence":0.9,"evidence":"  "}'
    res2 = parse_qwen_output(raw2)
    assert res2.parse_status == "failed"


def test_unknown_label_fails():
    # 17. unknown label such as synthetic fails parsing.
    raw = '{"label":"synthetic","confidence":0.9,"evidence":"x"}'
    res = parse_qwen_output(raw)
    assert res.parse_status == "failed"


def test_confidence_out_of_bounds_fails():
    # 18. confidence below 0.0 fails parsing.
    # 19. confidence above 1.0 fails parsing.
    res1 = parse_qwen_output('{"label":"fake","confidence":-0.1,"evidence":"x"}')
    assert res1.parse_status == "failed"
    res2 = parse_qwen_output('{"label":"fake","confidence":1.1,"evidence":"x"}')
    assert res2.parse_status == "failed"


def test_boolean_confidence_fails():
    # 20. boolean confidence fails parsing.
    res = parse_qwen_output('{"label":"fake","confidence":true,"evidence":"x"}')
    assert res.parse_status == "failed"


def test_multiple_competing_json_fails():
    # 21. multiple competing JSON objects fail recovery rather than selecting one silently.
    raw = """
{"label":"fake","confidence":0.9,"evidence":"x"}
Some text
{"label":"real","confidence":0.9,"evidence":"y"}
"""
    res = parse_qwen_output(raw)
    assert res.parse_status == "failed"


def test_parser_no_imports():
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")

    code = (
        "import sys\n"
        "from aiforensics.baselines.qwen_vl.parsing import parse_qwen_output\n"
        "if 'torch' in sys.modules:\n"
        "    sys.exit(1)\n"
    )

    env = {"PYTHONPATH": src_path}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(repo_root)
    )
    assert result.returncode == 0, f"Parser imported torch! {result.stderr}"


def test_qwen_disabled_deferred(tmp_path):
    # 23. Qwen disabled in a temporary config creates a deferred run artifact and exits 0.

    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.baselines.qwen_vl.enabled = False
    config.baselines.qwen_vl.allow_deferred = True

    (tmp_path / "y").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,d,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )
    (tmp_path / "1.jpg").write_text("imagebytes")

    adapter = QwenVLAdapter()

    (tmp_path / "run_dir").mkdir(exist_ok=True, parents=True)

    res = adapter.run(config=config, output_dir=tmp_path / "run_dir", run_id="run_dir")

    assert res.status == "deferred"
    assert res.reason == "qwen_vl is disabled in config"


def test_missing_deps_failed(tmp_path):
    # 25. the same setup problem produces failed when allow_deferred=false.
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "x"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen"
    config.datasets.synthbuster.manifest = tmp_path / "z"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.allow_deferred = False
    config.baselines.qwen_vl.cache_outputs = False

    (tmp_path / "y").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,d,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )
    (tmp_path / "1.jpg").write_text("imagebytes")

    adapter = QwenVLAdapter()

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("torch", "transformers", "qwen_vl_utils", "accelerate"):
            raise ImportError(f"No module named {name}")
        return original_import(name, *args, **kwargs)

    try:
        builtins.__import__ = mock_import
        (tmp_path / "run_dir").mkdir(exist_ok=True, parents=True)

        res = adapter.run(config=config, output_dir=tmp_path / "run_dir", run_id="run_dir")
    finally:
        builtins.__import__ = original_import

    assert res.status == "failed"


def test_model_setup_deferred(monkeypatch, tmp_path):
    # 26. model/processor setup failure is deferred when allowed.
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "x"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen"
    config.datasets.synthbuster.manifest = tmp_path / "z"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.allow_deferred = True
    config.baselines.qwen_vl.model_id = "M"
    config.baselines.qwen_vl.cache_outputs = False

    (tmp_path / "y").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,d,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )
    (tmp_path / "1.jpg").write_text("imagebytes")

    adapter = QwenVLAdapter()

    # Mock find_spec to return something so dependencies are "found"
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name, *args, **kwargs):
        if name in ("torch", "transformers", "qwen_vl_utils", "accelerate"):
            return "mocked_spec"
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    # Mock _get_qwen_device
    monkeypatch.setattr(adapter, "_get_qwen_device", lambda c: "cpu")

    # Mock _load_model to raise Exception
    def mock_load_model(cfg, device):
        from aiforensics.baselines.qwen_vl.adapter import BaselineDeferredError

        raise BaselineDeferredError("Fake load error")

    monkeypatch.setattr(adapter, "_load_model", mock_load_model)

    (tmp_path / "run_dir").mkdir(exist_ok=True, parents=True)

    res = adapter.run(config=config, output_dir=tmp_path / "run_dir", run_id="run_dir")
    assert res.status == "deferred"


def test_inference_failure_failed(monkeypatch, tmp_path):
    # 27. per-sample generation failure is failed, not deferred.
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "x"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen"
    config.datasets.synthbuster.manifest = tmp_path / "z"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.allow_deferred = True
    config.baselines.qwen_vl.model_id = "M"
    config.baselines.qwen_vl.cache_outputs = False

    (tmp_path / "y").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,d,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )
    (tmp_path / "1.jpg").write_text("imagebytes")

    adapter = QwenVLAdapter()

    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *args: (
            "mocked" if n in ("torch", "transformers", "qwen_vl_utils", "accelerate") else None
        ),
    )
    monkeypatch.setattr(adapter, "_get_qwen_device", lambda c: "cpu")
    monkeypatch.setattr(adapter, "_load_model", lambda c, d: ("model", "processor"))

    def mock_gen(*args, **kwargs):
        raise RuntimeError("Generation error")

    monkeypatch.setattr(adapter, "_generate_one_image", mock_gen)

    (tmp_path / "run_dir").mkdir(exist_ok=True, parents=True)

    res = adapter.run(config=config, output_dir=tmp_path / "run_dir", run_id="run_dir")
    assert res.status == "failed"


def test_parse_failure_produces_unknown(monkeypatch, tmp_path):
    # 28. a parse failure still produces one valid PredictionRecord with label_pred="unknown".
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "x"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen"
    config.datasets.synthbuster.manifest = tmp_path / "z"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.allow_deferred = True
    config.baselines.qwen_vl.model_id = "M"
    config.baselines.qwen_vl.cache_outputs = False

    (tmp_path / "y").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,d,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )
    (tmp_path / "1.jpg").write_text("imagebytes")

    adapter = QwenVLAdapter()

    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *args: (
            "mocked" if n in ("torch", "transformers", "qwen_vl_utils", "accelerate") else None
        ),
    )
    monkeypatch.setattr(adapter, "_get_qwen_device", lambda c: "cpu")
    monkeypatch.setattr(adapter, "_load_model", lambda c, d: ("model", "processor"))
    monkeypatch.setattr(
        adapter, "_generate_one_image", lambda *args, **kwargs: "Gibberish response"
    )

    (tmp_path / "run_dir").mkdir(exist_ok=True, parents=True)

    res = adapter.run(config=config, output_dir=tmp_path / "run_dir", run_id="run_dir")
    assert res.status == "completed"

    import json

    preds = [
        json.loads(line)
        for line in (tmp_path / "run_dir" / "predictions.jsonl").read_text().strip().split("\n")
    ]
    assert len(preds) == 1
    assert preds[0]["label_pred"] == "unknown"
    assert preds[0]["score_fake"] is None
    assert preds[0]["parse_status"] == "failed"


def test_prediction_validation_failure(monkeypatch, tmp_path):
    # 29. prediction validation failure does not leave predictions.jsonl.
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "x"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen"
    config.datasets.synthbuster.manifest = tmp_path / "z"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.allow_deferred = True
    config.baselines.qwen_vl.model_id = "M"
    config.baselines.qwen_vl.cache_outputs = False

    (tmp_path / "y").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,d,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )
    (tmp_path / "1.jpg").write_text("imagebytes")

    adapter = QwenVLAdapter()

    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *args: (
            "mocked" if n in ("torch", "transformers", "qwen_vl_utils", "accelerate") else None
        ),
    )
    monkeypatch.setattr(adapter, "_get_qwen_device", lambda c: "cpu")
    monkeypatch.setattr(adapter, "_load_model", lambda c, d: ("model", "processor"))

    # Return duplicate prediction
    def mock_run_inference(records, cfg, r_id, counts):
        from pathlib import Path

        from aiforensics.schemas.predictions import PredictionRecord

        p = PredictionRecord(
            sample_id="1",
            label_true="fake",
            label_pred="unknown",
            score_fake=None,
            model_name="qwen_vl",
            source="s",
            run_id="r",
            dataset="d",
            split="dev",
            path=Path("1.jpg"),
            checksum=None,
            prompt_id="p",
            raw_output="raw",
            explanation="",
            parse_status="failed",
        )
        return [p, p]  # duplicate to trigger validation error

    monkeypatch.setattr(adapter, "_run_inference", mock_run_inference)

    (tmp_path / "run_dir").mkdir(exist_ok=True, parents=True)

    res = adapter.run(config=config, output_dir=tmp_path / "run_dir", run_id="run_dir")
    assert res.status == "failed"
    assert "Prediction validation failed" in res.reason
    assert not (tmp_path / "run_dir" / "predictions.jsonl").exists()


def test_disabled_optional_datasets(tmp_path):
    # 30. disabled optional datasets are ignored even if their manifests exist.
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "x"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen"
    config.datasets.synthbuster.manifest = tmp_path / "z"
    config.baselines.qwen_vl.enabled = False
    config.baselines.qwen_vl.allow_deferred = True
    config.baselines.qwen_vl.model_id = "M"
    config.baselines.qwen_vl.cache_outputs = False

    (tmp_path / "y").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,d,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )
    (tmp_path / "unseen").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n2,2.jpg,fake,dev,s,unseen,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )

    adapter = QwenVLAdapter()
    records = adapter._load_manifests(config)
    assert len(records) == 1
    assert records[0].sample_id == "1"


def test_cache_hit_bypasses_model(monkeypatch, tmp_path):
    # 31. raw-output cache hit bypasses model generation and still runs the parser.
    # 36. model/processor are loaded once per run in mocked inference.
    # 37. raw model text is preserved in the prediction record.
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "x"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen"
    config.datasets.synthbuster.manifest = tmp_path / "z"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.allow_deferred = True
    config.baselines.qwen_vl.model_id = "M"
    config.baselines.qwen_vl.cache_outputs = True

    (tmp_path / "y").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,d,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )
    (tmp_path / "1.jpg").write_text("imagebytes")

    # Pre-populate cache
    adapter = QwenVLAdapter()
    import hashlib

    from aiforensics.cache.keys import cache_key

    csum = hashlib.sha256(b"imagebytes").hexdigest()
    key = cache_key(
        {
            "baseline": "qwen_vl",
            "sample_checksum": csum,
            "model_id": "M",
            "prompt_id": "qwen_json_v1",
            "dtype": "bfloat16",
            "temperature": "0.0",
            "max_new_tokens": "128",
            "output_cache_version": "qwen_vl_raw_v3",
        }
    )

    cache_dir = tmp_path / "cache" / "qwen_vl" / "raw_outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        '{"raw_output": "{\\"label\\": \\"fake\\", \\"confidence\\": 0.9, '
        '\\"evidence\\": \\"cached\\"}"}'
    )

    # Track model load
    loaded = False

    def mock_load_model(*args):
        nonlocal loaded
        loaded = True
        return "model", "processor"

    monkeypatch.setattr(adapter, "_load_model", mock_load_model)

    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *args: (
            "mocked" if n in ("torch", "transformers", "qwen_vl_utils", "accelerate") else None
        ),
    )

    (tmp_path / "run_dir").mkdir(exist_ok=True, parents=True)

    res = adapter.run(config=config, output_dir=tmp_path / "run_dir", run_id="run_dir")
    assert res.status == "completed"
    assert not loaded  # Model should not have been loaded since cache was hit

    import json

    preds = [
        json.loads(line)
        for line in (tmp_path / "run_dir" / "predictions.jsonl").read_text().strip().split("\n")
    ]
    assert preds[0]["explanation"] == "cached"
    assert preds[0]["raw_output"] == ('{"label": "fake", "confidence": 0.9, "evidence": "cached"}')


def test_cache_key_changes(tmp_path):
    # 32. cache key changes when image checksum changes.
    # 33. cache key changes when prompt_id changes.
    # 34. cache key changes when model_id, temperature, or max_new_tokens changes.
    from aiforensics.config.load import load_config
    from aiforensics.data.manifest import ManifestRecord

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "x"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen"
    config.datasets.synthbuster.manifest = tmp_path / "z"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.allow_deferred = True
    config.baselines.qwen_vl.model_id = "M"
    config.baselines.qwen_vl.cache_outputs = True

    (tmp_path / "1.jpg").write_text("1")
    (tmp_path / "2.jpg").write_text("2")

    import hashlib

    def csum(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    r1 = ManifestRecord(
        sample_id="1",
        label="fake",
        split="dev",
        source="s",
        dataset="d",
        path=tmp_path / "1.jpg",
        checksum=csum(tmp_path / "1.jpg"),
    )
    r2 = ManifestRecord(
        sample_id="2",
        label="fake",
        split="dev",
        source="s",
        dataset="d",
        path=tmp_path / "2.jpg",
        checksum=csum(tmp_path / "2.jpg"),
    )

    from aiforensics.cache.keys import cache_key

    def get_key(record, cfg):
        return cache_key(
            {
                "baseline": "qwen_vl",
                "sample_checksum": record.checksum,
                "model_id": cfg.baselines.qwen_vl.model_id,
                "prompt_id": cfg.baselines.qwen_vl.prompt_id,
                "dtype": cfg.baselines.qwen_vl.dtype,
                "temperature": str(cfg.baselines.qwen_vl.temperature),
                "max_new_tokens": str(cfg.baselines.qwen_vl.max_new_tokens),
                "output_cache_version": "qwen_vl_raw_v3",
            }
        )

    k1 = get_key(r1, config)
    k2 = get_key(r2, config)
    assert k1 != k2  # Checksum change

    c2 = config.model_copy(deep=True)
    c2.baselines.qwen_vl.prompt_id = "other"
    assert k1 != get_key(r1, c2)

    c3 = config.model_copy(deep=True)
    c3.baselines.qwen_vl.model_id = "other"
    assert k1 != get_key(r1, c3)

    c4 = config.model_copy(deep=True)
    c4.baselines.qwen_vl.temperature = 1.0
    assert k1 != get_key(r1, c4)

    c5 = config.model_copy(deep=True)
    c5.baselines.qwen_vl.max_new_tokens = 999
    assert k1 != get_key(r1, c5)


def test_corrupt_cache_recomputed(monkeypatch, tmp_path):
    # 35. corrupt cache entry is recomputed.
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.train_manifest = tmp_path / "x"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "y"
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen"
    config.datasets.synthbuster.manifest = tmp_path / "z"
    config.baselines.qwen_vl.enabled = True
    config.baselines.qwen_vl.allow_deferred = True
    config.baselines.qwen_vl.model_id = "M"
    config.baselines.qwen_vl.cache_outputs = True

    (tmp_path / "y").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,d,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )
    (tmp_path / "1.jpg").write_text("imagebytes")

    adapter = QwenVLAdapter()
    import hashlib

    from aiforensics.cache.keys import cache_key

    csum = hashlib.sha256(b"imagebytes").hexdigest()
    key = cache_key(
        {
            "baseline": "qwen_vl",
            "sample_checksum": csum,
            "model_id": "M",
            "prompt_id": "qwen_json_v1",
            "dtype": "bfloat16",
            "temperature": "0.0",
            "max_new_tokens": "128",
            "output_cache_version": "qwen_vl_raw_v3",
        }
    )

    cache_dir = tmp_path / "cache" / "qwen_vl" / "raw_outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    cache_file.touch()

    # Make file unreadable
    import os

    os.chmod(cache_file, 0o000)

    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *args: (
            "mocked" if n in ("torch", "transformers", "qwen_vl_utils", "accelerate") else None
        ),
    )
    monkeypatch.setattr(adapter, "_get_qwen_device", lambda c: "cpu")
    monkeypatch.setattr(adapter, "_load_model", lambda c, d: ("model", "processor"))

    def mock_gen(*args, **kwargs):
        return '{"label":"fake","confidence":0.9,"evidence":"x"}'

    monkeypatch.setattr(adapter, "_generate_one_image", mock_gen)

    (tmp_path / "run_dir").mkdir(exist_ok=True, parents=True)

    res = adapter.run(config=config, output_dir=tmp_path / "run_dir", run_id="run_dir")
    assert res.status == "completed"

    os.chmod(cache_file, 0o644)
    import json

    assert (
        json.loads(cache_file.read_text())["raw_output"]
        == '{"label":"fake","confidence":0.9,"evidence":"x"}'
    )


def test_multiple_fenced_json_fails():
    from aiforensics.baselines.qwen_vl.parsing import parse_qwen_output

    raw = """Here are two outputs:
```json
{"label": "real", "confidence": 0.9, "evidence": "a"}
```
And the other:
```
{"label": "fake", "confidence": 0.8, "evidence": "b"}
```
"""
    res = parse_qwen_output(raw)
    assert res.parse_status == "failed"
    assert res.label_pred == "unknown"


def test_json_with_braces_in_string():
    from aiforensics.baselines.qwen_vl.parsing import parse_qwen_output

    raw = """This is a JSON:
{"label": "real", "confidence": 0.9, "evidence": "braces { and }"}
"""
    res = parse_qwen_output(raw)
    assert res.parse_status == "recovered"
    assert res.label_pred == "real"


def test_adapter_name():
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter

    adapter = QwenVLAdapter()
    assert adapter.name == "qwen_vl"


def test_deterministic_prompt():
    from aiforensics.baselines.qwen_vl.prompt import get_prompt

    p1 = get_prompt("qwen_json_v1")
    p2 = get_prompt("qwen_json_v1")
    assert p1 == p2
    assert "JSON" in p1


def test_unsupported_prompt():
    import pytest

    from aiforensics.baselines.qwen_vl.prompt import get_prompt

    with pytest.raises(ValueError):
        get_prompt("unsupported_prompt_id")


def test_missing_deps_allow_deferred(tmp_path, monkeypatch):
    import importlib.util

    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    c = load_config("configs/phase_ab_smoke.yaml")
    c.paths.output_root = tmp_path
    c.paths.cache_root = tmp_path / "cache"
    c.paths.data_root = tmp_path
    c.baselines.qwen_vl.enabled = True
    c.baselines.qwen_vl.allow_deferred = True

    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name, *args, **kwargs):
        if name == "torch":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    adapter = QwenVLAdapter()

    # Needs valid manifest to reach model loading
    c.datasets.tiny_genimage.train_manifest = tmp_path / "manifest.csv"
    c.datasets.tiny_genimage.dev_manifest = tmp_path / "manifest_dev.csv"
    c.datasets.tiny_genimage.dev_manifest.write_text(
        "sample_id,label,dataset,split,source,path,checksum\nid3,real,tiny,dev,s,img.png,b29814cf5792e684cd75d6a7fce7a67a11887e312f87ca2ac2496d81f365ff72\nid4,fake,tiny,dev,s,img.png,b29814cf5792e684cd75d6a7fce7a67a11887e312f87ca2ac2496d81f365ff72"
    )
    c.datasets.genimage_unseen.enabled = False
    c.datasets.synthbuster.enabled = False
    c.datasets.tiny_genimage.train_manifest.write_text(
        "sample_id,label,dataset,split,source,path,checksum\nid1,real,tiny,train,s,img.png,b29814cf5792e684cd75d6a7fce7a67a11887e312f87ca2ac2496d81f365ff72\nid2,fake,tiny,train,s,img.png,b29814cf5792e684cd75d6a7fce7a67a11887e312f87ca2ac2496d81f365ff72"
    )
    (tmp_path / "img.png").write_bytes(b"img")

    res = adapter.run(config=c, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "deferred"


def test_missing_deps_not_allowed(tmp_path, monkeypatch):
    import importlib.util

    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    c = load_config("configs/phase_ab_smoke.yaml")
    c.paths.output_root = tmp_path
    c.paths.cache_root = tmp_path / "cache"
    c.paths.data_root = tmp_path
    c.baselines.qwen_vl.enabled = True
    c.baselines.qwen_vl.allow_deferred = False

    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name, *args, **kwargs):
        if name == "torch":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    adapter = QwenVLAdapter()

    c.datasets.tiny_genimage.train_manifest = tmp_path / "manifest.csv"
    c.datasets.tiny_genimage.dev_manifest = tmp_path / "manifest_dev.csv"
    c.datasets.tiny_genimage.dev_manifest.write_text(
        "sample_id,label,dataset,split,source,path,checksum\nid3,real,tiny,dev,s,img.png,b29814cf5792e684cd75d6a7fce7a67a11887e312f87ca2ac2496d81f365ff72\nid4,fake,tiny,dev,s,img.png,b29814cf5792e684cd75d6a7fce7a67a11887e312f87ca2ac2496d81f365ff72"
    )
    c.datasets.genimage_unseen.enabled = False
    c.datasets.synthbuster.enabled = False
    c.datasets.tiny_genimage.train_manifest.write_text(
        "sample_id,label,dataset,split,source,path,checksum\nid1,real,tiny,train,s,img.png,b29814cf5792e684cd75d6a7fce7a67a11887e312f87ca2ac2496d81f365ff72\nid2,fake,tiny,train,s,img.png,b29814cf5792e684cd75d6a7fce7a67a11887e312f87ca2ac2496d81f365ff72"
    )
    (tmp_path / "img.png").write_bytes(b"img")

    res = adapter.run(config=c, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "failed"


def test_model_loaded_once_even_with_multiple_items(tmp_path, monkeypatch):
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    c = load_config("configs/phase_ab_smoke.yaml")
    c.paths.output_root = tmp_path
    c.paths.cache_root = tmp_path / "cache"
    c.paths.data_root = tmp_path
    c.baselines.qwen_vl.enabled = True

    c.datasets.tiny_genimage.train_manifest = tmp_path / "manifest.csv"
    c.datasets.tiny_genimage.dev_manifest = tmp_path / "dev.csv"
    c.datasets.genimage_unseen.manifest = tmp_path / "unseen.csv"
    c.datasets.synthbuster.manifest = tmp_path / "synth.csv"
    c.datasets.genimage_unseen.enabled = False
    c.datasets.synthbuster.enabled = False
    c.datasets.tiny_genimage.train_manifest.write_text(
        "sample_id,label,dataset,split,source,path,checksum\nid1,real,t,train,s,img1.png,d7bdd545f09d8a73c2b990337c8211d708a04ccd9748627685e4fc79cc038039\nid2,fake,t,train,s,img2.png,6987740fb624e3e9943ec5d9ac5519b72cea1b35fb4bde5719df3923a36c08f7"
    )
    c.datasets.tiny_genimage.dev_manifest.write_text(
        "sample_id,label,dataset,split,source,path,checksum\nid3,real,t,dev,s,img1.png,d7bdd545f09d8a73c2b990337c8211d708a04ccd9748627685e4fc79cc038039\nid4,fake,t,dev,s,img2.png,6987740fb624e3e9943ec5d9ac5519b72cea1b35fb4bde5719df3923a36c08f7"
    )
    (tmp_path / "img1.png").write_bytes(b"img1")
    (tmp_path / "img2.png").write_bytes(b"img2")

    adapter = QwenVLAdapter()

    load_calls = 0
    generate_calls = 0

    def mock_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return "mock_model", "mock_processor"

    def mock_generate(*args, **kwargs):
        nonlocal generate_calls
        generate_calls += 1
        return '{"label": "real", "confidence": 0.9, "evidence": "x"}'

    monkeypatch.setattr(adapter, "_load_model", mock_load)
    monkeypatch.setattr(adapter, "_generate_one_image", mock_generate)

    # Also bypass imports and device check
    monkeypatch.setattr(adapter, "_get_qwen_device", lambda x: "cpu")
    import importlib.util

    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *a, **k: (
            "mock"
            if n in ("torch", "transformers", "qwen_vl_utils", "accelerate")
            else original_find_spec(n, *a, **k)
        ),
    )

    res = adapter.run(config=c, output_dir=tmp_path / "run_out", run_id="run_id")

    assert res.status == "completed", f"Deferred reason: {res.reason}"
    assert load_calls == 1
    assert generate_calls == 2


def test_prompt_content_rules():
    from aiforensics.baselines.qwen_vl.prompt import get_prompt

    p = get_prompt("qwen_json_v1")
    assert "label" in p
    assert "confidence" in p
    assert "evidence" in p
    assert "classifier_pred" not in p
    assert "fake_probability" not in p


def test_missing_tiny_dev_but_external_exists(tmp_path):
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "missing.csv"
    config.datasets.genimage_unseen.enabled = True
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen.csv"
    config.datasets.synthbuster.enabled = False

    (tmp_path / "unseen.csv").write_text(
        "sample_id,path,label,split,source,dataset,checksum\n1,1.jpg,fake,dev,s,unseen,"
        "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3\n"
    )

    adapter = QwenVLAdapter()
    records = adapter._load_manifests(config)

    assert len(records) == 1
    assert records[0].sample_id == "1"
    assert records[0].dataset == "unseen"


def test_all_manifests_missing_fails(tmp_path):
    import pytest

    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config
    from aiforensics.data.manifest import ManifestError

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "missing1.csv"
    config.datasets.genimage_unseen.manifest = tmp_path / "missing2.csv"
    config.datasets.synthbuster.manifest = tmp_path / "missing3.csv"

    adapter = QwenVLAdapter()
    with pytest.raises(ManifestError, match="No valid evaluation manifests found"):
        adapter._load_manifests(config)


def test_disabled_tiny_genimage_is_ignored(tmp_path):
    """datasets.tiny_genimage.enabled=false must exclude tiny from evaluation."""
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
    from aiforensics.config.load import load_config

    config = load_config("configs/phase_ab_smoke.yaml")
    config.paths.data_root = tmp_path
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "dev.csv"
    config.datasets.genimage_unseen.enabled = True
    config.datasets.genimage_unseen.manifest = tmp_path / "unseen.csv"
    config.datasets.synthbuster.enabled = False

    checksum = "59e40235e6bfac39e4af3ac2fdcca12fc4e21fed53b56935938f7541459c68a3"
    (tmp_path / "dev.csv").write_text(
        f"sample_id,path,label,split,source,dataset,checksum\ntiny-1,1.jpg,fake,dev,s,d,{checksum}\n"
    )
    (tmp_path / "unseen.csv").write_text(
        f"sample_id,path,label,split,source,dataset,checksum\next-1,2.jpg,fake,dev,s,u,{checksum}\n"
    )

    config.datasets.tiny_genimage.enabled = False
    records = QwenVLAdapter()._load_manifests(config)
    assert [r.sample_id for r in records] == ["ext-1"]
