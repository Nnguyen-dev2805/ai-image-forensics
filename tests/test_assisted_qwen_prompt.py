"""Task 9 contract tests for the assisted_qwen baseline.

Covers the minimum contracts from docs/specs/task-9-assisted-qwen-baseline-spec.md.
All tests run without the optional ``qwen`` dependency group and without network access.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
import yaml

from aiforensics.baselines.assisted_qwen.adapter import (
    AssistedInput,
    AssistedQwenAdapter,
    BaselineDeferredError,
)
from aiforensics.baselines.assisted_qwen.prompt import get_assisted_prompt
from aiforensics.baselines.base import RunResult
from aiforensics.cache.keys import cache_key
from aiforensics.cli.main import main
from aiforensics.config.load import load_config
from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import load_manifest
from aiforensics.schemas.predictions import (
    PredictionRecord,
    load_predictions,
    write_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "phase_ab_smoke.yaml"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CSV_HEADER = "sample_id,label,dataset,split,source,path,checksum"
DEFAULT_OUTPUT = '{"label": "real", "confidence": 0.99, "evidence": "x"}'
QWEN_MODULES = ("torch", "transformers", "qwen_vl_utils", "accelerate")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> AppConfig:
    config = load_config(SMOKE_CONFIG)
    config.paths.data_root = tmp_path
    config.paths.manifest_root = tmp_path / "manifests"
    config.paths.cache_root = tmp_path / "cache"
    config.paths.output_root = tmp_path / "outputs"
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "dev.csv"
    config.datasets.genimage_unseen.enabled = False
    config.datasets.synthbuster.enabled = False
    return config


def _add_sample(
    tmp_path: Path, sample_id: str = "1", content: bytes = b"", label: str = "real"
) -> dict:
    filename = f"{sample_id}.jpg"
    (tmp_path / filename).write_bytes(content)
    return {
        "sample_id": sample_id,
        "label": label,
        "path": filename,
        "checksum": hashlib.sha256(content).hexdigest(),
    }


def _write_manifest(path: Path, rows: list[dict]) -> None:
    lines = [CSV_HEADER]
    for row in rows:
        lines.append(
            f"{row['sample_id']},{row['label']},t,{row.get('split', 'dev')},"
            f"{row.get('source', 's')},{row['path']},{row['checksum']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clip_record(
    sample_id: str = "1",
    label_pred: Literal["real", "fake", "unknown"] = "fake",
    score_fake: float | None = 0.8,
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        label_true="real",
        label_pred=label_pred,
        score_fake=score_fake,
        model_name="clip_probe",
        source="t",
        run_id="run_clip",
        dataset="t",
        split="dev",
        path=Path("1.jpg"),
        checksum=EMPTY_SHA256,
    )


def _write_clip_run(
    output_root: Path,
    records: list[PredictionRecord] | None = None,
    status: str = "completed",
    run_name: str = "run_clip",
    config: AppConfig | None = None,
) -> Path:
    """Create a CLIP run directory, scope-stamped for ``config`` when given.

    Assistant discovery only accepts CLIP runs carrying the current config's
    run scope, so a factory that omits ``config`` models an unscoped/foreign
    run that must be ignored.
    """
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps({"baseline": "clip_probe", "status": status}), encoding="utf-8"
    )
    if records is not None:
        write_predictions(records, run_dir / "predictions.jsonl")
    if config is not None:
        _write_clip_scope(run_dir, config)
    return run_dir


def _write_clip_scope(run_dir: Path, config: AppConfig) -> None:
    from aiforensics.runs.scope import SCOPE_FILENAME, compute_run_scope, write_run_scope

    write_run_scope(run_dir / SCOPE_FILENAME, compute_run_scope(config))


def _assist_input(
    sample_id: str = "1",
    classifier_pred: Literal["real", "fake"] = "fake",
    fake_probability: float = 0.8,
) -> AssistedInput:
    return AssistedInput(
        sample_id=sample_id,
        classifier_pred=classifier_pred,
        fake_probability=fake_probability,
        source_prediction_files=(),
    )


def _patch_find_spec(monkeypatch: pytest.MonkeyPatch, available: bool = True) -> None:
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name, *args, **kwargs):
        if name in QWEN_MODULES:
            return "mocked" if available else None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)


def _mock_assistant_inputs(
    monkeypatch: pytest.MonkeyPatch,
    adapter: AssistedQwenAdapter,
    mapping: dict[str, AssistedInput],
) -> None:
    monkeypatch.setattr(adapter, "_discover_assistant_inputs", lambda config: mapping)


def _mock_qwen_runtime(monkeypatch: pytest.MonkeyPatch, output_text: str = DEFAULT_OUTPUT):
    import aiforensics.baselines.assisted_qwen.adapter as adapter_module

    _patch_find_spec(monkeypatch, available=True)
    monkeypatch.setattr(adapter_module, "get_qwen_device", lambda *args, **kwargs: "cuda")
    monkeypatch.setattr(adapter_module, "load_model", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(adapter_module, "generate_one_image", lambda *args, **kwargs: output_text)
    return adapter_module


def _setup_completed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_ids: tuple[str, ...] = ("1",),
    output_text: str = DEFAULT_OUTPUT,
) -> tuple[AppConfig, AssistedQwenAdapter]:
    config = _make_config(tmp_path)
    rows = [_add_sample(tmp_path, sample_id) for sample_id in sample_ids]
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, rows)
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    config.baselines.assisted_qwen.cache_outputs = False
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {sid: _assist_input(sid) for sid in sample_ids})
    _mock_qwen_runtime(monkeypatch, output_text)
    return config, adapter


def _write_cli_config(tmp_path: Path, config: AppConfig) -> Path:
    (tmp_path / "pyproject.toml").touch()
    cfg_path = tmp_path / "cli_config.yaml"
    cfg_path.write_text(yaml.safe_dump(json.loads(config.model_dump_json())), encoding="utf-8")
    return cfg_path


def _load_eval_records(config: AppConfig) -> list:
    return load_manifest(
        config.datasets.tiny_genimage.dev_manifest, data_root=config.paths.data_root
    )


def _invoke_inference(config: AppConfig, adapter: AssistedQwenAdapter, inputs) -> None:
    records = _load_eval_records(config)
    counts = {"parsed": 0, "recovered": 0, "failed": 0, "cache_hits": 0, "cache_misses": 0}
    adapter._run_inference(records, inputs, config, "variation_run", counts)


# ---------------------------------------------------------------------------
# Contracts 1-11: prompt builder
# ---------------------------------------------------------------------------


def test_adapter_name():
    assert AssistedQwenAdapter().name == "assisted_qwen"


def test_prompt_deterministic():
    p1 = get_assisted_prompt("assisted_qwen_json_v1", classifier_pred="fake", fake_probability=0.99)
    p2 = get_assisted_prompt("assisted_qwen_json_v1", classifier_pred="fake", fake_probability=0.99)
    assert p1 == p2


def test_prompt_includes_classifier_pred():
    p = get_assisted_prompt(
        "assisted_qwen_json_v1", classifier_pred="real", fake_probability=0.12345
    )
    assert "classifier_pred" in p
    assert '"classifier_pred": "real"' in p


def test_prompt_includes_fake_probability():
    p = get_assisted_prompt(
        "assisted_qwen_json_v1", classifier_pred="real", fake_probability=0.12345
    )
    assert "fake_probability" in p
    assert '"fake_probability": 0.12345' in p


def test_prompt_requests_fields():
    p = get_assisted_prompt(
        "assisted_qwen_json_v1", classifier_pred="real", fake_probability=0.12345
    )
    assert "label" in p
    assert "confidence" in p
    assert "evidence" in p


def test_prompt_no_ground_truth():
    p = get_assisted_prompt(
        "assisted_qwen_json_v1", classifier_pred="real", fake_probability=0.12345
    )
    low = p.lower()
    assert "ground truth" not in low
    assert "label_true" not in low
    assert "clip embeddings" not in low
    assert "npr" not in low
    assert "patch attribution" not in low
    # The prompt mentions "retrieval results" only inside its own prohibition
    # sentence, so assert that no actual retrieved content is interpolated.
    assert "retrieved examples" not in low
    assert "nearest neighbor" not in low


def test_prompt_unsupported_id():
    with pytest.raises(ValueError, match="Unsupported prompt_id"):
        get_assisted_prompt("invalid_id", classifier_pred="fake", fake_probability=0.5)


def test_prompt_invalid_classifier_pred():
    with pytest.raises(ValueError, match="Invalid classifier_pred"):
        get_assisted_prompt(
            "assisted_qwen_json_v1", classifier_pred="unknown", fake_probability=0.5
        )


def test_prompt_out_of_range_prob():
    with pytest.raises(ValueError, match="Invalid fake_probability"):
        get_assisted_prompt("assisted_qwen_json_v1", classifier_pred="fake", fake_probability=1.5)


def test_prompt_non_finite_prob():
    with pytest.raises(ValueError, match="Invalid fake_probability"):
        get_assisted_prompt(
            "assisted_qwen_json_v1", classifier_pred="fake", fake_probability=float("inf")
        )


def test_prompt_stable_formatting():
    p1 = get_assisted_prompt("assisted_qwen_json_v1", classifier_pred="fake", fake_probability=1.0)
    p2 = get_assisted_prompt("assisted_qwen_json_v1", classifier_pred="fake", fake_probability=1.0)
    assert p1 == p2
    # .12g renders 1.0 as "1" (no trailing .0) and keeps 0.5 stable.
    assert '"fake_probability": 1' in p1
    assert '"fake_probability": 1.0' not in p1
    p3 = get_assisted_prompt("assisted_qwen_json_v1", classifier_pred="fake", fake_probability=0.5)
    assert '"fake_probability": 0.5' in p3


# ---------------------------------------------------------------------------
# Contracts 12-22: CLIP assistant discovery and aggregation
# ---------------------------------------------------------------------------


def test_discover_ignores_failed_runs(tmp_path):
    config = _make_config(tmp_path)
    _write_clip_run(config.paths.output_root, [_clip_record()], status="failed", config=config)
    adapter = AssistedQwenAdapter()
    with pytest.raises(Exception, match="No completed clip_probe predictions found"):
        adapter._discover_assistant_inputs(config)


def test_discover_ignores_deferred_runs(tmp_path):
    config = _make_config(tmp_path)
    _write_clip_run(config.paths.output_root, [_clip_record()], status="deferred", config=config)
    adapter = AssistedQwenAdapter()
    with pytest.raises(Exception, match="No completed clip_probe predictions found"):
        adapter._discover_assistant_inputs(config)


def test_discover_loads_completed_predictions(tmp_path):
    config = _make_config(tmp_path)
    run_dir = _write_clip_run(
        config.paths.output_root, [_clip_record("1", "fake", 0.8)], config=config
    )
    adapter = AssistedQwenAdapter()
    inputs = adapter._discover_assistant_inputs(config)
    assert "1" in inputs
    assert inputs["1"].classifier_pred == "fake"
    assert inputs["1"].fake_probability == 0.8
    assert run_dir / "predictions.jsonl" in inputs["1"].source_prediction_files


def test_discover_missing_predictions_fails(tmp_path):
    config = _make_config(tmp_path)
    adapter = AssistedQwenAdapter()
    with pytest.raises(Exception, match="No completed clip_probe predictions found"):
        adapter._discover_assistant_inputs(config)


def test_missing_assistant_for_evaluation_sample_fails(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {})
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "failed"
    assert "Missing clip_probe assistant prediction for sample_id=1" in (res.reason or "")


def test_invalid_clip_score_none_fails(tmp_path):
    config = _make_config(tmp_path)
    _write_clip_run(config.paths.output_root, [_clip_record("1", "fake", None)], config=config)
    adapter = AssistedQwenAdapter()
    with pytest.raises(Exception, match="Invalid clip_probe prediction score"):
        adapter._discover_assistant_inputs(config)


def test_invalid_clip_label_unknown_fails(tmp_path):
    config = _make_config(tmp_path)
    _write_clip_run(config.paths.output_root, [_clip_record("1", "unknown", 0.5)], config=config)
    adapter = AssistedQwenAdapter()
    with pytest.raises(Exception, match="Invalid clip_probe prediction score"):
        adapter._discover_assistant_inputs(config)


def test_aggregate_mean_score(tmp_path):
    config = _make_config(tmp_path)
    config.baselines.clip_probe.seeds = [70, 71]
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "fake", 0.6)],
        run_name="001_clip_probe_seed70",
        config=config,
    )
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "fake", 0.8)],
        run_name="002_clip_probe_seed71",
        config=config,
    )
    adapter = AssistedQwenAdapter()
    inputs = adapter._discover_assistant_inputs(config)
    assert inputs["1"].fake_probability == pytest.approx(0.7)
    assert inputs["1"].classifier_pred == "fake"


def test_aggregate_boundary_half_maps_fake(tmp_path):
    config = _make_config(tmp_path)
    config.baselines.clip_probe.seeds = [70, 71]
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "real", 0.4)],
        run_name="001_clip_probe_seed70",
        config=config,
    )
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "fake", 0.6)],
        run_name="002_clip_probe_seed71",
        config=config,
    )
    adapter = AssistedQwenAdapter()
    inputs = adapter._discover_assistant_inputs(config)
    assert inputs["1"].fake_probability == pytest.approx(0.5)
    assert inputs["1"].classifier_pred == "fake"


def test_aggregate_maps_low_prob_to_real(tmp_path):
    config = _make_config(tmp_path)
    _write_clip_run(config.paths.output_root, [_clip_record("1", "real", 0.4)], config=config)
    adapter = AssistedQwenAdapter()
    inputs = adapter._discover_assistant_inputs(config)
    assert inputs["1"].fake_probability == pytest.approx(0.4)
    assert inputs["1"].classifier_pred == "real"


# ---------------------------------------------------------------------------
# Assistant input is bound to the current experiment, not to output_root history
# ---------------------------------------------------------------------------


def test_clip_run_without_scope_is_ignored(tmp_path):
    """A CLIP run with no run_scope.json cannot be attributed to this config."""
    config = _make_config(tmp_path)
    _write_clip_run(config.paths.output_root, [_clip_record("1", "fake", 0.8)])
    adapter = AssistedQwenAdapter()
    with pytest.raises(Exception, match="No completed clip_probe predictions found"):
        adapter._discover_assistant_inputs(config)


def test_clip_run_from_other_dataset_slice_is_ignored(tmp_path):
    """A CLIP run over a different evaluation slice must not feed this run."""
    config = _make_config(tmp_path)
    other = _make_config(tmp_path)
    other.datasets.genimage_unseen.enabled = True
    _write_clip_run(config.paths.output_root, [_clip_record("1", "fake", 0.8)], config=other)
    adapter = AssistedQwenAdapter()
    with pytest.raises(Exception, match="No completed clip_probe predictions found"):
        adapter._discover_assistant_inputs(config)


def test_unconfigured_seed_run_is_ignored(tmp_path):
    """Only seeds the current config declares may contribute assistant input."""
    config = _make_config(tmp_path)
    config.baselines.clip_probe.seeds = [70]
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "fake", 0.9)],
        run_name="001_clip_probe_seed70",
        config=config,
    )
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "real", 0.1)],
        run_name="002_clip_probe_seed999",
        config=config,
    )
    adapter = AssistedQwenAdapter()
    inputs = adapter._discover_assistant_inputs(config)
    assert adapter._counts["clip_files_used"] == 1
    assert inputs["1"].fake_probability == pytest.approx(0.9)


def test_reran_seed_uses_latest_run_only(tmp_path):
    """Re-running one seed replaces its prediction instead of averaging twice."""
    config = _make_config(tmp_path)
    config.baselines.clip_probe.seeds = [70]
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "real", 0.1)],
        run_name="20260101T000000000000Z_clip_probe_seed70",
        config=config,
    )
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "fake", 0.9)],
        run_name="20260102T000000000000Z_clip_probe_seed70",
        config=config,
    )
    adapter = AssistedQwenAdapter()
    inputs = adapter._discover_assistant_inputs(config)
    assert adapter._counts["clip_files_used"] == 1
    assert inputs["1"].fake_probability == pytest.approx(0.9)


def test_assistant_input_is_stable_when_foreign_history_grows(tmp_path):
    """Adding runs from another experiment must not change assistant input."""
    config = _make_config(tmp_path)
    config.baselines.clip_probe.seeds = [70]
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "fake", 0.9)],
        run_name="001_clip_probe_seed70",
        config=config,
    )
    before = AssistedQwenAdapter()._discover_assistant_inputs(config)

    other = _make_config(tmp_path)
    other.datasets.genimage_unseen.enabled = True
    _write_clip_run(
        config.paths.output_root,
        [_clip_record("1", "real", 0.1)],
        run_name="002_clip_probe_seed70",
        config=other,
    )
    after = AssistedQwenAdapter()._discover_assistant_inputs(config)

    assert before["1"].fake_probability == after["1"].fake_probability
    assert before["1"].classifier_pred == after["1"].classifier_pred


def test_extra_clip_records_ignored(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(
        monkeypatch,
        adapter,
        {"1": _assist_input("1"), "2": _assist_input("2", "real", 0.1)},
    )
    _mock_qwen_runtime(monkeypatch)
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert len(preds) == 1
    assert preds[0].sample_id == "1"


# ---------------------------------------------------------------------------
# Contracts 23-25: evaluation data selection
# ---------------------------------------------------------------------------


def test_disabled_optional_dataset_ignored(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    genimage_manifest = tmp_path / "genimage.csv"
    _write_manifest(
        genimage_manifest,
        [
            {
                "sample_id": "2",
                "label": "real",
                "path": "missing.jpg",
                "checksum": EMPTY_SHA256,
            }
        ],
    )
    synthbuster_manifest = tmp_path / "synthbuster.csv"
    _write_manifest(
        synthbuster_manifest,
        [
            {
                "sample_id": "3",
                "label": "real",
                "path": "missing.jpg",
                "checksum": EMPTY_SHA256,
            }
        ],
    )
    config.datasets.genimage_unseen.enabled = False
    config.datasets.genimage_unseen.manifest = genimage_manifest
    config.datasets.synthbuster.enabled = False
    config.datasets.synthbuster.manifest = synthbuster_manifest
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1")})
    _mock_qwen_runtime(monkeypatch)
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert [p.sample_id for p in preds] == ["1"]


def test_missing_tiny_dev_allowed_with_external(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "missing_dev.csv"
    external_manifest = tmp_path / "external.csv"
    row = _add_sample(tmp_path, "ext1")
    _write_manifest(external_manifest, [row])
    config.datasets.genimage_unseen.enabled = True
    config.datasets.genimage_unseen.manifest = external_manifest
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"ext1": _assist_input("ext1")})
    _mock_qwen_runtime(monkeypatch)
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert [p.sample_id for p in preds] == ["ext1"]


def test_invalid_existing_manifest_fails(tmp_path):
    config = _make_config(tmp_path)
    dev_manifest = tmp_path / "dev.csv"
    dev_manifest.write_text("sample_id,path\n1,1.jpg\n", encoding="utf-8")
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    adapter = AssistedQwenAdapter()
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "failed"
    assert "Missing required columns" in (res.reason or "")
    assert not (tmp_path / "run_out" / "predictions.jsonl").exists()


def test_disabled_tiny_genimage_is_ignored(tmp_path, monkeypatch):
    """datasets.tiny_genimage.enabled=false must exclude tiny from evaluation."""
    config = _make_config(tmp_path)
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [_add_sample(tmp_path, "tiny1")])
    external_manifest = tmp_path / "external.csv"
    _write_manifest(external_manifest, [_add_sample(tmp_path, "ext1")])
    config.datasets.genimage_unseen.enabled = True
    config.datasets.genimage_unseen.manifest = external_manifest
    config.datasets.tiny_genimage.enabled = False
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False

    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"ext1": _assist_input("ext1")})
    _mock_qwen_runtime(monkeypatch)
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert [p.sample_id for p in preds] == ["ext1"]


# ---------------------------------------------------------------------------
# Contracts 26-32: deferred, failed, and parse behavior
# ---------------------------------------------------------------------------


def test_cli_disabled_baseline_creates_deferred_artifact_exit_zero(tmp_path):
    config = _make_config(tmp_path)
    assert config.baselines.assisted_qwen.enabled is False
    cfg_path = _write_cli_config(tmp_path, config)
    exit_code = main(["run", "--baseline", "assisted_qwen", "--config", str(cfg_path)])
    assert exit_code == 0
    run_dirs = list((tmp_path / "outputs").glob("*_assisted_qwen"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["baseline"] == "assisted_qwen"
    assert status["status"] == "deferred"
    for artifact in ("config.yaml", "environment.json", "logs.txt"):
        assert (run_dir / artifact).exists()
    assert not (run_dir / "predictions.jsonl").exists()


def test_missing_qwen_deps_deferred(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = True
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1")})
    _patch_find_spec(monkeypatch, available=False)
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "deferred"
    assert "Missing Qwen dependencies" in (res.reason or "")


def test_missing_qwen_deps_failed(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1")})
    _patch_find_spec(monkeypatch, available=False)
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "failed"
    assert "Missing Qwen dependencies" in (res.reason or "")


def test_setup_failure_deferred(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = True
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1")})
    adapter_module = _mock_qwen_runtime(monkeypatch)

    def mock_load_model(*args, **kwargs):
        raise BaselineDeferredError("Model setup failed intentionally")

    monkeypatch.setattr(adapter_module, "load_model", mock_load_model)

    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "deferred"
    assert "Model setup failed" in (res.reason or "")


def test_per_sample_generation_failure_is_failed_not_deferred(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = True
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1")})
    adapter_module = _mock_qwen_runtime(monkeypatch)

    def mock_generate(*args, **kwargs):
        raise Exception("Generation failed intentionally")

    monkeypatch.setattr(adapter_module, "generate_one_image", mock_generate)

    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "failed"
    assert "Generation failed" in (res.reason or "")
    assert not (tmp_path / "run_out" / "predictions.jsonl").exists()


def test_parse_failure_produces_unknown(tmp_path, monkeypatch):
    config, adapter = _setup_completed_run(
        tmp_path, monkeypatch, output_text="garbage output not JSON"
    )
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert len(preds) == 1
    assert preds[0].label_pred == "unknown"
    assert preds[0].score_fake is None
    assert preds[0].parse_status == "failed"
    assert preds[0].raw_output == "garbage output not JSON"


def test_prediction_validation_failure_cleanup(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1")})
    adapter_module = _mock_qwen_runtime(monkeypatch)

    def mock_validate_predictions(*args, **kwargs):
        raise ValueError("Intentional validation failure")

    monkeypatch.setattr(adapter_module, "validate_predictions", mock_validate_predictions)

    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "failed"
    assert "Intentional validation failure" in (res.reason or "")
    assert not (tmp_path / "run_out" / "predictions.jsonl").exists()


# ---------------------------------------------------------------------------
# Contracts 33-43: raw-output cache
# ---------------------------------------------------------------------------


def _assisted_cache_key(config: AppConfig, csum: str, version: str = "assisted_qwen_raw_v1"):
    cfg = config.baselines.assisted_qwen
    return cache_key(
        {
            "baseline": "assisted_qwen",
            "sample_checksum": csum,
            "base_model_id": cfg.base_model_id,
            "prompt_id": cfg.prompt_id,
            "assistant_source": cfg.assistant_source,
            "classifier_pred": "fake",
            "fake_probability": format(0.8, ".12g"),
            "temperature": str(cfg.temperature),
            "max_new_tokens": str(cfg.max_new_tokens),
            "output_cache_version": version,
        }
    )


def _write_cache_entry(
    config: AppConfig,
    csum: str,
    payload: str,
    version: str = "assisted_qwen_raw_v1",
) -> Path:
    key = _assisted_cache_key(config, csum, version)
    cache_dir = config.paths.cache_root / "assisted_qwen" / "raw_outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    cache_path.write_text(payload, encoding="utf-8")
    return cache_path


def _counting_runtime(monkeypatch: pytest.MonkeyPatch) -> dict:
    import aiforensics.baselines.assisted_qwen.adapter as adapter_module

    calls = {"load": 0, "generate": 0}

    def mock_load(*args, **kwargs):
        calls["load"] += 1
        return None, None

    def mock_generate(*args, **kwargs):
        calls["generate"] += 1
        return DEFAULT_OUTPUT

    _patch_find_spec(monkeypatch, available=True)
    monkeypatch.setattr(adapter_module, "get_qwen_device", lambda *args, **kwargs: "cuda")
    monkeypatch.setattr(adapter_module, "load_model", mock_load)
    monkeypatch.setattr(adapter_module, "generate_one_image", mock_generate)
    return calls


def test_cache_hit_bypasses_model(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    config.baselines.assisted_qwen.cache_outputs = True
    _write_cache_entry(
        config,
        EMPTY_SHA256,
        json.dumps({"raw_output": '{"label": "fake", "confidence": 0.9, "evidence": "x"}'}),
    )

    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1", "fake", 0.8)})
    calls = _counting_runtime(monkeypatch)

    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    assert calls["load"] == 0
    assert calls["generate"] == 0
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert preds[0].label_pred == "fake"
    assert preds[0].score_fake == 0.9
    assert preds[0].parse_status == "parsed"
    assert adapter._counts["cache_hits"] == 1


def test_run_respects_cache_outputs_false(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    config.baselines.assisted_qwen.cache_outputs = False
    _write_cache_entry(
        config,
        EMPTY_SHA256,
        json.dumps({"raw_output": '{"label": "fake", "confidence": 0.9, "evidence": "x"}'}),
    )

    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1", "fake", 0.8)})
    calls = _counting_runtime(monkeypatch)

    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    assert calls["generate"] == 1
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert preds[0].label_pred == "real"
    assert adapter._counts["cache_hits"] == 0


def test_cache_key_version_change_invalidates_cache(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    config.baselines.assisted_qwen.cache_outputs = True
    _write_cache_entry(
        config,
        EMPTY_SHA256,
        json.dumps({"raw_output": '{"label": "fake", "confidence": 0.9, "evidence": "x"}'}),
        version="WRONG_VERSION_V0",
    )

    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1", "fake", 0.8)})
    calls = _counting_runtime(monkeypatch)

    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    assert calls["generate"] == 1
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert preds[0].label_pred == "real"


def test_corrupt_cache_entry_recomputed(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    config.baselines.assisted_qwen.cache_outputs = True
    cache_path = _write_cache_entry(config, EMPTY_SHA256, "{not valid json")

    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1", "fake", 0.8)})
    calls = _counting_runtime(monkeypatch)

    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    assert calls["generate"] == 1
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert preds[0].label_pred == "real"
    assert preds[0].raw_output == DEFAULT_OUTPUT
    rewritten = json.loads(cache_path.read_text(encoding="utf-8"))
    assert rewritten["raw_output"] == DEFAULT_OUTPUT


def test_cache_key_changes_with_checksum(tmp_path, monkeypatch):
    import aiforensics.baselines.assisted_qwen.adapter as adapter_module

    config = _make_config(tmp_path)
    row1 = _add_sample(tmp_path, "1", content=b"image-bytes-one")
    row2 = _add_sample(tmp_path, "2", content=b"image-bytes-two")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row1, row2])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    config.baselines.assisted_qwen.cache_outputs = True

    captured = []

    def mock_cache_key(parts):
        captured.append(dict(parts))
        return f"checksum_key_{len(captured)}"

    _patch_find_spec(monkeypatch, available=True)
    monkeypatch.setattr(adapter_module, "cache_key", mock_cache_key)
    monkeypatch.setattr(adapter_module, "get_qwen_device", lambda *args, **kwargs: "cuda")
    monkeypatch.setattr(adapter_module, "load_model", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(
        adapter_module, "generate_one_image", lambda *args, **kwargs: DEFAULT_OUTPUT
    )

    adapter = AssistedQwenAdapter()
    records = _load_eval_records(config)
    inputs = {"1": _assist_input("1"), "2": _assist_input("2")}
    counts = {"parsed": 0, "recovered": 0, "failed": 0, "cache_hits": 0, "cache_misses": 0}
    adapter._run_inference(records, inputs, config, "run_id", counts)

    assert captured[0]["sample_checksum"] == hashlib.sha256(b"image-bytes-one").hexdigest()
    assert captured[1]["sample_checksum"] == hashlib.sha256(b"image-bytes-two").hexdigest()
    assert captured[0]["sample_checksum"] != captured[1]["sample_checksum"]


def _variation_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import aiforensics.baselines.assisted_qwen.adapter as adapter_module

    config = _make_config(tmp_path)
    row = _add_sample(tmp_path, "1")
    _write_manifest(config.datasets.tiny_genimage.dev_manifest, [row])
    config.baselines.assisted_qwen.enabled = True
    config.baselines.assisted_qwen.allow_deferred = False
    config.baselines.assisted_qwen.cache_outputs = True

    captured = []

    def mock_cache_key(parts):
        captured.append(dict(parts))
        return f"variation_key_{len(captured)}"

    _patch_find_spec(monkeypatch, available=True)
    monkeypatch.setattr(adapter_module, "cache_key", mock_cache_key)
    monkeypatch.setattr(adapter_module, "get_qwen_device", lambda *args, **kwargs: "cuda")
    monkeypatch.setattr(adapter_module, "load_model", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(
        adapter_module, "generate_one_image", lambda *args, **kwargs: DEFAULT_OUTPUT
    )
    monkeypatch.setattr(adapter_module, "get_assisted_prompt", lambda *args, **kwargs: "prompt")
    adapter = AssistedQwenAdapter()
    _mock_assistant_inputs(monkeypatch, adapter, {"1": _assist_input("1", "fake", 0.8)})
    return config, adapter, captured


def test_cache_key_variations_temperature(tmp_path, monkeypatch):
    config, adapter, captured = _variation_setup(tmp_path, monkeypatch)
    inputs = {"1": _assist_input("1", "fake", 0.8)}
    config.baselines.assisted_qwen.temperature = 0.5
    _invoke_inference(config, adapter, inputs)
    config.baselines.assisted_qwen.temperature = 0.6
    _invoke_inference(config, adapter, inputs)
    assert captured[0]["temperature"] == "0.5"
    assert captured[1]["temperature"] == "0.6"


def test_cache_key_variations_prompt_id(tmp_path, monkeypatch):
    config, adapter, captured = _variation_setup(tmp_path, monkeypatch)
    inputs = {"1": _assist_input("1", "fake", 0.8)}
    config.baselines.assisted_qwen.prompt_id = "p1"
    _invoke_inference(config, adapter, inputs)
    config.baselines.assisted_qwen.prompt_id = "p2"
    _invoke_inference(config, adapter, inputs)
    assert captured[0]["prompt_id"] == "p1"
    assert captured[1]["prompt_id"] == "p2"


def test_cache_key_variations_max_new_tokens(tmp_path, monkeypatch):
    config, adapter, captured = _variation_setup(tmp_path, monkeypatch)
    inputs = {"1": _assist_input("1", "fake", 0.8)}
    config.baselines.assisted_qwen.max_new_tokens = 100
    _invoke_inference(config, adapter, inputs)
    config.baselines.assisted_qwen.max_new_tokens = 200
    _invoke_inference(config, adapter, inputs)
    assert captured[0]["max_new_tokens"] == "100"
    assert captured[1]["max_new_tokens"] == "200"


def test_cache_key_variations_base_model_id(tmp_path, monkeypatch):
    config, adapter, captured = _variation_setup(tmp_path, monkeypatch)
    inputs = {"1": _assist_input("1", "fake", 0.8)}
    config.baselines.assisted_qwen.base_model_id = "m1"
    _invoke_inference(config, adapter, inputs)
    config.baselines.assisted_qwen.base_model_id = "m2"
    _invoke_inference(config, adapter, inputs)
    assert captured[0]["base_model_id"] == "m1"
    assert captured[1]["base_model_id"] == "m2"


def test_cache_key_variations_classifier_pred(tmp_path, monkeypatch):
    config, adapter, captured = _variation_setup(tmp_path, monkeypatch)
    _invoke_inference(config, adapter, {"1": _assist_input("1", "real", 0.2)})
    _invoke_inference(config, adapter, {"1": _assist_input("1", "fake", 0.2)})
    assert captured[0]["classifier_pred"] == "real"
    assert captured[1]["classifier_pred"] == "fake"


def test_cache_key_variations_fake_probability(tmp_path, monkeypatch):
    config, adapter, captured = _variation_setup(tmp_path, monkeypatch)
    _invoke_inference(config, adapter, {"1": _assist_input("1", "fake", 0.1)})
    _invoke_inference(config, adapter, {"1": _assist_input("1", "fake", 0.2)})
    assert captured[0]["fake_probability"] == format(0.1, ".12g")
    assert captured[1]["fake_probability"] == format(0.2, ".12g")


def test_cache_key_variations_assistant_source(tmp_path, monkeypatch):
    config, adapter, captured = _variation_setup(tmp_path, monkeypatch)
    inputs = {"1": _assist_input("1", "fake", 0.8)}
    config.baselines.assisted_qwen.assistant_source = "s1"
    _invoke_inference(config, adapter, inputs)
    config.baselines.assisted_qwen.assistant_source = "s2"
    _invoke_inference(config, adapter, inputs)
    assert captured[0]["assistant_source"] == "s1"
    assert captured[1]["assistant_source"] == "s2"


# ---------------------------------------------------------------------------
# Contracts 38-40: runtime behavior and CLI artifacts
# ---------------------------------------------------------------------------


def test_model_processor_loaded_once_per_run(tmp_path, monkeypatch):
    config, adapter = _setup_completed_run(tmp_path, monkeypatch, sample_ids=("1", "2"))
    adapter_module = _mock_qwen_runtime(monkeypatch)
    calls = {"load": 0, "generate": 0}

    def mock_load(*args, **kwargs):
        calls["load"] += 1
        # Non-None sentinel so the adapter's `model is None` lazy-init check
        # sees the model as already loaded for the second record.
        return "model", "processor"

    def mock_generate(*args, **kwargs):
        calls["generate"] += 1
        return DEFAULT_OUTPUT

    monkeypatch.setattr(adapter_module, "load_model", mock_load)
    monkeypatch.setattr(adapter_module, "generate_one_image", mock_generate)

    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    assert calls["load"] == 1
    assert calls["generate"] == 2
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert len(preds) == 2


def test_raw_output_preserved(tmp_path, monkeypatch):
    raw = '```json\n{"label": "fake", "confidence": 0.99, "evidence": "x"}\n```'
    config, adapter = _setup_completed_run(tmp_path, monkeypatch, output_text=raw)
    res = adapter.run(config=config, output_dir=tmp_path / "run_out", run_id="run_id")
    assert res.status == "completed"
    preds = load_predictions(tmp_path / "run_out" / "predictions.jsonl")
    assert preds[0].raw_output == raw
    assert preds[0].parse_status == "recovered"
    assert preds[0].label_pred == "fake"
    assert preds[0].score_fake == pytest.approx(0.99)


def test_cli_creates_single_run_dir_for_assisted(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.assisted_qwen.enabled = True
    config.baselines.clip_probe.seeds = [1, 2, 3]

    def mock_run(self, *, config, output_dir, run_id, seed=None):
        (output_dir / "predictions.jsonl").touch()
        (output_dir / "logs.txt").touch()
        return RunResult(
            baseline="assisted_qwen",
            run_id=run_id,
            status="completed",
            output_dir=output_dir,
            prediction_path=output_dir / "predictions.jsonl",
            log_path=output_dir / "logs.txt",
            environment_path=output_dir / "environment.json",
            status_path=output_dir / "status.json",
            reason=None,
        )

    monkeypatch.setattr(AssistedQwenAdapter, "run", mock_run)
    cfg_path = _write_cli_config(tmp_path, config)
    exit_code = main(["run", "--baseline", "assisted_qwen", "--config", str(cfg_path)])
    assert exit_code == 0
    run_dirs = list((tmp_path / "outputs").glob("*_assisted_qwen"))
    assert len(run_dirs) == 1
    status = json.loads((run_dirs[0] / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"


# ---------------------------------------------------------------------------
# Contract 41: smoke CLI deferred run without Qwen imports
# ---------------------------------------------------------------------------


def test_smoke_cli_deferred_no_import_qwen(tmp_path):
    config = _make_config(tmp_path)
    cfg_path = _write_cli_config(tmp_path, config)
    outputs_root = tmp_path / "outputs"
    cfg_path_str = str(cfg_path)
    outputs_str = str(outputs_root)
    script = f"""
import json
import sys
from pathlib import Path

from aiforensics.cli.main import main

exit_code = main(["run", "--baseline", "assisted_qwen", "--config", {cfg_path_str!r}])
assert exit_code == 0, f"exit_code={{exit_code}}"
assert "transformers" not in sys.modules
assert "torch" not in sys.modules

run_dirs = list(Path({outputs_str!r}).glob("*_assisted_qwen"))
assert len(run_dirs) == 1, run_dirs
status = json.loads((run_dirs[0] / "status.json").read_text())
assert status["baseline"] == "assisted_qwen", status
assert status["status"] == "deferred", status
"""
    script_path = tmp_path / "runner.py"
    script_path.write_text(script, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    res = subprocess.run(
        [sys.executable, str(script_path)], capture_output=True, text=True, env=env
    )
    assert res.returncode == 0, res.stderr
