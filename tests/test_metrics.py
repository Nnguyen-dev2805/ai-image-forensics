import json
from pathlib import Path
from typing import Literal

import pytest

from aiforensics.cli.main import main
from aiforensics.evaluation.metrics import (
    METRIC_NAMES,
    MetricsError,
    compute_classification_metrics,
    compute_metrics_by_source,
    discover_prediction_files,
    evaluate_prediction_file,
    write_metrics_outputs,
)
from aiforensics.schemas.predictions import PredictionRecord


def prediction(
    sample_id: str,
    label_true: Literal["real", "fake"],
    label_pred: Literal["real", "fake", "unknown"],
    score_fake: float | None,
    source: str = "smoke",
    model_name: Literal["clip_probe", "qwen_vl", "npr", "assisted_qwen"] = "clip_probe",
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        label_true=label_true,
        label_pred=label_pred,
        score_fake=score_fake,
        model_name=model_name,
        source=source,
    )


def test_compute_empty():
    metrics = compute_classification_metrics([])
    for name in METRIC_NAMES:
        assert metrics[name] is None


def test_perfect_metrics():
    records = [
        prediction("real-1", "real", "real", 0.05),
        prediction("fake-1", "fake", "fake", 0.95),
    ]
    metrics = compute_classification_metrics(records)
    for name in METRIC_NAMES:
        assert metrics[name] == 1.0


def test_unknown_incorrect_accuracy():
    records = [
        prediction("r1", "real", "real", 0.1),
        prediction("r2", "real", "unknown", 0.5),  # incorrect
    ]
    metrics = compute_classification_metrics(records)
    assert metrics["accuracy"] == 0.5


def test_fake_unknown_counts_against_recall():
    records = [
        prediction("r1", "real", "real", 0.05),
        prediction("f1", "fake", "unknown", None),
    ]
    metrics = compute_classification_metrics(records)
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["recall"] == 0.0
    assert metrics["precision"] is None
    assert metrics["f1"] is None
    assert metrics["auroc"] is None


def test_precision_none():
    records = [
        prediction("r1", "real", "real", 0.1),
        prediction("f1", "fake", "real", 0.2),  # fn
    ]
    metrics = compute_classification_metrics(records)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] is None
    assert metrics["recall"] == 0.0


def test_recall_none():
    records = [
        prediction("r1", "real", "real", 0.1),
    ]
    metrics = compute_classification_metrics(records)
    assert metrics["recall"] is None
    assert metrics["precision"] is None
    assert metrics["balanced_accuracy"] is None


def test_auroc_none_scenarios():
    # Only one class
    rects1 = [prediction("r1", "real", "real", 0.1)]
    m1 = compute_classification_metrics(rects1)
    assert m1["auroc"] is None

    # None scores
    rects2 = [
        prediction("r1", "real", "real", None),
        prediction("f1", "fake", "fake", None),
    ]
    m2 = compute_classification_metrics(rects2)
    assert m2["auroc"] is None


def test_compute_metrics_by_source():
    records = [
        prediction("r1", "real", "real", 0.1, source="A"),
        prediction("f1", "fake", "fake", 0.9, source="A"),
        prediction("r2", "real", "fake", 0.8, source="B"),  # fp
    ]
    df = compute_metrics_by_source(records)
    assert len(df) == 2
    assert list(df.columns) == ["source", "n"] + list(METRIC_NAMES)
    assert list(df["source"]) == ["A", "B"]

    # A perfect
    assert df.loc[df["source"] == "A", "accuracy"].iloc[0] == 1.0

    # B fp -> precision 0.0, recall None
    b_prec = df.loc[df["source"] == "B", "precision"].iloc[0]
    assert b_prec == 0.0


def test_write_metrics_outputs(tmp_path):
    records = [
        prediction("r1", "real", "real", 0.1),
        prediction("f1", "fake", "fake", 0.9),
    ]
    j, c = write_metrics_outputs(records, tmp_path)
    assert j.exists()
    assert c.exists()

    d = json.loads(j.read_text())
    assert d["total_records"] == 2
    assert d["overall"]["accuracy"] == 1.0


def test_evaluate_prediction_file(tmp_path):
    p_file = tmp_path / "predictions.jsonl"
    r = prediction("r1", "real", "real", 0.1).model_dump(mode="json")
    p_file.write_text(json.dumps(r) + "\n")

    eval_j, eval_c = evaluate_prediction_file(p_file)
    assert eval_j.exists()
    assert eval_j.parent == tmp_path


def test_evaluate_prediction_file_fails_validation(tmp_path):
    p_file = tmp_path / "predictions.jsonl"
    r1 = prediction("r1", "real", "real", 0.1).model_dump(mode="json")
    r2 = prediction("r1", "real", "real", 0.1).model_dump(
        mode="json"
    )  # duplicate sample_id
    p_file.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")

    with pytest.raises(MetricsError, match="Prediction validation failed"):
        evaluate_prediction_file(p_file)


def test_discover_prediction_files(tmp_path):
    d1 = tmp_path / "run1"
    d1.mkdir()
    d2 = tmp_path / "run2"
    d2.mkdir()

    f1 = d1 / "predictions.jsonl"
    f1.touch()
    f2 = d2 / "predictions.jsonl"
    f2.touch()
    d1.joinpath("other.jsonl").touch()

    found = discover_prediction_files(tmp_path)
    assert len(found) == 2
    assert f1 in found
    assert f2 in found


def test_evaluate_prediction_file_fails_on_missing_mllm_fields(tmp_path):
    p_file = tmp_path / "predictions.jsonl"
    rec = prediction("q1", "fake", "fake", 0.9, model_name="qwen_vl")
    p_file.write_text(json.dumps(rec.model_dump(mode="json")) + "\n")

    with pytest.raises(MetricsError, match="Prediction validation failed"):
        evaluate_prediction_file(p_file)


def _write_tmp_config(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal valid smoke config inside tmp_path and return (config, output_root)."""
    # find_repo_root() needs a pyproject.toml marker above the config file.
    (tmp_path / "pyproject.toml").touch()
    out_root = tmp_path / "outputs"
    out_root.mkdir()

    config_text = f"""
project:
  name: ai-image-forensics
  phase: phase_ab_smoke
  description: Temporary config for Task 5 CLI tests.

paths:
  data_root: {tmp_path / "data"}
  manifest_root: {tmp_path / "manifests"}
  cache_root: {tmp_path / "cache"}
  output_root: {out_root}
  external_root: {tmp_path / "external"}

runtime:
  python: "3.10"
  seed: 70
  device: cpu
  batch_size: 2
  num_workers: 0
  fail_fast: false

datasets:
  tiny_genimage:
    enabled: false
    source: tmp
    use_original_split: false
    train_manifest: {tmp_path / "m_train.csv"}
    dev_manifest: {tmp_path / "m_dev.csv"}
  genimage_unseen:
    enabled: false
    preferred_generator: midjourney
    fallback_generators:
      - adm
    max_images: 8
    balance_labels: true
    split: external
    manifest: {tmp_path / "m_ext.csv"}
  synthbuster:
    enabled: false
    max_images: 8
    balance_labels: true
    split: external
    manifest: {tmp_path / "m_sb.csv"}

baselines:
  clip_probe:
    enabled: true
    model_family: synthetic
    model_name: smoke-embedding
    pretrained: none
    classifier: logistic_regression
    seeds:
      - 70
    cache_embeddings: false
  qwen_vl:
    enabled: false
    model_id: Qwen/Qwen2.5-VL-3B-Instruct
    prompt_id: qwen_json_v1
    temperature: 0.0
    max_new_tokens: 128
    cache_outputs: false
    allow_deferred: true
  assisted_qwen:
    enabled: false
    base_model_id: Qwen/Qwen2.5-VL-3B-Instruct
    prompt_id: assisted_qwen_json_v1
    assistant_source: clip_probe
    include_classifier_pred: true
    include_fake_probability: true
    temperature: 0.0
    max_new_tokens: 128
    cache_outputs: false
    allow_deferred: true
  npr:
    enabled: false
    repo_url: https://github.com/chuangchuangtan/NPR-DeepfakeDetection
    repo_commit: smoke-disabled
    checkpoint_path: {tmp_path / "NPR.pth"}
    checkpoint_sha256: smoke-disabled
    batch_size: 2
    allow_deferred: true

evaluation:
  labels:
    negative: real
    positive: fake
  metrics:
    - accuracy
    - balanced_accuracy
    - precision
    - recall
    - f1
    - auroc
  group_by:
    - source
    - split

report:
  filename: phase_ab_smoke_report.md
  include_failure_notes: true
  include_explanations_sample: false
  explanation_sample_size: 0
"""
    cfg_path = tmp_path / "phase_ab_smoke.yaml"
    cfg_path.write_text(config_text, encoding="utf-8")
    return cfg_path, out_root


def test_cli_evaluate_no_prediction_files_returns_zero(tmp_path, capsys):
    cfg_path, _out_root = _write_tmp_config(tmp_path)

    exit_code = main(["evaluate", "--config", str(cfg_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "prediction_files=0" in captured.out


def test_cli_evaluate_writes_metrics_for_existing_predictions(tmp_path, capsys):
    cfg_path, out_root = _write_tmp_config(tmp_path)

    run_dir = out_root / "run1"
    run_dir.mkdir(parents=True)
    p_file = run_dir / "predictions.jsonl"
    records = [
        prediction("r1", "real", "real", 0.1),
        prediction("f1", "fake", "fake", 0.9),
    ]
    from aiforensics.schemas.predictions import write_predictions

    write_predictions(records, p_file)

    exit_code = main(["evaluate", "--config", str(cfg_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "prediction_files=1" in captured.out

    metrics_json = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics_by_source.csv"
    assert metrics_json.exists()
    assert metrics_csv.exists()

    data = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert data["total_records"] == 2
    assert data["overall"]["accuracy"] == 1.0
