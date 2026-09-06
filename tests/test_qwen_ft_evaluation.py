"""Tests for label-only qwen_ft evaluation, validation, and reporting."""

from __future__ import annotations

import json
from pathlib import Path

from aiforensics.config.models import (
    AppConfig,
    AssistedQwenConfig,
    BaselinesConfig,
    ClipProbeConfig,
    DatasetsConfig,
    EvaluationConfig,
    GenImageUnseenConfig,
    LabelsConfig,
    NPRConfig,
    PathsConfig,
    ProjectConfig,
    QwenFTConfig,
    QwenVLConfig,
    ReportConfig,
    RuntimeConfig,
    SynthbusterConfig,
    TinyGenImageConfig,
)
from aiforensics.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    write_metrics_outputs,
)
from aiforensics.reporting import markdown as md
from aiforensics.reporting.markdown import RunSummary, build_phase_ab_report
from aiforensics.schemas.predictions import PredictionRecord, validate_predictions


def _qwen_ft_record(sample_id: str, label_true: str, label_pred: str) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        label_true=label_true,  # type: ignore[arg-type]
        label_pred=label_pred,  # type: ignore[arg-type]
        score_fake=None,
        model_name="qwen_ft",
        source="g",
    )


def test_label_only_qwen_ft_skips_auroc_but_keeps_classification_metrics() -> None:
    records = [
        _qwen_ft_record("a", "fake", "fake"),
        _qwen_ft_record("b", "real", "real"),
        _qwen_ft_record("c", "fake", "real"),
        _qwen_ft_record("d", "real", "fake"),
    ]

    metrics = compute_classification_metrics(records)

    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["auroc"] is None


def test_confusion_matrix_label_only() -> None:
    records = [
        _qwen_ft_record("a", "fake", "fake"),
        _qwen_ft_record("b", "real", "real"),
        _qwen_ft_record("c", "fake", "real"),
        _qwen_ft_record("d", "real", "fake"),
    ]

    result = compute_confusion_matrix(records)

    assert result["labels"] == ["real", "fake"]
    # [[tn, fp], [fn, tp]]
    assert result["matrix"] == [[1, 1], [1, 1]]


def test_write_metrics_outputs_writes_confusion_matrix(tmp_path: Path) -> None:
    records = [
        _qwen_ft_record("a", "fake", "fake"),
        _qwen_ft_record("b", "real", "real"),
    ]

    json_path, _csv_path = write_metrics_outputs(records, tmp_path)

    cm_path = json_path.parent / "confusion_matrix.json"
    assert cm_path.exists()
    payload = json.loads(cm_path.read_text(encoding="utf-8"))
    assert payload == {"labels": ["real", "fake"], "matrix": [[1, 0], [0, 1]]}


def test_validate_predictions_accepts_label_only_qwen_ft_records() -> None:
    record = PredictionRecord(
        sample_id="a",
        label_true="fake",
        label_pred="fake",
        score_fake=None,
        model_name="qwen_ft",
        source="g",
        prompt_id="qwen_ft_label_json_v1",
        raw_output='{"label":"fake"}',
        explanation="",
        parse_status="parsed",
    )

    result = validate_predictions([record])

    assert result.is_valid, result.errors


def test_validate_predictions_requires_mllm_fields_for_qwen_ft() -> None:
    record = PredictionRecord(
        sample_id="a",
        label_true="fake",
        label_pred="fake",
        score_fake=None,
        model_name="qwen_ft",
        source="g",
        parse_status="parsed",
    )

    # The evaluate path validates with MLLM enforcement on, so a qwen_ft
    # record missing its raw output cannot slip through.
    result = validate_predictions([record], require_mllm_fields=True)

    assert not result.is_valid
    assert any("raw_output" in error for error in result.errors)


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project=ProjectConfig(name="t", phase="phase_ab", description="d"),
        paths=PathsConfig(
            data_root=tmp_path / "data",
            manifest_root=tmp_path / "manifests",
            cache_root=tmp_path / ".cache",
            output_root=tmp_path / "outputs",
            external_root=tmp_path / "external",
        ),
        runtime=RuntimeConfig(
            python="3.10", seed=70, device="cpu", batch_size=1, num_workers=0, fail_fast=False
        ),
        datasets=DatasetsConfig(
            tiny_genimage=TinyGenImageConfig(
                enabled=False,
                source="s",
                use_original_split=False,
                train_manifest=tmp_path / "train.csv",
                dev_manifest=tmp_path / "dev.csv",
            ),
            genimage_unseen=GenImageUnseenConfig(
                enabled=False,
                generators=[],
                max_images=0,
                balance_labels=True,
                split="external",
                manifest=tmp_path / "ext.csv",
            ),
            synthbuster=SynthbusterConfig(
                enabled=False,
                max_images=0,
                balance_labels=True,
                split="external",
                manifest=tmp_path / "sb.csv",
            ),
        ),
        baselines=BaselinesConfig(
            clip_probe=ClipProbeConfig(
                enabled=False,
                model_family="synthetic",
                model_name="m",
                pretrained="none",
                classifier="lr",
                seeds=[70],
                cache_embeddings=False,
            ),
            qwen_vl=QwenVLConfig(
                enabled=False,
                model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                prompt_id="qwen_json_v1",
                temperature=0.0,
                max_new_tokens=32,
                cache_outputs=False,
            ),
            assisted_qwen=AssistedQwenConfig(
                enabled=False,
                base_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                prompt_id="assisted_qwen_json_v1",
                assistant_source="clip_probe",
                include_classifier_pred=True,
                include_fake_probability=True,
                temperature=0.0,
                max_new_tokens=32,
                cache_outputs=False,
            ),
            npr=NPRConfig(
                enabled=False,
                repo_url="https://github.com/example/NPR",
                repo_commit=None,
                checkpoint_path=tmp_path / "NPR.pth",
                checkpoint_sha256=None,
                batch_size=1,
                allow_deferred=True,
            ),
            qwen_ft=QwenFTConfig(
                enabled=True,
                adapter_uri="gs://aiforensics-qwen-ft-579187260419/checkpoints/p/final_adapter",
            ),
        ),
        evaluation=EvaluationConfig(
            labels=LabelsConfig(negative="real", positive="fake"),
            metrics=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc"],
            group_by=["source"],
        ),
        report=ReportConfig(
            filename="r.md",
            include_failure_notes=True,
            include_explanations_sample=False,
            explanation_sample_size=0,
        ),
    )


def _qwen_ft_summary(run_dir: Path) -> RunSummary:
    overall = {
        "accuracy": 0.75,
        "balanced_accuracy": 0.5,
        "precision": None,
        "recall": 0.0,
        "f1": None,
        "auroc": None,
    }
    return RunSummary(
        baseline="qwen_ft",
        seed=None,
        run_id="001_qwen_ft",
        status="completed",
        reason=None,
        run_dir=run_dir,
        started_at=None,
        ended_at=None,
        total_records=4,
        overall=overall,
        by_source=(),
        prediction_path=run_dir / "predictions.jsonl",
    )


def test_report_renders_qwen_ft_rows_and_label_only_note(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = _qwen_ft_summary(tmp_path / "outputs" / "001_qwen_ft")

    text = build_phase_ab_report(config, [run])

    assert "| qwen_ft |" in text
    assert (
        "qwen_ft is label-only in this phase; AUROC is skipped unless a real "
        "fake score is available." in text
    )
    # AUROC renders as N/A, not 0.
    assert "0.5000 | N/A" in text or "0.5000| N/A" in text


def test_report_lists_confusion_matrix_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "outputs" / "001_qwen_ft"
    run_dir.mkdir(parents=True)
    (run_dir / "confusion_matrix.json").write_text(
        json.dumps({"labels": ["real", "fake"], "matrix": [[1, 1], [1, 1]]}), encoding="utf-8"
    )
    run = _qwen_ft_summary(run_dir)

    text = build_phase_ab_report(config, [run])

    assert "Confusion matrix artifacts" in text
    assert "confusion_matrix.json" in text


def test_qwen_ft_slot_present_in_expected_slots(tmp_path: Path) -> None:
    config = _config(tmp_path)

    slots = md._expected_slots(config)

    assert slots[-1] == ("qwen_ft", None)
