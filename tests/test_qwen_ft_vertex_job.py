"""Tests for the Vertex AI CustomJob spec builder for qwen_ft training."""

from __future__ import annotations

import pytest

from aiforensics.finetune.vertex_job import build_vertex_custom_job_spec

BUCKET = "gs://aiforensics-qwen-ft-579187260419"
TRAINING_JSONL = f"{BUCKET}/data/protocol-a-small/qwen_train.jsonl"


def _build(**overrides):
    kwargs = {
        "project_id": "579187260419",
        "location": "asia-southeast1",
        "bucket_uri": BUCKET,
        "protocol": "protocol-a-small",
        "training_jsonl_uri": TRAINING_JSONL,
    }
    kwargs.update(overrides)
    return build_vertex_custom_job_spec(**kwargs)


def test_vertex_job_spec_uses_a100_and_gcs_checkpoints() -> None:
    spec = _build(
        machine_type="a2-highgpu-1g",
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
    )

    worker = spec["jobSpec"]["workerPoolSpecs"][0]
    assert worker["machineSpec"]["acceleratorType"] == "NVIDIA_TESLA_A100"
    assert worker["machineSpec"]["acceleratorCount"] == 1
    assert spec["jobSpec"]["baseOutputDirectory"]["outputUriPrefix"] == (
        f"{BUCKET}/checkpoints/protocol-a-small"
    )
    args = worker["containerSpec"]["args"]
    assert "--save_steps=100" in args
    assert "--save_total_limit=3" in args


def test_vertex_job_spec_lora_flags_match_qwen_finetune_framework() -> None:
    spec = _build()

    args = spec["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["args"]
    assert "--lora_enable=True" in args
    assert "--tune_mm_llm=True" in args
    assert "--tune_mm_vision=False" in args
    assert "--tune_mm_mlp=True" in args
    assert f"--data_path={TRAINING_JSONL}" in args
    assert f"--output_dir={BUCKET}/checkpoints/protocol-a-small" in args
    assert "--model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct" in args
    assert "--num_train_epochs=1" in args
    assert "--per_device_train_batch_size=1" in args
    assert "--gradient_accumulation_steps=16" in args
    assert "--learning_rate=2e-4" in args


def test_vertex_job_spec_display_name_and_single_replica() -> None:
    spec = _build()

    assert spec["displayName"] == "aiforensics-qwen-ft-protocol-a-small"
    worker = spec["jobSpec"]["workerPoolSpecs"][0]
    assert worker["replicaCount"] == "1"
    assert len(spec["jobSpec"]["workerPoolSpecs"]) == 1


def test_vertex_job_spec_defaults_to_project_artifact_registry_image() -> None:
    spec = _build()

    image_uri = spec["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["imageUri"]
    assert image_uri == (
        "asia-southeast1-docker.pkg.dev/579187260419/aiforensics/qwen-ft-trainer:latest"
    )


def test_vertex_job_spec_accepts_explicit_image_uri() -> None:
    spec = _build(image_uri="us-central1-docker.pkg.dev/proj/repo/img:v2")

    image_uri = spec["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["imageUri"]
    assert image_uri == "us-central1-docker.pkg.dev/proj/repo/img:v2"


def test_vertex_job_spec_rejects_non_gcs_bucket() -> None:
    with pytest.raises(ValueError, match="gs://"):
        _build(bucket_uri="s3://not-gcs")


def test_vertex_job_spec_rejects_non_gcs_training_jsonl() -> None:
    with pytest.raises(ValueError, match="gs://"):
        _build(training_jsonl_uri="/local/qwen_train.jsonl")


def test_vertex_job_spec_rejects_empty_protocol() -> None:
    with pytest.raises(ValueError, match="protocol"):
        _build(protocol="")


def test_vertex_job_spec_rejects_zero_accelerators() -> None:
    with pytest.raises(ValueError, match="accelerator_count"):
        _build(accelerator_count=0)
