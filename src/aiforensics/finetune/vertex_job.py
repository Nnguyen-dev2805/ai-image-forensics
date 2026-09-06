"""Vertex AI CustomJob spec for Qwen2.5-VL LoRA fine-tuning.

Builds the REST payload for ``projects.locations.customJobs.create``. Training
reads the label-only JSONL from GCS and writes LoRA checkpoints back to GCS,
so the job itself stays stateless and resumable.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DEFAULT_MODEL_ID", "build_vertex_custom_job_spec", "default_image_uri"]

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Checkpoint policy from the design spec: save every 100 steps, keep the
# latest three checkpoints, and rely on Trainer resume when a job restarts.
CHECKPOINT_ARGS = ("--save_strategy=steps", "--save_steps=100", "--save_total_limit=3")


def default_image_uri(project_id: str, location: str) -> str:
    """Artifact Registry image the training container (Task 5) will be pushed to."""
    return f"{location}-docker.pkg.dev/{project_id}/aiforensics/qwen-ft-trainer:latest"


def build_vertex_custom_job_spec(
    *,
    project_id: str,
    location: str,
    bucket_uri: str,
    protocol: str,
    training_jsonl_uri: str,
    machine_type: str = "a2-highgpu-1g",
    accelerator_type: str = "NVIDIA_TESLA_A100",
    accelerator_count: int = 1,
    model_id: str = DEFAULT_MODEL_ID,
    image_uri: str | None = None,
    learning_rate: str = "2e-4",
) -> dict[str, Any]:
    """Build the CustomJob REST payload for one LoRA fine-tuning protocol."""
    for name, value in {
        "project_id": project_id,
        "location": location,
        "protocol": protocol,
    }.items():
        if not value:
            raise ValueError(f"{name} must not be empty")
    for name, value in {
        "bucket_uri": bucket_uri,
        "training_jsonl_uri": training_jsonl_uri,
    }.items():
        if not value.startswith("gs://"):
            raise ValueError(f"{name} must be a gs:// URI, got: {value}")
    if accelerator_count < 1:
        raise ValueError("accelerator_count must be at least 1")

    checkpoint_prefix = f"{bucket_uri.rstrip('/')}/checkpoints/{protocol}"
    resolved_image_uri = image_uri or default_image_uri(project_id, location)

    training_args = [
        f"--model_name_or_path={model_id}",
        f"--data_path={training_jsonl_uri}",
        f"--output_dir={checkpoint_prefix}",
        # LoRA on the LLM and MLP projector; the vision tower stays frozen so
        # the CLIP-pretrained visual encoder is preserved.
        "--lora_enable=True",
        "--tune_mm_llm=True",
        "--tune_mm_vision=False",
        "--tune_mm_mlp=True",
        "--num_train_epochs=1",
        "--per_device_train_batch_size=1",
        "--gradient_accumulation_steps=16",
        f"--learning_rate={learning_rate}",
        "--logging_steps=10",
        *CHECKPOINT_ARGS,
    ]

    return {
        "displayName": f"aiforensics-qwen-ft-{protocol}",
        "jobSpec": {
            "baseOutputDirectory": {"outputUriPrefix": checkpoint_prefix},
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": machine_type,
                        "acceleratorType": accelerator_type,
                        "acceleratorCount": accelerator_count,
                    },
                    "replicaCount": "1",
                    "containerSpec": {
                        "imageUri": resolved_image_uri,
                        "args": training_args,
                    },
                }
            ],
        },
    }
