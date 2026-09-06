"""Tests for the qwen_ft training container entrypoint wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra.qwen_ft.train_qwen_ft import (
    build_parser,
    build_upstream_command,
    gcs_uri_to_local_path,
    has_final_adapter,
    is_gcs_uri,
    rewrite_jsonl_image_uris,
    write_qwen_dataset_registry,
)


def test_training_entrypoint_parses_required_args() -> None:
    args = build_parser().parse_args(
        [
            "--model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct",
            "--data_path=gs://bucket/data/qwen_train.jsonl",
            "--output_dir=gs://bucket/checkpoints/protocol-a-small",
            "--lora_enable=True",
        ]
    )

    assert args.model_name_or_path == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert args.data_path == "gs://bucket/data/qwen_train.jsonl"
    assert args.output_dir == "gs://bucket/checkpoints/protocol-a-small"
    assert args.lora_enable == "True"


def test_training_entrypoint_defaults_match_vertex_job_payload() -> None:
    """Upstream flags default to the same values the Task 4 spec builder sends."""
    args = build_parser().parse_args(
        [
            "--model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct",
            "--data_path=gs://bucket/data/qwen_train.jsonl",
            "--output_dir=gs://bucket/checkpoints/protocol-a-small",
        ]
    )

    assert args.lora_enable == "True"
    assert args.tune_mm_llm == "True"
    assert args.tune_mm_vision == "False"
    assert args.tune_mm_mlp == "True"
    assert args.num_train_epochs == "1"
    assert args.per_device_train_batch_size == "1"
    assert args.gradient_accumulation_steps == "16"
    assert args.learning_rate == "2e-4"
    assert args.save_strategy == "steps"
    assert args.save_steps == "100"
    assert args.save_total_limit == "3"
    assert args.logging_steps == "10"
    assert args.method == "lora"
    assert args.max_pixels == 200704
    assert args.min_pixels == 50176
    assert args.bf16 == "True"
    assert args.gradient_checkpointing == "True"


def test_build_upstream_command_rewrites_gcs_paths_and_drops_wrapper_args(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        [
            "--model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct",
            "--data_path=gs://bucket/data/qwen_train.jsonl",
            "--output_dir=gs://bucket/checkpoints/protocol-a-small",
            "--work_dir",
            str(tmp_path),
        ]
    )
    local_data = tmp_path / "data" / "qwen_train_local.jsonl"
    local_output = tmp_path / "output"

    command = build_upstream_command(args, local_data, local_output)

    assert "--dataset_use=qwen_ft" in command
    assert not any(part.startswith("--data_path=") for part in command)
    assert "--output_dir=" + str(local_output) in command
    assert "--lora_enable=True" in command
    assert "--max_pixels=200704" in command
    assert "--min_pixels=50176" in command
    assert "--bf16=True" in command
    assert "--gradient_checkpointing=True" in command
    assert not any(part.startswith("--training_script=") for part in command)
    assert not any(part.startswith("--work_dir=") for part in command)
    assert "gs://" not in " ".join(command)


def test_gcs_uri_helpers() -> None:
    assert is_gcs_uri("gs://bucket/obj.jsonl") is True
    assert is_gcs_uri("/local/obj.jsonl") is False
    assert gcs_uri_to_local_path("gs://bucket/dir/obj.jsonl", Path("/work")) == (
        Path("/work") / "bucket" / "dir" / "obj.jsonl"
    )


def test_rewrite_jsonl_image_uris_maps_gcs_to_downloaded_local_files(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "data"
    image_local = local_root / "bucket" / "data" / "train" / "g1" / "ai" / "0001.png"
    image_local.parent.mkdir(parents=True)
    image_local.write_bytes(b"fake-bytes")

    jsonl_path = tmp_path / "qwen_train.jsonl"
    rows = [
        {
            "image": "gs://bucket/data/train/g1/ai/0001.png",
            "conversations": [
                {"from": "human", "value": "prompt"},
                {"from": "gpt", "value": '{"label":"fake"}'},
            ],
        },
        {"image": "/already/local.png", "conversations": []},
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    rewritten = rewrite_jsonl_image_uris(jsonl_path, local_root)

    new_rows = [json.loads(line) for line in rewritten.read_text(encoding="utf-8").splitlines()]
    assert new_rows[0]["image"] == str(image_local)
    assert new_rows[1]["image"] == "/already/local.png"
    # The original JSONL stays untouched for provenance.
    assert json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])["image"] == (
        "gs://bucket/data/train/g1/ai/0001.png"
    )


def test_rewrite_jsonl_image_uris_raises_on_missing_download(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "qwen_train.jsonl"
    row = {"image": "gs://bucket/missing.png", "conversations": []}
    jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.png"):
        rewrite_jsonl_image_uris(jsonl_path, tmp_path / "data")


def test_has_final_adapter_requires_marker_file(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert has_final_adapter(empty_dir) is False

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    # Config alone is not sufficient; weights must also exist
    assert has_final_adapter(adapter_dir) is False

    # Both config and weights are required
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"lora-weights")
    assert has_final_adapter(adapter_dir) is True


def test_training_entrypoint_accepts_local_modal_args() -> None:
    args = build_parser().parse_args(
        [
            "--model-id",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--train-jsonl",
            "/vol/data/protocol-a-small/qwen_train_abs.jsonl",
            "--output-dir",
            "/vol/checkpoints/protocol-a-small",
            "--epochs",
            "1",
            "--learning-rate",
            "0.0002",
            "--method",
            "lora",
        ]
    )

    assert args.train_jsonl == "/vol/data/protocol-a-small/qwen_train_abs.jsonl"
    assert args.output_dir == "/vol/checkpoints/protocol-a-small"
    assert args.method == "lora"


def test_build_upstream_command_preserves_data_path_when_using_train_jsonl(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--model-id",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--train-jsonl",
            str(tmp_path / "train.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    command = build_upstream_command(args)
    assert "--dataset_use=qwen_ft" in command
    assert not any(part.startswith("--data_path=") for part in command)
    assert f"--output_dir={tmp_path / 'out'}" in command
    assert "--model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct" in command


def test_build_upstream_command_includes_lora_training_args(tmp_path: Path) -> None:
    command = build_upstream_command(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        train_jsonl=tmp_path / "train.jsonl",
        output_dir=tmp_path / "out",
        epochs=1.0,
        learning_rate=2e-4,
        method="lora",
        save_steps=100,
        save_total_limit=3,
    )

    text = " ".join(str(part) for part in command)
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in text
    assert "--dataset_use=qwen_ft" in text
    assert str(tmp_path / "train.jsonl") not in text
    assert "--lora_enable" in text or "lora" in text.lower()
    assert "--max_pixels=200704" in text
    assert "--min_pixels=50176" in text
    assert "--bf16=True" in text
    assert "--gradient_checkpointing=True" in text


def test_build_upstream_command_supports_custom_pixels_and_qlora(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct",
            "--train-jsonl",
            str(tmp_path / "train.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--max_pixels=100352",
            "--min_pixels=25088",
            "--method=qlora",
        ]
    )
    command = build_upstream_command(args)
    assert "--max_pixels=100352" in command
    assert "--min_pixels=25088" in command
    assert "--qlora=True" in command
    assert not any(part.startswith("--method=") for part in command)


def test_build_upstream_command_kwargs_pixels_and_qlora(tmp_path: Path) -> None:
    command = build_upstream_command(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        train_jsonl=tmp_path / "train.jsonl",
        output_dir=tmp_path / "out",
        max_pixels=100352,
        min_pixels=25088,
        method="qlora",
    )
    text = " ".join(command)
    assert "--max_pixels=100352" in text
    assert "--min_pixels=25088" in text
    assert "--qlora=True" in text


def test_write_qwen_dataset_registry_points_to_absolute_jsonl(tmp_path: Path) -> None:
    script = tmp_path / "qwen-vl-finetune" / "qwenvl" / "train" / "train_qwen.py"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    data_init = script.parents[1] / "data" / "__init__.py"
    data_init.parent.mkdir(parents=True)
    data_init.write_text("data_dict = {}\n", encoding="utf-8")

    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text("{}\n", encoding="utf-8")

    write_qwen_dataset_registry(script, train_jsonl, dataset_name="qwen_ft")

    source = data_init.read_text(encoding="utf-8")
    assert '"annotation_path": ' in source
    assert str(train_jsonl) in source
    assert '"data_path": ""' in source
    assert 'data_dict["qwen_ft"] = QWEN_FT_DATASET' in source


def test_write_qwen_dataset_registry_registers_val_and_patches_processor(tmp_path: Path) -> None:
    script = tmp_path / "qwen-vl-finetune" / "qwenvl" / "train" / "train_qwen.py"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    data_dir = script.parents[1] / "data"
    data_dir.mkdir(parents=True)
    data_init = data_dir / "__init__.py"
    data_init.write_text("data_dict = {}\n", encoding="utf-8")

    processor_file = data_dir / "data_processor.py"
    processor_file.write_text(
        "def make_supervised_data_module(processor, data_args):\n"
        "    train_dataset = LazySupervisedDataset(processor, data_args=data_args)\n"
        "    data_collator = DataCollator(processor.tokenizer)\n"
        "    return dict(\n"
        "        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator\n"
        "    )\n",
        encoding="utf-8",
    )

    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text("{}\n", encoding="utf-8")
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text("{}\n", encoding="utf-8")

    write_qwen_dataset_registry(script, train_jsonl, val_jsonl=val_jsonl)

    source = data_init.read_text(encoding="utf-8")
    assert 'data_dict["qwen_ft"] = QWEN_FT_DATASET' in source
    assert 'data_dict["qwen_ft_val"] = QWEN_FT_VAL_DATASET' in source
    assert str(val_jsonl) in source

    processor_code = processor_file.read_text(encoding="utf-8")
    assert "qwen_ft_val" in processor_code
    assert "eval_dataset=eval_dataset" in processor_code


def test_build_upstream_command_configures_eval_flags_when_val_jsonl_given(tmp_path: Path) -> None:
    # 1. Via args
    args = build_parser().parse_args(
        [
            "--model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct",
            "--train-jsonl=/tmp/train.jsonl",
            "--val-jsonl=/tmp/val.jsonl",
            "--output_dir=/tmp/out",
        ]
    )
    cmd = build_upstream_command(args, Path("/tmp/train.jsonl"), Path("/tmp/out"))
    cmd_text = " ".join(cmd)
    assert "--eval_strategy=steps" in cmd_text
    assert "--eval_steps=25" in cmd_text
    assert "--save_strategy=steps" in cmd_text
    assert "--save_steps=25" in cmd_text
    assert "--load_best_model_at_end=True" in cmd_text
    assert "--metric_for_best_model=loss" in cmd_text
    assert "--per_device_eval_batch_size=1" in cmd_text
    assert "--eval_accumulation_steps=1" in cmd_text
    assert "--prediction_loss_only=True" in cmd_text
    assert "--val_jsonl" not in cmd_text  # Should not forward wrapper-only flag

    # 2. Via kwargs
    cmd2 = build_upstream_command(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        train_jsonl="/tmp/train.jsonl",
        val_jsonl="/tmp/val.jsonl",
        output_dir="/tmp/out",
        eval_steps=50,
    )
    cmd2_text = " ".join(cmd2)
    assert "--eval_strategy=steps" in cmd2_text
    assert "--eval_steps=50" in cmd2_text
    assert "--save_steps=50" in cmd2_text
    assert "--load_best_model_at_end=True" in cmd2_text
    assert "--metric_for_best_model=loss" in cmd2_text
    assert "--per_device_eval_batch_size=1" in cmd2_text
    assert "--eval_accumulation_steps=1" in cmd2_text
    assert "--prediction_loss_only=True" in cmd2_text
