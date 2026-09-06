from __future__ import annotations

from pathlib import Path


def test_modal_app_declares_expected_functions() -> None:
    source = Path("infra/modal/qwen_ft_modal.py").read_text(encoding="utf-8")

    assert 'modal.App("aiforensics-qwen-ft")' in source
    assert 'modal.Volume.from_name("aiforensics-qwen-ft")' in source
    for name in [
        "inspect_volume",
        "unpack_archive",
        "prepare_from_raw_archive",
        "train_lora",
        "eval_small",
        "eval_full",
    ]:
        assert f"def {name}(" in source


def test_modal_app_does_not_require_gcp_secret() -> None:
    source = Path("infra/modal/qwen_ft_modal.py").read_text(encoding="utf-8")

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in source
    assert "gcloud" not in source
    assert "huggingface-secret" in source


def test_modal_app_mounts_repo_and_clones_qwen() -> None:
    source = Path("infra/modal/qwen_ft_modal.py").read_text(encoding="utf-8")

    assert "nvidia/cuda:12.4.1-devel-ubuntu22.04" in source
    assert "torch==2.5.1 torchvision==0.20.1" in source
    assert "https://download.pytorch.org/whl/cu124" in source
    assert "flash-attn" in source
    assert "--no-build-isolation" in source
    assert "git clone --depth 1" in source
    assert "https://github.com/QwenLM/Qwen2.5-VL.git /opt/qwen-vl-finetune" in source
    assert "add_local_dir" in source
    assert 'remote_path="/repo"' in source
    assert "pip install --no-cache-dir -e /repo" in source
    assert "parents[2]" not in source


def test_modal_train_lora_checks_both_config_and_weights() -> None:
    source = Path("infra/modal/qwen_ft_modal.py").read_text(encoding="utf-8")

    assert 'final_adapter / "adapter_config.json"' in source
    assert "adapter_model.safetensors" in source


def test_modal_prepare_from_raw_archive_keeps_raw_extract_off_volume() -> None:
    source = Path("infra/modal/qwen_ft_modal.py").read_text(encoding="utf-8")

    assert "def prepare_from_raw_archive(" in source
    assert "tempfile.TemporaryDirectory" in source
    assert "find_genimage_data_root" in source
    assert "write_export_archive" in source
    assert "zf.extractall(raw_extract_root)" in source
    assert 'zf.extractall(VOL_ROOT / "data")' in source


def test_modal_train_lora_forwards_pixel_bounds_and_method() -> None:
    source = Path("infra/modal/qwen_ft_modal.py").read_text(encoding="utf-8")

    assert "max_pixels: int = 200704" in source
    assert "min_pixels: int = 50176" in source
    assert 'method: str = "lora"' in source
    assert '"--max_pixels"' in source
    assert '"--min_pixels"' in source
    assert '"--method"' in source
    assert '"max_pixels": max_pixels' in source
    assert '"min_pixels": min_pixels' in source
    assert '"method": method' in source
    assert "max_pixels=max_pixels" in source
    assert "min_pixels=min_pixels" in source
    assert "method=method" in source
    assert "PYTORCH_CUDA_ALLOC_CONF" in source
    assert "bf16: bool = True" in source
    assert "gradient_checkpointing: bool = True" in source
    assert '"--bf16"' in source
    assert '"--gradient_checkpointing"' in source


def test_modal_train_lora_forwards_eval_memory_flags_and_cleans_checkpoints() -> None:
    source = Path("infra/modal/qwen_ft_modal.py").read_text(encoding="utf-8")

    assert "per_device_eval_batch_size: int = 1" in source
    assert "eval_accumulation_steps: int = 1" in source
    assert "prediction_loss_only: bool = True" in source
    assert '"--per-device-eval-batch-size"' in source
    assert '"--eval-accumulation-steps"' in source
    assert '"--prediction-loss-only"' in source
    assert "clean_checkpoints: bool = True" in source
