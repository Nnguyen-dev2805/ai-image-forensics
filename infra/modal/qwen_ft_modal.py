from __future__ import annotations

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

APP_NAME = "aiforensics-qwen-ft"
VOLUME_NAME = "aiforensics-qwen-ft"
VOL_ROOT = Path("/vol")

app = modal.App("aiforensics-qwen-ft")
volume = modal.Volume.from_name("aiforensics-qwen-ft")


def _find_repo_root() -> Path:
    """Find the local repo root without assuming Modal preserves file depth."""
    candidates = [Path.cwd(), Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    modal_repo = Path("/repo")
    if (modal_repo / "pyproject.toml").is_file():
        return modal_repo
    raise RuntimeError("Could not find ai-image-forensics repo root")


REPO_ROOT = _find_repo_root()

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .apt_install("git", "ffmpeg", "build-essential")
    .run_commands(
        "python -m pip install --no-cache-dir "
        "torch==2.5.1 torchvision==0.20.1 "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "accelerate",
        "bitsandbytes",
        "datasets",
        "einops",
        "ninja",
        "packaging",
        "peft",
        "pillow",
        "pydantic",
        "pyyaml",
        "qwen-vl-utils",
        "rich",
        "scikit-learn",
        "tqdm",
        "transformers",
        "typer",
        "wheel",
    )
    .pip_install(
        "flash-attn",
        extra_options="--no-build-isolation",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/QwenLM/Qwen2.5-VL.git /opt/qwen-vl-finetune"
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        copy=True,
        ignore=[
            ".venv",
            ".git",
            ".codegraph",
            "outputs",
            ".cache",
            "external",
            "__pycache__",
            "*.zip",
            ".pytest_cache",
            ".ruff_cache",
        ],
    )
    .workdir("/repo")
    .run_commands("pip install --no-cache-dir -e /repo")
)


@app.function(image=image, volumes={"/vol": volume}, timeout=300)
def inspect_volume() -> dict[str, object]:
    roots = ["archives", "data", "checkpoints", "eval", "cache"]
    return {
        "volume": VOLUME_NAME,
        "exists": VOL_ROOT.exists(),
        "roots": {name: (VOL_ROOT / name).exists() for name in roots},
        "archives": (
            sorted(p.name for p in (VOL_ROOT / "archives").glob("*.zip"))
            if (VOL_ROOT / "archives").exists()
            else []
        ),
    }


@app.function(image=image, volumes={"/vol": volume}, timeout=1800)
def unpack_archive(
    archive_name: str = "qwen_ft_protocol_a_small.zip",
    protocol: str = "protocol-a-small",
) -> dict[str, object]:
    archive_path = VOL_ROOT / "archives" / archive_name
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found in Modal Volume: {archive_path}")

    data_root = VOL_ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
        if not any(name.startswith(f"{protocol}/manifests/") for name in names):
            raise ValueError(
                "Archive does not look like a qwen_ft export zip. "
                "If this is a raw GenImage zip, run prepare_from_raw_archive instead."
            )
        zf.extractall(data_root)

    protocol_root = data_root / protocol
    train_manifest = protocol_root / "manifests" / "train.csv"
    eval_manifest = protocol_root / "manifests" / "eval_small.csv"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Missing train manifest: {train_manifest}")
    if not eval_manifest.is_file():
        raise FileNotFoundError(f"Missing eval manifest: {eval_manifest}")

    from aiforensics.finetune.qwen_data import rewrite_qwen_jsonl_image_paths

    abs_jsonl = rewrite_qwen_jsonl_image_paths(
        protocol_root / "qwen_train.jsonl",
        protocol_root / "qwen_train_abs.jsonl",
        protocol_root,
    )
    volume.commit()
    return {
        "protocol_root": str(protocol_root),
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "training_jsonl": str(abs_jsonl),
    }


@app.function(image=image, volumes={"/vol": volume}, timeout=60 * 60 * 2)
def prepare_from_raw_archive(
    archive_name: str = "qwen_ft_protocol_a_small.zip",
    protocol: str = "protocol-a-small",
    train_per_label: int = 100,
    eval_per_label: int = 50,
    eval_full_per_label: int = 0,
    seed: int = 70,
) -> dict[str, object]:
    archive_path = VOL_ROOT / "archives" / archive_name
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found in Modal Volume: {archive_path}")

    from aiforensics.finetune.genimage_export import (
        build_export_plan,
        find_genimage_data_root,
        write_export_archive,
    )
    from aiforensics.finetune.qwen_data import (
        rewrite_qwen_jsonl_image_paths,
        split_qwen_train_val,
    )

    with tempfile.TemporaryDirectory(prefix="qwen_ft_raw_") as tmp:
        tmp_root = Path(tmp)
        raw_extract_root = tmp_root / "raw"
        raw_extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(raw_extract_root)

        raw_data_root = find_genimage_data_root(raw_extract_root)
        plan = build_export_plan(
            data_root=raw_data_root,
            protocol=protocol,
            train_per_label=train_per_label,
            eval_per_label=eval_per_label,
            eval_full_per_label=eval_full_per_label,
            seed=seed,
        )
        prepared_archive = write_export_archive(
            plan,
            tmp_root / f"{protocol}.zip",
            tmp_root / "prepared_stage",
        )
        with zipfile.ZipFile(prepared_archive) as zf:
            zf.extractall(VOL_ROOT / "data")

    protocol_root = VOL_ROOT / "data" / protocol
    train_manifest = protocol_root / "manifests" / "train.csv"
    eval_manifest = protocol_root / "manifests" / "eval_small.csv"
    abs_jsonl = rewrite_qwen_jsonl_image_paths(
        protocol_root / "qwen_train.jsonl",
        protocol_root / "qwen_train_abs.jsonl",
        protocol_root,
    )
    split_qwen_train_val(
        abs_jsonl,
        protocol_root / "qwen_train_split_abs.jsonl",
        protocol_root / "qwen_val_abs.jsonl",
        val_ratio=0.2,
    )

    volume.commit()
    return {
        "protocol_root": str(protocol_root),
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "training_jsonl": str(abs_jsonl),
        "train_images": len(plan.train_records),
        "eval_images": len(plan.eval_records),
        "eval_full_images": len(plan.eval_full_records),
    }


def _run(cmd: list[str], cwd: Path | str | None = "/repo") -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100-40GB",
    timeout=60 * 60 * 8,
)
def train_lora(
    protocol: str = "protocol-a-small",
    epochs: float = 3.0,
    learning_rate: float = 2e-4,
    max_pixels: int = 200704,
    min_pixels: int = 50176,
    method: str = "lora",
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    val_ratio: float = 0.2,
    eval_steps: int = 25,
    clean_checkpoints: bool = True,
    per_device_eval_batch_size: int = 1,
    eval_accumulation_steps: int = 1,
    prediction_loss_only: bool = True,
) -> dict[str, object]:
    protocol_root = VOL_ROOT / "data" / protocol
    train_jsonl = protocol_root / "qwen_train_abs.jsonl"
    output_dir = VOL_ROOT / "checkpoints" / protocol
    final_adapter = output_dir / "final_adapter"
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Run unpack_archive first; missing {train_jsonl}")

    if clean_checkpoints and output_dir.exists():
        import shutil

        print(
            f"[modal] cleaning previous checkpoints in {output_dir} for fresh training...",
            flush=True,
        )
        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print(
            f"[modal] {output_dir} is now clean; ready for fresh 3-epoch training.",
            flush=True,
        )

    cmd = [
        "python",
        "/repo/infra/qwen_ft/train_qwen_ft.py",
        "--model-id",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        "--save-total-limit",
        "3",
        "--method",
        method,
        "--max_pixels",
        str(max_pixels),
        "--min_pixels",
        str(min_pixels),
        "--bf16",
        str(bf16),
        "--gradient_checkpointing",
        str(gradient_checkpointing),
    ]

    if val_ratio > 0:
        from aiforensics.finetune.qwen_data import split_qwen_train_val

        train_split_jsonl = protocol_root / "qwen_train_split_abs.jsonl"
        val_jsonl = protocol_root / "qwen_val_abs.jsonl"
        n_train, n_val = split_qwen_train_val(
            train_jsonl,
            train_split_jsonl,
            val_jsonl,
            val_ratio=val_ratio,
        )
        print(f"[modal] split training data into {n_train} train and {n_val} val", flush=True)
        cmd.extend(
            [
                "--train-jsonl",
                str(train_split_jsonl),
                "--val-jsonl",
                str(val_jsonl),
                "--eval-steps",
                str(eval_steps),
                "--save-steps",
                str(eval_steps),
                "--per-device-eval-batch-size",
                str(per_device_eval_batch_size),
                "--eval-accumulation-steps",
                str(eval_accumulation_steps),
                "--prediction-loss-only",
                str(prediction_loss_only),
            ]
        )
    else:
        cmd.extend(
            [
                "--train-jsonl",
                str(train_jsonl),
                "--save-steps",
                "100",
            ]
        )

    _run(cmd, cwd="/repo")
    has_config = (final_adapter / "adapter_config.json").is_file()
    has_weights = any(
        (final_adapter / marker).is_file()
        for marker in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "adapter_model.safetensors.index.json",
        )
    )
    if not (has_config and has_weights):
        raise FileNotFoundError(
            f"Expected final adapter with config and weights at {final_adapter}"
        )
    volume.commit()
    return {
        "adapter": str(final_adapter),
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
        "method": method,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "eval_accumulation_steps": eval_accumulation_steps,
        "prediction_loss_only": prediction_loss_only,
    }


def _eval(config_path: str) -> dict[str, object]:
    _run(["aiforensics", "run", "--baseline", "qwen_ft", "--config", config_path], cwd="/repo")
    _run(["aiforensics", "evaluate", "--config", config_path], cwd="/repo")
    _run(["aiforensics", "report", "--config", config_path], cwd="/repo")
    return {"config": config_path, "status": "completed"}


@app.function(image=image, volumes={"/vol": volume}, gpu="A100-40GB", timeout=60 * 60 * 4)
def eval_small(protocol: str = "protocol-a-small") -> dict[str, object]:
    result = _eval("configs/qwen_ft_protocol_a_small_modal.yaml")
    volume.commit()
    return result


@app.function(image=image, volumes={"/vol": volume}, gpu="A100-40GB", timeout=60 * 60 * 12)
def eval_full(protocol: str = "protocol-a-small") -> dict[str, object]:
    result = _eval("configs/qwen_ft_protocol_a_full_modal.yaml")
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    action: str = "inspect_volume",
    archive_name: str = "qwen_ft_protocol_a_small.zip",
    protocol: str = "protocol-a-small",
    max_pixels: int = 200704,
    min_pixels: int = 50176,
    method: str = "lora",
    bf16: bool = True,
    gradient_checkpointing: bool = True,
) -> None:
    actions = {
        "inspect_volume": lambda: inspect_volume.remote(),
        "unpack_archive": lambda: unpack_archive.remote(
            archive_name=archive_name, protocol=protocol
        ),
        "prepare_from_raw_archive": lambda: prepare_from_raw_archive.remote(
            archive_name=archive_name,
            protocol=protocol,
        ),
        "train_lora": lambda: train_lora.remote(
            protocol=protocol,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            method=method,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
        ),
        "eval_small": lambda: eval_small.remote(protocol=protocol),
        "eval_full": lambda: eval_full.remote(protocol=protocol),
    }
    if action not in actions:
        raise ValueError(f"Unknown action {action!r}; expected one of {sorted(actions)}")
    result = actions[action]()
    print(json.dumps(result, indent=2, sort_keys=True))
