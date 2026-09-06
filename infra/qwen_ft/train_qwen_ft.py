"""Vertex training entrypoint wrapper for qwen_ft LoRA fine-tuning.

Runs as the container ENTRYPOINT. Responsibilities:

- pull the training JSONL and every gs:// image it references from GCS to
  local disk, because the upstream Qwen2.5-VL fine-tuning script consumes
  local files only
- invoke the upstream training script with the LoRA flags passed through
- sync checkpoints and the final adapter back to the GCS output prefix
- exit non-zero when training fails or no adapter artifact was produced

Heavy dependencies (google-cloud-storage, torch) are imported lazily so the
test suite can exercise the parsing and rewriting helpers on any machine.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__all__ = [
    "ADAPTER_MARKERS",
    "build_parser",
    "build_upstream_command",
    "has_final_adapter",
    "rewrite_jsonl_image_uris",
    "write_qwen_dataset_registry",
]

# Arguments consumed by this wrapper; never forwarded to the upstream script.
WRAPPER_ONLY_ARGS = {
    "training_script",
    "work_dir",
    "local_work_dir",
    "method",
    "model_id",
    "train_jsonl",
    "val_jsonl",
    "epochs",
}

# Container-internal contract: upstream Qwen2.5-VL training script
DEFAULT_TRAINING_SCRIPT = "/opt/qwen-vl-finetune/qwen-vl-finetune/qwenvl/train/train_qwen.py"
FALLBACK_TRAINING_SCRIPTS = (
    "/opt/qwen-vl-finetune/qwenvl/train/train_qwen.py",
    "/opt/qwen-vl-finetune/qwen-vl-finetune/finetune.py",
)


def resolve_default_training_script() -> str:
    if Path(DEFAULT_TRAINING_SCRIPT).is_file():
        return DEFAULT_TRAINING_SCRIPT
    for fallback in FALLBACK_TRAINING_SCRIPTS:
        if Path(fallback).is_file():
            return fallback
    return DEFAULT_TRAINING_SCRIPT


# A LoRA run only counts as finished when both the adapter configuration
# AND valid weight artifacts exist.
ADAPTER_CONFIG_MARKER = "adapter_config.json"
ADAPTER_WEIGHT_MARKERS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.safetensors.index.json",
)
ADAPTER_MARKERS = (ADAPTER_CONFIG_MARKER, *ADAPTER_WEIGHT_MARKERS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qwen_ft training entrypoint")
    parser.add_argument("--training_script", default=resolve_default_training_script())
    parser.add_argument(
        "--work_dir",
        default=None,
        help="Local scratch directory; defaults to a fresh temp directory",
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-id", "--model_id", dest="model_id", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--train-jsonl", "--train_jsonl", dest="train_jsonl", default=None)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    # All remaining flags are forwarded verbatim to the upstream script; the
    # defaults mirror the Task 4 Vertex job spec payload.
    parser.add_argument("--lora_enable", default="True")
    parser.add_argument("--tune_mm_llm", default="True")
    parser.add_argument("--tune_mm_vision", default="False")
    parser.add_argument("--tune_mm_mlp", default="True")
    parser.add_argument("--num_train_epochs", default="1")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", default="1")
    parser.add_argument("--gradient_accumulation_steps", default="16")
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", default="2e-4")
    parser.add_argument("--val_jsonl", "--val-jsonl", dest="val_jsonl", default=None)
    parser.add_argument(
        "--eval_strategy",
        "--eval-strategy",
        "--evaluation_strategy",
        dest="eval_strategy",
        default=None,
    )
    parser.add_argument("--eval_steps", "--eval-steps", dest="eval_steps", default=None)
    parser.add_argument(
        "--load_best_model_at_end",
        "--load-best-model-at-end",
        dest="load_best_model_at_end",
        default=None,
    )
    parser.add_argument(
        "--metric_for_best_model",
        "--metric-for-best-model",
        dest="metric_for_best_model",
        default=None,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        "--per-device-eval-batch-size",
        dest="per_device_eval_batch_size",
        default=None,
    )
    parser.add_argument(
        "--eval_accumulation_steps",
        "--eval-accumulation-steps",
        dest="eval_accumulation_steps",
        default=None,
    )
    parser.add_argument(
        "--prediction_loss_only",
        "--prediction-loss-only",
        dest="prediction_loss_only",
        default=None,
    )
    parser.add_argument("--save_strategy", default="steps")
    parser.add_argument("--save_steps", "--save-steps", dest="save_steps", default="100")
    parser.add_argument(
        "--save_total_limit", "--save-total-limit", dest="save_total_limit", default="3"
    )
    parser.add_argument("--logging_steps", default="10")
    parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--max_pixels", "--max-pixels", dest="max_pixels", type=int, default=200704)
    parser.add_argument("--min_pixels", "--min-pixels", dest="min_pixels", type=int, default=50176)
    parser.add_argument("--bf16", default="True")
    parser.add_argument(
        "--gradient_checkpointing",
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        default="True",
    )
    parser.add_argument(
        "--local-work-dir", dest="local_work_dir", type=Path, default=Path("/tmp/qwen_ft")
    )
    return parser


def is_gcs_uri(value: str) -> bool:
    return value.startswith("gs://")


def gcs_uri_to_local_path(uri: str, local_root: Path) -> Path:
    """Map gs://<bucket>/<object> onto <local_root>/<bucket>/<object>."""
    return local_root / uri.removeprefix("gs://")


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    bucket_name, _, object_name = uri.removeprefix("gs://").partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"Malformed gs:// URI (expected gs://<bucket>/<object>): {uri}")
    return bucket_name, object_name


def download_gcs_file(uri: str, local_root: Path) -> Path:
    from google.cloud import storage

    bucket_name, object_name = _split_gcs_uri(uri)
    local_path = gcs_uri_to_local_path(uri, local_root)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket_name).blob(object_name).download_to_filename(local_path)
    return local_path


def rewrite_jsonl_image_uris(jsonl_path: Path, local_root: Path) -> Path:
    """Rewrite gs:// image references onto their downloaded local paths.

    Writes a sibling ``*_local.jsonl`` and leaves the original untouched for
    provenance. Raises when a referenced image was not downloaded.
    """
    rewritten = jsonl_path.with_name(jsonl_path.stem + "_local.jsonl")
    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        uri = row.get("image", "")
        if is_gcs_uri(uri):
            local_path = gcs_uri_to_local_path(uri, local_root)
            if not local_path.is_file():
                raise FileNotFoundError(
                    f"Downloaded training image missing: {local_path} (from {uri})"
                )
            row["image"] = str(local_path)

    with open(rewritten, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rewritten


def prepare_training_data(data_path: str, local_root: Path) -> Path:
    """Download a gs:// JSONL plus its images, or pass local paths through."""
    if not is_gcs_uri(data_path):
        return Path(data_path)

    local_jsonl = download_gcs_file(data_path, local_root)
    with open(local_jsonl, encoding="utf-8") as f:
        image_uris = [json.loads(line).get("image", "") for line in f if line.strip()]

    for uri in sorted({u for u in image_uris if is_gcs_uri(u)}):
        download_gcs_file(uri, local_root)

    return rewrite_jsonl_image_uris(local_jsonl, local_root)


def _patch_data_processor_eval_support(data_processor_file: Path) -> None:
    code = data_processor_file.read_text(encoding="utf-8")
    if "qwen_ft_val" in code:
        return

    target = "    train_dataset = LazySupervisedDataset(processor, data_args=data_args)"
    replacement = (
        "    train_dataset = LazySupervisedDataset(processor, data_args=data_args)\n"
        "    eval_dataset = None\n"
        "    try:\n"
        "        from . import data_dict\n"
        '        if "qwen_ft_val" in data_dict:\n'
        "            import copy\n"
        "            val_args = copy.copy(data_args)\n"
        '            val_args.dataset_use = "qwen_ft_val"\n'
        "            eval_dataset = LazySupervisedDataset(processor, data_args=val_args)\n"
        "    except Exception as _eval_err:\n"
        '        rank0_print(f"Warning: could not create eval_dataset: {_eval_err}")'
    )
    if target in code:
        code = code.replace(target, replacement, 1)
        code = code.replace("eval_dataset=None", "eval_dataset=eval_dataset")
        data_processor_file.write_text(code, encoding="utf-8")


def write_qwen_dataset_registry(
    training_script: str | Path,
    train_jsonl: Path,
    *,
    val_jsonl: Path | str | None = None,
    dataset_name: str = "qwen_ft",
    val_dataset_name: str = "qwen_ft_val",
) -> Path:
    """Register the qwen_ft JSONL in the upstream Qwen dataset registry."""
    training_script = Path(training_script)
    data_init = training_script.parents[1] / "data" / "__init__.py"
    if not data_init.is_file():
        raise FileNotFoundError(f"Qwen dataset registry not found: {data_init}")

    block = (
        "\n\n"
        "# Added by ai-image-forensics qwen_ft training wrapper.\n"
        "QWEN_FT_DATASET = {\n"
        f'    "annotation_path": "{Path(train_jsonl).resolve()}",\n'
        '    "data_path": "",\n'
        "}\n"
        f'data_dict["{dataset_name}"] = QWEN_FT_DATASET\n'
    )
    if val_jsonl is not None:
        block += (
            "QWEN_FT_VAL_DATASET = {\n"
            f'    "annotation_path": "{Path(val_jsonl).resolve()}",\n'
            '    "data_path": "",\n'
            "}\n"
            f'data_dict["{val_dataset_name}"] = QWEN_FT_VAL_DATASET\n'
        )

    source = data_init.read_text(encoding="utf-8")
    marker = f'data_dict["{dataset_name}"] = QWEN_FT_DATASET'
    if marker not in source:
        data_init.write_text(source.rstrip() + block, encoding="utf-8")
    elif val_jsonl is not None and f'data_dict["{val_dataset_name}"]' not in source:
        val_block = (
            "\n"
            "QWEN_FT_VAL_DATASET = {\n"
            f'    "annotation_path": "{Path(val_jsonl).resolve()}",\n'
            '    "data_path": "",\n'
            "}\n"
            f'data_dict["{val_dataset_name}"] = QWEN_FT_VAL_DATASET\n'
        )
        data_init.write_text(source.rstrip() + val_block, encoding="utf-8")

    if val_jsonl is not None:
        data_processor_file = training_script.parents[1] / "data" / "data_processor.py"
        if data_processor_file.is_file():
            _patch_data_processor_eval_support(data_processor_file)

    return data_init


def build_upstream_command(
    args: argparse.Namespace | None = None,
    data_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    model_id: str | None = None,
    train_jsonl: Path | str | None = None,
    val_jsonl: Path | str | None = None,
    epochs: float | int | str | None = None,
    learning_rate: float | str | None = None,
    method: str = "lora",
    max_pixels: int | str = 200704,
    min_pixels: int | str = 50176,
    bf16: bool | str = "True",
    gradient_checkpointing: bool | str = "True",
    save_steps: int | str = 100,
    save_total_limit: int | str = 3,
    eval_steps: int | str | None = None,
    eval_strategy: str | None = None,
    load_best_model_at_end: bool | str | None = None,
    metric_for_best_model: str | None = None,
    per_device_eval_batch_size: int | str = 1,
    eval_accumulation_steps: int | str = 1,
    prediction_loss_only: bool | str = "True",
) -> list[str]:
    """Forward arguments to the upstream training script."""
    if not isinstance(args, argparse.Namespace):
        script = resolve_default_training_script()
        command = [sys.executable, str(script)]
        resolved_model = model_id or "Qwen/Qwen2.5-VL-7B-Instruct"
        resolved_out = str(output_dir or "")
        resolved_epochs = str(epochs if epochs is not None else 1.0)
        resolved_lr = str(learning_rate if learning_rate is not None else 2e-4)

        command.extend(
            [
                f"--model_name_or_path={resolved_model}",
                "--dataset_use=qwen_ft",
                f"--output_dir={resolved_out}",
                f"--num_train_epochs={resolved_epochs}",
                f"--learning_rate={resolved_lr}",
                f"--save_steps={save_steps}",
                f"--save_total_limit={save_total_limit}",
                f"--max_pixels={max_pixels}",
                f"--min_pixels={min_pixels}",
                f"--bf16={bf16}",
                f"--gradient_checkpointing={gradient_checkpointing}",
                "--lora_enable=True",
                "--tune_mm_llm=True",
                "--tune_mm_vision=False",
                "--tune_mm_mlp=True",
            ]
        )
        if val_jsonl is not None:
            resolved_eval_strat = eval_strategy or "steps"
            resolved_eval_steps = str(eval_steps if eval_steps is not None else 25)
            command.extend(
                [
                    f"--eval_strategy={resolved_eval_strat}",
                    f"--eval_steps={resolved_eval_steps}",
                    "--save_strategy=steps",
                    f"--save_steps={resolved_eval_steps}",
                    "--load_best_model_at_end=True",
                    "--metric_for_best_model=loss",
                    f"--per_device_eval_batch_size={per_device_eval_batch_size}",
                    f"--eval_accumulation_steps={eval_accumulation_steps}",
                    f"--prediction_loss_only={prediction_loss_only}",
                ]
            )
        if method == "qlora":
            command.append("--qlora=True")
        return command

    command = [sys.executable, str(args.training_script)]
    resolved_model = (
        getattr(args, "model_id", None)
        or getattr(args, "model_name_or_path", None)
        or "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    resolved_out = str(output_dir or getattr(args, "output_dir", None) or "")
    resolved_epochs = (
        getattr(args, "epochs", None) or getattr(args, "num_train_epochs", None) or "1"
    )
    resolved_max_pixels = getattr(args, "max_pixels", max_pixels)
    resolved_min_pixels = getattr(args, "min_pixels", min_pixels)
    resolved_method = getattr(args, "method", method)
    resolved_bf16 = getattr(args, "bf16", bf16)
    resolved_grad_ckpt = getattr(args, "gradient_checkpointing", gradient_checkpointing)

    if getattr(args, "val_jsonl", None):
        if getattr(args, "eval_strategy", None) is None:
            args.eval_strategy = "steps"
        resolved_eval_steps = str(getattr(args, "eval_steps", None) or "25")
        args.eval_steps = resolved_eval_steps
        if str(getattr(args, "save_steps", None)) in ("100", "None"):
            args.save_steps = resolved_eval_steps
        if getattr(args, "save_strategy", None) is None:
            args.save_strategy = "steps"
        if getattr(args, "load_best_model_at_end", None) is None:
            args.load_best_model_at_end = "True"
        if getattr(args, "metric_for_best_model", None) is None:
            args.metric_for_best_model = "loss"
        if getattr(args, "per_device_eval_batch_size", None) is None:
            args.per_device_eval_batch_size = "1"
        if getattr(args, "eval_accumulation_steps", None) is None:
            args.eval_accumulation_steps = "1"
        if getattr(args, "prediction_loss_only", None) is None:
            args.prediction_loss_only = "True"

    command.extend(
        [
            f"--model_name_or_path={resolved_model}",
            "--dataset_use=qwen_ft",
            f"--output_dir={resolved_out}",
            f"--num_train_epochs={resolved_epochs}",
            f"--max_pixels={resolved_max_pixels}",
            f"--min_pixels={resolved_min_pixels}",
            f"--bf16={resolved_bf16}",
            f"--gradient_checkpointing={resolved_grad_ckpt}",
        ]
    )

    skip_args = WRAPPER_ONLY_ARGS | {
        "model_name_or_path",
        "model_id",
        "data_path",
        "train_jsonl",
        "output_dir",
        "num_train_epochs",
        "epochs",
        "max_pixels",
        "min_pixels",
        "bf16",
        "gradient_checkpointing",
    }
    for name, value in vars(args).items():
        if name in skip_args or value is None:
            continue
        command.append(f"--{name}={value}")
    if resolved_method == "qlora":
        command.append("--qlora=True")
    return command


def is_valid_adapter_dir(directory: Path) -> bool:
    """Check whether a directory contains a complete PEFT LoRA adapter.

    Requires adapter_config.json AND at least one weights artifact (safetensors or bin).
    """
    has_config = (directory / ADAPTER_CONFIG_MARKER).is_file()
    has_weights = any((directory / marker).is_file() for marker in ADAPTER_WEIGHT_MARKERS)
    return has_config and has_weights


def has_final_adapter(output_dir: Path) -> bool:
    return is_valid_adapter_dir(output_dir) or is_valid_adapter_dir(output_dir / "final_adapter")


def sync_output_to_gcs(local_output_dir: Path, gcs_prefix: str) -> int:
    """Upload every artifact file under the local output dir to the GCS prefix."""
    from google.cloud import storage

    bucket_name, prefix = _split_gcs_uri(gcs_prefix)
    bucket = storage.Client().bucket(bucket_name)

    uploaded = 0
    for path in sorted(local_output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_output_dir).as_posix()
        bucket.blob(f"{prefix.rstrip('/')}/{relative}").upload_from_filename(path)
        uploaded += 1
    return uploaded


def log_package_versions() -> None:
    """Record the training stack versions; the plan pins them after first run."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("torch", "transformers", "peft", "accelerate", "qwen-vl-utils"):
            try:
                print(f"[train] {package}=={version(package)}", flush=True)
            except PackageNotFoundError:
                print(f"[train] {package}: not installed", flush=True)
    except Exception as exc:
        print(f"[train] could not record package versions: {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = (
        Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="qwen_ft_train_"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    output_is_gcs = is_gcs_uri(args.output_dir)
    local_output_dir = work_dir / "output" if output_is_gcs else Path(args.output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)

    log_package_versions()
    print(
        f"[train] image bounds: min_pixels={args.min_pixels}, "
        f"max_pixels={args.max_pixels}, method={args.method}, "
        f"bf16={args.bf16}, grad_ckpt={args.gradient_checkpointing}",
        flush=True,
    )

    data_source = args.train_jsonl or args.data_path
    if not data_source:
        print(
            "[train] error: either --data_path or --train-jsonl is required",
            file=sys.stderr,
            flush=True,
        )
        return 1

    try:
        data_path = prepare_training_data(data_source, work_dir / "data")
    except Exception as exc:
        print(f"[train] data preparation failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"[train] training data ready: {data_path}", flush=True)

    val_path = None
    if getattr(args, "val_jsonl", None):
        try:
            val_path = prepare_training_data(args.val_jsonl, work_dir / "data_val")
        except Exception as exc:
            print(f"[train] validation data preparation failed: {exc}", file=sys.stderr, flush=True)
            return 1
        print(f"[train] validation data ready: {val_path}", flush=True)

    try:
        registry_path = write_qwen_dataset_registry(
            args.training_script, data_path, val_jsonl=val_path
        )
    except Exception as exc:
        print(f"[train] dataset registry setup failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"[train] dataset registry ready: {registry_path}", flush=True)

    command = build_upstream_command(args, data_path, local_output_dir)
    print(f"[train] upstream command: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            f"[train] upstream training failed with exit code {completed.returncode}",
            file=sys.stderr,
            flush=True,
        )
        return completed.returncode or 1

    final_adapter = local_output_dir / "final_adapter"
    final_adapter.mkdir(parents=True, exist_ok=True)
    for filename in ADAPTER_MARKERS:
        candidate = local_output_dir / filename
        if candidate.is_file():
            shutil.copy2(candidate, final_adapter / filename)
    if not any((final_adapter / m).is_file() for m in ADAPTER_MARKERS):
        checkpoints = sorted(
            local_output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if checkpoints:
            latest = checkpoints[-1]
            for filename in ADAPTER_MARKERS:
                candidate = latest / filename
                if candidate.is_file():
                    shutil.copy2(candidate, final_adapter / filename)

    if not has_final_adapter(local_output_dir):
        print(
            "[train] upstream training finished but produced no LoRA adapter artifacts",
            file=sys.stderr,
            flush=True,
        )
        return 1

    training_meta = {
        "model_id": getattr(args, "model_id", None) or args.model_name_or_path,
        "method": args.method,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "epochs": getattr(args, "epochs", None) or args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "save_steps": args.save_steps,
    }
    meta_json = json.dumps(training_meta, indent=2)
    (local_output_dir / "training_meta.json").write_text(meta_json, encoding="utf-8")
    (final_adapter / "training_meta.json").write_text(meta_json, encoding="utf-8")

    if output_is_gcs:
        uploaded = sync_output_to_gcs(local_output_dir, args.output_dir)
        print(f"[train] synced {uploaded} artifact file(s) to {args.output_dir}", flush=True)

    print("[train] training completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
