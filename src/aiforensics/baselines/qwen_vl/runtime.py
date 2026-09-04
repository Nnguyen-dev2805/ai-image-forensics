"""Qwen-VL runtime boundary: device resolution, model loading, and generation.

Every torch/transformers import lives here so adapters stay importable without
model runtimes installed. Two environment facts drive the design:

Multi-GPU. Passing a bare ``"cuda"`` device map pins the whole model to device 0,
which wastes any other visible GPU and turns a 2-GPU runtime into a 1-GPU one.
Weights are therefore sharded across visible GPUs, and a load that silently
spills to CPU or disk is rejected rather than left to run unusably slowly.

Compute dtype. ``bfloat16`` has no tensor-core support before Ampere, so on
Turing cards (T4) ``float16`` is materially faster at the same memory cost.
Because dtype changes numerical results, it is a configured, recorded choice
rather than a hidden default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_DTYPES",
    "QwenOutOfMemoryError",
    "describe_device_map",
    "generate_one_image",
    "get_qwen_device",
    "load_model",
    "resolve_torch_dtype",
]

DtypeName = Literal["bfloat16", "float16", "float32"]

SUPPORTED_DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32")

# Device-map placements that mean "not actually on a GPU".
_OFFLOAD_PLACEMENTS = ("cpu", "disk", "meta")


class QwenOutOfMemoryError(Exception):
    """Raised when inference exhausts GPU memory.

    Distinct from a generic failure: running out of VRAM is an environment
    limit, so adapters map it onto their ``deferred`` contract instead of
    reporting a broken run.
    """


def get_qwen_device(device_str: str, allow_deferred: bool, exception_cls: type[Exception]) -> str:
    def _raise(msg):
        if allow_deferred:
            raise exception_cls(msg)
        else:
            raise Exception(msg)

    import torch

    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            _raise("GPU unavailable for auto device")
    elif device_str == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            _raise("CUDA requested but unavailable")
    elif device_str == "cpu":
        _raise("CPU unsupported for real Qwen run")
    else:
        _raise(f"Unsupported device: {device_str}")


def resolve_torch_dtype(dtype_name: str):
    """Map a configured dtype name onto a torch dtype.

    Only the three dtypes Phase A/B allows are accepted; an unknown name is a
    configuration error rather than a silent fallback, because the dtype changes
    numerical results and is recorded in cache keys and reports.
    """
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported qwen dtype: {dtype_name!r}. Supported: {list(SUPPORTED_DTYPES)}"
        )
    return mapping[dtype_name]


def describe_device_map(model) -> dict[str, str]:
    """Return the model's parameter placement map, normalized to strings."""
    raw = getattr(model, "hf_device_map", None)
    if not isinstance(raw, dict):
        return {}
    return {str(module): str(placement) for module, placement in raw.items()}


def _offloaded_modules(device_map: dict[str, str]) -> list[str]:
    """List modules that ended up off the GPU."""
    offloaded = []
    for module, placement in sorted(device_map.items()):
        lowered = placement.lower()
        if any(marker in lowered for marker in _OFFLOAD_PLACEMENTS):
            offloaded.append(f"{module}={placement}")
    return offloaded


def load_model(
    model_id: str,
    device: str,
    allow_deferred: bool,
    exception_cls: type[Exception],
    *,
    dtype: str = "bfloat16",
):
    """Load the Qwen model sharded across visible GPUs at the configured dtype.

    ``device_map="auto"`` lets accelerate place weights across every visible
    GPU. That same setting will happily offload to CPU or disk when VRAM is
    short, which produces a run that appears to hang rather than fail, so any
    offloaded module is reported through the deferral path instead.
    """
    try:
        import importlib.util

        if not all(
            importlib.util.find_spec(m) is not None
            for m in ("torch", "transformers", "qwen_vl_utils", "accelerate")
        ):
            raise ImportError("Missing Qwen optional dependencies")

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        torch_dtype = resolve_torch_dtype(dtype)

        # "cuda" means "use the GPUs"; accelerate decides which ones. A bare
        # "cuda" map would pin every weight to device 0.
        device_map = "auto" if device == "cuda" else device

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model.eval()

        placement = describe_device_map(model)
        offloaded = _offloaded_modules(placement)
        if offloaded:
            raise RuntimeError(
                "Model weights were offloaded off-GPU, which makes inference "
                f"impractically slow: {offloaded[:5]}"
                + (f" (+{len(offloaded) - 5} more)" if len(offloaded) > 5 else "")
                + ". Use a runtime with more GPU memory, or a smaller model."
            )
        if placement:
            logger.info("Qwen weights placed on: %s", sorted({p for p in placement.values()}))

        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor
    except Exception as e:
        if allow_deferred:
            raise exception_cls(f"Model setup failed: {e}") from e
        else:
            raise Exception(f"Model setup failed: {e}") from e


def _input_device(model, fallback: str) -> str:
    """Pick the device that generation inputs must live on.

    With a sharded model the inputs belong on whichever device holds the first
    module, not on a bare ``"cuda"``.
    """
    device_map = describe_device_map(model)
    if device_map:
        first = device_map[sorted(device_map)[0]]
        if first.isdigit():
            return f"cuda:{first}"
        if first.lower() not in _OFFLOAD_PLACEMENTS:
            return first
    return fallback


def generate_one_image(
    model, processor, image_path: Path, prompt_text: str, device: str, max_new_tokens: int
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    try:
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(_input_device(model, device))

        with torch.no_grad():
            # do_sample=False is what makes decoding deterministic; temperature
            # is intentionally not passed, since generation ignores it in greedy
            # mode and newer transformers warn about it.
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text
    except torch.cuda.OutOfMemoryError as e:
        raise QwenOutOfMemoryError(f"GPU out of memory generating for {image_path}: {e}") from e
    except Exception as e:
        # Some stacks surface OOM as a plain RuntimeError; treat those as the
        # environment limit they are rather than a broken run.
        if "out of memory" in str(e).lower():
            raise QwenOutOfMemoryError(f"GPU out of memory generating for {image_path}: {e}") from e
        raise Exception(f"Inference failed for {image_path}: {e}") from e
