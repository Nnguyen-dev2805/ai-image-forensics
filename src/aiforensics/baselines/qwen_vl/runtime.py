import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


def load_model(model_id: str, device: str, allow_deferred: bool, exception_cls: type[Exception]):
    try:
        import importlib.util

        if not all(
            importlib.util.find_spec(m) is not None
            for m in ("torch", "transformers", "qwen_vl_utils", "accelerate")
        ):
            raise ImportError("Missing Qwen optional dependencies")

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor
    except Exception as e:
        if allow_deferred:
            raise exception_cls(f"Model setup failed: {e}") from e
        else:
            raise Exception(f"Model setup failed: {e}") from e


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
        inputs = inputs.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text
    except Exception as e:
        raise Exception(f"Inference failed for {image_path}: {e}") from e
