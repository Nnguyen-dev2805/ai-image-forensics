import logging
from pathlib import Path

from aiforensics.baselines.base import BaselineAdapter, RunResult
from aiforensics.baselines.qwen_vl.parsing import parse_qwen_output
from aiforensics.baselines.qwen_vl.prompt import get_prompt
from aiforensics.cache.keys import cache_key
from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import ManifestRecord, load_manifest
from aiforensics.schemas.predictions import (
    PredictionRecord,
    validate_predictions,
    write_predictions,
)

logger = logging.getLogger(__name__)


class BaselineDeferredError(Exception):
    """Raised when environment cannot support the requested run and allow_deferred is True."""

    pass


class QwenVLAdapter(BaselineAdapter):
    name = "qwen_vl"

    def run(
        self,
        *,
        config: AppConfig,
        output_dir: Path,
        run_id: str,
        seed: int | None = None,
    ) -> RunResult:
        run_dir = output_dir

        if not run_dir.exists():
            run_dir.mkdir(parents=True)

        import logging

        logger = logging.getLogger(__name__)

        self._counts = {
            "parsed": 0,
            "recovered": 0,
            "failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        try:
            if not config.baselines.qwen_vl.enabled:
                raise BaselineDeferredError("qwen_vl is disabled in config")

            if config.baselines.qwen_vl.temperature != 0.0:
                raise Exception("Qwen Task 8 requires temperature == 0.0")

            records = self._load_manifests(config)
            if not records:
                raise Exception("No evaluation records found")

            # We determine device upfront for logging
            device = "unknown"
            try:
                device = self._get_qwen_device(config)
            except BaselineDeferredError:
                device = "deferred"
            except Exception:
                device = "failed"

            predictions = self._run_inference(records, config, run_id, self._counts)
            val_result = validate_predictions(
                predictions,
                manifest_sample_ids={r.sample_id for r in records},
                require_mllm_fields=True,
            )
            if not val_result.is_valid:
                raise Exception(f"Prediction validation failed: {val_result.errors}")

            # Write predictions
            pred_file = run_dir / "predictions.jsonl"
            write_predictions(predictions, pred_file)

            # 7. Read-back validation
            from aiforensics.schemas.predictions import load_predictions

            loaded_preds = load_predictions(pred_file)
            val_loaded = validate_predictions(
                loaded_preds,
                manifest_sample_ids={r.sample_id for r in records},
                require_mllm_fields=True,
            )
            if not val_loaded.is_valid:
                raise Exception(f"Read-back prediction validation failed: {val_loaded.errors}")

            with open(run_dir / "logs.txt", "a", encoding="utf-8") as f:
                f.write(f"Run completed successfully for baseline {self.name}\n")
                f.write(f"Processed {len(records)} records.\n")
                f.write(f"Model ID: {config.baselines.qwen_vl.model_id}\n")
                f.write(f"Prompt ID: {config.baselines.qwen_vl.prompt_id}\n")
                f.write(f"Resolved device: {device}\n")
                f.write(f"Cache Hits: {self._counts['cache_hits']}\n")
                f.write(f"Cache Misses: {self._counts['cache_misses']}\n")
                f.write(
                    f"Parsed: {self._counts['parsed']}, Recovered: {self._counts['recovered']}, "
                    f"Failed (Unknown): {self._counts['failed']}\n"
                )

            return RunResult(
                baseline=self.name,
                run_id=run_id,
                status="completed",
                output_dir=run_dir,
                prediction_path=run_dir / "predictions.jsonl",
                log_path=run_dir / "logs.txt",
                environment_path=run_dir / "environment.json",
                status_path=run_dir / "status.json",
                reason=None,
            )

        except BaselineDeferredError as e:
            logger.info(f"Run deferred: {e}")
            try:
                if (run_dir / "predictions.jsonl").exists():
                    (run_dir / "predictions.jsonl").unlink()
            except Exception:
                pass
            with open(run_dir / "logs.txt", "a", encoding="utf-8") as f:
                f.write(f"Run deferred: {e}\n")

            return RunResult(
                baseline=self.name,
                run_id=run_id,
                status="deferred",
                output_dir=run_dir,
                prediction_path=None,
                log_path=run_dir / "logs.txt",
                environment_path=run_dir / "environment.json",
                status_path=run_dir / "status.json",
                reason=str(e),
            )
        except Exception as e:
            logger.error(f"Run failed: {e}")
            try:
                if (run_dir / "predictions.jsonl").exists():
                    (run_dir / "predictions.jsonl").unlink()
            except Exception:
                pass
            with open(run_dir / "logs.txt", "a", encoding="utf-8") as f:
                f.write(f"Run failed: {e}\n")

            return RunResult(
                baseline=self.name,
                run_id=run_id,
                status="failed",
                output_dir=run_dir,
                prediction_path=None,
                log_path=run_dir / "logs.txt",
                environment_path=run_dir / "environment.json",
                status_path=run_dir / "status.json",
                reason=str(e),
            )

    def _load_manifests(self, config: AppConfig) -> list[ManifestRecord]:
        records = []
        datasets_cfg = config.datasets

        if datasets_cfg.tiny_genimage.dev_manifest.exists():
            records.extend(
                load_manifest(
                    datasets_cfg.tiny_genimage.dev_manifest, data_root=config.paths.data_root
                )
            )
        else:
            logger.warning(f"Manifest missing: {datasets_cfg.tiny_genimage.dev_manifest}")

        if getattr(datasets_cfg.genimage_unseen, "enabled", True):
            if datasets_cfg.genimage_unseen.manifest.exists():
                records.extend(
                    load_manifest(
                        datasets_cfg.genimage_unseen.manifest, data_root=config.paths.data_root
                    )
                )
            else:
                logger.warning(f"Manifest missing: {datasets_cfg.genimage_unseen.manifest}")

        if getattr(datasets_cfg.synthbuster, "enabled", True):
            if datasets_cfg.synthbuster.manifest.exists():
                records.extend(
                    load_manifest(
                        datasets_cfg.synthbuster.manifest, data_root=config.paths.data_root
                    )
                )
            else:
                logger.warning(f"Manifest missing: {datasets_cfg.synthbuster.manifest}")

        if not records:
            from aiforensics.data.manifest import ManifestError

            raise ManifestError("No valid evaluation manifests found.")

        return records

    def _get_qwen_device(self, config: AppConfig) -> str:
        device_str = config.runtime.device
        allow_def = config.baselines.qwen_vl.allow_deferred

        def _raise(msg):
            if allow_def:
                raise BaselineDeferredError(msg)
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

    def _load_model(self, config: AppConfig, device: str):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.baselines.qwen_vl.model_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            model.eval()

            processor = AutoProcessor.from_pretrained(config.baselines.qwen_vl.model_id)
            return model, processor
        except Exception as e:
            if config.baselines.qwen_vl.allow_deferred:
                raise BaselineDeferredError(f"Model setup failed: {e}") from e
            else:
                raise Exception(f"Model setup failed: {e}") from e

    def _generate_one_image(
        self, model, processor, image_path: Path, prompt_text: str, device: str, max_new_tokens: int
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

    def _run_inference(
        self, records: list[ManifestRecord], config: AppConfig, run_id: str, counts: dict
    ) -> list[PredictionRecord]:
        qwen_cfg = config.baselines.qwen_vl

        prompt_text = get_prompt(qwen_cfg.prompt_id)
        cache_enabled = qwen_cfg.cache_outputs
        cache_dir = config.paths.cache_root / "qwen_vl" / "raw_outputs"

        if cache_enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)

        import hashlib
        import json

        def _get_cache_key(record: ManifestRecord) -> str:
            checksum = record.checksum
            if not checksum:
                if not record.path.exists():
                    raise Exception(f"Image not found: {record.path}")
                checksum = hashlib.sha256(record.path.read_bytes()).hexdigest()
            return cache_key(
                {
                    "baseline": "qwen_vl",
                    "sample_checksum": checksum,
                    "model_id": qwen_cfg.model_id,
                    "prompt_id": qwen_cfg.prompt_id,
                    "temperature": str(qwen_cfg.temperature),
                    "max_new_tokens": str(qwen_cfg.max_new_tokens),
                    "output_cache_version": "qwen_vl_raw_v2",  # Bump version since format changed
                }
            )

        model = None
        processor = None
        device = None

        predictions = []

        for record in records:
            if not record.path.exists():
                raise Exception(f"Image missing: {record.path}")

            if record.checksum:
                csum = hashlib.sha256(record.path.read_bytes()).hexdigest()
                if csum != record.checksum:
                    raise Exception(f"Checksum mismatch for {record.path}")

            raw_output = None
            cache_path = None

            if cache_enabled:
                ckey = _get_cache_key(record)
                cache_path = cache_dir / f"{ckey}.json"
                if cache_path.exists():
                    try:
                        raw_text = cache_path.read_text(encoding="utf-8")
                        cache_data = json.loads(raw_text)
                        if isinstance(cache_data, dict) and isinstance(
                            cache_data.get("raw_output"), str
                        ):
                            raw_output = cache_data["raw_output"]
                            counts["cache_hits"] += 1
                        else:
                            counts["cache_misses"] += 1
                    except Exception as e:
                        logger.warning(f"Cache read failed for {record.sample_id}: {e}")
                        counts["cache_misses"] += 1
                else:
                    counts["cache_misses"] += 1
            else:
                counts["cache_misses"] += 1

            if raw_output is None:
                if model is None:
                    try:
                        import importlib.util

                        if not all(
                            importlib.util.find_spec(m) is not None
                            for m in ("torch", "transformers", "qwen_vl_utils", "accelerate")
                        ):
                            raise ImportError("Missing Qwen optional dependencies")
                    except ImportError as e:
                        if qwen_cfg.allow_deferred:
                            raise BaselineDeferredError(f"Missing Qwen dependencies: {e}") from e
                        else:
                            raise Exception(f"Missing Qwen dependencies: {e}") from e

                    device = self._get_qwen_device(config)
                    model, processor = self._load_model(config, device)

                raw_output = self._generate_one_image(
                    model, processor, record.path, prompt_text, device, qwen_cfg.max_new_tokens
                )

                if cache_enabled and cache_path is not None:
                    try:
                        import os
                        import tempfile

                        tmp_fd, tmp_name = tempfile.mkstemp(
                            dir=cache_path.parent, prefix=".tmp", text=True
                        )
                        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                            json.dump({"raw_output": raw_output}, f)
                        os.replace(tmp_name, cache_path)
                    except Exception as e:
                        logger.warning(f"Cache write failed for {record.sample_id}: {e}")

            parse_result = parse_qwen_output(raw_output)

            if parse_result.parse_status == "parsed":
                counts["parsed"] += 1
            elif parse_result.parse_status == "recovered":
                counts["recovered"] += 1
            else:
                counts["failed"] += 1

            predictions.append(
                PredictionRecord(
                    sample_id=record.sample_id,
                    label_true=record.label,
                    label_pred=parse_result.label_pred,
                    score_fake=parse_result.score_fake,
                    model_name="qwen_vl",
                    source=record.source,
                    run_id=run_id,
                    dataset=record.dataset,
                    split=record.split,
                    path=record.path,
                    checksum=record.checksum,
                    prompt_id=qwen_cfg.prompt_id,
                    raw_output=raw_output,
                    explanation=parse_result.explanation,
                    parse_status=parse_result.parse_status,
                )
            )

        return predictions
