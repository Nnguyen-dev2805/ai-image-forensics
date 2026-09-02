import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from aiforensics.baselines.assisted_qwen.prompt import get_assisted_prompt
from aiforensics.baselines.base import BaselineAdapter, RunResult
from aiforensics.baselines.qwen_vl.cache import read_qwen_cache, write_qwen_cache
from aiforensics.baselines.qwen_vl.parsing import parse_qwen_output
from aiforensics.baselines.qwen_vl.runtime import (
    generate_one_image,
    get_qwen_device,
    load_model,
)
from aiforensics.cache.keys import cache_key
from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import ManifestRecord, load_manifest
from aiforensics.schemas.predictions import (
    PredictionRecord,
    load_predictions,
    validate_predictions,
    write_predictions,
)

logger = logging.getLogger(__name__)


class BaselineDeferredError(Exception):
    pass


class AssistedInput(BaseModel):
    sample_id: str
    classifier_pred: Literal["real", "fake"]
    fake_probability: float = Field(ge=0.0, le=1.0)
    source_prediction_files: tuple[Path, ...]


class AssistedQwenAdapter(BaselineAdapter):
    name = "assisted_qwen"

    def __init__(self):
        self._counts = {
            "parsed": 0,
            "recovered": 0,
            "failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "assistant_inputs_built": 0,
            "clip_files_used": 0,
        }

    def _validate_config(self, cfg) -> None:
        if not getattr(cfg, "include_classifier_pred", True):
            raise Exception("include_classifier_pred must be true")
        if not getattr(cfg, "include_fake_probability", True):
            raise Exception("include_fake_probability must be true")
        if cfg.temperature != 0.0:
            raise Exception("Assisted Qwen requires temperature == 0.0")

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

        self._counts = {
            "parsed": 0,
            "recovered": 0,
            "failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "assistant_inputs_built": 0,
            "clip_files_used": 0,
        }

        try:
            cfg = config.baselines.assisted_qwen

            if not getattr(cfg, "enabled", False):
                raise BaselineDeferredError("assisted_qwen is disabled in config")

            if cfg.prompt_id != "assisted_qwen_json_v1":
                raise Exception(f"Unsupported prompt_id: {cfg.prompt_id}")

            if getattr(cfg, "assistant_source", "") != "clip_probe":
                raise Exception(f"Unsupported assistant_source: {cfg.assistant_source}")

            self._validate_config(cfg)

            records = self._load_manifests(config)
            if not records:
                raise Exception("No evaluation records found")

            assistant_inputs = self._discover_assistant_inputs(config)

            # We determine device upfront for logging
            device = "unknown"
            try:
                device = get_qwen_device(
                    config.runtime.device, cfg.allow_deferred, BaselineDeferredError
                )
            except BaselineDeferredError:
                device = "deferred"
            except Exception:
                device = "failed"

            predictions = self._run_inference(
                records, assistant_inputs, config, run_id, self._counts
            )
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

            # Read-back validation
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
                f.write(f"Model ID: {cfg.base_model_id}\n")
                f.write(f"Prompt ID: {cfg.prompt_id}\n")
                f.write(f"Resolved device: {device}\n")
                f.write(f"CLIP Files Used: {self._counts['clip_files_used']}\n")
                if hasattr(self, "_clip_files"):
                    for cf in self._clip_files:
                        f.write(f"  - {cf}\n")
                f.write(f"Assistant Inputs Built: {self._counts['assistant_inputs_built']}\n")
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

    def _discover_assistant_inputs(self, config: AppConfig) -> dict[str, AssistedInput]:
        output_root = config.paths.output_root
        if not output_root.exists():
            raise Exception("No completed clip_probe predictions found for assisted_qwen")

        clip_files = []
        for path in output_root.rglob("predictions.jsonl"):
            status_file = path.parent / "status.json"
            if status_file.exists():
                try:
                    status_data = json.loads(status_file.read_text(encoding="utf-8"))
                    if (
                        status_data.get("baseline") == "clip_probe"
                        and status_data.get("status") == "completed"
                    ):
                        clip_files.append(path)
                except Exception:
                    pass

        if not clip_files:
            raise Exception("No completed clip_probe predictions found for assisted_qwen")

        clip_files.sort()
        self._counts["clip_files_used"] = len(clip_files)
        self._clip_files = clip_files

        from collections import defaultdict

        scores_by_sample = defaultdict(list)
        sources_by_sample = defaultdict(list)

        for p_file in clip_files:
            preds = load_predictions(p_file)
            val_res = validate_predictions(preds)
            if not val_res.is_valid:
                raise Exception(f"Invalid clip_probe predictions in {p_file}: {val_res.errors}")
            for p in preds:
                if p.model_name == "clip_probe":
                    if p.score_fake is None or p.label_pred == "unknown":
                        raise Exception(
                            f"Invalid clip_probe prediction score for "
                            f"sample {p.sample_id} in {p_file}"
                        )
                    scores_by_sample[p.sample_id].append(p.score_fake)
                    sources_by_sample[p.sample_id].append(p_file)

        inputs = {}
        for s_id, scores in scores_by_sample.items():
            import math

            for s in scores:
                if not math.isfinite(s) or not (0.0 <= s <= 1.0):
                    raise Exception(f"Invalid clip_probe score_fake for {s_id}")

            fake_prob = sum(scores) / len(scores)
            classifier_pred = "fake" if fake_prob >= 0.5 else "real"
            inputs[s_id] = AssistedInput(
                sample_id=s_id,
                classifier_pred=classifier_pred,
                fake_probability=fake_prob,
                source_prediction_files=tuple(sources_by_sample[s_id]),
            )

        self._counts["assistant_inputs_built"] = len(inputs)
        return inputs

    def _run_inference(
        self,
        records: list[ManifestRecord],
        assistant_inputs: dict[str, AssistedInput],
        config: AppConfig,
        run_id: str,
        counts: dict,
    ) -> list[PredictionRecord]:
        cfg = config.baselines.assisted_qwen

        cache_enabled = cfg.cache_outputs
        cache_dir = config.paths.cache_root / "assisted_qwen" / "raw_outputs"

        if cache_enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)

        model = None
        processor = None
        device = None

        predictions = []

        for record in records:
            if record.sample_id not in assistant_inputs:
                raise Exception(
                    f"Missing clip_probe assistant prediction for sample_id={record.sample_id}"
                )

            assist_in = assistant_inputs[record.sample_id]
            prompt_text = get_assisted_prompt(
                cfg.prompt_id,
                classifier_pred=assist_in.classifier_pred,
                fake_probability=assist_in.fake_probability,
            )

            if not record.path.exists():
                raise Exception(f"Image missing: {record.path}")

            checksum = record.checksum
            if checksum:
                csum = hashlib.sha256(record.path.read_bytes()).hexdigest()
                if csum != checksum:
                    raise Exception(f"Checksum mismatch for {record.path}")
            else:
                checksum = hashlib.sha256(record.path.read_bytes()).hexdigest()

            def _get_cache_key(c_sum=checksum, a_in=assist_in) -> str:
                return cache_key(
                    {
                        "baseline": "assisted_qwen",
                        "sample_checksum": c_sum,
                        "base_model_id": cfg.base_model_id,
                        "prompt_id": cfg.prompt_id,
                        "assistant_source": cfg.assistant_source,
                        "classifier_pred": a_in.classifier_pred,
                        "fake_probability": format(a_in.fake_probability, ".12g"),
                        "temperature": str(cfg.temperature),
                        "max_new_tokens": str(cfg.max_new_tokens),
                        "output_cache_version": "assisted_qwen_raw_v1",
                    }
                )

            raw_output = None
            cache_path = None

            if cache_enabled:
                ckey = _get_cache_key()
                cache_path = cache_dir / f"{ckey}.json"
                raw_output = read_qwen_cache(cache_path, record.sample_id, counts)
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
                        if cfg.allow_deferred:
                            raise BaselineDeferredError(f"Missing Qwen dependencies: {e}") from e
                        else:
                            raise Exception(f"Missing Qwen dependencies: {e}") from e

                    device = get_qwen_device(
                        config.runtime.device, cfg.allow_deferred, BaselineDeferredError
                    )
                    model, processor = load_model(
                        cfg.base_model_id, device, cfg.allow_deferred, BaselineDeferredError
                    )

                raw_output = generate_one_image(
                    model, processor, record.path, prompt_text, device, cfg.max_new_tokens
                )

                if cache_enabled and cache_path is not None:
                    write_qwen_cache(cache_path, record.sample_id, raw_output)

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
                    model_name="assisted_qwen",
                    source=record.source,
                    run_id=run_id,
                    dataset=record.dataset,
                    split=record.split,
                    path=record.path,
                    checksum=record.checksum,
                    prompt_id=cfg.prompt_id,
                    raw_output=raw_output,
                    explanation=parse_result.explanation,
                    parse_status=parse_result.parse_status,
                )
            )

        return predictions
