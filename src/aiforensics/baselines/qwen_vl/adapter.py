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
        from aiforensics.baselines.qwen_vl.runtime import get_qwen_device

        return get_qwen_device(
            config.runtime.device, config.baselines.qwen_vl.allow_deferred, BaselineDeferredError
        )

    def _load_model(self, config: AppConfig, device: str):
        from aiforensics.baselines.qwen_vl.runtime import load_model

        return load_model(
            config.baselines.qwen_vl.model_id,
            device,
            config.baselines.qwen_vl.allow_deferred,
            BaselineDeferredError,
        )

    def _generate_one_image(
        self, model, processor, image_path: Path, prompt_text: str, device: str, max_new_tokens: int
    ) -> str:
        from aiforensics.baselines.qwen_vl.runtime import generate_one_image

        return generate_one_image(model, processor, image_path, prompt_text, device, max_new_tokens)

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
                from aiforensics.baselines.qwen_vl.cache import read_qwen_cache

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
                    from aiforensics.baselines.qwen_vl.cache import write_qwen_cache

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
