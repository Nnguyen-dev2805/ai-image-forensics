"""Label-only inference adapter for the fine-tuned Qwen2.5-VL baseline (qwen_ft).

Mirrors the QwenVLAdapter flow with three differences: the base model is
wrapped in the PEFT LoRA adapter configured by ``adapter_uri``, generation is
capped at the label-only token budget, and predictions carry no score or
explanation because the fine-tuning stage trains label-only outputs.
"""

import hashlib
import logging
from pathlib import Path

from aiforensics.baselines.base import BaselineAdapter, RunResult
from aiforensics.baselines.qwen_ft.parsing import parse_qwen_ft_output
from aiforensics.baselines.qwen_vl.runtime import QwenOutOfMemoryError
from aiforensics.cache.keys import cache_key
from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import ManifestRecord
from aiforensics.data.selection import selected_evaluation_manifests
from aiforensics.finetune.qwen_data import INFERENCE_PROMPT_TEXT
from aiforensics.progress import progress_iter
from aiforensics.schemas.predictions import (
    PredictionRecord,
    validate_predictions,
    write_predictions,
)

logger = logging.getLogger(__name__)

# Special adapter URI that switches the adapter into CPU smoke mode: no model
# runtime is imported and deterministic labels are derived from fixture
# sample ids, so the smoke pipeline exercises the CLI contract without weights.
SMOKE_ADAPTER_URI = "smoke://adapter"


class BaselineDeferredError(Exception):
    """Raised when the environment cannot support the requested run."""


def compute_adapter_fingerprint(adapter_uri: str) -> str:
    """Compute a deterministic fingerprint for an adapter URI.

    If adapter_uri points to an existing local directory, fingerprint its files
    (names, sizes, mtimes) so retraining an adapter into the same path automatically
    invalidates cached outputs.
    """
    path = Path(adapter_uri)
    if path.is_dir():
        hasher = hashlib.sha256()
        marker_files = sorted(path.glob("adapter_*"))
        if not marker_files:
            marker_files = sorted(p for p in path.iterdir() if p.is_file())
        for f in marker_files:
            stat = f.stat()
            hasher.update(f.name.encode())
            hasher.update(str(stat.st_size).encode())
            hasher.update(str(stat.st_mtime_ns).encode())
        return hasher.hexdigest()
    if path.is_file():
        stat = path.stat()
        return hashlib.sha256(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    return hashlib.sha256(adapter_uri.encode()).hexdigest()


class QwenFTAdapter(BaselineAdapter):
    name = "qwen_ft"

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

        counts = {
            "parsed": 0,
            "recovered": 0,
            "failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        try:
            if not config.baselines.qwen_ft.enabled:
                raise BaselineDeferredError("qwen_ft is disabled in config")
            if not config.baselines.qwen_ft.adapter_uri:
                raise BaselineDeferredError(
                    "qwen_ft adapter_uri is not configured; train the LoRA adapter first"
                )
            if config.baselines.qwen_ft.temperature != 0.0:
                raise Exception("qwen_ft requires temperature == 0.0 for deterministic inference")

            records = self._load_manifests(config)
            if not records:
                raise Exception("No evaluation records found")

            predictions = self._run_inference(records, config, run_id, counts)

            # Label-only records: no evidence fields to enforce at this layer;
            # `evaluate` still enforces the MLLM fields qwen_ft always writes.
            val_result = validate_predictions(
                predictions,
                manifest_sample_ids={r.sample_id for r in records},
                require_mllm_fields=False,
            )
            if not val_result.is_valid:
                raise Exception(f"Prediction validation failed: {val_result.errors}")

            pred_file = run_dir / "predictions.jsonl"
            write_predictions(predictions, pred_file)

            from aiforensics.schemas.predictions import load_predictions

            loaded_preds = load_predictions(pred_file)
            val_loaded = validate_predictions(
                loaded_preds,
                manifest_sample_ids={r.sample_id for r in records},
                require_mllm_fields=False,
            )
            if not val_loaded.is_valid:
                raise Exception(f"Read-back prediction validation failed: {val_loaded.errors}")

            with open(run_dir / "logs.txt", "a", encoding="utf-8") as f:
                f.write(f"Run completed successfully for baseline {self.name}\n")
                f.write(f"Processed {len(records)} records.\n")
                f.write(f"Model ID: {config.baselines.qwen_ft.model_id}\n")
                f.write(f"Adapter URI: {config.baselines.qwen_ft.adapter_uri}\n")
                f.write(f"Prompt ID: {config.baselines.qwen_ft.prompt_id}\n")
                f.write(f"Compute dtype: {config.baselines.qwen_ft.dtype}\n")
                f.write(f"Min Pixels: {config.baselines.qwen_ft.min_pixels}\n")
                f.write(f"Max Pixels: {config.baselines.qwen_ft.max_pixels}\n")
                f.write(f"Cache Hits: {counts['cache_hits']}\n")
                f.write(f"Cache Misses: {counts['cache_misses']}\n")
                f.write(
                    f"Parsed: {counts['parsed']}, Recovered: {counts['recovered']}, "
                    f"Failed (Unknown): {counts['failed']}\n"
                )

            return RunResult(
                baseline=self.name,
                run_id=run_id,
                status="completed",
                output_dir=run_dir,
                prediction_path=pred_file,
                log_path=run_dir / "logs.txt",
                environment_path=run_dir / "environment.json",
                status_path=run_dir / "status.json",
                reason=None,
            )

        except QwenOutOfMemoryError as e:
            # Exhausted VRAM is an environment limit, not a broken run.
            return self._defer(config, run_dir, run_id, f"GPU out of memory: {e}")
        except BaselineDeferredError as e:
            return self._defer(config, run_dir, run_id, str(e))
        except Exception as e:
            logger.error("Run failed: %s", e)
            self._remove_partial_predictions(run_dir)
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

    def _defer(self, config: AppConfig, run_dir: Path, run_id: str, reason: str) -> RunResult:
        logger.info("Run deferred: %s", reason)
        self._remove_partial_predictions(run_dir)
        with open(run_dir / "logs.txt", "a", encoding="utf-8") as f:
            f.write(f"Run deferred: {reason}\n")
        return RunResult(
            baseline=self.name,
            run_id=run_id,
            status="deferred",
            output_dir=run_dir,
            prediction_path=None,
            log_path=run_dir / "logs.txt",
            environment_path=run_dir / "environment.json",
            status_path=run_dir / "status.json",
            reason=reason,
        )

    def _remove_partial_predictions(self, run_dir: Path) -> None:
        try:
            # A failed/deferred run must not leave a partial predictions file
            # that evaluate could mistake for a completed run's artifact.
            if (run_dir / "predictions.jsonl").exists():
                (run_dir / "predictions.jsonl").unlink()
        except Exception:
            pass

    def _load_manifests(self, config: AppConfig) -> list[ManifestRecord]:
        selection = selected_evaluation_manifests(config)
        for message in selection.warnings:
            logger.warning("%s", message)
        return list(selection.records)

    def _get_qwen_device(self, config: AppConfig) -> str:
        from aiforensics.baselines.qwen_vl.runtime import get_qwen_device

        return get_qwen_device(
            config.runtime.device,
            config.baselines.qwen_ft.allow_deferred,
            BaselineDeferredError,
        )

    def _resolve_adapter_path(self, config: AppConfig) -> str:
        """Return a local adapter directory, downloading a gs:// URI once."""
        adapter_cfg = config.baselines.qwen_ft
        if not adapter_cfg.adapter_uri.startswith("gs://"):
            return adapter_cfg.adapter_uri

        local_root = (
            Path(config.paths.cache_root)
            / "qwen_ft"
            / "adapters"
            / cache_key({"adapter_uri": adapter_cfg.adapter_uri})
        )
        marker = local_root / "adapter_config.json"
        if marker.is_file():
            return str(local_root)

        try:
            from google.cloud import storage
        except ImportError as e:
            message = "google-cloud-storage is required to fetch a gs:// adapter_uri"
            if adapter_cfg.allow_deferred:
                raise BaselineDeferredError(message) from e
            raise Exception(message) from e

        bucket_name, _, prefix = adapter_cfg.adapter_uri.removeprefix("gs://").partition("/")
        local_root.mkdir(parents=True, exist_ok=True)
        for blob in storage.Client().list_blobs(bucket_name, prefix=prefix.rstrip("/")):
            relative = blob.name[len(prefix) :].lstrip("/")
            if not relative:
                continue
            destination = local_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(destination)
        if not marker.is_file():
            message = f"No adapter_config.json under {adapter_cfg.adapter_uri}"
            if adapter_cfg.allow_deferred:
                raise BaselineDeferredError(message)
            raise Exception(message)
        return str(local_root)

    def _load_model(self, config: AppConfig, device: str):
        """Load the base model plus the PEFT LoRA adapter."""
        import importlib.util

        required = ("torch", "transformers", "qwen_vl_utils", "accelerate", "peft")
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            message = f"Missing qwen_ft dependencies: {missing}"
            if config.baselines.qwen_ft.allow_deferred:
                raise BaselineDeferredError(message)
            raise Exception(message)

        adapter_path = self._resolve_adapter_path(config)

        from aiforensics.baselines.qwen_vl.runtime import load_model

        base_model, processor = load_model(
            config.baselines.qwen_ft.model_id,
            device,
            config.baselines.qwen_ft.allow_deferred,
            BaselineDeferredError,
            dtype=config.baselines.qwen_ft.dtype,
            min_pixels=config.baselines.qwen_ft.min_pixels,
            max_pixels=config.baselines.qwen_ft.max_pixels,
        )

        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
        except Exception as e:
            message = f"LoRA adapter load failed from {adapter_path}: {e}"
            if config.baselines.qwen_ft.allow_deferred:
                raise BaselineDeferredError(message) from e
            raise Exception(message) from e
        return model, processor

    def _resolve_image_path(self, image_path: Path, data_root: Path) -> Path:
        if image_path.is_absolute():
            return image_path
        return data_root / image_path

    def _generate_one_image(
        self, model, processor, image_path: Path, prompt_text: str, device: str, max_new_tokens: int
    ) -> str:
        from aiforensics.baselines.qwen_vl.runtime import generate_one_image

        return generate_one_image(model, processor, image_path, prompt_text, device, max_new_tokens)

    def _smoke_generate(self, record: ManifestRecord) -> str:
        """Deterministic smoke output derived from the fixture sample id."""
        import json as _json

        label = "fake" if "fake" in record.sample_id else "real"
        return _json.dumps({"label": label}, separators=(",", ":"))

    def _get_cache_key(self, record: ManifestRecord, config: AppConfig) -> str:
        qwen_ft_cfg = config.baselines.qwen_ft
        checksum = record.checksum
        if not checksum:
            resolved_path = self._resolve_image_path(record.path, config.paths.data_root)
            if not resolved_path.exists():
                raise Exception(f"Image not found: {resolved_path}")
            checksum = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
        adapter_fingerprint = compute_adapter_fingerprint(qwen_ft_cfg.adapter_uri)
        return cache_key(
            {
                "baseline": "qwen_ft",
                "sample_checksum": checksum,
                "model_id": qwen_ft_cfg.model_id,
                "adapter_uri": qwen_ft_cfg.adapter_uri,
                "adapter_fingerprint": adapter_fingerprint,
                "prompt_id": qwen_ft_cfg.prompt_id,
                "dtype": qwen_ft_cfg.dtype,
                "temperature": str(qwen_ft_cfg.temperature),
                "max_new_tokens": str(qwen_ft_cfg.max_new_tokens),
                "min_pixels": str(qwen_ft_cfg.min_pixels),
                "max_pixels": str(qwen_ft_cfg.max_pixels),
                "output_cache_version": "qwen_ft_raw_v2",
            }
        )

    def _run_inference(
        self,
        records: list[ManifestRecord],
        config: AppConfig,
        run_id: str,
        counts: dict,
    ) -> list[PredictionRecord]:
        qwen_ft_cfg = config.baselines.qwen_ft
        logger.info(
            "qwen_ft image resolution bounds: min_pixels=%d, max_pixels=%d",
            qwen_ft_cfg.min_pixels,
            qwen_ft_cfg.max_pixels,
        )
        prompt_text = INFERENCE_PROMPT_TEXT
        smoke_mode = qwen_ft_cfg.adapter_uri == SMOKE_ADAPTER_URI
        cache_dir = Path(config.paths.cache_root) / "qwen_ft" / "raw_outputs"
        if qwen_ft_cfg.cache_outputs and not smoke_mode:
            cache_dir.mkdir(parents=True, exist_ok=True)

        model = None
        processor = None
        device = None

        from aiforensics.baselines.qwen_vl.cache import read_qwen_cache, write_qwen_cache

        predictions: list[PredictionRecord] = []
        progress_records = progress_iter("qwen_ft", records, log_every=50)
        for record in progress_records:
            resolved_path = self._resolve_image_path(record.path, config.paths.data_root)
            if not resolved_path.exists():
                raise Exception(f"Image missing: {resolved_path}")

            if record.checksum:
                actual = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
                if actual != record.checksum:
                    raise Exception(f"Checksum mismatch for {resolved_path}")

            raw_output = None
            cache_path = None
            if smoke_mode:
                raw_output = self._smoke_generate(record)
                counts["cache_misses"] += 1
            elif qwen_ft_cfg.cache_outputs:
                cache_path = cache_dir / f"{self._get_cache_key(record, config)}.json"
                raw_output = read_qwen_cache(cache_path, record.sample_id, counts)
            else:
                counts["cache_misses"] += 1

            if raw_output is None:
                if smoke_mode:
                    raise Exception("Smoke mode must always produce a raw output")
                if model is None:
                    device = self._get_qwen_device(config)
                    model, processor = self._load_model(config, device)
                raw_output = self._generate_one_image(
                    model,
                    processor,
                    resolved_path,
                    prompt_text,
                    device,
                    qwen_ft_cfg.max_new_tokens,
                )
                if cache_path is not None:
                    write_qwen_cache(cache_path, record.sample_id, raw_output)

            parse_result = parse_qwen_ft_output(raw_output)
            if parse_result.parse_status == "parsed":
                counts["parsed"] += 1
            elif parse_result.parse_status == "recovered":
                counts["recovered"] += 1
            else:
                counts["failed"] += 1

            progress_records.set_postfix(
                parsed=counts["parsed"],
                recovered=counts["recovered"],
                failed=counts["failed"],
            )

            predictions.append(
                PredictionRecord(
                    sample_id=record.sample_id,
                    label_true=record.label,
                    label_pred=parse_result.label_pred,
                    score_fake=None,
                    model_name="qwen_ft",
                    source=record.source,
                    run_id=run_id,
                    dataset=record.dataset,
                    split=record.split,
                    path=record.path,
                    checksum=record.checksum,
                    prompt_id=qwen_ft_cfg.prompt_id,
                    raw_output=raw_output,
                    explanation="",
                    parse_status=parse_result.parse_status,
                )
            )

        return predictions
