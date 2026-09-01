from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

from aiforensics.baselines.base import RunResult
from aiforensics.cache.keys import cache_key
from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import ManifestRecord, compute_sha256, load_manifest
from aiforensics.schemas.predictions import (
    PredictionRecord,
    validate_predictions,
    write_predictions,
)

logger = logging.getLogger(__name__)


class BaselineDeferredError(Exception):
    """Raised when a baseline run should be deferred (e.g., missing models, unsupported env)."""


def _smoke_image_embedding(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((8, 8))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    flat = pixels.reshape(-1)
    summary = np.array(
        [
            pixels[..., 0].mean(),
            pixels[..., 1].mean(),
            pixels[..., 2].mean(),
            pixels.std(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([flat, summary])


class ClipProbeAdapter:
    name: Literal["clip_probe"] = "clip_probe"

    def run(
        self,
        *,
        config: AppConfig,
        output_dir: Path,
        run_id: str,
        seed: int | None = None,
    ) -> RunResult:
        if seed is None:
            raise ValueError("Seed is required for clip_probe baseline")

        log_path = output_dir / "logs.txt"
        env_path = output_dir / "environment.json"
        status_path = output_dir / "status.json"

        def _fail(reason: str) -> RunResult:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[FAILED] {reason}\n")
            return RunResult(
                baseline=self.name,
                run_id=run_id,
                status="failed",
                output_dir=output_dir,
                log_path=log_path,
                environment_path=env_path,
                status_path=status_path,
                reason=reason,
            )

        def _defer(reason: str) -> RunResult:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[DEFERRED] {reason}\n")
            return RunResult(
                baseline=self.name,
                run_id=run_id,
                status="deferred",
                output_dir=output_dir,
                log_path=log_path,
                environment_path=env_path,
                status_path=status_path,
                reason=reason,
            )

        def _completed(prediction_path: Path) -> RunResult:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[COMPLETED] Run finished successfully.\n")
            return RunResult(
                baseline=self.name,
                run_id=run_id,
                status="completed",
                output_dir=output_dir,
                prediction_path=prediction_path,
                log_path=log_path,
                environment_path=env_path,
                status_path=status_path,
                reason=None,
            )

        # 1. Check if disabled
        if not config.baselines.clip_probe.enabled:
            return _defer("clip_probe is disabled in config")

        try:
            # 2. Extract dataset paths and validate
            train_records, eval_records = self._load_manifests(config, log_path)
            self._validate_records(train_records, eval_records)

            # 3. Check imports and get embeddings
            self._check_imports(config)
            x_train, y_train = self._load_data(train_records, config=config)
            x_eval, _ = self._load_data(eval_records, config=config)

            # 4. Logistic Regression
            classifier, probs = self._fit_and_predict(x_train, y_train, x_eval, seed)

            # 5. Prediction Generation & Validation
            predictions = self._generate_predictions(classifier, eval_records, probs, run_id)

            val_result = validate_predictions(predictions, require_mllm_fields=True)
            if not val_result.is_valid:
                return _fail(f"Prediction validation failed: {val_result.errors}")

            # 6. Prediction Writing
            pred_path = output_dir / "predictions.jsonl"
            write_predictions(predictions, pred_path)

            return _completed(pred_path)

        except BaselineDeferredError as e:
            return _defer(str(e))
        except Exception as e:
            return _fail(str(e))

    def _load_manifests(
        self, config: AppConfig, log_path: Path
    ) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
        data_root = config.paths.data_root
        train_manifest_path = config.datasets.tiny_genimage.train_manifest

        if not train_manifest_path.exists():
            raise Exception(f"Train manifest missing: {train_manifest_path}")

        try:
            train_records = load_manifest(train_manifest_path, data_root=data_root)
        except Exception as e:
            raise Exception(f"Train manifest validation failed: {e}") from e

        eval_records: list[ManifestRecord] = []

        # Check tiny_genimage dev
        dev_manifest_path = config.datasets.tiny_genimage.dev_manifest
        if dev_manifest_path.exists():
            try:
                eval_records.extend(load_manifest(dev_manifest_path, data_root=data_root))
            except Exception as e:
                raise Exception(f"Dev manifest validation failed: {e}") from e
        else:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[WARNING] Dev manifest missing: {dev_manifest_path}\n")

        # Check optional external datasets (Fix P2: checks for enabled flag)
        for opt_label, opt_enabled, opt_path in (
            (
                "genimage_unseen",
                config.datasets.genimage_unseen.enabled,
                config.datasets.genimage_unseen.manifest,
            ),
            (
                "synthbuster",
                config.datasets.synthbuster.enabled,
                config.datasets.synthbuster.manifest,
            ),
        ):
            if not opt_enabled:
                continue
            if not opt_path.exists():
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[WARNING] {opt_label} manifest missing: {opt_path}\n")
                continue
            try:
                eval_records.extend(load_manifest(opt_path, data_root=data_root))
            except Exception as e:
                raise Exception(f"{opt_label} manifest validation failed: {e}") from e

        if not eval_records:
            raise Exception("No evaluation manifest exists")

        return train_records, eval_records

    def _validate_records(
        self, train_records: list[ManifestRecord], eval_records: list[ManifestRecord]
    ) -> None:
        train_labels = {r.label for r in train_records}
        if "real" not in train_labels or "fake" not in train_labels:
            raise Exception("Training data requires both 'real' and 'fake' records")

        for r in train_records + eval_records:
            if not r.path.exists():
                raise Exception(f"Missing image file: {r.path}")
            if r.checksum is not None:
                actual = compute_sha256(r.path)
                if actual != r.checksum:
                    raise Exception(
                        f"Checksum mismatch for {r.sample_id}: expected {r.checksum} got {actual}"
                    )

    def _check_imports(self, config: AppConfig) -> None:
        model_family = config.baselines.clip_probe.model_family
        if model_family == "openclip":
            try:
                import open_clip  # noqa: F401
                import torch  # noqa: F401
            except ImportError as e:
                raise BaselineDeferredError(f"OpenCLIP or Torch unavailable: {e}") from e

    def _load_data(
        self,
        records: list[ManifestRecord],
        config: AppConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            model_family = config.baselines.clip_probe.model_family
            if model_family == "synthetic":
                embeddings = np.vstack([_smoke_image_embedding(r.path) for r in records])
            else:
                embeddings = self._get_openclip_embeddings(records, config=config)

            labels = np.array([1 if r.label == "fake" else 0 for r in records])
            return embeddings, labels
        except BaselineDeferredError:
            raise
        except Exception as e:
            raise Exception(f"Data loading/embedding failed: {e}") from e

    def _fit_and_predict(
        self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int
    ) -> tuple[LogisticRegression, np.ndarray]:
        classifier = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            solver="liblinear",
        )
        try:
            classifier.fit(x_train, y_train)
            probs = classifier.predict_proba(x_eval)
            return classifier, probs
        except Exception as e:
            raise Exception(f"Probe training or prediction failed: {e}") from e

    def _generate_predictions(
        self,
        classifier: LogisticRegression,
        eval_records: list[ManifestRecord],
        probs: np.ndarray,
        run_id: str,
    ) -> list[PredictionRecord]:
        fake_class_idx = list(classifier.classes_).index(1)
        predictions: list[PredictionRecord] = []

        for record, prob_row in zip(eval_records, probs, strict=True):
            score_fake = max(0.0, min(1.0, float(prob_row[fake_class_idx])))
            label_pred = "fake" if score_fake >= 0.5 else "real"

            predictions.append(
                PredictionRecord(
                    sample_id=record.sample_id,
                    label_true=record.label,
                    label_pred=label_pred,
                    score_fake=score_fake,
                    model_name=self.name,
                    source=record.source,
                    run_id=run_id,
                    dataset=record.dataset,
                    split=record.split,
                    path=record.path,
                    checksum=record.checksum,
                    parse_status="not_applicable",
                )
            )
        return predictions

    def _get_openclip_embeddings(
        self,
        records: list[ManifestRecord],
        config: AppConfig,
    ) -> np.ndarray:
        """Extract OpenCLIP embeddings in batches with optional cache."""
        import open_clip
        import torch

        cache_enabled = config.baselines.clip_probe.cache_embeddings
        batch_size = config.runtime.batch_size
        device = config.runtime.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=config.baselines.clip_probe.model_name,
                pretrained=config.baselines.clip_probe.pretrained,
                device=device,
            )
        except Exception as e:
            raise BaselineDeferredError(f"OpenCLIP model setup failure: {e}") from e
        model.eval()

        cache_dir = config.paths.cache_root / "clip_probe" / "embeddings"
        if cache_enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)

        cache_paths: list[Path | None] = []
        # Fix P1: Pre-allocate list and assign by index to maintain record order
        embeddings_by_index: list[np.ndarray | None] = [None] * len(records)
        to_compute_indices: list[int] = []

        for idx, record in enumerate(records):
            if cache_enabled and record.checksum is not None:
                key = cache_key(
                    {
                        "sample_checksum": record.checksum,
                        "model_family": config.baselines.clip_probe.model_family,
                        "model_name": config.baselines.clip_probe.model_name,
                        "pretrained": config.baselines.clip_probe.pretrained,
                        "embedding_version": "clip_probe_v1",
                    }
                )
                path = cache_dir / f"{key}.npy"
                cache_paths.append(path)
                if path.exists():
                    try:
                        embeddings_by_index[idx] = np.load(path)
                        continue
                    except Exception as e:
                        logger.warning(
                            "Failed loading cache for %s: %s; recomputing",
                            record.sample_id,
                            e,
                        )
                        # Corrupt cache — fall through to recompute and overwrite
                to_compute_indices.append(idx)
                continue
            cache_paths.append(None)
            to_compute_indices.append(idx)

        if not to_compute_indices:
            return np.vstack(embeddings_by_index)

        try:
            for batch_start in range(0, len(to_compute_indices), batch_size):
                batch_indices = to_compute_indices[batch_start : batch_start + batch_size]
                tensors = []
                for idx in batch_indices:
                    image = Image.open(records[idx].path).convert("RGB")
                    tensors.append(preprocess(image))
                batch_tensor = torch.stack(tensors).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(batch_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                batch_features = image_features.cpu().numpy().astype(np.float32)

                for offset, idx in enumerate(batch_indices):
                    emb = batch_features[offset].reshape(1, -1)
                    cache_path = cache_paths[idx]
                    if cache_enabled and cache_path is not None:
                        np.save(cache_path, emb.flatten())
                    embeddings_by_index[idx] = emb.flatten()
        except Exception as e:
            raise Exception(f"Failed embedding extraction: {e}") from e

        return np.vstack(embeddings_by_index)
