"""NPR external-baseline adapter: orchestration around the official NPR repo."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from aiforensics.baselines.base import RunResult
from aiforensics.baselines.npr import bridge
from aiforensics.baselines.npr import checkpoint as checkpoint_mod
from aiforensics.baselines.npr.checkout import (
    CheckoutInfo,
    ensure_npr_checkout,
    normalize_repo_url,
    resolve_repo_dir,
)
from aiforensics.baselines.npr.errors import (
    NPRConfigError,
    NPRDeferredError,
    NPRRuntimeError,
)
from aiforensics.baselines.npr.preprocess import PROFILE_NAME
from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import ManifestRecord, compute_sha256
from aiforensics.data.selection import selected_evaluation_manifests
from aiforensics.schemas.predictions import load_predictions, validate_predictions
from aiforensics.schemas.predictions import write_predictions as write_predictions_file

__all__ = ["NPRAdapter"]

logger = logging.getLogger(__name__)

_SMOKE_COMMIT_PLACEHOLDER = "smoke-disabled"
_COMMIT_SHA_LENGTH = 40


class NPRAdapter:
    """Phase A/B baseline adapter around the official pretrained NPR detector."""

    name = "npr"

    def run(
        self,
        *,
        config: AppConfig,
        output_dir: Path,
        run_id: str,
        seed: int | None = None,
    ) -> RunResult:
        npr_cfg = config.baselines.npr

        log_path = output_dir / "logs.txt"
        env_path = output_dir / "environment.json"
        status_path = output_dir / "status.json"

        def _log(line: str) -> None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        def _fail(reason: str) -> RunResult:
            _log(f"[FAILED] {reason}")
            self._remove_stale_predictions(output_dir)
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
            _log(f"[DEFERRED] {reason}")
            self._remove_stale_predictions(output_dir)
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
            _log("[COMPLETED] Run finished successfully.")
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

        # ---- 1. Disabled state must be validated before any placeholder checks. ----
        if not npr_cfg.enabled:
            return _defer("npr is disabled in config")

        # ---- 2. Commit pin validation (configuration failure, not deferral). ----
        repo_commit = (npr_cfg.repo_commit or "").strip().lower()
        if (
            not repo_commit
            or repo_commit == _SMOKE_COMMIT_PLACEHOLDER
            or len(repo_commit) != _COMMIT_SHA_LENGTH
            or any(c not in "0123456789abcdef" for c in repo_commit)
        ):
            return _fail(
                "Invalid or missing repo_commit for a real NPR run: "
                f"{npr_cfg.repo_commit!r}. A completed real run requires an exact "
                "40-character Git commit SHA of the official repository."
            )

        # ---- 3. Official repo URL validation. ----
        try:
            normalized_url = normalize_repo_url(npr_cfg.repo_url)
        except NPRConfigError as exc:
            return _fail(str(exc))
        repo_dir = resolve_repo_dir(config.paths.external_root)

        # ---- 4. Checkpoint existence (defer-eligible) before integrity checks. ----
        checkpoint_path = Path(npr_cfg.checkpoint_path)
        try:
            checkpoint_exists = checkpoint_path.exists() and checkpoint_path.is_file()
        except OSError as exc:
            return _fail(f"Could not inspect NPR checkpoint path: {exc}")
        if not checkpoint_exists:
            reason = f"NPR checkpoint not found: {checkpoint_path}"
            if npr_cfg.allow_deferred:
                return _defer(reason)
            return _fail(reason)

        # ---- 5. External checkout verification (narrow, mockable boundary). ----
        try:
            checkout_info: CheckoutInfo = ensure_npr_checkout(
                repo_dir=repo_dir,
                repo_url=normalized_url,
                repo_commit=repo_commit,
                allow_deferred=npr_cfg.allow_deferred,
            )
        except NPRDeferredError as exc:
            reason = f"NPR checkout unavailable: {exc}"
            if npr_cfg.allow_deferred:
                return _defer(reason)
            return _fail(reason)
        except NPRConfigError as exc:
            return _fail(f"NPR checkout verification failed: {exc}")

        # ---- 6. Checkpoint integrity. ----
        # A checksum mismatch is a hard failure (never deferred); a checkpoint
        # that vanished between the existence check and here defers only when
        # deferral is allowed.
        try:
            checksum_verified = checkpoint_mod.validate_checkpoint(
                checkpoint_path, npr_cfg.checkpoint_sha256
            )
        except FileNotFoundError as exc:
            if npr_cfg.allow_deferred:
                return _defer(str(exc))
            return _fail(str(exc))
        except ValueError as exc:
            return _fail(str(exc))
        _log(
            f"[INFO] checkpoint={checkpoint_path} checksum_verified={checksum_verified} "
            f"preprocessing_profile={PROFILE_NAME}"
        )

        # ---- 7. Evaluation record selection. ----
        try:
            eval_records = self._select_eval_records(config)
        except Exception as exc:
            return _fail(f"Evaluation record selection failed: {exc}")
        if not eval_records:
            return _fail("No evaluation records available for the NPR run")
        _log(f"[INFO] evaluation_samples={len(eval_records)}")

        # ---- 8. Image integrity. ----
        try:
            self._validate_images(eval_records)
        except Exception as exc:
            return _fail(str(exc))

        # ---- 9. Runtime availability (CUDA via torch; defer when unavailable). ----
        device = config.runtime.device
        try:
            resolved_device = self._check_runtime_availability(device)
        except NPRDeferredError as exc:
            reason = str(exc)
            if npr_cfg.allow_deferred:
                return _defer(reason)
            return _fail(reason)

        # ---- 10. Bridge artifacts + subprocess inference. ----
        input_path = output_dir / "npr_input.jsonl"
        scores_path = output_dir / "npr_scores.jsonl"
        try:
            rows = bridge.build_runtime_input_rows(eval_records)
            bridge.write_runtime_input_jsonl(rows, input_path)
        except Exception as exc:
            return _fail(f"Could not write NPR runtime input: {exc}")

        runtime_seed = config.runtime.seed if seed is None else seed
        command = [
            sys.executable,
            "-m",
            "aiforensics.baselines.npr.runtime",
            "--repo-dir",
            str(repo_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(scores_path),
            "--batch-size",
            str(npr_cfg.batch_size),
            "--seed",
            str(runtime_seed),
            "--device",
            resolved_device,
        ]
        _log(f"[INFO] runtime command: {' '.join(command)}")

        exit_code = self._run_subprocess(command, log_path)
        _log(f"[INFO] runtime exit code: {exit_code}")
        if exit_code != 0:
            if exit_code == 2:
                reason = "NPR runtime setup failed (model/CUDA/deps unavailable)"
                if npr_cfg.allow_deferred:
                    return _defer(reason)
                return _fail(reason)
            return _fail(f"NPR runtime inference failed with exit code {exit_code}")

        # ---- 11. Score validation and prediction mapping. ----
        try:
            score_rows = self._read_score_rows(scores_path)
            expected_ids = [r.sample_id for r in eval_records]
            scores = bridge.validate_runtime_scores(score_rows, expected_ids)
            predictions = bridge.build_npr_predictions(eval_records, scores, run_id=run_id)
        except NPRRuntimeError as exc:
            return _fail(f"Invalid NPR runtime scores: {exc}")

        validation = validate_predictions(
            predictions,
            manifest_sample_ids={r.sample_id for r in eval_records},
            require_mllm_fields=False,
        )
        if not validation.is_valid:
            return _fail(f"NPR prediction validation failed: {validation.errors}")

        prediction_path = output_dir / "predictions.jsonl"
        try:
            write_predictions_file(predictions, prediction_path)
        except Exception as exc:
            return _fail(f"Could not write predictions.jsonl: {exc}")

        try:
            loaded = load_predictions(prediction_path)
            recheck = validate_predictions(
                loaded,
                manifest_sample_ids={r.sample_id for r in eval_records},
                require_mllm_fields=False,
            )
        except Exception as exc:
            self._remove_stale_predictions(output_dir)
            return _fail(f"predictions.jsonl read-back validation failed: {exc}")
        if not recheck.is_valid:
            self._remove_stale_predictions(output_dir)
            return _fail(f"predictions.jsonl read-back invalid: {recheck.errors}")

        _log(
            f"[INFO] completed: baseline=npr repo_url={npr_cfg.repo_url} "
            f"repo_dir={repo_dir} configured_commit={repo_commit} "
            f"resolved_commit={checkout_info.resolved_commit} "
            f"checkout_action={checkout_info.action} "
            f"checkpoint={checkpoint_path} checksum_verified={checksum_verified} "
            f"preprocessing_profile={PROFILE_NAME} device={resolved_device} "
            f"seed={runtime_seed} batch_size={npr_cfg.batch_size} "
            f"samples={len(eval_records)}"
        )
        return _completed(prediction_path)

    # ------------------------------------------------------------------ helpers

    def _run_subprocess(self, command: list[str], log_path: Path) -> int:
        """Run the isolated NPR runtime subprocess; returns its exit code."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[FAILED] Could not start NPR runtime subprocess: {exc}\n")
            return 1
        if result.stdout:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(result.stdout)
        if result.stderr:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(result.stderr)
        return result.returncode

    def _read_score_rows(self, scores_path: Path) -> list[dict[str, object]]:
        if not scores_path.is_file():
            raise NPRRuntimeError(f"NPR runtime output missing: {scores_path}")
        rows: list[dict[str, object]] = []
        with open(scores_path, encoding="utf-8") as f:
            for i, line in enumerate(f.read().splitlines(), start=1):
                if not line.strip():
                    raise NPRRuntimeError(f"Blank runtime score row at line {i}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise NPRRuntimeError(
                        f"Malformed runtime score JSON at line {i}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise NPRRuntimeError(f"Runtime score row {i} is not a JSON object")
                rows.append(row)
        return rows

    def _check_runtime_availability(self, device: str) -> str:
        """Pre-flight the inference device without importing Torch in this process."""
        if device == "cpu":
            raise NPRDeferredError(
                "runtime.device=cpu is unsupported for real NPR inference in Task 10"
            )
        if device in ("auto", "cuda"):
            try:
                import importlib.util

                if importlib.util.find_spec("torch") is None:
                    raise NPRDeferredError("Torch is unavailable for NPR inference")
            except NPRDeferredError:
                raise
            except Exception as exc:
                raise NPRDeferredError(f"Could not inspect Torch availability: {exc}") from exc
            if device == "cuda":
                return "cuda"
            # auto: the runtime subprocess resolves auto -> cuda itself (it
            # imports torch anyway) and reports an unusable CUDA runtime via
            # exit code 2 (deferred when allowed).
            return "auto"
        raise NPRDeferredError(f"Unsupported runtime.device for NPR: {device!r}")

    def _select_eval_records(self, config: AppConfig) -> list[ManifestRecord]:
        """Select evaluation records with the shared Tasks 8/9/10 contract."""
        selection = selected_evaluation_manifests(config, strict=False)
        for message in selection.warnings:
            logger.warning("%s", message)

        if not selection.records:
            raise NPRConfigError(
                "No enabled evaluation manifest exists for the NPR run; "
                f"checked: {self._describe_candidates(config)}"
            )

        eval_records = list(selection.records)
        seen: set[str] = set()
        for record in eval_records:
            if record.sample_id in seen:
                raise NPRConfigError(
                    f"Duplicate evaluation sample_id across manifests: {record.sample_id}"
                )
            seen.add(record.sample_id)
        return eval_records

    @staticmethod
    def _describe_candidates(config: AppConfig) -> str:
        """Render the enabled-manifest candidates for a failure message."""
        datasets = config.datasets
        parts: list[str] = []
        if datasets.tiny_genimage.enabled:
            parts.append(f"tiny_genimage={datasets.tiny_genimage.dev_manifest}")
        if datasets.genimage_unseen.enabled:
            parts.append(f"genimage_unseen={datasets.genimage_unseen.manifest}")
        if datasets.synthbuster.enabled:
            parts.append(f"synthbuster={datasets.synthbuster.manifest}")
        return ", ".join(parts) if parts else "no dataset is enabled in config"

    def _validate_images(self, records: list[ManifestRecord]) -> None:
        for record in records:
            if not record.path.is_file():
                raise NPRConfigError(f"Missing image file: {record.path}")
            if record.checksum is not None:
                actual = compute_sha256(record.path)
                if actual != record.checksum:
                    raise NPRConfigError(
                        f"Checksum mismatch for {record.sample_id}: "
                        f"expected {record.checksum} got {actual}"
                    )

    @staticmethod
    def _remove_stale_predictions(output_dir: Path) -> None:
        predictions_path = output_dir / "predictions.jsonl"
        try:
            predictions_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove stale predictions.jsonl: %s", exc)
