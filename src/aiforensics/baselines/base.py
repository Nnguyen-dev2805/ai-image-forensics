"""Shared baseline protocol and run-result contract."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from aiforensics.config.models import AppConfig

__all__ = ["RunResult", "BaselineAdapter"]


class RunResult(BaseModel):
    baseline: str
    run_id: str
    status: Literal["completed", "failed", "deferred"]
    output_dir: Path
    prediction_path: Path | None = None
    log_path: Path
    environment_path: Path
    status_path: Path
    reason: str | None = None


class BaselineAdapter(Protocol):
    name: str

    def run(
        self,
        *,
        config: AppConfig,
        output_dir: Path,
        run_id: str,
        seed: int | None = None,
    ) -> RunResult: ...
