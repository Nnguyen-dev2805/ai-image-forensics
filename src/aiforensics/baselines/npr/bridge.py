"""Bridge contracts between the NPR adapter and its isolated runtime subprocess.

Covers runtime-input JSONL construction (no ground-truth labels), runtime-score
JSONL validation, and score-to-``PredictionRecord`` conversion. All helpers are
pure and free of Torch/NPR imports.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from aiforensics.baselines.npr.errors import NPRRuntimeError
from aiforensics.data.manifest import ManifestRecord
from aiforensics.schemas.predictions import PredictionRecord

__all__ = [
    "build_runtime_input_rows",
    "write_runtime_input_jsonl",
    "validate_runtime_scores",
    "build_npr_predictions",
    "label_pred_from_score",
]

NPR_DECISION_THRESHOLD = 0.5  # official NPR: fake iff sigmoid score > 0.5 (strict)


def build_runtime_input_rows(records: list[ManifestRecord]) -> list[dict[str, str]]:
    """Build one minimal row per evaluation record, preserving order.

    Rows contain only ``sample_id`` and an absolute resolved ``path``; ground
    truth never enters the runtime subprocess.
    """
    rows: list[dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "sample_id": record.sample_id,
                "path": str(record.path.resolve()),
            }
        )
    return rows


def write_runtime_input_jsonl(rows: list[dict[str, str]], path: Path) -> None:
    """Write the runtime-input JSONL bridge artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def validate_runtime_scores(
    rows: list[dict[str, object]],
    expected_ids: list[str],
) -> list[tuple[str, float]]:
    """Validate runtime score rows against the expected input sample ids.

    Requirements: exactly one row per input sample, no duplicate/unknown/missing
    sample ids, row order identical to the runtime input order, ``score_fake``
    numeric, finite, and within [0.0, 1.0].

    Returns validated ``(sample_id, score)`` pairs in input order.
    """
    expected_set = set(expected_ids)
    seen: set[str] = set()

    for i, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise NPRRuntimeError(f"Runtime score row {i} is not a JSON object")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise NPRRuntimeError(f"Runtime score row {i} has missing or invalid sample_id")
        if sample_id in seen:
            raise NPRRuntimeError(f"Duplicate runtime score sample_id: {sample_id}")
        if sample_id not in expected_set:
            raise NPRRuntimeError(f"Unknown runtime score sample_id: {sample_id}")

        score = row.get("score_fake")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise NPRRuntimeError(f"Runtime score for {sample_id} is not numeric: {score!r}")
        score_f = float(score)
        if not math.isfinite(score_f):
            raise NPRRuntimeError(f"Runtime score for {sample_id} is not finite: {score_f}")
        if not 0.0 <= score_f <= 1.0:
            raise NPRRuntimeError(
                f"Runtime score for {sample_id} out of range [0.0, 1.0]: {score_f}"
            )

        seen.add(sample_id)

    missing = [sid for sid in expected_ids if sid not in seen]
    if missing:
        raise NPRRuntimeError(f"Missing runtime scores for sample ids: {missing}")

    returned_ids = [row["sample_id"] for row in rows]
    if returned_ids != expected_ids:
        raise NPRRuntimeError(
            "Runtime score order misalignment: npr_scores.jsonl must preserve the "
            f"runtime input order (expected {expected_ids}, got {returned_ids})"
        )

    scores: list[tuple[str, float]] = []
    for expected_id, row in zip(expected_ids, rows, strict=True):
        score_value = row["score_fake"]
        if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
            raise NPRRuntimeError(f"Runtime score for {expected_id} is not numeric")
        scores.append((expected_id, float(score_value)))
    return scores


def label_pred_from_score(score_fake: float) -> Literal["fake", "real"]:
    """Official NPR decision semantics: fake iff score strictly greater than 0.5."""
    return "fake" if score_fake > NPR_DECISION_THRESHOLD else "real"


def build_npr_predictions(
    records: list[ManifestRecord],
    scores: list[tuple[str, float]],
    *,
    run_id: str,
) -> list[PredictionRecord]:
    """Convert validated runtime scores into shared ``PredictionRecord`` objects."""
    predictions: list[PredictionRecord] = []
    for record, (sample_id, score) in zip(records, scores, strict=True):
        if record.sample_id != sample_id:
            raise NPRRuntimeError(
                f"Score/order misalignment: expected {record.sample_id}, got {sample_id}"
            )
        predictions.append(
            PredictionRecord(
                sample_id=record.sample_id,
                label_true=record.label,
                label_pred=label_pred_from_score(score),
                score_fake=score,
                model_name="npr",
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
