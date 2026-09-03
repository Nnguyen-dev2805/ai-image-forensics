import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class PredictionError(ValueError):
    """Exception raised for prediction structural errors or invalid JSON."""

    pass


class PredictionRecord(BaseModel):
    sample_id: str = Field(min_length=1)
    label_true: Literal["real", "fake"]
    label_pred: Literal["real", "fake", "unknown"]
    score_fake: float | None = Field(ge=0.0, le=1.0)
    model_name: Literal["clip_probe", "qwen_vl", "npr", "assisted_qwen"]
    source: str = Field(min_length=1)

    run_id: str | None = None
    dataset: str | None = None
    split: str | None = None
    path: Path | None = None
    checksum: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")

    prompt_id: str | None = None
    raw_output: str | None = None
    explanation: str | None = None
    parse_status: Literal["parsed", "recovered", "failed", "not_applicable"] | None = None


class PredictionValidationResult(BaseModel):
    is_valid: bool
    total_records: int
    records_by_model: dict[str, int]
    records_by_label_true: dict[str, int]
    records_by_label_pred: dict[str, int]
    records_by_source: dict[str, int]
    duplicate_sample_ids: list[str]
    missing_manifest_sample_ids: list[str]
    errors: list[str]
    warnings: list[str]


def validate_prediction_record(record: Mapping[str, Any]) -> PredictionRecord:
    try:
        return PredictionRecord(**record)
    except ValidationError as e:
        # Extract the first error message for simpler error reporting
        err = e.errors()[0]
        loc = err["loc"][0] if err["loc"] else "unknown"
        msg = err["msg"]
        raise PredictionError(f"Invalid prediction record: Field '{loc}': {msg}") from e


def write_predictions(records: Iterable[PredictionRecord], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        for record in records:
            # Pydantic's model_dump with mode='json' serializes Path to string natively
            row = record.model_dump(mode="json")
            # Exclude None values except for required fields like score_fake
            row_clean = {k: v for k, v in row.items() if v is not None or k == "score_fake"}
            f.write(json.dumps(row_clean) + "\n")


def load_predictions(path: Path | str) -> list[PredictionRecord]:
    p = Path(path)
    if not p.is_file():
        raise PredictionError(f"Prediction file missing: {p}")

    if p.stat().st_size == 0:
        return []

    records = []
    with open(p, encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        return []

    for i, line in enumerate(lines, start=1):
        if not line.strip():
            raise PredictionError(f"Invalid JSON on line {i} in {p}: blank line not allowed")

        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise PredictionError(f"Invalid JSON on line {i} in {p}: {e}") from e

        try:
            record = PredictionRecord(**row)
            records.append(record)
        except ValidationError as e:
            err = e.errors()[0]
            loc = err["loc"][0] if err["loc"] else "unknown"
            msg = err["msg"]
            raise PredictionError(
                f"Invalid prediction record on line {i} in {p}: Field '{loc}': {msg}"
            ) from e

    return records


def validate_predictions(
    records: Iterable[PredictionRecord],
    *,
    manifest_sample_ids: set[str] | None = None,
    require_mllm_fields: bool = True,
) -> PredictionValidationResult:
    from collections import defaultdict

    records_by_model: dict[str, int] = defaultdict(int)
    records_by_label_true: dict[str, int] = defaultdict(int)
    records_by_label_pred: dict[str, int] = defaultdict(int)
    records_by_source: dict[str, int] = defaultdict(int)

    seen_ids = set()
    duplicate_sample_ids = []
    missing_manifest_sample_ids = []
    errors = []
    warnings = []

    total_records = 0
    mllm_models = {"qwen_vl", "assisted_qwen"}

    for record in records:
        total_records += 1
        records_by_model[record.model_name] += 1
        records_by_label_true[record.label_true] += 1
        records_by_label_pred[record.label_pred] += 1
        records_by_source[record.source] += 1

        if record.sample_id in seen_ids:
            duplicate_sample_ids.append(record.sample_id)
            errors.append(f"Duplicate sample_id found: {record.sample_id}")
        else:
            seen_ids.add(record.sample_id)

        if manifest_sample_ids is not None and record.sample_id not in manifest_sample_ids:
            missing_manifest_sample_ids.append(record.sample_id)
            errors.append(
                f"Prediction for sample_id '{record.sample_id}' not found in manifest_sample_ids."
            )

        if require_mllm_fields and record.model_name in mllm_models:
            missing = []
            if record.prompt_id is None:
                missing.append("prompt_id")
            if record.raw_output is None:
                missing.append("raw_output")
            if record.explanation is None:
                missing.append("explanation")
            if record.parse_status is None:
                missing.append("parse_status")
            if missing:
                errors.append(
                    f"MLLM fields missing for model '{record.model_name}' "
                    f"(sample {record.sample_id}): {missing}"
                )

        if record.model_name not in mllm_models:
            if record.parse_status not in (None, "not_applicable"):
                errors.append(
                    f"Model '{record.model_name}' (sample {record.sample_id}) has "
                    f"invalid parse_status '{record.parse_status}'. "
                    "Should be None or 'not_applicable'."
                )

    return PredictionValidationResult(
        is_valid=len(errors) == 0,
        total_records=total_records,
        records_by_model=dict(records_by_model),
        records_by_label_true=dict(records_by_label_true),
        records_by_label_pred=dict(records_by_label_pred),
        records_by_source=dict(records_by_source),
        duplicate_sample_ids=duplicate_sample_ids,
        missing_manifest_sample_ids=missing_manifest_sample_ids,
        errors=errors,
        warnings=warnings,
    )
