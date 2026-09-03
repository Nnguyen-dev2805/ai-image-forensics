# Task 4 Spec: Prediction Schema

## Goal

Implement the prediction schema layer for Phase A/B. This layer is the shared contract every baseline adapter must write and every evaluator/report reader must consume.

This task must produce typed prediction records, JSONL read/write helpers, and validation helpers. It must not implement model inference, metrics, run artifact management, reporting, cache keys, external repository integration, or baseline adapters.

## Required Reading

Before coding, read:

- `AGENTS.md`
- `CLAUDE.md` when using Claude Code
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/schemas/predictions-jsonl.md`
- `src/aiforensics/config/models.py`
- `src/aiforensics/data/manifest.py`
- `tests/test_manifest.py`

## Files To Create Or Modify

```text
src/aiforensics/schemas/__init__.py
src/aiforensics/schemas/predictions.py
tests/test_predictions_schema.py
```

Do not modify CLI behavior in Task 4 unless a later review finds an import or packaging issue. Prediction schema is library code that later tasks will call.

## Public Interface

Expose these imports from `src/aiforensics/schemas/__init__.py`:

```python
from aiforensics.schemas.predictions import (
    PredictionError,
    PredictionRecord,
    PredictionValidationResult,
    load_predictions,
    validate_prediction_record,
    validate_predictions,
    write_predictions,
)
```

Implement:

```python
class PredictionError(ValueError):
    ...
```

Implement:

```python
class PredictionRecord(BaseModel):
    sample_id: str
    label_true: Literal["real", "fake"]
    label_pred: Literal["real", "fake", "unknown"]
    score_fake: float | None
    model_name: Literal["clip_probe", "qwen_vl", "npr", "assisted_qwen"]
    source: str

    run_id: str | None = None
    dataset: str | None = None
    split: str | None = None
    path: Path | None = None
    checksum: str | None = None

    prompt_id: str | None = None
    raw_output: str | None = None
    explanation: str | None = None
    parse_status: Literal["parsed", "recovered", "failed", "not_applicable"] | None = None
```

Implement:

```python
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
```

Implement:

```python
def validate_prediction_record(record: Mapping[str, Any]) -> PredictionRecord:
    ...

def write_predictions(records: Iterable[PredictionRecord], path: Path | str) -> None:
    ...

def load_predictions(path: Path | str) -> list[PredictionRecord]:
    ...

def validate_predictions(
    records: Iterable[PredictionRecord],
    *,
    manifest_sample_ids: set[str] | None = None,
    require_mllm_fields: bool = True,
) -> PredictionValidationResult:
    ...
```

`validate_predictions()` is included in Task 4 because `docs/schemas/predictions-jsonl.md` requires duplicate prediction checks and manifest sample id checks, which cannot be performed on one record at a time.

## Field Rules

Required fields:

```text
sample_id
label_true
label_pred
score_fake
model_name
source
```

Rules:

- `sample_id` must be a non-empty string.
- `label_true` must be `real` or `fake`.
- `label_pred` must be `real`, `fake`, or `unknown`.
- `score_fake` must be `None` or a number in `[0.0, 1.0]`.
- `model_name` must be `clip_probe`, `qwen_vl`, `npr`, or `assisted_qwen`.
- `source` must be a non-empty string.
- `checksum`, when present, must match SHA-256 hex format: 64 lowercase or uppercase hexadecimal characters.
- `path`, when present, should be accepted as `Path`.

## MLLM Field Rules

MLLM baselines are:

```text
qwen_vl
assisted_qwen
```

For `qwen_vl` and `assisted_qwen`, `validate_predictions(..., require_mllm_fields=True)` must require:

```text
prompt_id
raw_output
explanation
parse_status
```

Allowed `parse_status` values:

```text
parsed
recovered
failed
not_applicable
```

For `clip_probe` and `npr`, `parse_status` may be omitted or set to `not_applicable`.

Record-level validation may allow missing MLLM fields so baseline adapters can construct partial objects first. Dataset-level validation must report missing MLLM fields when `require_mllm_fields=True`.

## JSONL Contract

`write_predictions()` must write UTF-8 JSONL:

- one JSON object per line,
- no surrounding JSON array,
- create parent directories when needed,
- preserve input record order,
- exclude fields with value `None`,
- serialize `Path` values as strings,
- append one trailing newline when at least one record is written.

`load_predictions()` must:

- raise `PredictionError` when the file does not exist,
- raise `PredictionError` for invalid JSON with the line number,
- raise `PredictionError` for an invalid record with the line number,
- return an empty list for an existing empty file.

Invalid rows must not be silently skipped.

## Dataset-Level Validation

`validate_predictions()` must collect all detectable validation issues into `PredictionValidationResult.errors`.

It must check:

- duplicate `sample_id` values,
- every record `sample_id` is present in `manifest_sample_ids` when that argument is provided,
- MLLM fields are present for `qwen_vl` and `assisted_qwen` when `require_mllm_fields=True`,
- `parse_status` for non-MLLM models is either omitted or `not_applicable`,
- summary counts by model, true label, predicted label, and source.

It should not check whether every manifest sample has a prediction yet. That completeness check belongs in a later evaluation task when the intended evaluated split is known.

## Tests

Create `tests/test_predictions_schema.py`.

Minimum tests:

1. `validate_prediction_record()` accepts a valid CLIP record with required fields.
2. `validate_prediction_record()` accepts `score_fake=None`.
3. `validate_prediction_record()` rejects missing `sample_id` with `PredictionError`.
4. `validate_prediction_record()` rejects invalid `label_true`.
5. `validate_prediction_record()` rejects invalid `label_pred`.
6. `validate_prediction_record()` rejects `score_fake < 0.0`.
7. `validate_prediction_record()` rejects `score_fake > 1.0`.
8. `validate_prediction_record()` rejects invalid `model_name`.
9. `validate_prediction_record()` rejects invalid `checksum` when present.
10. `write_predictions()` and `load_predictions()` round-trip records through JSONL.
11. `load_predictions()` raises `PredictionError` with the line number for invalid JSON.
12. `load_predictions()` raises `PredictionError` with the line number for invalid record data.
13. `validate_predictions()` reports duplicate `sample_id` values.
14. `validate_predictions()` reports `sample_id` values missing from `manifest_sample_ids`.
15. `validate_predictions()` reports missing MLLM fields for `qwen_vl`.
16. `validate_predictions()` accepts `clip_probe` records without MLLM fields.
17. `validate_predictions()` reports non-MLLM `parse_status` values other than `not_applicable`.
18. `validate_predictions()` returns expected summary counts.

Use `tmp_path` for all JSONL files. Do not write into `outputs/` in Task 4 tests.

## Example Valid Records

CLIP probe:

```python
{
    "sample_id": "smoke/dev/real_0002",
    "label_true": "real",
    "label_pred": "real",
    "score_fake": 0.12,
    "model_name": "clip_probe",
    "source": "smoke",
    "dataset": "smoke",
    "split": "dev",
    "checksum": "a" * 64,
}
```

Qwen-VL:

```python
{
    "sample_id": "smoke/dev/fake_0002",
    "label_true": "fake",
    "label_pred": "fake",
    "score_fake": 0.91,
    "model_name": "qwen_vl",
    "source": "smoke",
    "prompt_id": "qwen_json_v1",
    "raw_output": "{\"label\":\"fake\",\"confidence\":0.91,\"evidence\":\"synthetic texture\"}",
    "explanation": "Synthetic texture cues are visible.",
    "parse_status": "parsed",
}
```

## Error Message Requirements

Use clear `PredictionError` messages.

Examples:

```text
Prediction file missing: <path>
Invalid JSON on line 3 in <path>: <parser message>
Invalid prediction record on line 4 in <path>: Field 'score_fake': Input should be less than or equal to 1
Invalid prediction record: Field 'label_pred': Input should be 'real', 'fake' or 'unknown'
```

The exact Pydantic wording may vary. Tests should assert stable fragments such as field names, path names, and line numbers.

## Verification

Run:

```bash
pytest tests/test_predictions_schema.py -v
pytest tests/test_manifest.py -v
pytest tests/test_config.py -v
pytest tests/test_cli_smoke.py -v
python -m aiforensics.cli.main prepare --config configs/phase_ab_smoke.yaml
```

Expected:

- prediction schema tests pass,
- existing Task 1-3 tests still pass,
- smoke `prepare` still prints `valid=True`,
- Task 4 does not add model inference, metrics, reporting, cache, run artifact, external clone, or dataset download behavior.

When using this project environment, prefer:

```bash
uv run pytest tests/test_predictions_schema.py -v
uv run pytest
uv run python -m aiforensics.cli.main prepare --config configs/phase_ab_smoke.yaml
```

because the local shell may not provide a `python` executable outside the uv-managed environment.

## Done Criteria

Task 4 is complete when:

- `PredictionRecord` models the required and optional JSONL fields,
- record-level validation rejects invalid labels, invalid model names, missing required fields, invalid checksums, and out-of-range scores,
- JSONL writing creates parent directories, preserves record order, excludes `None` fields, and serializes `Path`,
- JSONL loading rejects missing files, invalid JSON, and invalid rows with line numbers,
- dataset-level validation reports duplicate predictions,
- dataset-level validation reports predictions whose sample ids are not present in the manifest when manifest ids are provided,
- dataset-level validation enforces MLLM fields for Qwen-based records,
- dataset-level validation allows non-MLLM records without MLLM fields,
- all Task 4 tests and existing Task 1-3 tests pass,
- no baseline adapter, metrics, report, cache, external repo, real dataset, or model inference behavior is implemented.
