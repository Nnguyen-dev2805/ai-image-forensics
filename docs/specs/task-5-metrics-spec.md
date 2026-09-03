# Task 5 Spec: Metrics

## Goal

Implement the metrics layer for Phase A/B. This layer converts validated `PredictionRecord` objects into overall classification metrics and per-source metrics that later reporting code can consume.

This task must produce pure metric functions, metric output writing helpers, tests, and minimal `aiforensics evaluate` wiring for existing prediction files. It must not implement baseline adapters, model inference, run directory creation, cache keys, report rendering, external repository integration, or dataset download logic.

## Prerequisites

Task 5 depends on Task 4 being complete.

Before starting Task 5, verify these Task 4 behaviors:

```bash
uv run pytest tests/test_predictions_schema.py -v
```

Expected:

- missing `score_fake` is rejected,
- `score_fake=None` is accepted when the field is present,
- blank lines inside non-empty JSONL prediction files are rejected,
- valid prediction records still load and write correctly.

## Required Reading

Before coding, read:

- `AGENTS.md`
- `CLAUDE.md` when using Claude Code
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/schemas/predictions-jsonl.md`
- `docs/specs/task-4-prediction-schema-spec.md`
- `src/aiforensics/schemas/predictions.py`
- `src/aiforensics/cli/main.py`
- `configs/phase_ab_smoke.yaml`

## Files To Create Or Modify

```text
src/aiforensics/evaluation/__init__.py
src/aiforensics/evaluation/metrics.py
src/aiforensics/cli/main.py
tests/test_metrics.py
tests/test_cli_smoke.py
```

Use `tests/test_metrics.py` for metric and evaluate-output tests. Update `tests/test_cli_smoke.py` only if the `evaluate` placeholder changes and existing CLI tests need a new assertion.

## Public Interface

Expose these imports from `src/aiforensics/evaluation/__init__.py`:

```python
from aiforensics.evaluation.metrics import (
    METRIC_NAMES,
    MetricsError,
    compute_classification_metrics,
    compute_metrics_by_source,
    discover_prediction_files,
    evaluate_prediction_file,
    write_metrics_outputs,
)
```

Implement:

```python
class MetricsError(ValueError):
    ...
```

Implement:

```python
METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
)
```

Implement:

```python
def compute_classification_metrics(
    records: Sequence[PredictionRecord],
) -> dict[str, float | None]:
    ...

def compute_metrics_by_source(
    records: Sequence[PredictionRecord],
) -> pandas.DataFrame:
    ...

def write_metrics_outputs(
    records: Sequence[PredictionRecord],
    output_dir: Path | str,
) -> tuple[Path, Path]:
    ...

def evaluate_prediction_file(
    prediction_path: Path | str,
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    ...

def discover_prediction_files(output_root: Path | str) -> list[Path]:
    ...
```

`evaluate_prediction_file()` should load predictions through `load_predictions()`, validate them through `validate_predictions()`, compute metrics, and write output files. It should raise `MetricsError` when prediction validation fails.

`discover_prediction_files()` should return sorted `predictions.jsonl` paths under the configured output root. It must skip files named anything else.

## Metric Semantics

Use `fake` as the positive class.

Truth mapping:

```text
real -> 0
fake -> 1
```

Prediction mapping:

```text
real -> 0
fake -> 1
unknown -> no binary class
```

Definitions:

- `accuracy`: exact label match rate. `unknown` is always incorrect.
- `balanced_accuracy`: average of real recall and fake recall. Return `None` unless both true classes are present.
- `precision`: fake precision, computed as `TP / (TP + FP)`. Return `None` when there are no predicted-fake records.
- `recall`: fake recall, computed as `TP / (TP + FN)`. Return `None` when there are no true-fake records.
- `f1`: fake F1 from precision and recall. Return `None` when precision or recall is `None`, or when `precision + recall == 0`.
- `auroc`: ROC AUC over records with non-`None` `score_fake`. Return `None` unless scored records contain at least one true `real` and one true `fake`.

Counts:

```text
TP = label_true == fake and label_pred == fake
FP = label_true == real and label_pred == fake
TN = label_true == real and label_pred == real
FN = label_true == fake and label_pred != fake
```

For real recall in `balanced_accuracy`:

```text
real_recall = count(label_true == real and label_pred == real) / count(label_true == real)
```

For fake recall in `balanced_accuracy`:

```text
fake_recall = count(label_true == fake and label_pred == fake) / count(label_true == fake)
```

This makes `unknown` reduce accuracy and class recall without pretending it is `real`.

## Empty Input Behavior

For `compute_classification_metrics([])`, return:

```python
{
    "accuracy": None,
    "balanced_accuracy": None,
    "precision": None,
    "recall": None,
    "f1": None,
    "auroc": None,
}
```

For `compute_metrics_by_source([])`, return an empty `pandas.DataFrame` with columns:

```text
source,n,accuracy,balanced_accuracy,precision,recall,f1,auroc
```

## Per-Source Metrics

`compute_metrics_by_source()` must:

- group records by `source`,
- sort rows by `source` ascending for deterministic output,
- compute the same metric set as `compute_classification_metrics()`,
- include an integer `n` column with the number of records for that source,
- return columns in this order:

```text
source,n,accuracy,balanced_accuracy,precision,recall,f1,auroc
```

Use `None` or pandas missing values for metrics that cannot be computed.

## Output Files

`write_metrics_outputs(records, output_dir)` must create `output_dir` and write:

```text
metrics.json
metrics_by_source.csv
```

`metrics.json` must be UTF-8 JSON with this shape:

```json
{
  "total_records": 4,
  "overall": {
    "accuracy": 0.75,
    "balanced_accuracy": 0.75,
    "precision": 1.0,
    "recall": 0.5,
    "f1": 0.6666666666666666,
    "auroc": 1.0
  }
}
```

`metrics_by_source.csv` must contain the per-source DataFrame columns described above.

Do not write report Markdown in Task 5.

## CLI Integration

Update `aiforensics evaluate --config <config>` so it:

1. loads config with `load_config(args.config)`,
2. discovers prediction files under `config.paths.output_root` by calling `discover_prediction_files(config.paths.output_root)`,
3. for each discovered `predictions.jsonl`, writes `metrics.json` and `metrics_by_source.csv` beside that prediction file,
4. prints a concise summary:

```text
[evaluate] project=<project.name> phase=<project.phase> prediction_files=<count> output_root=<output_root>
```

If no prediction files are found, print the same summary with `prediction_files=0` and return exit code `0`. This preserves the smoke CLI gate until Task 6 and Task 7 create real run artifacts.

If any prediction file fails validation, raise `MetricsError` or return exit code `1` with the failing path in the message. Tests may assert the exception path or non-zero exit behavior, but the behavior must be explicit.

Task 5 must not create run ids or run directories. It only evaluates prediction files that already exist.

## Tests

Create `tests/test_metrics.py`.

Minimum tests:

1. `compute_classification_metrics()` returns all `None` metrics for an empty list.
2. Perfect real/fake predictions produce `accuracy=1.0`, `balanced_accuracy=1.0`, `precision=1.0`, `recall=1.0`, `f1=1.0`, and `auroc=1.0`.
3. A record with `label_pred="unknown"` counts as incorrect for `accuracy`.
4. A fake record with `label_pred="unknown"` counts against fake `recall`.
5. `precision` returns `None` when no record is predicted `fake`.
6. `recall` returns `None` when no true-fake records exist.
7. `balanced_accuracy` returns `None` when only one true class is present.
8. `auroc` returns `None` when every `score_fake` is `None`.
9. `auroc` returns `None` when scored records contain only one true class.
10. `compute_metrics_by_source()` returns sorted rows and the expected columns.
11. `write_metrics_outputs()` writes `metrics.json` and `metrics_by_source.csv`.
12. `evaluate_prediction_file()` loads `predictions.jsonl` and writes metrics beside it by default.
13. `discover_prediction_files()` returns only files named `predictions.jsonl`, sorted deterministically.
14. `aiforensics evaluate --config <tmp_smoke_config>` returns `0` when no prediction files exist.
15. `aiforensics evaluate --config <tmp_smoke_config>` writes metrics for an existing `predictions.jsonl` under the configured output root.

Use `tmp_path` for all generated prediction, metric, CSV, and temporary config files. Do not write Task 5 test artifacts into repository `outputs/`.

## Example Test Records

Use this helper in `tests/test_metrics.py`:

```python
from aiforensics.schemas.predictions import PredictionRecord


def prediction(
    sample_id: str,
    label_true: str,
    label_pred: str,
    score_fake: float | None,
    source: str = "smoke",
    model_name: str = "clip_probe",
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        label_true=label_true,
        label_pred=label_pred,
        score_fake=score_fake,
        model_name=model_name,
        source=source,
    )
```

Example perfect records:

```python
records = [
    prediction("real-1", "real", "real", 0.05),
    prediction("fake-1", "fake", "fake", 0.95),
]
```

Expected:

```python
{
    "accuracy": 1.0,
    "balanced_accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "auroc": 1.0,
}
```

Example unknown prediction:

```python
records = [
    prediction("real-1", "real", "real", 0.05),
    prediction("fake-1", "fake", "unknown", None),
]
```

Expected important values:

```python
metrics["accuracy"] == 0.5
metrics["balanced_accuracy"] == 0.5
metrics["recall"] == 0.0
metrics["precision"] is None
metrics["f1"] is None
metrics["auroc"] is None
```

## Implementation Notes

Use manual counts for accuracy, balanced accuracy, precision, recall, and F1 so `unknown` behavior stays explicit.

Use `sklearn.metrics.roc_auc_score` only for AUROC:

```python
from sklearn.metrics import roc_auc_score
```

Before calling `roc_auc_score`, filter to records where `score_fake is not None` and verify that both true classes are present.

Use `json.dumps(..., indent=2)` for `metrics.json`. Use `pandas.DataFrame.to_csv(index=False)` for `metrics_by_source.csv`.

## Verification

Run:

```bash
pytest tests/test_metrics.py -v
pytest tests/test_predictions_schema.py -v
pytest tests/test_manifest.py -v
pytest tests/test_config.py -v
pytest tests/test_cli_smoke.py -v
python -m aiforensics.cli.main prepare --config configs/phase_ab_smoke.yaml
python -m aiforensics.cli.main evaluate --config configs/phase_ab_smoke.yaml
```

Expected:

- metric tests pass,
- prediction schema tests still pass,
- existing Task 1-3 tests still pass,
- smoke `prepare` still prints `valid=True`,
- smoke `evaluate` returns `0` when no prediction files are present,
- Task 5 does not add model inference, baseline adapters, run directory creation, cache keys, report rendering, external repo clone, or dataset download behavior.

When using this project environment, prefer:

```bash
uv run pytest tests/test_metrics.py -v
uv run pytest
uv run python -m aiforensics.cli.main prepare --config configs/phase_ab_smoke.yaml
uv run python -m aiforensics.cli.main evaluate --config configs/phase_ab_smoke.yaml
```

because the local shell may not provide a `python` executable outside the uv-managed environment.

## Done Criteria

Task 5 is complete when:

- `compute_classification_metrics()` computes accuracy, balanced accuracy, precision, recall, F1, and AUROC according to this spec,
- invalid or unavailable metric cases return `None` instead of crashing,
- `unknown` predictions are handled according to the metric semantics above,
- `compute_metrics_by_source()` returns deterministic per-source rows with the required columns,
- `write_metrics_outputs()` writes `metrics.json` and `metrics_by_source.csv`,
- `evaluate_prediction_file()` evaluates one prediction file and writes metrics beside it by default,
- `discover_prediction_files()` finds sorted `predictions.jsonl` files under an output root,
- `aiforensics evaluate --config configs/phase_ab_smoke.yaml` preserves the CLI contract,
- all Task 5 tests and existing Task 1-4 tests pass,
- no baseline adapter, model inference, cache key, run artifact helper, report generator, external repo integration, real dataset, or download behavior is implemented.
