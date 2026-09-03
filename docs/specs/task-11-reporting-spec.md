# Task 11 Spec: Phase A/B Reporting

## Goal

Implement the Phase A/B reporting layer and turn:

```bash
aiforensics report --config configs/phase_ab.yaml
```

from a placeholder into a deterministic Markdown report built from artifacts produced by
Tasks 6-10.

Task 11 must:

- discover run artifacts under `config.paths.output_root`,
- select the latest relevant run for each baseline slot instead of mixing historical runs,
- summarize configured datasets and observed evaluation coverage,
- report completed/failed/deferred/missing baseline status,
- compare overall and per-source metrics,
- aggregate CLIP probe seed metrics without recomputing predictions,
- include failure/deferred notes when configured,
- include a deterministic sample of MLLM explanations when configured,
- produce a conservative next-step recommendation from Phase A/B evidence,
- write the configured Markdown filename under `output_root`,
- remain CPU-only, network-free, model-free, and deterministic.

Task 11 must not run a baseline, download anything, import heavy model runtimes, recompute
classification metrics from predictions, modify run artifacts, or invent a research success
threshold that is not defined by the project.

## Prerequisites

Task 11 depends on Tasks 1-10 being complete.

Before changing reporting code, verify:

```bash
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
uv run pytest
```

Expected before Task 11 changes:

- the repository is fully Ruff-clean,
- the full test suite is green,
- `status.json` uses the Task 6 `RunStatus` contract,
- completed runs write `predictions.jsonl`,
- `aiforensics evaluate` writes `metrics.json` and `metrics_by_source.csv` beside each completed
  prediction file,
- CLIP probe may have one run per configured seed,
- Qwen-VL, Assisted Qwen, and NPR have one run slot each,
- failed/deferred runs may legitimately have no metrics.

From Task 10 onward the verification gate is repository-wide. Task 11 must not return to
scoped-only Ruff verification as the final gate.

## Required Reading

Before coding, read:

- `AGENTS.md`
- `CLAUDE.md` when using Claude Code
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/specs/task-5-metrics-spec.md`
- `docs/specs/task-6-run-artifacts-cache-keys-spec.md`
- `docs/specs/task-7-clip-probe-baseline-spec.md`
- `docs/specs/task-8-qwen-vl-baseline-spec.md`
- `docs/specs/task-9-assisted-qwen-baseline-spec.md`
- `docs/specs/task-10-npr-external-adapter-spec.md`
- `src/aiforensics/config/models.py`
- `src/aiforensics/evaluation/metrics.py`
- `src/aiforensics/runs/artifacts.py`
- `src/aiforensics/schemas/predictions.py`
- `src/aiforensics/cli/main.py`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`

## Scope Boundary

Task 11 owns reporting only.

It may:

- read `status.json`, `metrics.json`, `metrics_by_source.csv`, and selected
  `predictions.jsonl` files,
- inspect the current `AppConfig`,
- aggregate already-computed metric values across CLIP seeds,
- format Markdown,
- write one report file under `output_root`.

It must not:

- call any baseline adapter,
- invoke Torch, Transformers, OpenCLIP, NPR, CUDA, or model-loading code,
- invoke Git or network access,
- call `aiforensics evaluate` internally,
- recompute accuracy/F1/AUROC from prediction records,
- change `metrics.json` or `metrics_by_source.csv`,
- change prediction/status schemas,
- add new training, evaluation, or benchmark logic,
- delete or mutate historical run directories,
- treat smoke metrics as scientific evidence,
- invent a fixed Phase A/B go/no-go threshold.

## Existing Artifact Contract

Completed run directory:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  predictions.jsonl
  metrics.json
  metrics_by_source.csv
  logs.txt
  status.json
```

Failed/deferred run directory may contain only:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  logs.txt
  status.json
```

`status.json` is authoritative and follows the existing `RunStatus` model:

```python
class RunStatus(BaseModel):
    baseline: str
    status: Literal["completed", "failed", "deferred"]
    reason: str | None
    command: list[str]
    started_at: str
    ended_at: str
```

`metrics.json` currently follows:

```json
{
  "total_records": 100,
  "overall": {
    "accuracy": 0.91,
    "balanced_accuracy": 0.90,
    "precision": 0.89,
    "recall": 0.92,
    "f1": 0.905,
    "auroc": 0.96
  }
}
```

`metrics_by_source.csv` columns are:

```text
source,n,accuracy,balanced_accuracy,precision,recall,f1,auroc
```

Task 11 consumes these contracts; it does not redefine them.

## Files

Create:

```text
src/aiforensics/reporting/__init__.py
src/aiforensics/reporting/markdown.py
tests/test_reporting.py
```

Modify:

```text
src/aiforensics/cli/main.py
```

Do not modify Tasks 5-10 implementation merely to make reporting easier unless a genuine contract
bug is discovered and separately justified.

## Public Reporting Interface

`src/aiforensics/reporting/markdown.py` must expose:

```python
from collections.abc import Sequence
from pathlib import Path

from aiforensics.config.models import AppConfig


class ReportingError(ValueError):
    pass


def discover_run_summaries(config: AppConfig) -> list[RunSummary]: ...


def build_phase_ab_report(
    config: AppConfig,
    runs: Sequence[RunSummary],
) -> str: ...


def write_phase_ab_report(
    config: AppConfig,
    report_text: str,
) -> Path: ...
```

The master plan's required interface remains:

```python
build_phase_ab_report(config: AppConfig, runs: list[RunSummary]) -> str
```

Accepting `Sequence[RunSummary]` is allowed because it is a compatible, more general input
contract.

## Reporting Models

Keep reporting-owned models local to the reporting package. They are not new cross-project
schemas.

Use an immutable internal representation equivalent to:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MetricValue = float | None
ReportStatus = Literal["completed", "failed", "deferred", "missing"]


@dataclass(frozen=True)
class SourceMetricRow:
    source: str
    n: int
    accuracy: MetricValue
    balanced_accuracy: MetricValue
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    auroc: MetricValue


@dataclass(frozen=True)
class RunSummary:
    baseline: str
    seed: int | None
    run_id: str | None
    status: ReportStatus
    reason: str | None
    run_dir: Path | None
    started_at: str | None
    ended_at: str | None
    total_records: int | None
    overall: dict[str, MetricValue] | None
    by_source: tuple[SourceMetricRow, ...]
    prediction_path: Path | None
```

`missing` is a reporting-only state meaning no relevant run artifact exists. It must never be
written back into Task 6 `RunStatus`.

## Expected Run Slots

The report must show every configured baseline, including disabled or not-yet-run baselines.

Expected slots are deterministic:

```text
clip_probe enabled  -> one slot for every configured seed, ascending
clip_probe disabled -> one slot with seed=None
qwen_vl             -> one slot
assisted_qwen       -> one slot
npr                  -> one slot
```

Baseline display/order must always be:

```text
clip_probe
qwen_vl
assisted_qwen
npr
```

CLIP seed rows are ordered by integer seed.

## Run Discovery And Historical Run Policy

`output_root` may contain many historical runs from repeated smoke or development executions.
Task 11 must not combine all of them.

Treat the configured `output_root` as the Phase A/B experiment boundary for Task 11.

Discovery rules:

1. Scan immediate child directories of `config.paths.output_root`.
2. A candidate run must contain `status.json`.
3. Parse `status.json` through the existing `RunStatus` model. Do not use ad-hoc dict access for
   status validation.
4. Ignore status records whose `baseline` is not one of the four Phase A/B baselines.
5. Qwen-VL, Assisted Qwen, and NPR candidates belong to their single baseline slots.
6. For enabled CLIP probe, identify the seed from the CLI-created run-id suffix:

   ```text
   ..._clip_probe_seed70
   ..._clip_probe_seed71
   ..._clip_probe_seed72
   ```

7. A CLIP run without a `seed<N>` suffix belongs only to the disabled `seed=None` slot.
8. For every expected slot, choose the candidate with the greatest parsed `ended_at` timestamp.
9. Break an exact timestamp tie by lexicographically greatest run-directory name.
10. Historical candidates not selected by this policy must not contribute metrics, notes, or
    explanation samples.
11. If no candidate exists for an expected slot, synthesize a `RunSummary(status="missing")`.

Do not choose an older completed run merely because the latest run failed, deferred, or lacks
metrics. The latest selected run is the truthful current result for that slot.

## Artifact Validation Policy

Reporting must fail loudly on malformed selected artifacts rather than silently creating a
misleading report.

For a selected `completed` run:

- `predictions.jsonl` must exist as a regular file,
- `metrics.json` must exist as a regular file,
- `metrics_by_source.csv` must exist as a regular file,
- otherwise raise `ReportingError` with a message telling the user to run
  `aiforensics evaluate --config <config>` when metrics are missing.

For a selected `failed` or `deferred` run:

- metrics are not required,
- stale metric files, if present, must not be included in comparisons.

For `metrics.json`:

- root must be an object,
- `total_records` must be an integer `>= 0`,
- `overall` must be an object,
- all six keys from `METRIC_NAMES` must exist,
- metric values must be either `null` or finite numeric values in `[0.0, 1.0]`.

For `metrics_by_source.csv`:

- require semantic columns `source`, `n`, and all six metric names; extra columns may be ignored,
- `source` must be non-empty,
- `n` must be an integer `>= 0`,
- metric values must be empty/NaN or finite numeric values in `[0.0, 1.0]`,
- duplicate `source` rows are invalid,
- rows are normalized into ascending `source` order.

Malformed JSON, malformed CSV, invalid values, or inconsistent selected artifact structure must
raise `ReportingError` with the affected path.

## Dataset Summary

The report must include a dataset summary without adding new manifest-processing logic.

Render configured datasets in this order:

```text
tiny_genimage
genimage_unseen
synthbuster
```

For each dataset show at least:

- dataset name,
- enabled/disabled,
- configured evaluation split/source descriptor,
- configured evaluation manifest path relative to the repository when practical.

For `tiny_genimage`, use `dev_manifest` as the evaluation manifest.

Also render observed evaluation coverage from completed selected runs:

```text
baseline / seed | total records | source counts
```

Source counts come from `metrics_by_source.csv`; do not reopen manifests to derive them.

If there are no completed selected runs, the configured dataset table still renders and observed
coverage says `N/A`.

If completed baselines disagree on total record count or source counts, do not hide the mismatch.
Add a visible coverage warning and let the recommendation logic treat the comparison as not ready.

## Baseline Status Table

Render one row per expected run slot with columns:

```text
Baseline | Seed | Configured | Status | Run ID | Reason
```

Rules:

- `Configured` is `enabled` or `disabled` from the current config.
- `Seed` is the integer CLIP seed or `N/A`.
- `Run ID` is the selected directory name or `N/A`.
- `Reason` is the selected status reason or `N/A`.
- `missing` remains visibly different from `deferred`.
- Never describe an absent run as successful.

Dynamic Markdown table cells must escape `|` and replace embedded newlines with spaces so a
runtime error message cannot corrupt the report table.

## Overall Metrics Table

Use the Task 5 metric order exactly:

```text
accuracy
balanced_accuracy
precision
recall
f1
auroc
```

Render values to four decimal places and use `N/A` for unavailable values.

### CLIP Seed Aggregation

CLIP probe is the only Phase A/B baseline with configured multi-seed execution.

Do not concatenate CLIP predictions and recompute metrics.

For each metric, aggregate the already-computed values across selected completed CLIP seed runs:

- exclude `None` values for that metric,
- zero valid values -> `N/A`,
- one valid value -> render that value only,
- two or more valid values -> render `mean +/- sample_std`,
- use `statistics.mean` and `statistics.stdev`,
- format mean and std to four decimals.

The CLIP row must show completion coverage such as:

```text
2/3 seeds completed
```

Do not claim a full multi-seed result when a configured seed is failed, deferred, or missing.

### Single-Run Baselines

Qwen-VL, Assisted Qwen, and NPR use metric values from their selected completed run directly.

Failed/deferred/missing baselines render `N/A` metrics.

## Per-Source Metrics Table

Render:

```text
Baseline | Source | n | Accuracy | Balanced Accuracy | Precision | Recall | F1 | AUROC
```

For Qwen-VL, Assisted Qwen, and NPR, use selected completed run source rows directly.

For CLIP probe:

1. Aggregate source rows by source across selected completed seeds.
2. All contributing completed seeds for a source must agree on `n`.
3. If the same source has different `n` values across CLIP seeds, raise `ReportingError`; seed
   aggregation would otherwise be scientifically misleading.
4. Aggregate each metric with the same `mean +/- sample_std` rules as overall CLIP metrics.

Sort by baseline order, then source name ascending.

## Failure And Deferred Notes

When:

```yaml
report:
  include_failure_notes: true
```

include a `Failure and Deferred Notes` section listing selected runs whose status is `failed`,
`deferred`, or `missing`.

Each note must identify baseline, seed when relevant, status, and reason.

`missing` uses:

```text
No run artifact found under the configured output_root.
```

When `include_failure_notes=false`, omit this section entirely. The status table still shows the
status and reason.

## Explanation Samples

When:

```yaml
report:
  include_explanations_sample: true
  explanation_sample_size: 20
```

include an `Explanation Samples` section.

Rules:

1. Read predictions only from selected completed `qwen_vl` and `assisted_qwen` runs.
2. Use existing `load_predictions()`.
3. Keep records with a non-empty `explanation`.
4. Never render full `raw_output`.
5. Sort candidates by `sample_id`, then baseline order (`qwen_vl` before `assisted_qwen`).
6. Take at most `explanation_sample_size` records total across both baselines.
7. Render at least baseline, sample id, true label, predicted label, parse status, and explanation.
8. Escape Markdown table/control characters in dynamic text.
9. If no explanation is available, render `No explanation samples available.`.

When `include_explanations_sample=false`, do not open prediction files solely for explanation
sampling and omit the section.

When `explanation_sample_size == 0`, render no samples even if the include flag is true.

## Next-Step Recommendation

The recommendation must be deterministic and conservative. Task 11 summarizes evidence; it must
not manufacture a publication claim.

### Smoke Phase

If:

```python
config.project.phase == "phase_ab_smoke"
```

the recommendation must explicitly state that smoke metrics validate pipeline behavior only and
must not be used as scientific evidence or as a later-phase go/no-go decision.

### Full Phase A/B Readiness

For a non-smoke Phase A/B config:

1. Build the set of expected slots whose baseline is currently enabled.
2. If any enabled slot is `failed`, `deferred`, or `missing`, state that comparative Phase A/B
   evidence is incomplete and identify unresolved slots.
3. If completed selected runs disagree on observed total/source coverage, state that evaluation
   coverage must be aligned before comparing baselines.
4. Do not select a winner while either condition above is true.

### Evidence Summary When Comparable

When all enabled expected slots are completed and observed coverage is aligned:

1. Identify the best observed baseline by `balanced_accuracy`.
2. For CLIP use aggregate mean balanced accuracy.
3. Break a balanced-accuracy tie by F1, then AUROC, then fixed baseline order.
4. State the best observed baseline and value without claiming statistical significance.
5. If Qwen-VL and Assisted Qwen are both enabled/completed with non-null balanced accuracy,
   compute:

   ```text
   assisted_delta = assisted_qwen_balanced_accuracy - qwen_vl_balanced_accuracy
   ```

6. If `assisted_delta > 0`, report the measured improvement and recommend prioritizing analysis of
   classifier-assisted fusion/evidence transfer in the next experiment.
7. If `assisted_delta <= 0`, report that classifier assistance did not improve observed balanced
   accuracy and recommend error analysis before treating assistance/fusion as validated.
8. Classification improvement must never be presented as proof of explanation faithfulness.

No hard rule such as `AUROC > 0.9 -> proceed` may be introduced in Task 11 because the research
plan defines no such threshold.

## Markdown Output Contract

Use these sections in this order:

```markdown
# Phase A/B Baseline Report

## Project
## Dataset Summary
## Baseline Status
## Overall Metrics
## Per-Source Metrics
## Failure and Deferred Notes        # conditional
## Explanation Samples               # conditional
## Next-Step Recommendation
```

`Project` must include project name, phase, description, and configured output root.

Do not add a wall-clock `generated_at` timestamp. The same artifacts and config should generate
byte-for-byte stable report content.

Do not embed absolute local paths when a path can be expressed relative to the repository or
output root. Reports should remain readable after copying artifacts between local, Colab, and
Kaggle.

Use plain ASCII generated formatting:

- `N/A` for unavailable values,
- `+/-` for mean/std,
- no decorative Unicode characters.

## Report Filename Safety

Output path:

```python
config.paths.output_root / config.report.filename
```

Before writing, require:

- filename is non-empty after stripping,
- filename is a basename only,
- filename is not absolute,
- filename contains no `..` traversal component,
- filename ends in `.md` case-insensitively.

These must fail with `ReportingError`:

```text
../report.md
/tmp/report.md
subdir/report.md
report.txt
```

Create `output_root` when absent. Do not create arbitrary directories from the report filename.

Write UTF-8 and end the report with exactly one newline.

## CLI Contract

Replace the current `_cmd_report()` placeholder while preserving:

```bash
aiforensics report --config <config>
```

Conceptually:

```python
def _cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        runs = discover_run_summaries(config)
        text = build_phase_ab_report(config, runs)
        path = write_phase_ab_report(config, text)
    except ReportingError as exc:
        print(f"Error generating report: {exc}")
        return 1

    ...
    return 0
```

Successful summary:

```text
[report] project=<name> phase=<phase> runs=<N> completed=<N> failed=<N> deferred=<N> missing=<N> path=<report_path>
```

Exit behavior:

- report successfully written -> exit `0`, even if experiment runs are failed/deferred/missing,
- malformed selected artifact -> exit `1`,
- completed selected run missing metrics -> exit `1`,
- unsafe report filename -> exit `1`.

Failed/deferred experiment outcomes are report content, not failures of the reporting command.

## Import And Runtime Safety

Importing or executing reporting code must not:

- import `torch`,
- import `transformers`,
- import `open_clip`,
- clone/fetch NPR,
- access network,
- initialize CUDA,
- instantiate a baseline adapter.

The reporting module may use the standard library, existing Pydantic config/status models, Pandas
if useful for CSV reading, and existing lightweight schema helpers.

## Tests

Create `tests/test_reporting.py` using `tmp_path` and synthetic artifacts. Tests must not depend on
the repository's existing `outputs/` history.

Default tests must be CPU-only, network-free, and model-free.

Minimum behavioral coverage:

1. required title and section order are stable,
2. expected baseline order is stable,
3. enabled CLIP creates one expected slot per configured seed,
4. disabled CLIP creates a seed-less slot,
5. missing run slot becomes reporting status `missing`,
6. latest Qwen run is selected over an older Qwen run,
7. latest failed run is selected over an older completed run,
8. CLIP seeds are selected independently,
9. `ended_at` drives selection and directory name is the tie-breaker,
10. unknown-baseline artifacts do not enter the report,
11. malformed selected `status.json` raises `ReportingError`,
12. completed run missing predictions raises `ReportingError`,
13. completed run missing metrics raises `ReportingError` with evaluate guidance,
14. failed/deferred run without metrics remains reportable,
15. malformed/out-of-range/NaN/infinite overall metrics fail,
16. malformed or duplicate per-source rows fail,
17. dataset config summary includes all three datasets,
18. observed coverage comes from metric artifacts,
19. coverage mismatch is visible,
20. single CLIP seed renders a scalar without fake std,
21. multiple CLIP seeds render mean plus sample std,
22. CLIP `None` metric values are excluded metric-by-metric,
23. partial CLIP completion shows completed/expected seed count,
24. CLIP per-source `n` mismatch across seeds fails,
25. single-run metrics render to four decimals and unavailable metrics use `N/A`,
26. per-source rows use fixed baseline/source order,
27. failure/deferred/missing notes obey config,
28. Markdown dynamic cells cannot break tables with `|` or newlines,
29. explanation section obeys config,
30. explanation sampling never renders `raw_output`,
31. explanation sampling is deterministic and capped,
32. zero explanation sample size emits no samples,
33. smoke recommendation rejects scientific interpretation,
34. incomplete full Phase A/B does not select a winner,
35. coverage mismatch prevents comparative recommendation,
36. comparable full runs identify best observed balanced accuracy,
37. Assisted Qwen positive delta is reported numerically,
38. Assisted Qwen non-positive delta does not claim improvement,
39. recommendation never equates classification gain with explanation faithfulness,
40. unsafe absolute/traversal/non-Markdown report filenames fail,
41. CLI writes exactly `config.report.filename` under `output_root`,
42. CLI returns `0` with deferred/missing experiment runs when report generation succeeds,
43. CLI returns `1` for malformed selected artifacts,
44. repeated render with identical inputs is byte-for-byte deterministic,
45. importing/reporting does not require Torch/Transformers/OpenCLIP/NPR runtime dependencies.

Do not chase a test-count number. This is a behavior contract; combine closely related cases with
parametrization where clearer.

## TDD Implementation Order

### Step 1: Reporting models and formatting helpers

Add `ReportingError`, `RunSummary`, `SourceMetricRow`, metric formatting, Markdown cell escaping,
and report filename validation. Write failing tests first.

### Step 2: Artifact parsing

Add narrow helpers for `RunStatus`, metrics JSON, and per-source CSV validation. The renderer must
not open JSON/CSV itself. Write malformed-artifact tests first.

### Step 3: Run discovery and latest-slot selection

Implement `discover_run_summaries(config)`. Cover:

```text
old completed -> new failed
old failed -> new completed
independent CLIP seed histories
missing slot
```

### Step 4: Pure Markdown rendering

`build_phase_ab_report()` consumes prepared `RunSummary` values and performs no filesystem
discovery. Add section/table tests before implementation.

### Step 5: CLIP aggregation and coverage checks

Use only values already present in selected metric artifacts. Cover mean/std, partial seeds,
per-source aggregation, and coverage mismatch.

### Step 6: Explanation sampling

Keep prediction reading narrow and only active when configured. Cover deterministic selection,
sample cap, and no `raw_output` leakage.

### Step 7: Recommendation rules

Cover smoke, incomplete, coverage mismatch, Assisted Qwen positive delta, and non-positive delta.
Do not introduce an arbitrary success threshold.

### Step 8: CLI and report writer

Replace `_cmd_report()` placeholder while keeping the public CLI unchanged. Add CLI success/error
tests.

### Step 9: Full verification

Do not stop after `tests/test_reporting.py` passes.

## Smoke Behavior

For `configs/phase_ab_smoke.yaml`:

- latest CLIP seed 70 may be `completed`,
- Qwen-VL may be `deferred`,
- Assisted Qwen may be `deferred`,
- NPR may be `deferred`,
- report still exits `0`,
- report contains their actual recorded statuses/reasons,
- metrics appear only for completed selected runs,
- recommendation explicitly says smoke results are not scientific evidence.

The report must not trigger Qwen/NPR dependency checks merely to explain an earlier deferred run;
it reads the recorded `status.json` reason.

## Verification Gate

Run:

```bash
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
uv run pytest

uv run aiforensics prepare --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline qwen_vl --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline assisted_qwen --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline npr --config configs/phase_ab_smoke.yaml
uv run aiforensics evaluate --config configs/phase_ab_smoke.yaml
uv run aiforensics report --config configs/phase_ab_smoke.yaml
```

Expected:

- Ruff check passes,
- Ruff format check passes,
- full pytest passes with no regression in Tasks 1-10,
- smoke prepare succeeds,
- smoke CLIP probe completes,
- smoke Qwen-VL/Assisted Qwen/NPR complete or defer according to existing contracts,
- evaluate exits `0`,
- report exits `0`,
- `outputs/smoke/phase_ab_smoke_report.md` exists,
- generated Markdown includes all required unconditional sections,
- `report` itself triggers no network/model work.

## Acceptance Criteria

Task 11 is complete when:

- `_cmd_report()` is no longer a placeholder,
- report discovery selects only the latest run per expected slot,
- historical runs do not contaminate comparisons,
- malformed selected artifacts fail loudly,
- failed/deferred/missing experiments remain reportable,
- completed runs require metrics produced by `evaluate`,
- CLIP seed aggregation reports mean/sample-std without recomputing metrics,
- partial CLIP completion is visible,
- per-source metrics are deterministic,
- configured and observed dataset coverage is visible,
- coverage mismatches are visible and block comparative recommendation,
- failure/deferred notes obey config,
- explanation sampling obeys config and never exposes `raw_output`,
- smoke report cannot be mistaken for research evidence,
- full Phase A/B recommendation is conservative and evidence-based,
- report filename cannot escape `output_root`,
- output is deterministic for identical inputs,
- reporting imports no heavyweight model runtime,
- full Ruff, full pytest, and smoke gate are green.

## Explicitly Deferred To Later Tasks

Task 11 does not implement:

- Colab/Kaggle notebooks or runbooks (Task 12),
- README quickstart/final project documentation (Task 13),
- HTML/PDF dashboards,
- charts/plots,
- confidence intervals or statistical significance tests,
- automatic experiment registry/database,
- historical trend visualization,
- new metrics such as EER,
- explanation correctness/faithfulness scoring,
- model training or fine-tuning,
- later research phases.

