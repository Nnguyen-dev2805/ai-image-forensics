# Task 6 Spec: Run Artifacts and Cache Keys

## Goal

Implement the shared run-artifact and deterministic cache-key infrastructure for Phase A/B.

Task 6 is the bridge between the schema/evaluation foundation from Tasks 1-5 and the real baseline adapters that start in Task 7. It must give every later baseline a consistent way to:

- create an isolated run directory,
- record the runtime environment,
- record completed/failed/deferred run status,
- build stable cache keys from semantic inputs.

Task 6 must not implement CLIP, Qwen-VL, assisted Qwen, NPR inference, prediction generation, report rendering, dataset downloads, external repository cloning, or cache value storage.

## Prerequisites

Task 6 depends on Tasks 1-5 being complete.

Before starting Task 6, verify the current behavioral foundation:

```bash
uv run pytest
```

Expected: all existing Task 1-5 tests pass before Task 6 changes are introduced.

## Required Reading

Before coding, read:

- `AGENTS.md`
- `CLAUDE.md` when using Claude Code
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/schemas/predictions-jsonl.md`
- `docs/specs/task-5-metrics-spec.md`
- `src/aiforensics/config/models.py`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`

## Files To Create

```text
src/aiforensics/runs/__init__.py
src/aiforensics/runs/artifacts.py
src/aiforensics/cache/__init__.py
src/aiforensics/cache/keys.py
tests/test_run_artifacts.py
tests/test_cache_keys.py
```

Do not modify CLI behavior in Task 6. `aiforensics run --baseline ...` remains the responsibility of Task 7 and later baseline tasks.

## Public Interface

Expose these imports from `src/aiforensics/runs/__init__.py`:

```python
from aiforensics.runs.artifacts import (
    RunStatus,
    create_run_dir,
    write_environment,
    write_status,
)
```

Expose this import from `src/aiforensics/cache/__init__.py`:

```python
from aiforensics.cache.keys import cache_key
```

Implement:

```python
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel


class RunStatus(BaseModel):
    baseline: str
    status: Literal["completed", "failed", "deferred"]
    reason: str | None = None
    command: list[str]
    started_at: str
    ended_at: str


def create_run_dir(
    output_root: Path,
    baseline: str,
    run_name: str | None = None,
) -> Path:
    ...


def write_environment(path: Path) -> None:
    ...


def write_status(path: Path, status: RunStatus) -> None:
    ...


def cache_key(parts: Mapping[str, str]) -> str:
    ...
```

Keep these helpers model-agnostic. Nothing in these modules should import OpenCLIP, Transformers, Torch, NPR code, or baseline adapters.

## Run Status Contract

`RunStatus` is the shared metadata object used when a run finishes or cannot produce predictions.

Required fields:

| Field | Meaning |
| --- | --- |
| `baseline` | Baseline id such as `clip_probe`, `qwen_vl`, `assisted_qwen`, or `npr`. |
| `status` | Exactly one of `completed`, `failed`, or `deferred`. |
| `reason` | Human-readable explanation; normally `None` for a successful run. |
| `command` | CLI argv as a list of strings. Keep it structured rather than shell-escaped. |
| `started_at` | UTC ISO-8601 timestamp. |
| `ended_at` | UTC ISO-8601 timestamp. |

Semantic meaning:

- `completed`: the baseline finished its intended run.
- `failed`: the baseline attempted to run and encountered a real execution error.
- `deferred`: the baseline intentionally did not run because a required environment capability, dependency, repository checkout, checkpoint, GPU, or model asset was unavailable.

Task 6 defines and serializes this contract only. Baseline-specific decisions about when to use `failed` versus `deferred` are implemented in later tasks.

## Run Directory Semantics

`create_run_dir()` must create one new directory directly below the configured output root.

Shape:

```text
<output_root>/<run_id>/
```

Use this run-id format:

```text
YYYYMMDDTHHMMSSffffffZ_<baseline>[_<run_name>]
```

Example:

```text
outputs/20260901T041530123456Z_clip_probe_seed70/
```

Requirements:

- timestamp is UTC,
- timestamp includes microseconds to reduce collisions,
- `baseline` is always included,
- `run_name` is appended only when provided and non-empty,
- path is created with parents as needed,
- the returned path is the created run directory,
- an existing run directory must never be silently reused or overwritten.

### Safe Name Rules

Normalize `baseline` and `run_name` into path-safe slugs:

- trim leading/trailing whitespace,
- lowercase,
- replace each run of characters outside `[a-z0-9._-]` with `-`,
- collapse repeated `-`,
- strip leading/trailing `-`, `_`, and `.`,
- reject a value that becomes empty after normalization.

Do not allow `/`, `\\`, `..`, or user-controlled absolute paths to escape `output_root`.

`run_name=None` means no custom suffix. An empty or whitespace-only explicit `run_name` should be rejected with `ValueError` rather than silently treated as `None`.

### Collision Behavior

Directory creation must be exclusive. If the generated path already exists, generate a fresh timestamp and retry a small bounded number of times. If a unique directory still cannot be created, raise `FileExistsError`.

Do not delete, clear, or reuse an existing run directory.

## Environment Metadata

`write_environment(path)` receives the exact target JSON path, normally:

```python
run_dir / "environment.json"
```

It must create the parent directory when needed and write UTF-8 JSON.

Required JSON shape:

```json
{
  "captured_at": "2026-09-01T04:15:30.123456+00:00",
  "python_version": "3.10.20",
  "python_executable": "/path/to/python",
  "platform": "macOS-...",
  "command": ["aiforensics", "run", "--baseline", "clip_probe"],
  "packages": {
    "numpy": "...",
    "pandas": "...",
    "pydantic": "...",
    "scikit-learn": "..."
  }
}
```

Implementation requirements:

- `captured_at` uses timezone-aware UTC ISO-8601,
- `python_version` comes from the running interpreter,
- `python_executable` comes from `sys.executable`,
- `platform` comes from the standard library `platform` module,
- `command` comes from `sys.argv`,
- `packages` uses `importlib.metadata.version()` rather than importing heavyweight packages,
- missing optional packages are skipped instead of raising,
- output uses `json.dumps(..., indent=2)` and ends with a newline,
- no environment variables, API keys, tokens, passwords, or full environment dumps are written.

Capture these package names when installed:

```text
numpy
pandas
pydantic
PyYAML
scikit-learn
Pillow
pytest
ruff
```

Later baseline tasks may extend environment metadata with model-specific package versions if needed. Task 6 should keep the base metadata lightweight and CPU-safe.

## Status Output

`write_status(path, status)` receives the exact target JSON path, normally:

```python
run_dir / "status.json"
```

It must:

- create the parent directory if needed,
- serialize `RunStatus` without changing field names,
- write UTF-8 JSON using `indent=2`,
- end the file with a newline,
- never write a partial temporary status shape.

Example:

```json
{
  "baseline": "qwen_vl",
  "status": "deferred",
  "reason": "CUDA device is unavailable",
  "command": [
    "aiforensics",
    "run",
    "--baseline",
    "qwen_vl"
  ],
  "started_at": "2026-09-01T04:15:30+00:00",
  "ended_at": "2026-09-01T04:15:31+00:00"
}
```

Do not create `predictions.jsonl` from this helper. Failure/deferred runs without predictions will be handled by later baseline adapters.

## Cache Key Contract

`cache_key(parts)` produces a deterministic SHA-256 key for expensive deterministic outputs.

Canonicalization algorithm:

1. require a non-empty mapping,
2. require every key and value to be a string,
3. sort by key,
4. serialize the sorted mapping as canonical JSON with no insignificant whitespace,
5. hash the UTF-8 bytes with SHA-256,
6. return the lowercase 64-character hexadecimal digest.

Recommended canonical serialization:

```python
canonical = json.dumps(
    dict(sorted(parts.items())),
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
)
```

Then:

```python
hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

The implementation must not build cache keys by naively concatenating strings because values containing separators can create ambiguous encodings.

### Determinism Requirements

These two inputs must produce the same key:

```python
cache_key({"model": "clip", "checksum": "abc"})
cache_key({"checksum": "abc", "model": "clip"})
```

Changing any key or value must change the digest with normal cryptographic-hash expectations.

### Intended Later Usage

Task 7 CLIP embedding cache keys should be composed from semantic inputs such as:

```text
sample_checksum
model_id
pretrained_id
preprocessing_config_hash
```

Task 8/9 MLLM raw-output cache keys should additionally include values such as:

```text
prompt_id
temperature
max_new_tokens
baseline_config_hash
```

Task 6 only provides the key builder. It must not implement cache directories, cache reads, cache writes, eviction, or locking.

## Artifact Boundary For Task 6

The full Phase A/B run artifact contract is:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  predictions.jsonl
  metrics.json
  metrics_by_source.csv
  logs.txt
```

or, when a baseline cannot produce predictions:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  status.json
  logs.txt
```

Task 6 implements only the shared primitives needed for:

```text
<run_id>/
environment.json
status.json
```

The following remain for later tasks:

- config snapshot writing,
- baseline logs,
- `predictions.jsonl`,
- model inference,
- CLI run orchestration,
- metrics invocation after a run,
- report generation.

This boundary prevents Task 6 from prematurely designing baseline behavior that belongs to Task 7-10.

## Tests: Run Artifacts

Create `tests/test_run_artifacts.py`.

Minimum tests:

1. `create_run_dir()` creates a directory under `tmp_path` and returns it.
2. The run directory name contains the normalized baseline.
3. A valid `run_name` appears in the run directory name.
4. `run_name=None` omits the custom suffix.
5. unsafe characters in baseline/run name are normalized without escaping the output root.
6. an empty baseline is rejected.
7. a whitespace-only explicit `run_name` is rejected.
8. an existing generated path is not silently reused.
9. `write_environment()` creates valid JSON with all required top-level keys.
10. `write_environment()` records `command` as a JSON list.
11. `write_environment()` skips unavailable package versions rather than failing.
12. `RunStatus` accepts exactly `completed`, `failed`, and `deferred` status values.
13. `RunStatus` rejects another status value such as `running`.
14. `write_status()` writes all six required fields.
15. `write_status()` preserves `reason=None` as JSON `null` for completed runs.

Use `tmp_path` and monkeypatching for filesystem/time/runtime observations. Tests must not create files under repository `outputs/`.

Do not assert exact Python executable paths, platform strings, or installed package versions because those vary between local, Colab, Kaggle, and CI environments.

## Tests: Cache Keys

Create `tests/test_cache_keys.py`.

Minimum tests:

1. the return value is a lowercase 64-character SHA-256 hex digest,
2. the same mapping produces the same key across repeated calls,
3. insertion order does not affect the key,
4. changing one value changes the key,
5. changing one key name changes the key,
6. values containing punctuation or separators do not create ambiguous collisions,
7. Unicode string values are deterministic,
8. an empty mapping is rejected with `ValueError`,
9. non-string keys are rejected,
10. non-string values are rejected.

These tests must not write cache files because Task 6 implements key construction only.

## Implementation Notes

Prefer standard-library modules for Task 6:

```python
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
```

Use Pydantic only for the small `RunStatus` contract because Pydantic is already part of the project and gives runtime validation for the status literal.

Avoid adding a general utility abstraction for JSON writing in Task 6. Two explicit small writers are easier to review and can be refactored later only if real duplication grows.

Keep all timestamps timezone-aware. Do not use naive `datetime.now()`.

Do not call `pip freeze`, `uv pip freeze`, subprocesses, network APIs, Git, CUDA, or model libraries from `write_environment()`.

## Ruff Scope

The repository currently has pre-existing Ruff debt outside Task 6. Keep Task 6 lint/format changes scoped to files created by this task.

Run:

```bash
uv run --extra dev ruff check \
  src/aiforensics/runs \
  src/aiforensics/cache \
  tests/test_run_artifacts.py \
  tests/test_cache_keys.py

uv run --extra dev ruff format --check \
  src/aiforensics/runs \
  src/aiforensics/cache \
  tests/test_run_artifacts.py \
  tests/test_cache_keys.py
```

Do not mass-format unrelated Task 1-5 files as part of Task 6.

## Verification

Task-specific verification:

```bash
uv run pytest tests/test_run_artifacts.py tests/test_cache_keys.py -v
```

Regression verification:

```bash
uv run pytest
```

Existing smoke behavior should remain intact:

```bash
uv run python -m aiforensics.cli.main prepare --config configs/phase_ab_smoke.yaml
uv run python -m aiforensics.cli.main evaluate --config configs/phase_ab_smoke.yaml
```

Expected:

- Task 6 tests pass,
- all existing Task 1-5 tests still pass,
- smoke `prepare` still succeeds,
- smoke `evaluate` still succeeds,
- no baseline model is downloaded or loaded,
- no repository `outputs/` artifacts are created by tests,
- no existing run directory is overwritten,
- no cache values are stored yet.

## Done Criteria

Task 6 is complete when:

- `create_run_dir()` creates isolated, path-safe run directories under the configured output root,
- existing run directories are never silently reused or overwritten,
- `RunStatus` represents exactly `completed`, `failed`, and `deferred` run outcomes,
- `write_environment()` writes portable, secret-safe runtime metadata,
- `write_status()` writes the architecture-required status fields,
- `cache_key()` returns deterministic order-independent SHA-256 digests from canonical string mappings,
- malformed cache-key input is rejected explicitly,
- Task 6 tests and all existing Task 1-5 tests pass,
- Ruff passes for Task 6 files,
- CLI run orchestration, model inference, prediction generation, config snapshots, logs, reporting, cache storage, and external integrations remain unimplemented.

