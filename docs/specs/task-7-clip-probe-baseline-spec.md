# Task 7 Spec: CLIP Probe Baseline

## Goal

Implement the first real Phase A/B baseline: `clip_probe`.

Task 7 turns `aiforensics run --baseline clip_probe --config ...` from a placeholder into a working baseline run. It must create run artifacts, train a logistic-regression probe over image embeddings, write valid `predictions.jsonl`, and keep the smoke path CPU-safe with no model downloads.

Task 7 must not implement Qwen-VL, assisted Qwen, NPR, report rendering, dataset downloading, external repository cloning, or explanation scoring.

## Prerequisites

Task 7 depends on Tasks 1-6 being complete.

Before starting Task 7, verify:

```bash
uv run pytest
uv run pytest tests/test_run_artifacts.py tests/test_cache_keys.py -v
```

Expected:

- existing Task 1-6 tests pass,
- `create_run_dir()`, `write_environment()`, `write_status()`, and `cache_key()` are available,
- `RunStatus` rejects non-UTC timestamps,
- Task 6 files pass their scoped Ruff checks.

## Required Reading

Before coding, read:

- `AGENTS.md`
- `CLAUDE.md` when using Claude Code
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/schemas/manifest.md`
- `docs/schemas/predictions-jsonl.md`
- `docs/specs/task-6-run-artifacts-cache-keys-spec.md`
- `src/aiforensics/cli/main.py`
- `src/aiforensics/config/models.py`
- `src/aiforensics/data/manifest.py`
- `src/aiforensics/schemas/predictions.py`
- `src/aiforensics/runs/artifacts.py`
- `src/aiforensics/cache/keys.py`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`

## Files To Create Or Modify

```text
src/aiforensics/baselines/__init__.py
src/aiforensics/baselines/base.py
src/aiforensics/baselines/clip_probe/__init__.py
src/aiforensics/baselines/clip_probe/adapter.py
src/aiforensics/cli/main.py
pyproject.toml
tests/test_clip_probe_smoke.py
tests/test_cli_smoke.py
```

Modify `pyproject.toml` only for optional CLIP dependencies. Do not add OpenCLIP or Torch to default dependencies required by smoke tests.

## Public Interface

Expose these imports from `src/aiforensics/baselines/__init__.py`:

```python
from aiforensics.baselines.base import BaselineAdapter, RunResult
```

Expose these imports from `src/aiforensics/baselines/clip_probe/__init__.py`:

```python
from aiforensics.baselines.clip_probe.adapter import ClipProbeAdapter
```

Create `src/aiforensics/baselines/base.py`:

```python
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from aiforensics.config.models import AppConfig


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
    ) -> RunResult:
        ...
```

Implement `src/aiforensics/baselines/clip_probe/adapter.py`:

```python
class ClipProbeAdapter:
    name = "clip_probe"

    def run(
        self,
        *,
        config: AppConfig,
        output_dir: Path,
        run_id: str,
        seed: int | None = None,
    ) -> RunResult:
        ...
```

`seed` is required for `clip_probe`. If `seed is None`, raise `ValueError`.

## CLI Contract

Update only `aiforensics run --baseline clip_probe`.

These commands must remain valid:

```bash
aiforensics run --baseline clip_probe --config configs/phase_ab.yaml
aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
```

Task 7 behavior:

1. load config with `load_config(args.config)`,
2. if `args.baseline != "clip_probe"`, preserve the existing placeholder behavior for later tasks,
3. if `config.baselines.clip_probe.enabled` is false, create a deferred run artifact and return `0`,
4. for each seed in `config.baselines.clip_probe.seeds`, create one run directory,
5. copy the input config file to `run_dir / "config.yaml"`,
6. write `run_dir / "environment.json"` using `write_environment()`,
7. call `ClipProbeAdapter.run(...)`,
8. write `run_dir / "status.json"` using `write_status()` for completed, failed, and deferred runs,
9. write `run_dir / "logs.txt"` with concise human-readable run notes,
10. print a concise summary,
11. return `0` only when every seed run is `completed` or `deferred`,
12. return `1` when any seed run is `failed`.

Suggested summary:

```text
[run] baseline=clip_probe runs=<n> completed=<n> failed=<n> deferred=<n> output_root=<output_root>
```

For smoke config, the command should create exactly one seed run because `configs/phase_ab_smoke.yaml` currently has one seed.

## Run Directory And Artifact Contract

For each seed, create the run directory with:

```python
create_run_dir(config.paths.output_root, "clip_probe", run_name=f"seed{seed}")
```

The returned directory name is the `run_id`.

Completed CLIP probe run artifact:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  status.json
  predictions.jsonl
  logs.txt
```

Failed or deferred CLIP probe run artifact:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  status.json
  logs.txt
```

Do not write metrics in `run`. Metrics are produced by `aiforensics evaluate`.

Do not write report Markdown in Task 7.

## Data Selection

Use manifests from config, not hardcoded paths.

Training records:

```python
config.datasets.tiny_genimage.train_manifest
```

Evaluation records:

```python
config.datasets.tiny_genimage.dev_manifest
```

Additionally include these manifests when their dataset configs are enabled and the files exist:

```python
config.datasets.genimage_unseen.manifest
config.datasets.synthbuster.manifest
```

Task 7 does not download or build real datasets. It consumes manifest files that already exist.

Load manifests using:

```python
load_manifest(path, data_root=config.paths.data_root)
```

Validation rules:

- training manifest must exist,
- at least one evaluation manifest must exist,
- training records must include both `real` and `fake`,
- evaluation records may contain either or both labels,
- missing image files or checksum mismatches should fail the run with a clear `status.json` reason,
- duplicate prediction `sample_id` values must not be written.

If an enabled non-smoke external manifest is missing, record a warning in `logs.txt` and continue with the manifests that exist. If no evaluation manifest exists, mark the run `failed`.

## Smoke Embedding Path

When:

```python
config.baselines.clip_probe.model_family == "synthetic"
```

use a deterministic CPU-safe embedding path.

Requirements:

- no OpenCLIP import,
- no Torch import,
- no model download,
- no network access,
- no GPU requirement,
- image files are still opened through PIL so smoke tests exercise real image I/O,
- embeddings are deterministic for the same image bytes and config.

Recommended smoke feature extraction:

```python
from PIL import Image
import numpy as np


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
```

The smoke path does not need to achieve high accuracy. It only needs to produce valid predictions and exercise the full baseline pipeline.

## Real OpenCLIP Path

When:

```python
config.baselines.clip_probe.model_family == "openclip"
```

use OpenCLIP image embeddings.

Configuration values:

```python
config.baselines.clip_probe.model_name      # e.g. "ViT-L-14"
config.baselines.clip_probe.pretrained      # e.g. "openai"
config.baselines.clip_probe.cache_embeddings
config.runtime.device                       # "auto", "cpu", or an explicit device
config.runtime.batch_size
config.paths.cache_root
```

Real path requirements:

- import OpenCLIP and Torch lazily inside the real path only,
- do not import model libraries at module import time,
- use `open_clip.create_model_and_transforms(...)`,
- put the model in eval mode,
- use `torch.no_grad()` for embedding extraction,
- normalize image embeddings before training the classifier,
- process records in batches using `config.runtime.batch_size`,
- support CPU execution, even if slow,
- when `device == "auto"`, use CUDA only if Torch reports it available,
- if OpenCLIP, Torch, model weights, or device setup are unavailable, return a `deferred` run result with a clear reason rather than crashing,
- do not cache processed images.

Optional dependencies:

```toml
[project.optional-dependencies]
clip = [
    "open_clip_torch",
]
```

Do not add this to default dependencies. Smoke tests must pass without installing the `clip` extra.

## Embedding Cache

Use `cache_key()` from Task 6 for real OpenCLIP embeddings when:

```python
config.baselines.clip_probe.cache_embeddings is True
```

Cache directory:

```text
<config.paths.cache_root>/clip_probe/embeddings/
```

Cache file:

```text
<cache_key>.npy
```

Cache key parts:

```python
{
    "sample_checksum": record.checksum,
    "model_family": config.baselines.clip_probe.model_family,
    "model_name": config.baselines.clip_probe.model_name,
    "pretrained": config.baselines.clip_probe.pretrained,
    "embedding_version": "clip_probe_v1",
}
```

Requirements:

- use `numpy.save()` and `numpy.load()` for `.npy` files,
- create cache parent directories as needed,
- if a cache file is corrupt or unreadable, recompute the embedding and overwrite that cache file,
- never cache predictions as a replacement for writing `predictions.jsonl`,
- never write cache files during smoke tests when `cache_embeddings: false`.

## Linear Probe

Train a scikit-learn logistic regression classifier for each configured seed.

Label mapping:

```text
real -> 0
fake -> 1
```

Classifier:

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(
    random_state=seed,
    max_iter=1000,
    solver="liblinear",
)
```

Training requirements:

- train only on `config.datasets.tiny_genimage.train_manifest`,
- reject training data with fewer than two labels,
- use embeddings as `float32` or `float64` numeric arrays,
- fit one classifier per seed,
- do not shuffle records unless the shuffle is deterministic from `seed`.

Prediction requirements:

- predict every evaluation record,
- use `predict_proba()` to compute `score_fake`,
- map `score_fake >= 0.5` to `label_pred="fake"`, otherwise `label_pred="real"`,
- clamp only for tiny numeric drift outside `[0.0, 1.0]`,
- do not emit `unknown` for normal CLIP predictions,
- if a sample cannot be embedded, fail the run rather than silently dropping it.

## Prediction Records

Write predictions with `write_predictions()` from `src/aiforensics/schemas/predictions.py`.

Every CLIP prediction must include:

```python
PredictionRecord(
    sample_id=record.sample_id,
    label_true=record.label,
    label_pred="real" or "fake",
    score_fake=float_probability,
    model_name="clip_probe",
    source=record.source,
    run_id=run_id,
    dataset=record.dataset,
    split=record.split,
    path=record.path,
    checksum=record.checksum,
    parse_status="not_applicable",
)
```

Do not include MLLM-only fields:

```text
prompt_id
raw_output
explanation
```

After writing, load and validate the file:

```python
records = load_predictions(prediction_path)
result = validate_predictions(records, require_mllm_fields=True)
```

The validation must pass. `parse_status="not_applicable"` is valid for `clip_probe`.

## Failure And Deferred Behavior

Use `RunStatus` and `write_status()` for every run.

Completed:

```python
RunStatus(
    baseline="clip_probe",
    status="completed",
    reason=None,
    command=list(sys.argv),
    started_at=started_at,
    ended_at=ended_at,
)
```

Failed examples:

- train manifest missing,
- no evaluation manifest exists,
- manifest validation fails,
- training data contains only one class,
- prediction writing or validation fails,
- image file cannot be opened.

Deferred examples:

- config disables `clip_probe`,
- real OpenCLIP path is requested but OpenCLIP/Torch is unavailable,
- model weights cannot be resolved in the current environment.

`started_at` and `ended_at` must be UTC ISO-8601 strings accepted by `RunStatus`.

Failed/deferred runs must still write:

```text
config.yaml
environment.json
status.json
logs.txt
```

## Tests

Create `tests/test_clip_probe_smoke.py`.

Minimum tests:

1. `RunResult` can represent a completed run with all required paths.
2. `ClipProbeAdapter.name == "clip_probe"`.
3. smoke embedding extraction is deterministic for the same image.
4. smoke embedding extraction changes for different smoke fixture images.
5. the adapter rejects `seed=None`.
6. the adapter runs on `configs/phase_ab_smoke.yaml` copied to `tmp_path`.
7. the smoke adapter writes `predictions.jsonl`.
8. smoke predictions load through `load_predictions()`.
9. smoke predictions pass `validate_predictions(..., require_mllm_fields=True)`.
10. every smoke prediction has `model_name="clip_probe"`.
11. every smoke prediction has `run_id` equal to the run directory name.
12. every smoke prediction has `parse_status="not_applicable"`.
13. every smoke prediction has `score_fake` in `[0.0, 1.0]`.
14. the smoke run writes `status.json` with `status="completed"`.
15. the smoke run writes `environment.json`.
16. the smoke run writes `logs.txt`.
17. the smoke run writes a `config.yaml` snapshot when invoked through CLI.
18. `aiforensics run --baseline clip_probe --config <tmp_smoke_config>` returns `0`.
19. CLI smoke run creates exactly one run directory for the one configured seed.
20. `discover_prediction_files(output_root)` finds the smoke prediction file after CLI run.
21. `evaluate_prediction_file(prediction_path)` writes `metrics.json` and `metrics_by_source.csv`.
22. multiple seeds in a temporary smoke config create one run directory per seed.
23. disabling `clip_probe` in a temporary config writes a deferred `status.json` and exits `0`.
24. a training manifest with only one class writes a failed `status.json` and exits `1`.
25. the smoke path does not require `open_clip` or `torch` to be importable.

Use `tmp_path` for every test-generated config, output root, cache root, status file, metric file, and run directory. Tests must not create artifacts under repository `outputs/`.

### Temporary Smoke Config Helper

Use a helper like this in tests:

```python
import yaml
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "phase_ab_smoke.yaml"


def write_tmp_smoke_config(tmp_path: Path, **overrides: object) -> Path:
    data = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    data["paths"]["output_root"] = str(tmp_path / "outputs")
    data["paths"]["cache_root"] = str(tmp_path / "cache")

    for dotted_key, value in overrides.items():
        target = data
        parts = dotted_key.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value

    config_path = tmp_path / "phase_ab_smoke.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path
```

Then call:

```python
exit_code = main(
    [
        "run",
        "--baseline",
        "clip_probe",
        "--config",
        str(config_path),
    ]
)
```

## CLI Smoke Expectations

After:

```bash
uv run python -m aiforensics.cli.main run --baseline clip_probe --config configs/phase_ab_smoke.yaml
```

there should be a new run directory under:

```text
outputs/smoke/
```

For local test code, use a temporary config so the test output root is under `tmp_path` instead.

Expected files for the smoke run:

```text
<tmp_output_root>/<run_id>/config.yaml
<tmp_output_root>/<run_id>/environment.json
<tmp_output_root>/<run_id>/status.json
<tmp_output_root>/<run_id>/predictions.jsonl
<tmp_output_root>/<run_id>/logs.txt
```

Expected prediction count for the current smoke config:

```text
2 records
```

because `tests/fixtures/manifests/smoke_dev.csv` currently contains two dev records.

## Ruff Scope

The repository still has Ruff debt in pre-Task-6 files. Keep Task 7 lint and format changes scoped to Task 7 files plus `src/aiforensics/cli/main.py` if it is modified.

Run:

```bash
uv run --extra dev ruff check \
  src/aiforensics/baselines \
  src/aiforensics/cli/main.py \
  tests/test_clip_probe_smoke.py

uv run --extra dev ruff format --check \
  src/aiforensics/baselines \
  src/aiforensics/cli/main.py \
  tests/test_clip_probe_smoke.py
```

Do not mass-format unrelated Task 1-6 files as part of Task 7.

## Verification

Task-specific verification:

```bash
uv run pytest tests/test_clip_probe_smoke.py -v
```

Regression verification:

```bash
uv run pytest
```

Smoke CLI verification:

```bash
uv run python -m aiforensics.cli.main prepare --config configs/phase_ab_smoke.yaml
uv run python -m aiforensics.cli.main run --baseline clip_probe --config configs/phase_ab_smoke.yaml
uv run python -m aiforensics.cli.main evaluate --config configs/phase_ab_smoke.yaml
```

Expected:

- `prepare` succeeds,
- `run --baseline clip_probe` creates a run directory,
- `run --baseline clip_probe` writes valid `predictions.jsonl`,
- `evaluate` discovers the prediction file and writes metrics beside it,
- smoke tests do not import OpenCLIP or Torch,
- smoke tests do not download model weights,
- no Qwen, assisted Qwen, or NPR behavior is implemented.

If `report` is still a placeholder, it may remain unchanged in Task 7.

## Done Criteria

Task 7 is complete when:

- `src/aiforensics/baselines/base.py` defines `RunResult` and `BaselineAdapter`,
- `ClipProbeAdapter.name` is exactly `clip_probe`,
- smoke config uses deterministic CPU-safe image embeddings,
- real config has a lazy OpenCLIP embedding path,
- real OpenCLIP dependencies are optional and do not affect smoke tests,
- real embedding cache uses `cache_key()` and `.npy` files under `config.paths.cache_root`,
- one run directory is created per configured CLIP seed,
- each run writes `config.yaml`, `environment.json`, `status.json`, `logs.txt`, and completed runs write `predictions.jsonl`,
- `predictions.jsonl` follows `docs/schemas/predictions-jsonl.md`,
- completed CLIP predictions validate with `validate_predictions(..., require_mllm_fields=True)`,
- `aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml` returns `0`,
- `aiforensics evaluate --config configs/phase_ab_smoke.yaml` can evaluate the generated CLIP predictions,
- all Task 7 tests and existing Task 1-6 tests pass,
- scoped Ruff checks pass for Task 7 files,
- Qwen-VL, assisted Qwen, NPR, report rendering, dataset downloading, external repo cloning, and explanation scoring remain unimplemented.

