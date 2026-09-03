# Task 2 Spec: Config Loading

## Goal

Implement typed YAML config loading for the Phase A/B pipeline. This task turns `configs/phase_ab.yaml` and `configs/phase_ab_smoke.yaml` into validated Python objects that later tasks can reuse.

Do not implement manifest building, model inference, metrics, caching, reporting, downloads, or dataset access in this task.

## Required Reading

Before coding, read:

- `AGENTS.md`
- `CLAUDE.md` when using Claude Code
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`

## Files To Create Or Modify

```text
src/aiforensics/config/__init__.py
src/aiforensics/config/models.py
src/aiforensics/config/load.py
src/aiforensics/cli/main.py
tests/test_config.py
tests/test_cli_smoke.py
```

## Public Interface

Expose these imports from `src/aiforensics/config/__init__.py`:

```python
from aiforensics.config.load import ConfigError, load_config
from aiforensics.config.models import AppConfig
```

Implement:

```python
def load_config(path: Path | str) -> AppConfig:
    ...
```

Implement:

```python
class ConfigError(ValueError):
    ...
```

`load_config()` must raise `ConfigError` for missing files, invalid YAML structure, missing required sections, and validation errors. Error messages must include the config path and the invalid or missing section name when available.

## Config Model Requirements

Use Pydantic models. Keep the models explicit and typed; do not pass raw dictionaries through the app after loading.

Required top-level sections:

```text
project
paths
runtime
datasets
baselines
evaluation
report
```

Required model names:

```python
AppConfig
ProjectConfig
PathsConfig
RuntimeConfig
DatasetsConfig
TinyGenImageConfig
GenImageUnseenConfig
SynthbusterConfig
BaselinesConfig
ClipProbeConfig
QwenVLConfig
AssistedQwenConfig
NPRConfig
EvaluationConfig
LabelsConfig
ReportConfig
```

## Required Fields

`ProjectConfig`:

```text
name: str
phase: str
description: str
```

`PathsConfig`:

```text
data_root: Path
manifest_root: Path
cache_root: Path
output_root: Path
external_root: Path
```

`RuntimeConfig`:

```text
python: str
seed: int
device: str
batch_size: int
num_workers: int
fail_fast: bool
```

`DatasetsConfig`:

```text
tiny_genimage: TinyGenImageConfig
genimage_unseen: GenImageUnseenConfig
synthbuster: SynthbusterConfig
```

`TinyGenImageConfig`:

```text
enabled: bool
source: str
use_original_split: bool
train_manifest: Path
dev_manifest: Path
```

`GenImageUnseenConfig`:

```text
enabled: bool
preferred_generator: str
fallback_generators: list[str]
max_images: int
balance_labels: bool
split: str
manifest: Path
```

`SynthbusterConfig`:

```text
enabled: bool
max_images: int
balance_labels: bool
split: str
manifest: Path
```

`BaselinesConfig`:

```text
clip_probe: ClipProbeConfig
qwen_vl: QwenVLConfig
assisted_qwen: AssistedQwenConfig
npr: NPRConfig
```

`ClipProbeConfig`:

```text
enabled: bool
model_family: str
model_name: str
pretrained: str
classifier: str
seeds: list[int]
cache_embeddings: bool
```

`QwenVLConfig`:

```text
enabled: bool
model_id: str
prompt_id: str
temperature: float
max_new_tokens: int
cache_outputs: bool
allow_deferred: bool = True
```

`AssistedQwenConfig`:

```text
enabled: bool
base_model_id: str
prompt_id: str
assistant_source: str
include_classifier_pred: bool
include_fake_probability: bool
temperature: float
max_new_tokens: int
cache_outputs: bool
allow_deferred: bool = True
```

`NPRConfig`:

```text
enabled: bool
repo_url: str
repo_commit: str | None
checkpoint_path: Path
checkpoint_sha256: str | None
batch_size: int
allow_deferred: bool
```

`EvaluationConfig`:

```text
labels: LabelsConfig
metrics: list[str]
group_by: list[str]
```

`LabelsConfig`:

```text
negative: str
positive: str
```

`ReportConfig`:

```text
filename: str
include_failure_notes: bool
include_explanations_sample: bool
explanation_sample_size: int
```

## Path Resolution

Resolve path fields after YAML validation.

Rules:

- Absolute paths remain absolute.
- Relative paths are resolved against the repository root by default.
- The repository root is the nearest parent directory containing `pyproject.toml`.
- The config file path itself may be absolute or relative.
- `AppConfig` should expose resolved `Path` objects for every path field.

Examples:

```python
config = load_config("configs/phase_ab_smoke.yaml")
assert config.paths.data_root.is_absolute()
assert config.paths.data_root.name == "smoke_data"
```

For manifest paths inside dataset configs:

```python
assert config.datasets.tiny_genimage.train_manifest.is_absolute()
assert config.datasets.tiny_genimage.train_manifest.name == "smoke_train.csv"
```

## Validation Rules

Validate these constraints:

- all required top-level sections exist
- `runtime.batch_size > 0`
- `runtime.num_workers >= 0`
- `clip_probe.seeds` is not empty when `clip_probe.enabled` is true
- `qwen_vl.temperature >= 0`
- `qwen_vl.max_new_tokens > 0`
- `assisted_qwen.temperature >= 0`
- `assisted_qwen.max_new_tokens > 0`
- `npr.batch_size > 0`
- `evaluation.labels.negative == "real"`
- `evaluation.labels.positive == "fake"`
- `report.explanation_sample_size >= 0`

Keep validation focused on config shape and values. Do not check whether data files, model weights, external repos, or manifest files exist in Task 2.

## CLI Integration

Update `src/aiforensics/cli/main.py` so each placeholder command calls `load_config(args.config)` before printing.

Placeholder output should include:

- command name
- config path
- `project.name`
- `project.phase`

For `run`, also include:

- selected baseline

Example output:

```text
[prepare] placeholder: project=ai-image-forensics phase=phase_ab_smoke config=configs/phase_ab_smoke.yaml
```

If config loading fails, let `ConfigError` surface for now. Do not build a custom CLI error UI in Task 2.

## Tests

Create `tests/test_config.py`.

Minimum tests:

1. `load_config("configs/phase_ab_smoke.yaml")` returns `AppConfig`.
2. Smoke config resolves `paths.data_root` to an absolute path ending in `tests/fixtures/smoke_data`.
3. Smoke config preserves project values: `name == "ai-image-forensics"` and `phase == "phase_ab_smoke"`.
4. Smoke config loads baseline values: `clip_probe.enabled is True`, `clip_probe.model_family == "synthetic"`, `qwen_vl.enabled is False`, and `npr.allow_deferred is True`.
5. Full config loads `npr.repo_commit is None` and `npr.checkpoint_sha256 is None`.
6. A missing top-level section such as `paths` raises `ConfigError` and the message contains `paths`.
7. Invalid numeric values such as `runtime.batch_size: 0` raise `ConfigError` and the message contains `batch_size`.
8. `load_config()` raises `ConfigError` for a missing config file.

Update `tests/test_cli_smoke.py` so existing CLI smoke tests still pass after commands load the smoke config.

Add one CLI test proving a command prints the project phase after loading config:

```python
exit_code = main(["prepare", "--config", SMOKE_CONFIG])
captured = capsys.readouterr()
assert exit_code == 0
assert "phase_ab_smoke" in captured.out
```

## Verification

Run:

```bash
pytest tests/test_config.py -v
pytest tests/test_cli_smoke.py -v
python -m aiforensics.cli.main prepare --config configs/phase_ab_smoke.yaml
python -m aiforensics.cli.main run --baseline clip_probe --config configs/phase_ab_smoke.yaml
```

Expected:

- config tests pass
- CLI smoke tests pass
- CLI placeholder commands print project and phase values from the config

## Done Criteria

Task 2 is complete when:

- `load_config()` returns typed `AppConfig` objects for both Phase A/B configs,
- every path field in the returned config is a resolved `Path`,
- invalid config shape and invalid key values raise `ConfigError`,
- CLI placeholders load config before printing,
- Task 1 CLI behavior still works,
- no manifest, dataset, model, metric, cache, report, or download behavior is implemented.

