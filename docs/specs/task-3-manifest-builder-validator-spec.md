# Task 3 Spec: Manifest Builder and Validator

## Goal

Implement the manifest layer for Phase A/B. A manifest is the source of truth for which images are in the experiment, what their labels are, where they came from, which split they belong to, and what their file checksum is.

This task must produce a working `aiforensics prepare` command for the smoke config. It must not implement model inference, prediction schema, metrics, reporting, external downloads, or real dataset download logic.

## Required Reading

Before coding, read:

- `AGENTS.md`
- `CLAUDE.md` when using Claude Code
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/schemas/manifest.md`
- `configs/phase_ab_smoke.yaml`
- `src/aiforensics/config/models.py`
- `src/aiforensics/config/load.py`
- `src/aiforensics/cli/main.py`

## Files To Create Or Modify

```text
src/aiforensics/data/__init__.py
src/aiforensics/data/manifest.py
src/aiforensics/cli/main.py
tests/test_manifest.py
tests/fixtures/smoke_data/
tests/fixtures/manifests/
```

Small PNG fixtures and CSV smoke manifests may be committed under `tests/fixtures/`. Do not place fixtures under ignored `data/` or `manifests/` directories.

## Public Interface

Expose these imports from `src/aiforensics/data/__init__.py`:

```python
from aiforensics.data.manifest import (
    ManifestError,
    ManifestRecord,
    ManifestValidationResult,
    compute_sha256,
    load_manifest,
    prepare_smoke_manifest,
    validate_manifest,
    write_manifest,
)
```

Implement:

```python
class ManifestError(ValueError):
    ...
```

Implement:

```python
class ManifestRecord(BaseModel):
    sample_id: str
    path: Path
    label: Literal["real", "fake"]
    source: str
    split: Literal["train", "dev", "test", "external", "smoke"]
    checksum: str
```

Optional fields from `docs/schemas/manifest.md` may be accepted when present:

```python
dataset: str | None = None
generator: str | None = None
width: int | None = None
height: int | None = None
mime_type: str | None = None
license: str | None = None
notes: str | None = None
```

Implement:

```python
class ManifestValidationResult(BaseModel):
    is_valid: bool
    total_records: int
    records_by_label: dict[str, int]
    records_by_split: dict[str, int]
    records_by_source: dict[str, int]
    duplicate_sample_ids: list[str]
    duplicate_checksums: dict[str, list[str]]
    missing_files: list[str]
    checksum_mismatches: list[str]
    errors: list[str]
    warnings: list[str]
```

Implement:

```python
def compute_sha256(path: Path) -> str:
    ...

def load_manifest(path: Path | str, *, data_root: Path | None = None) -> list[ManifestRecord]:
    ...

def write_manifest(records: list[ManifestRecord], path: Path | str) -> None:
    ...

def validate_manifest(records: list[ManifestRecord]) -> ManifestValidationResult:
    ...

def prepare_smoke_manifest(config: AppConfig) -> ManifestValidationResult:
    ...
```

## Manifest CSV Contract

Use CSV with UTF-8 encoding.

Required columns:

```text
sample_id,path,label,source,split,checksum
```

Write columns in this order. Optional columns may appear after the required columns.

## Path Behavior

`load_manifest()` must support both absolute and relative image paths:

- absolute manifest paths remain absolute,
- relative manifest paths are resolved against `data_root` when `data_root` is provided,
- relative manifest paths are resolved against the manifest file parent when `data_root` is not provided.

Returned `ManifestRecord.path` values should be absolute `Path` objects.

`write_manifest()` should write path values as strings exactly as stored in the records. It should create the output parent directory when needed.

## Validation Rules

`validate_manifest()` must check:

- manifest has at least one record,
- `sample_id` values are non-empty,
- `sample_id` values are unique,
- `label` values are `real` or `fake`,
- `split` values are `train`, `dev`, `test`, `external`, or `smoke`,
- image paths exist,
- image paths point to files,
- checksums are valid SHA-256 hex strings,
- checksums match image bytes when the file exists,
- duplicate checksums are reported,
- each split present in the manifest has both `real` and `fake` labels when that split contains at least two records.

Validation should collect all detectable errors into `ManifestValidationResult.errors` instead of failing at the first invalid record. CSV parsing and required-column failures may raise `ManifestError` because records cannot be built safely.

## Smoke Manifest Behavior

`prepare_smoke_manifest(config)` should create deterministic tiny smoke fixtures when the smoke files do not already exist.

Use `config.paths.data_root` and `config.paths.manifest_root`.

Create four small RGB PNG files:

```text
tests/fixtures/smoke_data/real_0001.png
tests/fixtures/smoke_data/real_0002.png
tests/fixtures/smoke_data/fake_0001.png
tests/fixtures/smoke_data/fake_0002.png
```

Use simple deterministic colors so the files are stable.

Write:

```text
tests/fixtures/manifests/smoke_train.csv
tests/fixtures/manifests/smoke_dev.csv
```

Recommended split:

```text
smoke/train/real_0001,real_0001.png,real,smoke,train,<sha256>
smoke/train/fake_0001,fake_0001.png,fake,smoke,train,<sha256>
smoke/dev/real_0002,real_0002.png,real,smoke,dev,<sha256>
smoke/dev/fake_0002,fake_0002.png,fake,smoke,dev,<sha256>
```

Paths in smoke CSV files should be relative to `config.paths.data_root`.

Return validation for the combined train and dev records.

## CLI Integration

Update `aiforensics prepare` so it:

1. loads config with `load_config(args.config)`,
2. when `config.project.phase == "phase_ab_smoke"`, calls `prepare_smoke_manifest(config)`,
3. creates `config.paths.output_root`,
4. writes a JSON validation summary to:

```text
<output_root>/manifest_validation.json
```

5. prints a concise message containing:

```text
[prepare]
project=<project.name>
phase=<project.phase>
records=<total_records>
valid=<is_valid>
summary=<summary_path>
```

For non-smoke configs in Task 3, `prepare` may validate configured manifests if they exist. If they do not exist, raise `ManifestError` with a clear message that real dataset manifest building is not implemented in Task 3.

## Tests

Create `tests/test_manifest.py`.

Minimum tests:

1. `compute_sha256()` returns the known SHA-256 for a small file.
2. `write_manifest()` then `load_manifest()` round-trips two records.
3. `load_manifest()` raises `ManifestError` when required columns are missing.
4. `validate_manifest()` reports duplicate `sample_id` values.
5. `validate_manifest()` reports duplicate checksums.
6. `validate_manifest()` reports checksum mismatches.
7. `validate_manifest()` reports missing image files.
8. `validate_manifest()` reports a split with only one label when the split has at least two records.
9. `prepare_smoke_manifest(load_config("configs/phase_ab_smoke.yaml"))` creates smoke images and smoke manifest CSV files.
10. `aiforensics prepare --config configs/phase_ab_smoke.yaml` writes `outputs/smoke/manifest_validation.json`.

Update existing CLI tests only as needed so Task 1 and Task 2 behavior still passes.

## Test Data Policy

Tests should use `tmp_path` for most generated files. The smoke prepare test may write to the configured smoke paths because those paths are part of the smoke contract.

If tests write `outputs/smoke/manifest_validation.json`, clean it in the test or leave it ignored by git through `.gitignore`.

## Verification

Run:

```bash
pytest tests/test_manifest.py -v
pytest tests/test_config.py -v
pytest tests/test_cli_smoke.py -v
python -m aiforensics.cli.main prepare --config configs/phase_ab_smoke.yaml
```

Expected:

- manifest tests pass,
- config tests still pass,
- CLI smoke tests still pass,
- `prepare` prints `valid=True`,
- `outputs/smoke/manifest_validation.json` exists and contains `is_valid: true`.

## Done Criteria

Task 3 is complete when:

- manifest records can be loaded from CSV,
- manifest records can be written to CSV,
- required columns are enforced,
- labels and splits are validated,
- image existence is checked,
- SHA-256 checksums are computed and verified,
- duplicate sample ids and duplicate checksums are reported,
- smoke image fixtures and smoke manifests can be created deterministically,
- `aiforensics prepare --config configs/phase_ab_smoke.yaml` writes a validation summary,
- no model inference, prediction schema, metric computation, report generation, external repo cloning, or dataset download behavior is implemented.

