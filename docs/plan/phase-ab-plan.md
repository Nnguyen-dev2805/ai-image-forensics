# Phase A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Phase A/B baseline suite for AI-generated image detection.

**Architecture:** Implement a manifest-first Python package with a stable CLI, shared schemas, baseline adapters, evaluation, and reports. Keep the monorepo layout broad, but only implement the Phase A/B functionality described in `docs/phase-ab-architecture.md`.

**Tech Stack:** Python 3.10, YAML config, pytest, scikit-learn metrics, PIL/Pillow image fixtures, OpenCLIP for CLIP probe, Hugging Face Transformers for Qwen-VL, and external official NPR repository integration.

**Spec:** `docs/phase-ab-architecture.md`

## Global Constraints

- Keep importable code under `src/aiforensics/`.
- Use `configs/phase_ab.yaml` and `configs/phase_ab_smoke.yaml`.
- Do not hardcode local, Colab, or Kaggle paths.
- Do not commit datasets, model weights, downloaded repos, caches, checkpoints, or generated outputs.
- Keep notebooks as thin wrappers around the CLI.
- Use NPR only as a pinned external baseline from `https://github.com/chuangchuangtan/NPR-DeepfakeDetection`.
- Every completed baseline writes `predictions.jsonl` following `docs/schemas/predictions-jsonl.md`.
- Every task must leave the repo in a state where the relevant tests for that task can run.

---

## File Structure To Create

- `pyproject.toml`: package metadata, pytest settings, console script entrypoint.
- `requirements.txt`: Python 3.10 runtime dependencies for Phase A/B.
- `src/aiforensics/__init__.py`: package marker and version.
- `src/aiforensics/cli/main.py`: CLI entrypoint and subcommands.
- `src/aiforensics/config/models.py`: typed config dataclasses.
- `src/aiforensics/config/load.py`: YAML config loading and path resolution.
- `src/aiforensics/data/manifest.py`: manifest records, loading, writing, validation, checksum, duplicate checks.
- `src/aiforensics/schemas/predictions.py`: prediction records and JSONL validation.
- `src/aiforensics/evaluation/metrics.py`: classification and per-source metrics.
- `src/aiforensics/cache/keys.py`: deterministic cache key helpers.
- `src/aiforensics/runs/artifacts.py`: run id generation, output directory creation, metadata writing.
- `src/aiforensics/baselines/base.py`: baseline adapter protocol and run result model.
- `src/aiforensics/baselines/clip_probe/adapter.py`: CLIP linear probe baseline.
- `src/aiforensics/baselines/qwen_vl/adapter.py`: Qwen-VL base MLLM baseline.
- `src/aiforensics/baselines/assisted_qwen/adapter.py`: classifier-assisted Qwen baseline.
- `src/aiforensics/baselines/npr/adapter.py`: external NPR baseline adapter.
- `src/aiforensics/reporting/markdown.py`: Phase A/B Markdown report.
- `tests/fixtures/`: tiny synthetic fixture data for smoke tests.
- `tests/`: unit tests and CLI smoke tests.
- `notebooks/colab_phase_ab.ipynb`: thin Colab wrapper.
- `notebooks/kaggle_phase_ab.ipynb`: thin Kaggle wrapper.

---

## Task 1: Package and CLI Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/aiforensics/__init__.py`
- Create: `src/aiforensics/cli/main.py`
- Create: `tests/test_cli_smoke.py`

**Interfaces:**
- Produces CLI command `aiforensics`.
- Produces subcommands `prepare`, `run`, `evaluate`, and `report`.

- [ ] **Step 1: Create package metadata**

Create `pyproject.toml` with a console script:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aiforensics"
version = "0.1.0"
requires-python = ">=3.10,<3.11"
dependencies = []

[project.scripts]
aiforensics = "aiforensics.cli.main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create initial requirements**

Create `requirements.txt` with:

```text
pyyaml>=6.0.1
pydantic>=2.8.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
pillow>=10.0.0
pytest>=8.0.0
tqdm>=4.66.0
```

- [ ] **Step 3: Implement CLI skeleton**

Implement `src/aiforensics/cli/main.py` using `argparse`. Each command should accept `--config` and return a clear placeholder message while later tasks fill behavior.

- [ ] **Step 4: Add CLI smoke test**

Add tests that call `main(["prepare", "--config", "configs/phase_ab_smoke.yaml"])` and assert the command is recognized.

- [ ] **Step 5: Verify**

Run:

```bash
pytest tests/test_cli_smoke.py -v
```

Expected: tests pass.

---

## Task 2: Config Loading

**Files:**
- Create: `src/aiforensics/config/models.py`
- Create: `src/aiforensics/config/load.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces `load_config(path: Path) -> AppConfig`.
- Produces typed config objects for paths, datasets, baselines, cache, and outputs.

- [ ] **Step 1: Define config dataclasses or Pydantic models**

Represent these sections:

```python
paths: data_root, manifest_root, cache_root, output_root, external_root
datasets: tiny_genimage, genimage_unseen, synthbuster
baselines: clip_probe, qwen_vl, assisted_qwen, npr
evaluation: metrics, group_by
runtime: seed, device, batch_size
```

- [ ] **Step 2: Implement YAML loading and path resolution**

Resolve relative paths relative to the repository root or current working directory consistently. Preserve user-configured absolute paths.

- [ ] **Step 3: Test valid config loading**

Load `configs/phase_ab_smoke.yaml` and assert paths and baseline names match the config.

- [ ] **Step 4: Test invalid config failure**

Assert missing required sections raise an explicit exception with the missing section name.

- [ ] **Step 5: Verify**

Run:

```bash
pytest tests/test_config.py -v
```

Expected: tests pass.

---

## Task 3: Manifest Builder and Validator

**Files:**
- Create: `src/aiforensics/data/manifest.py`
- Create: `tests/test_manifest.py`
- Create: `tests/fixtures/images/`

**Interfaces:**
- Produces `ManifestRecord`.
- Produces `load_manifest(path: Path) -> list[ManifestRecord]`.
- Produces `write_manifest(records: list[ManifestRecord], path: Path) -> None`.
- Produces `validate_manifest(records: list[ManifestRecord]) -> ManifestValidationResult`.
- Produces `compute_sha256(path: Path) -> str`.

- [ ] **Step 1: Create tiny image fixtures**

Use PIL in tests to generate tiny RGB PNG files in a temporary directory. Do not commit large fixtures.

- [ ] **Step 2: Implement manifest record validation**

Enforce required fields from `docs/schemas/manifest.md`: `sample_id`, `path`, `label`, `source`, `split`, and `checksum`.

- [ ] **Step 3: Implement checksum and duplicate detection**

Duplicate detection should report repeated checksums across records.

- [ ] **Step 4: Wire `aiforensics prepare`**

For the smoke config, create or validate a small manifest and write a validation summary under the configured output root.

- [ ] **Step 5: Verify**

Run:

```bash
pytest tests/test_manifest.py -v
aiforensics prepare --config configs/phase_ab_smoke.yaml
```

Expected: tests pass and prepare writes a manifest validation artifact.

---

## Task 4: Prediction Schema

**Files:**
- Create: `src/aiforensics/schemas/predictions.py`
- Create: `tests/test_predictions_schema.py`

**Interfaces:**
- Produces `PredictionRecord`.
- Produces `write_predictions(records: Iterable[PredictionRecord], path: Path) -> None`.
- Produces `load_predictions(path: Path) -> list[PredictionRecord]`.
- Produces `validate_prediction_record(record: Mapping[str, Any]) -> PredictionRecord`.

- [ ] **Step 1: Implement required fields**

Required fields are defined in `docs/schemas/predictions-jsonl.md`.

- [ ] **Step 2: Implement optional MLLM fields**

Support `raw_output`, `prompt_id`, `explanation`, and `parse_status`.

- [ ] **Step 3: Test valid and invalid records**

Assert invalid labels, missing ids, and out-of-range scores fail with explicit messages.

- [ ] **Step 4: Verify**

Run:

```bash
pytest tests/test_predictions_schema.py -v
```

Expected: tests pass.

---

## Task 5: Metrics

**Files:**
- Create: `src/aiforensics/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces `compute_classification_metrics(records: list[PredictionRecord]) -> dict[str, float | None]`.
- Produces `compute_metrics_by_source(records: list[PredictionRecord]) -> pandas.DataFrame`.

- [ ] **Step 1: Implement classification metrics**

Compute accuracy, balanced accuracy, precision, recall, F1, and AUROC when `score_fake` exists for both classes.

- [ ] **Step 2: Implement per-source metrics**

Group by `source` and compute the same metrics where valid.

- [ ] **Step 3: Wire `aiforensics evaluate`**

Read prediction files, validate schema, write `metrics.json` and `metrics_by_source.csv`.

- [ ] **Step 4: Verify**

Run:

```bash
pytest tests/test_metrics.py -v
```

Expected: tests pass.

---

## Task 6: Run Artifacts and Cache Keys

**Files:**
- Create: `src/aiforensics/runs/artifacts.py`
- Create: `src/aiforensics/cache/keys.py`
- Create: `tests/test_run_artifacts.py`
- Create: `tests/test_cache_keys.py`

**Interfaces:**
- Produces `create_run_dir(output_root: Path, baseline: str, run_name: str | None) -> Path`.
- Produces `write_environment(path: Path) -> None`.
- Produces `write_status(path: Path, status: RunStatus) -> None`.
- Produces `cache_key(parts: Mapping[str, str]) -> str`.

- [ ] **Step 1: Implement stable run directories**

Run directories should live under `outputs/<run_id>/` or the configured output root.

- [ ] **Step 2: Implement environment metadata**

Capture Python version, platform, package versions when available, command, and timestamp.

- [ ] **Step 3: Implement deterministic cache keys**

Hash sorted key/value parts so key order does not change the result.

- [ ] **Step 4: Verify**

Run:

```bash
pytest tests/test_run_artifacts.py tests/test_cache_keys.py -v
```

Expected: tests pass.

---

## Task 7: CLIP Probe Baseline

**Files:**
- Create: `src/aiforensics/baselines/base.py`
- Create: `src/aiforensics/baselines/clip_probe/adapter.py`
- Create: `tests/test_clip_probe_smoke.py`

**Interfaces:**
- Produces baseline name `clip_probe`.
- Produces predictions following `docs/schemas/predictions-jsonl.md`.

- [ ] **Step 1: Define baseline adapter protocol**

Create a small shared protocol and `RunResult` model used by all adapters.

- [ ] **Step 2: Implement a smoke-safe CLIP adapter path**

The smoke config may use cached or synthetic embeddings so tests do not require model downloads.

- [ ] **Step 3: Implement real CLIP path**

Use OpenCLIP `ViT-L-14` for real runs. Cache embeddings by checksum, model id, and preprocessing config.

- [ ] **Step 4: Train linear probe**

Train logistic regression for configured seeds and write predictions for eval splits.

- [ ] **Step 5: Wire `aiforensics run --baseline clip_probe`**

The CLI should create run artifacts and call the adapter.

- [ ] **Step 6: Verify**

Run:

```bash
pytest tests/test_clip_probe_smoke.py -v
aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
```

Expected: smoke run completes and writes predictions.

---

## Task 8: Qwen-VL Baseline

**Files:**
- Create: `src/aiforensics/baselines/qwen_vl/adapter.py`
- Create: `tests/test_qwen_prompt_parsing.py`

**Interfaces:**
- Produces baseline name `qwen_vl`.
- Produces structured MLLM prediction records with `raw_output`, `prompt_id`, `explanation`, and `parse_status`.

- [ ] **Step 1: Define prompt template**

Prompt must request JSON with `label`, `confidence`, and `evidence`.

- [ ] **Step 2: Implement output parser**

Parse valid JSON and robustly fail invalid output with `parse_status`.

- [ ] **Step 3: Implement deferred behavior**

If model dependencies, GPU, or weights are unavailable, write `status.json` with `status="deferred"` and a clear reason.

- [ ] **Step 4: Wire `aiforensics run --baseline qwen_vl`**

Real inference uses `Qwen/Qwen2.5-VL-3B-Instruct` from config.

- [ ] **Step 5: Verify parser tests**

Run:

```bash
pytest tests/test_qwen_prompt_parsing.py -v
```

Expected: parser tests pass without downloading Qwen.

---

## Task 9: Assisted Qwen Baseline

**Files:**
- Create: `src/aiforensics/baselines/assisted_qwen/adapter.py`
- Create: `tests/test_assisted_qwen_prompt.py`

**Interfaces:**
- Produces baseline name `assisted_qwen`.
- Consumes CLIP probe predictions or scores configured as assistant input.

- [ ] **Step 1: Define assisted prompt**

Prompt must include `classifier_pred` and `fake_probability`.

- [ ] **Step 2: Implement CLIP prediction lookup**

Join assisted inputs by `sample_id`; fail clearly when required scores are missing.

- [ ] **Step 3: Reuse Qwen parser**

Use the same parsing and deferred behavior as `qwen_vl`.

- [ ] **Step 4: Verify**

Run:

```bash
pytest tests/test_assisted_qwen_prompt.py -v
```

Expected: prompt construction and missing-score behavior are tested.

---

## Task 10: NPR External Adapter

**Files:**
- Create: `src/aiforensics/baselines/npr/adapter.py`
- Create: `tests/test_npr_adapter.py`

**Interfaces:**
- Produces baseline name `npr`.
- Clones or locates official NPR repo outside committed source.
- Converts NPR outputs to shared predictions.

- [ ] **Step 1: Implement config-driven external repo location**

Use `paths.external_root` and the configured NPR repo URL and commit.

- [ ] **Step 2: Implement dry-run/smoke mode**

Smoke tests must not clone the repo or download weights. They should validate command construction and conversion logic.

- [ ] **Step 3: Implement real run command**

Run official NPR inference script when the external repo and checkpoint are available.

- [ ] **Step 4: Implement deferred status**

Write `status.json` with `status="deferred"` when dependencies, repo checkout, or checkpoint are missing.

- [ ] **Step 5: Verify**

Run:

```bash
pytest tests/test_npr_adapter.py -v
```

Expected: command construction, conversion, and deferred behavior pass.

---

## Task 11: Reporting

**Files:**
- Create: `src/aiforensics/reporting/markdown.py`
- Create: `tests/test_reporting.py`

**Interfaces:**
- Produces `build_phase_ab_report(config: AppConfig, runs: list[RunSummary]) -> str`.
- Produces a Markdown report from metrics and status files.

- [ ] **Step 1: Implement report sections**

Include dataset summary, baseline status table, overall metrics table, per-source metrics table, failure/deferred notes, and next-step recommendation.

- [ ] **Step 2: Wire `aiforensics report`**

Write a report under the configured output root.

- [ ] **Step 3: Verify**

Run:

```bash
pytest tests/test_reporting.py -v
aiforensics report --config configs/phase_ab_smoke.yaml
```

Expected: report file is generated.

---

## Task 12: Colab and Kaggle Wrappers

**Files:**
- Create: `notebooks/colab_phase_ab.ipynb`
- Create: `notebooks/kaggle_phase_ab.ipynb`
- Create: `docs/runbook-colab-kaggle.md`

**Interfaces:**
- Notebooks install the package, mount or point to storage, and call the same CLI commands.

- [ ] **Step 1: Create thin notebooks**

Each notebook should avoid duplicated logic. It should call shell commands for install, prepare, run, evaluate, and report.

- [ ] **Step 2: Create runbook**

Document how to set `data_root`, `manifest_root`, `cache_root`, `output_root`, and `external_root` for Colab and Kaggle.

- [ ] **Step 3: Verify manually**

Open notebooks and confirm they contain no core Python pipeline logic beyond path setup and CLI calls.

---

## Task 13: Final Verification

**Files:**
- Modify: `README.md`
- Verify: all Phase A/B files.

**Interfaces:**
- Produces a user-facing quickstart.

- [ ] **Step 1: Write README quickstart**

Document installation, smoke run, full Phase A/B intended run, artifact locations, and known GPU requirements.

- [ ] **Step 2: Run full smoke gate**

Run:

```bash
pytest
aiforensics prepare --config configs/phase_ab_smoke.yaml
aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
aiforensics evaluate --config configs/phase_ab_smoke.yaml
aiforensics report --config configs/phase_ab_smoke.yaml
```

Expected: tests pass and smoke artifacts are generated.

- [ ] **Step 3: Record unsupported environment behavior**

If Qwen or NPR cannot run locally, confirm their adapters produce `deferred` status artifacts with logs.

