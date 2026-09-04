# Phase A/B Architecture

## Purpose

Phase A/B builds a reproducible baseline suite for AI-generated image detection. It should answer whether the project should proceed to later stages involving trained fusion, token-space alignment, and explanation reasoning.

The system compares four baselines on a controlled dataset protocol:

- `qwen_vl`: base open-source MLLM prompted to classify real/fake and return structured JSON.
- `clip_probe`: frozen CLIP ViT-L/14 image embeddings with a CPU logistic-regression linear probe.
- `npr`: pretrained NPR detector run through the official external repository.
- `assisted_qwen`: Qwen-VL prompted with the image plus the CLIP probe label and fake probability.

## Scope

Phase A/B includes reproducible data manifests, baseline inference, evaluation, reports, Colab/Kaggle entry notebooks, and smoke tests.

Phase A/B excludes custom Stage 1 fusion training, Stage 2 projector/token-space alignment training, Stage 3 explanation fine-tuning, Docker-first infrastructure, dashboards, and publication-grade explanation evaluation.

## Repository Layout

The repo uses a monorepo-style layout, while the implemented surface remains focused on Phase A/B.

```text
src/aiforensics/
  cli/
  config/
  data/
  schemas/
  baselines/
    clip_probe/
    qwen_vl/
    assisted_qwen/
    npr/
  evaluation/
  reporting/
  cache/
  utils/

configs/
notebooks/
tests/
docs/
  schemas/
  research/
external/
outputs/
infra/
scripts/
```

Directory responsibilities:

- `src/aiforensics/cli/`: command entrypoints for `prepare`, `run`, `evaluate`, and `report`.
- `src/aiforensics/config/`: YAML loading, config dataclasses, path resolution, and config validation.
- `src/aiforensics/data/`: manifest building, split validation, checksum calculation, duplicate checks, and dataset summaries.
- `src/aiforensics/schemas/`: typed records and validators for manifests, predictions, metrics, and run metadata.
- `src/aiforensics/baselines/`: one adapter per baseline. Each adapter owns model loading, preprocessing, inference, and conversion to the shared prediction schema.
- `src/aiforensics/evaluation/`: metrics from manifest plus prediction files.
- `src/aiforensics/reporting/`: Markdown/CSV report generation from metrics and run metadata.
- `src/aiforensics/cache/`: cache key construction and cache read/write helpers.
- `src/aiforensics/utils/`: small shared utilities with no model-specific behavior.
- `external/`: runtime location for cloned external repositories such as NPR. Downloaded content is ignored by git.
- `outputs/`: generated run artifacts. Ignored by git.
- `infra/`: reserved for future infrastructure. Phase A/B does not require Docker or CI.

## Data Protocol

Phase A/B uses a manifest-first protocol. Raw datasets may live anywhere; the code sees them through manifest files and config paths.

Datasets:

- Tiny-GenImage for train/dev using the original dataset split when available.
- GenImage unseen-generator subset for cross-generator testing, prioritizing Midjourney.
- Synthbuster as a small external test set.

Each manifest row follows `docs/schemas/manifest.md` and includes:

- stable sample id
- image path
- label
- source/generator
- split
- checksum

The `prepare` command builds or validates manifests, verifies labels and paths, computes checksums, checks duplicates by checksum, and writes summary artifacts.

## Baseline Interface

Each baseline adapter should expose the same conceptual interface:

```python
class BaselineAdapter:
    name: str

    def run(self, manifest_path: Path, output_dir: Path, config: BaselineConfig) -> RunResult:
        ...
```

`RunResult` should include:

- baseline name
- run id
- status: `completed`, `failed`, or `deferred`
- output directory
- prediction path when available
- log path
- metadata path
- failure reason when status is not `completed`

Every completed adapter writes `predictions.jsonl` following `docs/schemas/predictions-jsonl.md`.

## Baselines

### CLIP Probe

`clip_probe` uses frozen OpenCLIP `ViT-L-14` image embeddings. It caches embeddings by image checksum plus CLIP model id and preprocessing config. The classifier is a CPU logistic-regression linear probe trained on the configured train split and evaluated on configured dev/test splits.

Run policy:

- Train with three seeds.
- Store one run artifact per seed or one combined artifact with seed-level metrics.
- Report aggregate metrics and per-source metrics.

### Qwen-VL

`qwen_vl` uses `Qwen/Qwen2.5-VL-3B-Instruct` as the single Phase A/B MLLM. It runs deterministic inference with one structured prompt. The model must return parseable JSON with `label`, `confidence`, and `evidence`.

The adapter stores:

- parsed label and fake score
- raw model output
- prompt id
- explanation/evidence text
- parse status

### Assisted Qwen

`assisted_qwen` uses the image plus the CLIP probe prediction. The prompt includes `classifier_pred` and `fake_probability`. It should not include retrieved examples, patch attribution, or unimplemented forensic maps in Phase A/B.

This baseline tests whether a classifier score improves MLLM detection and explanation behavior compared with base Qwen-VL.

### NPR External Baseline

`npr` uses the official NPR repository:

`https://github.com/chuangchuangtan/NPR-DeepfakeDetection`

The project must not copy NPR source code into this repo. The adapter should clone or locate the official repo at runtime, pin the commit in config, verify the checkpoint when a checksum is configured, run official inference code, and convert outputs to the shared prediction schema.

If NPR cannot run because of dependency or environment incompatibility, the adapter records `deferred` with logs and does not block other baselines.

## CLI Flow

```text
prepare
  -> read config
  -> build/validate manifests
  -> compute checksums and duplicate report
  -> write prepared manifest summaries

run --baseline <name>
  -> read config and manifest
  -> create outputs/<run_id>/
  -> write config.yaml, run_scope.json, and environment.json
  -> execute baseline adapter
  -> write predictions.jsonl or failure/deferred metadata

evaluate
  -> read manifests and prediction files
  -> skip runs whose run_scope.json does not match the current config
  -> validate prediction schema against the current manifest sample ids
  -> compute metrics
  -> write metrics.json and metrics_by_source.csv

report
  -> read run metadata and metrics for the current run scope only
  -> write Markdown report comparing baselines
```

## Run Scope

A single `output_root` accumulates runs from different configs and dataset
slices. Each run directory therefore records a `run_scope.json` fingerprint of
the evaluation setup that produced it: project phase, `data_root`, which dataset
slices are enabled, the manifests they resolve to, and a digest of the exact
evaluation sample ids.

`evaluate`, `report`, and `assisted_qwen` all filter by that fingerprint, so
artifacts from an unrelated experiment cannot enter the current one. Because the
fingerprint covers the evaluation slice rather than model or seed choices,
re-running the same experiment reproduces the same scope, while changing a
dataset flag, manifest, or phase makes older runs foreign.

## Artifact Contract

Every run writes:

```text
outputs/<run_id>/
  config.yaml
  run_scope.json
  environment.json
  predictions.jsonl
  metrics.json
  metrics_by_source.csv
  logs.txt
```

If a run fails before predictions are available, it still writes:

```text
outputs/<run_id>/
  config.yaml
  run_scope.json
  environment.json
  status.json
  logs.txt
```

`status.json` should include `baseline`, `status`, `reason`, `command`, `started_at`, and `ended_at`.

## Metrics

The required classification metrics are:

- accuracy
- balanced accuracy
- precision
- recall
- F1
- AUROC when scores are available
- per-source metrics

Explanation text is stored for analysis, but not scored automatically in Phase A/B.

## Caching

Cache expensive deterministic outputs using keys built from:

- sample checksum
- model id
- preprocessing config
- prompt id for MLLM outputs
- baseline version/config hash

Cache CLIP embeddings and Qwen raw outputs. Do not cache processed images unless a future task explicitly adds that behavior.

## Platform Policy

The same repository code must run locally, on Colab, and on Kaggle. Platform differences belong in config values and thin notebooks, not in core logic.

Use Python 3.10-compatible code and `requirements.txt` for the initial implementation. Docker is not required in Phase A/B.

## Acceptance Gate

Phase A/B implementation is acceptable when:

- CPU tests pass.
- `prepare` runs with the smoke config.
- `clip_probe` runs on the smoke config.
- `evaluate` runs on smoke outputs.
- `report` generates a Markdown report.
- Qwen/NPR either run in the available environment or write clear `failed`/`deferred` artifacts with logs.

