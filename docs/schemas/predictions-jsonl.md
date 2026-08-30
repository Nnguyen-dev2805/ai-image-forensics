# Predictions JSONL Schema

## Purpose

Every baseline writes predictions in one shared JSONL format so evaluation and reporting stay model-agnostic.

Each line is one JSON object for one image sample.

## Required Fields

```json
{
  "sample_id": "tiny-genimage/dev/midjourney/000001",
  "label_true": "fake",
  "label_pred": "fake",
  "score_fake": 0.91,
  "model_name": "clip_probe",
  "source": "midjourney"
}
```

### `sample_id`

Must match a `sample_id` from the manifest.

### `label_true`

Ground-truth label from the manifest.

Allowed values:

```text
real
fake
```

### `label_pred`

Predicted label.

Allowed values:

```text
real
fake
unknown
```

Use `unknown` only when inference completed but a prediction could not be parsed for this sample.

### `score_fake`

Numeric fake probability or fake score in `[0.0, 1.0]`.

Use `null` only when the baseline cannot produce a meaningful score. AUROC should be skipped for records without usable scores.

### `model_name`

Baseline or model identifier.

Phase A/B values:

```text
clip_probe
qwen_vl
npr
assisted_qwen
```

### `source`

Source/generator copied from the manifest for grouping and per-source metrics.

## Optional Common Fields

```json
{
  "run_id": "2026-08-29_clip_probe_seed0",
  "dataset": "tiny-genimage",
  "split": "dev",
  "path": "relative/or/absolute/path.png",
  "checksum": "sha256..."
}
```

## Optional MLLM Fields

Qwen-based baselines should include these fields:

```json
{
  "prompt_id": "qwen_json_v1",
  "raw_output": "{\"label\":\"fake\",\"confidence\":0.91,\"evidence\":\"...\"}",
  "explanation": "Texture and edge patterns appear inconsistent with a natural camera image.",
  "parse_status": "parsed"
}
```

Allowed `parse_status` values:

```text
parsed
recovered
failed
not_applicable
```

`clip_probe` and `npr` should use `not_applicable` or omit `parse_status`.

## Validation Rules

Prediction validation must check:

- required fields exist
- labels use allowed values
- `score_fake` is `null` or in `[0.0, 1.0]`
- `sample_id` exists in the evaluated manifest
- duplicate predictions for the same `sample_id` are reported
- `model_name` is one of the configured baselines
- MLLM records include `prompt_id`, `raw_output`, `explanation`, and `parse_status`

Invalid prediction rows must not be silently dropped.

## Failure and Deferred Runs

If a baseline fails before per-sample predictions exist, write `status.json` and `logs.txt` instead of a partial `predictions.jsonl`.

Use partial predictions only when inference processed some samples and the adapter can clearly mark missing or unparseable records.

