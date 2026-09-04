# Manifest Schema

## Purpose

Manifests are the source of truth for dataset membership, labels, sources, and splits. Raw datasets may live in different places on local machines, Colab, or Kaggle; code should operate through manifests and config paths.

## File Format

Use CSV for Phase A/B manifests. UTF-8 encoding is required. One row represents one image sample.

Required columns:

```text
sample_id,path,label,source,split,checksum
```

Optional columns:

```text
dataset,generator,width,height,mime_type,license,notes
```

## Required Fields

### `sample_id`

Stable string id for the sample. It must be unique within a manifest.

Recommended format:

```text
<dataset>/<split>/<source>/<relative_stem>
```

Example:

```text
tiny-genimage/train/stable-diffusion/000001
```

### `path`

Path to the image file. It may be absolute or relative to the configured `data_root`.

The path must point to a readable image file during manifest validation.

### `label`

Ground-truth binary label.

Allowed values:

```text
real
fake
```

### `source`

Dataset source or generator name used for grouping and per-source metrics.

Examples:

```text
tiny-genimage
midjourney
synthbuster
stable-diffusion
imagenet-real
```

### `split`

Evaluation split.

Allowed values:

```text
train
dev
test
external
smoke
```

Use the original dataset split when available. Do not randomly rebuild official splits unless the config explicitly asks for a derived split.

### `checksum`

SHA-256 checksum of the image bytes.

The prepare command should compute this value when building manifests and verify it when validating existing manifests.

## Validation Rules

Manifest validation must check:

- all required columns exist
- `sample_id` values are unique
- `label` is `real` or `fake`
- `split` is one of the allowed values
- image paths exist and are readable
- checksum values match image bytes
- duplicate checksums are reported
- each configured split has both labels when used for metrics

Validation must not silently skip invalid rows. It should either write a validation report or fail with a clear error depending on the CLI mode.

## Phase A/B Manifests

The intended manifests are:

```text
manifests/tiny_genimage_train.csv
manifests/tiny_genimage_dev.csv
manifests/genimage_unseen_external.csv
manifests/synthbuster_external.csv
manifests/smoke.csv
```

The smoke manifest should use tiny synthetic or fixture images and must not require a real dataset download.

