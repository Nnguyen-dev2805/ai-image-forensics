# Colab and Kaggle Runbook

## Purpose

Run the existing Phase A/B baseline suite on Google Colab or Kaggle without
creating a second implementation of the pipeline. The two notebooks

```text
notebooks/colab_phase_ab.ipynb
notebooks/kaggle_phase_ab.ipynb
```

are thin wrappers: they install the package, provision a compatible Python
runtime, point the pipeline at storage you control, and call the same public
`aiforensics` CLI you would run locally.

## What Task 12 Does and Does Not Do

Does:

- install the repository package and its optional extras,
- report a minimal runtime preflight (Python version, working directory, GPU),
- provision a Python 3.10 environment for the CLI when the kernel is newer,
- mount or point at user-controlled storage,
- generate a runtime YAML config by changing **path values only**,
- call the public CLI in the required order,
- show where run artifacts and the Markdown report were written.

Does not:

- download or build research datasets,
- parse manifests, images, or predictions,
- load models or run inference,
- compute metrics or render the report,
- clone or patch the NPR repository,
- decide research outcomes,
- convert a genuine CLI failure into an apparent success.

Every behaviour in the second list already belongs to the package (Tasks 1-11)
and must stay there. A useful review rule: if a notebook cell would still be
useful inside `src/aiforensics/`, it does not belong in the notebook.

## Prerequisites

- A checkout of this repository reachable from the notebook runtime, either
  already present or cloned from a repository URL you control.
- Python 3.10 for the CLI. `pyproject.toml` declares
  `requires-python = ">=3.10,<3.11"`. The notebook kernel itself may be newer;
  see [GPU and Optional Dependencies](#gpu-and-optional-dependencies).
- For a full Phase A/B run: provisioned images, provisioned manifests, a CUDA
  GPU, network access for model weights, and the official NPR checkpoint.
- For smoke verification: nothing beyond the repository itself.

Never edit `pyproject.toml`, `configs/phase_ab.yaml`, or
`configs/phase_ab_smoke.yaml` to make a hosted run succeed.

## Shared Phase A/B Command Order

Both notebooks and local execution use the same public commands, in this order:

```bash
aiforensics prepare --config "$AIF_CONFIG"
aiforensics run --baseline clip_probe --config "$AIF_CONFIG"
aiforensics run --baseline qwen_vl --config "$AIF_CONFIG"
aiforensics run --baseline npr --config "$AIF_CONFIG"
aiforensics run --baseline assisted_qwen --config "$AIF_CONFIG"
aiforensics evaluate --config "$AIF_CONFIG"
aiforensics report --config "$AIF_CONFIG"
```

Why the order matters:

- `prepare` validates manifests before any baseline consumes them.
- `assisted_qwen` runs **after** `clip_probe` because its Phase A/B contract
  consumes CLIP assistant predictions from a completed CLIP run.
- `evaluate` reads prediction artifacts, so it follows every baseline.
- `report` reads metric artifacts, so it follows `evaluate`.

One `clip_probe` command covers every configured seed. Do not loop over seeds in
the notebook; the CLI owns that.

Do not call `_cmd_*()` functions or instantiate adapters from a notebook.
Never append `|| true` to any command: a real CLI failure must stay visible.

## Required Storage and Config Paths

The pipeline uses five project roots. Each is a visible, user-editable notebook
variable.

| Config key | Notebook variable | Contains | Writable | Persistence |
| --- | --- | --- | --- | --- |
| `data_root` | `DATA_ROOT` | research images referenced by manifests | read is enough | useful; re-uploading data is expensive |
| `manifest_root` | `MANIFEST_ROOT` | dataset manifest CSVs | read is enough | useful; manifests are provisioned artifacts |
| `cache_root` | `CACHE_ROOT` | CLIP embedding cache, MLLM output cache | **required** | optional, but persistence avoids recomputation |
| `output_root` | `OUTPUT_ROOT` | run directories, metrics, the report | **required** | **required** if results must outlive the session |
| `external_root` | `EXTERNAL_ROOT` | official NPR checkout managed by the adapter | **required** | optional; a fresh clone needs network |

Dataset manifest paths and the NPR checkpoint path are separate config fields
today, so setting `MANIFEST_ROOT` alone does not relocate them. Both notebooks
expose them individually:

| Config key | Notebook variable | Default filename |
| --- | --- | --- |
| `datasets.tiny_genimage.train_manifest` | `TINY_TRAIN_MANIFEST` | `tiny_genimage_train.csv` |
| `datasets.tiny_genimage.dev_manifest` | `TINY_DEV_MANIFEST` | `tiny_genimage_dev.csv` |
| `datasets.genimage_unseen.manifest` | `GENIMAGE_UNSEEN_MANIFEST` | `genimage_unseen_external.csv` |
| `datasets.synthbuster.manifest` | `SYNTHBUSTER_MANIFEST` | `synthbuster_external.csv` |
| `baselines.npr.checkpoint_path` | `NPR_CHECKPOINT_PATH` | `NPR.pth` |

The generated runtime config is written to

```text
<REPO_ROOT>/.cache/aiforensics-notebook/phase_ab_colab.yaml
<REPO_ROOT>/.cache/aiforensics-notebook/phase_ab_kaggle.yaml
```

Two constraints force that location: the config loader finds the repository root
by walking up to `pyproject.toml`, so the config must live under `REPO_ROOT`; and
`.cache/` is git-ignored, so generated configs are never committed.

The notebooks change path values only, plus the `BUILD_MANIFESTS` switch below.
Dataset enable flags, generator lists, model ids, prompt ids, the CLIP seed list,
the metric list, report policy, and the pinned NPR repository URL and commit stay
exactly as committed.

## Data and Manifest Provisioning

**`aiforensics prepare` is not a research-dataset downloader.** It never fetches
images. It has two modes over data you already provisioned:

1. **validate only** (default) — checks the manifest CSVs the config points at,
2. **build then validate** (`--build-manifests`) — derives those CSVs from a
   GenImage-layout `data_root`, then validates them.

Building expects this on-disk layout, which GenImage-style releases (including
Tiny-GenImage) already use:

```text
<data_root>/<generator>/<train|val>/<ai|nature>/*.png|*.jpg|*.jpeg
```

The mapping is fixed: `ai` is label `fake`, `nature` is label `real`, `train`
becomes split `train`, and `val` becomes split `dev`. Which generator directories
are in-distribution versus held out comes from the config, not the notebook:

| Config field | Role |
| --- | --- |
| `datasets.tiny_genimage.generators` | in-distribution; produces train + dev manifests |
| `datasets.genimage_unseen.generators` | held out; produces the external manifest |
| `max_images` | **per-generator** cap per manifest; `0` means no cap |
| `balance_labels` | split the cap evenly between real and fake |

The cap is per generator, not a pooled total, so per-source metrics stay
comparable instead of letting one large generator dominate the sample.

Two leakage rules are enforced, not advised:

- a generator listed in both roles is rejected outright,
- one image never enters two manifests. Content is deduplicated by SHA-256
  across the whole build with precedence train > dev > external, because
  GenImage reuses the same real ImageNet photographs across generators. The
  build reports how many duplicates it skipped.

`prepare` also reports the file-extension mix per label. If real images are all
PNG and fake images all JPEG, container format becomes a shortcut a detector can
learn instead of generation artifacts, so the build says so:

```text
[prepare] format warning: tiny_genimage split=train: real={.png:400} vs fake={.jpg:400};
container format may act as a shortcut feature
```

Before a full run:

1. raw images must already be available to the runtime or mounted storage,
2. manifests for every **enabled** dataset must exist, or `BUILD_MANIFESTS` must
   be true so they are built in-session,
3. manifest image paths must resolve under the configured `data_root`,
4. `prepare` validates those artifacts and writes
   `<OUTPUT_ROOT>/manifest_validation.json`,
5. a missing dataset is a data-provisioning problem. Do not add hidden download
   logic to the notebook.

Because building **overwrites** the manifest CSVs, it stays opt-in. Set
`BUILD_MANIFESTS = False` once manifests are provisioned and you want them left
untouched.

An example layout (not a required host path):

```text
<PERSIST_ROOT>/
  data/
    ... research images ...
  manifests/
    tiny_genimage_train.csv
    tiny_genimage_dev.csv
    genimage_unseen_external.csv
    synthbuster_external.csv
  cache/
  outputs/
  external/
  checkpoints/
    NPR.pth
```

Use `configs/phase_ab_smoke.yaml` to exercise the pipeline without research data.

## Google Colab

Open `notebooks/colab_phase_ab.ipynb` and work top to bottom.

Storage choices, both supported by the notebook:

1. **Ephemeral runtime storage** (`USE_DRIVE = False`). Fast, needs no mount, and
   **disappears when the runtime is recycled**. Fine for smoke verification.
2. **Mounted Drive storage** (`USE_DRIVE = True`). The notebook mounts Drive at
   `DRIVE_MOUNT_POINT` and derives `PERSIST_ROOT` from `DRIVE_PROJECT_SUBPATH`.

`DRIVE_PROJECT_SUBPATH` ships as the placeholder
`MyDrive/<your-folder>/ai-image-forensics`. It is an example: change it. No
personal Drive folder is assumed anywhere.

Notes:

- Colab sessions have wall-clock and idle limits; long real runs can be
  interrupted. Put `output_root` and `cache_root` on Drive when a run must
  survive.
- Do not put tokens in notebook source. Use an environment variable or a Colab
  secret and document only the variable name.

## Kaggle

Open `notebooks/kaggle_phase_ab.ipynb` and work top to bottom.

Kaggle separates read-only inputs from writable working storage, and the notebook
keeps that split explicit:

- `/kaggle/input/<your-dataset>` is **read-only**. Research images, provisioned
  manifests, and the NPR checkpoint normally live in attached datasets. The
  notebook maps `DATA_ROOT` and `NPR_CHECKPOINT_PATH` here.
- `/kaggle/working` is **writable** and ephemeral. The notebook maps
  `CACHE_ROOT`, `OUTPUT_ROOT`, and `EXTERNAL_ROOT` here.

`MANIFEST_ROOT` depends on `BUILD_MANIFESTS`, because the two modes have opposite
storage requirements:

| `BUILD_MANIFESTS` | `MANIFEST_ROOT` | Why |
| --- | --- | --- |
| `True` | `/kaggle/working/manifests` | building writes CSVs, and `/kaggle/input` is read-only |
| `False` | `INPUT_MANIFEST_DIR` | manifests are already provisioned in an attached dataset |

Building only **reads** images from `DATA_ROOT`, so a read-only image dataset is
fine. Built manifests live in `/kaggle/working` and disappear with the session;
save the notebook output, or attach them as a dataset and switch
`BUILD_MANIFESTS` to `False`, to reuse the exact same manifests later.

Never point `cache_root`, `output_root`, or `external_root` at `/kaggle/input`;
the adapters must be able to write there.

Other Kaggle specifics:

- No dataset slug, username, or competition path is hardcoded. Replace the
  `<...>` placeholders with the datasets you attached. Give the **full**
  directory path: Kaggle mounts datasets under more than one shape, so
  `/kaggle/input/<slug>` and `/kaggle/input/datasets/<owner>/<slug>` both occur.
- Section 7 lists the generator directories it can see under `DATA_ROOT` and
  flags any generator the config asks for but cannot find. A wrong `DATA_ROOT` is
  the most common first-run failure, and this surfaces it before `prepare` runs.
- To keep artifacts after the session ends, save/version them through Kaggle's
  normal notebook output workflow.
- Package installation, Python 3.10 provisioning, model downloads, and the NPR
  clone all need the notebook **Internet** setting enabled. With Internet off,
  use a pre-provisioned environment and attach models/checkpoints as datasets.

## NPR Checkpoint and External Repository

NPR stays an external pinned baseline:

- the adapter owns clone, fetch, and checkout verification of the **official**
  repository at the **pinned** commit recorded in `configs/phase_ab.yaml`,
- the notebook must not vendor, clone, or patch NPR source,
- `external_root` must be writable, because the verified checkout lives there,
- `checkpoint_path` must point at the operator-provided NPR checkpoint,
- when a real 64-character SHA-256 is configured in `checkpoint_sha256`, it must
  match the checkpoint bytes,
- a checksum or integrity failure is a **failure**, never an environment
  deferral,
- missing checkpoint and network-unavailable clone/fetch follow the adapter's
  `allow_deferred` policy: `deferred` when deferral is allowed, `failed` when it
  is not.

Do not add a cell that downloads a checkpoint from an unverified third-party URL.

## GPU and Optional Dependencies

Optional dependency groups come from `pyproject.toml`:

| Extra | Used by |
| --- | --- |
| `clip` | CLIP probe (`open_clip_torch`) |
| `qwen` | Qwen-VL and Assisted Qwen (`torch`, `transformers`, `accelerate`, `qwen-vl-utils`) |
| `npr` | NPR runtime bridge (`torch`) |

The notebooks install them together:

```bash
python -m pip install -e ".[clip,qwen,npr]"
```

Device behaviour stays in the adapters:

- real Qwen-VL and Assisted Qwen inference expects CUDA,
- leave `runtime.device: auto` alone for the normal hosted-GPU flow; the adapter
  resolves it,
- missing GPU, missing dependencies, or unavailable model weights follow each
  adapter's `allow_deferred` contract,
- a failure that happens **after** per-sample inference has started is a real
  failure. The notebook must not turn it into a deferred success.

CLIP notes: one CLI command owns every configured seed, and the embedding cache
lives under `cache_root`, so putting that root on persistent storage lets seeds
and sessions reuse embeddings.

### Provisioning Python 3.10 for the CLI

The repository requires `>=3.10,<3.11`. Hosted kernels are often newer. That is
fine: the kernel never imports `aiforensics`, the CLI does. The notebooks
therefore run this flow.

1. **Preflight** reports the kernel version (informational) and searches for a
   Python 3.10 interpreter for the CLI.
2. If none is found, it says so and points at the optional provisioning cell.
   It does **not** fail merely because the kernel is 3.11 or newer.
3. **Optional provisioning** creates a dedicated 3.10 environment:

   ```bash
   pip install uv
   uv python install 3.10
   uv venv --seed --no-project --python 3.10 "$CLI_VENV_PATH"
   ```

   `--seed` installs pip inside the environment, so the normal
   `python -m pip install -e ".[clip,qwen,npr]"` cell works unchanged. The
   notebook then prepends `<CLI_VENV_PATH>/bin` to `PATH`, which is why every
   later cell can call `aiforensics` with no path prefix.
4. **Verification** checks that `python` on `PATH` is 3.10.x, and after the
   install cell that `aiforensics` resolves from that environment.
5. The run **fails** only if no valid Python 3.10 environment exists after
   provisioning.

This satisfies the version contract honestly: the package is installed under a
real 3.10 interpreter. Do not patch `pyproject.toml` to bypass the contract;
widening `requires-python` would need separate repository-wide validation.

Provisioning needs **network access** for `uv`, the interpreter download, and the
dependencies. On Kaggle that means enabling Internet, or using an environment
that is already provisioned with Python 3.10. On Colab the environment is
ephemeral: `CLI_VENV_PATH` must be recreated after a runtime recycle. Placing it
on Drive can help, but virtual environments are not fully relocatable, so
recreating it is usually simpler than reusing a stale one.

## Smoke Verification

The smoke flow proves that installation and the CLI pipeline work in the hosted
environment.

> Smoke metrics are pipeline checks, **not scientific evidence**. They must never
> be used as a later-phase go/no-go decision.

The notebook smoke section generates a copy of `configs/phase_ab_smoke.yaml` that
relocates `cache_root` and `output_root` only. Smoke `data_root` and smoke
manifests keep pointing at the committed repository fixtures, because those
fixtures *are* the smoke dataset. The committed smoke config is never modified,
and smoke mode is never pointed at external research data.

Expected smoke outcome with the committed config: CLIP probe completes, and
Qwen-VL, Assisted Qwen, and NPR record `deferred` because they are disabled.
`evaluate` and `report` still exit `0`, and the report is written to
`<smoke output_root>/phase_ab_smoke_report.md`.

## Full Phase A/B Run

Preconditions: provisioned data and manifests, writable `cache_root`,
`output_root`, `external_root`, a CUDA GPU, model access, and the NPR checkpoint.

Run the seven commands from
[Shared Phase A/B Command Order](#shared-phase-ab-command-order) against the
generated runtime config. Each notebook cell uses `set -euo pipefail`, so a real
failure stops the cell and stays visible.

What to expect:

- `prepare` writes `manifest_validation.json` and exits non-zero if validation
  fails,
- `clip_probe` produces one run directory per configured seed,
- `qwen_vl` and `npr` either complete or record an honest `deferred`/`failed`
  status,
- `assisted_qwen` needs completed CLIP artifacts,
- `evaluate` writes `metrics.json` and `metrics_by_source.csv` beside each
  completed `predictions.jsonl` that belongs to the current config, and reports
  how many runs it skipped as out of scope,
- `report` writes the configured Markdown report and exits `0` even when some
  baselines are `deferred`, `failed`, or `missing`.

## Artifacts and Persistence

```text
<OUTPUT_ROOT>/manifest_validation.json
<OUTPUT_ROOT>/<run_id>/config.yaml
<OUTPUT_ROOT>/<run_id>/run_scope.json
<OUTPUT_ROOT>/<run_id>/environment.json
<OUTPUT_ROOT>/<run_id>/logs.txt
<OUTPUT_ROOT>/<run_id>/status.json
<OUTPUT_ROOT>/<run_id>/predictions.jsonl
<OUTPUT_ROOT>/<run_id>/metrics.json
<OUTPUT_ROOT>/<run_id>/metrics_by_source.csv
<OUTPUT_ROOT>/<configured report filename>
```

`status.json` is authoritative for a run's outcome. `run_scope.json` records the
experiment identity: project phase, `data_root`, which dataset slices are
enabled, their manifests, and a digest of the evaluation sample ids.

`evaluate`, `report`, and `assisted_qwen` only consider runs whose
`run_scope.json` matches the config you pass them, so one `output_root` can hold
smoke, full, and external-only experiments without cross-contamination. Two
consequences worth knowing:

- Runs created before scopes existed carry no `run_scope.json` and are skipped;
  re-run those baselines if you still need them in a report.
- Switching a dataset flag or manifest changes the scope, so previously
  completed runs become `missing` for the new config rather than being reused.

Persistence:

- **Colab**: everything outside mounted Drive is lost when the runtime recycles.
- **Kaggle**: `/kaggle/working` is lost when the session ends unless you save the
  notebook output.
- Cache reuse across sessions requires a persistent `cache_root`.
- Reusing the NPR checkout across sessions requires a persistent `external_root`.

Never commit datasets, checkpoints, caches, generated outputs, or generated
runtime configs.

## Failure / Deferred Troubleshooting

First, the distinction that matters:

- **`deferred`** means the baseline could not run for an environment reason its
  adapter recognises (disabled in config, missing dependency, no CUDA, missing
  checkpoint, repository unreachable) while `allow_deferred: true`. The pipeline
  continues and the report shows the status honestly.
- **`failed`** means something the adapter treats as a real error: a checksum
  mismatch, an invalid configuration, a crash after inference started, or the
  same environment problem while `allow_deferred: false`.
- **`missing`** is a reporting-only state meaning no run artifact exists for a
  slot. It is never written into `status.json`.

Never edit a `status.json` by hand and never disable validation to move past an
error.

| Symptom | Cause | Action |
| --- | --- | --- |
| pip refuses to install the package | runtime Python is not 3.10.x | run the optional provisioning cell; do not patch `pyproject.toml` |
| provisioning cell fails while fetching `uv` or the interpreter | no network | enable Kaggle Internet, or use a pre-provisioned Python 3.10 environment |
| verification cell raises after provisioning | no valid Python 3.10 on `PATH` | re-run provisioning, check `CLI_VENV_PATH`, confirm `<venv>/bin` is first on `PATH` |
| `aiforensics: command not found` | install cell did not run, or `PATH` was reset | re-run the install cell, then the CLI verification cell |
| `nvidia-smi` missing, baselines defer | no GPU in this runtime | select a GPU runtime; CPU inference is not supported for the heavy baselines |
| Qwen baselines defer | `qwen` extra missing, or model weights unreachable | install extras, enable network/model access; do not call Transformers from the notebook |
| `prepare` reports missing manifests | manifests not provisioned and `BUILD_MANIFESTS` is false | set `BUILD_MANIFESTS = True` to build them from `data_root`, or provision the CSVs; `prepare` never downloads datasets |
| `prepare --build-manifests` says a generator directory does not exist | `DATA_ROOT` is wrong, or the config names a generator the dataset does not ship | compare against the generator list printed by section 7; the error also lists what it did find |
| `prepare --build-manifests` fails writing CSVs | `MANIFEST_ROOT` is under `/kaggle/input` | set `BUILD_MANIFESTS = True` so the manifest root moves to `/kaggle/working` |
| build reports "no train records" or "no dev records" | the in-distribution generator has no `train/` or `val/` subdirectory with images | pick a generator that ships both splits, or move it to `genimage_unseen`, which uses all splits |
| build skipped many duplicate images | the same real photographs appear across generators | expected; deduplication is what keeps a trained image out of the evaluation set |
| `format warning` after building | real and fake images use different container formats | investigate before trusting metrics: a detector can learn PNG-vs-JPEG instead of generation artifacts |
| manifest references a missing image | manifest paths do not resolve under `data_root` | fix `DATA_ROOT` or rebuild manifests with `--build-manifests` |
| NPR defers on a missing checkpoint | `checkpoint_path` does not exist | provide the official checkpoint; never fetch an unverified one |
| NPR defers on clone/fetch | repository unreachable | enable network, or pre-provision `external_root` with the verified checkout |
| NPR **fails** on checksum | checkpoint bytes do not match `checkpoint_sha256` | replace the checkpoint; this is not an environment deferral |
| permission error writing artifacts | `cache_root`/`output_root`/`external_root` point at read-only storage | move them to writable storage (`/kaggle/working`, Drive, or runtime disk) |
| artifacts vanished | hosted session reset, ephemeral storage | use persistent storage, or save/version the outputs before the session ends |
| report generated with incomplete baselines | some enabled slots are `deferred`/`failed`/`missing` | expected; the report says evidence is incomplete and selects no winner |

## Reproducibility Checklist

Before trusting a hosted result:

- [ ] the CLI ran under Python 3.10 (verification cells passed),
- [ ] `REPO_ROOT` points at the intended commit of this repository,
- [ ] the runtime config changed **path values only**,
- [ ] dataset enable flags, generator lists, model ids, prompt ids, CLIP seeds,
      metrics, report policy, and the pinned NPR commit are unchanged,
- [ ] the generator directories section 7 found match the ones the config names,
- [ ] `prepare` validated the manifests, and any `format warning` was reviewed,
- [ ] the manifests used are recorded: built in-session (`BUILD_MANIFESTS = True`,
      so they vanish with the session unless saved) or provisioned as a dataset,
- [ ] the seven CLI commands ran in the documented order,
- [ ] `assisted_qwen` ran after `clip_probe` completed,
- [ ] every baseline's `status.json` reflects a real outcome, not a suppressed
      error,
- [ ] NPR integrity checks passed, or its status honestly records why not,
- [ ] `evaluate` produced metric artifacts for every completed run,
- [ ] `report` exited `0` and the Markdown report exists,
- [ ] smoke results are labelled as pipeline checks, not scientific evidence,
- [ ] artifacts needed later were copied to persistent storage,
- [ ] notebook outputs were cleared before committing.
