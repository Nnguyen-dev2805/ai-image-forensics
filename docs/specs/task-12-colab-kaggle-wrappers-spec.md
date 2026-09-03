# Task 12 Spec: Colab and Kaggle Wrappers

## Goal

Implement thin Google Colab and Kaggle entry notebooks plus an operator runbook for running the
existing Phase A/B CLI in hosted notebook environments.

Create:

```text
notebooks/colab_phase_ab.ipynb
notebooks/kaggle_phase_ab.ipynb
docs/runbook-colab-kaggle.md
```

Task 12 must make the existing local workflow understandable and reproducible on Colab/Kaggle
without creating a second implementation of the pipeline.

The notebooks must only:

- install the repository package and optional runtime dependencies,
- perform minimal platform preflight checks,
- mount or point to user-controlled storage,
- define user-editable path inputs,
- create a runtime YAML config by changing environment/path values only,
- call the existing `aiforensics` CLI commands,
- show where run artifacts and the final Markdown report were written.

The notebooks must not implement dataset parsing, manifest validation, model loading, inference,
metrics, reporting, caching, NPR checkout logic, prediction parsing, or research decision logic.
Those behaviors already belong to Tasks 1-11.

Task 12 is documentation/orchestration work. It must not change scientific behavior.

## Prerequisites

Task 12 depends on Tasks 1-11 being complete.

Before changing notebook/runbook files, verify:

```bash
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
uv run pytest
```

Expected before Task 12 changes:

- the full repository test suite is green,
- all four baseline CLI adapters already exist,
- `aiforensics prepare` validates smoke data and existing real manifests,
- `aiforensics evaluate` writes metric artifacts,
- `aiforensics report` writes the deterministic Phase A/B Markdown report,
- Qwen-VL, Assisted Qwen, and NPR already own their deferred/failed environment behavior,
- notebooks do not need direct access to any internal baseline adapter.

Task 12 must not weaken the repository-wide verification gate.

## Required Reading

Before implementation, read:

- `AGENTS.md`
- `CLAUDE.md` when using Claude Code
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/schemas/manifest.md`
- `docs/schemas/predictions-jsonl.md`
- `docs/specs/task-3-manifest-builder-validator-spec.md`
- `docs/specs/task-6-run-artifacts-cache-keys-spec.md`
- `docs/specs/task-7-clip-probe-baseline-spec.md`
- `docs/specs/task-8-qwen-vl-baseline-spec.md`
- `docs/specs/task-9-assisted-qwen-baseline-spec.md`
- `docs/specs/task-10-npr-external-adapter-spec.md`
- `docs/specs/task-11-reporting-spec.md`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`
- `pyproject.toml`
- `src/aiforensics/config/load.py`
- `src/aiforensics/config/models.py`
- `src/aiforensics/cli/main.py`

## Scope Boundary

Task 12 owns only hosted-notebook wrappers and their operator documentation.

It may:

- add the two `.ipynb` files,
- add the Colab/Kaggle runbook,
- add lightweight deterministic tests that inspect notebook JSON and documentation,
- use standard-library/PyYAML code inside notebooks only for path/config setup,
- use platform-specific mounting APIs in the corresponding notebook,
- call shell commands or subprocesses that invoke the public CLI.

It must not:

- add a new model or baseline,
- add or change classification metrics,
- change report recommendation logic,
- change prediction/manifest/status schemas,
- duplicate baseline inference inside notebooks,
- import baseline adapters from notebook cells,
- download/build real research datasets inside core package code,
- copy NPR source into this repository,
- hardcode a user's Drive folder, Kaggle dataset slug, username, home directory, or secret,
- embed API keys/tokens in notebook JSON,
- bypass checksum/manifest validation,
- silently modify `configs/phase_ab.yaml` in place,
- silently remove the project's Python-version requirement,
- make Task 12 tests depend on network, GPU, model downloads, or notebook execution.

If implementation discovers that a hosted platform's current runtime cannot satisfy the repository's
existing `requires-python` contract, document that as an environment preflight failure. Do not edit
`pyproject.toml` merely to make the notebook install succeed unless Python compatibility is separately
validated across the repository and explicitly approved as a scope expansion.

## Existing CLI Contract

The notebooks must use the existing public CLI unchanged:

```bash
aiforensics prepare --config <runtime-config>
aiforensics run --baseline clip_probe --config <runtime-config>
aiforensics run --baseline qwen_vl --config <runtime-config>
aiforensics run --baseline npr --config <runtime-config>
aiforensics run --baseline assisted_qwen --config <runtime-config>
aiforensics evaluate --config <runtime-config>
aiforensics report --config <runtime-config>
```

Do not call `_cmd_*()` functions directly.

Do not instantiate adapters directly from a notebook.

Run ordering matters:

```text
prepare
  -> clip_probe
  -> qwen_vl
  -> npr
  -> assisted_qwen
  -> evaluate
  -> report
```

`assisted_qwen` must run after `clip_probe` because its Phase A/B contract consumes CLIP assistant
predictions.

Qwen/NPR environment unavailability is already handled by their adapters. The notebook must not
invent a second deferred/failure policy.

## Important Existing Data Limitation

Task 3 intentionally did not implement real dataset download/build logic.

For non-smoke `phase_ab` runs, `aiforensics prepare` validates configured manifest files that already
exist. Therefore Task 12 must not imply that running the notebook automatically provisions the real
research datasets.

The runbook must state clearly:

1. raw image data must already be available to the notebook runtime or mounted storage,
2. enabled dataset manifests must already exist for a full comparable Phase A/B experiment,
3. manifest image paths must resolve under the configured `data_root` according to the existing
   manifest contract,
4. `prepare` validates these artifacts; it is not a research-dataset downloader,
5. missing real manifests are an operator/data-provisioning problem, not a reason to add hidden
   notebook download logic.

The smoke config remains available for verifying the CLI pipeline without real research datasets.

## Files

Create:

```text
notebooks/colab_phase_ab.ipynb
notebooks/kaggle_phase_ab.ipynb
docs/runbook-colab-kaggle.md
tests/test_notebook_wrappers.py
```

Do not modify `src/aiforensics/` for Task 12 unless a genuine pre-existing portability bug prevents
the documented public CLI contract from working and the fix is independently justified.

Do not modify `configs/phase_ab.yaml` or `configs/phase_ab_smoke.yaml` in place.

## Notebook Design Principles

Both notebooks must be thin wrappers around the same repository code.

Allowed notebook Python logic:

- `pathlib.Path`, `os`, `sys`, `subprocess`, `yaml`, and platform mounting imports,
- validating user-entered paths,
- deriving storage roots from a small set of user-controlled variables,
- copying `configs/phase_ab.yaml` into a generated runtime config,
- replacing path/config values described in this spec,
- setting environment variables used by later shell cells,
- displaying artifact/report locations.

Disallowed notebook Python logic:

- opening images for preprocessing,
- reading manifests to select samples,
- computing checksums,
- building train/dev splits,
- loading Torch/Transformers/OpenCLIP directly,
- constructing prompts,
- parsing MLLM output,
- calculating accuracy/F1/AUROC,
- scanning `status.json` to implement custom run selection,
- generating report Markdown itself,
- cloning/checking out NPR manually when the NPR adapter already owns that behavior.

A useful rule for review:

> If a notebook cell would still be useful as part of `src/aiforensics/`, it probably does not belong
> in the notebook.

## Notebook Serialization Contract

Commit normal Jupyter notebook version-4 JSON.

Both notebooks must:

- use `nbformat: 4`,
- contain Markdown explanations before destructive/expensive steps,
- have all committed code-cell `execution_count` values set to `null`,
- have all committed cell `outputs` arrays empty,
- contain no saved model output, report text, local absolute user paths, secrets, or access tokens,
- avoid widget state and large metadata blobs,
- use a standard Python 3 kernelspec where practical.

Notebook outputs must be cleared before commit even if manual verification was performed.

## Shared Notebook Flow

Both notebooks should follow the same conceptual sections and ordering.

### 1. Title and Scope

First Markdown section must explain:

- this notebook is only a wrapper around the repository CLI,
- full Phase A/B requires pre-provisioned manifests/data,
- heavy baselines may require CUDA and network/model access,
- smoke mode is appropriate for pipeline verification but not scientific evidence.

### 2. Runtime Preflight

Show at least:

```text
Python version
current working directory
GPU visibility / nvidia-smi when available
```

The notebook must check the repository's current Python compatibility before package installation.

For the current project contract this means Python 3.10.x because `pyproject.toml` declares:

```text
>=3.10,<3.11
```

Do not patch package metadata from inside the notebook to bypass an incompatible runtime.

GPU preflight is informational. Actual baseline device/deferred behavior stays inside the adapters.

### 3. Repository Location

Expose a user-editable repository location rather than assuming a personal path.

The notebook may support either:

- a repository already present in the runtime, or
- an explicit user-supplied Git clone URL/location.

Do not hardcode a private repository URL or personal Git credential.

After setup, validate that:

```text
<repo_root>/pyproject.toml
<repo_root>/configs/phase_ab.yaml
```

exist before continuing.

### 4. Install Package

Hosted notebooks should not require `uv` to be preinstalled.

Use the repository's package metadata with pip, for example from `repo_root`:

```bash
python -m pip install -e ".[clip,qwen,npr]"
```

The exact install cell may be split into a base install and optional heavy extras if that is clearer,
but dependency names must come from `pyproject.toml`; do not duplicate pinned package lists in the
notebook.

Do not install the `dev` extra unless it is specifically needed for notebook verification.

The runbook must explain that model weights and the NPR checkpoint are not packaged with the repo.

### 5. Configure Storage Inputs

The notebooks must expose user-controlled values for at least:

```text
REPO_ROOT
DATA_ROOT
MANIFEST_ROOT
CACHE_ROOT
OUTPUT_ROOT
EXTERNAL_ROOT
NPR_CHECKPOINT_PATH
```

Use descriptive variables; do not hide these inside an opaque helper.

All five project roots required by the master plan must be visible:

```text
data_root
manifest_root
cache_root
output_root
external_root
```

The user must be able to point them to mounted/persistent storage without editing core package code.

### 6. Generate Runtime Config

Never edit `configs/phase_ab.yaml` in place.

Load it as the template and write a generated config under an ignored directory inside the repository,
for example:

```text
<repo_root>/.cache/aiforensics-notebook/phase_ab_colab.yaml
<repo_root>/.cache/aiforensics-notebook/phase_ab_kaggle.yaml
```

The generated config must remain underneath `repo_root` because the current config loader discovers
the repository root by walking upward to `pyproject.toml`.

The runtime config must set these absolute values from notebook inputs:

```yaml
paths:
  data_root: <DATA_ROOT>
  manifest_root: <MANIFEST_ROOT>
  cache_root: <CACHE_ROOT>
  output_root: <OUTPUT_ROOT>
  external_root: <EXTERNAL_ROOT>
```

Because dataset manifest paths are explicit config fields today, setting `manifest_root` alone does
not relocate them. The generated config must also map:

```text
datasets.tiny_genimage.train_manifest
    -> MANIFEST_ROOT / tiny_genimage_train.csv

datasets.tiny_genimage.dev_manifest
    -> MANIFEST_ROOT / tiny_genimage_dev.csv

datasets.genimage_unseen.manifest
    -> MANIFEST_ROOT / genimage_midjourney_external.csv

datasets.synthbuster.manifest
    -> MANIFEST_ROOT / synthbuster_external.csv
```

The notebook may expose these four paths individually if the operator uses different filenames.
The important requirement is that the values are explicit and user-controlled.

Also set:

```text
baselines.npr.checkpoint_path -> NPR_CHECKPOINT_PATH
```

Do not change:

- dataset enabled/disabled flags,
- model ids,
- prompt ids,
- CLIP seed list,
- metric list,
- report policy,
- NPR repo URL or pinned commit,
- model thresholds/decision behavior,

unless the notebook presents the change as an explicit user-editable experiment override rather than
an invisible platform default.

For the default Task 12 notebook flow, path changes are sufficient.

### 7. Validate Provisioned Inputs

Before the full real run, print/check the expected paths without reading dataset contents into custom
notebook logic.

At minimum identify:

- runtime config path,
- data root,
- manifest paths for enabled datasets,
- output root,
- cache root,
- external root,
- NPR checkpoint path.

A missing NPR checkpoint should be explained before the NPR command. Do not download an unofficial
checkpoint automatically.

Do not duplicate manifest validation; `aiforensics prepare` remains authoritative.

### 8. Execute CLI Pipeline

Use the generated runtime config and call the public CLI.

Recommended command cell:

```bash
set -euo pipefail

aiforensics prepare --config "$AIF_CONFIG"
aiforensics run --baseline clip_probe --config "$AIF_CONFIG"
aiforensics run --baseline qwen_vl --config "$AIF_CONFIG"
aiforensics run --baseline npr --config "$AIF_CONFIG"
aiforensics run --baseline assisted_qwen --config "$AIF_CONFIG"
aiforensics evaluate --config "$AIF_CONFIG"
aiforensics report --config "$AIF_CONFIG"
```

Equivalent separate cells are acceptable and often easier to recover/re-run, provided the order is
preserved and each cell visibly calls the CLI.

Do not use `|| true` or blanket exception suppression to make failed commands appear successful.

A baseline that legitimately returns a deferred artifact according to its adapter contract is an
acceptable environment outcome. A real CLI exit failure should remain visible to the operator.

### 9. Artifact Inspection

The final notebook section should point the user to, not reimplement parsing of:

```text
<OUTPUT_ROOT>/manifest_validation.json
<OUTPUT_ROOT>/<run_id>/status.json
<OUTPUT_ROOT>/<run_id>/predictions.jsonl
<OUTPUT_ROOT>/<run_id>/metrics.json
<OUTPUT_ROOT>/<run_id>/metrics_by_source.csv
<OUTPUT_ROOT>/<configured report filename>
```

It is acceptable to list directories or print the report path.

Do not add a second custom report renderer in the notebook.

## Colab-Specific Contract

`notebooks/colab_phase_ab.ipynb` may import:

```python
from google.colab import drive
```

for an explicit optional Drive-mount cell.

The mount location must be represented by a visible user-editable variable. Storage roots must be
derived from user inputs, not from a personal folder name.

The notebook must support at least these two conceptual storage choices:

1. ephemeral runtime storage for quick smoke/experimentation,
2. mounted Drive storage for caches, outputs, external checkout, checkpoint, and/or research data.

The runbook must explain that ephemeral Colab storage disappears when the runtime is recycled.

Do not assume a specific Google Drive directory such as:

```text
MyDrive/ai-image-forensics
```

without presenting it as an example that the user must choose/change.

Do not store tokens in notebook source. If a future model/download requires authentication, use an
environment variable or Colab secret mechanism and document only the variable name.

## Kaggle-Specific Contract

`notebooks/kaggle_phase_ab.ipynb` must distinguish read-only input data from writable working/output
storage.

Expose explicit user-editable inputs for:

- repository location,
- mounted Kaggle input dataset location(s),
- writable cache/output/external location.

Do not hardcode a Kaggle dataset slug, username, or competition path.

The runbook must explain:

- `/kaggle/input`-style mounted datasets are normally treated as read-only inputs,
- generated manifests/config/cache/output must go to writable storage,
- outputs that need to survive the session must be saved/versioned through Kaggle's normal notebook
  output workflow,
- Internet/model downloads may depend on notebook Internet settings and the current environment.

These platform examples belong only in the Kaggle notebook/runbook; they must never leak into core
package defaults.

## NPR Hosted-Environment Guidance

The runbook must make the NPR contract explicit:

- NPR remains an external pinned baseline,
- the adapter owns official repository clone/fetch/checkout verification,
- the notebook must not vendor or patch NPR source,
- `external_root` must be writable,
- `checkpoint_path` must point to the operator-provided NPR checkpoint,
- when a real 64-character checkpoint SHA-256 is configured, it must match,
- checksum/integrity failures are not environment deferrals,
- network-unavailable clone/fetch and missing checkpoint behavior continues to follow Task 10's
  `allow_deferred` policy.

Do not add a notebook cell that downloads a checkpoint from an unverified third-party URL.

## Qwen Hosted-Environment Guidance

The runbook must state:

- Qwen-VL and Assisted Qwen use the `qwen` optional dependency group,
- real Phase A/B Qwen inference expects CUDA according to Tasks 8/9,
- `runtime.device=auto` should be left to adapter resolution for the normal hosted-GPU flow,
- missing GPU/dependencies/model availability follows each adapter's existing `allow_deferred`
  contract,
- a failure after per-sample inference begins is not something the notebook should convert into a
  deferred success.

Do not call Transformers or Torch model APIs directly from the notebooks.

## CLIP Hosted-Environment Guidance

The runbook must state:

- CLIP uses the `clip` optional dependency group,
- the one CLI command owns all configured seeds,
- the notebook must not loop over seeds itself,
- embedding cache should live under configured `cache_root`, which may be placed on persistent
  storage when reuse across sessions is desired.

## Smoke Workflow

Both notebooks should include a clearly separated optional smoke section or documented smoke command
sequence using:

```text
configs/phase_ab_smoke.yaml
```

The smoke flow is for proving that package installation and the CLI pipeline work in the hosted
environment.

It must preserve the Task 11 warning:

> Smoke metrics are pipeline checks, not scientific evidence.

Do not modify the committed smoke config to point at external research data.

A notebook may copy the smoke config to a generated runtime config if output/cache roots need to be
relocated, but must not turn smoke mode into a real benchmark implicitly.

## Runbook Contract

Create `docs/runbook-colab-kaggle.md` as an operator-focused document.

Required sections, in this order:

```markdown
# Colab and Kaggle Runbook

## Purpose
## What Task 12 Does and Does Not Do
## Prerequisites
## Shared Phase A/B Command Order
## Required Storage and Config Paths
## Data and Manifest Provisioning
## Google Colab
## Kaggle
## NPR Checkpoint and External Repository
## GPU and Optional Dependencies
## Smoke Verification
## Full Phase A/B Run
## Artifacts and Persistence
## Failure / Deferred Troubleshooting
## Reproducibility Checklist
```

### Required path documentation

Explain all of:

```text
data_root
manifest_root
cache_root
output_root
external_root
```

For each path state:

- what it contains,
- whether it must be writable,
- whether persistence across sessions is useful/required,
- which notebook variable controls it.

Also document the explicit dataset manifest paths and NPR checkpoint path because they are separate
config fields today.

### Data provisioning section

Must say explicitly that real dataset download/build is not implemented by `prepare`.

Provide a conceptual expected layout, for example:

```text
<PERSIST_ROOT>/
  data/
    ... research images ...
  manifests/
    tiny_genimage_train.csv
    tiny_genimage_dev.csv
    genimage_midjourney_external.csv
    synthbuster_external.csv
  cache/
  outputs/
  external/
  checkpoints/
    NPR.pth
```

This is an example layout, not a hardcoded required host path.

### Full command section

Include exactly the public command order from this spec and explain why Assisted Qwen follows CLIP.

Do not replace it with Python calls into internal modules.

### Troubleshooting section

Cover at least:

- unsupported Python runtime/package install refusal,
- GPU unavailable,
- Qwen optional dependencies/model resolution unavailable,
- missing real manifest,
- manifest references missing image,
- missing NPR checkpoint,
- NPR repository network unavailable,
- NPR integrity/checksum failure,
- read-only output/cache/external path,
- hosted session reset losing ephemeral artifacts,
- `deferred` versus `failed` meaning,
- report generated with incomplete baselines.

Do not recommend suppressing validation or changing status files manually.

## Tests

Create `tests/test_notebook_wrappers.py`.

Tests must use only local files and the Python standard library unless an already-required lightweight
dependency is needed. They must never execute the notebooks, install packages, access network, load
models, or require GPU.

Minimum tests:

1. both notebook files exist and parse as JSON,
2. both notebooks declare `nbformat == 4`,
3. every committed code cell has `execution_count is None`,
4. every committed code cell has an empty `outputs` list,
5. both notebooks mention/use a generated runtime config rather than modifying
   `configs/phase_ab.yaml` in place,
6. both notebooks visibly expose `DATA_ROOT`, `MANIFEST_ROOT`, `CACHE_ROOT`, `OUTPUT_ROOT`, and
   `EXTERNAL_ROOT`,
7. both notebooks expose `NPR_CHECKPOINT_PATH`,
8. both notebooks contain the `prepare` CLI command,
9. both notebooks contain exactly one visible Phase A/B `clip_probe` command in the full-run section
   rather than a notebook-side seed loop,
10. both notebooks contain `qwen_vl`, `npr`, and `assisted_qwen` CLI commands,
11. both notebooks contain `evaluate` and `report` CLI commands,
12. full-run command ordering is `prepare -> clip_probe -> qwen_vl -> npr -> assisted_qwen ->
    evaluate -> report`,
13. notebook code does not import `aiforensics.baselines` or any baseline adapter,
14. notebook code does not import Torch, Transformers, or OpenCLIP for direct model inference,
15. notebook source contains no obvious embedded secret assignments such as literal HF/API tokens,
16. Colab notebook contains the optional Colab storage-mount guidance,
17. Kaggle notebook contains read-only-input versus writable-output guidance,
18. runbook documents all five required roots,
19. runbook explicitly states that real dataset provisioning is not implemented by `prepare`,
20. runbook documents NPR checkpoint provisioning and official external-repo behavior,
21. runbook documents smoke metrics as non-scientific pipeline evidence,
22. runbook contains the shared public CLI sequence,
23. runbook documents deferred versus failed outcomes,
24. notebooks do not contain saved output text from a prior execution.

Tests should inspect semantic markers conservatively. Avoid brittle assertions on notebook cell UUIDs,
exact Markdown prose, or incidental formatting.

## Manual Verification

Static tests cannot prove that Google Colab or Kaggle currently provides a compatible runtime, GPU,
Internet setting, or model availability. Manual hosted verification is therefore part of Task 12.

For each platform, perform at least the following when access is available:

1. open the committed notebook,
2. select a runtime satisfying the repository Python contract,
3. run the preflight cell,
4. install the package from the repository,
5. configure writable output/cache/external paths,
6. run the smoke pipeline,
7. verify smoke `report` exits `0`,
8. verify the smoke report exists,
9. clear notebook outputs before commit.

The real Phase A/B model run is recommended when GPU/data/checkpoint access is available but must not
become a default CI requirement.

For a real hosted run, verify additionally:

- provisioned real manifests pass `prepare`,
- CLIP produces configured seed run artifacts,
- Qwen-VL completes or records an honest deferred/failed status,
- NPR completes or records an honest deferred/failed status according to Task 10,
- Assisted Qwen runs only after CLIP artifacts exist,
- `evaluate` consumes completed prediction artifacts,
- `report` produces the configured Markdown report,
- output/cache persistence matches the chosen platform storage setup.

## Automated Verification Gate

After Task 12 implementation run:

```bash
uv run --extra dev pytest tests/test_notebook_wrappers.py -v
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
uv run pytest
```

Also run the existing local smoke gate to ensure documentation/notebook work did not break the public
CLI:

```bash
uv run aiforensics prepare --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline qwen_vl --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline npr --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline assisted_qwen --config configs/phase_ab_smoke.yaml
uv run aiforensics evaluate --config configs/phase_ab_smoke.yaml
uv run aiforensics report --config configs/phase_ab_smoke.yaml
```

Expected:

- notebook-wrapper tests pass without network/GPU,
- repository-wide Ruff remains green,
- full pytest remains green,
- smoke commands preserve their Task 7-11 contracts,
- smoke report is generated,
- no notebook test imports heavyweight model runtimes.

## Review Checklist

A reviewer should confirm:

### Thin-wrapper rule

- no model/pipeline implementation exists in notebook cells,
- notebooks call the CLI rather than Python adapter APIs,
- platform-specific behavior is limited to mounting/path/install setup.

### Portability

- no personal absolute paths,
- no hardcoded Kaggle dataset slug,
- no private repo URL or credentials,
- storage roots are user controlled,
- runtime config uses absolute mounted paths,
- runtime config itself is generated under the repository so `load_config()` can find
  `pyproject.toml`.

### Scientific integrity

- smoke is not described as research evidence,
- real dataset provisioning is explicit,
- notebook does not change baseline parameters invisibly,
- notebook does not catch/convert genuine failures,
- deferred behavior remains owned by baseline adapters,
- report remains produced by Task 11.

### Repository hygiene

- notebook outputs cleared,
- no checkpoints/data/cache/generated outputs committed,
- no generated runtime YAML committed,
- no new core dependency added for notebook-only convenience.

## Suggested Implementation Order

1. add failing static notebook-wrapper tests,
2. create the Colab notebook skeleton with cleared outputs,
3. create the Kaggle notebook using the same shared conceptual flow,
4. implement only path/runtime-config setup in notebook Python cells,
5. add public CLI command cells in the required order,
6. write the runbook from the exact notebook behavior,
7. run notebook static tests,
8. run repository-wide Ruff/full pytest,
9. run the existing local smoke gate,
10. manually smoke-test on Colab/Kaggle when those environments are available,
11. clear all notebook outputs and re-run static tests before commit.

## Acceptance Criteria

Task 12 is complete when:

- `notebooks/colab_phase_ab.ipynb` exists and is a thin CLI wrapper,
- `notebooks/kaggle_phase_ab.ipynb` exists and is a thin CLI wrapper,
- `docs/runbook-colab-kaggle.md` documents the hosted workflow,
- both notebooks expose user-controlled storage roots,
- both generate a runtime config without mutating the canonical config,
- all explicit manifest paths and NPR checkpoint path can be relocated,
- both call the same public CLI contract as local execution,
- Assisted Qwen follows CLIP,
- notebooks do not duplicate core pipeline/model/reporting logic,
- real dataset provisioning limitations are explicit,
- NPR checkpoint/external-repo requirements are explicit,
- Qwen/NPR GPU/deferred behavior is not reimplemented,
- smoke workflow is clearly distinguished from scientific Phase A/B evidence,
- committed notebooks contain no outputs/secrets/personal paths,
- notebook structural tests pass offline,
- full Ruff/full pytest remain green,
- existing local smoke gate remains green,
- manual hosted smoke verification is recorded when platform access is available.

## Explicitly Deferred To Task 13 or Later

Task 12 does not implement:

- final README quickstart/project documentation,
- broad Python-version expansion,
- CI execution of Colab/Kaggle,
- real dataset downloader/builders,
- automatic Kaggle Dataset creation/versioning,
- automatic Google Drive synchronization,
- secret-management infrastructure,
- checkpoint hosting/distribution,
- notebook widgets/dashboard UI,
- HTML/PDF reporting,
- new metrics/statistical tests,
- model training/fine-tuning,
- later research phases.

Task 13 owns final repository verification/documentation polish after the hosted wrappers exist.
