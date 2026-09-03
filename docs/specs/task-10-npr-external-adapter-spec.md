# Task 10 Spec: NPR External Adapter

## Goal

Implement the fourth Phase A/B baseline: `npr`.

Task 10 turns:

```bash
aiforensics run --baseline npr --config configs/phase_ab.yaml
```

from a placeholder into a reproducible adapter around the official NPR repository:

```text
https://github.com/chuangchuangtan/NPR-DeepfakeDetection
```

The baseline must:

- keep NPR source outside this repository,
- locate or clone the official NPR repository under `paths.external_root`,
- execute only a pinned NPR commit for a completed real run,
- verify the configured checkpoint when a SHA-256 is provided,
- run NPR inference without modifying the external checkout,
- obtain one fake score per evaluation image,
- map those scores into the shared `PredictionRecord` schema,
- create the standard run artifacts,
- defer cleanly when the environment cannot support NPR and `allow_deferred=true`,
- keep default tests and the smoke gate completely network-free and model-download-free.

Task 10 must not implement NPR training, fine-tuning, model fusion, Qwen changes, new metrics,
report rendering, dataset downloading, checkpoint downloading, or modifications to the official NPR
source tree.

## Research Facts Locked By This Spec

The implementation must account for the actual behavior of the official repository rather than
assuming its CLI already matches this project's manifest-first contract.

Official references:

```text
README:
https://github.com/chuangchuangtan/NPR-DeepfakeDetection

test.py:
https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/main/test.py

validate.py:
https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/main/validate.py

data/datasets.py:
https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/main/data/datasets.py
```

Important facts:

1. The README's documented detector command runs `test.py` with `NPR.pth`.
2. `test.py` contains benchmark-specific hard-coded dataset roots and prints aggregate metrics.
3. `test.py` is therefore not sufficient by itself to produce this project's per-image
   `predictions.jsonl`.
4. The official inference path uses `networks.resnet.resnet50(num_classes=1)` and a sigmoid model
   output.
5. `validate.py` treats a sigmoid score greater than `0.5` as fake.
6. The official README documents a special GenImage test preprocessing protocol: replace resize
   with translate-and-duplicate for undersized images, use crop size 224, disable test-time flip,
   and test with cropping enabled.

Do not implement Task 10 by invoking `test.py` and attempting to scrape aggregate stdout into
per-image predictions. That cannot satisfy the shared prediction schema.

## Prerequisites

Task 10 depends on Tasks 1-9 being complete.

Before starting Task 10, verify:

```bash
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
uv run pytest
```

Expected before Task 10 changes:

- the repository is fully Ruff-clean,
- the full test suite is green,
- `RunResult` and `BaselineAdapter` exist,
- run artifact helpers are stable,
- manifest and prediction schemas are stable,
- `clip_probe`, `qwen_vl`, and `assisted_qwen` behavior must not regress.

From Task 10 onward, do not return to scoped-only Ruff verification. The final Task 10 gate is full
repository Ruff plus full pytest.

## Required Reading

Before coding, read:

- `AGENTS.md`
- `CLAUDE.md` when using Claude Code
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/schemas/manifest.md`
- `docs/schemas/predictions-jsonl.md`
- `docs/specs/task-6-run-artifacts-cache-keys-spec.md`
- `docs/specs/task-7-clip-probe-baseline-spec.md`
- `docs/specs/task-8-qwen-vl-baseline-spec.md`
- `docs/specs/task-9-assisted-qwen-baseline-spec.md`
- `src/aiforensics/baselines/base.py`
- `src/aiforensics/cli/main.py`
- `src/aiforensics/config/models.py`
- `src/aiforensics/config/load.py`
- `src/aiforensics/data/manifest.py`
- `src/aiforensics/schemas/predictions.py`
- `src/aiforensics/runs/artifacts.py`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`
- the pinned external NPR README, `test.py`, `validate.py`, `data/datasets.py`, and
  `networks/resnet.py` before implementing the real runtime bridge.

## Scope Boundary

Task 10 is an external pretrained detector adapter.

It is not a reimplementation of NPR.

The project may own only the integration code needed to:

- manage the external checkout,
- validate the checkpoint,
- adapt manifest records into deterministic image inputs,
- invoke the official NPR network from that checkout,
- collect per-image scores,
- convert those scores to the shared schema.

The project must not:

- copy `networks/`, `data/`, `options/`, `test.py`, or `validate.py` from NPR into
  `src/aiforensics/`,
- vendor NPR code into this repository,
- patch files inside `external/NPR-DeepfakeDetection`,
- train NPR,
- download datasets,
- silently switch to a fork,
- silently switch to a different checkpoint,
- silently run a floating branch or latest `main`,
- parse aggregate NPR benchmark metrics as if they were per-image predictions.

## Files To Create Or Modify

Required:

```text
src/aiforensics/baselines/npr/__init__.py
src/aiforensics/baselines/npr/adapter.py
src/aiforensics/baselines/npr/runtime.py
src/aiforensics/cli/main.py
tests/test_npr_adapter.py
tests/test_cli_smoke.py
```

Potentially modify:

```text
pyproject.toml
configs/phase_ab.yaml
```

`pyproject.toml` may add an `npr` optional dependency group if required by the verified pinned NPR
checkout. Do not add Torch, Torchvision, SciPy, or OpenCV to the base dependency list merely to
make NPR importable.

`configs/phase_ab.yaml` may replace `repo_commit: null` with an exact verified official commit SHA.
Do not invent a commit hash. If the implementation environment cannot verify an official commit,
leave the value unchanged and the real run must not claim reproducible completion.

Do not modify:

- shared prediction schema,
- evaluation metric semantics,
- Qwen prompts or parsers,
- CLIP probe semantics,
- reporting code,
- dataset preparation behavior.

## Public Interface

Expose:

```python
from aiforensics.baselines.npr.adapter import NPRAdapter
```

Implement:

```python
class NPRAdapter:
    name = "npr"

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

`seed` does not create multiple NPR runs. NPR has one configured run per CLI invocation.

The adapter module must not import Torch, Torchvision, OpenCV, SciPy, or modules from the external
NPR checkout at module import time.

## Config Contract

Use the existing fields:

```python
config.paths.external_root

config.baselines.npr.enabled
config.baselines.npr.repo_url
config.baselines.npr.repo_commit
config.baselines.npr.checkpoint_path
config.baselines.npr.checkpoint_sha256
config.baselines.npr.batch_size
config.baselines.npr.allow_deferred

config.runtime.device
config.runtime.seed
```

Do not add hidden environment-specific paths.

### Supported Repository

Task 10 supports only the official NPR repository:

```text
https://github.com/chuangchuangtan/NPR-DeepfakeDetection
```

Normalization may accept the same URL with a trailing slash or `.git`, but a fork or unrelated URL
must fail with a clear configuration error.

### Commit Pinning

A completed real NPR run requires an exact Git commit SHA.

Requirements:

- `repo_commit` must resolve to one exact commit in the official checkout,
- branch names such as `main` are not accepted as the final reproducibility pin,
- `repo_commit=null`, empty, or an obvious smoke placeholder must not produce a completed real run,
- validate the disabled state before validating the smoke placeholder, so the existing smoke config
  remains network-free.

For an enabled real run, missing or invalid commit configuration is a configuration failure, not a
GPU/environment defer condition.

## External Repository Location

The deterministic checkout location is:

```python
repo_dir = config.paths.external_root / "NPR-DeepfakeDetection"
```

Do not clone inside `src/`, `tests/`, `outputs/`, or a notebook directory.

The checkout is runtime state and must remain ignored by Git.

## External Checkout Lifecycle

Checkout management must be isolated behind a narrow helper so tests can mock it without invoking
Git or the network.

Recommended interface:

```python
def ensure_npr_checkout(
    *,
    repo_dir: Path,
    repo_url: str,
    repo_commit: str,
    allow_deferred: bool,
) -> CheckoutInfo:
    ...
```

`CheckoutInfo` should expose at least:

```text
repo_dir
resolved_commit
```

Rules:

1. If the baseline is disabled, do not call checkout management at all.
2. If `repo_dir` does not exist, cloning is allowed for a real enabled run.
3. Clone only the configured official URL.
4. Check out the configured commit in detached-HEAD state.
5. If `repo_dir` already exists, verify it is a Git checkout of the official repository.
6. Verify the working tree is clean before using it.
7. Verify `HEAD` resolves exactly to the configured commit before inference.
8. Never run `git reset --hard`, `git clean`, or any destructive command against an existing
   checkout.
9. Never overwrite user changes in an existing checkout.
10. If a clean checkout is at another commit, obtaining and checking out the configured commit is
    allowed.
11. If network access is unavailable while cloning/fetching and `allow_deferred=true`, defer.
12. If the checkout exists but is corrupt, dirty, points at another repository, or cannot resolve a
    successfully fetched configured commit, fail.

The adapter must log the resolved commit used for every completed run.

## Checkpoint Contract

Use:

```python
checkpoint_path = config.baselines.npr.checkpoint_path
```

Task 10 does not add checkpoint download logic.

Before real inference:

1. `checkpoint_path` must exist.
2. It must be a regular file.
3. If `checkpoint_sha256` is configured with a real 64-character SHA-256 value, compute the file
   SHA-256 and require an exact case-insensitive match.
4. A checksum mismatch is a hard integrity failure and must never be converted to deferred.
5. If no checkpoint checksum is configured, completion is allowed by the architecture, but log that
   checksum verification was skipped.

Missing checkpoint behavior:

- `allow_deferred=true` -> deferred,
- `allow_deferred=false` -> failed.

Do not silently substitute another `.pth` file.

## Data Selection

NPR requires no training split in Task 10.

Use the same evaluation selection contract as Tasks 8 and 9.

Primary evaluation manifest:

```python
config.datasets.tiny_genimage.dev_manifest
```

Additionally include these only when the dataset is enabled and the manifest exists:

```python
config.datasets.genimage_unseen.manifest
config.datasets.synthbuster.manifest
```

Load with:

```python
load_manifest(path, data_root=config.paths.data_root)
```

Requirements:

- the tiny dev manifest may be absent only if at least one enabled external evaluation manifest
  exists,
- an enabled external manifest that is missing should warn and continue,
- a disabled external dataset must be ignored even if its manifest exists,
- an existing invalid manifest must fail rather than be skipped,
- if no evaluation records remain, fail,
- preserve manifest ordering and dataset ordering deterministically,
- duplicate `sample_id` across the combined evaluation records must fail,
- missing image files must fail,
- configured image checksum mismatch must fail,
- do not silently drop samples.

## NPR Phase A/B Preprocessing Profile

Task 10 uses one explicit preprocessing profile for the Phase A/B manifest bridge:

```text
npr_genimage_v1
```

This profile follows the official NPR README's documented GenImage test guidance because
Tiny-GenImage and GenImage are the primary Phase A/B data family.

For each RGB image:

1. Open with Pillow and convert to RGB.
2. Use `crop_size = 224`.
3. If either image side is smaller than 224, tile/translate-duplicate the image in each dimension
   until both dimensions are at least 224.
4. Do not resize an image merely because it is larger than 224.
5. Apply deterministic center crop to 224 x 224.
6. Do not apply test-time horizontal flip.
7. Convert to tensor.
8. Normalize with ImageNet values:

   ```text
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
   ```

9. Preserve evaluation ordering.

Do not copy NPR's dataset classes into this repository. Implement only the narrow manifest adapter
needed to reproduce this documented preprocessing behavior.

The preprocessing profile name must be logged for completed runs.

## Device Contract

The official NPR testing path is CUDA-based. Task 10 should preserve that runtime assumption rather
than quietly changing the research baseline to CPU inference.

Rules:

- `runtime.device=auto`: use CUDA when available; otherwise treat the environment as unsupported,
- `runtime.device=cuda`: require CUDA,
- `runtime.device=cpu`: real NPR inference is unsupported in Task 10,
- unsupported device/CUDA availability is deferred when allowed and failed otherwise,
- smoke/disabled NPR must not import Torch merely to inspect CUDA.

Do not add MPS behavior in Task 10.

## Runtime Isolation

Do not import the external NPR checkout directly into the long-lived CLI process.

Use a narrow subprocess boundary so NPR's top-level module names such as `networks`, `data`, and
`options` cannot pollute or collide with this project's imports.

Recommended project-owned runner:

```text
src/aiforensics/baselines/npr/runtime.py
```

Recommended invocation shape:

```bash
<sys.executable> -m aiforensics.baselines.npr.runtime \
  --repo-dir <external/NPR-DeepfakeDetection> \
  --checkpoint <NPR.pth> \
  --input-jsonl <run_dir/npr_input.jsonl> \
  --output-jsonl <run_dir/npr_scores.jsonl> \
  --batch-size <configured batch size> \
  --seed <runtime.seed> \
  --device cuda
```

Use `sys.executable`; do not use a literal `python` command.

The adapter owns subprocess construction and status handling. The runtime owns only real NPR model
setup, preprocessing, inference, and score emission.

## Runtime Input Contract

The adapter may write an internal input JSONL under the run directory.

Each row should contain only what the runtime needs, for example:

```json
{
  "sample_id": "sample-001",
  "path": "/absolute/path/image.png"
}
```

Requirements:

- one row per evaluation record,
- sample ids unique,
- absolute resolved image paths,
- stable ordering identical to the selected evaluation records,
- do not include `label_true` in runtime input.

Ground-truth labels are not needed for inference and must not be sent into the NPR runtime process.

## Real NPR Runtime Contract

Inside the subprocess only:

1. Add the verified external checkout to `sys.path` for the lifetime of the runner process.
2. Import the official network from that checkout:

   ```python
   from networks.resnet import resnet50
   ```

3. Construct:

   ```python
   model = resnet50(num_classes=1)
   ```

4. Load the configured checkpoint using the state-dict structure expected by the pinned official
   testing path. Do not add silent fallback heuristics for unrelated checkpoint formats.
5. Move the model to CUDA and call `eval()`.
6. Set Python, NumPy, and Torch random seeds from `config.runtime.seed` before inference.
7. Apply `npr_genimage_v1` preprocessing.
8. Run batches under `torch.no_grad()`.
9. Convert model logits with sigmoid.
10. Emit exactly one finite score in `[0.0, 1.0]` per input sample.

Model and checkpoint must be loaded once per runner process, not once per image.

## Runtime Score Output Contract

The runtime output JSONL should be intentionally small:

```json
{
  "sample_id": "sample-001",
  "score_fake": 0.912345
}
```

Requirements:

- exactly one score row per input sample,
- no duplicate sample id,
- no unknown sample id,
- no missing sample id,
- `score_fake` must be numeric, finite, and within `[0.0, 1.0]`,
- runtime output ordering should match input ordering,
- malformed output fails the run.

The adapter must validate this contract before building `PredictionRecord` objects.

Do not scrape model scores from human-readable stdout.

## Shared Prediction Mapping

For every evaluation `ManifestRecord`, emit one:

```python
PredictionRecord(
    sample_id=record.sample_id,
    label_true=record.label,
    label_pred=...,
    score_fake=score,
    model_name="npr",
    source=record.source,
    run_id=run_id,
    dataset=record.dataset,
    split=record.split,
    path=record.path,
    checksum=record.checksum,
    parse_status="not_applicable",
)
```

Use official NPR threshold semantics:

```python
label_pred = "fake" if score_fake > 0.5 else "real"
```

Note the strict `>` comparison. Do not silently change this to `>= 0.5` because the official NPR
validation code uses `> 0.5`.

NPR is not an MLLM. Do not populate:

- `prompt_id`,
- `raw_output`,
- `explanation`.

`parse_status="not_applicable"` is recommended for consistency with the CLIP probe baseline.

Validate with the shared `validate_predictions()` before writing.

After writing `predictions.jsonl`, read it back with `load_predictions()` and validate the loaded
records again. A completed run must never point at an invalid prediction file.

## CLI Contract

Wire:

```bash
aiforensics run --baseline npr --config <config>
```

to `NPRAdapter`.

One CLI invocation creates exactly one NPR run directory.

CLI behavior should match Tasks 8 and 9:

- completed -> exit `0`,
- deferred -> exit `0`,
- failed -> exit `1`.

Suggested summary:

```text
[run] baseline=npr runs=1 completed=<0|1> failed=<0|1> deferred=<0|1> output_root=<output_root>
```

Do not create one run per dataset or batch.

## Run Directory And Artifact Contract

Create the run directory with:

```python
create_run_dir(config.paths.output_root, "npr")
```

Completed run:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  status.json
  predictions.jsonl
  logs.txt
  npr_input.jsonl
  npr_scores.jsonl
```

`npr_input.jsonl` and `npr_scores.jsonl` are allowed as inspectable bridge artifacts. They must not
contain ground-truth labels.

Failed/deferred run:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  status.json
  logs.txt
```

If bridge files were created before a failure, they may remain for diagnosis, but a failed or
deferred run must not leave a stale `predictions.jsonl`.

Do not write metrics inside `run`. Metrics belong to `aiforensics evaluate`.

## Failure And Deferred Behavior

Use `RunStatus` and `write_status()` for every CLI run.

### Completed

A run is `completed` only when all of the following are true:

- the official repo checkout is verified,
- `HEAD` equals the configured pinned commit,
- the working tree is clean,
- the checkpoint exists,
- configured checkpoint checksum verification passes,
- CUDA runtime is usable,
- every selected evaluation image is valid,
- runtime produces exactly one valid score per sample,
- every `PredictionRecord` validates,
- written predictions pass read-back validation.

### Failed

Examples that must fail:

- unsupported/forked `repo_url`,
- missing or invalid real-run commit pin,
- existing checkout is not the official repository,
- existing checkout is dirty,
- configured commit cannot be resolved after repository access succeeds,
- checkpoint SHA-256 mismatch,
- invalid existing manifest,
- duplicate evaluation sample ids,
- missing image,
- image checksum mismatch,
- runtime subprocess starts successfully but inference crashes,
- model/checkpoint state-dict incompatibility,
- runtime produces duplicate, extra, missing, malformed, NaN, infinite, or out-of-range scores,
- prediction validation fails,
- prediction write/read-back validation fails,
- a defer-eligible environment condition occurs while `allow_deferred=false`.

### Deferred When `allow_deferred=true`

Examples:

- `npr.enabled` is false,
- the official repo is absent and network is unavailable for clone,
- the configured commit is not present locally and network is unavailable for fetch,
- checkpoint is missing,
- NPR optional dependencies are unavailable,
- CUDA is unavailable,
- `runtime.device=cpu`,
- current platform/runtime cannot initialize the pinned NPR code before per-sample inference begins.

Once real model inference has begun, an inference crash is a failed run, not a deferred run.

Integrity problems are never deferred.

`started_at` and `ended_at` must remain UTC ISO-8601 timestamps accepted by `RunStatus`.

## Logging

Keep logs concise and diagnostic.

Include:

- baseline name,
- official repo URL,
- resolved repo directory,
- configured commit,
- resolved checkout commit,
- checkout action: reused, cloned, fetched, or checked out,
- checkpoint path,
- whether checkpoint checksum verification ran,
- preprocessing profile `npr_genimage_v1`,
- resolved device,
- runtime seed,
- batch size,
- number of evaluation samples,
- subprocess command without secrets,
- subprocess exit code,
- completed/failed/deferred reason.

Do not log:

- model tensors,
- full checkpoint content,
- secrets or credentials,
- every per-image score by default,
- ground-truth labels passed to the runtime process.

## Dependency Policy

Default installation and default tests must not require NPR's heavy dependencies.

If an optional dependency group is needed, prefer:

```toml
[project.optional-dependencies]
npr = [
    # only dependencies verified as required by the pinned runtime bridge
]
```

Do not blindly copy every item from the external repository requirements into base dependencies.

The real runner should perform lazy dependency checks and produce a clear deferred/failed reason.

## Tests

Create:

```text
tests/test_npr_adapter.py
```

All default tests must run:

- without network access,
- without cloning NPR,
- without a real checkpoint,
- without CUDA,
- without importing Torch/Torchvision unless a narrow pure-runtime test explicitly injects a fake
  boundary.

Minimum coverage:

1. `NPRAdapter.name == "npr"`.
2. disabled NPR defers before repo validation.
3. disabled smoke NPR does not run Git.
4. disabled smoke NPR does not inspect checkpoint.
5. disabled smoke NPR does not import Torch.
6. official repo URL is accepted.
7. fork/unrelated repo URL fails.
8. repo path resolves under `paths.external_root/NPR-DeepfakeDetection`.
9. missing real-run commit pin fails clearly.
10. checkout helper can represent an already-correct clean checkout.
11. dirty existing checkout fails.
12. wrong existing remote fails.
13. commit mismatch cannot be silently accepted.
14. missing repo/network failure defers when allowed.
15. the same checkout setup failure fails when `allow_deferred=false`.
16. missing checkpoint defers when allowed.
17. the same missing checkpoint fails when `allow_deferred=false`.
18. checkpoint SHA-256 match passes.
19. checkpoint SHA-256 mismatch fails and is never deferred.
20. disabled optional datasets are ignored even when manifests exist.
21. missing tiny dev is allowed when an enabled external manifest exists.
22. existing invalid manifest fails rather than being skipped.
23. duplicate sample ids across selected manifests fail.
24. missing image fails.
25. image checksum mismatch fails.
26. preprocessing tiles an undersized image until both sides support a 224 crop.
27. preprocessing leaves larger image dimensions un-resized before center crop.
28. preprocessing output is exactly 224 x 224.
29. preprocessing is deterministic for the same image.
30. runtime input JSONL excludes ground-truth labels.
31. runtime input preserves evaluation order.
32. valid runtime score rows convert one-to-one to predictions.
33. `score_fake > 0.5` maps to `label_pred="fake"`.
34. `score_fake == 0.5` maps to `label_pred="real"`.
35. score below `0.5` maps to `label_pred="real"`.
36. NaN/Inf/out-of-range runtime score fails.
37. duplicate runtime score sample id fails.
38. unknown extra runtime sample id fails.
39. missing runtime sample id fails.
40. NPR prediction has `model_name="npr"` and `parse_status="not_applicable"`.
41. NPR prediction contains no MLLM-only fields.
42. prediction validation failure leaves no completed `predictions.jsonl`.
43. subprocess command uses `sys.executable`, not literal `python`.
44. runtime/model setup failure defers when allowed.
45. per-sample/batch inference failure after setup fails rather than defers.
46. model/checkpoint load boundary is called once per mocked runner execution.
47. CLI creates exactly one NPR run directory.
48. CLI completed NPR returns `0`.
49. CLI deferred NPR returns `0`.
50. CLI failed NPR returns `1`.
51. smoke config NPR invocation creates a deferred artifact and performs no network/model work.

Do not chase an exact test count if a smaller set covers multiple contracts clearly. Do not weaken a
contract merely to reduce the test count.

Use `tmp_path` for generated configs, fake external roots, fake checkouts, checkpoints, manifests,
run directories, and subprocess bridge artifacts.

Tests must not create artifacts under repository `external/` or `outputs/`.

## Mocking Rules

Mock narrow external boundaries rather than simulating Git, CUDA, and Torch internals throughout the
adapter.

Preferred boundaries:

```text
ensure/verify external checkout
check runtime availability
run NPR subprocess
load/execute model inside runtime-specific tests
```

Keep pure or nearly pure helpers for:

```text
repo URL normalization/validation
checkpoint SHA-256 validation
evaluation-record selection
preprocessing profile construction
runtime input JSONL construction
runtime score JSONL validation
PredictionRecord construction
```

Do not mock `subprocess.run` globally for every test if a smaller project helper can be mocked.

## Smoke Configuration Expectations

Keep NPR disabled in:

```text
configs/phase_ab_smoke.yaml
```

The existing smoke config intentionally contains placeholder NPR commit/checksum values. Because
`enabled=false`, they must not be validated or resolved.

Running:

```bash
uv run aiforensics run --baseline npr --config configs/phase_ab_smoke.yaml
```

must:

- create one run directory,
- write `config.yaml`, `environment.json`, `status.json`, and `logs.txt`,
- set status to `deferred`,
- return exit code `0`,
- not run Git,
- not access the network,
- not clone/fetch NPR,
- not inspect/download the checkpoint,
- not import Torch/Torchvision,
- not require CUDA,
- not write `predictions.jsonl`.

## Real Environment Verification

Real NPR inference is not required for the default CPU test gate.

When a suitable Colab/Kaggle CUDA environment is available and the real config contains a verified
commit pin plus a usable checkpoint, run:

```bash
uv run --extra npr aiforensics run --baseline npr --config configs/phase_ab.yaml
```

Expected real behavior:

- official checkout is at the exact configured commit,
- external checkout remains unmodified,
- checkpoint integrity is checked when configured,
- model loads once,
- every evaluation sample produces exactly one fake score,
- score semantics match official NPR sigmoid output,
- every evaluation sample produces exactly one shared prediction,
- completed `predictions.jsonl` validates and can be consumed by `aiforensics evaluate` without NPR
  special-casing.

If the environment cannot support NPR and `allow_deferred=true`, a deferred artifact is acceptable.

## Verification

Task-specific tests:

```bash
uv run pytest tests/test_npr_adapter.py -v
uv run pytest tests/test_cli_smoke.py -v
```

Full repository quality gate:

```bash
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
uv run pytest
```

Smoke CLI gate:

```bash
uv run aiforensics prepare --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline qwen_vl --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline assisted_qwen --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline npr --config configs/phase_ab_smoke.yaml
uv run aiforensics evaluate --config configs/phase_ab_smoke.yaml
uv run aiforensics report --config configs/phase_ab_smoke.yaml
```

Expected:

- full Ruff check passes,
- full Ruff format check passes,
- full pytest passes,
- `prepare` passes,
- CLIP smoke run completes,
- Qwen-VL smoke run defers,
- assisted-Qwen smoke run defers,
- NPR smoke run defers with zero external/model work,
- evaluate still reads completed prediction artifacts,
- report preserves current behavior,
- no Task 1-9 regression is introduced.

## Suggested Implementation Order

1. add `tests/test_npr_adapter.py` with disabled/config/repo/checkpoint pure tests,
2. create the `npr` package and public adapter skeleton,
3. implement official repo URL validation and deterministic repo path,
4. implement checkout verification/management behind one narrow helper,
5. implement checkpoint existence and optional SHA-256 validation,
6. implement evaluation-manifest selection and image integrity checks,
7. implement and test `npr_genimage_v1` preprocessing,
8. implement runtime input and score JSONL contracts,
9. implement the isolated NPR runtime subprocess,
10. implement score-to-`PredictionRecord` conversion,
11. validate and read back `predictions.jsonl`,
12. wire the NPR CLI branch and run artifacts,
13. add deferred/failed behavior tests,
14. add smoke CLI coverage,
15. run full Ruff, full pytest, and full smoke gate,
16. perform optional real CUDA verification only when the environment supports it.

## Acceptance Checklist

Task 10 is complete when:

- `NPRAdapter.name` is exactly `npr`,
- CLI routes `npr` to the real adapter,
- one NPR CLI invocation creates exactly one run directory,
- only the official NPR repository is accepted,
- completed real runs use an exact pinned commit,
- external NPR source is not copied into this repository,
- existing dirty external checkout is never overwritten or silently used,
- no destructive Git command is used to repair an external checkout,
- checkpoint path is explicit,
- configured checkpoint SHA-256 is enforced,
- checksum mismatch is always failed,
- smoke/disabled NPR performs no repo/checkpoint/model/network work,
- evaluation selection matches Tasks 8 and 9,
- no training manifest is required,
- missing/corrupt images fail clearly,
- `npr_genimage_v1` preprocessing is explicit and deterministic,
- ground-truth labels never enter the NPR runtime input,
- real NPR model code comes from the verified external checkout,
- real model/checkpoint load occurs once per runner process,
- official sigmoid score is mapped directly to `score_fake`,
- official strict `score_fake > 0.5` decision semantics are preserved,
- every selected sample produces exactly one `PredictionRecord`,
- NPR records use `model_name="npr"`,
- NPR records do not populate MLLM-only fields,
- predictions validate before write and after read-back,
- failed/deferred runs do not leave stale `predictions.jsonl`,
- environment limitations defer only when allowed,
- inference/runtime/data/integrity failures are not mislabeled as deferred,
- default tests require no network, clone, checkpoint, Torch, or CUDA,
- full `ruff check src tests` is green,
- full `ruff format --check src tests` is green,
- full pytest is green,
- full smoke gate is green.
