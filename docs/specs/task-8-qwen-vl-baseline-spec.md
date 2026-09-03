# Task 8 Spec: Qwen-VL Baseline

## Goal

Implement the second real Phase A/B baseline: `qwen_vl`.

Task 8 turns:

```bash
aiforensics run --baseline qwen_vl --config configs/phase_ab.yaml
```

from a placeholder into a working multimodal baseline using the configured
`Qwen/Qwen2.5-VL-3B-Instruct` model.

The baseline must:

- classify each evaluation image as `real` or `fake`,
- request one structured JSON response per image,
- preserve the exact raw model output,
- parse `label`, `confidence`, and `evidence` deterministically,
- map the parsed response into the shared `PredictionRecord` schema,
- mark unparseable model text with `label_pred="unknown"` rather than silently dropping it,
- cache raw model outputs when enabled,
- create the standard run artifacts,
- defer cleanly when the real Qwen runtime cannot be used in the current environment,
- keep all parser and smoke tests runnable without downloading Qwen weights.

Task 8 must not implement assisted Qwen, CLIP-conditioned prompting, NPR, report rendering,
dataset downloading, explanation scoring, RAG, few-shot retrieval, patch attribution, or forensic maps.

## Prerequisites

Task 8 depends on Tasks 1-7 being complete.

Before starting Task 8, verify:

```bash
uv run pytest
uv run pytest tests/test_clip_probe_smoke.py -v
```

Expected:

- all Task 1-7 tests pass,
- `RunResult` and `BaselineAdapter` exist,
- `create_run_dir()`, `write_environment()`, `write_status()`, and `cache_key()` are available,
- `PredictionRecord` already supports MLLM fields,
- `clip_probe` behavior remains stable,
- Task 7 scoped Ruff checks pass.

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
- `src/aiforensics/baselines/base.py`
- `src/aiforensics/cli/main.py`
- `src/aiforensics/config/models.py`
- `src/aiforensics/data/manifest.py`
- `src/aiforensics/schemas/predictions.py`
- `src/aiforensics/runs/artifacts.py`
- `src/aiforensics/cache/keys.py`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`

## Scope Boundary

Task 8 is the unassisted MLLM baseline.

The prompt may contain only:

- the current image,
- the fixed Task 8 forensic-classification instruction,
- the required response schema.

The prompt must not contain:

- `clip_probe` prediction,
- CLIP fake probability,
- training examples,
- retrieved examples,
- nearest neighbors,
- NPR outputs,
- external forensic maps,
- hidden ground-truth labels.

Those additions belong to later tasks, especially Task 9.

## Files To Create Or Modify

Required files:

```text
src/aiforensics/baselines/qwen_vl/__init__.py
src/aiforensics/baselines/qwen_vl/adapter.py
src/aiforensics/cli/main.py
pyproject.toml
uv.lock
tests/test_qwen_prompt_parsing.py
tests/test_cli_smoke.py
```

Modify `src/aiforensics/config/models.py` only if a small validation rule is required by this spec.

If `adapter.py` becomes difficult to navigate, it is acceptable to split pure prompt/parser logic into:

```text
src/aiforensics/baselines/qwen_vl/prompt.py
src/aiforensics/baselines/qwen_vl/parsing.py
```

Do not create abstractions for Task 9 before Task 9 actually needs them.

## Public Interface

Expose this import from `src/aiforensics/baselines/qwen_vl/__init__.py`:

```python
from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter
```

Implement:

```python
class QwenVLAdapter:
    name = "qwen_vl"

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

`qwen_vl` does not use a training seed in Task 8. `seed` may be ignored when `None`.

The adapter must not load Qwen dependencies at module import time.

## Prompt Contract

Task 8 supports exactly one prompt version:

```text
qwen_json_v1
```

The config value:

```python
config.baselines.qwen_vl.prompt_id
```

must select that prompt explicitly. An unsupported prompt id must fail the run with a clear reason;
do not silently substitute another prompt.

Recommended prompt content:

```text
You are an image-forensics classifier.

Classify the provided image as either "real" or "fake".

Definitions:
- "real": a natural camera photograph.
- "fake": an AI-generated or synthetic image.

Return exactly one JSON object and no Markdown or surrounding prose:
{
  "label": "real" or "fake",
  "confidence": a number from 0.0 to 1.0 representing confidence in the chosen label,
  "evidence": "a concise image-based reason for the decision"
}

Use only evidence visible in the image. Do not claim access to metadata, hidden detectors,
classifier scores, retrieval results, or tools that were not provided.
```

Requirements:

- prompt text is deterministic for the same `prompt_id`,
- no sample label or manifest ground truth is interpolated into the prompt,
- no CLIP result is interpolated into the prompt,
- the response request uses exactly the semantic fields `label`, `confidence`, and `evidence`,
- `confidence` means confidence in the selected label, not automatically probability of `fake`,
- prompt changes require a new `prompt_id`.

## Parser Contract

Implement a pure parser that can be unit-tested without Torch, Transformers, Qwen weights, GPU,
or network access.

Suggested interface:

```python
class QwenParseResult(BaseModel):
    label_pred: Literal["real", "fake", "unknown"]
    score_fake: float | None
    explanation: str
    parse_status: Literal["parsed", "recovered", "failed"]


def parse_qwen_output(raw_output: str) -> QwenParseResult:
    ...
```

### Strict Parse

First attempt:

```python
json.loads(raw_output.strip())
```

A strict parse is `parse_status="parsed"` only when:

- the complete non-whitespace output is one JSON object,
- `label` is exactly `real` or `fake`,
- `confidence` is numeric, finite, and in `[0.0, 1.0]`,
- booleans are not accepted as numeric confidence values,
- `evidence` is a non-empty string after trimming.

Additional JSON keys may be ignored. Required keys may not be missing.

### Bounded Recovery

If strict parsing fails, allow only deterministic bounded recovery.

Allowed recovery cases:

1. exactly one fenced JSON object, for example:

   ````text
   ```json
   {"label":"fake","confidence":0.9,"evidence":"..."}
   ```
   ````

2. surrounding prose containing exactly one recoverable top-level JSON object,
3. `label` differs only by surrounding whitespace or ASCII case, such as `" Fake "`.

Recovered output uses:

```text
parse_status = "recovered"
```

Do not implement broad heuristic repair such as:

- converting arbitrary Python dict syntax to JSON,
- inventing missing keys,
- converting percentages like `90` to `0.9`,
- guessing malformed numeric strings,
- mapping synonyms such as `synthetic`, `AI`, or `authentic` to labels,
- selecting one object when multiple competing JSON objects are present.

For surrounding-prose recovery, use a balanced-object parser or another structured method. Do not use
a greedy regular expression that can merge unrelated braces.

### Failed Parse

If no valid result can be produced:

```python
QwenParseResult(
    label_pred="unknown",
    score_fake=None,
    explanation="",
    parse_status="failed",
)
```

The exact model text is still preserved separately in `PredictionRecord.raw_output`.

An unparseable model response is a per-sample parse failure, not automatically a failed run.

### Confidence Mapping

The prompt defines `confidence` as confidence in the predicted label.

Map to the shared fake score as:

```python
if label == "fake":
    score_fake = confidence
else:
    score_fake = 1.0 - confidence
```

Examples:

```text
label=fake, confidence=0.90 -> score_fake=0.90
label=real, confidence=0.90 -> score_fake=0.10
```

Do not treat confidence in `real` as fake probability directly.

## Data Selection

Qwen-VL requires no training split in Task 8.

Evaluation records come from:

```python
config.datasets.tiny_genimage.dev_manifest
```

Additionally include these manifests only when the dataset is enabled and the file exists:

```python
config.datasets.genimage_unseen.manifest
config.datasets.synthbuster.manifest
```

Use:

```python
load_manifest(path, data_root=config.paths.data_root)
```

Requirements:

- the tiny dev manifest may be absent only if at least one enabled external evaluation manifest exists,
- an enabled external manifest that is missing should produce a warning and continue,
- disabled external datasets must be ignored even when their manifest file exists,
- no training manifest is required for Qwen-VL,
- if no evaluation records remain, fail the run,
- missing image files fail the run,
- checksum mismatch fails the run,
- preserve evaluation-record ordering,
- do not silently drop any sample.

## Real Qwen Inference Path

When:

```python
config.baselines.qwen_vl.enabled is True
```

run the model configured by:

```python
config.baselines.qwen_vl.model_id
```

The Phase A/B real config currently uses:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Use the Hugging Face Qwen2.5-VL implementation, with lazy imports inside the real inference path.

Expected model/processor APIs are based on:

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
```

and, when used by the selected Transformers flow:

```python
from qwen_vl_utils import process_vision_info
```

Requirements:

- load the model once per run, not once per sample,
- load the processor once per run,
- use `config.baselines.qwen_vl.model_id`, never a hardcoded checkpoint inside the adapter,
- put the model in evaluation mode,
- use inference/no-grad mode,
- run one deterministic prompt template for every sample,
- use `config.baselines.qwen_vl.max_new_tokens`,
- Task 8 Phase A/B requires `temperature == 0.0`,
- use greedy/deterministic generation (`do_sample=False`),
- decode only newly generated tokens, not the input prompt,
- preserve the decoded model text exactly as `raw_output`, except normal library-level decoding,
- parse the output only after generation has completed,
- do not expose the ground-truth label to the model.

If `temperature` is not `0.0`, fail early with a clear configuration error. Do not silently change it.

## Device Policy

Task 8 real Qwen inference is intended for GPU environments such as Colab or Kaggle.

Resolve `config.runtime.device` as follows:

```text
auto -> CUDA when available, otherwise environment unavailable
cuda -> use CUDA only when available
cpu  -> unsupported for the Task 8 real Qwen run
```

This policy avoids accidentally starting a multi-billion-parameter CPU run during smoke or local tests.

When the environment cannot support the requested real Qwen run:

- return `deferred` when `config.baselines.qwen_vl.allow_deferred is True`,
- return `failed` when `allow_deferred is False`.

Environment/setup examples:

- Torch unavailable,
- Transformers unavailable,
- `qwen_vl_utils` unavailable when required by the implementation,
- CUDA requested but unavailable,
- model weights cannot be resolved,
- processor cannot be loaded,
- model cannot be placed on the requested device because of environment/resource limitations.

Do not classify per-sample image or generation bugs as deferred environment failures.

## Optional Dependencies

Qwen dependencies must remain optional.

Add a dedicated optional dependency group in `pyproject.toml`, for example:

```toml
[project.optional-dependencies]
qwen = [
    "torch",
    "transformers",
    "accelerate",
    "qwen-vl-utils",
]
```

Choose mutually compatible package versions for Qwen2.5-VL and let `uv.lock` pin the resolved
environment.

Do not add Qwen, Transformers, Torch, Accelerate, or `qwen-vl-utils` to the default dependencies.

The following must work without the `qwen` extra:

```bash
uv run pytest tests/test_qwen_prompt_parsing.py -v
uv run pytest
```

## Raw Output Cache

When:

```python
config.baselines.qwen_vl.cache_outputs is True
```

cache only the decoded raw model output.

Cache directory:

```text
<config.paths.cache_root>/qwen_vl/raw_outputs/
```

Recommended cache file:

```text
<cache_key>.txt
```

Use `cache_key()` from Task 6.

The key must include all semantic inputs that can change the generated answer:

```python
{
    "baseline": "qwen_vl",
    "sample_checksum": sample_checksum,
    "model_id": config.baselines.qwen_vl.model_id,
    "prompt_id": config.baselines.qwen_vl.prompt_id,
    "temperature": str(config.baselines.qwen_vl.temperature),
    "max_new_tokens": str(config.baselines.qwen_vl.max_new_tokens),
    "output_cache_version": "qwen_vl_raw_v1",
}
```

If a manifest checksum is absent, compute the SHA-256 of the image bytes for the cache key.

Requirements:

- cache hits bypass model generation for that sample,
- cache hits still run through the current parser,
- changing image bytes, model id, prompt id, temperature, or max tokens changes the key,
- corrupt/unreadable cache entries are treated as misses and recomputed,
- cache writes happen only after a complete raw output was generated,
- do not cache parsed predictions as a replacement for `predictions.jsonl`,
- do not cache processed images or tensors,
- do not write Qwen cache files when `cache_outputs` is false.

## Prediction Records

Build one `PredictionRecord` for every evaluation sample.

For parsed/recovered responses:

```python
PredictionRecord(
    sample_id=record.sample_id,
    label_true=record.label,
    label_pred=parse_result.label_pred,
    score_fake=parse_result.score_fake,
    model_name="qwen_vl",
    source=record.source,
    run_id=run_id,
    dataset=record.dataset,
    split=record.split,
    path=record.path,
    checksum=record.checksum,
    prompt_id=config.baselines.qwen_vl.prompt_id,
    raw_output=raw_output,
    explanation=parse_result.explanation,
    parse_status=parse_result.parse_status,
)
```

For unparseable model text:

```text
label_pred = "unknown"
score_fake = null
explanation = ""
parse_status = "failed"
raw_output = exact decoded model output
```

Requirements:

- every evaluation sample produces exactly one prediction when inference itself succeeds,
- parser failure does not silently remove a sample,
- preserve `raw_output` even when parsing fails,
- `explanation` is the parsed `evidence` string for parsed/recovered outputs,
- do not use `not_applicable` for Qwen-VL,
- do not write duplicate `sample_id` predictions.

Before writing `predictions.jsonl`, validate the full in-memory list:

```python
validation = validate_predictions(
    predictions,
    manifest_sample_ids={record.sample_id for record in eval_records},
    require_mllm_fields=True,
)
```

If validation fails, fail the run and do not leave an invalid `predictions.jsonl` artifact.

After writing, load the file again with `load_predictions()` and validate it once more to protect
the serialization boundary.

## Per-Sample Failure Semantics

Distinguish model-output parse failure from inference failure.

### Parse failure

The model returned text successfully, but the text cannot be parsed.

Behavior:

```text
run continues
prediction is written
label_pred = unknown
score_fake = null
parse_status = failed
```

### Inference failure

Examples:

- image cannot be opened or processed,
- processor fails for the sample,
- tensor transfer fails after setup,
- `model.generate()` fails during sample inference,
- decoding fails.

Behavior:

```text
run = failed
exit code = 1
no partial predictions.jsonl
```

Task 8 should accumulate predictions in memory and write the file only after all samples have
completed and validation succeeds.

## CLI Contract

Update `aiforensics run --baseline qwen_vl` while preserving Task 7 behavior exactly.

These commands must remain valid:

```bash
aiforensics run --baseline clip_probe --config configs/phase_ab.yaml
aiforensics run --baseline qwen_vl --config configs/phase_ab.yaml
aiforensics run --baseline qwen_vl --config configs/phase_ab_smoke.yaml
```

Task 8 CLI behavior:

1. load config with `load_config(args.config)`,
2. preserve the existing `clip_probe` branch,
3. route `qwen_vl` to `QwenVLAdapter`,
4. preserve placeholder behavior for `npr` and `assisted_qwen`,
5. create exactly one Qwen run directory per CLI invocation,
6. copy the input config to `run_dir / "config.yaml"`,
7. write `environment.json`,
8. create/touch `logs.txt`,
9. invoke the adapter,
10. write `status.json` for completed, failed, or deferred results,
11. print one concise summary,
12. return `0` for completed or deferred,
13. return `1` for failed.

Suggested summary:

```text
[run] baseline=qwen_vl runs=1 completed=<0|1> failed=<0|1> deferred=<0|1> output_root=<output_root>
```

Because `QwenVLConfig` has no seed list, do not create multiple Task 8 runs for CLIP seeds.

## Run Directory And Artifact Contract

Create the run directory with:

```python
create_run_dir(config.paths.output_root, "qwen_vl")
```

The directory name is the `run_id`.

Completed run:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  status.json
  predictions.jsonl
  logs.txt
```

Failed/deferred run:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  status.json
  logs.txt
```

Do not write metrics inside `run`. Metrics belong to `aiforensics evaluate`.

Do not write report Markdown in Task 8.

## Failure And Deferred Behavior

Use `RunStatus` and `write_status()` for every CLI run.

Completed examples:

- all samples completed inference,
- some responses parsed normally,
- some responses required bounded parser recovery,
- some model responses were unparseable and were recorded as `unknown`,
- all prediction records validate.

Failed examples:

- no evaluation manifest is available,
- manifest validation fails,
- image file is missing,
- image checksum does not match,
- unsupported `prompt_id`,
- non-zero Task 8 temperature,
- per-sample image processing fails,
- model generation fails after setup,
- prediction validation fails,
- prediction writing or read-back validation fails,
- a defer-eligible environment problem occurs while `allow_deferred` is false.

Deferred examples when `allow_deferred` is true:

- `qwen_vl.enabled` is false,
- Qwen optional dependencies are unavailable,
- GPU is unavailable for the requested real run,
- model weights cannot be resolved,
- processor/model setup cannot complete because of the current environment.

`started_at` and `ended_at` must be UTC ISO-8601 timestamps accepted by `RunStatus`.

## Logging

Keep logs concise and useful.

Include:

- baseline and model id,
- prompt id,
- resolved device,
- number of evaluation samples,
- cache hit/miss counts when cache is enabled,
- parse-status counts at completion,
- clear failed/deferred reason.

Do not log:

- model weights,
- full tensors,
- secrets or tokens,
- every raw model output by default.

Raw output already belongs in `predictions.jsonl` and raw-output cache files.

## Tests

Create `tests/test_qwen_prompt_parsing.py`.

All tests in this file must run without installing the Qwen optional dependency group and without
network access.

Minimum tests:

1. `QwenVLAdapter.name == "qwen_vl"`.
2. `qwen_json_v1` prompt is deterministic.
3. prompt requests `label`, `confidence`, and `evidence`.
4. prompt does not contain CLIP prediction or fake probability fields.
5. unsupported prompt id fails clearly.
6. valid compact JSON parses with `parse_status="parsed"`.
7. valid pretty JSON parses with `parse_status="parsed"`.
8. `label=fake, confidence=0.9` maps to `score_fake=0.9`.
9. `label=real, confidence=0.9` maps to `score_fake=0.1`.
10. fenced JSON is recovered with `parse_status="recovered"`.
11. one JSON object surrounded by prose is recovered.
12. label case/whitespace-only normalization is `recovered`.
13. malformed JSON returns `unknown`, `None`, empty explanation, and `parse_status="failed"`.
14. missing `label` fails parsing.
15. missing `confidence` fails parsing.
16. missing/empty `evidence` fails parsing.
17. unknown label such as `synthetic` fails parsing.
18. confidence below `0.0` fails parsing.
19. confidence above `1.0` fails parsing.
20. boolean confidence fails parsing.
21. multiple competing JSON objects fail recovery rather than selecting one silently.
22. parser never imports Torch, Transformers, or Qwen utilities.
23. Qwen disabled in a temporary config creates a deferred run artifact and exits `0`.
24. missing Qwen dependencies produce deferred when `allow_deferred=true`.
25. the same setup problem produces failed when `allow_deferred=false`.
26. model/processor setup failure is deferred when allowed.
27. per-sample generation failure is failed, not deferred.
28. a parse failure still produces one valid `PredictionRecord` with `label_pred="unknown"`.
29. prediction validation failure does not leave `predictions.jsonl`.
30. disabled optional datasets are ignored even if their manifests exist.
31. raw-output cache hit bypasses model generation and still runs the parser.
32. cache key changes when image checksum changes.
33. cache key changes when `prompt_id` changes.
34. cache key changes when `model_id`, `temperature`, or `max_new_tokens` changes.
35. corrupt cache entry is recomputed.
36. model/processor are loaded once per run in mocked inference.
37. raw model text is preserved in the prediction record.

Use `tmp_path` for every generated config, output root, cache root, run directory, and cache file.

Tests must not create artifacts under repository `outputs/`.

## Mocking Rules

Real Qwen integration tests are optional and must not be part of the default test suite.

Default tests should mock the narrow external boundary rather than recreate a full fake Transformers
framework.

Preferred boundaries to isolate:

```text
load model/processor
generate raw text for one image
```

Keep pure functions for:

```text
prompt selection
output parsing
cache-key composition
PredictionRecord construction
```

This keeps parser/cache tests small and prevents brittle mocks of internal Torch tensor behavior.

## Smoke Configuration Expectations

Do not enable Qwen in `configs/phase_ab_smoke.yaml`.

The existing smoke config intentionally has:

```yaml
qwen_vl:
  enabled: false
  model_id: Qwen/Qwen2.5-VL-3B-Instruct
  prompt_id: qwen_json_v1
  temperature: 0.0
  max_new_tokens: 128
  cache_outputs: false
  allow_deferred: true
```

Running:

```bash
uv run aiforensics run --baseline qwen_vl --config configs/phase_ab_smoke.yaml
```

must:

- not import Qwen model libraries,
- not download weights,
- not require GPU,
- create one deferred run artifact,
- return exit code `0`.

## Real Environment Verification

Real model inference is not required for the default CI/smoke gate because it requires optional
dependencies and model weights.

When a suitable Colab/Kaggle/GPU environment is available, install the optional Qwen dependency
group and run:

```bash
uv run --extra qwen aiforensics run --baseline qwen_vl --config configs/phase_ab.yaml
```

Expected real-run behavior:

- model and processor load once,
- Qwen receives only image + Task 8 prompt,
- every evaluation sample produces one prediction unless inference fails,
- completed predictions contain MLLM fields,
- output validates through the shared prediction schema,
- repeated execution can reuse raw-output cache when enabled.

If the environment cannot support Qwen and `allow_deferred=true`, a deferred artifact is an
acceptable result.

## Ruff Scope

Keep Task 8 lint/format changes scoped to Task 8 files plus small directly required edits.

Run:

```bash
uv run --extra dev ruff check \
  src/aiforensics/baselines/qwen_vl \
  src/aiforensics/cli/main.py \
  src/aiforensics/config/models.py \
  tests/test_qwen_prompt_parsing.py \
  tests/test_cli_smoke.py

uv run --extra dev ruff format --check \
  src/aiforensics/baselines/qwen_vl \
  src/aiforensics/cli/main.py \
  src/aiforensics/config/models.py \
  tests/test_qwen_prompt_parsing.py \
  tests/test_cli_smoke.py
```

If `src/aiforensics/config/models.py` or `tests/test_cli_smoke.py` is untouched, it may be omitted
from the scoped command.

Do not mass-format unrelated Task 1-7 files as part of Task 8.

## Verification

Task-specific parser/unit verification:

```bash
uv run pytest tests/test_qwen_prompt_parsing.py -v
```

Regression verification:

```bash
uv run pytest
```

Smoke CLI verification:

```bash
uv run aiforensics prepare --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline qwen_vl --config configs/phase_ab_smoke.yaml
uv run aiforensics evaluate --config configs/phase_ab_smoke.yaml
uv run aiforensics report --config configs/phase_ab_smoke.yaml
```

Expected:

- `prepare` passes,
- existing CLIP smoke run still completes,
- Qwen smoke run is deferred without importing/downloading Qwen,
- `evaluate` still evaluates completed prediction artifacts,
- `report` preserves its current behavior,
- all existing Task 1-7 tests remain green,
- no assisted-Qwen or NPR implementation is added.

## Implementation Order

Recommended implementation order for a coding agent:

1. add the `qwen_vl` package and public adapter skeleton,
2. implement `qwen_json_v1` prompt selection,
3. implement and fully test the pure parser,
4. implement evaluation-manifest loading and record validation,
5. implement raw-output cache key/read/write behavior,
6. isolate lazy Qwen model setup behind a small helper,
7. implement one-image generation behind a small helper,
8. map parse results to `PredictionRecord`,
9. validate all predictions before writing,
10. wire the Qwen CLI branch and run artifacts,
11. add deferred/failed behavior tests,
12. add cache/inference boundary tests,
13. run scoped Ruff and the full regression suite,
14. run the smoke CLI gate.

## Done Criteria

Task 8 is complete when:

- `QwenVLAdapter.name` is exactly `qwen_vl`,
- `qwen_json_v1` is explicit and deterministic,
- the prompt asks for `label`, `confidence`, and `evidence`,
- the prompt contains no CLIP/NPR/ground-truth assistance,
- parser supports `parsed`, bounded `recovered`, and `failed` states,
- confidence maps correctly to `score_fake`, including `real -> 1 - confidence`,
- failed parse produces `unknown` rather than dropping the sample,
- real Qwen dependencies are lazy and optional,
- Qwen model/processor load once per run,
- Task 8 generation is deterministic with temperature `0.0`,
- GPU/setup limitations produce deferred or failed according to `allow_deferred`,
- per-sample inference failures are failed rather than deferred,
- raw outputs are cached with semantic Task 6 cache keys when enabled,
- cache hits preserve ordering and still pass through the parser,
- one Qwen run directory is created per CLI invocation,
- completed runs write valid MLLM `predictions.jsonl`,
- failed/deferred runs do not leave partial prediction artifacts,
- Qwen smoke config remains model-download-free and GPU-free,
- `aiforensics run --baseline qwen_vl --config configs/phase_ab_smoke.yaml` returns `0` with deferred status,
- all Task 8 tests pass,
- all existing Task 1-7 tests pass,
- scoped Ruff checks pass,
- assisted Qwen, NPR, report rendering, dataset downloading, explanation scoring, RAG, and patch attribution remain out of scope.
