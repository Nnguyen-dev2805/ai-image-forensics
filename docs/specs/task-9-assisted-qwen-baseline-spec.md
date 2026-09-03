# Task 9 Spec: Assisted Qwen Baseline

## Goal

Implement the third real Phase A/B baseline: `assisted_qwen`.

Task 9 turns:

```bash
aiforensics run --baseline assisted_qwen --config configs/phase_ab.yaml
```

from a placeholder into a working classifier-assisted multimodal baseline using the configured
Qwen model plus `clip_probe` predictions as assistant input.

The baseline must:

- classify each evaluation image as `real` or `fake`,
- include the CLIP probe classifier decision and fake probability in the prompt,
- request one structured JSON response per image,
- preserve the exact raw model output,
- reuse the deterministic Qwen parser from Task 8,
- map every sample into the shared `PredictionRecord` schema,
- mark unparseable model text with `label_pred="unknown"` rather than dropping it,
- cache raw model outputs when enabled,
- create the standard run artifacts,
- defer cleanly when the real Qwen runtime cannot be used in the current environment,
- fail clearly when required CLIP assistant scores are missing,
- keep all prompt, lookup, parser, and smoke tests runnable without downloading Qwen weights.

Task 9 must not implement NPR, trained fusion, score calibration, report rendering, dataset
downloading, explanation scoring, RAG, few-shot retrieval, patch attribution, forensic maps, or
new evaluation metrics.

## Prerequisites

Task 9 depends on Tasks 1-8 being complete.

Before starting Task 9, verify:

```bash
uv run pytest
uv run pytest tests/test_qwen_prompt_parsing.py tests/test_cli_smoke.py -v
```

Expected:

- all Task 1-8 tests pass,
- `RunResult` and `BaselineAdapter` exist,
- `create_run_dir()`, `write_environment()`, `write_status()`, and `cache_key()` are available,
- `PredictionRecord` supports MLLM fields,
- `clip_probe` writes valid `predictions.jsonl`,
- `qwen_vl` parser, prompt, deferred behavior, and CLI behavior remain stable,
- Task 8 scoped Ruff checks pass.

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
- `src/aiforensics/baselines/base.py`
- `src/aiforensics/baselines/clip_probe/adapter.py`
- `src/aiforensics/baselines/qwen_vl/adapter.py`
- `src/aiforensics/baselines/qwen_vl/parsing.py`
- `src/aiforensics/baselines/qwen_vl/prompt.py`
- `src/aiforensics/cli/main.py`
- `src/aiforensics/config/models.py`
- `src/aiforensics/data/manifest.py`
- `src/aiforensics/schemas/predictions.py`
- `src/aiforensics/runs/artifacts.py`
- `src/aiforensics/cache/keys.py`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`

## Scope Boundary

Task 9 is a prompt-assisted MLLM baseline.

The prompt may contain only:

- the current image,
- the fixed assisted-Qwen forensic-classification instruction,
- `classifier_pred` derived from CLIP probe predictions,
- `fake_probability` derived from CLIP probe predictions,
- the required response schema.

The prompt must not contain:

- ground-truth labels,
- CLIP embeddings,
- CLIP training data,
- CLIP model internals,
- retrieved examples,
- nearest neighbors,
- NPR outputs,
- external forensic maps,
- patch attributions,
- hidden detector metadata,
- metric results from evaluation.

This task tests whether a simple classifier signal improves Qwen behavior. It is not a fusion model.

## Files To Create Or Modify

Required files:

```text
src/aiforensics/baselines/assisted_qwen/__init__.py
src/aiforensics/baselines/assisted_qwen/adapter.py
src/aiforensics/baselines/assisted_qwen/prompt.py
src/aiforensics/cli/main.py
tests/test_assisted_qwen_prompt.py
tests/test_cli_smoke.py
```

Optional refactor files, only if they reduce duplication between `qwen_vl` and `assisted_qwen`:

```text
src/aiforensics/baselines/qwen_vl/runtime.py
src/aiforensics/baselines/qwen_vl/cache.py
```

Do not modify prediction schema, metric computation, report generation, NPR code, or dataset
download behavior in Task 9.

Modify `src/aiforensics/config/models.py` only if a small validation rule is needed for
`AssistedQwenConfig`.

Do not add new default dependencies. Reuse the existing `qwen` optional dependency group from
Task 8.

## Public Interface

Expose this import from `src/aiforensics/baselines/assisted_qwen/__init__.py`:

```python
from aiforensics.baselines.assisted_qwen.adapter import AssistedQwenAdapter
```

Implement:

```python
class AssistedQwenAdapter:
    name = "assisted_qwen"

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

`assisted_qwen` does not use a training seed in Task 9. `seed` may be ignored when `None`.

The adapter must not load Torch, Transformers, Qwen weights, or `qwen_vl_utils` at module import
time.

## Config Contract

Use the existing config fields:

```python
config.baselines.assisted_qwen.enabled
config.baselines.assisted_qwen.base_model_id
config.baselines.assisted_qwen.prompt_id
config.baselines.assisted_qwen.assistant_source
config.baselines.assisted_qwen.include_classifier_pred
config.baselines.assisted_qwen.include_fake_probability
config.baselines.assisted_qwen.temperature
config.baselines.assisted_qwen.max_new_tokens
config.baselines.assisted_qwen.cache_outputs
config.baselines.assisted_qwen.allow_deferred
```

Task 9 supports only:

```text
assistant_source = "clip_probe"
include_classifier_pred = true
include_fake_probability = true
temperature = 0.0
prompt_id = "assisted_qwen_json_v1"
```

If any of these values are unsupported, fail the run with a clear configuration error. Do not
silently substitute defaults.

## Assisted Prompt Contract

Task 9 supports exactly one prompt version:

```text
assisted_qwen_json_v1
```

Implement a pure prompt builder:

```python
def get_assisted_prompt(
    prompt_id: str,
    *,
    classifier_pred: str,
    fake_probability: float,
) -> str:
    ...
```

`classifier_pred` must be exactly `real` or `fake`.

`fake_probability` must be numeric, finite, and in `[0.0, 1.0]`.

The prompt must include the literal field names:

```text
classifier_pred
fake_probability
label
confidence
evidence
```

Recommended prompt content:

```text
You are an image-forensics classifier.

Classify the provided image as either "real" or "fake".

Definitions:
- "real": a natural camera photograph.
- "fake": an AI-generated or synthetic image.

You are also given an auxiliary classifier signal:
{
  "classifier_pred": "<real or fake>",
  "fake_probability": <number from 0.0 to 1.0>
}

Use the auxiliary classifier signal as supporting evidence, but make the final decision from the
image and the signal together. The auxiliary classifier may be wrong.

Return exactly one JSON object and no Markdown or surrounding prose:
{
  "label": "real" or "fake",
  "confidence": a number from 0.0 to 1.0 representing confidence in the chosen label,
  "evidence": "a concise image-based reason for the decision"
}

Use only evidence visible in the image plus the provided auxiliary classifier signal. Do not claim
access to metadata, hidden detectors, retrieval results, tools, training labels, or scores that were
not provided.
```

Requirements:

- prompt text is deterministic for the same inputs,
- prompt changes require a new `prompt_id`,
- no sample ground truth is interpolated into the prompt,
- no CLIP embeddings or CLIP training examples are interpolated into the prompt,
- no Qwen-VL baseline output is interpolated into the prompt,
- no NPR output is interpolated into the prompt,
- `fake_probability` is rendered with a stable text representation,
- the response request uses exactly the semantic output fields `label`, `confidence`, and
  `evidence`,
- `confidence` means confidence in the selected label, not automatically probability of `fake`.

Use this stable fake-probability text format:

```python
fake_probability_text = format(fake_probability, ".12g")
```

Use the same formatted value in the prompt and the raw-output cache key.

## Parser Contract

Reuse the Task 8 parser:

```python
from aiforensics.baselines.qwen_vl.parsing import QwenParseResult, parse_qwen_output
```

Do not implement a second parser for `assisted_qwen`.

For parsed or recovered responses:

```python
label=fake, confidence=0.90 -> score_fake=0.90
label=real, confidence=0.90 -> score_fake=0.10
```

For unparseable model text, write one prediction with:

```python
label_pred = "unknown"
score_fake = None
explanation = ""
parse_status = "failed"
```

An unparseable model response is a per-sample parse failure, not automatically a failed run.

## Data Selection

`assisted_qwen` uses the same evaluation records as `qwen_vl`.

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

- the tiny dev manifest may be absent only if at least one enabled external evaluation manifest
  exists,
- an enabled external manifest that is missing should produce a warning and continue,
- disabled external datasets must be ignored even when their manifest file exists,
- no training manifest is required for `assisted_qwen`,
- if no evaluation records remain, fail the run,
- missing image files fail the run,
- checksum mismatch fails the run,
- preserve evaluation-record ordering,
- do not silently drop any sample.

## CLIP Assistant Input Discovery

Task 9 consumes completed `clip_probe` predictions from:

```python
config.paths.output_root
```

Do not add a new config path in Task 9 unless the user explicitly asks for it.

Discovery rules:

1. Recursively find `predictions.jsonl` files under `config.paths.output_root`.
2. Keep only files whose parent directory has `status.json` with:

   ```json
   {
     "baseline": "clip_probe",
     "status": "completed"
   }
   ```

3. Load each kept file with `load_predictions(path)`.
4. Validate each file with `validate_predictions(records, require_mllm_fields=True)`.
5. Keep only records with `model_name == "clip_probe"`.
6. Ignore failed or deferred CLIP run directories.
7. Sort prediction file paths lexicographically before processing for deterministic behavior.

If no completed CLIP prediction file is available, fail the run with:

```text
No completed clip_probe predictions found for assisted_qwen
```

This is a data dependency failure, not a deferred Qwen environment problem.

## CLIP Assistant Input Aggregation

Because `clip_probe` may run multiple seeds and `assisted_qwen` has no seed list, Task 9 must
aggregate CLIP predictions deterministically into one assistant input per `sample_id`.

Suggested typed helper:

```python
from pydantic import BaseModel, Field


class AssistedInput(BaseModel):
    sample_id: str
    classifier_pred: Literal["real", "fake"]
    fake_probability: float = Field(ge=0.0, le=1.0)
    source_prediction_files: tuple[Path, ...]
```

Aggregation rules:

1. For each evaluation `sample_id`, collect all completed `clip_probe` records with that
   `sample_id`.
2. Each collected CLIP record must have `score_fake` as a finite float in `[0.0, 1.0]`.
3. Each collected CLIP record must have `label_pred` equal to `real` or `fake`.
4. If an evaluation sample has no CLIP record, fail clearly:

   ```text
   Missing clip_probe assistant prediction for sample_id=<sample_id>
   ```

5. Compute:

   ```python
   fake_probability = sum(score_fake_values) / len(score_fake_values)
   classifier_pred = "fake" if fake_probability >= 0.5 else "real"
   ```

6. Extra CLIP records whose `sample_id` is not in the evaluation manifest may be ignored with a
   warning.
7. Do not use ground-truth labels when deriving `classifier_pred`.
8. Do not silently drop evaluation samples.

## Real Qwen Inference Path

Use the Hugging Face Qwen2.5-VL implementation as in Task 8.

The configured model id is:

```python
config.baselines.assisted_qwen.base_model_id
```

Expected APIs:

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
```

Requirements:

- load the model once per run, not once per sample,
- load the processor once per run,
- use `config.baselines.assisted_qwen.base_model_id`, never a hardcoded checkpoint inside the
  adapter,
- put the model in evaluation mode,
- use inference/no-grad mode,
- run one deterministic assisted prompt template for every sample,
- use `config.baselines.assisted_qwen.max_new_tokens`,
- Task 9 Phase A/B requires `temperature == 0.0`,
- use greedy/deterministic generation with `do_sample=False`,
- decode only newly generated tokens, not the input prompt,
- preserve the decoded model text exactly as `raw_output`, except normal library-level decoding,
- parse the output only after generation has completed,
- do not expose the ground-truth label to the model.

If shared Qwen runtime logic is extracted from Task 8, keep Task 8 behavior byte-for-byte equivalent
from the CLI user's perspective.

## Device Policy

Use the same device policy as Task 8.

Resolve `config.runtime.device` as follows:

```text
auto -> CUDA when available, otherwise environment unavailable
cuda -> use CUDA only when available
cpu  -> unsupported for the Task 9 real Qwen run
```

When the environment cannot support the requested real Qwen run:

- return `deferred` when `config.baselines.assisted_qwen.allow_deferred is True`,
- return `failed` when `allow_deferred is False`.

Environment/setup examples:

- Torch unavailable,
- Transformers unavailable,
- `qwen_vl_utils` unavailable,
- CUDA requested but unavailable,
- model weights cannot be resolved,
- processor cannot be loaded,
- model cannot be placed on the requested device because of environment/resource limitations.

Do not classify missing CLIP predictions, missing images, invalid manifests, per-sample image
processing errors, or generation errors after setup as deferred environment failures.

## Optional Dependencies

Reuse the Task 8 optional dependency group:

```toml
[project.optional-dependencies]
qwen = [
    "torch",
    "transformers",
    "accelerate",
    "qwen-vl-utils",
]
```

Do not add Qwen, Transformers, Torch, Accelerate, or `qwen-vl-utils` to default dependencies.

The following must work without the `qwen` extra:

```bash
uv run pytest tests/test_assisted_qwen_prompt.py -v
uv run pytest
```

## Raw Output Cache

When:

```python
config.baselines.assisted_qwen.cache_outputs is True
```

cache only the decoded raw model output.

Cache directory:

```text
<config.paths.cache_root>/assisted_qwen/raw_outputs/
```

Recommended cache file:

```text
<cache_key>.json
```

Use `cache_key()` from Task 6.

The key must include all semantic inputs that can change the generated answer:

```python
{
    "baseline": "assisted_qwen",
    "sample_checksum": sample_checksum,
    "base_model_id": config.baselines.assisted_qwen.base_model_id,
    "prompt_id": config.baselines.assisted_qwen.prompt_id,
    "assistant_source": config.baselines.assisted_qwen.assistant_source,
    "classifier_pred": classifier_pred,
    "fake_probability": fake_probability_text,
    "temperature": str(config.baselines.assisted_qwen.temperature),
    "max_new_tokens": str(config.baselines.assisted_qwen.max_new_tokens),
    "output_cache_version": "assisted_qwen_raw_v1",
}
```

If a manifest checksum is absent, compute the SHA-256 of the image bytes for the cache key.

Requirements:

- cache hits bypass model generation for that sample,
- cache hits still run through the current parser,
- changing image bytes, model id, prompt id, classifier prediction, fake probability, temperature,
  or max tokens changes the key,
- corrupt/unreadable cache entries are treated as misses and recomputed,
- cache writes happen only after a complete raw output was generated,
- do not cache parsed predictions as a replacement for `predictions.jsonl`,
- do not cache processed images or tensors,
- do not write assisted-Qwen cache files when `cache_outputs` is false.

Recommended cache payload:

```json
{
  "raw_output": "{\"label\":\"fake\",\"confidence\":0.91,\"evidence\":\"...\"}"
}
```

Treat the cache entry as corrupt unless the decoded JSON is a dict and `raw_output` is a string.

## Prediction Records

Build one `PredictionRecord` for every evaluation sample.

For parsed/recovered responses:

```python
PredictionRecord(
    sample_id=record.sample_id,
    label_true=record.label,
    label_pred=parse_result.label_pred,
    score_fake=parse_result.score_fake,
    model_name="assisted_qwen",
    source=record.source,
    run_id=run_id,
    dataset=record.dataset,
    split=record.split,
    path=record.path,
    checksum=record.checksum,
    prompt_id=config.baselines.assisted_qwen.prompt_id,
    raw_output=raw_output,
    explanation=parse_result.explanation,
    parse_status=parse_result.parse_status,
)
```

For failed parse responses:

```python
PredictionRecord(
    sample_id=record.sample_id,
    label_true=record.label,
    label_pred="unknown",
    score_fake=None,
    model_name="assisted_qwen",
    source=record.source,
    run_id=run_id,
    dataset=record.dataset,
    split=record.split,
    path=record.path,
    checksum=record.checksum,
    prompt_id=config.baselines.assisted_qwen.prompt_id,
    raw_output=raw_output,
    explanation="",
    parse_status="failed",
)
```

Requirements:

- exactly one prediction per evaluation sample,
- preserve evaluation-record ordering,
- no duplicate `sample_id` records,
- `model_name` is exactly `assisted_qwen`,
- `prompt_id` is exactly `config.baselines.assisted_qwen.prompt_id`,
- `raw_output` is the exact decoded model text or cached decoded model text,
- do not use `not_applicable` for `assisted_qwen`,
- validate predictions in memory before writing,
- write `predictions.jsonl`,
- read `predictions.jsonl` back and validate again after writing,
- delete `predictions.jsonl` if the run ends as failed or deferred.

Do not add non-schema assistant-input fields to `PredictionRecord` in Task 9. Record assistant input
source files and aggregate counts in `logs.txt`.

## CLI Contract

Update `aiforensics run --baseline assisted_qwen` while preserving Task 7 and Task 8 behavior.

These commands must remain valid:

```bash
aiforensics run --baseline clip_probe --config configs/phase_ab.yaml
aiforensics run --baseline qwen_vl --config configs/phase_ab.yaml
aiforensics run --baseline assisted_qwen --config configs/phase_ab.yaml
aiforensics run --baseline assisted_qwen --config configs/phase_ab_smoke.yaml
```

Task 9 CLI behavior:

1. load config with `load_config(args.config)`,
2. preserve existing `clip_probe` behavior exactly,
3. preserve existing `qwen_vl` behavior exactly,
4. route `assisted_qwen` to `AssistedQwenAdapter`,
5. preserve placeholder behavior for `npr`,
6. create exactly one assisted-Qwen run directory per CLI invocation,
7. copy the input config file to `run_dir / "config.yaml"`,
8. write `run_dir / "environment.json"` using `write_environment()`,
9. call `AssistedQwenAdapter.run(...)`,
10. write `run_dir / "status.json"` using `write_status()` for completed, failed, and deferred,
11. write `run_dir / "logs.txt"` with concise human-readable run notes,
12. print a concise summary,
13. return `0` only when the assisted-Qwen run is `completed` or `deferred`,
14. return `1` when the assisted-Qwen run is `failed`.

Suggested summary:

```text
[run] baseline=assisted_qwen runs=1 completed=<0|1> failed=<0|1> deferred=<0|1> output_root=<output_root>
```

Because `AssistedQwenConfig` has no seed list, do not create multiple Task 9 runs for CLIP seeds.

## Run Directory And Artifact Contract

Create the run directory with:

```python
create_run_dir(config.paths.output_root, "assisted_qwen")
```

The returned directory name is the `run_id`.

Completed assisted-Qwen run artifact:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  status.json
  predictions.jsonl
  logs.txt
```

Failed or deferred assisted-Qwen run artifact:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  status.json
  logs.txt
```

Do not write metrics inside `run`. Metrics belong to `aiforensics evaluate`.

Do not write report Markdown in Task 9.

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
- no completed CLIP probe predictions are available,
- required CLIP assistant score is missing for an evaluation sample,
- CLIP assistant score is invalid or non-finite,
- image file is missing,
- image checksum does not match,
- unsupported `prompt_id`,
- unsupported `assistant_source`,
- `include_classifier_pred` is false,
- `include_fake_probability` is false,
- non-zero Task 9 temperature,
- per-sample image processing fails,
- model generation fails after setup,
- prediction validation fails,
- prediction writing or read-back validation fails,
- a defer-eligible environment problem occurs while `allow_deferred` is false.

Deferred examples when `allow_deferred` is true:

- `assisted_qwen.enabled` is false,
- Qwen optional dependencies are unavailable,
- GPU is unavailable for the requested real run,
- model weights cannot be resolved,
- processor/model setup cannot complete because of the current environment.

`started_at` and `ended_at` must be UTC ISO-8601 timestamps accepted by `RunStatus`.

## Logging

Keep logs concise and useful.

Include:

- baseline and base model id,
- prompt id,
- resolved device,
- number of evaluation samples,
- number of CLIP prediction files used,
- CLIP prediction file paths or run ids,
- number of assistant inputs built,
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

Create `tests/test_assisted_qwen_prompt.py`.

All tests in this file must run without installing the Qwen optional dependency group and without
network access.

Minimum tests:

1. `AssistedQwenAdapter.name == "assisted_qwen"`.
2. `assisted_qwen_json_v1` prompt is deterministic for identical inputs.
3. prompt includes `classifier_pred`.
4. prompt includes `fake_probability`.
5. prompt requests output fields `label`, `confidence`, and `evidence`.
6. prompt does not contain ground truth, CLIP embeddings, NPR, retrieval, or patch attribution.
7. unsupported prompt id fails clearly.
8. invalid `classifier_pred` fails clearly.
9. out-of-range `fake_probability` fails clearly.
10. non-finite `fake_probability` fails clearly.
11. fake probability text formatting is stable.
12. CLIP assistant discovery ignores failed CLIP runs.
13. CLIP assistant discovery ignores deferred CLIP runs.
14. CLIP assistant discovery loads completed CLIP `predictions.jsonl` files.
15. missing completed CLIP predictions fail clearly.
16. missing CLIP prediction for one evaluation sample fails clearly.
17. invalid CLIP `score_fake=None` fails clearly.
18. invalid CLIP `label_pred="unknown"` fails clearly.
19. multiple CLIP seed predictions aggregate by mean `score_fake`.
20. aggregate `fake_probability >= 0.5` maps to `classifier_pred="fake"`.
21. aggregate `fake_probability < 0.5` maps to `classifier_pred="real"`.
22. extra CLIP records not in evaluation manifests do not create extra assisted predictions.
23. disabled optional datasets are ignored even if their manifests exist.
24. missing tiny dev is allowed when an enabled external evaluation manifest exists.
25. invalid existing manifest fails rather than being silently skipped.
26. assisted Qwen disabled creates a deferred run artifact and exits `0`.
27. missing Qwen dependencies produce deferred when `allow_deferred=true`.
28. the same setup problem produces failed when `allow_deferred=false`.
29. model/processor setup failure is deferred when allowed.
30. per-sample generation failure is failed, not deferred.
31. parse failure still produces one valid `PredictionRecord` with `label_pred="unknown"`.
32. prediction validation failure does not leave `predictions.jsonl`.
33. raw-output cache hit bypasses model generation and still runs the parser.
34. cache key changes when image checksum changes.
35. cache key changes when `prompt_id` changes.
36. cache key changes when model id, classifier prediction, fake probability, temperature, or
    `max_new_tokens` changes.
37. corrupt cache entry is recomputed.
38. model/processor are loaded once per run in mocked inference.
39. raw model text is preserved in the prediction record.
40. CLI `run --baseline assisted_qwen` creates one run directory, not one per CLIP seed.
41. smoke config `run --baseline assisted_qwen` returns `0` with deferred status and does not import
    Qwen model libraries.

Use `tmp_path` for every generated config, output root, cache root, run directory, CLIP prediction
file, and assisted cache file.

Tests must not create artifacts under repository `outputs/`.

## Mocking Rules

Real Qwen integration tests are optional and must not be part of the default test suite.

Default tests should mock these narrow boundaries:

```text
load model/processor
generate raw text for one image
```

Keep pure functions for:

```text
assisted prompt selection
assistant input discovery
assistant input aggregation
cache-key composition
PredictionRecord construction
```

This keeps tests small and prevents brittle mocks of internal Torch tensor behavior.

## Smoke Configuration Expectations

Do not enable assisted Qwen in `configs/phase_ab_smoke.yaml`.

The existing smoke config intentionally has:

```yaml
assisted_qwen:
  enabled: false
  base_model_id: Qwen/Qwen2.5-VL-3B-Instruct
  prompt_id: assisted_qwen_json_v1
  assistant_source: clip_probe
  include_classifier_pred: true
  include_fake_probability: true
  temperature: 0.0
  max_new_tokens: 128
  cache_outputs: false
  allow_deferred: true
```

This command must return `0` and create a deferred run artifact:

```bash
uv run aiforensics run --baseline assisted_qwen --config configs/phase_ab_smoke.yaml
```

Smoke assisted-Qwen run must:

- not import Qwen model libraries,
- not download model weights,
- not require GPU,
- not require completed CLIP predictions when `assisted_qwen.enabled` is false.

## Real Run Expectations

When a suitable Colab/Kaggle/GPU environment is available:

```bash
uv run aiforensics prepare --config configs/phase_ab.yaml
uv run --extra clip aiforensics run --baseline clip_probe --config configs/phase_ab.yaml
uv run --extra qwen aiforensics run --baseline assisted_qwen --config configs/phase_ab.yaml
```

Expected:

- CLIP probe completed predictions exist under `config.paths.output_root`,
- assisted Qwen receives image + assisted prompt only,
- no ground truth reaches the prompt,
- every evaluation sample has a CLIP assistant input,
- `predictions.jsonl` contains one `assisted_qwen` record per evaluation sample,
- `evaluate` can read assisted-Qwen predictions without special casing.

If the environment cannot support Qwen and `allow_deferred=true`, a deferred artifact is an
acceptable Task 9 real-run outcome.

## Suggested Implementation Order

1. create `tests/test_assisted_qwen_prompt.py` with pure prompt tests,
2. implement `src/aiforensics/baselines/assisted_qwen/prompt.py`,
3. add assistant input discovery and aggregation tests,
4. implement assistant input loading helpers in `adapter.py`,
5. add mocked adapter run tests for disabled, missing input, parse failure, and validation failure,
6. implement `AssistedQwenAdapter.run`,
7. add cache tests,
8. implement assisted raw-output cache,
9. optionally extract shared Qwen runtime helpers if duplication with Task 8 becomes awkward,
10. wire the assisted-Qwen CLI branch and run artifacts,
11. add CLI smoke tests,
12. run verification.

## Verification

Run scoped tests:

```bash
uv run pytest tests/test_assisted_qwen_prompt.py -v
uv run pytest tests/test_cli_smoke.py -v
```

Run scoped Ruff:

```bash
uv run --extra dev ruff check \
  src/aiforensics/baselines/assisted_qwen \
  src/aiforensics/baselines/qwen_vl \
  src/aiforensics/cli/main.py \
  tests/test_assisted_qwen_prompt.py \
  tests/test_cli_smoke.py

uv run --extra dev ruff format --check \
  src/aiforensics/baselines/assisted_qwen \
  src/aiforensics/baselines/qwen_vl \
  src/aiforensics/cli/main.py \
  tests/test_assisted_qwen_prompt.py \
  tests/test_cli_smoke.py
```

Run full tests:

```bash
uv run pytest
```

Run smoke gate:

```bash
uv run aiforensics prepare --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline qwen_vl --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline assisted_qwen --config configs/phase_ab_smoke.yaml
uv run aiforensics evaluate --config configs/phase_ab_smoke.yaml
uv run aiforensics report --config configs/phase_ab_smoke.yaml
```

Expected:

- `tests/test_assisted_qwen_prompt.py` passes without Qwen dependencies,
- existing Task 1-8 tests keep passing,
- scoped Ruff passes,
- smoke `clip_probe` run completes,
- smoke `qwen_vl` run is deferred,
- smoke `assisted_qwen` run is deferred,
- smoke `evaluate` and `report` still exit successfully,
- no NPR implementation is added.

The repository currently has pre-existing Ruff debt outside the scoped Task 9 files. Keep Task 9
lint and format changes scoped unless the user explicitly asks for a broader cleanup.

## Acceptance Checklist

- `AssistedQwenAdapter.name` is exactly `assisted_qwen`,
- CLI routes `assisted_qwen` to the real adapter,
- CLI preserves `clip_probe`, `qwen_vl`, and `npr` behavior,
- prompt contains `classifier_pred` and `fake_probability`,
- prompt contains no labels, training examples, retrieval, NPR output, patch attribution, or hidden
  detector claims,
- CLIP predictions are discovered only from completed `clip_probe` runs,
- all evaluation samples require an assistant input,
- multiple CLIP seeds are aggregated deterministically,
- `fake_probability` and `classifier_pred` are derived only from CLIP predictions,
- Task 8 Qwen parser is reused,
- real Qwen dependencies are lazy and optional,
- model/processor load once per run,
- raw-output cache is sample/model/prompt/assistant-input aware,
- corrupt cache recomputes instead of crashing,
- every completed run writes valid `predictions.jsonl`,
- failed/deferred runs do not leave stale `predictions.jsonl`,
- one assisted-Qwen run directory is created per CLI invocation,
- run artifacts include `config.yaml`, `environment.json`, `status.json`, and `logs.txt`,
- smoke config remains model-download-free and GPU-free,
- `aiforensics run --baseline assisted_qwen --config configs/phase_ab_smoke.yaml` returns `0` with
  deferred status,
- NPR, trained fusion, report rendering, dataset downloading, explanation scoring, RAG, few-shot
  retrieval, and patch attribution remain out of scope.
