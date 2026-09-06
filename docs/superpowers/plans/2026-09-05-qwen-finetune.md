# Qwen Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible `qwen_ft` pipeline that fine-tunes Qwen2.5-VL-7B with LoRA on GenImage real/fake labels, evaluates it with the existing Phase A/B metrics, and compares it against the zero-shot Qwen baseline.

**Architecture:** Keep training, inference, and evaluation as separate stages. Kaggle prepares and uploads bounded data subsets to GCS, Vertex CustomJob performs LoRA training, and a `qwen_ft` inference runner emits the same `predictions.jsonl` contract used by the existing evaluator.

**Tech Stack:** Python 3.10, Pydantic, YAML, pytest, Google Cloud Storage, Vertex AI CustomJob, Hugging Face Transformers, PEFT/LoRA, Qwen2.5-VL fine-tuning framework, existing `aiforensics` CLI.

**Spec:** `docs/superpowers/specs/2026-09-05-qwen-finetune-design.md`

## Global Constraints

- Baseline name is `qwen_ft`.
- Model is `Qwen/Qwen2.5-VL-7B-Instruct`.
- First real run uses 7B; there is no 3B fallback.
- Training target is label-only JSON: `{"label":"real"}` or `{"label":"fake"}`.
- Do not train `confidence` or `evidence` in this stage.
- LoRA is primary; QLoRA is fallback only when LoRA OOMs.
- Training runs on Vertex AI CustomJob, not inside Kaggle or Colab.
- Kaggle filters and uploads data to GCS; Vertex reads from GCS.
- Default bucket is `gs://aiforensics-qwen-ft-579187260419`.
- First training size is 100 real and 100 fake images per generator across all seven GenImage generators.
- Eval-small size is 50 real and 50 fake images per generator.
- Eval-full is the current 7000-image GenImage validation set.
- Checkpoints save every 100 steps, keep the latest three checkpoints, write to GCS, and support resume.
- Do not commit datasets, credentials, access tokens, model weights, checkpoints, caches, or generated outputs.
- Existing Phase A/B CLI behavior must remain stable for `clip_probe`, `qwen_vl`, `npr`, and `assisted_qwen`.

---

## File Structure

- Create `configs/qwen_ft_protocol_a_small.yaml`: config for the first all-generator LoRA experiment.
- Create `configs/qwen_ft_protocol_b_seen_unseen.yaml`: config for the seen/unseen follow-up experiment.
- Create `src/aiforensics/baselines/qwen_ft/__init__.py`: package export for the fine-tuned baseline.
- Create `src/aiforensics/baselines/qwen_ft/adapter.py`: inference adapter that loads base Qwen plus a LoRA adapter and writes label-only predictions.
- Create `src/aiforensics/baselines/qwen_ft/parsing.py`: parser for `{"label":"real"}` and `{"label":"fake"}` outputs.
- Create `src/aiforensics/finetune/__init__.py`: fine-tuning utilities package.
- Create `src/aiforensics/finetune/genimage_export.py`: balanced GenImage subset selection and upload manifest generation.
- Create `src/aiforensics/finetune/qwen_data.py`: conversion from manifest rows to Qwen image-conversation JSONL.
- Create `src/aiforensics/finetune/vertex_job.py`: Vertex CustomJob spec builder and submitter.
- Create `scripts/export_genimage_qwen_ft_subset.py`: Kaggle/local CLI script for exporting selected images and manifests.
- Create `scripts/submit_qwen_ft_vertex_job.py`: CLI script that submits a Vertex training job from a YAML config.
- Create `scripts/build_qwen_ft_notebooks.py`: notebook builder for thin Kaggle and Vertex submission notebooks.
- Create `notebooks/kaggle_qwen_ft_export.ipynb`: generated notebook that filters GenImage subsets and uploads to GCS.
- Create `notebooks/vertex_qwen_ft_submit_and_eval.ipynb`: generated notebook that submits training and runs eval.
- Modify `src/aiforensics/config/models.py`: add `QwenFTConfig` and include it in `BaselinesConfig`.
- Modify `src/aiforensics/cli/main.py`: accept `qwen_ft` in `aiforensics run --baseline`.
- Modify `src/aiforensics/schemas/predictions.py`: allow `model_name="qwen_ft"` and allow MLLM label-only records without strict evidence fields when the adapter does not request MLLM strict validation.
- Modify `src/aiforensics/evaluation/metrics.py`: skip AUROC when all `score_fake` values are null and write an explicit null metric.
- Modify `src/aiforensics/reporting/markdown.py`: render `qwen_ft` rows, parse failure stats, and confusion matrix links.
- Test `tests/test_qwen_ft_parsing.py`: label-only parser behavior.
- Test `tests/test_qwen_ft_data_export.py`: balanced train/eval subset selection.
- Test `tests/test_qwen_ft_training_json.py`: Qwen conversation JSONL output.
- Test `tests/test_qwen_ft_vertex_job.py`: Vertex job spec fields, GCS paths, checkpoint settings.
- Test `tests/test_qwen_ft_adapter.py`: fake model runner writes valid `predictions.jsonl`.
- Test `tests/test_qwen_ft_cli.py`: CLI dispatch and report integration.

---

### Task 1: Add `qwen_ft` Config And CLI Registration

**Files:**
- Modify: `src/aiforensics/config/models.py`
- Modify: `src/aiforensics/cli/main.py`
- Create: `configs/qwen_ft_protocol_a_small.yaml`
- Create: `configs/qwen_ft_protocol_b_seen_unseen.yaml`
- Create: `tests/test_qwen_ft_cli.py`

**Interfaces:**
- Consumes: existing `AppConfig`, `BaselinesConfig`, CLI `run --baseline`.
- Produces: `QwenFTConfig`, config sections `baselines.qwen_ft`, and CLI dispatch for `qwen_ft`.

- [ ] **Step 1: Write failing config test**

Add this test:

```python
from pathlib import Path

from aiforensics.config import load_config


def test_qwen_ft_protocol_a_config_loads():
    cfg = load_config(Path("configs/qwen_ft_protocol_a_small.yaml"))

    assert cfg.baselines.qwen_ft.enabled is True
    assert cfg.baselines.qwen_ft.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg.baselines.qwen_ft.adapter_uri.startswith("gs://aiforensics-qwen-ft-579187260419/")
    assert cfg.baselines.qwen_ft.prompt_id == "qwen_ft_label_json_v1"
    assert cfg.baselines.qwen_ft.output_fields == ["label"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_qwen_ft_cli.py::test_qwen_ft_protocol_a_config_loads -v
```

Expected: FAIL because `qwen_ft` config does not exist yet.

- [ ] **Step 3: Add config model**

Add this Pydantic model in `src/aiforensics/config/models.py`:

```python
class QwenFTConfig(BaseModel):
    enabled: bool
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_uri: str
    prompt_id: str = "qwen_ft_label_json_v1"
    temperature: float = Field(default=0.0, ge=0.0)
    max_new_tokens: int = Field(default=32, gt=0)
    cache_outputs: bool = True
    allow_deferred: bool = True
    dtype: Literal["bfloat16", "float16", "float32"] = "float16"
    output_fields: list[Literal["label"]] = ["label"]
```

Extend `BaselinesConfig`:

```python
class BaselinesConfig(BaseModel):
    clip_probe: ClipProbeConfig
    qwen_vl: QwenVLConfig
    assisted_qwen: AssistedQwenConfig
    npr: NPRConfig
    qwen_ft: QwenFTConfig
```

- [ ] **Step 4: Add protocol A config**

Create `configs/qwen_ft_protocol_a_small.yaml` with:

```yaml
project:
  name: ai-image-forensics
  phase: qwen_ft_protocol_a_small
  description: Qwen2.5-VL LoRA fine-tuning, all GenImage generators, small train and eval.

paths:
  data_root: data
  manifest_root: manifests/qwen_ft/protocol_a_small
  cache_root: .cache/aiforensics-qwen-ft-a-small
  output_root: outputs/qwen-ft-protocol-a-small
  external_root: external

runtime:
  python: "3.10"
  seed: 70
  device: auto
  batch_size: 1
  num_workers: 2
  fail_fast: false

datasets:
  tiny_genimage:
    enabled: false
    source: TheKernel01/Tiny-GenImage
    use_original_split: true
    train_manifest: manifests/qwen_ft/protocol_a_small/train.csv
    dev_manifest: manifests/qwen_ft/protocol_a_small/eval_small.csv
    generators: []
    max_images: 0
    balance_labels: true
  genimage_unseen:
    enabled: true
    generators:
      - imagenet_ai_0419_biggan
      - imagenet_ai_0419_vqdm
      - imagenet_ai_0424_sdv5
      - imagenet_ai_0424_wukong
      - imagenet_ai_0508_adm
      - imagenet_glide
      - imagenet_midjourney
    max_images: 100
    balance_labels: true
    split: external
    manifest: manifests/qwen_ft/protocol_a_small/eval_small.csv
  synthbuster:
    enabled: false
    max_images: 0
    balance_labels: true
    split: external
    manifest: manifests/synthbuster_external.csv

baselines:
  clip_probe:
    enabled: false
    model_family: openclip
    model_name: ViT-L-14-quickgelu
    pretrained: openai
    classifier: logistic_regression
    seeds: [70]
    cache_embeddings: true
  qwen_vl:
    enabled: false
    model_id: Qwen/Qwen2.5-VL-7B-Instruct
    prompt_id: qwen_json_v1
    temperature: 0.0
    max_new_tokens: 128
    cache_outputs: false
    allow_deferred: true
  assisted_qwen:
    enabled: false
    base_model_id: Qwen/Qwen2.5-VL-7B-Instruct
    prompt_id: assisted_qwen_json_v1
    assistant_source: clip_probe
    include_classifier_pred: true
    include_fake_probability: true
    temperature: 0.0
    max_new_tokens: 128
    cache_outputs: false
    allow_deferred: true
  npr:
    enabled: false
    repo_url: https://github.com/chuangchuangtan/NPR-DeepfakeDetection
    repo_commit: smoke-disabled
    checkpoint_path: external/NPR-DeepfakeDetection/NPR.pth
    checkpoint_sha256: smoke-disabled
    batch_size: 2
    allow_deferred: true
  qwen_ft:
    enabled: true
    model_id: Qwen/Qwen2.5-VL-7B-Instruct
    adapter_uri: gs://aiforensics-qwen-ft-579187260419/checkpoints/protocol-a-small/final_adapter
    prompt_id: qwen_ft_label_json_v1
    temperature: 0.0
    max_new_tokens: 32
    cache_outputs: true
    allow_deferred: true
    dtype: float16
    output_fields:
      - label

evaluation:
  labels:
    negative: real
    positive: fake
  metrics:
    - accuracy
    - balanced_accuracy
    - precision
    - recall
    - f1
    - auroc
  group_by:
    - source
    - split

report:
  filename: qwen_ft_protocol_a_small_report.md
  include_failure_notes: true
  include_explanations_sample: false
  explanation_sample_size: 0
```

- [ ] **Step 5: Add protocol B config**

Create `configs/qwen_ft_protocol_b_seen_unseen.yaml` with the same structure as protocol A, but set:

```yaml
project:
  phase: qwen_ft_protocol_b_seen_unseen
  description: Qwen2.5-VL LoRA fine-tuning, seen GenImage generators with unseen-generator evaluation.
paths:
  manifest_root: manifests/qwen_ft/protocol_b_seen_unseen
  cache_root: .cache/aiforensics-qwen-ft-b-seen-unseen
  output_root: outputs/qwen-ft-protocol-b-seen-unseen
baselines:
  qwen_ft:
    adapter_uri: gs://aiforensics-qwen-ft-579187260419/checkpoints/protocol-b-seen-unseen/final_adapter
report:
  filename: qwen_ft_protocol_b_seen_unseen_report.md
```

The export script in Task 2 will create protocol B manifests with seen train
generators and unseen eval generators.

- [ ] **Step 6: Register CLI dispatch**

In `src/aiforensics/cli/main.py`, extend the baseline adapter factory or match
block so:

```python
elif baseline == "qwen_ft":
    from aiforensics.baselines.qwen_ft import QwenFTAdapter

    adapter = QwenFTAdapter()
```

- [ ] **Step 7: Verify**

Run:

```bash
uv run pytest tests/test_qwen_ft_cli.py -v
uv run pytest tests/test_cli_smoke.py -v
```

Expected: new config tests pass and existing CLI smoke behavior remains stable.

- [ ] **Step 8: Commit**

```bash
git add src/aiforensics/config/models.py src/aiforensics/cli/main.py configs/qwen_ft_protocol_a_small.yaml configs/qwen_ft_protocol_b_seen_unseen.yaml tests/test_qwen_ft_cli.py
git commit -m "feat: register qwen fine-tuned baseline config"
```

---

### Task 2: Build Balanced GenImage Export To GCS

**Files:**
- Create: `src/aiforensics/finetune/__init__.py`
- Create: `src/aiforensics/finetune/genimage_export.py`
- Create: `scripts/export_genimage_qwen_ft_subset.py`
- Create: `tests/test_qwen_ft_data_export.py`

**Interfaces:**
- Consumes: GenImage layout `<root>/<generator>/<split>/{ai,nature}/*`.
- Produces: `ExportPlan`, `build_export_plan(...)`, `write_export_manifests(...)`.

- [ ] **Step 1: Write failing balanced selection test**

Add synthetic directories:

```python
from pathlib import Path

from PIL import Image

from aiforensics.finetune.genimage_export import build_export_plan


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(path)


def test_build_export_plan_balances_labels_per_generator(tmp_path):
    for generator in ("imagenet_ai_0419_biggan", "imagenet_glide"):
        for split in ("train", "val"):
            for label_dir in ("ai", "nature"):
                for idx in range(3):
                    _image(tmp_path / generator / split / label_dir / f"{idx}.png")

    plan = build_export_plan(
        data_root=tmp_path,
        train_generators=["imagenet_ai_0419_biggan", "imagenet_glide"],
        eval_generators=["imagenet_ai_0419_biggan", "imagenet_glide"],
        train_per_label=2,
        eval_per_label=1,
        seed=70,
    )

    assert len(plan.train_records) == 8
    assert len(plan.eval_records) == 4
    assert {r.label for r in plan.train_records} == {"real", "fake"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_qwen_ft_data_export.py::test_build_export_plan_balances_labels_per_generator -v
```

Expected: FAIL because `genimage_export.py` does not exist.

- [ ] **Step 3: Implement export planning**

Define:

```python
@dataclass(frozen=True)
class ExportRecord:
    sample_id: str
    source_path: Path
    relative_gcs_path: str
    label: Literal["real", "fake"]
    source: str
    split: Literal["train", "external"]
    checksum: str


@dataclass(frozen=True)
class ExportPlan:
    train_records: list[ExportRecord]
    eval_records: list[ExportRecord]
```

Implement mapping:

```python
LABEL_DIRS = {"ai": "fake", "nature": "real"}
SPLITS = {"train": "train", "val": "external"}
```

Selection rules:

- Sort candidate paths before sampling.
- Use `random.Random(seed).sample(...)`.
- Select exactly `train_per_label` per label and generator from `train`.
- Select exactly `eval_per_label` per label and generator from `val`.
- Raise `ValueError` naming the generator, split, label directory, requested
  count, and available count when there are not enough images.

- [ ] **Step 4: Write manifests**

Implement:

```python
def write_export_manifests(plan: ExportPlan, manifest_dir: Path) -> dict[str, Path]:
    ...
```

Write:

```text
train.csv
eval_small.csv
```

with columns:

```text
sample_id,path,label,source,split,checksum,dataset,generator
```

Use GCS destination paths in `path`, not Kaggle local paths.

- [ ] **Step 5: Add export script**

Create `scripts/export_genimage_qwen_ft_subset.py` with argparse flags:

```text
--data-root
--bucket-uri
--protocol protocol-a-small|protocol-b-seen-unseen
--work-dir
--seed
--train-per-label
--eval-per-label
--upload
```

When `--upload` is set, use `gcloud storage cp` for files listed by the plan.
When `--upload` is absent, write manifests and print a dry-run summary.

- [ ] **Step 6: Verify**

Run:

```bash
uv run pytest tests/test_qwen_ft_data_export.py -v
uv run ruff check src/aiforensics/finetune scripts/export_genimage_qwen_ft_subset.py tests/test_qwen_ft_data_export.py
```

Expected: tests pass and lint passes.

- [ ] **Step 7: Commit**

```bash
git add src/aiforensics/finetune scripts/export_genimage_qwen_ft_subset.py tests/test_qwen_ft_data_export.py
git commit -m "feat: export qwen fine-tuning genimage subsets"
```

---

### Task 3: Convert Manifests To Qwen Label-Only Training JSONL

**Files:**
- Create: `src/aiforensics/finetune/qwen_data.py`
- Create: `tests/test_qwen_ft_training_json.py`

**Interfaces:**
- Consumes: manifest CSV records.
- Produces: `write_qwen_label_jsonl(manifest_path: Path, output_path: Path) -> None`.

- [ ] **Step 1: Write failing training JSON test**

Add:

```python
import json

from aiforensics.finetune.qwen_data import write_qwen_label_jsonl


def test_write_qwen_label_jsonl_uses_label_only_response(tmp_path):
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "sample_id,path,label,source,split,checksum\n"
        "s1,gs://bucket/a.png,fake,g1,train,abc\n"
        "s2,gs://bucket/b.png,real,g1,train,def\n",
        encoding="utf-8",
    )
    out = tmp_path / "qwen_train.jsonl"

    write_qwen_label_jsonl(manifest, out)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["image"] == "gs://bucket/a.png"
    assert json.loads(rows[0]["conversations"][1]["value"]) == {"label": "fake"}
    assert json.loads(rows[1]["conversations"][1]["value"]) == {"label": "real"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_qwen_ft_training_json.py::test_write_qwen_label_jsonl_uses_label_only_response -v
```

Expected: FAIL because `qwen_data.py` does not exist.

- [ ] **Step 3: Implement conversion**

Use this prompt text exactly:

```text
<image>
You are an image-forensics classifier. Classify the image as either "real" or "fake". Return exactly one JSON object with one key: label. The label must be "real" or "fake".
```

Write one JSONL row per manifest row:

```python
{
    "image": row["path"],
    "conversations": [
        {"from": "human", "value": PROMPT_TEXT},
        {"from": "gpt", "value": json.dumps({"label": row["label"]}, separators=(",", ":"))},
    ],
}
```

- [ ] **Step 4: Integrate conversion into export script**

After `train.csv` is written, call:

```python
write_qwen_label_jsonl(train_manifest, manifest_dir / "qwen_train.jsonl")
```

Upload `qwen_train.jsonl` to:

```text
gs://aiforensics-qwen-ft-579187260419/data/<protocol>/qwen_train.jsonl
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run pytest tests/test_qwen_ft_training_json.py tests/test_qwen_ft_data_export.py -v
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/aiforensics/finetune/qwen_data.py scripts/export_genimage_qwen_ft_subset.py tests/test_qwen_ft_training_json.py tests/test_qwen_ft_data_export.py
git commit -m "feat: build qwen label-only fine-tuning data"
```

---

### Task 4: Build Vertex Training Job Spec

**Files:**
- Create: `src/aiforensics/finetune/vertex_job.py`
- Create: `scripts/submit_qwen_ft_vertex_job.py`
- Create: `tests/test_qwen_ft_vertex_job.py`

**Interfaces:**
- Consumes: project id, region, bucket uri, protocol name, training JSONL URI.
- Produces: `build_vertex_custom_job_spec(...) -> dict[str, object]`.

- [ ] **Step 1: Write failing Vertex spec test**

Add:

```python
from aiforensics.finetune.vertex_job import build_vertex_custom_job_spec


def test_vertex_job_spec_uses_a100_and_gcs_checkpoints():
    spec = build_vertex_custom_job_spec(
        project_id="579187260419",
        location="asia-southeast1",
        bucket_uri="gs://aiforensics-qwen-ft-579187260419",
        protocol="protocol-a-small",
        training_jsonl_uri="gs://aiforensics-qwen-ft-579187260419/data/protocol-a-small/qwen_train.jsonl",
        machine_type="a2-highgpu-1g",
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
    )

    worker = spec["jobSpec"]["workerPoolSpecs"][0]
    assert worker["machineSpec"]["acceleratorType"] == "NVIDIA_TESLA_A100"
    assert worker["machineSpec"]["acceleratorCount"] == 1
    assert spec["jobSpec"]["baseOutputDirectory"]["outputUriPrefix"] == "gs://aiforensics-qwen-ft-579187260419/checkpoints/protocol-a-small"
    args = worker["containerSpec"]["args"]
    assert "--save_steps=100" in args
    assert "--save_total_limit=3" in args
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_qwen_ft_vertex_job.py::test_vertex_job_spec_uses_a100_and_gcs_checkpoints -v
```

Expected: FAIL because `vertex_job.py` does not exist.

- [ ] **Step 3: Implement spec builder**

Build a CustomJob REST payload with:

```python
displayName = f"aiforensics-qwen-ft-{protocol}"
baseOutputDirectory.outputUriPrefix = f"{bucket_uri}/checkpoints/{protocol}"
```

Worker pool:

```python
{
    "machineSpec": {
        "machineType": machine_type,
        "acceleratorType": accelerator_type,
        "acceleratorCount": accelerator_count,
    },
    "replicaCount": "1",
    "containerSpec": {
        "imageUri": image_uri,
        "args": [
            f"--model_name_or_path={model_id}",
            f"--data_path={training_jsonl_uri}",
            f"--output_dir={bucket_uri}/checkpoints/{protocol}",
            "--lora_enable=True",
            "--tune_mm_llm=True",
            "--tune_mm_vision=False",
            "--tune_mm_mlp=True",
            "--num_train_epochs=1",
            "--per_device_train_batch_size=1",
            "--gradient_accumulation_steps=16",
            "--learning_rate=1e-6",
            "--save_strategy=steps",
            "--save_steps=100",
            "--save_total_limit=3",
            "--logging_steps=10",
        ],
    },
}
```

Use an explicit image URI owned by this project after Task 5 creates the
training container. Until then, tests may pass `image_uri` explicitly.

- [ ] **Step 4: Add submit script**

Create `scripts/submit_qwen_ft_vertex_job.py` with:

```text
--project-id
--location
--bucket-uri
--protocol
--training-jsonl-uri
--image-uri
--machine-type
--accelerator-type
--accelerator-count
--dry-run
```

When `--dry-run` is set, print the JSON payload. When not dry-run, call the
Vertex AI REST API using `google.auth.default(scopes=[...])` and
`google.auth.transport.requests.AuthorizedSession`.

- [ ] **Step 5: Verify**

Run:

```bash
uv run pytest tests/test_qwen_ft_vertex_job.py -v
uv run ruff check src/aiforensics/finetune/vertex_job.py scripts/submit_qwen_ft_vertex_job.py tests/test_qwen_ft_vertex_job.py
```

Expected: tests pass and lint passes.

- [ ] **Step 6: Commit**

```bash
git add src/aiforensics/finetune/vertex_job.py scripts/submit_qwen_ft_vertex_job.py tests/test_qwen_ft_vertex_job.py
git commit -m "feat: build vertex qwen fine-tuning job spec"
```

---

### Task 5: Add Training Container Recipe

**Files:**
- Create: `infra/qwen_ft/Dockerfile`
- Create: `infra/qwen_ft/train_qwen_ft.py`
- Create: `infra/qwen_ft/requirements.txt`
- Create: `tests/test_qwen_ft_training_entrypoint.py`

**Interfaces:**
- Consumes: Qwen training JSONL path and output directory.
- Produces: LoRA adapter checkpoints under the requested output directory.

- [ ] **Step 1: Write entrypoint argument test**

Add:

```python
from infra.qwen_ft.train_qwen_ft import build_parser


def test_training_entrypoint_parses_required_args():
    args = build_parser().parse_args(
        [
            "--model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct",
            "--data_path=gs://bucket/data/qwen_train.jsonl",
            "--output_dir=gs://bucket/checkpoints/protocol-a-small",
            "--lora_enable=True",
        ]
    )

    assert args.model_name_or_path == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert args.data_path == "gs://bucket/data/qwen_train.jsonl"
    assert args.output_dir == "gs://bucket/checkpoints/protocol-a-small"
    assert args.lora_enable == "True"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_qwen_ft_training_entrypoint.py::test_training_entrypoint_parses_required_args -v
```

Expected: FAIL because `infra/qwen_ft/train_qwen_ft.py` does not exist.

- [ ] **Step 3: Implement training entrypoint wrapper**

The wrapper must:

- Parse the exact arguments used by Task 4.
- Copy GCS `data_path` to local disk before training when the upstream Qwen
  script does not accept GCS URIs directly.
- Invoke the Qwen2.5-VL fine-tuning module or script with LoRA flags.
- Sync `output_dir` checkpoints and final adapter back to GCS.
- Exit non-zero when training fails before writing a final adapter.

- [ ] **Step 4: Create requirements**

Use package families:

```text
torch
torchvision
transformers
accelerate
peft
bitsandbytes
qwen-vl-utils
google-cloud-storage
Pillow
```

Pin exact versions after the first successful container build in a follow-up
commit. The first container task records installed versions in the training log.

- [ ] **Step 5: Create Dockerfile**

Base on a CUDA/PyTorch image compatible with A100 and install:

```text
infra/qwen_ft/requirements.txt
Qwen2.5-VL fine-tuning source
```

Set the entrypoint:

```dockerfile
ENTRYPOINT ["python", "/app/train_qwen_ft.py"]
```

- [ ] **Step 6: Verify locally without GPU**

Run:

```bash
uv run pytest tests/test_qwen_ft_training_entrypoint.py -v
uv run ruff check infra/qwen_ft tests/test_qwen_ft_training_entrypoint.py
```

Expected: parser tests and lint pass. GPU training is not required locally.

- [ ] **Step 7: Commit**

```bash
git add infra/qwen_ft tests/test_qwen_ft_training_entrypoint.py
git commit -m "feat: add qwen fine-tuning training container"
```

---

### Task 6: Implement Label-Only `qwen_ft` Inference Adapter

**Files:**
- Create: `src/aiforensics/baselines/qwen_ft/__init__.py`
- Create: `src/aiforensics/baselines/qwen_ft/parsing.py`
- Create: `src/aiforensics/baselines/qwen_ft/adapter.py`
- Create: `tests/test_qwen_ft_parsing.py`
- Create: `tests/test_qwen_ft_adapter.py`

**Interfaces:**
- Consumes: evaluation manifests from `selected_evaluation_manifests`.
- Produces: completed run directory with `predictions.jsonl`.

- [ ] **Step 1: Write parser tests**

Add:

```python
from aiforensics.baselines.qwen_ft.parsing import parse_qwen_ft_output


def test_parse_label_only_fake():
    parsed = parse_qwen_ft_output('{"label":"fake"}')

    assert parsed.label_pred == "fake"
    assert parsed.score_fake is None
    assert parsed.parse_status == "parsed"


def test_parse_label_only_real():
    parsed = parse_qwen_ft_output('{"label":"real"}')

    assert parsed.label_pred == "real"
    assert parsed.score_fake is None
    assert parsed.parse_status == "parsed"


def test_parse_label_only_invalid_label_fails():
    parsed = parse_qwen_ft_output('{"label":"synthetic"}')

    assert parsed.label_pred == "unknown"
    assert parsed.score_fake is None
    assert parsed.parse_status == "failed"
```

- [ ] **Step 2: Run parser tests to verify failure**

Run:

```bash
uv run pytest tests/test_qwen_ft_parsing.py -v
```

Expected: FAIL because parser module does not exist.

- [ ] **Step 3: Implement parser**

Return a small Pydantic or dataclass result:

```python
@dataclass(frozen=True)
class QwenFTParseResult:
    label_pred: Literal["real", "fake", "unknown"]
    score_fake: float | None
    parse_status: Literal["parsed", "recovered", "failed"]
    raw_output: str
```

`parse_qwen_ft_output` accepts strict JSON first and then recovers the first JSON
object embedded in surrounding text. It accepts only `real` and `fake`.

- [ ] **Step 4: Write adapter test with fake generator**

Patch the model call so no GPU is needed:

```python
def test_qwen_ft_adapter_writes_label_only_predictions(tmp_path, monkeypatch):
    cfg = make_smoke_config_with_qwen_ft(tmp_path)
    adapter = QwenFTAdapter()

    monkeypatch.setattr(adapter, "_generate_one_image", lambda *args, **kwargs: '{"label":"fake"}')

    result = adapter.run(tmp_path / "run", cfg)

    assert result.status == "completed"
    preds = load_predictions(result.prediction_path)
    assert preds[0].model_name == "qwen_ft"
    assert preds[0].label_pred == "fake"
    assert preds[0].score_fake is None
    assert preds[0].parse_status == "parsed"
```

- [ ] **Step 5: Implement adapter**

Follow the structure of `QwenVLAdapter`, with these differences:

- Use `config.baselines.qwen_ft`.
- Load base model plus PEFT adapter from `adapter_uri`.
- Use prompt id `qwen_ft_label_json_v1`.
- Generate at most 32 new tokens.
- Write `explanation=""`.
- Write `score_fake=None`.
- Validate predictions without requiring MLLM evidence fields.

- [ ] **Step 6: Verify**

Run:

```bash
uv run pytest tests/test_qwen_ft_parsing.py tests/test_qwen_ft_adapter.py -v
uv run ruff check src/aiforensics/baselines/qwen_ft tests/test_qwen_ft_parsing.py tests/test_qwen_ft_adapter.py
```

Expected: tests pass and lint passes.

- [ ] **Step 7: Commit**

```bash
git add src/aiforensics/baselines/qwen_ft tests/test_qwen_ft_parsing.py tests/test_qwen_ft_adapter.py
git commit -m "feat: add qwen fine-tuned inference adapter"
```

---

### Task 7: Update Evaluation And Reporting For Label-Only Outputs

**Files:**
- Modify: `src/aiforensics/evaluation/metrics.py`
- Modify: `src/aiforensics/reporting/markdown.py`
- Modify: `src/aiforensics/schemas/predictions.py`
- Create: `tests/test_qwen_ft_evaluation.py`

**Interfaces:**
- Consumes: `qwen_ft` predictions with `score_fake=None`.
- Produces: metrics with AUROC null and classification metrics populated.

- [ ] **Step 1: Write failing evaluation test**

Add:

```python
from aiforensics.evaluation.metrics import compute_classification_metrics
from aiforensics.schemas.predictions import PredictionRecord


def test_label_only_qwen_ft_skips_auroc_but_keeps_classification_metrics():
    records = [
        PredictionRecord(sample_id="a", label_true="fake", label_pred="fake", score_fake=None, model_name="qwen_ft", source="g"),
        PredictionRecord(sample_id="b", label_true="real", label_pred="real", score_fake=None, model_name="qwen_ft", source="g"),
        PredictionRecord(sample_id="c", label_true="fake", label_pred="real", score_fake=None, model_name="qwen_ft", source="g"),
        PredictionRecord(sample_id="d", label_true="real", label_pred="fake", score_fake=None, model_name="qwen_ft", source="g"),
    ]

    metrics = compute_classification_metrics(records)

    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["auroc"] is None
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_qwen_ft_evaluation.py::test_label_only_qwen_ft_skips_auroc_but_keeps_classification_metrics -v
```

Expected: FAIL until `qwen_ft` model name and null AUROC behavior are supported.

- [ ] **Step 3: Support `qwen_ft` model name**

Update prediction model validation so `model_name` accepts:

```text
qwen_ft
```

Keep existing names valid.

- [ ] **Step 4: Ensure AUROC skip behavior**

When no records have usable `score_fake`, write:

```python
metrics["auroc"] = None
```

Do not convert null scores to hard-label scores.

- [ ] **Step 5: Add confusion matrix output**

Add a compact artifact:

```json
{
  "labels": ["real", "fake"],
  "matrix": [[tn, fp], [fn, tp]]
}
```

Write it as `confusion_matrix.json` beside `metrics.json`.

- [ ] **Step 6: Render report**

In the Markdown report:

- Show `qwen_ft` in baseline status and metric tables.
- Render AUROC as `N/A` when it is null.
- Add a short note: `qwen_ft is label-only in this phase; AUROC is skipped unless a real fake score is available.`

- [ ] **Step 7: Verify**

Run:

```bash
uv run pytest tests/test_qwen_ft_evaluation.py tests/test_reporting.py -v
```

Expected: tests pass and existing report tests remain stable.

- [ ] **Step 8: Commit**

```bash
git add src/aiforensics/evaluation/metrics.py src/aiforensics/reporting/markdown.py src/aiforensics/schemas/predictions.py tests/test_qwen_ft_evaluation.py tests/test_reporting.py
git commit -m "feat: report qwen fine-tuned label-only metrics"
```

---

### Task 8: Generate Thin Notebooks For Export, Submit, And Eval

**Files:**
- Create: `scripts/build_qwen_ft_notebooks.py`
- Create: `notebooks/kaggle_qwen_ft_export.ipynb`
- Create: `notebooks/vertex_qwen_ft_submit_and_eval.ipynb`
- Create: `tests/test_qwen_ft_notebooks.py`

**Interfaces:**
- Consumes: scripts from Tasks 2 and 4.
- Produces: reproducible notebooks that call scripts instead of embedding business logic.

- [ ] **Step 1: Write notebook structure test**

Add:

```python
import json
from pathlib import Path


def test_kaggle_qwen_ft_export_notebook_calls_export_script():
    notebook = json.loads(Path("notebooks/kaggle_qwen_ft_export.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "export_genimage_qwen_ft_subset.py" in source
    assert "--protocol protocol-a-small" in source
    assert "--train-per-label 100" in source
    assert "--eval-per-label 50" in source
    assert "--bucket-uri gs://aiforensics-qwen-ft-579187260419" in source
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_qwen_ft_notebooks.py::test_kaggle_qwen_ft_export_notebook_calls_export_script -v
```

Expected: FAIL because notebooks do not exist.

- [ ] **Step 3: Implement notebook builder**

The Kaggle notebook must contain cells for:

```text
1. Install repo and dependencies.
2. Authenticate GCP without printing keys or tokens.
3. Verify GenImage data root.
4. Run export script with protocol-a-small, train-per-label 100, eval-per-label 50.
5. Print GCS paths for train.csv, eval_small.csv, and qwen_train.jsonl.
```

The Vertex notebook must contain cells for:

```text
1. Verify gcloud auth.
2. Dry-run Vertex job JSON.
3. Submit Vertex job.
4. After adapter exists, run qwen_ft eval-small.
5. If eval-small passes parse failure threshold, run eval-full.
6. Generate report.
```

- [ ] **Step 4: Generate notebooks**

Run:

```bash
uv run python scripts/build_qwen_ft_notebooks.py
```

Expected output:

```text
wrote notebooks/kaggle_qwen_ft_export.ipynb
wrote notebooks/vertex_qwen_ft_submit_and_eval.ipynb
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run pytest tests/test_qwen_ft_notebooks.py -v
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_qwen_ft_notebooks.py notebooks/kaggle_qwen_ft_export.ipynb notebooks/vertex_qwen_ft_submit_and_eval.ipynb tests/test_qwen_ft_notebooks.py
git commit -m "feat: add qwen fine-tuning notebooks"
```

---

### Task 9: End-To-End Smoke Without Real Qwen Training

**Files:**
- Create: `configs/qwen_ft_smoke.yaml`
- Create: `tests/test_qwen_ft_smoke_pipeline.py`

**Interfaces:**
- Consumes: smoke fixture images and fake adapter generation.
- Produces: a complete prepare -> run -> evaluate -> report cycle for `qwen_ft`.

- [ ] **Step 1: Add smoke config**

Create `configs/qwen_ft_smoke.yaml` using fixture manifests and:

```yaml
project:
  phase: qwen_ft_smoke
paths:
  data_root: tests/fixtures/smoke_data
  manifest_root: tests/fixtures/manifests
  cache_root: .cache/aiforensics-qwen-ft-smoke
  output_root: outputs/qwen-ft-smoke
baselines:
  qwen_ft:
    enabled: true
    model_id: Qwen/Qwen2.5-VL-7B-Instruct
    adapter_uri: smoke://adapter
    prompt_id: qwen_ft_label_json_v1
    temperature: 0.0
    max_new_tokens: 32
    cache_outputs: false
    allow_deferred: true
    dtype: float16
    output_fields:
      - label
```

- [ ] **Step 2: Write smoke pipeline test**

Patch `_generate_one_image` to return deterministic labels, then run:

```python
def test_qwen_ft_smoke_pipeline(cli_runner, monkeypatch):
    monkeypatch.setenv("AIF_QWEN_FT_SMOKE", "1")

    assert cli_runner(["prepare", "--config", "configs/qwen_ft_smoke.yaml"]) == 0
    assert cli_runner(["run", "--baseline", "qwen_ft", "--config", "configs/qwen_ft_smoke.yaml"]) == 0
    assert cli_runner(["evaluate", "--config", "configs/qwen_ft_smoke.yaml"]) == 0
    assert cli_runner(["report", "--config", "configs/qwen_ft_smoke.yaml"]) == 0
```

- [ ] **Step 3: Add smoke mode in adapter**

When `adapter_uri == "smoke://adapter"`, do not import torch, transformers, or
PEFT. Return labels based on fixture sample ids so the test can run on CPU.

- [ ] **Step 4: Verify**

Run:

```bash
uv run pytest tests/test_qwen_ft_smoke_pipeline.py -v
uv run aiforensics prepare --config configs/qwen_ft_smoke.yaml
uv run aiforensics run --baseline qwen_ft --config configs/qwen_ft_smoke.yaml
uv run aiforensics evaluate --config configs/qwen_ft_smoke.yaml
uv run aiforensics report --config configs/qwen_ft_smoke.yaml
```

Expected: every command exits zero and writes a `qwen_ft_smoke` report.

- [ ] **Step 5: Commit**

```bash
git add configs/qwen_ft_smoke.yaml tests/test_qwen_ft_smoke_pipeline.py src/aiforensics/baselines/qwen_ft/adapter.py
git commit -m "test: add qwen fine-tuned smoke pipeline"
```

---

### Task 10: Documentation And Final Verification Gate

**Files:**
- Modify: `docs/runbook-colab-kaggle.md`
- Create: `docs/runbook-qwen-finetune-vertex.md`
- Modify: `docs/schemas/predictions-jsonl.md`

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: operator instructions for Kaggle export, Vertex training, eval-small, and eval-full.

- [ ] **Step 1: Document the operator flow**

Create `docs/runbook-qwen-finetune-vertex.md` with these sections:

```text
1. Create GCS bucket
2. Run Kaggle export notebook
3. Verify GCS artifacts
4. Build/push training container
5. Dry-run Vertex job
6. Submit Vertex job
7. Monitor logs and checkpoints
8. Run eval-small
9. Run eval-full
10. Compare against zero-shot Qwen report
11. Save artifacts
```

- [ ] **Step 2: Document prediction schema extension**

Update `docs/schemas/predictions-jsonl.md`:

- Add `qwen_ft` to Phase A/B values.
- State that `qwen_ft` phase-1 predictions use `score_fake: null`.
- State that `qwen_ft` phase-1 explanations are not evaluated.

- [ ] **Step 3: Run targeted tests**

Run:

```bash
uv run pytest tests/test_qwen_ft_cli.py tests/test_qwen_ft_data_export.py tests/test_qwen_ft_training_json.py tests/test_qwen_ft_vertex_job.py tests/test_qwen_ft_parsing.py tests/test_qwen_ft_adapter.py tests/test_qwen_ft_evaluation.py tests/test_qwen_ft_notebooks.py tests/test_qwen_ft_smoke_pipeline.py -v
```

Expected: all targeted tests pass.

- [ ] **Step 4: Run full verification gate**

Run:

```bash
uv run --extra dev ruff check src tests scripts infra
uv run --extra dev ruff format --check src tests scripts infra
uv run pytest
uv run aiforensics prepare --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
uv run aiforensics evaluate --config configs/phase_ab_smoke.yaml
uv run aiforensics report --config configs/phase_ab_smoke.yaml
uv run aiforensics prepare --config configs/qwen_ft_smoke.yaml
uv run aiforensics run --baseline qwen_ft --config configs/qwen_ft_smoke.yaml
uv run aiforensics evaluate --config configs/qwen_ft_smoke.yaml
uv run aiforensics report --config configs/qwen_ft_smoke.yaml
```

Expected: every command exits zero.

- [ ] **Step 5: Commit**

```bash
git add docs/runbook-colab-kaggle.md docs/runbook-qwen-finetune-vertex.md docs/schemas/predictions-jsonl.md
git commit -m "docs: document qwen fine-tuning operation"
```

---

## Execution Order

Implement tasks in order. Do not start real Vertex training until Tasks 1
through 10 pass locally. The first real cloud run should be:

```text
protocol-a-small
train-per-label: 100
eval-per-label: 50
model: Qwen/Qwen2.5-VL-7B-Instruct
method: LoRA
accelerator: A100 40GB
```

After eval-small passes parser and artifact checks, run eval-full on the 7000
validation images. Protocol B starts only after Protocol A has a valid full
report.

## Plan Self-Review

- Spec coverage: every spec decision maps to at least one task.
- Placeholder scan: no unfinished placeholder markers are present.
- Type consistency: `qwen_ft`, `QwenFTConfig`, `QwenFTAdapter`,
  `parse_qwen_ft_output`, `write_qwen_label_jsonl`,
  `build_export_plan`, and `build_vertex_custom_job_spec` are used
  consistently across tasks.
- Scope check: this plan covers the fine-tuning pipeline only. It does not add
  explanation training, CLIP/NPR fusion, or publication-grade faithfulness
  evaluation.
