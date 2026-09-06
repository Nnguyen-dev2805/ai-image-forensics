# Qwen Fine-Tuning Modal Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the primary Qwen fine-tuning execution path with a Modal Volume + Kaggle zip workflow while preserving the existing `qwen_ft` CLI, evaluator, and report integration.

**Architecture:** Keep `qwen_ft` inference/evaluation inside the existing `aiforensics` CLI. Add a local archive export mode for Kaggle, a Modal app that unpacks data, fine-tunes LoRA, and runs eval, and Modal-specific configs/docs so the main path no longer depends on GCS or Vertex quota.

**Tech Stack:** Python 3.10, pytest, Pydantic/YAML configs, Modal, Modal Volume `aiforensics-qwen-ft`, Hugging Face Transformers, PEFT LoRA, Qwen2.5-VL fine-tuning utilities, existing `aiforensics` CLI.

**Spec:** `docs/superpowers/specs/2026-09-05-qwen-finetune-modal-pivot.md`

## Global Constraints

- Primary execution platform is Modal, not Vertex.
- Primary storage is Modal Volume `aiforensics-qwen-ft`.
- Kaggle must export a zip archive; it must not require GCS credentials.
- Do not require GCP service-account JSON for the Modal path.
- Keep existing Vertex/GCS files as legacy fallback unless the user explicitly asks to delete them.
- Baseline name remains `qwen_ft`.
- Model remains `Qwen/Qwen2.5-VL-7B-Instruct`.
- First real run uses 7B; there is no 3B fallback.
- Training target remains label-only JSON: `{"label":"real"}` or `{"label":"fake"}`.
- Do not train `confidence` or `evidence`.
- LoRA is primary; QLoRA is fallback only after a real LoRA OOM.
- First train size is 100 real and 100 fake images per generator across all seven GenImage generators.
- Eval-small is 50 real and 50 fake images per generator.
- Eval-full is the current 7000-image GenImage validation set.
- Do not commit datasets, credentials, access tokens, model weights, checkpoints, caches, zip archives, or generated outputs.
- Existing Phase A/B CLI behavior must remain stable for `clip_probe`, `qwen_vl`, `npr`, and `assisted_qwen`.

---

## File Structure

- Modify `src/aiforensics/finetune/genimage_export.py`: add relative-path archive manifest support without breaking the existing GCS writer.
- Modify `src/aiforensics/finetune/qwen_data.py`: add a helper to rewrite Qwen JSONL image paths to absolute Modal paths.
- Modify `scripts/export_genimage_qwen_ft_subset.py`: add `--archive-path` and `--path-mode modal-volume`, producing a Kaggle-downloadable zip.
- Create `infra/modal/qwen_ft_modal.py`: Modal app with `inspect_volume`, `unpack_archive`, `train_lora`, `eval_small`, and `eval_full` functions.
- Create `configs/qwen_ft_protocol_a_small_modal.yaml`: Modal runtime config for eval-small.
- Create `configs/qwen_ft_protocol_a_full_modal.yaml`: Modal runtime config for eval-full.
- Modify `scripts/build_qwen_ft_notebooks.py`: generate a Modal zip export notebook or add a separate builder if cleaner.
- Create `notebooks/kaggle_qwen_ft_modal_export.ipynb`: thin notebook that exports the zip only.
- Create `docs/runbook-qwen-finetune-modal.md`: exact setup/run/download instructions for the user.
- Modify `docs/runbook-qwen-finetune-vertex.md`: mark as legacy/fallback.
- Modify `pyproject.toml`: add a `qwen-ft` optional dependency group if missing.
- Add tests:
  - `tests/test_qwen_ft_modal_archive_export.py`
  - `tests/test_qwen_ft_modal_path_rewrite.py`
  - `tests/test_qwen_ft_modal_configs.py`
  - `tests/test_qwen_ft_modal_app.py`
  - update `tests/test_qwen_ft_notebooks.py`

---

### Task 1: Add Modal Archive Export Primitives

**Files:**
- Modify: `src/aiforensics/finetune/genimage_export.py`
- Test: `tests/test_qwen_ft_modal_archive_export.py`

**Interfaces:**
- Consumes: existing `ExportPlan`, `ExportItem`, and balanced GenImage selection logic.
- Produces:
  - `ArchiveManifestMode = Literal["gcs", "relative"]`
  - `write_relative_export_manifests(plan: ExportPlan, manifest_dir: Path) -> dict[str, Path]`
  - `write_export_archive(plan: ExportPlan, archive_path: Path, staging_dir: Path) -> Path`

- [ ] **Step 1: Write failing test for relative manifest paths**

```python
from pathlib import Path

from aiforensics.finetune.genimage_export import (
    build_export_plan,
    write_relative_export_manifests,
)


def test_write_relative_export_manifests_uses_protocol_relative_paths(tmp_path: Path):
    data_root = tmp_path / "genimage"
    image = data_root / "imagenet_ai_0419_biggan" / "train" / "ai" / "0.JPEG"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake image")

    real = data_root / "imagenet_ai_0419_biggan" / "train" / "nature" / "1.JPEG"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"real image")

    plan = build_export_plan(
        data_root=data_root,
        protocol="protocol-a-small",
        train_per_label=1,
        eval_per_label=0,
        seed=70,
    )
    written = write_relative_export_manifests(plan, tmp_path / "manifests")

    text = written["train"].read_text(encoding="utf-8")
    assert "gs://" not in text
    assert "train/imagenet_ai_0419_biggan/ai/0.JPEG" in text
    assert "train/imagenet_ai_0419_biggan/nature/1.JPEG" in text
```

- [ ] **Step 2: Run the new test and confirm it fails**

```bash
uv run pytest tests/test_qwen_ft_modal_archive_export.py::test_write_relative_export_manifests_uses_protocol_relative_paths -v
```

Expected: FAIL because `write_relative_export_manifests` does not exist.

- [ ] **Step 3: Implement the relative manifest writer**

Add a small helper beside the existing GCS manifest writer:

```python
def _relative_export_path(item: ExportItem) -> str:
    split_dir = "eval-small" if item.split == "eval" else "train"
    return f"{split_dir}/{item.generator}/{item.source_label}/{item.source_path.name}"


def write_relative_export_manifests(plan: ExportPlan, manifest_dir: Path) -> dict[str, Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_path = manifest_dir / "train.csv"
    eval_path = manifest_dir / "eval_small.csv"
    _write_manifest_csv(train_path, [item.to_manifest_row(_relative_export_path(item)) for item in plan.train_items])
    _write_manifest_csv(eval_path, [item.to_manifest_row(_relative_export_path(item)) for item in plan.eval_items])
    return {"train": train_path, "eval_small": eval_path}
```

If `_write_manifest_csv` or `to_manifest_row` has a different current name, keep the existing code style and introduce the smallest equivalent helper.

- [ ] **Step 4: Run the focused test**

```bash
uv run pytest tests/test_qwen_ft_modal_archive_export.py::test_write_relative_export_manifests_uses_protocol_relative_paths -v
```

Expected: PASS.

- [ ] **Step 5: Write failing test for zip layout**

```python
import zipfile
from pathlib import Path

from aiforensics.finetune.genimage_export import build_export_plan, write_export_archive


def test_write_export_archive_contains_images_manifests_and_training_json(tmp_path: Path):
    data_root = tmp_path / "genimage"
    for source_label, payload in {"ai": b"fake", "nature": b"real"}.items():
        image = data_root / "imagenet_ai_0419_biggan" / "train" / source_label / f"{source_label}.JPEG"
        image.parent.mkdir(parents=True)
        image.write_bytes(payload)

    plan = build_export_plan(
        data_root=data_root,
        protocol="protocol-a-small",
        train_per_label=1,
        eval_per_label=0,
        seed=70,
    )
    archive = write_export_archive(plan, tmp_path / "qwen_ft_protocol_a_small.zip", tmp_path / "stage")

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())

    assert "protocol-a-small/manifests/train.csv" in names
    assert "protocol-a-small/qwen_train.jsonl" in names
    assert "protocol-a-small/train/imagenet_ai_0419_biggan/ai/ai.JPEG" in names
    assert "protocol-a-small/train/imagenet_ai_0419_biggan/nature/nature.JPEG" in names
```

- [ ] **Step 6: Implement `write_export_archive`**

Implement staging with `shutil.copy2`, `zipfile.ZipFile`, and the existing Qwen JSONL writer:

```python
def write_export_archive(plan: ExportPlan, archive_path: Path, staging_dir: Path) -> Path:
    import shutil
    import zipfile

    protocol_root = staging_dir / plan.protocol
    if protocol_root.exists():
        shutil.rmtree(protocol_root)
    (protocol_root / "manifests").mkdir(parents=True, exist_ok=True)

    for item in [*plan.train_items, *plan.eval_items]:
        destination = protocol_root / _relative_export_path(item)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source_path, destination)

    manifests = write_relative_export_manifests(plan, protocol_root / "manifests")
    from aiforensics.finetune.qwen_data import write_qwen_training_jsonl

    write_qwen_training_jsonl(manifests["train"], protocol_root / "qwen_train.jsonl")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(protocol_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging_dir))
    return archive_path
```

- [ ] **Step 7: Run archive export tests**

```bash
uv run pytest tests/test_qwen_ft_modal_archive_export.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit this task**

```bash
git add src/aiforensics/finetune/genimage_export.py tests/test_qwen_ft_modal_archive_export.py
git commit -m "feat: add modal qwen ft archive export"
```

---

### Task 2: Add Modal Path Rewriting For Qwen Training JSONL

**Files:**
- Modify: `src/aiforensics/finetune/qwen_data.py`
- Test: `tests/test_qwen_ft_modal_path_rewrite.py`

**Interfaces:**
- Consumes: `qwen_train.jsonl` with relative `image` values.
- Produces: `rewrite_qwen_jsonl_image_paths(input_jsonl: Path, output_jsonl: Path, image_root: Path) -> Path`

- [ ] **Step 1: Write failing path rewrite test**

```python
import json
from pathlib import Path

from aiforensics.finetune.qwen_data import rewrite_qwen_jsonl_image_paths


def test_rewrite_qwen_jsonl_image_paths_makes_relative_images_absolute(tmp_path: Path):
    image_root = tmp_path / "protocol-a-small"
    image = image_root / "train" / "gen" / "ai" / "0.JPEG"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")

    source = tmp_path / "qwen_train.jsonl"
    source.write_text(
        json.dumps({"image": "train/gen/ai/0.JPEG", "conversations": []}) + "\n",
        encoding="utf-8",
    )
    output = rewrite_qwen_jsonl_image_paths(source, tmp_path / "qwen_train_abs.jsonl", image_root)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["image"] == str(image)
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
uv run pytest tests/test_qwen_ft_modal_path_rewrite.py::test_rewrite_qwen_jsonl_image_paths_makes_relative_images_absolute -v
```

Expected: FAIL because the function does not exist.

- [ ] **Step 3: Implement rewrite helper**

```python
def rewrite_qwen_jsonl_image_paths(input_jsonl: Path, output_jsonl: Path, image_root: Path) -> Path:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with input_jsonl.open(encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            image_value = str(row["image"])
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = image_root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Training image not found: {image_path}")
            row["image"] = str(image_path)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_jsonl
```

- [ ] **Step 4: Run path rewrite tests**

```bash
uv run pytest tests/test_qwen_ft_modal_path_rewrite.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit this task**

```bash
git add src/aiforensics/finetune/qwen_data.py tests/test_qwen_ft_modal_path_rewrite.py
git commit -m "feat: rewrite qwen ft training paths for modal"
```

---

### Task 3: Add Kaggle Zip Export CLI And Notebook

**Files:**
- Modify: `scripts/export_genimage_qwen_ft_subset.py`
- Modify: `scripts/build_qwen_ft_notebooks.py`
- Create: `notebooks/kaggle_qwen_ft_modal_export.ipynb`
- Modify: `tests/test_qwen_ft_notebooks.py`

**Interfaces:**
- Consumes: Task 1 `write_export_archive`.
- Produces:
  - CLI flag `--archive-path PATH`
  - CLI flag `--path-mode {gcs,modal-volume}`
  - notebook section `AIF_SECTION: modal_zip_export`

- [ ] **Step 1: Add failing CLI behavior test**

```python
from pathlib import Path

from scripts.export_genimage_qwen_ft_subset import build_parser


def test_export_script_accepts_modal_archive_flags():
    args = build_parser().parse_args(
        [
            "--data-root",
            "/kaggle/input/genimage",
            "--protocol",
            "protocol-a-small",
            "--path-mode",
            "modal-volume",
            "--archive-path",
            "/kaggle/working/qwen_ft_protocol_a_small.zip",
        ]
    )

    assert args.path_mode == "modal-volume"
    assert args.archive_path == Path("/kaggle/working/qwen_ft_protocol_a_small.zip")
```

- [ ] **Step 2: Run the focused test**

```bash
uv run pytest tests/test_qwen_ft_notebooks.py::test_export_script_accepts_modal_archive_flags -v
```

Expected: FAIL until flags are added.

- [ ] **Step 3: Implement CLI flags**

In `build_parser()`, add:

```python
parser.add_argument("--path-mode", choices=["gcs", "modal-volume"], default="gcs")
parser.add_argument("--archive-path", type=Path)
```

In `main()`, route:

```python
if args.path_mode == "modal-volume":
    if args.archive_path is None:
        parser.error("--archive-path is required when --path-mode=modal-volume")
    archive = write_export_archive(plan, args.archive_path, args.work_dir / "modal_stage")
    print(f"archive_path={archive}")
    return 0
```

Keep the current GCS upload path unchanged.

- [ ] **Step 4: Add notebook generation test**

```python
import nbformat
from pathlib import Path


def test_modal_export_notebook_uses_archive_mode():
    notebook = nbformat.read(Path("notebooks/kaggle_qwen_ft_modal_export.ipynb"), as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "--path-mode modal-volume" in source
    assert "--archive-path" in source
    assert "gcloud storage" not in source
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in source
```

- [ ] **Step 5: Generate the notebook**

Add a new notebook builder entry that writes cells for:

```python
DATA_ROOT = Path("/kaggle/input/genimage")
ARCHIVE_PATH = Path("/kaggle/working/qwen_ft_protocol_a_small.zip")
```

and a shell cell:

```bash
python scripts/export_genimage_qwen_ft_subset.py \
  --data-root "$DATA_ROOT" \
  --protocol protocol-a-small \
  --train-per-label 100 \
  --eval-per-label 50 \
  --seed 70 \
  --path-mode modal-volume \
  --archive-path "$ARCHIVE_PATH"
```

- [ ] **Step 6: Run notebook builder and tests**

```bash
uv run python scripts/build_qwen_ft_notebooks.py
uv run pytest tests/test_qwen_ft_notebooks.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit this task**

```bash
git add scripts/export_genimage_qwen_ft_subset.py scripts/build_qwen_ft_notebooks.py notebooks/kaggle_qwen_ft_modal_export.ipynb tests/test_qwen_ft_notebooks.py
git commit -m "feat: add kaggle modal zip export notebook"
```

---

### Task 4: Add Modal Runtime Configs

**Files:**
- Create: `configs/qwen_ft_protocol_a_small_modal.yaml`
- Create: `configs/qwen_ft_protocol_a_full_modal.yaml`
- Test: `tests/test_qwen_ft_modal_configs.py`

**Interfaces:**
- Consumes: existing config loader and `qwen_ft` baseline config.
- Produces: Modal configs that point at `/vol/data`, `/vol/checkpoints`, and `/vol/eval`.

- [ ] **Step 1: Write config tests**

```python
from pathlib import Path

from aiforensics.config import load_config


def test_modal_eval_small_config_points_to_volume_paths():
    cfg = load_config(Path("configs/qwen_ft_protocol_a_small_modal.yaml"))

    assert str(cfg.paths.data_root) == "/vol/data/protocol-a-small"
    assert str(cfg.paths.manifest_root) == "/vol/data/protocol-a-small/manifests"
    assert str(cfg.paths.output_root) == "/vol/eval/protocol-a-small/eval-small"
    assert cfg.baselines.qwen_ft.adapter_uri == "/vol/checkpoints/protocol-a-small/final_adapter"
    assert cfg.datasets.genimage_unseen.manifest == Path("/vol/data/protocol-a-small/manifests/eval_small.csv")


def test_modal_eval_full_config_points_to_full_manifest():
    cfg = load_config(Path("configs/qwen_ft_protocol_a_full_modal.yaml"))

    assert str(cfg.paths.output_root) == "/vol/eval/protocol-a-small/eval-full"
    assert "eval_full" in str(cfg.datasets.genimage_unseen.manifest)
```

- [ ] **Step 2: Create eval-small Modal config**

Use this exact structure, adjusting only if current config models require extra fields:

```yaml
project:
  name: ai-image-forensics
  phase: qwen_ft_protocol_a_small_modal
  description: Modal eval-small for Qwen2.5-VL LoRA fine-tuned on GenImage protocol A.

paths:
  data_root: /vol/data/protocol-a-small
  manifest_root: /vol/data/protocol-a-small/manifests
  cache_root: /vol/cache/aiforensics-qwen-ft-a-small
  output_root: /vol/eval/protocol-a-small/eval-small
  external_root: /vol/external

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
    train_manifest: /vol/data/protocol-a-small/manifests/train.csv
    dev_manifest: /vol/data/protocol-a-small/manifests/eval_small.csv
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
    max_images: 700
    balance_labels: true
    split: external
    manifest: /vol/data/protocol-a-small/manifests/eval_small.csv
  synthbuster:
    enabled: false
    max_images: 0
    balance_labels: true
    split: external
    manifest: /vol/data/protocol-a-small/manifests/unused_synthbuster.csv

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
    checkpoint_path: /vol/external/npr/NPR.pth
    repo_dir: /vol/external/npr/NPR-DeepfakeDetection
    batch_size: 16
    allow_deferred: true
  qwen_ft:
    enabled: true
    model_id: Qwen/Qwen2.5-VL-7B-Instruct
    adapter_uri: /vol/checkpoints/protocol-a-small/final_adapter
    prompt_id: qwen_ft_label_json_v1
    temperature: 0.0
    max_new_tokens: 32
    cache_outputs: true
    allow_deferred: false
    dtype: float16
    output_fields: [label]
```

- [ ] **Step 3: Create eval-full config**

Copy eval-small config and change:

```yaml
project:
  phase: qwen_ft_protocol_a_full_modal
paths:
  output_root: /vol/eval/protocol-a-small/eval-full
datasets:
  genimage_unseen:
    max_images: 7000
    manifest: /vol/data/protocol-a-small/manifests/eval_full.csv
```

- [ ] **Step 4: Run config tests**

```bash
uv run pytest tests/test_qwen_ft_modal_configs.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit this task**

```bash
git add configs/qwen_ft_protocol_a_small_modal.yaml configs/qwen_ft_protocol_a_full_modal.yaml tests/test_qwen_ft_modal_configs.py
git commit -m "feat: add modal qwen ft configs"
```

---

### Task 5: Add Modal App For Unpack, Train, And Eval

**Files:**
- Create: `infra/modal/qwen_ft_modal.py`
- Test: `tests/test_qwen_ft_modal_app.py`

**Interfaces:**
- Consumes: Modal Volume `aiforensics-qwen-ft`, archive layout from Task 1, configs from Task 4.
- Produces Modal functions:
  - `inspect_volume() -> dict[str, object]`
  - `unpack_archive(archive_name: str = "qwen_ft_protocol_a_small.zip", protocol: str = "protocol-a-small") -> dict[str, object]`
  - `train_lora(protocol: str = "protocol-a-small", epochs: float = 1.0, learning_rate: float = 2e-4) -> dict[str, object]`
  - `eval_small(protocol: str = "protocol-a-small") -> dict[str, object]`
  - `eval_full(protocol: str = "protocol-a-small") -> dict[str, object]`

- [ ] **Step 1: Write static Modal app tests**

```python
from pathlib import Path


def test_modal_app_declares_expected_functions():
    source = Path("infra/modal/qwen_ft_modal.py").read_text(encoding="utf-8")

    assert 'modal.App("aiforensics-qwen-ft")' in source
    assert 'modal.Volume.from_name("aiforensics-qwen-ft")' in source
    for name in ["inspect_volume", "unpack_archive", "train_lora", "eval_small", "eval_full"]:
        assert f"def {name}(" in source


def test_modal_app_does_not_require_gcp_secret():
    source = Path("infra/modal/qwen_ft_modal.py").read_text(encoding="utf-8")

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in source
    assert "gcloud" not in source
    assert "huggingface-secret" in source
```

- [ ] **Step 2: Create Modal image and app skeleton**

Use this skeleton:

```python
from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import modal

APP_NAME = "aiforensics-qwen-ft"
VOLUME_NAME = "aiforensics-qwen-ft"
VOL_ROOT = Path("/vol")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "accelerate",
        "bitsandbytes",
        "datasets",
        "einops",
        "peft",
        "pillow",
        "qwen-vl-utils",
        "torch",
        "torchvision",
        "transformers",
    )
    .pip_install("-e", ".")
)
```

If Modal cannot install `-e .` from the local source in this form, replace it with Modal's supported local source mount pattern and run `python -m pip install -e /repo` inside the function before invoking the CLI.

- [ ] **Step 3: Implement `inspect_volume`**

```python
@app.function(image=image, volumes={"/vol": volume}, timeout=300)
def inspect_volume() -> dict[str, object]:
    roots = ["archives", "data", "checkpoints", "eval", "cache"]
    return {
        "volume": VOLUME_NAME,
        "exists": VOL_ROOT.exists(),
        "roots": {name: (VOL_ROOT / name).exists() for name in roots},
        "archives": sorted(p.name for p in (VOL_ROOT / "archives").glob("*.zip"))
        if (VOL_ROOT / "archives").exists()
        else [],
    }
```

- [ ] **Step 4: Implement `unpack_archive`**

```python
@app.function(image=image, volumes={"/vol": volume}, timeout=1800)
def unpack_archive(
    archive_name: str = "qwen_ft_protocol_a_small.zip",
    protocol: str = "protocol-a-small",
) -> dict[str, object]:
    archive_path = VOL_ROOT / "archives" / archive_name
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found in Modal Volume: {archive_path}")

    data_root = VOL_ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(data_root)

    protocol_root = data_root / protocol
    train_manifest = protocol_root / "manifests" / "train.csv"
    eval_manifest = protocol_root / "manifests" / "eval_small.csv"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Missing train manifest: {train_manifest}")
    if not eval_manifest.is_file():
        raise FileNotFoundError(f"Missing eval manifest: {eval_manifest}")

    from aiforensics.finetune.qwen_data import rewrite_qwen_jsonl_image_paths

    abs_jsonl = rewrite_qwen_jsonl_image_paths(
        protocol_root / "qwen_train.jsonl",
        protocol_root / "qwen_train_abs.jsonl",
        protocol_root,
    )
    volume.commit()
    return {
        "protocol_root": str(protocol_root),
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "training_jsonl": str(abs_jsonl),
    }
```

- [ ] **Step 5: Implement command runner**

```python
def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
```

- [ ] **Step 6: Implement `train_lora` with conservative defaults**

Start with a command wrapper that calls the existing training entrypoint or upstream Qwen fine-tuning command. Use the current `infra/qwen_ft/train_qwen_ft.py` if it can accept local JSONL; otherwise call the upstream script directly.

```python
@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100-40GB",
    timeout=60 * 60 * 8,
)
def train_lora(
    protocol: str = "protocol-a-small",
    epochs: float = 1.0,
    learning_rate: float = 2e-4,
) -> dict[str, object]:
    protocol_root = VOL_ROOT / "data" / protocol
    train_jsonl = protocol_root / "qwen_train_abs.jsonl"
    output_dir = VOL_ROOT / "checkpoints" / protocol
    final_adapter = output_dir / "final_adapter"
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Run unpack_archive first; missing {train_jsonl}")

    _run(
        [
            "python",
            "infra/qwen_ft/train_qwen_ft.py",
            "--model-id",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--train-jsonl",
            str(train_jsonl),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(epochs),
            "--learning-rate",
            str(learning_rate),
            "--save-steps",
            "100",
            "--save-total-limit",
            "3",
            "--method",
            "lora",
        ]
    )
    if not (final_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Expected final adapter at {final_adapter}")
    volume.commit()
    return {"adapter": str(final_adapter)}
```

If Modal rejects `A100-40GB` availability, the executor should try Modal's available A100 string for the workspace, then `H100` only after confirming cost/availability with the user.

- [ ] **Step 7: Implement eval functions**

```python
def _eval(config_path: str) -> dict[str, object]:
    _run(["aiforensics", "run", "--baseline", "qwen_ft", "--config", config_path])
    _run(["aiforensics", "evaluate", "--config", config_path])
    _run(["aiforensics", "report", "--config", config_path])
    return {"config": config_path, "status": "completed"}


@app.function(image=image, volumes={"/vol": volume}, gpu="A100-40GB", timeout=60 * 60 * 4)
def eval_small(protocol: str = "protocol-a-small") -> dict[str, object]:
    result = _eval("configs/qwen_ft_protocol_a_small_modal.yaml")
    volume.commit()
    return result


@app.function(image=image, volumes={"/vol": volume}, gpu="A100-40GB", timeout=60 * 60 * 12)
def eval_full(protocol: str = "protocol-a-small") -> dict[str, object]:
    result = _eval("configs/qwen_ft_protocol_a_full_modal.yaml")
    volume.commit()
    return result
```

- [ ] **Step 8: Run static tests**

```bash
uv run pytest tests/test_qwen_ft_modal_app.py -v
```

Expected: PASS.

- [ ] **Step 9: Run Modal smoke functions**

```bash
python3 -m modal run infra/modal/qwen_ft_modal.py::inspect_volume
```

Expected: returns the volume name and either an empty archive list or the uploaded zip.

- [ ] **Step 10: Commit this task**

```bash
git add infra/modal/qwen_ft_modal.py tests/test_qwen_ft_modal_app.py
git commit -m "feat: add modal qwen ft runner"
```

---

### Task 6: Make Training Entrypoint Work With Local Modal Data

**Files:**
- Modify: `infra/qwen_ft/train_qwen_ft.py`
- Modify: `infra/qwen_ft/requirements.txt`
- Modify: `pyproject.toml`
- Test: `tests/test_qwen_ft_training_entrypoint.py`

**Interfaces:**
- Consumes: `/vol/data/protocol-a-small/qwen_train_abs.jsonl`.
- Produces: a LoRA adapter directory containing `adapter_config.json`.

- [ ] **Step 1: Add parser test for local args**

```python
from infra.qwen_ft.train_qwen_ft import build_parser


def test_training_entrypoint_accepts_local_modal_args():
    args = build_parser().parse_args(
        [
            "--model-id",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--train-jsonl",
            "/vol/data/protocol-a-small/qwen_train_abs.jsonl",
            "--output-dir",
            "/vol/checkpoints/protocol-a-small",
            "--epochs",
            "1",
            "--learning-rate",
            "0.0002",
            "--method",
            "lora",
        ]
    )

    assert args.train_jsonl == "/vol/data/protocol-a-small/qwen_train_abs.jsonl"
    assert args.output_dir == "/vol/checkpoints/protocol-a-small"
    assert args.method == "lora"
```

- [ ] **Step 2: Run focused parser test**

```bash
uv run pytest tests/test_qwen_ft_training_entrypoint.py::test_training_entrypoint_accepts_local_modal_args -v
```

Expected: FAIL if the current parser is still GCS-only or uses different names.

- [ ] **Step 3: Add local-first CLI args**

Ensure `build_parser()` accepts:

```python
parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--train-jsonl", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--epochs", type=float, default=1.0)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
parser.add_argument("--save-steps", type=int, default=100)
parser.add_argument("--save-total-limit", type=int, default=3)
parser.add_argument("--local-work-dir", type=Path, default=Path("/tmp/qwen_ft"))
```

Keep existing GCS helpers only as compatibility helpers.

- [ ] **Step 4: Add command construction test**

```python
from pathlib import Path

from infra.qwen_ft.train_qwen_ft import build_upstream_command


def test_build_upstream_command_includes_lora_training_args(tmp_path: Path):
    command = build_upstream_command(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        train_jsonl=tmp_path / "train.jsonl",
        output_dir=tmp_path / "out",
        epochs=1.0,
        learning_rate=2e-4,
        method="lora",
        save_steps=100,
        save_total_limit=3,
    )

    text = " ".join(str(part) for part in command)
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in text
    assert str(tmp_path / "train.jsonl") in text
    assert "--lora_enable" in text or "lora" in text.lower()
```

- [ ] **Step 5: Implement local training command construction**

Update `build_upstream_command` so it receives local paths directly. The exact upstream command may differ based on the Qwen fine-tuning repo version, but it must include:

```python
[
    "accelerate",
    "launch",
    "qwen-vl-finetune/qwenvl/train/train_qwen.py",
    "--model_name_or_path",
    model_id,
    "--data_path",
    str(train_jsonl),
    "--output_dir",
    str(output_dir),
    "--num_train_epochs",
    str(epochs),
    "--learning_rate",
    str(learning_rate),
    "--save_steps",
    str(save_steps),
    "--save_total_limit",
    str(save_total_limit),
]
```

Add the LoRA flags required by the actual upstream Qwen script already used in this repo. For QLoRA, add the corresponding 4-bit/bitsandbytes flags only under `method == "qlora"`.

- [ ] **Step 6: Ensure final adapter copy exists**

After upstream training completes, normalize output:

```python
final_adapter = output_dir / "final_adapter"
final_adapter.mkdir(parents=True, exist_ok=True)
for filename in ["adapter_config.json", "adapter_model.safetensors"]:
    candidate = output_dir / filename
    if candidate.is_file():
        shutil.copy2(candidate, final_adapter / filename)
```

If upstream writes adapter files only inside the last checkpoint, copy from the numerically latest `checkpoint-*` directory.

- [ ] **Step 7: Run training entrypoint tests**

```bash
uv run pytest tests/test_qwen_ft_training_entrypoint.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit this task**

```bash
git add infra/qwen_ft/train_qwen_ft.py infra/qwen_ft/requirements.txt pyproject.toml tests/test_qwen_ft_training_entrypoint.py
git commit -m "feat: support local modal qwen ft training"
```

---

### Task 7: Fix `qwen_ft` Adapter Path Handling For Modal Manifests

**Files:**
- Modify: `src/aiforensics/baselines/qwen_ft/adapter.py`
- Test: `tests/test_qwen_ft_adapter.py`

**Interfaces:**
- Consumes: manifest image paths that may be absolute or relative to `config.paths.data_root`.
- Produces: robust local `Path` resolution before checksum and inference.

- [ ] **Step 1: Add failing relative path test**

```python
from pathlib import Path

from aiforensics.baselines.qwen_ft.adapter import QwenFTAdapter


def test_qwen_ft_adapter_resolves_relative_manifest_path_against_data_root(tmp_path: Path):
    adapter = QwenFTAdapter()
    image = tmp_path / "data" / "eval-small" / "gen" / "ai" / "0.JPEG"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")

    resolved = adapter._resolve_image_path(Path("eval-small/gen/ai/0.JPEG"), tmp_path / "data")

    assert resolved == image
```

- [ ] **Step 2: Run focused test**

```bash
uv run pytest tests/test_qwen_ft_adapter.py::test_qwen_ft_adapter_resolves_relative_manifest_path_against_data_root -v
```

Expected: FAIL because `_resolve_image_path` does not exist.

- [ ] **Step 3: Implement image path resolver**

```python
def _resolve_image_path(self, image_path: Path, data_root: Path) -> Path:
    if image_path.is_absolute():
        return image_path
    return data_root / image_path
```

Use this resolver anywhere the adapter currently does `record.path.exists()`, checksum validation, cache key creation, and `_generate_one_image`.

- [ ] **Step 4: Run adapter tests**

```bash
uv run pytest tests/test_qwen_ft_adapter.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit this task**

```bash
git add src/aiforensics/baselines/qwen_ft/adapter.py tests/test_qwen_ft_adapter.py
git commit -m "fix: resolve qwen ft modal manifest paths"
```

---

### Task 8: Add Modal Runbook And Mark Vertex As Legacy

**Files:**
- Create: `docs/runbook-qwen-finetune-modal.md`
- Modify: `docs/runbook-qwen-finetune-vertex.md`

**Interfaces:**
- Consumes: scripts/configs from Tasks 3-5.
- Produces: exact user commands for setup, upload, unpack, train, eval, and download results.

- [ ] **Step 1: Write Modal runbook**

Create `docs/runbook-qwen-finetune-modal.md` with these sections and commands:

```markdown
# Qwen Fine-Tuning On Modal

## One-Time Setup

```bash
python3 -m modal profile current
python3 -m modal volume list
python3 -m modal secret list
```

Expected:

- profile is `tnhatnguyen-dev2805`
- volume includes `aiforensics-qwen-ft`
- secret includes `huggingface-secret`

## Export Data On Kaggle

Open `notebooks/kaggle_qwen_ft_modal_export.ipynb`, run all cells, then save the notebook with output enabled and download:

```text
/kaggle/working/qwen_ft_protocol_a_small.zip
```

## Upload Zip To Modal Volume

```bash
python3 -m modal volume put aiforensics-qwen-ft \
  ~/aiforensics-modal-data/qwen_ft_protocol_a_small.zip \
  /archives/qwen_ft_protocol_a_small.zip
```

## Inspect And Unpack

```bash
python3 -m modal run infra/modal/qwen_ft_modal.py::inspect_volume
python3 -m modal run infra/modal/qwen_ft_modal.py::unpack_archive
```

## Train LoRA

```bash
python3 -m modal run infra/modal/qwen_ft_modal.py::train_lora
```

## Eval Small

```bash
python3 -m modal run infra/modal/qwen_ft_modal.py::eval_small
```

## Download Results

```bash
python3 -m modal volume get aiforensics-qwen-ft \
  /eval/protocol-a-small/eval-small \
  ~/aiforensics-modal-results/eval-small
```
```

- [ ] **Step 2: Mark Vertex runbook as fallback**

Add this notice at the top of `docs/runbook-qwen-finetune-vertex.md`:

```markdown
> Legacy fallback: the primary path is now Modal because the Vertex A100 quota
> request for `travel-agent-vllm` was denied. Use this runbook only if Vertex
> quota becomes available later.
```

- [ ] **Step 3: Commit docs**

```bash
git add docs/runbook-qwen-finetune-modal.md docs/runbook-qwen-finetune-vertex.md
git commit -m "docs: add modal qwen ft runbook"
```

---

### Task 9: Verification Gate

**Files:**
- No new files.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified implementation and a clear list of any environment-only blockers.

- [ ] **Step 1: Run focused Qwen FT tests**

```bash
uv run pytest \
  tests/test_qwen_ft_modal_archive_export.py \
  tests/test_qwen_ft_modal_path_rewrite.py \
  tests/test_qwen_ft_modal_configs.py \
  tests/test_qwen_ft_modal_app.py \
  tests/test_qwen_ft_adapter.py \
  tests/test_qwen_ft_notebooks.py \
  tests/test_qwen_ft_training_entrypoint.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run project lint**

```bash
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
```

Expected: PASS. If formatting fails, run `uv run --extra dev ruff format src tests` and repeat.

- [ ] **Step 3: Run full tests**

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 4: Run existing smoke pipeline**

```bash
uv run aiforensics prepare --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
uv run aiforensics evaluate --config configs/phase_ab_smoke.yaml
uv run aiforensics report --config configs/phase_ab_smoke.yaml
```

Expected: PASS and no regression to Phase A/B smoke.

- [ ] **Step 5: Run Modal inspect smoke**

```bash
python3 -m modal run infra/modal/qwen_ft_modal.py::inspect_volume
```

Expected: returns `volume: aiforensics-qwen-ft`.

- [ ] **Step 6: Commit verification note if docs changed**

If any docs or generated notebooks changed during verification:

```bash
git add docs notebooks
git commit -m "docs: refresh qwen ft modal instructions"
```

---

## Self-Review

- Spec coverage: the plan covers the Modal pivot, Kaggle zip export, Modal Volume layout, local path rewriting, training/eval commands, docs, and verification.
- Placeholder scan: no task uses `TBD`; the only conditional language is reserved for provider/runtime differences that the implementer must verify while running Modal.
- Type consistency: `write_export_archive`, `write_relative_export_manifests`, `rewrite_qwen_jsonl_image_paths`, and the Modal function names are defined before later tasks depend on them.
- Known risk: Modal GPU availability and exact upstream Qwen fine-tuning command may require a small adjustment during Task 5 or Task 6. Any such adjustment must be captured in tests/docs before proceeding.
