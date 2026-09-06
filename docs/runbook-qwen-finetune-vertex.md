# qwen_ft Fine-Tuning on Vertex AI — Operator Runbook

> Legacy fallback: the primary path is now Modal because the Vertex A100 quota
> request for `travel-agent-vllm` was denied. Use this runbook only if Vertex
> quota becomes available later.

This runbook operates the `qwen_ft` pipeline: LoRA fine-tuning of
`Qwen/Qwen2.5-VL-7B-Instruct` on GenImage real/fake labels, evaluated with the
existing Phase A/B metrics and compared against the zero-shot Qwen baseline.

Design spec: `docs/superpowers/specs/2026-09-05-qwen-finetune-design.md`.

Fixed decisions (do not change per run): baseline name `qwen_ft`, model 7B (no
3B fallback), LoRA first (QLoRA only after a proven OOM), label-only training
target `{"label":"real"}` / `{"label":"fake"}`, A100 40GB accelerator,
training as a Vertex AI CustomJob — never inside Kaggle or Colab.

Default bucket: `gs://aiforensics-qwen-ft-579187260419`.

## 1. Create GCS bucket

```bash
gcloud storage buckets create gs://aiforensics-qwen-ft-579187260419 \
    --location=asia-southeast1 --uniform-bucket-level-access
```

The Vertex CustomJob service agent needs write access to the bucket (the
project-level Vertex AI service agent usually already has it).

## 2. Run the Kaggle export notebook

Open `notebooks/kaggle_qwen_ft_export.ipynb` on Kaggle:

1. Attach the GenImage dataset (layout
   `<DATA_ROOT>/<generator>/<train|val>/{ai,nature}/*`).
2. Add the Kaggle Secret `GOOGLE_APPLICATION_CREDENTIALS` containing the
   service-account JSON.
3. Run the cells in order; section 4 dry-runs the export, section 5 uploads
   with `--upload`.

Protocol A defaults: `--train-per-label 100`, `--eval-per-label 50`, seed 70,
7 generators.

## 3. Verify GCS artifacts

```bash
gcloud storage ls gs://aiforensics-qwen-ft-579187260419/data/protocol-a-small/
# expect:
#   data/protocol-a-small/train/<generator>/{ai,nature}/*.png
#   data/protocol-a-small/eval-small/<generator>/{ai,nature}/*.png
#   data/protocol-a-small/manifests/train.csv
#   data/protocol-a-small/manifests/eval_small.csv
#   data/protocol-a-small/qwen_train.jsonl
```

Spot-check counts: 1400 train rows and 700 eval rows in the CSVs, and the same
number of lines in `qwen_train.jsonl`.

## 4. Build/push the training container

```bash
cd infra/qwen_ft
docker build -t asia-southeast1-docker.pkg.dev/579187260419/aiforensics/qwen-ft-trainer:latest .
docker push asia-southeast1-docker.pkg.dev/579187260419/aiforensics/qwen-ft-trainer:latest
```

Create the Artifact Registry repository first if it does not exist:

```bash
gcloud artifacts repositories create aiforensics \
    --repository-format=docker --location=asia-southeast1
```

Record the installed package versions from the first training log and pin
`requirements.txt` and the upstream Qwen2.5-VL clone commit in a follow-up
commit.

## 5. Dry-run the Vertex job

```bash
uv run python scripts/submit_qwen_ft_vertex_job.py \
    --project-id 579187260419 \
    --location asia-southeast1 \
    --protocol protocol-a-small \
    --dry-run
```

Check: `NVIDIA_TESLA_A100` count 1, output prefix
`gs://aiforensics-qwen-ft-579187260419/checkpoints/protocol-a-small`,
`--save_steps=100`, `--save_total_limit=3`, LoRA flags
(`--tune_mm_vision=False`).

## 6. Submit the Vertex job

Remove `--dry-run` from the command above. The job reads the training JSONL
from GCS and writes checkpoints back to GCS.

## 7. Monitor logs and checkpoints

```bash
gcloud ai custom-jobs list --region=asia-southeast1
gcloud ai custom-jobs tail-logs <JOB_NAME> --region=asia-southeast1
gcloud storage ls gs://aiforensics-qwen-ft-579187260419/checkpoints/protocol-a-small/
```

The run is usable for evaluation once
`checkpoints/protocol-a-small/final_adapter/adapter_config.json` exists. A job
that stops early resumes from the latest checkpoint when resubmitted (Trainer
resume).

## 8. Run eval-small

On a GPU-capable runtime (Kaggle/Colab GPU or a GPU VM) with the repository
installed and `GOOGLE_APPLICATION_CREDENTIALS` set:

```bash
aiforensics run --baseline qwen_ft --config configs/qwen_ft_protocol_a_small.yaml
aiforensics evaluate --config configs/qwen_ft_protocol_a_small.yaml
```

The adapter downloads `final_adapter` from GCS into the configured cache root
on first use. Eval-small is the 700-image manifest shipped by the export.

## 9. Run eval-full

Only after the parse-failure gate passes (< 2% failed parses on eval-small;
the submit-and-eval notebook enforces this in section 7). Point the config's
evaluation manifest at the 7000-image all-validation CSV (produced by the
`kaggle_vertex_qwen_all_val` flow), relocate `output_root`/`cache_root`, and
re-run `run --baseline qwen_ft`, `evaluate`. The
`vertex_qwen_ft_submit_and_eval.ipynb` notebook automates this.

## 10. Compare against zero-shot Qwen

```bash
aiforensics report --config configs/qwen_ft_protocol_a_small.yaml
```

Success criteria from the design spec:

- balanced accuracy improves from 0.5533 to at least 0.65
- fake recall improves from 0.1429 to at least 0.45
- parse failure stays below 2%
- Protocol B unseen-generator performance does not collapse relative to the
  zero-shot baseline

AUROC is skipped for `qwen_ft`: label-only outputs carry `score_fake: null`.

Protocol B (`configs/qwen_ft_protocol_b_seen_unseen.yaml`) starts only after
Protocol A has a valid full report. Its export uses the same script with
`--protocol protocol-b-seen-unseen` (4 seen training generators, 3 unseen
evaluation generators).

## 11. Save artifacts

A run without saved artifacts is not a valid experiment result. Preserve for
every experiment:

- LoRA adapter (`final_adapter/`) and checkpoints (GCS)
- `vertex_custom_job.json` payload and training config (GCS `jobs/`)
- training and inference logs (GCS and run directories)
- locally: `predictions.jsonl`, `metrics.json`, `metrics_by_source.csv`,
  `confusion_matrix.json`, the Markdown report, and `status.json` under
  `outputs/qwen-ft-protocol-a-small/`

Risk controls to keep in force: never train on validation images, never select
training examples using validation metrics, never write service-account JSON,
datasets, or checkpoints into git, and always record the run scope so foreign
artifacts cannot enter a report.
