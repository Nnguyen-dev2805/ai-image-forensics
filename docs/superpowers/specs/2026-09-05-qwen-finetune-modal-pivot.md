# Qwen Fine-Tuning Modal Pivot Spec

## Purpose

Pivot the existing `qwen_ft` fine-tuning implementation from the rejected
Vertex A100 path to a Modal-first path. The classification goal is unchanged:
fine-tune `Qwen/Qwen2.5-VL-7B-Instruct` with LoRA for GenImage `real`/`fake`
classification, then evaluate with the existing Phase A/B artifact/report
pipeline.

## Updated Decisions

- Platform is Modal, not Vertex AI CustomJob, for this execution path.
- Durable training/evaluation storage is Modal Volume `aiforensics-qwen-ft`.
- Kaggle exports a zip archive containing only the selected GenImage subset.
- The user downloads the Kaggle zip locally and uploads it to the Modal Volume.
- No GCS bucket, GCP service-account secret, Vertex endpoint, or Vertex quota is
  required for the Modal path.
- Keep the existing Vertex/GCS files as a legacy fallback, but do not make the
  primary notebook/runbook depend on them.
- Modal still needs a Hugging Face token secret to download gated or rate-limited
  model assets.

## Unchanged Fine-Tuning Protocol

- Baseline name for reports: `qwen_ft`.
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`.
- Model size fallback: none for the real experiment.
- Training method: LoRA first; QLoRA only if LoRA runs out of memory.
- Training output format is label-only JSON:

```json
{"label":"real"}
```

or:

```json
{"label":"fake"}
```

- Do not train `confidence`.
- Do not train `evidence`.
- Protocol A first: train all seven generators and evaluate all seven validation
  generators.
- Protocol B after: train seen generators and evaluate unseen generators.
- First training size: 100 real and 100 fake images per generator.
- Eval-small: 50 real and 50 fake images per generator.
- Eval-full: 7000 GenImage validation images.

## Modal Volume Layout

The Modal Volume must use this layout:

```text
/vol/
  archives/
    qwen_ft_protocol_a_small.zip
  data/
    protocol-a-small/
      train/
      eval-small/
      manifests/
        train.csv
        eval_small.csv
      qwen_train.jsonl
      qwen_train_abs.jsonl
  checkpoints/
    protocol-a-small/
      checkpoint-*/
      final_adapter/
  eval/
    protocol-a-small/
      eval-small/
        <aiforensics run/evaluate/report artifacts>
      eval-full/
        <aiforensics run/evaluate/report artifacts>
```

Manifest image paths inside the archive should be relative paths rooted at the
protocol directory, such as:

```text
train/imagenet_ai_0419_biggan/ai/example.JPEG
eval-small/imagenet_ai_0419_biggan/nature/example.JPEG
```

Modal unpacking may generate `qwen_train_abs.jsonl` with absolute paths because
many Qwen training scripts expect local filesystem image paths.

## Success Criteria

- A Kaggle notebook can create `qwen_ft_protocol_a_small.zip` without GCS.
- The zip can be uploaded to Modal Volume `aiforensics-qwen-ft`.
- A Modal smoke command can unpack the archive and verify manifests/images.
- A Modal training command can produce a LoRA adapter under
  `/vol/checkpoints/protocol-a-small/final_adapter`.
- A Modal eval-small command can run `qwen_ft`, `evaluate`, and `report` using
  the existing CLI and produce `predictions.jsonl`, `metrics.json`,
  `metrics_by_source.csv`, and a Markdown report under `/vol/eval`.
- The local smoke test path still runs without Qwen weights.
