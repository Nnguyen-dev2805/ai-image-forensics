# Qwen Fine-Tuning Design Spec

## Purpose

Fine-tune `Qwen/Qwen2.5-VL-7B-Instruct` for GenImage `real`/`fake`
classification and compare the result against the current zero-shot Qwen
baseline. This stage focuses only on classification quality. It does not train
or evaluate explanation faithfulness.

## Baseline To Beat

The frozen baseline is the current `qwen_vertex_all_val` result:

- Evaluation set: GenImage validation split, seven generators, 1000 images per
  generator.
- Total evaluation size: 7000 images.
- Baseline model: zero-shot Qwen2.5-VL-7B-Instruct served through Vertex.
- Balanced accuracy: `0.5533`.
- Fake recall: `0.1429`.
- AUROC: `0.5571`, retained only as a zero-shot reference because the first
  fine-tuning stage is label-only.

The key failure mode to fix is false negatives: zero-shot Qwen predicts many
fake images as real.

## Final Decisions

- Baseline name for reports: `qwen_ft`.
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`.
- Model size fallback: none. The first real experiment must use 7B.
- Training method: LoRA first; QLoRA only if LoRA runs out of memory.
- Full fine-tuning: excluded from this stage.
- Platform: Google Vertex AI CustomJob.
- Accelerator target: A100 40GB.
- Durable storage: a new Google Cloud Storage bucket dedicated to this project.
- Default bucket name: `gs://aiforensics-qwen-ft-579187260419`.
- Training execution: not inside Kaggle or Colab. A notebook may submit or
  monitor jobs, but training itself runs as a Vertex job.
- Data ingress: Kaggle notebook filters the needed GenImage subsets and uploads
  them to GCS. Vertex jobs read only from GCS.
- Training output format: label-only JSON.
- Evidence/explanation: not trained in this stage.
- Confidence: not trained in this stage.
- Evaluation: subset first, then full 7000-image validation manifest.
- Checkpointing: save every 100 steps for small runs, keep the latest three
  checkpoints, write outputs to GCS, and support resume from checkpoint.

## Training Target

Each training answer is exactly one JSON object:

```json
{"label":"real"}
```

or:

```json
{"label":"fake"}
```

No `confidence` field is used because a hard-coded confidence target would teach
the model to emit artificial certainty. No `evidence` field is used because the
current dataset labels do not provide faithful forensic explanations.

## Protocol A: All Generators To All Validation

This is the first protocol.

Training data:

- Generators: `imagenet_ai_0419_biggan`, `imagenet_ai_0419_vqdm`,
  `imagenet_ai_0424_sdv5`, `imagenet_ai_0424_wukong`, `imagenet_ai_0508_adm`,
  `imagenet_glide`, `imagenet_midjourney`.
- Split: GenImage `train`.
- Size: 100 real and 100 fake images per generator.
- Total size: about 1400 training images.

Evaluation data:

- First pass: 50 real and 50 fake images per generator, about 700 validation
  images.
- Final pass: all 7000 validation images from the current all-val manifest.

Purpose:

- Verify that fine-tuning improves classification over zero-shot Qwen on the
  same validation distribution.
- Catch output-format, parser, and inference-cost problems before running the
  full validation set.

## Protocol B: Seen And Unseen Generators

This is the second protocol.

Seen training generators:

- `imagenet_ai_0419_biggan`
- `imagenet_ai_0419_vqdm`
- `imagenet_ai_0508_adm`
- `imagenet_glide`

Unseen evaluation generators:

- `imagenet_ai_0424_sdv5`
- `imagenet_ai_0424_wukong`
- `imagenet_midjourney`

Purpose:

- Test whether fine-tuning learns a general real/fake signal or only memorizes
  generator-specific artifacts.
- Compare seen-generator gains against unseen-generator gains.

## Metrics

Primary metrics:

- balanced accuracy
- fake recall
- precision
- F1
- accuracy
- per-generator metrics
- confusion matrix
- parse failure rate

Secondary metrics:

- AUROC only when a meaningful `score_fake` exists. For label-only `qwen_ft`
  predictions, `score_fake` is `null` and AUROC is skipped.

Success criteria:

- Balanced accuracy improves from `0.5533` to at least `0.65`.
- Fake recall improves from `0.1429` to at least `0.45`.
- Parse failure remains below `2%`.
- Protocol B unseen-generator performance does not collapse relative to the
  zero-shot baseline.
- A run without saved artifacts is not counted as a valid experiment result.

## Artifact Contract

Every valid `qwen_ft` training experiment must preserve:

- LoRA adapter or QLoRA adapter.
- Training configuration YAML.
- Vertex job specification JSON.
- Training manifest CSV.
- Fine-tuning conversation JSON.
- Evaluation manifest CSV.
- `predictions.jsonl`.
- `metrics.json`.
- `metrics_by_source.csv`.
- Confusion matrix artifact.
- Markdown report.
- Training logs.
- Inference logs.
- Run status JSON.

GCS layout:

```text
gs://aiforensics-qwen-ft-579187260419/
  data/
    protocol-a-small/
      train/
      eval-small/
      eval-full/
      manifests/
      qwen_train.jsonl
  jobs/
    protocol-a-small/
      vertex_custom_job.json
      training_config.yaml
  checkpoints/
    protocol-a-small/
      checkpoint-*/
      final_adapter/
  eval/
    protocol-a-small/
      qwen_ft_eval_small/
      qwen_ft_eval_full/
```

Local output layout follows the existing project convention under a configured
`output_root`.

## Execution Flow

1. Kaggle notebook verifies GenImage train/val layout.
2. Kaggle notebook builds balanced train and eval-small manifests.
3. Kaggle notebook uploads selected images, manifests, and label-only Qwen
   training JSON to GCS.
4. Local CLI or notebook submits a Vertex CustomJob.
5. Vertex CustomJob fine-tunes LoRA on Qwen2.5-VL-7B-Instruct.
6. Vertex job writes checkpoints and final adapter to GCS.
7. Inference runner loads base Qwen plus the adapter and emits
   `qwen_ft` predictions.
8. Existing `aiforensics evaluate` and `aiforensics report` produce metrics and
   the comparison report.
9. If eval-small passes parser and artifact checks, run eval-full.

## Risk Controls

- Do not train on validation images.
- Do not use validation metrics to choose examples inside the training subset.
- Do not hard-code Kaggle paths in package code.
- Do not write service account JSON, private keys, access tokens, datasets,
  checkpoints, caches, or generated outputs into git.
- Do not rely on a notebook runtime for training completion.
- Record run scope fingerprints so old or foreign run artifacts do not enter a
  new report.
- Treat label-only outputs as classification outputs only; do not report
  explanation quality.

## References

- Qwen2.5-VL fine-tuning framework:
  https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune
- Google Vertex AI CustomJob:
  https://cloud.google.com/vertex-ai/docs/training/create-custom-job
- Hugging Face PEFT:
  https://huggingface.co/docs/peft/index
- Hugging Face Trainer checkpoint/resume behavior:
  https://huggingface.co/docs/transformers/trainer
- Existing project architecture:
  `docs/architecture/phase-ab-architecture.md`
- Existing manifest schema:
  `docs/schemas/manifest.md`
- Existing prediction schema:
  `docs/schemas/predictions-jsonl.md`
