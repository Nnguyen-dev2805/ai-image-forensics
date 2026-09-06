# Qwen2.5-VL Fine-Tuning Execution Notes

## Goal

Fine-tune Qwen2.5-VL-7B-Instruct for GenImage real/fake detection, then
compare it against the current zero-shot Vertex Qwen baseline on the same
GenImage validation manifest.

Current baseline:

- Experiment: `qwen_vertex_all_val`
- Evaluation set: 7 GenImage validation generators, 1000 images each
- Zero-shot Qwen balanced accuracy: 0.5533
- Zero-shot Qwen fake recall: 0.1429

## Recommended Training Strategy

Start with LoRA or QLoRA, not full fine-tuning.

Reasons:

- The main failure mode is not parsing or serving; it is the decision boundary:
  Qwen predicts too many fake images as real.
- Parameter-efficient fine-tuning is much cheaper to iterate than full
  fine-tuning.
- It reduces the risk of damaging the base MLLM's general visual-language
  behavior before the detection protocol is proven.

The official Qwen VL fine-tuning framework supports:

- image/video conversation-style annotations
- dataset registration through `data/__init__.py`
- component-level tuning flags such as `tune_mm_llm`, `tune_mm_vision`, and
  `tune_mm_mlp`
- LoRA options such as `lora_enable`, `lora_r`, `lora_alpha`, and
  `lora_dropout`
- training-resolution controls through `min_pixels` and `max_pixels`

Primary source:

- https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune

## Data Format

Training records should follow Qwen VL's image conversation format:

```json
{
  "image": "/absolute/or/data-root-relative/path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nYou are an image-forensics classifier. Classify the image as real or fake. Return exactly one JSON object with keys: label, confidence, evidence."
    },
    {
      "from": "gpt",
      "value": "{\"label\":\"fake\",\"confidence\":0.95,\"evidence\":\"The image is labeled as AI-generated in the GenImage training split.\"}"
    }
  ]
}
```

For the first training stage, the `evidence` field should be treated as a
controlled placeholder, not as faithful forensic ground truth. The important
supervision is the label and the model's habit of returning valid JSON.

## Evaluation Protocol

Use three levels:

1. Quick sanity:
   - Train on a small balanced subset from GenImage train.
   - Evaluate on the existing 7000-image GenImage val manifest.
   - Purpose: prove training, inference, parsing, and reporting work.

2. Main seen/unseen protocol:
   - Train seen generators: BigGAN, VQDM, ADM, GLIDE.
   - Evaluate seen generators: BigGAN, VQDM, ADM, GLIDE val.
   - Evaluate unseen generators: SD v1.5, Wukong, Midjourney val.
   - Purpose: distinguish generator memorization from generalization.

3. Optional leave-one-generator-out:
   - Train on six generators and evaluate on the held-out generator.
   - Purpose: map generator transfer behavior.

Never train on the validation manifest used for the current Qwen report.

## Recommended Execution Environment

Prefer Vertex AI CustomJob for real fine-tuning. Use Kaggle only for quick
dataset conversion or smoke tests.

Why Vertex CustomJob:

- The job continues without depending on an interactive notebook tab.
- The training container, machine type, accelerator, and disk can be declared.
- Output can be written to a Cloud Storage base output directory.
- Logs can be monitored separately from the notebook.

Google Cloud documentation states that CustomJob is the basic way to run custom
ML training code, and that the model output directory can be a Cloud Storage URI
passed through the `baseOutputDirectory` API field.

Primary source:

- https://cloud.google.com/vertex-ai/docs/training/create-custom-job

## Checkpointing and Resume

Use frequent step checkpoints:

- `save_strategy="steps"`
- `save_steps` small enough for the expected runtime, such as 100-500 for early
  tests
- `save_total_limit=2` or `3`
- persist checkpoints to GCS after each save or set the training output
  directory to a mounted/synced durable location

Hugging Face Trainer supports resuming with `resume_from_checkpoint=True` or a
specific checkpoint path. This is essential for preemptions or failed jobs.

Primary source:

- https://huggingface.co/docs/transformers/v5.2.0/trainer

## Evaluation Integration With This Repo

The existing repo evaluation expects each model run to emit:

- `predictions.jsonl`
- `metrics.json`
- `metrics_by_source.csv`
- `logs.txt`
- `status.json`
- `run_scope.json`

For the first fine-tuned Qwen experiment, the cleanest integration path is:

1. Train a LoRA adapter and save it under a durable artifact directory.
2. Run inference with base Qwen2.5-VL plus the LoRA adapter.
3. Emit the same prediction schema as `qwen_vl`.
4. Run existing `aiforensics evaluate` and `aiforensics report`.

If the fine-tuned model is later deployed behind a Vertex OpenAI-compatible
endpoint, the existing `vertex_openai` provider can be reused by pointing the
config to the new endpoint/model id. If evaluating the adapter locally, the repo
will need a small adapter-loading path for PEFT/LoRA.

## Success Criteria

The fine-tuning run should be considered worth continuing only if:

- balanced accuracy improves from 0.5533 to at least about 0.65 on the selected
  validation protocol
- fake recall improves from 0.1429 to at least about 0.45
- AUROC improves, not only hard-label accuracy
- parse failure remains below 1-2%
- unseen generator performance does not collapse

If performance improves only on seen generators but not unseen generators, the
model likely learned generator-specific shortcuts rather than general forensic
evidence.
