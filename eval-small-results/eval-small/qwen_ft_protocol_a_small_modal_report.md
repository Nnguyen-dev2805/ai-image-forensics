# Phase A/B Baseline Report

## Project

- **Project:** ai-image-forensics
- **Phase:** qwen_ft_protocol_a_small_modal
- **Description:** Modal eval-small for Qwen2.5-VL LoRA fine-tuned on GenImage protocol A.
- **Output root:** eval-small

## Dataset Summary

Configured datasets:

| Dataset | Enabled | Evaluation split/source | Manifest |
| --- | --- | --- | --- |
| tiny_genimage | disabled | source=TheKernel01/Tiny-GenImage; evaluation split=dev (dev_manifest) | eval_small.csv |
| genimage_unseen | enabled | generators=imagenet_ai_0419_biggan,imagenet_ai_0419_vqdm,imagenet_ai_0424_sdv5,imagenet_ai_0424_wukong,imagenet_ai_0508_adm,imagenet_glide,imagenet_midjourney; split=external | eval_small.csv |
| synthbuster | disabled | split=external | unused_synthbuster.csv |

Observed evaluation coverage (from selected completed runs):

| Baseline | Seed | Total records | Source counts |
| --- | --- | --- | --- |
| qwen_ft | N/A | 700 | imagenet_ai_0419_biggan (n=100); imagenet_ai_0419_vqdm (n=100); imagenet_ai_0424_sdv5 (n=100); imagenet_ai_0424_wukong (n=100); imagenet_ai_0508_adm (n=100); imagenet_glide (n=100); imagenet_midjourney (n=100) |

## Baseline Status

One row per expected run slot; `missing` means no run artifact exists under the configured output_root and is not a successful outcome.

| Baseline | Seed | Configured | Status | Run ID | Reason |
| --- | --- | --- | --- | --- | --- |
| clip_probe | N/A | disabled | missing | N/A | N/A |
| qwen_vl | N/A | disabled | missing | N/A | N/A |
| assisted_qwen | N/A | disabled | missing | N/A | N/A |
| npr | N/A | disabled | missing | N/A | N/A |
| qwen_ft | N/A | enabled | completed | 20260906T031410114429Z_qwen_ft | N/A |

MLLM compute precision (affects numerical results):

| Baseline | Model | Compute dtype |
| --- | --- | --- |
| qwen_vl | Qwen/Qwen2.5-VL-7B-Instruct | bfloat16 |
| assisted_qwen | Qwen/Qwen2.5-VL-7B-Instruct | bfloat16 |
| qwen_ft | Qwen/Qwen2.5-VL-7B-Instruct | float16 |

MLLM prediction and parse statistics:

| Baseline | Total Images | Succeeded | Parsed (Strict) | Recovered | Failed | Success Rate |
| --- | --- | --- | --- | --- | --- | --- |
| qwen_ft | 700 | 700 | 700 | 0 | 0 | 100.00% |

## Overall Metrics

Values are rendered from artifacts produced by `aiforensics evaluate`; no metric is recomputed here.

| Baseline | accuracy | balanced_accuracy | precision | recall | f1 | auroc | Completion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_probe | N/A | N/A | N/A | N/A | N/A | N/A | 0/1 seeds completed |
| qwen_vl | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| assisted_qwen | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| npr | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| qwen_ft | 0.9671 | 0.9671 | 0.9455 | 0.9914 | 0.9679 | N/A | completed |

qwen_ft is label-only in this phase; AUROC is skipped unless a real fake score is available.

Confusion matrix artifacts (label order: real, fake):

| Baseline | Artifact |
| --- | --- |
| qwen_ft | confusion_matrix.json |

## Per-Source Metrics

| Baseline | Source | n | Accuracy | Balanced_accuracy | Precision | Recall | F1 | Auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_ft | imagenet_ai_0419_biggan | 100 | 0.9800 | 0.9800 | 0.9615 | 1.0000 | 0.9804 | N/A |
| qwen_ft | imagenet_ai_0419_vqdm | 100 | 0.9600 | 0.9600 | 0.9600 | 0.9600 | 0.9600 | N/A |
| qwen_ft | imagenet_ai_0424_sdv5 | 100 | 0.9600 | 0.9600 | 0.9259 | 1.0000 | 0.9615 | N/A |
| qwen_ft | imagenet_ai_0424_wukong | 100 | 0.9400 | 0.9400 | 0.8929 | 1.0000 | 0.9434 | N/A |
| qwen_ft | imagenet_ai_0508_adm | 100 | 0.9600 | 0.9600 | 0.9423 | 0.9800 | 0.9608 | N/A |
| qwen_ft | imagenet_glide | 100 | 0.9900 | 0.9900 | 0.9804 | 1.0000 | 0.9901 | N/A |
| qwen_ft | imagenet_midjourney | 100 | 0.9800 | 0.9800 | 0.9615 | 1.0000 | 0.9804 | N/A |

## Failure and Deferred Notes

- clip_probe (seed N/A): missing - No run artifact found under the configured output_root.
- qwen_vl (seed N/A): missing - No run artifact found under the configured output_root.
- assisted_qwen (seed N/A): missing - No run artifact found under the configured output_root.
- npr (seed N/A): missing - No run artifact found under the configured output_root.

## Next-Step Recommendation

Best observed baseline by balanced_accuracy: qwen_ft (0.9671).
No claim of statistical significance is made.

Classification improvement must never be interpreted as proof of explanation faithfulness.
