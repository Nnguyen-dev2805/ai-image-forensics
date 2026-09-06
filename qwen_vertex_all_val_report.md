# Phase A/B Baseline Report

## Project

- **Project:** ai-image-forensics
- **Phase:** qwen_vertex_all_val
- **Description:** Evaluation of Qwen-VL via Vertex AI across all GenImage val generators.
- **Output root:** outputs-qwen-all-val

## Dataset Summary

Configured datasets:

| Dataset | Enabled | Evaluation split/source | Manifest |
| --- | --- | --- | --- |
| tiny_genimage | disabled | source=TheKernel01/Tiny-GenImage; evaluation split=dev (dev_manifest) | manifests/tiny_genimage_dev.csv |
| genimage_unseen | enabled | generators=imagenet_ai_0419_biggan,imagenet_ai_0419_vqdm,imagenet_ai_0424_sdv5,imagenet_ai_0424_wukong,imagenet_ai_0508_adm,imagenet_glide,imagenet_midjourney; split=external | genimage_all_val.csv |
| synthbuster | disabled | split=external | manifests/synthbuster_external.csv |

Observed evaluation coverage (from selected completed runs):

| Baseline | Seed | Total records | Source counts |
| --- | --- | --- | --- |
| qwen_vl | N/A | 7000 | imagenet_ai_0419_biggan (n=1000); imagenet_ai_0419_vqdm (n=1000); imagenet_ai_0424_sdv5 (n=1000); imagenet_ai_0424_wukong (n=1000); imagenet_ai_0508_adm (n=1000); imagenet_glide (n=1000); imagenet_midjourney (n=1000) |

## Baseline Status

One row per expected run slot; `missing` means no run artifact exists under the configured output_root and is not a successful outcome.

| Baseline | Seed | Configured | Status | Run ID | Reason |
| --- | --- | --- | --- | --- | --- |
| clip_probe | N/A | disabled | missing | N/A | N/A |
| qwen_vl | N/A | enabled | completed | 20260905T081635098212Z_qwen_vl | N/A |
| assisted_qwen | N/A | disabled | missing | N/A | N/A |
| npr | N/A | disabled | missing | N/A | N/A |

MLLM compute precision (affects numerical results):

| Baseline | Model | Compute dtype |
| --- | --- | --- |
| qwen_vl | Qwen/Qwen2.5-VL-7B-Instruct | float16 |
| assisted_qwen | Qwen/Qwen2.5-VL-7B-Instruct | float16 |

MLLM prediction and parse statistics:

| Baseline | Total Images | Succeeded | Parsed (Strict) | Recovered | Failed | Success Rate |
| --- | --- | --- | --- | --- | --- | --- |
| qwen_vl | 7000 | 6972 | 6972 | 0 | 28 | 99.60% |

## Overall Metrics

Values are rendered from artifacts produced by `aiforensics evaluate`; no metric is recomputed here.

| Baseline | accuracy | balanced_accuracy | precision | recall | f1 | auroc | Completion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_probe | N/A | N/A | N/A | N/A | N/A | N/A | 0/1 seeds completed |
| qwen_vl | 0.5533 | 0.5533 | 0.8347 | 0.1429 | 0.2440 | 0.5571 | completed |
| assisted_qwen | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| npr | N/A | N/A | N/A | N/A | N/A | N/A | missing |

## Per-Source Metrics

| Baseline | Source | n | Accuracy | Balanced_accuracy | Precision | Recall | F1 | Auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_vl | imagenet_ai_0419_biggan | 1000 | 0.5470 | 0.5470 | 0.8228 | 0.1300 | 0.2245 | 0.5510 |
| qwen_vl | imagenet_ai_0419_vqdm | 1000 | 0.5490 | 0.5490 | 0.8293 | 0.1360 | 0.2337 | 0.5538 |
| qwen_vl | imagenet_ai_0424_sdv5 | 1000 | 0.5180 | 0.5180 | 0.7407 | 0.0800 | 0.1444 | 0.5257 |
| qwen_vl | imagenet_ai_0424_wukong | 1000 | 0.6170 | 0.6170 | 0.9220 | 0.2600 | 0.4056 | 0.6190 |
| qwen_vl | imagenet_ai_0508_adm | 1000 | 0.5240 | 0.5240 | 0.7288 | 0.0860 | 0.1538 | 0.5270 |
| qwen_vl | imagenet_glide | 1000 | 0.5730 | 0.5730 | 0.8800 | 0.1760 | 0.2933 | 0.5757 |
| qwen_vl | imagenet_midjourney | 1000 | 0.5450 | 0.5450 | 0.7857 | 0.1320 | 0.2260 | 0.5477 |

## Failure and Deferred Notes

- clip_probe (seed N/A): missing - No run artifact found under the configured output_root.
- assisted_qwen (seed N/A): missing - No run artifact found under the configured output_root.
- npr (seed N/A): missing - No run artifact found under the configured output_root.

## Explanation Samples

| Baseline | Sample ID | True | Predicted | Parse status | Explanation |
| --- | --- | --- | --- | --- | --- |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/002_biggan_00127 | fake | real | parsed | The image shows a realistic shark with natural lighting and water texture, indicating it is likely a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/002_biggan_00143 | fake | real | parsed | The image shows a clear underwater scene with a fish, which appears to be a real photograph due to the natural lighting and texture of the water and the fish's scales. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/003_biggan_00035 | fake | real | parsed | The image shows a natural underwater scene with a shark, which appears to be a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/006_biggan_00035 | fake | real | parsed | The image shows a natural underwater scene with clear details of the sand, water, and a fish, which appears to be a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/006_biggan_00039 | fake | real | parsed | The image shows a natural scene with clear details of a stingray in water, which appears consistent with a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/008_biggan_00039 | fake | real | parsed | The image shows a natural texture and lighting consistent with a real photograph of a chicken. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/008_biggan_00127 | fake | fake | parsed | The image exhibits unnatural coloration and texture, particularly around the eyes and mouth area, which is inconsistent with a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/011_biggan_00035 | fake | real | parsed | The image shows a bird with natural feather patterns and a realistic background, suggesting it is a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/011_biggan_00074 | fake | real | parsed | The image shows a bird with natural lighting and shadows, which are consistent with a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/014_biggan_00035 | fake | real | parsed | The image shows a bird with natural coloration and texture, which is consistent with a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/014_biggan_00143 | fake | real | parsed | The image shows a natural scene with a bird and greenery, which appears to be captured by a real camera. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/016_biggan_00020 | fake | real | parsed | The image shows a bird with natural textures and colors, typical of a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/016_biggan_00127 | fake | real | parsed | The image shows a bird with natural plumage and realistic details such as feathers, beak, and eye, which are consistent with a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/016_biggan_00143 | fake | real | parsed | The image shows a bird on grass with natural lighting and shadows, which appear consistent with a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/018_biggan_00020 | fake | real | parsed | The bird appears to have natural feathers and a realistic texture, with no signs of artificial enhancement or distortion. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/020_biggan_00143 | fake | real | parsed | The image shows a natural scene with a bird on a rock in a flowing stream, which appears to be captured by a real camera. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/023_biggan_00143 | fake | real | parsed | The image shows a bird perched on a branch with natural lighting and shadows, which appear consistent with a real photograph. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/024_biggan_00127 | fake | real | parsed | The image shows a detailed texture consistent with a real owl, with natural lighting and shadows that suggest a real-world setting. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/031_biggan_00143 | fake | real | parsed | The image shows a natural leaf with a small insect, which appears to be a real photograph due to the realistic texture and lighting. |
| qwen_vl | genimage/val/imagenet_ai_0419_biggan/fake/033_biggan_00035 | fake | real | parsed | The image shows a clear, detailed view of a sea turtle in natural water, with realistic lighting and shadows that suggest it was taken by a camera. |

## Next-Step Recommendation

Best observed baseline by balanced_accuracy: qwen_vl (0.5533).
No claim of statistical significance is made.

Classification improvement must never be interpreted as proof of explanation faithfulness.
