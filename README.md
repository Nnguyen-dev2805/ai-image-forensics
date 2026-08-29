# AI Image Forensics

This project explores explainable AI-generated image detection.

The core idea is that fake-image detection should not rely only on an MLLM's general visual semantics. A stronger system should separate three abilities:

1. train a visual forensic expert to detect real/fake signals,
2. align forensic features into an MLLM-readable token space,
3. train the MLLM to produce faithful natural-language explanations.

The first implementation focus is Phase A/B: build reproducible baselines for dataset protocol, CLIP linear probing, base MLLM detection, NPR, and classifier-assisted MLLM prompting. This baseline phase decides whether the later training stages are worth pursuing.
