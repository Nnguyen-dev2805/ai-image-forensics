import math


def get_assisted_prompt(
    prompt_id: str,
    *,
    classifier_pred: str,
    fake_probability: float,
) -> str:
    if prompt_id != "assisted_qwen_json_v1":
        raise ValueError(f"Unsupported prompt_id: {prompt_id}")

    if classifier_pred not in ("real", "fake"):
        raise ValueError(f"Invalid classifier_pred: {classifier_pred}")

    if not math.isfinite(fake_probability) or not (0.0 <= fake_probability <= 1.0):
        raise ValueError(f"Invalid fake_probability: {fake_probability}")

    fake_probability_text = format(fake_probability, ".12g")

    return f"""You are an image-forensics classifier.

Classify the provided image as either "real" or "fake".

Definitions:
- "real": a natural camera photograph.
- "fake": an AI-generated or synthetic image.

You are also given an auxiliary classifier signal:
{{
  "classifier_pred": "{classifier_pred}",
  "fake_probability": {fake_probability_text}
}}

Use the auxiliary classifier signal as supporting evidence, but make the final decision from the
image and the signal together. The auxiliary classifier may be wrong.

Return exactly one JSON object and no Markdown or surrounding prose:
{{
  "label": "real" or "fake",
  "confidence": a number from 0.0 to 1.0 representing confidence in the chosen label,
  "evidence": "a concise image-based reason for the decision"
}}

Use only evidence visible in the image plus the provided auxiliary classifier signal. Do not claim
access to metadata, hidden detectors, retrieval results, tools, training labels, or scores that were
not provided."""
