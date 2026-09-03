import json
from typing import Literal

from pydantic import BaseModel


class QwenParseResult(BaseModel):
    label_pred: Literal["real", "fake", "unknown"]
    score_fake: float | None
    explanation: str
    parse_status: Literal["parsed", "recovered", "failed"]


def _map_confidence(label: str, confidence: float) -> float:
    if label == "fake":
        return confidence
    return 1.0 - confidence


def _fail() -> QwenParseResult:
    return QwenParseResult(
        label_pred="unknown",
        score_fake=None,
        explanation="",
        parse_status="failed",
    )


def parse_qwen_output(raw_output: str) -> QwenParseResult:
    def _validate(
        data: dict, base_status: Literal["parsed", "recovered"]
    ) -> QwenParseResult | None:
        if "label" not in data or "confidence" not in data or "evidence" not in data:
            return None

        label_raw = data["label"]
        if not isinstance(label_raw, str):
            return None

        label = label_raw.strip().lower()
        if label not in ("real", "fake"):
            return None

        # Check if label needed normalization
        status = base_status
        if base_status == "parsed" and label_raw not in ("real", "fake"):
            status = "recovered"

        confidence = data["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None
        if not (0.0 <= confidence <= 1.0):
            return None

        evidence = data["evidence"]
        if not isinstance(evidence, str):
            return None
        evidence = evidence.strip()
        if not evidence:
            return None

        return QwenParseResult(
            label_pred=label,
            score_fake=_map_confidence(label, float(confidence)),
            explanation=evidence,
            parse_status=status,
        )

    # 1. Strict parse
    try:
        data = json.loads(raw_output.strip())
        if isinstance(data, dict):
            res = _validate(data, "parsed")
            if res:
                return res
    except json.JSONDecodeError:
        pass

    # 2. Extract ALL valid JSON dicts that pass schema from anywhere in the text
    decoder = json.JSONDecoder()
    valid_results = []

    idx = 0
    while idx < len(raw_output):
        idx = raw_output.find("{", idx)
        if idx == -1:
            break
        try:
            data, end_idx = decoder.raw_decode(raw_output[idx:])
            if isinstance(data, dict):
                res = _validate(data, "recovered")
                if res:
                    valid_results.append(res)
            idx += end_idx
        except json.JSONDecodeError:
            idx += 1

    if len(valid_results) == 1:
        return valid_results[0]

    return _fail()
