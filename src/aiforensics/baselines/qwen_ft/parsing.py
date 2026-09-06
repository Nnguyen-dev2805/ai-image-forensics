"""Label-only output parsing for the qwen_ft baseline.

The fine-tuned model is trained to emit exactly ``{"label":"real"}`` or
``{"label":"fake"}``. Parsing accepts strict JSON first, then recovers the
first JSON object embedded in surrounding text; anything else fails without
guessing a score (label-only outputs have no ``score_fake``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

__all__ = ["QwenFTParseResult", "parse_qwen_ft_output"]

ALLOWED_LABELS = ("real", "fake")


@dataclass(frozen=True)
class QwenFTParseResult:
    label_pred: Literal["real", "fake", "unknown"]
    score_fake: float | None
    parse_status: Literal["parsed", "recovered", "failed"]
    raw_output: str


def _label_from_object(payload: object) -> Literal["real", "fake"] | None:
    """Return the label when the payload is an object with a valid label."""
    if not isinstance(payload, dict):
        return None
    label = payload.get("label")
    if label in ALLOWED_LABELS:
        return label  # type: ignore[return-value]
    return None


def parse_qwen_ft_output(raw_output: str) -> QwenFTParseResult:
    def _result(
        label_pred: Literal["real", "fake", "unknown"],
        parse_status: Literal["parsed", "recovered", "failed"],
    ) -> QwenFTParseResult:
        return QwenFTParseResult(
            label_pred=label_pred,
            score_fake=None,
            parse_status=parse_status,
            raw_output=raw_output,
        )

    try:
        label = _label_from_object(json.loads(raw_output))
    except (json.JSONDecodeError, ValueError):
        label = None
    if label is not None:
        return _result(label, "parsed")

    # Recovery: scan every {...} candidate and accept the first object whose
    # label parses. Non-greedy matching would break on nested braces, so
    # candidates are enumerated by brace depth from the first "{".
    start = raw_output.find("{")
    while start != -1:
        for end in range(len(raw_output), start, -1):
            candidate = raw_output[start:end]
            if not candidate.endswith("}"):
                continue
            try:
                label = _label_from_object(json.loads(candidate))
            except (json.JSONDecodeError, ValueError):
                continue
            if label is not None:
                return _result(label, "recovered")
        start = raw_output.find("{", start + 1)

    return _result("unknown", "failed")
