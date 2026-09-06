"""Tests for the qwen_ft label-only output parser."""

from __future__ import annotations

from aiforensics.baselines.qwen_ft.parsing import parse_qwen_ft_output


def test_parse_label_only_fake() -> None:
    parsed = parse_qwen_ft_output('{"label":"fake"}')

    assert parsed.label_pred == "fake"
    assert parsed.score_fake is None
    assert parsed.parse_status == "parsed"


def test_parse_label_only_real() -> None:
    parsed = parse_qwen_ft_output('{"label":"real"}')

    assert parsed.label_pred == "real"
    assert parsed.score_fake is None
    assert parsed.parse_status == "parsed"


def test_parse_label_only_invalid_label_fails() -> None:
    parsed = parse_qwen_ft_output('{"label":"synthetic"}')

    assert parsed.label_pred == "unknown"
    assert parsed.score_fake is None
    assert parsed.parse_status == "failed"


def test_parse_recovers_first_json_object_in_surrounding_text() -> None:
    raw = 'Sure! Here is my answer: {"label":"fake"} hope that helps.'

    parsed = parse_qwen_ft_output(raw)

    assert parsed.label_pred == "fake"
    assert parsed.score_fake is None
    assert parsed.parse_status == "recovered"


def test_parse_preserves_raw_output() -> None:
    raw = '{"label":"real"}'

    parsed = parse_qwen_ft_output(raw)

    assert parsed.raw_output == raw


def test_parse_garbage_fails() -> None:
    for raw in ("", "no json here", "{not json"):
        parsed = parse_qwen_ft_output(raw)

        assert parsed.label_pred == "unknown"
        assert parsed.score_fake is None
        assert parsed.parse_status == "failed"
        assert parsed.raw_output == raw


def test_parse_label_only_ignores_extra_keys() -> None:
    parsed = parse_qwen_ft_output('{"label":"fake","confidence":0.9}')

    assert parsed.label_pred == "fake"
    assert parsed.parse_status == "parsed"
