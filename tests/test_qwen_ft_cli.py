"""Tests for qwen_ft config loading and CLI registration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aiforensics.cli.main import SUPPORTED_BASELINES, build_parser
from aiforensics.config import load_config
from aiforensics.config.models import QwenFTConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_qwen_ft_protocol_a_config_loads() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "qwen_ft_protocol_a_small.yaml")

    assert cfg.project.phase == "qwen_ft_protocol_a_small"
    assert cfg.baselines.qwen_ft.enabled is True
    assert cfg.baselines.qwen_ft.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg.baselines.qwen_ft.adapter_uri.startswith("gs://aiforensics-qwen-ft-579187260419/")
    assert cfg.baselines.qwen_ft.prompt_id == "qwen_ft_label_json_v1"
    assert cfg.baselines.qwen_ft.output_fields == ["label"]


def test_qwen_ft_protocol_b_config_loads() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "qwen_ft_protocol_b_seen_unseen.yaml")

    assert cfg.project.phase == "qwen_ft_protocol_b_seen_unseen"
    assert cfg.baselines.qwen_ft.enabled is True
    assert cfg.baselines.qwen_ft.adapter_uri == (
        "gs://aiforensics-qwen-ft-579187260419/checkpoints/protocol-b-seen-unseen/final_adapter"
    )
    assert cfg.baselines.qwen_ft.prompt_id == "qwen_ft_label_json_v1"
    assert cfg.datasets.genimage_unseen.generators == [
        "imagenet_ai_0424_sdv5",
        "imagenet_ai_0424_wukong",
        "imagenet_midjourney",
    ]


def test_configs_without_qwen_ft_section_default_to_disabled() -> None:
    """Pre-qwen_ft configs must keep loading: the section is optional and off by default."""
    cfg = load_config(REPO_ROOT / "configs" / "phase_ab_smoke.yaml")

    assert cfg.baselines.qwen_ft.enabled is False
    assert cfg.baselines.qwen_ft.adapter_uri == ""


def test_enabled_qwen_ft_requires_adapter_uri() -> None:
    with pytest.raises(ValidationError):
        QwenFTConfig(enabled=True, adapter_uri="")


def test_disabled_qwen_ft_without_adapter_uri_is_allowed() -> None:
    cfg = QwenFTConfig(enabled=False)

    assert cfg.adapter_uri == ""


def test_run_baseline_choices_include_qwen_ft() -> None:
    assert "qwen_ft" in SUPPORTED_BASELINES

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--baseline",
            "qwen_ft",
            "--config",
            str(REPO_ROOT / "configs" / "phase_ab_smoke.yaml"),
        ]
    )

    assert args.baseline == "qwen_ft"
