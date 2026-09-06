"""Tests for rewriting qwen_train.jsonl image paths onto Modal Volume locations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiforensics.finetune.qwen_data import rewrite_qwen_jsonl_image_paths


def test_rewrite_qwen_jsonl_image_paths_makes_relative_images_absolute(tmp_path: Path) -> None:
    image_root = tmp_path / "protocol-a-small"
    image = image_root / "train" / "gen" / "ai" / "0.JPEG"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")

    source = tmp_path / "qwen_train.jsonl"
    source.write_text(
        json.dumps({"image": "train/gen/ai/0.JPEG", "conversations": []}) + "\n",
        encoding="utf-8",
    )

    output = rewrite_qwen_jsonl_image_paths(source, tmp_path / "qwen_train_abs.jsonl", image_root)

    assert output == tmp_path / "qwen_train_abs.jsonl"
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["image"] == str(image)
    # The original relative JSONL stays untouched for provenance.
    original = json.loads(source.read_text(encoding="utf-8"))
    assert original["image"] == "train/gen/ai/0.JPEG"


def test_rewrite_keeps_already_absolute_paths(tmp_path: Path) -> None:
    image = tmp_path / "elsewhere" / "1.JPEG"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")

    source = tmp_path / "qwen_train.jsonl"
    source.write_text(
        json.dumps({"image": str(image), "conversations": []}) + "\n", encoding="utf-8"
    )

    output = rewrite_qwen_jsonl_image_paths(source, tmp_path / "abs.jsonl", tmp_path / "root")

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["image"] == str(image)


def test_rewrite_raises_when_training_image_missing(tmp_path: Path) -> None:
    source = tmp_path / "qwen_train.jsonl"
    source.write_text(
        json.dumps({"image": "train/gen/ai/missing.JPEG", "conversations": []}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing.JPEG"):
        rewrite_qwen_jsonl_image_paths(source, tmp_path / "abs.jsonl", tmp_path / "root")
    # No partial output file is left behind.
    assert not (tmp_path / "abs.jsonl").exists()
