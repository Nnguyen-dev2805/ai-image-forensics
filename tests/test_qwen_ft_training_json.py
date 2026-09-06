"""Tests for manifest-to-Qwen label-only training JSONL conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiforensics.finetune.qwen_data import (
    PROMPT_TEXT,
    split_qwen_train_val,
    write_qwen_label_jsonl,
)


def test_write_qwen_label_jsonl_uses_label_only_response(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "sample_id,path,label,source,split,checksum\n"
        "s1,gs://bucket/a.png,fake,g1,train,abc\n"
        "s2,gs://bucket/b.png,real,g1,train,def\n",
        encoding="utf-8",
    )
    out = tmp_path / "qwen_train.jsonl"

    write_qwen_label_jsonl(manifest, out)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["image"] == "gs://bucket/a.png"
    assert json.loads(rows[0]["conversations"][1]["value"]) == {"label": "fake"}
    assert json.loads(rows[1]["conversations"][1]["value"]) == {"label": "real"}
    assert rows[0]["conversations"][0]["from"] == "human"
    assert rows[0]["conversations"][1]["from"] == "gpt"


def test_prompt_text_matches_label_only_contract() -> None:
    assert PROMPT_TEXT.startswith("<image>\n")
    assert '"real"' in PROMPT_TEXT
    assert '"fake"' in PROMPT_TEXT
    assert "confidence" not in PROMPT_TEXT
    assert "evidence" not in PROMPT_TEXT


def test_write_qwen_label_jsonl_serializes_compact_label_json(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "sample_id,path,label,source,split,checksum\ns1,gs://bucket/a.png,fake,g1,train,abc\n",
        encoding="utf-8",
    )

    write_qwen_label_jsonl(manifest, tmp_path / "out.jsonl")

    row = json.loads((tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()[0])
    # Compact separators so the training target is exactly one JSON object.
    assert row["conversations"][1]["value"] == '{"label":"fake"}'


def test_write_qwen_label_jsonl_rejects_unknown_label(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "sample_id,path,label,source,split,checksum\ns1,gs://bucket/a.png,synthetic,g1,train,abc\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="synthetic"):
        write_qwen_label_jsonl(manifest, tmp_path / "out.jsonl")


def test_write_qwen_label_jsonl_requires_path_and_label_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text("sample_id,label\ns1,fake\n", encoding="utf-8")

    with pytest.raises(ValueError, match="path"):
        write_qwen_label_jsonl(manifest, tmp_path / "out.jsonl")


def test_split_qwen_train_val_stratifies_per_generator_and_label(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "train.jsonl"
    rows = []
    # 2 generators, each having 10 real and 10 fake = 40 samples total
    for gen in ["gen_a", "gen_b"]:
        for label in ["real", "fake"]:
            for i in range(10):
                rows.append(
                    {
                        "image": f"/data/train/{gen}/{label}/img_{i:03d}.png",
                        "conversations": [
                            {"from": "human", "value": "Classify"},
                            {"from": "gpt", "value": json.dumps({"label": label})},
                        ],
                    }
                )
    input_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    train_out = tmp_path / "train_split.jsonl"
    val_out = tmp_path / "val_split.jsonl"

    n_train, n_val = split_qwen_train_val(input_jsonl, train_out, val_out, val_ratio=0.2, seed=70)

    # 40 total, 20% is 8 val (2 per group of 10) and 32 train (8 per group of 10)
    assert n_train == 32
    assert n_val == 8

    train_rows = [json.loads(line) for line in train_out.read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(line) for line in val_out.read_text(encoding="utf-8").splitlines()]

    assert len(train_rows) == 32
    assert len(val_rows) == 8

    # Ensure each (generator, label) has exactly 2 in val and 8 in train
    for gen in ["gen_a", "gen_b"]:
        for label in ["real", "fake"]:
            val_subset = [
                r
                for r in val_rows
                if f"/{gen}/{label}/" in r["image"]
                and json.loads(r["conversations"][1]["value"])["label"] == label
            ]
            train_subset = [
                r
                for r in train_rows
                if f"/{gen}/{label}/" in r["image"]
                and json.loads(r["conversations"][1]["value"])["label"] == label
            ]
            assert len(val_subset) == 2
            assert len(train_subset) == 8


def test_split_qwen_train_val_is_deterministic(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "train.jsonl"
    rows = [
        {
            "image": f"/data/train/gen_a/ai/img_{i}.png",
            "conversations": [
                {"from": "human", "value": "Classify"},
                {"from": "gpt", "value": '{"label":"fake"}'},
            ],
        }
        for i in range(20)
    ]
    input_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    t1, v1 = tmp_path / "t1.jsonl", tmp_path / "v1.jsonl"
    t2, v2 = tmp_path / "t2.jsonl", tmp_path / "v2.jsonl"

    split_qwen_train_val(input_jsonl, t1, v1, val_ratio=0.2, seed=42)
    split_qwen_train_val(input_jsonl, t2, v2, val_ratio=0.2, seed=42)

    assert t1.read_text() == t2.read_text()
    assert v1.read_text() == v2.read_text()


def test_split_qwen_train_val_rejects_invalid_ratio(tmp_path: Path) -> None:
    dummy = tmp_path / "dummy.jsonl"
    dummy.write_text("{}\n")
    with pytest.raises(ValueError, match="val_ratio"):
        split_qwen_train_val(dummy, tmp_path / "t.jsonl", tmp_path / "v.jsonl", val_ratio=1.5)
