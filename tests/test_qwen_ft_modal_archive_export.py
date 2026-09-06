"""Tests for Modal archive export: relative manifests and the Kaggle zip."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from aiforensics.finetune.genimage_export import (
    ALL_GENERATORS,
    build_export_plan,
    find_genimage_data_root,
    write_export_archive,
    write_relative_export_manifests,
)


def _make_protocol_tree(tmp_path: Path, images_per_label: int = 1) -> Path:
    for generator in ALL_GENERATORS:
        for split in ("train", "val"):
            for label_dir in ("ai", "nature"):
                for idx in range(images_per_label):
                    image = tmp_path / generator / split / label_dir / f"{idx:05d}.JPEG"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(f"{label_dir}-{idx}".encode())
    return tmp_path


def _protocol_plan(data_root: Path, train_per_label: int, eval_per_label: int):
    return build_export_plan(
        data_root=data_root,
        protocol="protocol-a-small",
        train_per_label=train_per_label,
        eval_per_label=eval_per_label,
        seed=70,
    )


def test_write_relative_export_manifests_uses_protocol_relative_paths(tmp_path: Path) -> None:
    data_root = _make_protocol_tree(tmp_path / "genimage")
    plan = _protocol_plan(data_root, train_per_label=1, eval_per_label=0)

    written = write_relative_export_manifests(plan, tmp_path / "manifests")

    assert set(written) == {"train", "eval_small"}
    text = written["train"].read_text(encoding="utf-8")
    assert "gs://" not in text
    assert "train/imagenet_ai_0419_biggan/ai/00000.JPEG" in text
    assert "train/imagenet_ai_0419_biggan/nature/00000.JPEG" in text
    header = text.splitlines()[0]
    assert header == "sample_id,path,label,source,split,checksum,dataset,generator"
    eval_text = written["eval_small"].read_text(encoding="utf-8")
    assert "eval-small/" not in eval_text  # eval_per_label=0 -> header-only file


def test_build_export_plan_accepts_protocol_kwarg(tmp_path: Path) -> None:
    data_root = _make_protocol_tree(tmp_path / "genimage")
    plan = _protocol_plan(data_root, train_per_label=1, eval_per_label=0)

    assert plan.protocol == "protocol-a-small"
    assert len(plan.train_records) == 14  # 7 generators x 2 labels


def test_build_export_plan_rejects_unknown_protocol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown protocol"):
        build_export_plan(
            data_root=tmp_path,
            protocol="nope",
            train_per_label=1,
            eval_per_label=0,
            seed=70,
        )


def test_build_export_plan_rejects_protocol_and_generators_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either protocol or explicit generator"):
        build_export_plan(
            data_root=tmp_path,
            protocol="protocol-a-small",
            train_generators=["imagenet_glide"],
            train_per_label=1,
            eval_per_label=0,
            seed=70,
        )


def test_write_export_archive_contains_images_manifests_and_training_json(
    tmp_path: Path,
) -> None:
    data_root = _make_protocol_tree(tmp_path / "genimage")
    plan = _protocol_plan(data_root, train_per_label=1, eval_per_label=0)

    archive = write_export_archive(
        plan, tmp_path / "qwen_ft_protocol_a_small.zip", tmp_path / "stage"
    )

    assert archive == tmp_path / "qwen_ft_protocol_a_small.zip"
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        with zf.open("protocol-a-small/qwen_train.jsonl") as handle:
            first_row = json.loads(handle.read().decode("utf-8").splitlines()[0])
        train_manifest = zf.read("protocol-a-small/manifests/train.csv").decode("utf-8")

    assert "protocol-a-small/manifests/train.csv" in names
    assert "protocol-a-small/manifests/eval_small.csv" in names
    assert "protocol-a-small/qwen_train.jsonl" in names
    assert "protocol-a-small/train/imagenet_ai_0419_biggan/ai/00000.JPEG" in names
    assert "protocol-a-small/train/imagenet_ai_0419_biggan/nature/00000.JPEG" in names
    # The training JSONL keeps protocol-relative image paths for the Modal unpack step.
    assert first_row["image"] == "train/imagenet_ai_0419_biggan/ai/00000.JPEG"
    assert json.loads(first_row["conversations"][1]["value"]) == {"label": "fake"}
    assert "gs://" not in train_manifest


def test_write_export_archive_includes_eval_full_when_requested(tmp_path: Path) -> None:
    data_root = _make_protocol_tree(tmp_path / "genimage")
    plan = build_export_plan(
        data_root=data_root,
        protocol="protocol-a-small",
        train_per_label=1,
        eval_per_label=1,
        eval_full_per_label=1,
        seed=70,
    )

    assert len(plan.eval_full_records) == 14

    archive = write_export_archive(
        plan, tmp_path / "qwen_ft_protocol_a_full.zip", tmp_path / "stage"
    )

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())

    assert "protocol-a-small/manifests/eval_full.csv" in names
    assert "protocol-a-small/eval-full/imagenet_ai_0419_biggan/ai/00000.JPEG" in names
    assert "protocol-a-small/eval-full/imagenet_ai_0419_biggan/nature/00000.JPEG" in names


def test_find_genimage_data_root_accepts_raw_zip_layout(tmp_path: Path) -> None:
    data_root = _make_protocol_tree(tmp_path / "raw")

    assert find_genimage_data_root(tmp_path) == data_root


def test_find_genimage_data_root_rejects_non_genimage_tree(tmp_path: Path) -> None:
    (tmp_path / "not_genimage" / "train").mkdir(parents=True)

    with pytest.raises(ValueError, match="Could not find GenImage data root"):
        find_genimage_data_root(tmp_path)
