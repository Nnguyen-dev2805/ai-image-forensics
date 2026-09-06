"""Tests for balanced GenImage export planning and manifest writing."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from aiforensics.finetune.genimage_export import (
    PROTOCOLS,
    build_export_plan,
    write_export_manifests,
)


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(path)


def _make_genimage_tree(tmp_path: Path, generators: tuple[str, ...], images_per_label: int) -> None:
    for generator in generators:
        for split in ("train", "val"):
            for label_dir in ("ai", "nature"):
                for idx in range(images_per_label):
                    _image(tmp_path / generator / split / label_dir / f"{idx:05d}.png")


def test_build_export_plan_balances_labels_per_generator(tmp_path: Path) -> None:
    generators = ("imagenet_ai_0419_biggan", "imagenet_glide")
    _make_genimage_tree(tmp_path, generators, images_per_label=3)

    plan = build_export_plan(
        data_root=tmp_path,
        train_generators=list(generators),
        eval_generators=list(generators),
        train_per_label=2,
        eval_per_label=1,
        seed=70,
    )

    assert len(plan.train_records) == 8
    assert len(plan.eval_records) == 4
    assert {r.label for r in plan.train_records} == {"real", "fake"}
    assert {r.label for r in plan.eval_records} == {"real", "fake"}


def test_build_export_plan_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    generators = ("imagenet_ai_0419_biggan",)
    _make_genimage_tree(tmp_path, generators, images_per_label=10)

    kwargs = {
        "data_root": tmp_path,
        "train_generators": list(generators),
        "eval_generators": list(generators),
        "train_per_label": 4,
        "eval_per_label": 3,
        "seed": 70,
    }
    first = build_export_plan(**kwargs)
    second = build_export_plan(**kwargs)

    assert [r.sample_id for r in first.train_records] == [r.sample_id for r in second.train_records]
    assert [r.sample_id for r in first.eval_records] == [r.sample_id for r in second.eval_records]


def test_build_export_plan_train_and_eval_come_from_different_splits(tmp_path: Path) -> None:
    generators = ("imagenet_glide",)
    _make_genimage_tree(tmp_path, generators, images_per_label=2)

    plan = build_export_plan(
        data_root=tmp_path,
        train_generators=list(generators),
        eval_generators=list(generators),
        train_per_label=2,
        eval_per_label=2,
        seed=70,
    )

    assert {r.split for r in plan.train_records} == {"train"}
    assert {r.split for r in plan.eval_records} == {"external"}
    assert {r.source_path.parent.parent.name for r in plan.train_records} == {"train"}
    assert {r.source_path.parent.parent.name for r in plan.eval_records} == {"val"}


def test_build_export_plan_raises_when_not_enough_images(tmp_path: Path) -> None:
    generators = ("imagenet_glide",)
    _make_genimage_tree(tmp_path, generators, images_per_label=1)

    with pytest.raises(ValueError, match=r"imagenet_glide.*train.*ai.*requested 5.*available 1"):
        build_export_plan(
            data_root=tmp_path,
            train_generators=list(generators),
            eval_generators=[],
            train_per_label=5,
            eval_per_label=0,
            seed=70,
        )


def test_export_records_carry_checksums_and_gcs_relative_paths(tmp_path: Path) -> None:
    generators = ("imagenet_glide",)
    _make_genimage_tree(tmp_path, generators, images_per_label=1)

    plan = build_export_plan(
        data_root=tmp_path,
        train_generators=list(generators),
        eval_generators=[],
        train_per_label=1,
        eval_per_label=0,
        seed=70,
    )

    record = plan.train_records[0]
    source = record.source_path
    assert record.checksum == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record.relative_gcs_path.startswith("train/imagenet_glide/")
    assert record.sample_id == f"genimage/train/imagenet_glide/{source.parent.name}/{source.stem}"


def test_write_export_manifests_uses_gcs_paths_and_schema_columns(tmp_path: Path) -> None:
    generators = ("imagenet_glide",)
    _make_genimage_tree(tmp_path, generators, images_per_label=1)

    plan = build_export_plan(
        data_root=tmp_path,
        train_generators=list(generators),
        eval_generators=list(generators),
        train_per_label=1,
        eval_per_label=1,
        seed=70,
    )
    gcs_base_uri = "gs://aiforensics-qwen-ft-579187260419/data/protocol-a-small"
    manifests = write_export_manifests(plan, tmp_path / "manifests", gcs_base_uri=gcs_base_uri)

    assert set(manifests) == {"train", "eval_small"}
    for key in ("train", "eval_small"):
        header = manifests[key].read_text(encoding="utf-8").splitlines()[0]
        assert header == "sample_id,path,label,source,split,checksum,dataset,generator"

    with open(manifests["train"], encoding="utf-8", newline="") as f:
        train_rows = list(csv.DictReader(f))
    with open(manifests["eval_small"], encoding="utf-8", newline="") as f:
        eval_rows = list(csv.DictReader(f))

    assert len(train_rows) == 2
    assert len(eval_rows) == 2
    assert train_rows[0]["split"] == "train"
    assert eval_rows[0]["split"] == "external"
    for row in [*train_rows, *eval_rows]:
        assert row["path"].startswith(f"{gcs_base_uri}/train/") or row["path"].startswith(
            f"{gcs_base_uri}/eval-small/"
        )
        assert row["dataset"] == "genimage"
        assert row["generator"] == "imagenet_glide"
        assert row["source"] == "imagenet_glide"
        assert len(row["checksum"]) == 64


def test_write_export_manifests_rejects_non_gcs_base_uri(tmp_path: Path) -> None:
    generators = ("imagenet_glide",)
    _make_genimage_tree(tmp_path, generators, images_per_label=1)
    plan = build_export_plan(
        data_root=tmp_path,
        train_generators=list(generators),
        eval_generators=[],
        train_per_label=1,
        eval_per_label=0,
        seed=70,
    )

    with pytest.raises(ValueError, match="gs://"):
        write_export_manifests(plan, tmp_path / "manifests", gcs_base_uri="/local/path")


def test_protocols_match_design_spec_generator_sets() -> None:
    all_generators = {
        "imagenet_ai_0419_biggan",
        "imagenet_ai_0419_vqdm",
        "imagenet_ai_0424_sdv5",
        "imagenet_ai_0424_wukong",
        "imagenet_ai_0508_adm",
        "imagenet_glide",
        "imagenet_midjourney",
    }
    seen = {
        "imagenet_ai_0419_biggan",
        "imagenet_ai_0419_vqdm",
        "imagenet_ai_0508_adm",
        "imagenet_glide",
    }
    unseen = {"imagenet_ai_0424_sdv5", "imagenet_ai_0424_wukong", "imagenet_midjourney"}

    protocol_a = PROTOCOLS["protocol-a-small"]
    protocol_b = PROTOCOLS["protocol-b-seen-unseen"]

    assert set(protocol_a.train_generators) == all_generators
    assert set(protocol_a.eval_generators) == all_generators
    assert set(protocol_b.train_generators) == seen
    assert set(protocol_b.eval_generators) == unseen
    assert not (set(protocol_b.train_generators) & set(protocol_b.eval_generators))
