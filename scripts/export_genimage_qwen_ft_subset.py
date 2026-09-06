#!/usr/bin/env python
"""Export balanced GenImage subsets for qwen_ft fine-tuning.

Builds deterministic label-balanced train/eval manifests from a GenImage-layout
data root, and optionally uploads the selected images plus manifests to the
GCS bucket layout expected by the Vertex training job:

    gs://<bucket>/data/<protocol>/train/...          selected training images
    gs://<bucket>/data/<protocol>/eval-small/...     selected evaluation images
    gs://<bucket>/data/<protocol>/manifests/...      train.csv, eval_small.csv

Designed to run in the Kaggle export notebook or locally; without ``--upload``
it prints a dry-run summary instead of touching GCS.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path

from aiforensics.finetune.genimage_export import (
    PROTOCOLS,
    ExportPlan,
    build_export_plan,
    write_export_archive,
    write_export_manifests,
)
from aiforensics.finetune.qwen_data import write_qwen_label_jsonl

DEFAULT_BUCKET_URI = "gs://aiforensics-qwen-ft-579187260419"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export balanced GenImage subsets for qwen_ft fine-tuning."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="GenImage-layout root: <root>/<generator>/<split>/{ai,nature}/*",
    )
    parser.add_argument(
        "--bucket-uri",
        default=DEFAULT_BUCKET_URI,
        help="GCS bucket root for the qwen_ft project (default: %(default)s)",
    )
    parser.add_argument(
        "--protocol",
        choices=sorted(PROTOCOLS),
        default="protocol-a-small",
        help="Which protocol's generator sets to export (default: %(default)s)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Local directory that receives the manifest CSVs (required for "
            "--path-mode gcs; staging falls back to the archive's parent for modal-volume)"
        ),
    )
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--train-per-label", type=int, default=100)
    parser.add_argument("--eval-per-label", type=int, default=50)
    parser.add_argument(
        "--eval-full-per-label",
        type=int,
        default=0,
        help=(
            "Number of images per label per generator for eval_full "
            "(default: %(default)s; 500 for full 7,000 val set)"
        ),
    )
    parser.add_argument(
        "--include-eval-full",
        action="store_true",
        help=(
            "Include full evaluation set in export "
            "(sets eval-full-per-label to 500 if not specified)"
        ),
    )
    parser.add_argument(
        "--path-mode",
        choices=["gcs", "modal-volume"],
        default="gcs",
        help=(
            "gcs: manifests with GCS URIs plus optional upload (legacy Vertex path). "
            "modal-volume: a protocol-relative zip archive for the Modal Volume."
        ),
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Zip destination when --path-mode=modal-volume",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload selected images and manifests with `gcloud storage cp`",
    )
    return parser


def _print_summary(
    plan: ExportPlan,
    gcs_base_uri: str,
    manifests: dict[str, Path],
    qwen_jsonl_path: Path,
) -> None:
    train_labels = Counter(record.label for record in plan.train_records)
    eval_labels = Counter(record.label for record in plan.eval_records)
    print(f"[export] train images={len(plan.train_records)} {dict(train_labels)}")
    print(f"[export] eval images={len(plan.eval_records)} {dict(eval_labels)}")
    if plan.eval_full_records:
        eval_full_labels = Counter(record.label for record in plan.eval_full_records)
        print(f"[export] eval-full images={len(plan.eval_full_records)} {dict(eval_full_labels)}")
    print(f"[export] gcs data prefix={gcs_base_uri}")
    for name, path in manifests.items():
        print(f"[export] manifest {name}={path}")
    print(f"[export] qwen_train.jsonl={qwen_jsonl_path}")


def _upload(
    plan: ExportPlan,
    manifests: dict[str, Path],
    qwen_jsonl_path: Path,
    gcs_base_uri: str,
) -> None:
    pairs: list[tuple[Path, str]] = [
        (record.source_path, f"{gcs_base_uri}/{record.relative_gcs_path}")
        for record in [*plan.train_records, *plan.eval_records, *plan.eval_full_records]
    ]
    pairs.extend(
        (manifest_path, f"{gcs_base_uri}/manifests/{manifest_path.name}")
        for manifest_path in manifests.values()
    )
    pairs.append((qwen_jsonl_path, f"{gcs_base_uri}/qwen_train.jsonl"))

    for source, destination in pairs:
        print(f"[upload] {source} -> {destination}")
        subprocess.run(
            ["gcloud", "storage", "cp", str(source), destination],
            check=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    eval_full_per_label = args.eval_full_per_label
    if args.include_eval_full and eval_full_per_label == 0:
        eval_full_per_label = 500

    if args.path_mode == "modal-volume":
        if args.archive_path is None:
            parser.error("--archive-path is required when --path-mode=modal-volume")
        staging_dir = (args.work_dir or args.archive_path.parent) / "modal_stage"
        plan = build_export_plan(
            data_root=args.data_root,
            protocol=args.protocol,
            train_per_label=args.train_per_label,
            eval_per_label=args.eval_per_label,
            eval_full_per_label=eval_full_per_label,
            seed=args.seed,
        )
        train_labels = Counter(record.label for record in plan.train_records)
        eval_labels = Counter(record.label for record in plan.eval_records)
        print(f"[export] train images={len(plan.train_records)} {dict(train_labels)}")
        print(f"[export] eval images={len(plan.eval_records)} {dict(eval_labels)}")
        if plan.eval_full_records:
            eval_full_labels = Counter(record.label for record in plan.eval_full_records)
            print(
                f"[export] eval-full images={len(plan.eval_full_records)} {dict(eval_full_labels)}"
            )
        archive = write_export_archive(plan, args.archive_path, staging_dir)
        print(f"archive_path={archive}")
        return 0

    if args.work_dir is None:
        parser.error("--work-dir is required when --path-mode=gcs")

    gcs_base_uri = f"{args.bucket_uri.rstrip('/')}/data/{args.protocol}"
    plan = build_export_plan(
        data_root=args.data_root,
        protocol=args.protocol,
        train_per_label=args.train_per_label,
        eval_per_label=args.eval_per_label,
        eval_full_per_label=eval_full_per_label,
        seed=args.seed,
    )
    manifests = write_export_manifests(plan, args.work_dir, gcs_base_uri=gcs_base_uri)
    qwen_jsonl_path = args.work_dir / "qwen_train.jsonl"
    write_qwen_label_jsonl(manifests["train"], qwen_jsonl_path)
    _print_summary(plan, gcs_base_uri, manifests, qwen_jsonl_path)

    if args.upload:
        _upload(plan, manifests, qwen_jsonl_path, gcs_base_uri)
    else:
        print("[export] dry-run: pass --upload to copy files to GCS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
