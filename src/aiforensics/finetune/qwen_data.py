"""Convert manifest CSV rows into label-only Qwen image-conversation JSONL.

Reads manifest CSVs directly with ``csv.DictReader`` instead of
``load_manifest`` because export manifests reference ``gs://`` URIs that the
generic loader would resolve as local filesystem paths. Each output row
follows the Qwen2.5-VL fine-tuning conversation format: one human turn with
the classification prompt and one gpt turn whose value is the compact label
JSON object.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

__all__ = [
    "ALLOWED_LABELS",
    "INFERENCE_PROMPT_TEXT",
    "PROMPT_TEXT",
    "rewrite_qwen_jsonl_image_paths",
    "split_qwen_train_val",
    "write_qwen_label_jsonl",
]

# The <image> placeholder on its own line is the Qwen-VL image marker.
PROMPT_TEXT = (
    "<image>\n"
    "You are an image-forensics classifier. Classify the image as either "
    '"real" or "fake". Return exactly one JSON object with one key: label. '
    'The label must be "real" or "fake".'
)

ALLOWED_LABELS = ("real", "fake")

# Inference uses the training prompt minus the Qwen-VL <image> marker: at
# inference time the image is a separate message part, so deriving the text
# from PROMPT_TEXT keeps training and inference prompts identical by
# construction.
INFERENCE_PROMPT_TEXT = PROMPT_TEXT.removeprefix("<image>\n")


def write_qwen_label_jsonl(manifest_path: Path, output_path: Path) -> None:
    """Write one label-only conversation row per manifest row.

    All rows are validated before the output file is created so a bad manifest
    cannot leave a partial training JSONL behind.
    """
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = {"path", "label"} - fieldnames
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required column(s): {sorted(missing)}"
            )

        rows: list[tuple[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            image = (row.get("path") or "").strip()
            label = (row.get("label") or "").strip()
            if not image:
                raise ValueError(f"Manifest {manifest_path} row {line_number} has an empty path")
            if label not in ALLOWED_LABELS:
                raise ValueError(
                    f"Manifest {manifest_path} row {line_number} has invalid label "
                    f"{label!r}; expected one of {ALLOWED_LABELS}"
                )
            rows.append((image, label))

    with open(output_path, "w", encoding="utf-8") as f:
        for image, label in rows:
            record = {
                "image": image,
                "conversations": [
                    {"from": "human", "value": PROMPT_TEXT},
                    {"from": "gpt", "value": json.dumps({"label": label}, separators=(",", ":"))},
                ],
            }
            f.write(json.dumps(record) + "\n")


def rewrite_qwen_jsonl_image_paths(input_jsonl: Path, output_jsonl: Path, image_root: Path) -> Path:
    """Rewrite relative ``image`` values onto ``image_root`` as absolute paths.

    The Modal unpack step uses this because Qwen training scripts expect local
    filesystem image paths. Already-absolute values pass through unchanged.
    All rows are resolved and verified before the output file is created, so a
    missing image cannot leave a partial ``qwen_train_abs.jsonl`` behind.
    """
    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)
    image_root = Path(image_root)

    with open(input_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    resolved: list[dict] = []
    for line_number, row in enumerate(rows, start=1):
        image_path = Path(str(row["image"]))
        if not image_path.is_absolute():
            image_path = image_root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Training image not found for row {line_number}: {image_path}")
        row["image"] = str(image_path)
        resolved.append(row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in resolved:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_jsonl


def _extract_group_key(record: dict) -> tuple[str, str]:
    image_path = Path(str(record.get("image", "")))
    parts = image_path.parts
    generator = parts[-3] if len(parts) >= 3 else "unknown"
    label = "unknown"
    conversations = record.get("conversations")
    if isinstance(conversations, list) and len(conversations) >= 2:
        try:
            val = json.loads(conversations[1].get("value", "{}"))
            if isinstance(val, dict):
                label = str(val.get("label", "unknown"))
        except Exception:
            pass
    return generator, label


def split_qwen_train_val(
    input_jsonl: Path,
    train_output: Path,
    val_output: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 70,
) -> tuple[int, int]:
    """Split a Qwen conversation JSONL into stratified train and validation subsets.

    Grouping is done per (generator, label) so that each generator and label
    proportion is preserved in both splits. Sampling is deterministic for a
    fixed seed.
    """
    input_jsonl = Path(input_jsonl)
    train_output = Path(train_output)
    val_output = Path(val_output)

    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must satisfy 0.0 <= val_ratio < 1.0; got {val_ratio}")

    with open(input_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        train_output.parent.mkdir(parents=True, exist_ok=True)
        val_output.parent.mkdir(parents=True, exist_ok=True)
        train_output.write_text("", encoding="utf-8")
        val_output.write_text("", encoding="utf-8")
        return 0, 0

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = _extract_group_key(row)
        groups.setdefault(key, []).append(row)

    rng = random.Random(seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: str(r.get("image", "")))
        n_val = int(round(len(group) * val_ratio))
        if val_ratio > 0.0 and n_val == 0 and len(group) >= 2:
            n_val = 1
        val_indices = set(rng.sample(range(len(group)), n_val)) if n_val > 0 else set()
        for idx, item in enumerate(group):
            if idx in val_indices:
                val_rows.append(item)
            else:
                train_rows.append(item)

    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    with open(train_output, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(val_output, "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(train_rows), len(val_rows)
