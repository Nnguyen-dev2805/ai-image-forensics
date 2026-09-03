"""Isolated NPR inference runtime, executed as its own subprocess.

Run via ``python -m aiforensics.baselines.npr.runtime`` with the arguments built
by the adapter. The external NPR checkout is added to ``sys.path`` for the
lifetime of this process only, so NPR's top-level module names (``networks``,
``data``, ``options``) never leak into the long-lived CLI process.

Exit-code protocol:
- 0: success, ``--output-jsonl`` written with one score per input row,
- 2: setup failure before per-sample inference (defer-eligible),
- 1: inference/output failure after setup (failed).

Torch is imported lazily inside functions; importing this module is safe
without Torch installed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Callable
from pathlib import Path

__all__ = ["main", "parse_args", "run_runtime"]

SETUP_EXIT_CODE = 2
INFERENCE_EXIT_CODE = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aiforensics.baselines.npr.runtime")
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_runtime_device(device: str):
    """Resolve the requested device to the torch CUDA device.

    ``auto`` is resolved here (not in the adapter) because CUDA availability can
    only be confirmed after importing torch, which this subprocess does anyway.
    CPU inference is unsupported in Task 10: NPR on CPU is impractically slow,
    so an unusable CUDA runtime is a setup failure (exit code 2).
    """
    import torch

    if device not in ("auto", "cuda"):
        raise RuntimeError(f"Unsupported runtime device for NPR inference: {device!r}")
    if not torch.cuda.is_available():
        if device == "auto":
            raise RuntimeError(
                "runtime.device=auto resolved to no usable CUDA: "
                "CUDA is not available in the NPR runtime process"
            )
        raise RuntimeError("CUDA is not available in the NPR runtime process")
    return torch.device("cuda")


def _load_model(repo_dir: Path, checkpoint: Path, device):
    """Load the official NPR network once per runner process."""
    sys.path.insert(0, str(repo_dir))
    import torch
    from networks.resnet import resnet50

    model = resnet50(num_classes=1)
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _infer_batch(model, batch_array, device) -> list[float]:
    import torch

    with torch.no_grad():
        inputs = torch.from_numpy(batch_array).to(device)
        logits = model(inputs)
        return logits.sigmoid().flatten().tolist()


def _read_input_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Runtime input file missing: {path}")
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f.read().splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"Blank runtime input row at line {i}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Runtime input row {i} is not a JSON object")
            sample_id = row.get("sample_id")
            image_path = row.get("path")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Runtime input row {i} has invalid sample_id")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"Runtime input row {i} has invalid path")
            rows.append({"sample_id": sample_id, "path": image_path})
    return rows


def _run_inference(
    rows: list[dict[str, str]],
    model,
    *,
    batch_size: int,
    device,
    batch_runner: Callable,
) -> list[tuple[str, float]]:
    from PIL import Image

    from aiforensics.baselines.npr.preprocess import preprocess_npr_genimage_v1

    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}")

    results: list[tuple[str, float]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        arrays = []
        for row in chunk:
            with Image.open(row["path"]) as img:
                arrays.append(preprocess_npr_genimage_v1(img))

        import numpy as np

        batch_array = np.stack(arrays).astype(np.float32, copy=False)
        scores = batch_runner(model, batch_array, device)
        if len(scores) != len(chunk):
            raise RuntimeError(f"Batch produced {len(scores)} scores for {len(chunk)} inputs")
        for row, score in zip(chunk, scores, strict=True):
            score_f = float(score)
            if not math.isfinite(score_f) or not 0.0 <= score_f <= 1.0:
                raise RuntimeError(f"Non-finite or out-of-range score for {row['sample_id']}")
            results.append((row["sample_id"], score_f))
    return results


def _write_output_rows(path: Path, results: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample_id, score in results:
            f.write(json.dumps({"sample_id": sample_id, "score_fake": score}) + "\n")


def run_runtime(
    args: argparse.Namespace,
    *,
    seed_setter: Callable = _set_seeds,
    device_resolver: Callable = _resolve_runtime_device,
    model_loader: Callable = _load_model,
    batch_runner: Callable = _infer_batch,
) -> int:
    """Execute one NPR inference pass; see module docstring for the exit protocol.

    ``seed_setter``, ``device_resolver``, and ``model_loader`` are injection
    points so tests can exercise the runner without Torch or CUDA.
    """
    try:
        rows = _read_input_rows(args.input_jsonl)
    except Exception as exc:
        print(f"[npr-runtime] input error: {exc}", file=sys.stderr)
        return INFERENCE_EXIT_CODE

    try:
        seed_setter(args.seed)
        device = device_resolver(args.device)
        model = model_loader(args.repo_dir, args.checkpoint, device)
    except Exception as exc:
        print(f"[npr-runtime] setup error: {exc}", file=sys.stderr)
        return SETUP_EXIT_CODE

    try:
        results = _run_inference(
            rows,
            model,
            batch_size=args.batch_size,
            device=device,
            batch_runner=batch_runner,
        )
        _write_output_rows(args.output_jsonl, results)
    except Exception as exc:
        print(f"[npr-runtime] inference error: {exc}", file=sys.stderr)
        return INFERENCE_EXIT_CODE

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_runtime(args)


if __name__ == "__main__":
    raise SystemExit(main())
