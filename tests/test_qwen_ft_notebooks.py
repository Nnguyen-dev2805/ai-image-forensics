"""Tests for the generated qwen_ft thin notebooks."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"], f"notebook has no cells: {path.name}"
    # Generated notebooks must stay clean: no outputs, no execution counts.
    for cell in notebook["cells"]:
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_kaggle_qwen_ft_export_notebook_calls_export_script() -> None:
    source = _notebook_source(NOTEBOOK_DIR / "kaggle_qwen_ft_export.ipynb")

    assert "export_genimage_qwen_ft_subset.py" in source
    assert "--protocol protocol-a-small" in source
    assert "--train-per-label 100" in source
    assert "--eval-per-label 50" in source
    assert "--bucket-uri gs://aiforensics-qwen-ft-579187260419" in source


def test_kaggle_export_notebook_authenticates_without_printing_credentials() -> None:
    source = _notebook_source(NOTEBOOK_DIR / "kaggle_qwen_ft_export.ipynb")

    assert "UserSecretsClient" in source
    assert "gcloud auth activate-service-account" in source
    assert "credential material is never printed" in source.lower()


def test_vertex_notebook_dry_runs_then_submits_vertex_job() -> None:
    source = _notebook_source(NOTEBOOK_DIR / "vertex_qwen_ft_submit_and_eval.ipynb")

    assert "submit_qwen_ft_vertex_job.py" in source
    assert "--dry-run" in source
    assert "--protocol protocol-a-small" in source
    assert "--machine-type a2-highgpu-1g" in source
    assert "--accelerator-type NVIDIA_TESLA_A100" in source


def test_vertex_notebook_runs_eval_small_then_eval_full_and_report() -> None:
    source = _notebook_source(NOTEBOOK_DIR / "vertex_qwen_ft_submit_and_eval.ipynb")

    assert "aiforensics run --baseline qwen_ft" in source
    assert "aiforensics evaluate" in source
    assert "aiforensics report" in source
    # Parse-failure gate decides whether eval-full runs.
    assert "parse_failure_rate" in source
    assert "0.02" in source


def test_notebooks_embed_no_business_logic() -> None:
    """Thin-wrapper contract: heavy lifting must stay in the package/scripts."""
    for name in (
        "kaggle_qwen_ft_export.ipynb",
        "vertex_qwen_ft_submit_and_eval.ipynb",
    ):
        source = _notebook_source(NOTEBOOK_DIR / name)
        assert "import torch" not in source
        assert "from peft" not in source
        assert "Qwen2_5_VLForConditionalGeneration" not in source
        assert "compute_classification_metrics" not in source


def test_export_script_accepts_modal_archive_flags() -> None:
    from scripts.export_genimage_qwen_ft_subset import build_parser

    args = build_parser().parse_args(
        [
            "--data-root",
            "/kaggle/input/genimage",
            "--protocol",
            "protocol-a-small",
            "--path-mode",
            "modal-volume",
            "--archive-path",
            "/kaggle/working/qwen_ft_protocol_a_small.zip",
        ]
    )

    assert args.path_mode == "modal-volume"
    assert args.archive_path == Path("/kaggle/working/qwen_ft_protocol_a_small.zip")


def test_modal_export_notebook_uses_archive_mode() -> None:
    source = _notebook_source(NOTEBOOK_DIR / "kaggle_qwen_ft_modal_export.ipynb")

    assert "AIF_SECTION: modal_zip_export" in source
    assert "--path-mode modal-volume" in source
    assert "--archive-path" in source
    assert "--train-per-label 100" in source
    assert "--eval-per-label 50" in source
    # The Modal path must not need any GCP credential or tooling.
    assert "gcloud storage" not in source
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in source
