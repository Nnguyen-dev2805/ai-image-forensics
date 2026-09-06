"""End-to-end qwen_ft smoke pipeline without real Qwen training.

Uses the ``smoke://adapter`` mode of the qwen_ft adapter so the full
prepare -> run -> evaluate -> report cycle runs on CPU fixtures without
importing torch, transformers, or PEFT.
"""

from __future__ import annotations

import builtins
import json
import pathlib

import pytest
import yaml

from aiforensics.cli.main import main

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = REPO_ROOT / "configs" / "qwen_ft_smoke.yaml"


def _build_tmp_config(tmp_path: pathlib.Path) -> pathlib.Path:
    cfg_path = tmp_path / "tmp_qwen_ft_smoke.yaml"
    with open(SMOKE_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["paths"]["output_root"] = str(tmp_path / "outputs")
    data["paths"]["cache_root"] = str(tmp_path / "cache")
    # Smoke fixtures stay in the repository; only writable roots move.
    data["paths"]["data_root"] = str(REPO_ROOT / data["paths"]["data_root"])
    data["paths"]["manifest_root"] = str(REPO_ROOT / data["paths"]["manifest_root"])
    data["datasets"]["tiny_genimage"]["train_manifest"] = str(
        REPO_ROOT / data["datasets"]["tiny_genimage"]["train_manifest"]
    )
    data["datasets"]["tiny_genimage"]["dev_manifest"] = str(
        REPO_ROOT / data["datasets"]["tiny_genimage"]["dev_manifest"]
    )

    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    (tmp_path / "pyproject.toml").touch()
    return cfg_path


def test_qwen_ft_smoke_pipeline(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIF_QWEN_FT_SMOKE", "1")
    cfg_path = _build_tmp_config(tmp_path)

    def guarded_import(name, *args, **kwargs):
        if name in ("torch", "transformers", "peft"):
            raise AssertionError("smoke qwen_ft must not import model runtimes")
        return real_import(name, *args, **kwargs)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert main(["prepare", "--config", str(cfg_path)]) == 0
    assert main(["run", "--baseline", "qwen_ft", "--config", str(cfg_path)]) == 0
    assert main(["evaluate", "--config", str(cfg_path)]) == 0
    assert main(["report", "--config", str(cfg_path)]) == 0

    run_dirs = list((tmp_path / "outputs").glob("*_qwen_ft"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "confusion_matrix.json").exists()

    records = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert all(record["model_name"] == "qwen_ft" for record in records)
    assert all(record["score_fake"] is None for record in records)
    # Deterministic smoke labels follow the fixture sample ids.
    by_label_true = {record["label_true"]: record["label_pred"] for record in records}
    assert by_label_true == {"fake": "fake", "real": "real"}

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"]["accuracy"] == 1.0
    assert metrics["overall"]["auroc"] is None

    report_path = tmp_path / "outputs" / "qwen_ft_smoke_report.md"
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "| qwen_ft |" in report_text
    assert "qwen_ft is label-only in this phase" in report_text
    # The smoke disclaimer must guard the recommendation.
    assert "not scientific evidence" in report_text
