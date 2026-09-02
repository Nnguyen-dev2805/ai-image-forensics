"""Smoke tests for the aiforensics CLI skeleton."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

from aiforensics.cli.main import main

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = REPO_ROOT / "configs" / "phase_ab_smoke.yaml"
SMOKE_CONFIG = os.fspath(SMOKE_CONFIG_PATH)


def _build_tmp_config(tmp_path: pathlib.Path) -> pathlib.Path:
    cfg_path = tmp_path / "tmp_smoke.yaml"
    with open(SMOKE_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["paths"]["output_root"] = str(tmp_path / "outputs")
    data["paths"]["cache_root"] = str(tmp_path / "cache")
    data["paths"]["data_root"] = str(REPO_ROOT / data["paths"]["data_root"])
    data["paths"]["manifest_root"] = str(REPO_ROOT / data["paths"]["manifest_root"])
    data["datasets"]["tiny_genimage"]["train_manifest"] = str(
        REPO_ROOT / data["datasets"]["tiny_genimage"]["train_manifest"]
    )
    data["datasets"]["tiny_genimage"]["dev_manifest"] = str(
        REPO_ROOT / data["datasets"]["tiny_genimage"]["dev_manifest"]
    )
    data["datasets"]["genimage_unseen"]["manifest"] = str(
        REPO_ROOT / data["datasets"]["genimage_unseen"]["manifest"]
    )
    data["datasets"]["synthbuster"]["manifest"] = str(
        REPO_ROOT / data["datasets"]["synthbuster"]["manifest"]
    )
    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    (tmp_path / "pyproject.toml").touch()
    return cfg_path


@pytest.mark.parametrize(
    ("argv", "expected_substring"),
    [
        (["prepare", "--config", SMOKE_CONFIG], "prepare"),
        (["run", "--baseline", "clip_probe", "--config", SMOKE_CONFIG], "clip_probe"),
        (["run", "--baseline", "qwen_vl", "--config", SMOKE_CONFIG], "qwen_vl"),
        (["run", "--baseline", "assisted_qwen", "--config", SMOKE_CONFIG], "assisted_qwen"),
        (["evaluate", "--config", SMOKE_CONFIG], "evaluate"),
        (["report", "--config", SMOKE_CONFIG], "report"),
    ],
)
def test_subcommand_prints_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
    argv: list[str],
    expected_substring: str,
) -> None:
    cfg_path = _build_tmp_config(tmp_path)
    patched_argv = [arg if arg != SMOKE_CONFIG else str(cfg_path) for arg in argv]

    exit_code = main(patched_argv)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert expected_substring in captured.out

    if argv[0] not in ("prepare", "evaluate", "run"):
        assert str(cfg_path) in captured.out


def test_cli_prints_project_phase(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    cfg_path = _build_tmp_config(tmp_path)
    exit_code = main(["prepare", "--config", str(cfg_path)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "phase_ab_smoke" in captured.out


def test_run_with_invalid_baseline_raises_system_exit() -> None:
    cfg_path = REPO_ROOT / "configs" / "phase_ab_smoke.yaml"
    with pytest.raises(SystemExit):
        main(["run", "--baseline", "invalid", "--config", str(cfg_path)])


def test_module_invocation_help_exits_successfully() -> None:
    """Exercise the real `python -m aiforensics.cli.main --help` flow.

    Guards the `if __name__ == "__main__":` block at the bottom of
    `src/aiforensics/cli/main.py`, which is a Done Criterion in the spec.
    Spawns a subprocess so the `__main__` guard actually runs, instead
    of calling the parser in-process.
    """
    src_path = str(REPO_ROOT / "src")
    env = {**os.environ, "PYTHONPATH": src_path}
    result = subprocess.run(
        [sys.executable, "-m", "aiforensics.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "COMMAND" in result.stdout


def test_run_qwen_vl_artifacts(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Need to patch qwen dependencies to make it 'deferred' or 'completed' and check run_dir
    import importlib.util

    cfg_path = _build_tmp_config(tmp_path)

    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name, *args, **kwargs):
        if name in ("torch", "transformers", "qwen_vl_utils", "accelerate"):
            return "mocked_spec"
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    # We will mock the adapter run itself for "completed" run to verify artifacts are placed
    from aiforensics.baselines.base import RunResult
    from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter

    def mock_run(self, config, output_dir, run_id):
        (output_dir / "predictions.jsonl").touch()
        (output_dir / "logs.txt").touch()
        return RunResult(
            baseline="qwen_vl",
            run_id=run_id,
            status="completed",
            output_dir=output_dir,
            prediction_path=output_dir / "predictions.jsonl",
            log_path=output_dir / "logs.txt",
            environment_path=output_dir / "environment.json",
            status_path=output_dir / "status.json",
            reason=None,
        )

    monkeypatch.setattr(QwenVLAdapter, "run", mock_run)

    main(["prepare", "--config", str(cfg_path)])
    exit_code = main(["run", "--baseline", "qwen_vl", "--config", str(cfg_path)])
    assert exit_code == 0

    # Check if run_dir contains predictions.jsonl
    outputs_dir = tmp_path / "outputs"
    run_dirs = list(outputs_dir.glob("*_qwen_vl"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "logs.txt").exists()
    assert not (outputs_dir / "predictions.jsonl").exists()
