"""Smoke tests for the aiforensics CLI skeleton."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from aiforensics.cli.main import build_parser, main


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = REPO_ROOT / "configs" / "phase_ab_smoke.yaml"
SMOKE_CONFIG = os.fspath(SMOKE_CONFIG_PATH)


@pytest.mark.parametrize(
    ("argv", "expected_substring"),
    [
        (["prepare", "--config", SMOKE_CONFIG], "prepare"),
        (["run", "--baseline", "clip_probe", "--config", SMOKE_CONFIG], "clip_probe"),
        (["evaluate", "--config", SMOKE_CONFIG], "evaluate"),
        (["report", "--config", SMOKE_CONFIG], "report"),
    ],
)
def test_subcommand_prints_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected_substring: str,
) -> None:
    exit_code = main(argv)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert expected_substring in captured.out

    # Only placeholders print the config path explicitly now
    if argv[0] != "prepare":
        assert SMOKE_CONFIG in captured.out


def test_cli_prints_project_phase(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["prepare", "--config", SMOKE_CONFIG])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "phase_ab_smoke" in captured.out


def test_run_with_invalid_baseline_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--baseline", "invalid", "--config", SMOKE_CONFIG])


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
