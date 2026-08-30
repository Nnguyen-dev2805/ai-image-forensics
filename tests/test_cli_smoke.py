"""Smoke tests for the aiforensics CLI skeleton."""

from __future__ import annotations

import pytest

from aiforensics.cli.main import build_parser, main


SMOKE_CONFIG = "configs/phase_ab_smoke.yaml"


def test_prepare_command_prints_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["prepare", "--config", SMOKE_CONFIG])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "prepare" in captured.out
    assert SMOKE_CONFIG in captured.out


def test_run_clip_probe_command_prints_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["run", "--baseline", "clip_probe", "--config", SMOKE_CONFIG]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "clip_probe" in captured.out
    assert SMOKE_CONFIG in captured.out


def test_evaluate_command_prints_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["evaluate", "--config", SMOKE_CONFIG])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "evaluate" in captured.out
    assert SMOKE_CONFIG in captured.out


def test_report_command_prints_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["report", "--config", SMOKE_CONFIG])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "report" in captured.out
    assert SMOKE_CONFIG in captured.out


def test_run_with_invalid_baseline_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--baseline", "invalid", "--config", SMOKE_CONFIG])


def test_module_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
