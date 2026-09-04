import json
import re
import sys
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import aiforensics.runs.artifacts as artifacts
from aiforensics.runs import (
    RunStatus,
    clip_seed_from_run_id,
    create_run_dir,
    write_environment,
    write_status,
)


def _make_status(status: str = "completed", reason: str | None = None) -> RunStatus:
    return RunStatus(
        baseline="qwen_vl",
        status=status,
        reason=reason,
        command=["aiforensics", "run", "--baseline", "qwen_vl"],
        started_at="2026-09-01T04:15:30+00:00",
        ended_at="2026-09-01T04:15:31+00:00",
    )


def test_create_run_dir_creates_directory_under_output_root(tmp_path):
    run_dir = create_run_dir(tmp_path, "clip_probe")
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path


def test_run_dir_name_uses_utc_timestamp_format(tmp_path):
    run_dir = create_run_dir(tmp_path, "clip_probe")
    assert re.fullmatch(r"\d{8}T\d{12}Z_clip_probe", run_dir.name)


def test_run_dir_name_contains_normalized_baseline(tmp_path):
    run_dir = create_run_dir(tmp_path, "CLIP Probe")
    assert "clip-probe" in run_dir.name


def test_valid_run_name_appears_in_run_dir_name(tmp_path):
    run_dir = create_run_dir(tmp_path, "clip_probe", run_name="seed70")
    assert "seed70" in run_dir.name
    assert run_dir.name.endswith("_clip_probe_seed70")


class TestClipSeedFromRunId:
    """The run-id seed convention has one parser shared by every consumer."""

    def test_seed_is_parsed_from_cli_created_run_id(self, tmp_path):
        run_dir = create_run_dir(tmp_path, "clip_probe", run_name="seed71")
        assert clip_seed_from_run_id(run_dir.name) == 71

    def test_multi_digit_seed_is_parsed(self):
        assert clip_seed_from_run_id("20260101T000000000000Z_clip_probe_seed1234") == 1234

    def test_suffixless_clip_run_has_no_seed(self):
        assert clip_seed_from_run_id("20260101T000000000000Z_clip_probe") is None

    def test_other_baseline_run_has_no_seed(self):
        assert clip_seed_from_run_id("20260101T000000000000Z_qwen_vl") is None

    def test_suffix_must_be_at_the_end(self):
        assert clip_seed_from_run_id("001_clip_probe_seed70_extra") is None


def test_run_name_none_omits_suffix(tmp_path):
    run_dir = create_run_dir(tmp_path, "clip_probe", run_name=None)
    assert re.fullmatch(r"\d{8}T\d{12}Z_clip_probe", run_dir.name)


def test_unsafe_characters_are_normalized_without_escaping(tmp_path):
    run_dir = create_run_dir(tmp_path, "../ETC/Passwd", run_name="/abs/path")
    assert run_dir.parent == tmp_path
    assert "/" not in run_dir.name
    assert "\\" not in run_dir.name
    assert ".." not in run_dir.name
    assert "etc-passwd" in run_dir.name
    assert "abs-path" in run_dir.name


def test_empty_baseline_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        create_run_dir(tmp_path, "")
    with pytest.raises(ValueError):
        create_run_dir(tmp_path, "   ")
    with pytest.raises(ValueError):
        create_run_dir(tmp_path, "///")


def test_whitespace_only_run_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="run_name"):
        create_run_dir(tmp_path, "clip_probe", run_name="")
    with pytest.raises(ValueError, match="run_name"):
        create_run_dir(tmp_path, "clip_probe", run_name="   ")


def test_existing_run_dir_is_not_silently_reused(tmp_path, monkeypatch):
    fixed = datetime(2026, 9, 1, 4, 15, 30, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(artifacts, "_utc_now", lambda: fixed)

    first = create_run_dir(tmp_path, "clip_probe")
    marker = first / "logs.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "clip_probe")

    assert marker.read_text(encoding="utf-8") == "keep me"


def test_write_environment_creates_valid_json_with_required_keys(tmp_path):
    env_path = tmp_path / "run1" / "environment.json"
    write_environment(env_path)

    data = json.loads(env_path.read_text(encoding="utf-8"))
    for key in (
        "captured_at",
        "python_version",
        "python_executable",
        "platform",
        "command",
        "packages",
    ):
        assert key in data
    assert data["captured_at"].endswith("+00:00")
    assert env_path.read_text(encoding="utf-8").endswith("\n")


def test_write_environment_records_command_as_json_list(tmp_path, monkeypatch):
    argv = ["aiforensics", "run", "--baseline", "clip_probe"]
    monkeypatch.setattr(sys, "argv", argv)

    env_path = tmp_path / "environment.json"
    write_environment(env_path)

    data = json.loads(env_path.read_text(encoding="utf-8"))
    assert isinstance(data["command"], list)
    assert data["command"] == argv


def test_write_environment_skips_unavailable_packages(tmp_path, monkeypatch):
    monkeypatch.setattr(
        artifacts,
        "_TRACKED_PACKAGES",
        ("numpy", "definitely-not-a-real-package-xyz"),
    )

    env_path = tmp_path / "environment.json"
    write_environment(env_path)

    data = json.loads(env_path.read_text(encoding="utf-8"))
    assert "numpy" in data["packages"]
    assert "definitely-not-a-real-package-xyz" not in data["packages"]


@pytest.mark.parametrize("status_value", ["completed", "failed", "deferred"])
def test_run_status_accepts_valid_statuses(status_value):
    status = _make_status(status=status_value)
    assert status.status == status_value


def test_run_status_rejects_invalid_status():
    with pytest.raises(ValidationError):
        _make_status(status="running")


def test_run_status_rejects_non_iso8601_timestamp():
    with pytest.raises(ValidationError, match="ISO-8601"):
        RunStatus(
            baseline="clip_probe",
            status="completed",
            reason=None,
            command=["aiforensics"],
            started_at="not-a-timestamp",
            ended_at="2026-09-01T04:15:31+00:00",
        )


def test_run_status_rejects_naive_timestamp_without_timezone():
    with pytest.raises(ValidationError, match="UTC offset"):
        RunStatus(
            baseline="clip_probe",
            status="completed",
            reason=None,
            command=["aiforensics"],
            started_at="2026-09-01T04:15:30",
            ended_at="2026-09-01T04:15:31+00:00",
        )


def test_run_status_rejects_non_utc_offset_timestamp():
    with pytest.raises(ValidationError, match="must be a UTC timestamp"):
        RunStatus(
            baseline="clip_probe",
            status="completed",
            reason=None,
            command=["aiforensics"],
            started_at="2026-09-01T04:15:30+07:00",
            ended_at="2026-09-01T04:15:31+00:00",
        )


def test_run_status_accepts_explicit_zero_offset_timestamp():
    status = RunStatus(
        baseline="clip_probe",
        status="completed",
        reason=None,
        command=["aiforensics"],
        started_at="2026-09-01T04:15:30+00:00",
        ended_at="2026-09-01T04:15:31Z",
    )
    assert status.started_at == "2026-09-01T04:15:30+00:00"


def test_run_status_accepts_z_suffix_timestamp():
    status = RunStatus(
        baseline="clip_probe",
        status="completed",
        reason=None,
        command=["aiforensics"],
        started_at="2026-09-01T04:15:30Z",
        ended_at="2026-09-01T04:15:31Z",
    )
    assert status.started_at == "2026-09-01T04:15:30Z"


def test_write_status_writes_all_six_required_fields(tmp_path):
    status_path = tmp_path / "run1" / "status.json"
    write_status(status_path, _make_status(status="failed", reason="boom"))

    data = json.loads(status_path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {
        "baseline",
        "status",
        "reason",
        "command",
        "started_at",
        "ended_at",
    }
    assert data["status"] == "failed"
    assert data["reason"] == "boom"
    assert status_path.read_text(encoding="utf-8").endswith("\n")


def test_write_status_preserves_reason_none_as_json_null(tmp_path):
    status_path = tmp_path / "status.json"
    write_status(status_path, _make_status())

    raw = status_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["reason"] is None
    assert '"reason": null' in raw
