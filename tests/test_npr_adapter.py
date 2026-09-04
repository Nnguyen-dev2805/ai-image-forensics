"""Task 10 tests: NPR external adapter.

All default tests are network-free, clone-free, checkpoint-free, CUDA-free, and
Torch-free. External boundaries (checkout management, subprocess execution,
runtime availability) are mocked narrowly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aiforensics.baselines.base import RunResult
from aiforensics.baselines.npr import bridge
from aiforensics.baselines.npr.adapter import NPRAdapter
from aiforensics.baselines.npr.checkout import (
    OFFICIAL_NPR_REPO_URL,
    CheckoutInfo,
    ensure_npr_checkout,
    normalize_repo_url,
    resolve_repo_dir,
)
from aiforensics.baselines.npr.checkpoint import validate_checkpoint
from aiforensics.baselines.npr.errors import (
    NPRConfigError,
    NPRDeferredError,
    NPRRuntimeError,
)
from aiforensics.baselines.npr.preprocess import (
    CROP_SIZE,
    center_crop,
    preprocess_npr_genimage_v1,
    translate_duplicate,
)
from aiforensics.config.models import AppConfig
from aiforensics.data.manifest import ManifestRecord, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_COMMIT = "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a"


# ---------------------------------------------------------------------------
# fixtures / factories
# ---------------------------------------------------------------------------


def _make_record(
    tmp_path: Path,
    sample_id: str = "sample-001",
    label: str = "real",
    dataset: str | None = "tiny-genimage",
    with_image: bool = True,
) -> ManifestRecord:
    image_path = tmp_path / f"{sample_id}.png"
    if with_image:
        Image.new("RGB", (32, 32), color=(128, 64, 32)).save(image_path)
    checksum = hashlib.sha256(image_path.read_bytes()).hexdigest() if with_image else ("0" * 64)
    return ManifestRecord(
        sample_id=sample_id,
        path=image_path,
        label=label,  # type: ignore[arg-type]
        source="test-source",
        split="dev",  # type: ignore[arg-type]
        checksum=checksum,
        dataset=dataset,
    )


def _make_config(tmp_path: Path, **overrides) -> AppConfig:
    """Build an AppConfig directly (no YAML load, works under tmp_path)."""
    external_root = tmp_path / "external"
    external_root.mkdir(exist_ok=True)
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    checkpoint = tmp_path / "NPR.pth"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"fake-checkpoint-bytes")

    npr_values: dict = {
        "enabled": True,
        "repo_url": OFFICIAL_NPR_REPO_URL,
        "repo_commit": VERIFIED_COMMIT,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": None,
        "batch_size": 2,
        "allow_deferred": True,
    }
    runtime_values: dict = {
        "python": "3.10",
        "seed": 70,
        "device": "auto",
        "batch_size": 2,
        "num_workers": 0,
        "fail_fast": False,
    }
    for key, value in overrides.items():
        if key.startswith("npr_"):
            npr_values[key.removeprefix("npr_")] = value
        elif key.startswith("runtime_"):
            runtime_values[key.removeprefix("runtime_")] = value
        elif key == "dev_manifest":
            pass
        else:
            raise AssertionError(f"Unsupported override: {key}")

    from aiforensics.config.models import (
        AppConfig,
        AssistedQwenConfig,
        BaselinesConfig,
        ClipProbeConfig,
        DatasetsConfig,
        EvaluationConfig,
        GenImageUnseenConfig,
        LabelsConfig,
        NPRConfig,
        PathsConfig,
        ProjectConfig,
        QwenVLConfig,
        ReportConfig,
        RuntimeConfig,
        SynthbusterConfig,
        TinyGenImageConfig,
    )

    dev_manifest = overrides.get("dev_manifest", tmp_path / "manifests" / "dev_manifest.csv")
    config = AppConfig(
        project=ProjectConfig(name="npr-tests", phase="phase_ab", description="tests"),
        paths=PathsConfig(
            data_root=data_root,
            manifest_root=tmp_path / "manifests",
            cache_root=tmp_path / "cache",
            output_root=tmp_path / "outputs",
            external_root=external_root,
        ),
        runtime=RuntimeConfig(**runtime_values),
        datasets=DatasetsConfig(
            tiny_genimage=TinyGenImageConfig(
                enabled=True,
                source="tests",
                use_original_split=False,
                train_manifest=tmp_path / "manifests" / "train.csv",
                dev_manifest=dev_manifest,
            ),
            genimage_unseen=GenImageUnseenConfig(
                enabled=False,
                generators=["imagenet_midjourney"],
                max_images=8,
                balance_labels=True,
                split="external",
                manifest=tmp_path / "manifests" / "genimage_external.csv",
            ),
            synthbuster=SynthbusterConfig(
                enabled=False,
                max_images=8,
                balance_labels=True,
                split="external",
                manifest=tmp_path / "manifests" / "synthbuster_external.csv",
            ),
        ),
        baselines=BaselinesConfig(
            clip_probe=ClipProbeConfig(
                enabled=False,
                model_family="synthetic",
                model_name="smoke-embedding",
                pretrained="none",
                classifier="logistic_regression",
                seeds=[70],
                cache_embeddings=False,
            ),
            qwen_vl=QwenVLConfig(
                enabled=False,
                model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                prompt_id="qwen_json_v1",
                temperature=0.0,
                max_new_tokens=128,
                cache_outputs=False,
                allow_deferred=True,
            ),
            assisted_qwen=AssistedQwenConfig(
                enabled=False,
                base_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                prompt_id="assisted_qwen_json_v1",
                assistant_source="clip_probe",
                include_classifier_pred=True,
                include_fake_probability=True,
                temperature=0.0,
                max_new_tokens=128,
                cache_outputs=False,
                allow_deferred=True,
            ),
            npr=NPRConfig(**npr_values),
        ),
        evaluation=EvaluationConfig(
            labels=LabelsConfig(negative="real", positive="fake"),
            metrics=["accuracy"],
            group_by=["source"],
        ),
        report=ReportConfig(
            filename="report.md",
            include_failure_notes=True,
            include_explanations_sample=False,
            explanation_sample_size=0,
        ),
    )
    return config


def _write_dev_manifest(tmp_path: Path, records: list[ManifestRecord]) -> Path:
    manifest_path = tmp_path / "dev_manifest.csv"
    write_manifest(records, manifest_path)
    return manifest_path


@pytest.fixture()
def eval_setup(tmp_path: Path):
    """Config + two dev records; returns (config, records)."""
    records = [
        _make_record(tmp_path, "sample-001", "real"),
        _make_record(tmp_path, "sample-002", "fake"),
    ]
    manifest_path = _write_dev_manifest(tmp_path, records)
    config = _make_config(tmp_path)
    config.datasets.tiny_genimage.dev_manifest = manifest_path
    return config, records


class _FakeCheckout:
    """Records calls and returns a canned CheckoutInfo."""

    def __init__(self, info: CheckoutInfo | None = None, exc: Exception | None = None):
        self.calls: list[dict] = []
        self.info = info or CheckoutInfo(
            repo_dir=Path("/fake/repo"), resolved_commit=VERIFIED_COMMIT, action="reused"
        )
        self.exc = exc

    def __call__(self, *, repo_dir, repo_url, repo_commit, allow_deferred):
        self.calls.append(
            {
                "repo_dir": repo_dir,
                "repo_url": repo_url,
                "repo_commit": repo_commit,
                "allow_deferred": allow_deferred,
            }
        )
        if self.exc is not None:
            raise self.exc
        return self.info


def _patch_subprocess(monkeypatch, exit_code: int = 0, calls: list | None = None):
    """Patch adapter subprocess execution; optionally capture commands."""

    def fake_run(command, capture_output, text, check):
        if calls is not None:
            calls.append(command)
        return type("R", (), {"returncode": exit_code, "stdout": "", "stderr": ""})()

    import aiforensics.baselines.npr.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod.subprocess, "run", fake_run)


def _write_scores(path: Path, scores: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample_id, score in scores:
            f.write(json.dumps({"sample_id": sample_id, "score_fake": score}) + "\n")


def _run_with_mocks(
    monkeypatch,
    config: AppConfig,
    output_dir: Path,
    *,
    fake_checkout: _FakeCheckout | None = None,
    scores: list[tuple[str, float]] | None = None,
    subprocess_exit: int = 0,
    subprocess_calls: list | None = None,
    runtime_seed: int = 70,
) -> RunResult:
    """Run NPRAdapter with checkout + subprocess boundaries mocked.

    ``ensure_npr_checkout`` is always mocked (defaulting to a canned clean
    checkout) so tests stay network-free and clone-free; tests that exercise
    the real checkout helper call it directly with patched git commands.
    """
    if fake_checkout is None:
        fake_checkout = _FakeCheckout()
    monkeypatch.setattr(NPRAdapter, "_check_runtime_availability", lambda self, device: device)
    import aiforensics.baselines.npr.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "ensure_npr_checkout", fake_checkout)
    scores = scores or [("sample-001", 0.9), ("sample-002", 0.1)]

    original_run = NPRAdapter._run_subprocess

    def fake_run_subprocess(self, command, log_path):
        if subprocess_calls is not None:
            subprocess_calls.append(command)
        # emulate runtime writing scores
        output_jsonl = Path(command[command.index("--output-jsonl") + 1])
        if subprocess_exit == 0:
            _write_scores(output_jsonl, scores)
        return subprocess_exit

    monkeypatch.setattr(NPRAdapter, "_run_subprocess", fake_run_subprocess)

    try:
        return NPRAdapter().run(
            config=config,
            output_dir=output_dir,
            run_id=output_dir.name,
            seed=runtime_seed,
        )
    finally:
        # restore for subsequent tests
        NPRAdapter._run_subprocess = original_run  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 1. identity
# ---------------------------------------------------------------------------


def test_npr_adapter_name():
    assert NPRAdapter.name == "npr"


# ---------------------------------------------------------------------------
# 2-5. disabled behavior (validated before any placeholder/smoke checks)
# ---------------------------------------------------------------------------


def test_disabled_npr_defers_before_repo_validation(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.npr.enabled = False
    # Deliberately corrupt remaining settings; disabled must win regardless.
    config.baselines.npr.repo_commit = "smoke-disabled"
    config.baselines.npr.checkpoint_path = tmp_path / "does-not-exist.pth"

    fake_checkout = _FakeCheckout(exc=AssertionError("checkout must not be called"))
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    result = _run_with_mocks(monkeypatch, config, output_dir, fake_checkout=fake_checkout)
    assert result.status == "deferred"
    assert fake_checkout.calls == []


def test_disabled_smoke_npr_does_not_run_git(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.npr.enabled = False

    real_subprocess_run = __import__("subprocess").run

    def no_git(command, *args, **kwargs):
        if isinstance(command, list) and command and command[0] == "git":
            raise AssertionError("git must not run for disabled NPR")
        return real_subprocess_run(command, *args, **kwargs)

    import subprocess as subprocess_mod

    monkeypatch.setattr(subprocess_mod, "run", no_git)

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = NPRAdapter().run(config=config, output_dir=output_dir, run_id=output_dir.name)
    assert result.status == "deferred"


def test_disabled_smoke_npr_does_not_inspect_checkpoint(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.npr.enabled = False
    config.baselines.npr.checkpoint_path = Path("/definitely/not/a/file.pth")

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    called = {"checkpoint": False}
    real_validate = validate_checkpoint

    def spy_validate(path, sha):
        called["checkpoint"] = True
        return real_validate(path, sha)

    import aiforensics.baselines.npr.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod.checkpoint_mod, "validate_checkpoint", spy_validate)

    result = NPRAdapter().run(config=config, output_dir=output_dir, run_id=output_dir.name)
    assert result.status == "deferred"
    assert called["checkpoint"] is False


def test_disabled_smoke_npr_does_not_import_torch(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.npr.enabled = False

    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("torch must not be imported for disabled NPR")
        return real_import(name, *args, **kwargs)

    import builtins

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = NPRAdapter().run(config=config, output_dir=output_dir, run_id=output_dir.name)
    assert result.status == "deferred"


# ---------------------------------------------------------------------------
# 6-8. repo URL / path
# ---------------------------------------------------------------------------


def test_official_repo_url_accepted():
    assert normalize_repo_url(OFFICIAL_NPR_REPO_URL) == OFFICIAL_NPR_REPO_URL


def test_official_repo_url_trailing_slash_and_git_suffix():
    assert normalize_repo_url(OFFICIAL_NPR_REPO_URL + "/") == OFFICIAL_NPR_REPO_URL
    assert normalize_repo_url(OFFICIAL_NPR_REPO_URL + ".git") == OFFICIAL_NPR_REPO_URL


def test_fork_repo_url_fails():
    with pytest.raises(NPRConfigError):
        normalize_repo_url("https://github.com/someone-else/NPR-DeepfakeDetection")


def test_repo_path_resolves_under_external_root(tmp_path):
    repo_dir = resolve_repo_dir(tmp_path)
    assert repo_dir == tmp_path / "NPR-DeepfakeDetection"


# ---------------------------------------------------------------------------
# 9. commit pin
# ---------------------------------------------------------------------------


def test_missing_real_run_commit_pin_fails(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.npr.repo_commit = None

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "failed"
    assert "repo_commit" in (result.reason or "")


def test_branch_name_commit_pin_fails(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.npr.repo_commit = "main"

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "failed"


def test_smoke_placeholder_commit_pin_fails_when_enabled(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.npr.repo_commit = "smoke-disabled"

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "failed"


# ---------------------------------------------------------------------------
# 10-15. checkout helper
# ---------------------------------------------------------------------------


def test_checkout_helper_represents_clean_checkout(tmp_path):
    info = CheckoutInfo(repo_dir=tmp_path, resolved_commit=VERIFIED_COMMIT, action="reused")
    assert info.resolved_commit == VERIFIED_COMMIT
    assert info.repo_dir == tmp_path


def test_ensure_checkout_accepts_existing_clean_checkout(tmp_path, monkeypatch):
    """A fully mocked 'clean official checkout at the configured commit' is reused."""
    repo_dir = tmp_path / "NPR-DeepfakeDetection"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    import aiforensics.baselines.npr.checkout as checkout_mod

    monkeypatch.setattr(
        checkout_mod,
        "_run_git",
        lambda repo, *args: (
            OFFICIAL_NPR_REPO_URL
            if args[:2] == ("remote", "get-url")
            else ("" if args[:1] == ("status",) else VERIFIED_COMMIT)
        ),
    )

    info = ensure_npr_checkout(
        repo_dir=repo_dir,
        repo_url=OFFICIAL_NPR_REPO_URL,
        repo_commit=VERIFIED_COMMIT,
        allow_deferred=True,
    )
    assert info.action == "reused"
    assert info.resolved_commit == VERIFIED_COMMIT


def test_dirty_existing_checkout_fails(tmp_path, monkeypatch):
    repo_dir = tmp_path / "NPR-DeepfakeDetection"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    import aiforensics.baselines.npr.checkout as checkout_mod

    def fake_git(repo, *args):
        if args[:2] == ("remote", "get-url"):
            return OFFICIAL_NPR_REPO_URL
        if args[:1] == ("status",):
            return " M dirty_file.py"
        raise AssertionError("unexpected git call")

    monkeypatch.setattr(checkout_mod, "_run_git", fake_git)

    with pytest.raises(NPRConfigError, match="uncommitted changes"):
        ensure_npr_checkout(
            repo_dir=repo_dir,
            repo_url=OFFICIAL_NPR_REPO_URL,
            repo_commit=VERIFIED_COMMIT,
            allow_deferred=True,
        )


def test_wrong_existing_remote_fails(tmp_path, monkeypatch):
    repo_dir = tmp_path / "NPR-DeepfakeDetection"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    import aiforensics.baselines.npr.checkout as checkout_mod

    monkeypatch.setattr(
        checkout_mod,
        "_run_git",
        lambda repo, *args: (
            "https://github.com/fork/NPR-DeepfakeDetection"
            if args[:2] == ("remote", "get-url")
            else ""
        ),
    )

    with pytest.raises(NPRConfigError, match="Unsupported NPR repo_url"):
        ensure_npr_checkout(
            repo_dir=repo_dir,
            repo_url=OFFICIAL_NPR_REPO_URL,
            repo_commit=VERIFIED_COMMIT,
            allow_deferred=True,
        )


def test_commit_mismatch_cannot_be_silently_accepted(tmp_path, monkeypatch):
    repo_dir = tmp_path / "NPR-DeepfakeDetection"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    import aiforensics.baselines.npr.checkout as checkout_mod

    def fake_git(repo, *args):
        if args[:2] == ("remote", "get-url"):
            return OFFICIAL_NPR_REPO_URL
        if args[:1] == ("status",):
            return ""
        if args[:1] == ("rev-parse",):
            return "f" * 40  # HEAD is a different commit
        raise AssertionError("unexpected git call")

    monkeypatch.setattr(checkout_mod, "_run_git", fake_git)
    monkeypatch.setattr(
        checkout_mod,
        "_fetch_commit",
        lambda repo, commit, *, allow_deferred: False,
    )

    with pytest.raises(NPRConfigError, match="could not be resolved"):
        ensure_npr_checkout(
            repo_dir=repo_dir,
            repo_url=OFFICIAL_NPR_REPO_URL,
            repo_commit=VERIFIED_COMMIT,
            allow_deferred=True,
        )


def test_missing_repo_network_failure_defers_when_allowed(tmp_path, monkeypatch):
    repo_dir = tmp_path / "NPR-DeepfakeDetection"

    import subprocess as subprocess_mod

    def failing_clone(command, capture_output, text, check):
        raise OSError("network unreachable")

    monkeypatch.setattr(subprocess_mod, "run", failing_clone)

    with pytest.raises(NPRDeferredError):
        ensure_npr_checkout(
            repo_dir=repo_dir,
            repo_url=OFFICIAL_NPR_REPO_URL,
            repo_commit=VERIFIED_COMMIT,
            allow_deferred=True,
        )


def test_same_checkout_setup_failure_fails_when_not_allowed(tmp_path, monkeypatch):
    repo_dir = tmp_path / "NPR-DeepfakeDetection"

    import subprocess as subprocess_mod

    def failing_clone(command, capture_output, text, check):
        raise OSError("network unreachable")

    monkeypatch.setattr(subprocess_mod, "run", failing_clone)

    with pytest.raises(NPRConfigError):
        ensure_npr_checkout(
            repo_dir=repo_dir,
            repo_url=OFFICIAL_NPR_REPO_URL,
            repo_commit=VERIFIED_COMMIT,
            allow_deferred=False,
        )


def test_clone_creates_missing_external_root(tmp_path, monkeypatch):
    """ensure_npr_checkout must create external_root before cloning instead of
    misreporting a missing parent directory as a network failure."""
    external_root = tmp_path / "external"
    repo_dir = external_root / "NPR-DeepfakeDetection"
    assert not external_root.exists()

    import subprocess as subprocess_mod

    observed: dict = {}

    def fake_clone(command, capture_output, text, check):
        observed["parent_existed"] = repo_dir.parent.is_dir()
        observed["destination"] = command[-1]
        # Simulate a successful clone by materializing the checkout skeleton.
        Path(command[-1]).mkdir(parents=True, exist_ok=True)
        (Path(command[-1]) / ".git").mkdir()
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess_mod, "run", fake_clone)

    import aiforensics.baselines.npr.checkout as checkout_mod

    monkeypatch.setattr(checkout_mod, "_git_version", lambda: True)
    monkeypatch.setattr(
        checkout_mod,
        "_run_git",
        lambda repo, *args: VERIFIED_COMMIT,  # checkout + rev-parse HEAD
    )

    info = ensure_npr_checkout(
        repo_dir=repo_dir,
        repo_url=OFFICIAL_NPR_REPO_URL,
        repo_commit=VERIFIED_COMMIT,
        allow_deferred=True,
    )
    assert observed["parent_existed"] is True
    assert observed["destination"] == str(repo_dir)
    assert info.action == "cloned"


# ---------------------------------------------------------------------------
# 16-19. checkpoint validation
# ---------------------------------------------------------------------------


def test_missing_checkpoint_defers_when_allowed(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.npr.checkpoint_path = tmp_path / "missing.pth"

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "deferred"


def test_missing_checkpoint_fails_when_not_allowed(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.baselines.npr.checkpoint_path = tmp_path / "missing.pth"
    config.baselines.npr.allow_deferred = False

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "failed"


def test_checkpoint_sha256_match_passes(tmp_path):
    checkpoint = tmp_path / "NPR.pth"
    checkpoint.write_bytes(b"deterministic-bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert validate_checkpoint(checkpoint, digest) is True


def test_checkpoint_sha256_mismatch_fails_and_never_defers(tmp_path, monkeypatch):
    checkpoint = tmp_path / "NPR.pth"
    checkpoint.write_bytes(b"deterministic-bytes")

    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint(checkpoint, "a" * 64)

    # The adapter must convert this to failed regardless of allow_deferred.
    config = _make_config(tmp_path)
    config.baselines.npr.checkpoint_path = checkpoint
    config.baselines.npr.checkpoint_sha256 = "b" * 64

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "failed"


# ---------------------------------------------------------------------------
# 20-25. evaluation record selection
# ---------------------------------------------------------------------------


def test_disabled_optional_datasets_ignored_even_when_manifests_exist(tmp_path):
    config = _make_config(tmp_path)
    # smoke config has genimage_unseen/synthbuster disabled by default
    config.datasets.genimage_unseen.manifest = tmp_path / "exists_but_disabled.csv"
    config.datasets.synthbuster.manifest = tmp_path / "also_disabled.csv"
    config.datasets.genimage_unseen.manifest.write_text("x")
    config.datasets.synthbuster.manifest.write_text("x")

    records = [_make_record(tmp_path, "sample-001", "real")]
    config.datasets.tiny_genimage.dev_manifest = _write_dev_manifest(tmp_path, records)

    selected = NPRAdapter()._select_eval_records(config)
    assert [r.sample_id for r in selected] == ["sample-001"]


def test_missing_tiny_dev_allowed_when_external_manifest_exists(tmp_path):
    config = _make_config(tmp_path)
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "nope.csv"
    external = tmp_path / "genimage_external.csv"
    external_records = [_make_record(tmp_path, "ext-001", "fake")]
    write_manifest(external_records, external)
    config.datasets.genimage_unseen.enabled = True
    config.datasets.genimage_unseen.manifest = external

    selected = NPRAdapter()._select_eval_records(config)
    assert [r.sample_id for r in selected] == ["ext-001"]


def test_missing_tiny_dev_and_no_external_fails(tmp_path):
    config = _make_config(tmp_path)
    config.datasets.tiny_genimage.dev_manifest = tmp_path / "nope.csv"

    with pytest.raises(NPRConfigError, match="No enabled evaluation manifest exists"):
        NPRAdapter()._select_eval_records(config)


def test_disabled_tiny_genimage_is_not_loaded(tmp_path):
    """datasets.tiny_genimage.enabled=false excludes tiny even when it exists."""
    config = _make_config(tmp_path)
    tiny_record = _make_record(tmp_path, "tiny-001", "real")
    config.datasets.tiny_genimage.dev_manifest = _write_dev_manifest(tmp_path, [tiny_record])
    external = tmp_path / "external.csv"
    write_manifest([_make_record(tmp_path, "ext-001", "fake")], external)
    config.datasets.genimage_unseen.enabled = True
    config.datasets.genimage_unseen.manifest = external
    config.datasets.tiny_genimage.enabled = False

    selected = NPRAdapter()._select_eval_records(config)
    assert [r.sample_id for r in selected] == ["ext-001"]


def test_all_datasets_disabled_fails(tmp_path):
    """A config that enables no dataset has nothing to evaluate."""
    config = _make_config(tmp_path)
    config.datasets.tiny_genimage.dev_manifest = _write_dev_manifest(
        tmp_path, [_make_record(tmp_path, "tiny-001", "real")]
    )
    config.datasets.tiny_genimage.enabled = False
    config.datasets.genimage_unseen.enabled = False
    config.datasets.synthbuster.enabled = False

    with pytest.raises(NPRConfigError, match="no dataset is enabled in config"):
        NPRAdapter()._select_eval_records(config)


def test_existing_invalid_manifest_fails_rather_than_skipped(tmp_path):
    config = _make_config(tmp_path)
    bad_manifest = tmp_path / "bad.csv"
    bad_manifest.write_text("not,a,valid,manifest\n1,2,3\n")
    config.datasets.tiny_genimage.dev_manifest = bad_manifest

    from aiforensics.data.manifest import ManifestError

    with pytest.raises(ManifestError):
        NPRAdapter()._select_eval_records(config)


def test_duplicate_sample_ids_across_manifests_fail(tmp_path):
    config = _make_config(tmp_path)
    shared_id_record = _make_record(tmp_path, "dup-001", "real")
    config.datasets.tiny_genimage.dev_manifest = _write_dev_manifest(tmp_path, [shared_id_record])
    external = tmp_path / "external.csv"
    write_manifest([_make_record(tmp_path, "dup-001", "fake")], external)
    config.datasets.genimage_unseen.enabled = True
    config.datasets.genimage_unseen.manifest = external

    with pytest.raises(NPRConfigError, match="Duplicate evaluation sample_id"):
        NPRAdapter()._select_eval_records(config)


def test_missing_image_fails(tmp_path, eval_setup):
    config, records = eval_setup
    records[0].path.unlink()
    with pytest.raises(NPRConfigError, match="Missing image"):
        NPRAdapter()._validate_images(records)


def test_image_checksum_mismatch_fails(tmp_path, eval_setup):
    config, records = eval_setup
    records[0].checksum = "c" * 64
    with pytest.raises(NPRConfigError, match="Checksum mismatch"):
        NPRAdapter()._validate_images(records)


# ---------------------------------------------------------------------------
# 26-29. npr_genimage_v1 preprocessing
# ---------------------------------------------------------------------------


def test_preprocessing_tiles_undersized_image():
    img = Image.new("RGB", (100, 50), color=(10, 200, 30))
    tiled = translate_duplicate(img)
    assert tiled.size[0] >= CROP_SIZE
    assert tiled.size[1] >= CROP_SIZE
    # Tiling must duplicate, not resize: each tile keeps original pixels.
    assert tiled.size[0] % 100 == 0
    assert tiled.size[1] % 50 == 0


def test_preprocessing_leaves_larger_image_unresized_before_crop():
    img = Image.new("RGB", (300, 280))
    tiled = translate_duplicate(img)
    assert tiled.size == (300, 280)


def test_preprocessing_output_is_exactly_224():
    for size in [(100, 50), (300, 280), (224, 224)]:
        arr = preprocess_npr_genimage_v1(Image.new("RGB", size))
        assert arr.shape == (3, CROP_SIZE, CROP_SIZE)


def test_preprocessing_is_deterministic():
    rng = np.random.default_rng(70)
    pixels = rng.integers(0, 255, size=(180, 160, 3), dtype=np.uint8)
    img = Image.fromarray(pixels)
    a = preprocess_npr_genimage_v1(img)
    b = preprocess_npr_genimage_v1(img)
    assert np.array_equal(a, b)


def test_preprocessing_uses_imagenet_normalization():
    gray = Image.new("RGB", (CROP_SIZE, CROP_SIZE), color=(127, 127, 127))
    arr = preprocess_npr_genimage_v1(gray)
    expected = (127.0 / 255.0 - 0.485) / 0.229
    assert arr.shape == (3, CROP_SIZE, CROP_SIZE)
    assert np.allclose(arr[0], expected, atol=1e-4)


def test_center_crop_matches_torchvision_rounding():
    # torchvision CenterCrop rounds with int(floor + 0.5) semantics.
    img = Image.new("RGB", (351, 229))
    cropped = center_crop(img, CROP_SIZE)
    assert cropped.size == (CROP_SIZE, CROP_SIZE)
    left = int(round((351 - CROP_SIZE) / 2.0))
    top = int(round((229 - CROP_SIZE) / 2.0))
    assert np.array_equal(
        np.asarray(cropped), np.asarray(img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE)))
    )


# ---------------------------------------------------------------------------
# 30-31. runtime input JSONL
# ---------------------------------------------------------------------------


def test_runtime_input_excludes_ground_truth(tmp_path, eval_setup):
    config, records = eval_setup
    rows = bridge.build_runtime_input_rows(records)
    for row in rows:
        assert set(row.keys()) == {"sample_id", "path"}
        assert "label" not in json.dumps(row)


def test_runtime_input_preserves_order_and_absolute_paths(tmp_path, eval_setup):
    config, records = eval_setup
    rows = bridge.build_runtime_input_rows(records)
    assert [r["sample_id"] for r in rows] == ["sample-001", "sample-002"]
    for row in rows:
        assert Path(row["path"]).is_absolute()

    out = tmp_path / "npr_input.jsonl"
    bridge.write_runtime_input_jsonl(rows, out)
    loaded = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["sample_id"] for r in loaded] == ["sample-001", "sample-002"]


# ---------------------------------------------------------------------------
# 32-35, 40-41. score-to-prediction conversion
# ---------------------------------------------------------------------------


def test_valid_scores_convert_one_to_one(eval_setup):
    config, records = eval_setup
    scores = [("sample-001", 0.9), ("sample-002", 0.123456)]
    predictions = bridge.build_npr_predictions(records, scores, run_id="run-1")
    assert len(predictions) == 2
    assert [p.sample_id for p in predictions] == ["sample-001", "sample-002"]


def test_score_strictly_above_half_is_fake(eval_setup):
    config, records = eval_setup
    predictions = bridge.build_npr_predictions(
        records, [("sample-001", 0.5000001), ("sample-002", 0.8)], run_id="run-1"
    )
    assert predictions[0].label_pred == "fake"


def test_score_exactly_half_is_real(eval_setup):
    config, records = eval_setup
    predictions = bridge.build_npr_predictions(
        records, [("sample-001", 0.5), ("sample-002", 0.2)], run_id="run-1"
    )
    assert predictions[0].label_pred == "real"


def test_score_below_half_is_real(eval_setup):
    config, records = eval_setup
    predictions = bridge.build_npr_predictions(
        records, [("sample-001", 0.4999), ("sample-002", 0.1)], run_id="run-1"
    )
    assert predictions[0].label_pred == "real"


def test_npr_prediction_metadata_fields(eval_setup):
    config, records = eval_setup
    predictions = bridge.build_npr_predictions(
        records, [("sample-001", 0.9), ("sample-002", 0.1)], run_id="run-1"
    )
    rec = predictions[0]
    assert rec.model_name == "npr"
    assert rec.parse_status == "not_applicable"
    assert rec.prompt_id is None
    assert rec.raw_output is None
    assert rec.explanation is None


def test_score_order_misalignment_fails(eval_setup):
    config, records = eval_setup
    with pytest.raises(NPRRuntimeError, match="misalignment"):
        bridge.build_npr_predictions(
            records, [("sample-002", 0.9), ("sample-001", 0.1)], run_id="run-1"
        )


# ---------------------------------------------------------------------------
# 36-39. runtime score validation
# ---------------------------------------------------------------------------


def test_nan_inf_out_of_range_scores_fail():
    with pytest.raises(NPRRuntimeError):
        bridge.validate_runtime_scores([{"sample_id": "a", "score_fake": float("nan")}], ["a"])
    with pytest.raises(NPRRuntimeError):
        bridge.validate_runtime_scores([{"sample_id": "a", "score_fake": float("inf")}], ["a"])
    with pytest.raises(NPRRuntimeError):
        bridge.validate_runtime_scores([{"sample_id": "a", "score_fake": 1.5}], ["a"])
    with pytest.raises(NPRRuntimeError):
        bridge.validate_runtime_scores([{"sample_id": "a", "score_fake": -0.1}], ["a"])


def test_duplicate_runtime_score_sample_id_fails():
    with pytest.raises(NPRRuntimeError, match="Duplicate"):
        bridge.validate_runtime_scores(
            [
                {"sample_id": "a", "score_fake": 0.1},
                {"sample_id": "a", "score_fake": 0.2},
            ],
            ["a"],
        )


def test_unknown_extra_runtime_sample_id_fails():
    with pytest.raises(NPRRuntimeError, match="Unknown"):
        bridge.validate_runtime_scores(
            [
                {"sample_id": "a", "score_fake": 0.1},
                {"sample_id": "extra", "score_fake": 0.2},
            ],
            ["a"],
        )


def test_missing_runtime_sample_id_fails():
    with pytest.raises(NPRRuntimeError, match="Missing"):
        bridge.validate_runtime_scores([{"sample_id": "a", "score_fake": 0.1}], ["a", "b"])


def test_non_numeric_runtime_score_fails():
    with pytest.raises(NPRRuntimeError):
        bridge.validate_runtime_scores([{"sample_id": "a", "score_fake": "0.9"}], ["a"])


# ---------------------------------------------------------------------------
# 42-46. subprocess boundary + adapter orchestration
# ---------------------------------------------------------------------------


def test_subprocess_command_uses_sys_executable(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    calls: list[list[str]] = []
    _run_with_mocks(monkeypatch, config, output_dir, subprocess_calls=calls)
    assert calls, "subprocess must be invoked"
    command = calls[0]
    assert command[0] == sys.executable
    assert command[0] != "python"
    assert command[1:3] == ["-m", "aiforensics.baselines.npr.runtime"]


def test_runtime_setup_failure_defers_when_allowed(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir, subprocess_exit=2)
    assert result.status == "deferred"


def test_runtime_setup_failure_fails_when_not_allowed(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    config.baselines.npr.allow_deferred = False
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir, subprocess_exit=2)
    assert result.status == "failed"


def test_inference_failure_after_setup_fails(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir, subprocess_exit=1)
    assert result.status == "failed"


def test_invalid_scores_fail_the_run(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(
        monkeypatch,
        config,
        output_dir,
        scores=[("sample-001", float("nan")), ("sample-002", 0.1)],
    )
    assert result.status == "failed"


def test_completed_run_writes_and_reads_back_predictions(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "completed"
    assert result.prediction_path is not None
    assert result.prediction_path.exists()

    from aiforensics.schemas.predictions import load_predictions, validate_predictions

    loaded = load_predictions(result.prediction_path)
    assert len(loaded) == 2
    recheck = validate_predictions(loaded, require_mllm_fields=False)
    assert recheck.is_valid


def test_bridge_artifacts_written_on_completed_run(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "completed"
    assert (output_dir / "npr_input.jsonl").exists()
    assert (output_dir / "npr_scores.jsonl").exists()
    input_rows = [
        json.loads(line) for line in (output_dir / "npr_input.jsonl").read_text().splitlines()
    ]
    assert all("label" not in row for row in input_rows)


def test_failed_run_leaves_no_stale_predictions(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "predictions.jsonl").touch()  # stale artifact from a prior attempt

    result = _run_with_mocks(monkeypatch, config, output_dir, scores=[("sample-001", 0.9)])
    assert result.status == "failed"  # missing score for sample-002
    assert not (output_dir / "predictions.jsonl").exists()


def test_cuda_device_unavailable_defers_when_allowed(tmp_path, eval_setup, monkeypatch):
    import aiforensics.baselines.npr.adapter as adapter_mod

    config, records = eval_setup

    def no_cuda(self, device):
        raise NPRDeferredError("CUDA is not available")

    monkeypatch.setattr(adapter_mod.NPRAdapter, "_check_runtime_availability", no_cuda)
    monkeypatch.setattr(adapter_mod, "ensure_npr_checkout", _FakeCheckout())

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = NPRAdapter().run(config=config, output_dir=output_dir, run_id=output_dir.name)
    assert result.status == "deferred"


def test_cpu_device_defers(tmp_path, eval_setup, monkeypatch):
    import aiforensics.baselines.npr.adapter as adapter_mod

    config, records = eval_setup
    config.runtime.device = "cpu"
    monkeypatch.setattr(adapter_mod, "ensure_npr_checkout", _FakeCheckout())
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = NPRAdapter().run(config=config, output_dir=output_dir, run_id=output_dir.name)
    assert result.status == "deferred"


# ---------------------------------------------------------------------------
# runtime runner (pure, injected boundaries)
# ---------------------------------------------------------------------------


def _write_runtime_input(path: Path, sample_ids: list[str], tmp_path: Path):
    rows = []
    for sid in sample_ids:
        img = tmp_path / f"{sid}.png"
        Image.new("RGB", (224, 224)).save(img)
        rows.append({"sample_id": sid, "path": str(img)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _runtime_args(tmp_path: Path, sample_ids: list[str]):
    import argparse

    input_path = tmp_path / "npr_input.jsonl"
    _write_runtime_input(input_path, sample_ids, tmp_path)
    return argparse.Namespace(
        repo_dir=tmp_path / "repo",
        checkpoint=tmp_path / "NPR.pth",
        input_jsonl=input_path,
        output_jsonl=tmp_path / "npr_scores.jsonl",
        batch_size=2,
        seed=70,
        device="cuda",
    )


def _install_fake_torch(monkeypatch) -> types.ModuleType:
    """Install a minimal fake torch module so runtime paths stay Torch-free."""
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda seed: None  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def test_runtime_setup_failure_returns_exit_code_2(tmp_path):
    from aiforensics.baselines.npr import runtime as runtime_mod

    args = _runtime_args(tmp_path, ["s1"])

    def failing_loader(repo_dir, checkpoint, device):
        raise RuntimeError("no CUDA")

    code = runtime_mod.run_runtime(
        args,
        seed_setter=lambda seed: None,
        device_resolver=lambda device: "cuda",
        model_loader=failing_loader,
    )
    assert code == runtime_mod.SETUP_EXIT_CODE


def test_runtime_inference_failure_returns_exit_code_1(tmp_path):
    from aiforensics.baselines.npr import runtime as runtime_mod

    args = _runtime_args(tmp_path, ["s1"])

    def ok_loader(repo_dir, checkpoint, device):
        return object()

    def failing_batch(model, batch_array, device):
        raise RuntimeError("inference crashed")

    code = runtime_mod.run_runtime(
        args,
        seed_setter=lambda seed: None,
        device_resolver=lambda device: "cuda",
        model_loader=ok_loader,
        batch_runner=failing_batch,
    )
    assert code == runtime_mod.INFERENCE_EXIT_CODE


def test_runtime_success_emits_one_score_per_sample(tmp_path):
    from aiforensics.baselines.npr import runtime as runtime_mod

    args = _runtime_args(tmp_path, ["s1", "s2", "s3"])
    loader_calls = []

    def ok_loader(repo_dir, checkpoint, device):
        loader_calls.append(1)
        return object()

    def fake_batch(model, batch_array, device):
        return [0.75] * len(batch_array)

    code = runtime_mod.run_runtime(
        args,
        seed_setter=lambda seed: None,
        device_resolver=lambda device: "cuda",
        model_loader=ok_loader,
        batch_runner=fake_batch,
    )
    assert code == 0
    assert len(loader_calls) == 1  # model loaded once per runner process

    rows = [json.loads(line) for line in args.output_jsonl.read_text().splitlines()]
    assert [r["sample_id"] for r in rows] == ["s1", "s2", "s3"]
    assert all(0.0 <= r["score_fake"] <= 1.0 for r in rows)


def test_runtime_rejects_non_cuda_device(tmp_path, monkeypatch):
    """The default resolver rejects non-CUDA devices without importing real Torch."""
    from aiforensics.baselines.npr import runtime as runtime_mod

    _install_fake_torch(monkeypatch)

    args = _runtime_args(tmp_path, ["s1"])
    args.device = "cpu"
    code = runtime_mod.run_runtime(args)
    assert code == runtime_mod.SETUP_EXIT_CODE


# ---------------------------------------------------------------------------
# review fixes: auto-device resolution, fetch defer policy, checksum format,
# score order validation
# ---------------------------------------------------------------------------


def test_runtime_resolves_auto_device_via_device_resolver(tmp_path):
    """runtime accepts 'auto' and passes it through to the injected resolver."""
    from aiforensics.baselines.npr import runtime as runtime_mod

    args = _runtime_args(tmp_path, ["s1"])
    args.device = "auto"
    seen_devices: list[str] = []

    def recording_resolver(device):
        seen_devices.append(device)
        return "cuda"

    code = runtime_mod.run_runtime(
        args,
        seed_setter=lambda seed: None,
        device_resolver=recording_resolver,
        model_loader=lambda repo_dir, checkpoint, device: object(),
        batch_runner=lambda model, batch_array, device: [0.5] * len(batch_array),
    )
    assert code == 0
    assert seen_devices == ["auto"]


def test_runtime_auto_device_rejected_error_message(tmp_path, monkeypatch):
    """Default _resolve_runtime_device rejects 'auto' when torch reports no CUDA."""
    from aiforensics.baselines.npr import runtime as runtime_mod

    _install_fake_torch(monkeypatch)

    with pytest.raises(RuntimeError, match="auto resolved to no usable CUDA"):
        runtime_mod._resolve_runtime_device("auto")


def test_existing_checkout_fetch_failure_fails_when_not_allowed(tmp_path, monkeypatch):
    """Existing checkout at wrong commit + fetch failure must NOT defer when
    allow_deferred=False; it records failed via NPRConfigError."""
    repo_dir = tmp_path / "NPR-DeepfakeDetection"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    import aiforensics.baselines.npr.checkout as checkout_mod

    def fake_git(repo, *args):
        if args[:2] == ("remote", "get-url"):
            return OFFICIAL_NPR_REPO_URL
        if args[:1] == ("status",):
            return ""
        if args[:1] == ("rev-parse",):
            return "f" * 40  # HEAD at a different commit
        raise AssertionError("unexpected git call")

    monkeypatch.setattr(checkout_mod, "_run_git", fake_git)

    def failing_fetch(command, capture_output, text, check):
        assert command[:2] == ["git", "-C"]
        assert command[3:5] == ["fetch", "origin"]
        return type("R", (), {"returncode": 128, "stdout": "", "stderr": "network down"})()

    monkeypatch.setattr(subprocess, "run", failing_fetch)

    with pytest.raises(NPRConfigError, match="Could not fetch configured commit"):
        ensure_npr_checkout(
            repo_dir=repo_dir,
            repo_url=OFFICIAL_NPR_REPO_URL,
            repo_commit=VERIFIED_COMMIT,
            allow_deferred=False,
        )


def test_existing_checkout_fetch_failure_defers_when_allowed(tmp_path, monkeypatch):
    repo_dir = tmp_path / "NPR-DeepfakeDetection"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    import aiforensics.baselines.npr.checkout as checkout_mod

    def fake_git(repo, *args):
        if args[:2] == ("remote", "get-url"):
            return OFFICIAL_NPR_REPO_URL
        if args[:1] == ("status",):
            return ""
        if args[:1] == ("rev-parse",):
            return "f" * 40
        raise AssertionError("unexpected git call")

    monkeypatch.setattr(checkout_mod, "_run_git", fake_git)

    def failing_fetch(command, capture_output, text, check):
        return type("R", (), {"returncode": 128, "stdout": "", "stderr": "network down"})()

    monkeypatch.setattr(subprocess, "run", failing_fetch)

    with pytest.raises(NPRDeferredError, match="Could not fetch configured commit"):
        ensure_npr_checkout(
            repo_dir=repo_dir,
            repo_url=OFFICIAL_NPR_REPO_URL,
            repo_commit=VERIFIED_COMMIT,
            allow_deferred=True,
        )


def test_adapter_checkout_failure_fails_when_not_allowed(tmp_path, eval_setup, monkeypatch):
    """The adapter maps NPRDeferredError from checkout to failed when
    allow_deferred=False (existing-checkout fetch failure included)."""
    config, records = eval_setup
    config.baselines.npr.allow_deferred = False

    fake_checkout = _FakeCheckout(exc=NPRDeferredError("Could not fetch configured commit: x"))
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir, fake_checkout=fake_checkout)
    assert result.status == "failed"
    assert "checkout unavailable" in (result.reason or "")


def test_checkpoint_invalid_checksum_format_fails(tmp_path):
    """A malformed configured checksum fails loudly instead of silently skipping."""
    checkpoint = tmp_path / "NPR.pth"
    checkpoint.write_bytes(b"deterministic-bytes")
    with pytest.raises(ValueError, match="not a 64-character hex"):
        validate_checkpoint(checkpoint, "xyz")


def test_adapter_invalid_checksum_format_fails(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    config.baselines.npr.checkpoint_sha256 = "deadbeef"  # typo: not 64 hex chars
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "failed"
    assert "not a 64-character hex" in (result.reason or "")


def test_checkpoint_vanishing_mid_run_fails_when_not_allowed(tmp_path, eval_setup, monkeypatch):
    """A checkpoint removed between the existence check and integrity validation
    must record failed when allow_deferred=False (never unconditional defer)."""
    import aiforensics.baselines.npr.adapter as adapter_mod

    config, records = eval_setup
    config.baselines.npr.allow_deferred = False

    def vanishing_validate(path, sha256):
        raise FileNotFoundError(f"NPR checkpoint not found: {path}")

    monkeypatch.setattr(adapter_mod.checkpoint_mod, "validate_checkpoint", vanishing_validate)

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "failed"
    assert "checkpoint not found" in (result.reason or "").lower()


def test_checkpoint_vanishing_mid_run_defers_when_allowed(tmp_path, eval_setup, monkeypatch):
    import aiforensics.baselines.npr.adapter as adapter_mod

    config, records = eval_setup

    def vanishing_validate(path, sha256):
        raise FileNotFoundError(f"NPR checkpoint not found: {path}")

    monkeypatch.setattr(adapter_mod.checkpoint_mod, "validate_checkpoint", vanishing_validate)

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "deferred"


def test_checkout_mkdir_failure_is_config_error(tmp_path, monkeypatch):
    """A blocked external_root path (file in the way) surfaces as NPRConfigError,
    not a raw OSError leaking out of the checkout boundary."""
    external_root = tmp_path / "external"
    external_root.write_text("occupies the external_root path")  # exists as a file
    repo_dir = external_root / "NPR-DeepfakeDetection"

    import aiforensics.baselines.npr.checkout as checkout_mod

    monkeypatch.setattr(checkout_mod, "_git_version", lambda: True)

    with pytest.raises(NPRConfigError, match="Could not create NPR checkout parent"):
        ensure_npr_checkout(
            repo_dir=repo_dir,
            repo_url=OFFICIAL_NPR_REPO_URL,
            repo_commit=VERIFIED_COMMIT,
            allow_deferred=False,
        )


def test_runtime_score_wrong_order_fails_adapter_run(tmp_path, eval_setup, monkeypatch):
    """Reordered npr_scores.jsonl must fail the run (input-order contract)."""
    config, records = eval_setup
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(
        monkeypatch,
        config,
        output_dir,
        scores=[("sample-002", 0.1), ("sample-001", 0.9)],
    )
    assert result.status == "failed"
    assert "order misalignment" in (result.reason or "")


def test_runtime_scores_reordered_set_fails_validation():
    with pytest.raises(NPRRuntimeError, match="order misalignment"):
        bridge.validate_runtime_scores(
            [
                {"sample_id": "b", "score_fake": 0.2},
                {"sample_id": "a", "score_fake": 0.1},
            ],
            ["a", "b"],
        )


def test_auto_device_passed_through_to_runtime_command(tmp_path, eval_setup, monkeypatch):
    """runtime.device=auto must reach the runtime untouched so the runtime
    process resolves it against real CUDA availability."""
    config, records = eval_setup
    assert config.runtime.device == "auto"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    calls: list[list[str]] = []
    result = _run_with_mocks(monkeypatch, config, output_dir, subprocess_calls=calls)
    assert result.status == "completed"
    device_index = calls[0].index("--device")
    assert calls[0][device_index + 1] == "auto"


def test_completed_log_contains_audit_fields(tmp_path, eval_setup, monkeypatch):
    config, records = eval_setup
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    result = _run_with_mocks(monkeypatch, config, output_dir)
    assert result.status == "completed"
    log_text = (output_dir / "logs.txt").read_text(encoding="utf-8")
    completed = [line for line in log_text.splitlines() if "completed:" in line]
    assert len(completed) == 1
    for field in (
        "repo_url=",
        "repo_dir=",
        "configured_commit=",
        "resolved_commit=",
        "checkout_action=",
        "checkpoint=",
        "preprocessing_profile=npr_genimage_v1",
        "device=auto",
    ):
        assert field in completed[0], f"missing audit field {field!r} in completed log"
    assert f"configured_commit={VERIFIED_COMMIT}" in completed[0]
