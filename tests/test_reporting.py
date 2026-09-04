"""Task 11 tests: Phase A/B reporting.

All tests are CPU-only, network-free, and model-free. Synthetic run artifacts
are built under ``tmp_path``; the repository's own ``outputs/`` history is never
touched.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

import pytest

from aiforensics.evaluation.metrics import METRIC_NAMES
from aiforensics.reporting import markdown as md
from aiforensics.reporting.markdown import ReportingError
from aiforensics.runs.artifacts import write_status

# ---------------------------------------------------------------------------
# fixtures / artifact factories
# ---------------------------------------------------------------------------

_TS_EARLY = "2026-01-01T00:00:00+00:00"
_TS_LATE = "2026-01-02T00:00:00+00:00"


def _make_config(tmp_path: Path, *, phase: str = "phase_ab", **report_overrides):
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

    output_root = tmp_path / "outputs"

    report_values = {
        "filename": "phase_ab_report.md",
        "include_failure_notes": True,
        "include_explanations_sample": False,
        "explanation_sample_size": 0,
    }
    report_values.update(report_overrides)

    return AppConfig(
        project=ProjectConfig(
            name="ai-image-forensics",
            phase=phase,
            description="Test project",
        ),
        paths=PathsConfig(
            data_root=tmp_path / "data",
            manifest_root=tmp_path / "manifests",
            cache_root=tmp_path / "cache",
            output_root=output_root,
            external_root=tmp_path / "external",
        ),
        runtime=RuntimeConfig(
            python="3.10",
            seed=70,
            device="auto",
            batch_size=2,
            num_workers=0,
            fail_fast=False,
        ),
        datasets=DatasetsConfig(
            tiny_genimage=TinyGenImageConfig(
                enabled=True,
                source="huggingface",
                use_original_split=True,
                train_manifest=tmp_path / "manifests" / "tiny_train.csv",
                dev_manifest=tmp_path / "manifests" / "tiny_dev.csv",
            ),
            genimage_unseen=GenImageUnseenConfig(
                enabled=False,
                preferred_generator="sd15",
                fallback_generators=[],
                max_images=10,
                balance_labels=True,
                split="dev",
                manifest=tmp_path / "manifests" / "genimage_unseen.csv",
            ),
            synthbuster=SynthbusterConfig(
                enabled=False,
                max_images=10,
                balance_labels=True,
                split="dev",
                manifest=tmp_path / "manifests" / "synthbuster.csv",
            ),
        ),
        baselines=BaselinesConfig(
            clip_probe=ClipProbeConfig(
                enabled=True,
                model_family="synthetic",
                model_name="tiny",
                pretrained="",
                classifier="logreg",
                seeds=[70],
                cache_embeddings=False,
            ),
            qwen_vl=QwenVLConfig(
                enabled=True,
                model_id="test/model",
                prompt_id="p1",
                temperature=0.0,
                max_new_tokens=16,
                cache_outputs=False,
                allow_deferred=True,
            ),
            assisted_qwen=AssistedQwenConfig(
                enabled=True,
                base_model_id="test/model",
                prompt_id="p1",
                assistant_source="clip_probe",
                include_classifier_pred=True,
                include_fake_probability=True,
                temperature=0.0,
                max_new_tokens=16,
                cache_outputs=False,
                allow_deferred=True,
            ),
            npr=NPRConfig(
                enabled=True,
                repo_url="https://github.com/Chenyu-Research-Team/NPR-DeepfakeDetection",
                repo_commit=None,
                checkpoint_path=tmp_path / "NPR.pth",
                checkpoint_sha256=None,
                batch_size=2,
                allow_deferred=True,
            ),
        ),
        evaluation=EvaluationConfig(
            labels=LabelsConfig(negative="real", positive="fake"),
            metrics=["accuracy", "balanced_accuracy"],
            group_by=["source"],
        ),
        report=ReportConfig(**report_values),
    )


def _write_status(
    run_dir: Path,
    baseline: str,
    status: str,
    *,
    reason: str | None = None,
    ended_at: str = _TS_EARLY,
    config=None,
) -> Path:
    from aiforensics.runs.artifacts import RunStatus

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "status.json"
    write_status(
        path,
        RunStatus(
            baseline=baseline,
            status=status,  # type: ignore[arg-type]
            reason=reason,
            command=["aiforensics", "run"],
            started_at=_TS_EARLY,
            ended_at=ended_at,
        ),
    )
    if config is not None:
        _write_scope(run_dir, config)
    return path


def _write_scope(run_dir: Path, config) -> None:
    """Stamp a run directory with the scope the CLI would write for ``config``.

    Discovery only selects runs belonging to the current experiment, so an
    artifact factory must declare which config produced it. Passing no config
    models a pre-scope or foreign run directory.
    """
    from aiforensics.runs.scope import SCOPE_FILENAME, compute_run_scope, write_run_scope

    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_scope(run_dir / SCOPE_FILENAME, compute_run_scope(config))


def _overall_metrics(**overrides) -> dict[str, float | None]:
    values: dict[str, float | None] = {name: 0.9 for name in METRIC_NAMES}
    values.update(overrides)
    return values


def _write_metrics(
    run_dir: Path,
    *,
    total_records: int = 2,
    overall: dict[str, float | None] | None = None,
    by_source: list[tuple[str, int, dict[str, float | None]]] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "total_records": total_records,
        "overall": overall if overall is not None else _overall_metrics(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    if by_source is None:
        by_source = [("tiny-genimage-dev", total_records, _overall_metrics())]

    header = ["source", "n", *METRIC_NAMES]
    lines = [",".join(header)]
    for source, n, values in by_source:
        cells = [source, str(n)]
        for name in METRIC_NAMES:
            value = values.get(name)
            cells.append("" if value is None else repr(float(value)))
        lines.append(",".join(cells))
    (run_dir / "metrics_by_source.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _complete_run(
    output_root: Path,
    run_id: str,
    baseline: str,
    *,
    ended_at: str = _TS_EARLY,
    reason: str | None = None,
    overall: dict[str, float | None] | None = None,
    by_source: list[tuple[str, int, dict[str, float | None]]] | None = None,
    total_records: int = 2,
    config=None,
) -> Path:
    run_dir = output_root / run_id
    _write_status(run_dir, baseline, "completed", reason=reason, ended_at=ended_at, config=config)
    _write_metrics(
        run_dir,
        total_records=total_records,
        overall=overall,
        by_source=by_source,
    )
    (run_dir / "predictions.jsonl").touch()
    return run_dir


# ---------------------------------------------------------------------------
# Step 1: models and formatting helpers
# ---------------------------------------------------------------------------


class TestMetricFormatting:
    def test_fmt_metric_renders_four_decimals(self):
        assert md._fmt_metric(0.91234) == "0.9123"

    def test_fmt_metric_none_is_na(self):
        assert md._fmt_metric(None) == "N/A"


class TestMeanStdFormatting:
    def test_empty_values_render_na(self):
        assert md._fmt_mean_std([]) == "N/A"

    def test_single_value_renders_scalar_without_fake_std(self):
        assert md._fmt_mean_std([0.9]) == "0.9000"

    def test_multiple_values_render_mean_plus_sample_std(self):
        values = [0.90, 0.92, 0.94]
        expected = f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}"
        assert md._fmt_mean_std(values) == expected


class TestCellEscaping:
    def test_pipe_is_escaped(self):
        assert md._escape_cell("a|b") == r"a\|b"

    def test_newlines_become_spaces(self):
        assert md._escape_cell("line1\nline2") == "line1 line2"
        assert md._escape_cell("line1\r\nline2") == "line1 line2"

    def test_plain_text_unchanged(self):
        assert md._escape_cell("all good") == "all good"

    def test_pipe_and_newline_combined(self):
        assert md._escape_cell("bad|msg\nnext") == r"bad\|msg next"


class TestReportPathResolution:
    @pytest.mark.parametrize("filename", ["report.md", "REPORT.MD", "phase_ab_smoke_report.md"])
    def test_accepts_markdown_basenames(self, tmp_path, filename):
        config = _make_config(tmp_path)
        config.report.filename = filename
        path = md._resolve_report_path(config)
        assert path == config.paths.output_root / filename

    @pytest.mark.parametrize(
        "filename",
        [
            "../report.md",
            "/tmp/report.md",
            "subdir/report.md",
            "report.txt",
            "report",
            "",
            "   ",
        ],
    )
    def test_rejects_unsafe_filenames(self, tmp_path, filename):
        config = _make_config(tmp_path)
        config.report.filename = filename
        with pytest.raises(ReportingError):
            md._resolve_report_path(config)


# ---------------------------------------------------------------------------
# Step 2: artifact parsing
# ---------------------------------------------------------------------------


class TestStatusParsing:
    def test_valid_status_parses_with_ended_timestamp(self, tmp_path):
        run_dir = tmp_path / "run"
        path = _write_status(run_dir, "qwen_vl", "completed", ended_at=_TS_LATE)
        status, ended_at = md._parse_status(path)
        assert status.baseline == "qwen_vl"
        assert status.status == "completed"
        assert ended_at == datetime.fromisoformat(_TS_LATE)

    def test_malformed_json_raises_with_path(self, tmp_path):
        path = tmp_path / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ReportingError, match="status.json"):
            md._parse_status(path)

    def test_invalid_status_payload_raises(self, tmp_path):
        path = tmp_path / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": "qwen_vl",
            "status": "bogus",
            "reason": None,
            "command": ["aiforensics"],
            "started_at": _TS_EARLY,
            "ended_at": _TS_EARLY,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ReportingError, match="status.json"):
            md._parse_status(path)

    def test_non_utc_timestamp_raises(self, tmp_path):
        path = tmp_path / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": "qwen_vl",
            "status": "completed",
            "reason": None,
            "command": ["aiforensics"],
            "started_at": "2026-01-01T00:00:00+07:00",
            "ended_at": "2026-01-01T00:00:00+07:00",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ReportingError, match="status.json"):
            md._parse_status(path)

    def test_non_object_payload_raises(self, tmp_path):
        path = tmp_path / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ReportingError, match="status.json"):
            md._parse_status(path)


class TestMetricsJsonParsing:
    def test_valid_metrics_parse(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_metrics(run_dir, total_records=7, overall=_overall_metrics(auroc=None))
        total, overall = md._load_metrics_json(run_dir / "metrics.json")
        assert total == 7
        assert overall["auroc"] is None
        assert overall["accuracy"] == 0.9

    def test_non_object_root_raises(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ReportingError, match="metrics.json"):
            md._load_metrics_json(path)

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(ReportingError, match="metrics.json"):
            md._load_metrics_json(path)

    @pytest.mark.parametrize("bad_total", [-1, 1.5, "3", None, True])
    def test_invalid_total_records_raises(self, tmp_path, bad_total):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps({"total_records": bad_total, "overall": _overall_metrics()}),
            encoding="utf-8",
        )
        with pytest.raises(ReportingError, match="total_records"):
            md._load_metrics_json(run_dir / "metrics.json")

    def test_missing_metric_key_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        partial = _overall_metrics()
        partial.pop("auroc")
        (run_dir / "metrics.json").write_text(
            json.dumps({"total_records": 2, "overall": partial}),
            encoding="utf-8",
        )
        with pytest.raises(ReportingError, match="auroc"):
            md._load_metrics_json(run_dir / "metrics.json")

    @pytest.mark.parametrize("bad_value", [-0.1, 1.5, float("nan"), float("inf"), "0.9", True])
    def test_invalid_metric_values_raise(self, tmp_path, bad_value):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "total_records": 2,
                    "overall": _overall_metrics(accuracy=bad_value),
                }
            ).replace("NaN", "1e999"),  # json.loads accepts bare NaN/Infinity too
            encoding="utf-8",
        )
        with pytest.raises(ReportingError, match="accuracy"):
            md._load_metrics_json(run_dir / "metrics.json")

    def test_null_metric_value_is_allowed(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_metrics(run_dir, overall=_overall_metrics(auroc=None, f1=None))
        _, overall = md._load_metrics_json(run_dir / "metrics.json")
        assert overall["auroc"] is None
        assert overall["f1"] is None


class TestMetricsBySourceCsvParsing:
    def test_valid_rows_parse_sorted_by_source(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_metrics(
            run_dir,
            by_source=[
                ("zeta-source", 3, _overall_metrics()),
                ("alpha-source", 1, _overall_metrics(accuracy=0.5)),
            ],
        )
        rows = md._load_metrics_by_source_csv(run_dir / "metrics_by_source.csv")
        assert [r.source for r in rows] == ["alpha-source", "zeta-source"]
        assert rows[0].n == 1
        assert rows[0].accuracy == 0.5
        assert rows[1].n == 3

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "metrics_by_source.csv"
        path.write_text("source,n\nalpha,2\n", encoding="utf-8")
        with pytest.raises(ReportingError, match="metrics_by_source.csv"):
            md._load_metrics_by_source_csv(path)

    def test_duplicate_source_rows_raise(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_metrics(
            run_dir,
            by_source=[
                ("dup", 1, _overall_metrics()),
                ("dup", 2, _overall_metrics()),
            ],
        )
        with pytest.raises(ReportingError, match="[Dd]uplicate"):
            md._load_metrics_by_source_csv(run_dir / "metrics_by_source.csv")

    def test_empty_source_raises(self, tmp_path):
        path = tmp_path / "metrics_by_source.csv"
        header = ["source", "n", *METRIC_NAMES]
        line = ",".join(header)
        row = ",".join(["", "2", *["0.9"] * len(METRIC_NAMES)])
        path.write_text(f"{line}\n{row}\n", encoding="utf-8")
        with pytest.raises(ReportingError, match="metrics_by_source.csv"):
            md._load_metrics_by_source_csv(path)

    def test_negative_n_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_metrics(
            run_dir,
            by_source=[("src", -1, _overall_metrics())],
        )
        with pytest.raises(ReportingError, match="metrics_by_source.csv"):
            md._load_metrics_by_source_csv(run_dir / "metrics_by_source.csv")

    def test_blank_metric_value_parses_as_none(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_metrics(
            run_dir,
            by_source=[("src", 2, _overall_metrics(auroc=None))],
        )
        rows = md._load_metrics_by_source_csv(run_dir / "metrics_by_source.csv")
        assert rows[0].auroc is None

    def test_out_of_range_metric_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_metrics(
            run_dir,
            by_source=[("src", 2, _overall_metrics(accuracy=1.5))],
        )
        with pytest.raises(ReportingError, match="accuracy"):
            md._load_metrics_by_source_csv(run_dir / "metrics_by_source.csv")

    def test_malformed_csv_raises(self, tmp_path):
        path = tmp_path / "metrics_by_source.csv"
        header = ["source", "n", *METRIC_NAMES]
        path.write_text(
            ",".join(header) + "\nbroken,row,with,bad,data,xx,yy\n",
            encoding="utf-8",
        )
        with pytest.raises(ReportingError, match="metrics_by_source.csv"):
            md._load_metrics_by_source_csv(path)


# ---------------------------------------------------------------------------
# Step 3: run discovery and latest-slot selection
# ---------------------------------------------------------------------------


def _deferred_run(
    output_root: Path,
    run_id: str,
    baseline: str,
    *,
    ended_at: str = _TS_EARLY,
    reason: str = "deps unavailable",
    seed: int | None = None,
    config=None,
) -> Path:
    run_dir = output_root / run_id
    _write_status(run_dir, baseline, "deferred", reason=reason, ended_at=ended_at, config=config)
    return run_dir


def _failed_run(
    output_root: Path,
    run_id: str,
    baseline: str,
    *,
    ended_at: str = _TS_EARLY,
    reason: str = "boom",
    config=None,
) -> Path:
    run_dir = output_root / run_id
    _write_status(run_dir, baseline, "failed", reason=reason, ended_at=ended_at, config=config)
    return run_dir


class TestExpectedSlots:
    def test_enabled_clip_creates_one_slot_per_seed_ascending(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.seeds = [72, 70, 71]
        slots = md._expected_slots(config)
        clip_slots = [(b, s) for b, s in slots if b == "clip_probe"]
        assert clip_slots == [("clip_probe", 70), ("clip_probe", 71), ("clip_probe", 72)]

    def test_disabled_clip_creates_single_seedless_slot(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.enabled = False
        slots = md._expected_slots(config)
        assert slots == (
            ("clip_probe", None),
            ("qwen_vl", None),
            ("assisted_qwen", None),
            ("npr", None),
        )

    def test_slot_order_is_fixed(self, tmp_path):
        config = _make_config(tmp_path)
        slots = md._expected_slots(config)
        assert [b for b, _ in slots] == ["clip_probe", "qwen_vl", "assisted_qwen", "npr"]


class TestDiscovery:
    def test_missing_slot_becomes_reporting_missing(self, tmp_path):
        config = _make_config(tmp_path)
        config.paths.output_root.mkdir()
        runs = md.discover_run_summaries(config)
        by_slot = {(r.baseline, r.seed): r for r in runs}
        assert len(runs) == 4
        assert by_slot[("qwen_vl", None)].status == "missing"
        assert by_slot[("qwen_vl", None)].run_id is None

    def test_latest_qwen_run_selected_over_older(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        old = _complete_run(root, "001_qwen_vl", "qwen_vl", ended_at=_TS_EARLY, config=config)
        _write_metrics(old, overall=_overall_metrics(accuracy=0.5))
        new_dir = root / "002_qwen_vl"
        _write_status(new_dir, "qwen_vl", "completed", ended_at=_TS_LATE, config=config)
        _write_metrics(new_dir, overall=_overall_metrics(accuracy=0.99))
        (new_dir / "predictions.jsonl").touch()

        runs = md.discover_run_summaries(config)
        qwen = next(r for r in runs if r.baseline == "qwen_vl")
        assert qwen.run_id == "002_qwen_vl"
        assert qwen.overall is not None
        assert qwen.overall["accuracy"] == 0.99

    def test_latest_failed_run_selected_over_older_completed(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_qwen_vl", "qwen_vl", ended_at=_TS_EARLY, config=config)
        _failed_run(
            root, "002_qwen_vl", "qwen_vl", ended_at=_TS_LATE, reason="exploded", config=config
        )

        runs = md.discover_run_summaries(config)
        qwen = next(r for r in runs if r.baseline == "qwen_vl")
        assert qwen.status == "failed"
        assert qwen.reason == "exploded"
        assert qwen.overall is None
        assert qwen.by_source == ()

    def test_clip_seeds_selected_independently(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.seeds = [70, 71]
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_clip_probe_seed70", "clip_probe", ended_at=_TS_LATE, config=config)
        _failed_run(root, "002_clip_probe_seed71", "clip_probe", ended_at=_TS_LATE, config=config)

        runs = md.discover_run_summaries(config)
        by_seed = {r.seed: r for r in runs if r.baseline == "clip_probe"}
        assert by_seed[70].status == "completed"
        assert by_seed[70].run_id == "001_clip_probe_seed70"
        assert by_seed[71].status == "failed"

    def test_ended_at_drives_selection_and_name_is_tiebreaker(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        # Same ended_at: lexicographically greatest directory name wins.
        dir_a = root / "run_a"
        dir_b = root / "run_b"
        for run_dir in (dir_a, dir_b):
            _write_status(run_dir, "qwen_vl", "completed", ended_at=_TS_LATE, config=config)
            _write_metrics(run_dir)
            (run_dir / "predictions.jsonl").touch()
        _write_metrics(dir_a, total_records=1)
        _write_metrics(dir_b, total_records=2)

        runs = md.discover_run_summaries(config)
        qwen = next(r for r in runs if r.baseline == "qwen_vl")
        assert qwen.run_id == "run_b"
        assert qwen.total_records == 2

    def test_unknown_baseline_artifacts_do_not_enter_report(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_something", "mystery_baseline", config=config)
        _complete_run(root, "002_clip_probe_seed70", "clip_probe", config=config)

        runs = md.discover_run_summaries(config)
        assert {r.baseline for r in runs} == {"clip_probe", "qwen_vl", "assisted_qwen", "npr"}
        clip = next(r for r in runs if r.baseline == "clip_probe")
        assert clip.status == "completed"

    def test_clip_run_without_suffix_ignored_when_clip_enabled(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_clip_probe", "clip_probe", config=config)  # no seed suffix
        runs = md.discover_run_summaries(config)
        clip = next(r for r in runs if r.baseline == "clip_probe")
        assert clip.status == "missing"

    def test_disabled_clip_uses_suffixless_slot(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.enabled = False
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_clip_probe", "clip_probe", config=config)
        _complete_run(root, "002_clip_probe_seed70", "clip_probe", config=config)

        runs = md.discover_run_summaries(config)
        clip = next(r for r in runs if r.baseline == "clip_probe")
        assert clip.seed is None
        assert clip.run_id == "001_clip_probe"
        assert clip.status == "completed"

    def test_malformed_status_json_anywhere_raises(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_clip_probe_seed70", "clip_probe", config=config)
        bad = root / "002_garbage"
        bad.mkdir()
        (bad / "status.json").write_text("not json", encoding="utf-8")

        with pytest.raises(ReportingError, match="status.json"):
            md.discover_run_summaries(config)

    def test_completed_run_missing_metrics_raises_with_evaluate_guidance(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        run_dir = root / "001_qwen_vl"
        _write_status(run_dir, "qwen_vl", "completed", ended_at=_TS_LATE, config=config)
        (run_dir / "predictions.jsonl").touch()

        with pytest.raises(ReportingError, match="aiforensics evaluate"):
            md.discover_run_summaries(config)

    def test_completed_run_missing_predictions_raises(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        run_dir = root / "001_qwen_vl"
        _write_status(run_dir, "qwen_vl", "completed", ended_at=_TS_LATE, config=config)
        _write_metrics(run_dir)

        with pytest.raises(ReportingError, match="predictions.jsonl"):
            md.discover_run_summaries(config)

    def test_failed_run_with_stale_metrics_excludes_them(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        run_dir = _failed_run(root, "001_qwen_vl", "qwen_vl", ended_at=_TS_LATE, config=config)
        # Stale metrics from an earlier completed attempt must not be included.
        _write_metrics(run_dir, total_records=99, overall=_overall_metrics(accuracy=1.0))

        runs = md.discover_run_summaries(config)
        qwen = next(r for r in runs if r.baseline == "qwen_vl")
        assert qwen.status == "failed"
        assert qwen.overall is None
        assert qwen.by_source == ()

    def test_deferred_run_records_reason(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _deferred_run(
            root, "001_npr", "npr", ended_at=_TS_LATE, reason="CUDA unavailable", config=config
        )

        runs = md.discover_run_summaries(config)
        npr = next(r for r in runs if r.baseline == "npr")
        assert npr.status == "deferred"
        assert npr.reason == "CUDA unavailable"

    def test_run_without_scope_is_not_selected(self, tmp_path):
        """A run directory with no run_scope.json cannot be proven to belong here."""
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_qwen_vl", "qwen_vl", ended_at=_TS_LATE)  # no config -> no scope

        runs = md.discover_run_summaries(config)
        qwen = next(r for r in runs if r.baseline == "qwen_vl")
        assert qwen.status == "missing"

    def test_run_from_different_dataset_slice_is_not_selected(self, tmp_path):
        """Changing the evaluation slice makes older runs foreign, not current."""
        other = _make_config(tmp_path)
        other.datasets.genimage_unseen.enabled = True
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_qwen_vl", "qwen_vl", ended_at=_TS_LATE, config=other)

        runs = md.discover_run_summaries(config)
        qwen = next(r for r in runs if r.baseline == "qwen_vl")
        assert qwen.status == "missing"

    def test_same_config_scope_is_selected(self, tmp_path):
        """The positive control for scope filtering: same config still matches."""
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_qwen_vl", "qwen_vl", ended_at=_TS_LATE, config=config)

        runs = md.discover_run_summaries(config)
        qwen = next(r for r in runs if r.baseline == "qwen_vl")
        assert qwen.status == "completed"
        assert qwen.run_id == "001_qwen_vl"

    def test_discovery_is_network_free(self, tmp_path, monkeypatch):
        """Reporting discovery must not invoke git or network helpers."""
        import subprocess as subprocess_mod

        def no_git(command, *args, **kwargs):
            raise AssertionError(f"network/git must not run during discovery: {command}")

        monkeypatch.setattr(subprocess_mod, "run", no_git)

        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_clip_probe_seed70", "clip_probe", config=config)

        runs = md.discover_run_summaries(config)
        assert len(runs) == 4  # clip completed + 3 missing slots


# ---------------------------------------------------------------------------
# Steps 4-7: report rendering (pure), aggregation, explanations, recommendation
# ---------------------------------------------------------------------------


def _summary(
    baseline: str,
    status: str,
    *,
    seed: int | None = None,
    run_id: str | None = "run",
    reason: str | None = None,
    total_records: int | None = 2,
    overall: dict[str, float | None] | None = None,
    by_source: tuple = (),
) -> md.RunSummary:
    return md.RunSummary(
        baseline=baseline,
        seed=seed,
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        reason=reason,
        run_dir=Path("outputs") / (run_id or "none"),
        started_at=_TS_EARLY,
        ended_at=_TS_EARLY,
        total_records=total_records,
        overall=overall,
        by_source=by_source,
        prediction_path=Path("outputs") / (run_id or "none") / "predictions.jsonl",
    )


def _source_row(
    source: str,
    n: int,
    *,
    accuracy: float | None = 0.9,
    balanced_accuracy: float | None = 0.88,
    precision: float | None = 0.91,
    recall: float | None = 0.87,
    f1: float | None = 0.89,
    auroc: float | None = 0.95,
) -> md.SourceMetricRow:
    return md.SourceMetricRow(
        source=source,
        n=n,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=auroc,
    )


_SECTION_ORDER = [
    "# Phase A/B Baseline Report",
    "## Project",
    "## Dataset Summary",
    "## Baseline Status",
    "## Overall Metrics",
    "## Per-Source Metrics",
    "## Next-Step Recommendation",
]


class TestSectionOrder:
    def test_unconditional_sections_in_order(self, tmp_path):
        config = _make_config(tmp_path)
        runs = md.discover_run_summaries(config) if False else []
        text = md.build_phase_ab_report(config, runs)
        positions = [text.index(section) for section in _SECTION_ORDER]
        assert positions == sorted(positions)

    def test_conditional_sections_render_between_per_source_and_recommendation(self, tmp_path):
        config = _make_config(
            tmp_path,
            include_failure_notes=True,
            include_explanations_sample=True,
            explanation_sample_size=5,
        )
        runs = []
        text = md.build_phase_ab_report(config, runs)
        idx = {
            name: text.index(name)
            for name in [
                "## Per-Source Metrics",
                "## Failure and Deferred Notes",
                "## Explanation Samples",
                "## Next-Step Recommendation",
            ]
        }
        values = list(idx.values())
        assert values == sorted(values)

    def test_failure_notes_omitted_when_disabled(self, tmp_path):
        config = _make_config(tmp_path, include_failure_notes=False)
        text = md.build_phase_ab_report(config, [])
        assert "## Failure and Deferred Notes" not in text

    def test_explanations_omitted_when_disabled(self, tmp_path):
        config = _make_config(
            tmp_path, include_explanations_sample=False, explanation_sample_size=10
        )
        text = md.build_phase_ab_report(config, [])
        assert "## Explanation Samples" not in text


class TestProjectSection:
    def test_project_fields_present_with_relative_output_root(self, tmp_path):
        config = _make_config(tmp_path)
        text = md.build_phase_ab_report(config, [])
        assert f"**Project:** {config.project.name}" in text
        assert f"**Phase:** {config.project.phase}" in text
        assert f"**Description:** {config.project.description}" in text
        assert f"**Output root:** {config.paths.output_root.name}" in text
        assert str(tmp_path) not in text


class TestDatasetSummary:
    def test_all_three_datasets_rendered(self, tmp_path):
        config = _make_config(tmp_path)
        text = md.build_phase_ab_report(config, [])
        for name in ("tiny_genimage", "genimage_unseen", "synthbuster"):
            assert name in text
        assert "enabled" in text
        assert "disabled" in text

    def test_observed_coverage_na_without_completed_runs(self, tmp_path):
        config = _make_config(tmp_path)
        text = md.build_phase_ab_report(config, [_summary("qwen_vl", "deferred")])
        assert "N/A" in text


class TestBaselineStatusTable:
    def test_fixed_baseline_and_slot_rows(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.seeds = [70, 71]
        runs = [
            _summary("clip_probe", "completed", seed=70, run_id="run70"),
            _summary("clip_probe", "failed", seed=71, run_id="run71", reason="boom"),
            _summary("qwen_vl", "deferred", run_id="q", reason="no model"),
            _summary("assisted_qwen", "missing", run_id=None),
            _summary("npr", "completed", run_id="n"),
        ]
        text = md.build_phase_ab_report(config, runs)
        lines = text.splitlines()
        statuses = [ln for ln in lines if ln.startswith("|")]
        joined = "\n".join(statuses)
        for token in ("run70", "run71", "boom", "no model"):
            assert token in joined
        assert "missing" in joined

    def test_cell_escaping_in_reason(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary(
                "qwen_vl",
                "failed",
                run_id="q1",
                reason="bad|msg\nsecond line",
            )
        ]
        text = md.build_phase_ab_report(config, runs)
        status_section = text.split("## Baseline Status")[1].split("## ")[0]
        assert "bad\\|msg second line" in status_section
        # The table row containing the reason stays on a single line.
        for line in status_section.splitlines():
            if "second line" in line:
                assert line.startswith("|")


class TestOverallMetrics:
    def test_single_run_metrics_four_decimals_and_na(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary(
                "qwen_vl",
                "completed",
                overall=_overall_metrics(accuracy=0.91234, auroc=None),
            )
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "0.9123" in text
        assert "N/A" in text

    def test_clip_multi_seed_mean_plus_sample_std(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.seeds = [70, 71, 72]
        values = [0.90, 0.92, 0.94]
        runs = [
            _summary("clip_probe", "completed", seed=s, overall=_overall_metrics(accuracy=v))
            for s, v in zip([70, 71, 72], values, strict=True)
        ]
        text = md.build_phase_ab_report(config, runs)
        expected = f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}"
        assert expected in text

    def test_clip_single_seed_renders_scalar_without_fake_std(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [_summary("clip_probe", "completed", seed=70, overall=_overall_metrics())]
        text = md.build_phase_ab_report(config, runs)
        assert "0.9000 +/-" not in text
        assert "0.9000" in text

    def test_clip_none_values_excluded_metric_by_metric(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary(
                "clip_probe",
                "completed",
                seed=s,
                overall=_overall_metrics(auroc=None),
            )
            for s in (70, 71)
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "auroc" in text.lower()

    def test_partial_clip_completion_shows_coverage(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.seeds = [70, 71, 72]
        runs = [
            _summary("clip_probe", "completed", seed=70),
            _summary("clip_probe", "failed", seed=71, run_id="f", reason="x"),
            _summary("clip_probe", "missing", seed=72, run_id=None),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "1/3 seeds completed" in text

    def test_clip_per_source_n_mismatch_across_seeds_raises(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.seeds = [70, 71]
        runs = [
            _summary(
                "clip_probe",
                "completed",
                seed=70,
                by_source=(_source_row("src-a", 10),),
            ),
            _summary(
                "clip_probe",
                "completed",
                seed=71,
                by_source=(_source_row("src-a", 12),),
            ),
        ]
        with pytest.raises(ReportingError, match="disagree on n"):
            md.build_phase_ab_report(config, runs)


class TestPerSourceTable:
    def test_rows_sorted_by_baseline_then_source(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary(
                "npr",
                "completed",
                by_source=(_source_row("zzz", 5), _source_row("aaa", 3)),
            ),
            _summary(
                "qwen_vl",
                "completed",
                by_source=(_source_row("mmm", 4),),
            ),
        ]
        text = md.build_phase_ab_report(config, runs)
        positions = [text.index(x) for x in ["| qwen_vl | mmm |", "| npr | aaa |", "| npr | zzz |"]]
        assert positions == sorted(positions)

    def test_failed_baseline_shows_no_source_rows(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [_summary("qwen_vl", "failed", reason="x")]
        text = md.build_phase_ab_report(config, runs)
        per_source = text.split("## Per-Source Metrics")[1].split("## ")[0]
        assert "| qwen_vl |" not in per_source


class TestFailureNotes:
    def test_notes_include_baseline_status_reason(self, tmp_path):
        config = _make_config(tmp_path, include_failure_notes=True)
        runs = [
            _summary("qwen_vl", "deferred", run_id="q", reason="deps missing"),
            _summary("npr", "missing", run_id=None),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "deps missing" in text
        assert "No run artifact found under the configured output_root." in text

    def test_completed_runs_not_in_failure_notes(self, tmp_path):
        config = _make_config(tmp_path, include_failure_notes=True)
        runs = [_summary("qwen_vl", "completed", run_id="q", reason=None)]
        text = md.build_phase_ab_report(config, runs)
        section = text.split("## Failure and Deferred Notes")[1].split("## ")[0]
        assert "qwen_vl" not in section


class TestExplanationSamples:
    def _make_prediction(self, sample_id: str, model: str, **overrides):
        row = {
            "sample_id": sample_id,
            "label_true": "fake",
            "label_pred": "fake",
            "score_fake": 0.9,
            "model_name": model,
            "source": "tiny-genimage-dev",
            "prompt_id": "p1",
            "raw_output": "RAW OUTPUT TEXT",
            "explanation": "explanation for " + sample_id,
            "parse_status": "parsed",
        }
        row.update(overrides)
        return row

    def _write_predictions(self, run_dir: Path, rows: list[dict]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_samples_sorted_and_capped(self, tmp_path):
        config = _make_config(
            tmp_path,
            include_explanations_sample=True,
            explanation_sample_size=3,
        )
        root = config.paths.output_root
        root.mkdir(parents=True)
        qwen_dir = root / "001_qwen_vl"
        _write_status(qwen_dir, "qwen_vl", "completed", ended_at=_TS_LATE, config=config)
        _write_metrics(qwen_dir)
        self._write_predictions(
            qwen_dir,
            [
                self._make_prediction("s-003", "qwen_vl"),
                self._make_prediction("s-001", "qwen_vl"),
                self._make_prediction("s-002", "qwen_vl"),
                self._make_prediction("s-004", "qwen_vl"),
            ],
        )
        assisted_dir = root / "002_assisted_qwen"
        _write_status(assisted_dir, "assisted_qwen", "completed", ended_at=_TS_LATE, config=config)
        _write_metrics(assisted_dir)
        self._write_predictions(
            assisted_dir,
            [self._make_prediction("s-001", "assisted_qwen")],
        )

        runs = md.discover_run_summaries(config)
        text = md.build_phase_ab_report(config, runs)
        section = text.split("## Explanation Samples")[1].split("## ")[0]
        # 4 qwen + 1 assisted candidates, capped at 3, sorted by sample_id then
        # baseline order: s-001 (assisted beats qwen at same id? No - qwen_vl
        # comes first in baseline order), s-002, s-003.
        assert "s-004" not in section

    def test_no_raw_output_leakage(self, tmp_path):
        config = _make_config(
            tmp_path,
            include_explanations_sample=True,
            explanation_sample_size=5,
        )
        root = config.paths.output_root
        root.mkdir(parents=True)
        qwen_dir = root / "001_qwen_vl"
        _write_status(qwen_dir, "qwen_vl", "completed", ended_at=_TS_LATE, config=config)
        _write_metrics(qwen_dir)
        self._write_predictions(qwen_dir, [self._make_prediction("s-001", "qwen_vl")])

        runs = md.discover_run_summaries(config)
        text = md.build_phase_ab_report(config, runs)
        assert "RAW OUTPUT TEXT" not in text

    def test_no_explanations_message(self, tmp_path):
        config = _make_config(
            tmp_path,
            include_explanations_sample=True,
            explanation_sample_size=5,
        )
        runs = [_summary("qwen_vl", "completed", run_id="q")]
        text = md.build_phase_ab_report(config, runs)
        assert "No explanation samples available." in text

    def test_zero_sample_size_renders_no_samples(self, tmp_path):
        config = _make_config(
            tmp_path,
            include_explanations_sample=True,
            explanation_sample_size=0,
        )
        runs = [_summary("qwen_vl", "completed", run_id="q")]
        text = md.build_phase_ab_report(config, runs)
        section = text.split("## Explanation Samples")[1].split("## ")[0]
        assert "s-001" not in section


class TestRecommendation:
    def test_smoke_phase_rejects_scientific_interpretation(self, tmp_path):
        config = _make_config(tmp_path, phase="phase_ab_smoke")
        text = md.build_phase_ab_report(config, [])
        assert "not scientific evidence" in text
        assert "go/no-go" in text

    def test_incomplete_phase_ab_does_not_select_winner(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary("clip_probe", "completed", seed=70),
            _summary("qwen_vl", "deferred", reason="no GPU"),
            _summary("assisted_qwen", "missing", run_id=None),
            _summary("npr", "missing", run_id=None),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "incomplete" in text.lower()
        assert "qwen_vl" in text  # unresolved slots identified

    def test_coverage_mismatch_blocks_comparison(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary("clip_probe", "completed", seed=70, total_records=10),
            _summary("qwen_vl", "completed", total_records=12),
            _summary("assisted_qwen", "completed", total_records=12),
            _summary("npr", "completed", total_records=12),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "coverage must be aligned" in text

    def test_comparable_runs_identify_best_balanced_accuracy(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary(
                "clip_probe",
                "completed",
                seed=70,
                overall=_overall_metrics(balanced_accuracy=0.80),
            ),
            _summary(
                "qwen_vl",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.90),
            ),
            _summary(
                "assisted_qwen",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.85),
            ),
            _summary(
                "npr",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.70),
            ),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "qwen_vl" in text
        assert "0.9000" in text

    def test_assisted_positive_delta_reported_numerically(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary("clip_probe", "completed", seed=70, overall=_overall_metrics()),
            _summary(
                "qwen_vl",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.80, f1=0.8, auroc=0.8),
            ),
            _summary(
                "assisted_qwen",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.85, f1=0.85, auroc=0.9),
            ),
            _summary("npr", "completed", total_records=2, overall=_overall_metrics()),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "+0.0500" in text

    def test_assisted_non_positive_delta_no_improvement_claim(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary("clip_probe", "completed", seed=70, overall=_overall_metrics()),
            _summary(
                "qwen_vl",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.85),
            ),
            _summary(
                "assisted_qwen",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.80),
            ),
            _summary("npr", "completed", total_records=2, overall=_overall_metrics()),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "did not improve" in text

    def test_classification_gain_never_claims_explanation_faithfulness(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary("clip_probe", "completed", seed=70, overall=_overall_metrics()),
            _summary(
                "qwen_vl",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.70),
            ),
            _summary(
                "assisted_qwen",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.95),
            ),
            _summary("npr", "completed", total_records=2, overall=_overall_metrics()),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "explanation faithfulness" in text.lower()


class TestDeterminism:
    def test_repeated_render_byte_for_byte_identical(self, tmp_path):
        config = _make_config(tmp_path)
        root = config.paths.output_root
        root.mkdir(parents=True)
        _complete_run(root, "001_clip_probe_seed70", "clip_probe", ended_at=_TS_LATE, config=config)
        _failed_run(root, "002_qwen_vl", "qwen_vl", ended_at=_TS_LATE, config=config)

        runs = md.discover_run_summaries(config)
        text1 = md.build_phase_ab_report(config, runs)
        text2 = md.build_phase_ab_report(config, md.discover_run_summaries(config))
        assert text1 == text2

    def test_report_ends_with_single_newline(self, tmp_path):
        config = _make_config(tmp_path)
        text = md.build_phase_ab_report(config, [])
        assert text.endswith("\n")
        assert not text.endswith("\n\n")


class TestWritePhaseAbReport:
    def test_writes_configured_filename_under_output_root(self, tmp_path):
        config = _make_config(tmp_path)
        config.report.filename = "phase_ab_report.md"
        path = md.write_phase_ab_report(config, "# report\n")
        assert path == config.paths.output_root / "phase_ab_report.md"
        assert path.read_text(encoding="utf-8") == "# report\n"

    def test_creates_output_root_when_absent(self, tmp_path):
        config = _make_config(tmp_path)
        assert not config.paths.output_root.exists()
        path = md.write_phase_ab_report(config, "# report\n")
        assert path.exists()


# ---------------------------------------------------------------------------
# Step 8: CLI contract
# ---------------------------------------------------------------------------


class TestReportCli:
    def _write_cli_config(self, tmp_path: Path) -> Path:
        import yaml

        smoke = Path(__file__).resolve().parents[1] / "configs" / "phase_ab_smoke.yaml"
        with open(smoke, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["paths"]["output_root"] = str(tmp_path / "outputs")
        data["paths"]["cache_root"] = str(tmp_path / "cache")
        data["paths"]["data_root"] = str(tmp_path / "data")
        data["paths"]["manifest_root"] = str(tmp_path / "manifests")
        for ds in ("tiny_genimage", "genimage_unseen", "synthbuster"):
            data["datasets"][ds]["manifest"] = str(tmp_path / "manifests" / f"{ds}.csv")
        data["datasets"]["tiny_genimage"]["train_manifest"] = str(
            tmp_path / "manifests" / "tiny_train.csv"
        )
        data["datasets"]["tiny_genimage"]["dev_manifest"] = str(
            tmp_path / "manifests" / "tiny_dev.csv"
        )
        cfg_path = tmp_path / "report_cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        (tmp_path / "pyproject.toml").touch()
        return cfg_path

    def _run_cli(self, tmp_path: Path):
        from aiforensics.cli.main import main

        cfg = self._write_cli_config(tmp_path)
        exit_code = main(["report", "--config", str(cfg)])
        return exit_code, tmp_path / "outputs"

    def test_cli_writes_configured_filename_and_exits_zero(self, tmp_path, capsys):
        exit_code, output_root = self._run_cli(tmp_path)
        assert exit_code == 0
        report_path = output_root / "phase_ab_smoke_report.md"
        assert report_path.is_file()
        text = report_path.read_text(encoding="utf-8")
        for section in _SECTION_ORDER:
            assert section in text
        out = capsys.readouterr().out
        assert "[report] project=" in out
        assert "runs=4" in out
        assert f"path={report_path}" in out

    def test_cli_exit_zero_with_deferred_and_missing_runs(self, tmp_path):
        config = _make_config(tmp_path, phase="phase_ab_smoke")
        config.report.filename = "phase_ab_smoke_report.md"
        root = config.paths.output_root
        root.mkdir(parents=True)
        _deferred_run(root, "001_qwen_vl", "qwen_vl", ended_at=_TS_LATE)
        # clip_probe disabled in smoke config; npr/assisted missing -> no artifacts

        # Route the CLI through the same config object via a YAML round-trip.
        import yaml

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
        (tmp_path / "pyproject.toml").touch()

        from aiforensics.cli.main import main

        exit_code = main(["report", "--config", str(cfg_path)])
        assert exit_code == 0
        assert (root / "phase_ab_smoke_report.md").is_file()

    def test_cli_exit_one_on_malformed_artifact(self, tmp_path, capsys):
        exit_code, output_root = self._run_cli(tmp_path)
        assert exit_code == 0

        # Create one selected completed run, then corrupt its metrics artifact.
        # Smoke config enables clip_probe with seed 70, so the run id must
        # carry the matching seed suffix to land in the expected slot, and the
        # run must carry the CLI config's scope to be selected at all.
        from aiforensics.config.load import load_config

        cli_config = load_config(tmp_path / "report_cfg.yaml")
        completed = _complete_run(
            output_root,
            "001_clip_probe_seed70",
            "clip_probe",
            ended_at=_TS_LATE,
            config=cli_config,
        )
        (completed / "metrics.json").write_text("{broken", encoding="utf-8")

        import yaml

        cfg_path = tmp_path / "report_cfg.yaml"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg2 = tmp_path / "report_cfg2.yaml"
        cfg2.write_text(yaml.safe_dump(data), encoding="utf-8")

        from aiforensics.cli.main import main

        exit_code = main(["report", "--config", str(cfg2)])
        assert exit_code == 1
        assert "Error generating report" in capsys.readouterr().out

    def test_cli_exit_one_on_unsafe_filename(self, tmp_path, capsys):
        import yaml

        cfg = self._write_cli_config(tmp_path)
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        data["report"]["filename"] = "../evil.md"
        cfg2 = tmp_path / "unsafe_cfg.yaml"
        cfg2.write_text(yaml.safe_dump(data), encoding="utf-8")

        from aiforensics.cli.main import main

        exit_code = main(["report", "--config", str(cfg2)])
        assert exit_code == 1
        assert "Error generating report" in capsys.readouterr().out

    def test_cli_exit_one_on_invalid_evaluation_manifest(self, tmp_path, capsys):
        """A corrupt manifest blocks scope computation as a reporting error."""
        import yaml

        cfg = self._write_cli_config(tmp_path)
        bad_manifest = tmp_path / "bad_dev.csv"
        bad_manifest.write_text("not,a,manifest\n1,2\n", encoding="utf-8")

        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        data["datasets"]["tiny_genimage"]["dev_manifest"] = str(bad_manifest)
        cfg2 = tmp_path / "bad_manifest_cfg.yaml"
        cfg2.write_text(yaml.safe_dump(data), encoding="utf-8")

        from aiforensics.cli.main import main

        exit_code = main(["report", "--config", str(cfg2)])
        out = capsys.readouterr().out
        assert exit_code == 1
        assert "Error generating report" in out
        assert "run scope" in out


# ---------------------------------------------------------------------------
# import/runtime safety (coverage point 45)
# ---------------------------------------------------------------------------


def test_reporting_import_does_not_pull_model_runtimes():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import aiforensics.reporting.markdown\n"
        "banned = {'torch', 'transformers', 'open_clip'}\n"
        "loaded = banned & set(sys.modules)\n"
        "assert not loaded, f'model runtimes imported at reporting import: {loaded}'\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# ---------------------------------------------------------------------------
# review fixes: single CLIP aggregate row, CSV error contract, enabled-only
# recommendation, write error boundary
# ---------------------------------------------------------------------------


class TestClipSingleAggregateRow:
    def test_overall_metrics_render_one_clip_row_for_multi_seed(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.seeds = [70, 71, 72]
        values = [0.90, 0.92, 0.94]
        runs = [
            _summary(
                "clip_probe",
                "completed",
                seed=seed,
                overall=_overall_metrics(accuracy=value),
            )
            for seed, value in zip([70, 71, 72], values, strict=True)
        ]
        text = md.build_phase_ab_report(config, runs)
        overall = text.split("## Overall Metrics")[1].split("## ")[0]
        clip_rows = [ln for ln in overall.splitlines() if ln.startswith("| clip_probe |")]
        assert len(clip_rows) == 1
        expected = f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}"
        assert expected in clip_rows[0]
        assert "3/3 seeds completed" in clip_rows[0]

    def test_per_source_renders_one_row_per_source_for_multi_seed(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.seeds = [70, 71]
        runs = [
            _summary(
                "clip_probe",
                "completed",
                seed=seed,
                by_source=(_source_row("src-a", 10),),
            )
            for seed in (70, 71)
        ]
        text = md.build_phase_ab_report(config, runs)
        per_source = text.split("## Per-Source Metrics")[1].split("## ")[0]
        rows = [ln for ln in per_source.splitlines() if ln.startswith("| clip_probe |")]
        assert len(rows) == 1
        assert "| clip_probe | src-a | 10 |" in rows[0]

    def test_zero_completed_clip_renders_single_na_row(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.clip_probe.seeds = [70, 71]
        runs = [
            _summary("clip_probe", "failed", seed=70, run_id="f1", reason="x"),
            _summary("clip_probe", "missing", seed=71, run_id=None),
        ]
        text = md.build_phase_ab_report(config, runs)
        overall = text.split("## Overall Metrics")[1].split("## ")[0]
        clip_rows = [ln for ln in overall.splitlines() if ln.startswith("| clip_probe |")]
        assert len(clip_rows) == 1
        assert "0/2 seeds completed" in clip_rows[0]


class TestCsvMetricErrorContract:
    def test_non_numeric_metric_value_raises_reporting_error(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        path = run_dir / "metrics_by_source.csv"
        header = ["source", "n", *METRIC_NAMES]
        cells = ["src", "2", "oops", *["0.9"] * (len(METRIC_NAMES) - 1)]
        path.write_text(",".join(header) + "\n" + ",".join(cells) + "\n", encoding="utf-8")
        with pytest.raises(ReportingError, match="accuracy"):
            md._load_metrics_by_source_csv(path)

    def test_non_numeric_n_raises_reporting_error(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        path = run_dir / "metrics_by_source.csv"
        header = ["source", "n", *METRIC_NAMES]
        cells = ["src", "many", *["0.9"] * len(METRIC_NAMES)]
        path.write_text(",".join(header) + "\n" + ",".join(cells) + "\n", encoding="utf-8")
        with pytest.raises(ReportingError, match="metrics_by_source.csv"):
            md._load_metrics_by_source_csv(path)


class TestEnabledOnlyRecommendation:
    def test_disabled_completed_baseline_never_wins(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.qwen_vl.enabled = False
        runs = [
            _summary(
                "clip_probe",
                "completed",
                seed=70,
                overall=_overall_metrics(balanced_accuracy=0.80),
            ),
            _summary(
                "qwen_vl",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.99),
            ),
            _summary(
                "assisted_qwen",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.85),
            ),
            _summary(
                "npr",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.70),
            ),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "Best observed baseline by balanced_accuracy: assisted_qwen (0.8500)." in text
        assert "Best observed baseline by balanced_accuracy: qwen_vl (0.9900)." not in text

    def test_delta_requires_both_qwen_and_assisted_enabled(self, tmp_path):
        config = _make_config(tmp_path)
        config.baselines.qwen_vl.enabled = False
        runs = [
            _summary("clip_probe", "completed", seed=70, overall=_overall_metrics()),
            _summary(
                "qwen_vl",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.99),
            ),
            _summary(
                "assisted_qwen",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.85),
            ),
            _summary(
                "npr",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.70),
            ),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "improved observed balanced_accuracy by" not in text
        assert "did not improve observed balanced" not in text

    def test_delta_rendered_when_both_enabled(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            _summary("clip_probe", "completed", seed=70, overall=_overall_metrics()),
            _summary(
                "qwen_vl",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.80),
            ),
            _summary(
                "assisted_qwen",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.85),
            ),
            _summary(
                "npr",
                "completed",
                total_records=2,
                overall=_overall_metrics(balanced_accuracy=0.70),
            ),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "+0.0500" in text


class TestWriteErrorBoundary:
    def test_write_wraps_oserror_as_reporting_error(self, tmp_path):
        config = _make_config(tmp_path)
        blocked = tmp_path / "blocked"
        blocked.write_text("occupies the output_root path", encoding="utf-8")
        config.paths.output_root = blocked
        with pytest.raises(ReportingError, match="Could not write report"):
            md.write_phase_ab_report(config, "# report\n")


# ---------------------------------------------------------------------------
# review fix: None metric is worst in tie-breaks, never best
# ---------------------------------------------------------------------------


class TestTieBreakNoneIsWorst:
    def _full_run(self, baseline: str, ba: float, f1: float | None, auroc: float | None):
        return _summary(
            baseline,
            "completed",
            seed=70 if baseline == "clip_probe" else None,
            total_records=2,
            overall=_overall_metrics(balanced_accuracy=ba, f1=f1, auroc=auroc),
        )

    def test_ba_tie_f1_none_loses_to_real_f1(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            self._full_run("clip_probe", 0.90, 0.80, 0.80),
            self._full_run("qwen_vl", 0.90, None, 0.95),
            self._full_run("assisted_qwen", 0.70, 0.70, 0.70),
            self._full_run("npr", 0.60, 0.60, 0.60),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "Best observed baseline by balanced_accuracy: clip_probe (0.9000)." in text
        assert "Best observed baseline by balanced_accuracy: qwen_vl" not in text

    def test_ba_and_f1_tie_auroc_none_loses(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            self._full_run("clip_probe", 0.90, 0.80, 0.80),
            self._full_run("qwen_vl", 0.90, 0.80, None),
            self._full_run("assisted_qwen", 0.70, 0.70, 0.70),
            self._full_run("npr", 0.60, 0.60, 0.60),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "Best observed baseline by balanced_accuracy: clip_probe (0.9000)." in text
        assert "Best observed baseline by balanced_accuracy: qwen_vl" not in text

    def test_all_metrics_tie_resolves_by_baseline_order(self, tmp_path):
        config = _make_config(tmp_path)
        runs = [
            self._full_run("qwen_vl", 0.90, 0.80, 0.80),
            self._full_run("clip_probe", 0.90, 0.80, 0.80),
            self._full_run("assisted_qwen", 0.70, 0.70, 0.70),
            self._full_run("npr", 0.60, 0.60, 0.60),
        ]
        text = md.build_phase_ab_report(config, runs)
        assert "Best observed baseline by balanced_accuracy: clip_probe (0.9000)." in text
