"""Deterministic Phase A/B Markdown reporting.

Reporting is a read-only layer over run artifacts produced by Tasks 6-10: it
never runs baselines, recomputes metrics, imports model runtimes, or touches
the network. The same artifacts and config must produce byte-for-byte stable
report content.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from aiforensics.config.models import AppConfig
from aiforensics.evaluation.metrics import METRIC_NAMES
from aiforensics.runs.artifacts import RunStatus, clip_seed_from_run_id
from aiforensics.runs.scope import compute_run_scope, scope_matches

__all__ = [
    "ReportingError",
    "RunSummary",
    "SourceMetricRow",
    "discover_run_summaries",
    "build_phase_ab_report",
    "write_phase_ab_report",
]


class ReportingError(ValueError):
    """Raised when selected run artifacts or report settings are invalid."""


MetricValue = float | None
ReportStatus = Literal["completed", "failed", "deferred", "missing"]

_BASELINE_ORDER: tuple[str, ...] = ("clip_probe", "qwen_vl", "assisted_qwen", "npr")

# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class SourceMetricRow:
    source: str
    n: int
    accuracy: MetricValue
    balanced_accuracy: MetricValue
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    auroc: MetricValue


@dataclass(frozen=True)
class RunSummary:
    baseline: str
    seed: int | None
    run_id: str | None
    status: ReportStatus
    reason: str | None
    run_dir: Path | None
    started_at: str | None
    ended_at: str | None
    total_records: int | None
    overall: dict[str, MetricValue] | None
    by_source: tuple[SourceMetricRow, ...]
    prediction_path: Path | None


# ------------------------------------------------------------------ formatting helpers


def _fmt_metric(value: MetricValue) -> str:
    """Render one metric value to four decimals, or N/A when unavailable."""
    return "N/A" if value is None else f"{value:.4f}"


def _fmt_mean_std(values: Sequence[MetricValue]) -> str:
    """Render already-computed CLIP metric values as mean +/- sample_std.

    Zero valid values -> N/A; one valid value -> the value itself; two or more
    -> mean +/- sample standard deviation, both to four decimals.
    """
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return "N/A"
    if len(valid) == 1:
        return f"{valid[0]:.4f}"
    return f"{statistics.mean(valid):.4f} +/- {statistics.stdev(valid):.4f}"


def _escape_cell(text: object) -> str:
    """Make dynamic text safe inside a Markdown table cell.

    Pipe characters are escaped and embedded newlines are collapsed to spaces
    so a runtime error message cannot corrupt the report table.
    """
    s = "" if text is None else str(text)
    return s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _resolve_report_path(config: AppConfig) -> Path:
    """Validate ``config.report.filename`` and resolve it under ``output_root``."""
    raw = config.report.filename
    if not isinstance(raw, str):
        raise ReportingError(f"report.filename must be a string, got {type(raw).__name__}")
    filename = raw.strip()
    if not filename:
        raise ReportingError("report.filename is empty")
    candidate = Path(filename)
    if candidate.is_absolute() or candidate.name != filename:
        raise ReportingError(
            f"report.filename must be a plain basename without directory "
            f"components, got: {filename!r}"
        )
    if ".." in candidate.parts:
        raise ReportingError(
            f"report.filename must not contain '..' traversal components, got: {filename!r}"
        )
    if candidate.suffix.lower() != ".md":
        raise ReportingError(
            f"report.filename must end with .md (case-insensitive), got: {filename!r}"
        )
    return config.paths.output_root / candidate


# ----------------------------------------------------------------- artifact parsing


def _parse_status(status_path: Path) -> tuple[RunStatus, datetime]:
    """Parse one ``status.json`` through the Task 6 ``RunStatus`` model.

    Returns the validated status record and its parsed ``ended_at`` timestamp
    (used for latest-run selection ordering).
    """
    try:
        raw = status_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportingError(f"Could not read status file {status_path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReportingError(f"Malformed status.json at {status_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportingError(f"status.json must contain a JSON object at {status_path}")
    try:
        status = RunStatus(**payload)
    except Exception as exc:
        raise ReportingError(f"Invalid status.json at {status_path}: {exc}") from exc

    normalized = (
        status.ended_at[:-1] + "+00:00" if status.ended_at.endswith("Z") else status.ended_at
    )
    ended_at = datetime.fromisoformat(normalized)
    return status, ended_at


def _parse_metric_value(value: object, name: str, path: Path) -> MetricValue:
    """Validate one metric value: null or finite numeric in [0.0, 1.0]."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportingError(f"Metric {name!r} must be null or numeric at {path}, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ReportingError(f"Metric {name!r} must be finite at {path}, got {parsed}")
    if not 0.0 <= parsed <= 1.0:
        raise ReportingError(f"Metric {name!r} out of range [0.0, 1.0] at {path}: {parsed}")
    return parsed


def _load_metrics_json(path: Path) -> tuple[int, dict[str, MetricValue]]:
    """Load and validate ``metrics.json``; returns (total_records, overall)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportingError(f"Could not read metrics file {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReportingError(f"Malformed metrics.json at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportingError(f"metrics.json root must be an object at {path}")

    total_records = payload.get("total_records")
    if isinstance(total_records, bool) or not isinstance(total_records, int) or total_records < 0:
        raise ReportingError(
            f"metrics.json total_records must be a non-negative integer at {path}, "
            f"got {total_records!r}"
        )

    overall = payload.get("overall")
    if not isinstance(overall, dict):
        raise ReportingError(f"metrics.json overall must be an object at {path}")

    parsed: dict[str, MetricValue] = {}
    for name in METRIC_NAMES:
        if name not in overall:
            raise ReportingError(f"metrics.json overall is missing metric {name!r} at {path}")
        parsed[name] = _parse_metric_value(overall[name], name, path)
    return total_records, parsed


def _load_metrics_by_source_csv(path: Path) -> tuple[SourceMetricRow, ...]:
    """Load and validate ``metrics_by_source.csv`` into ascending source order."""
    try:
        frame = pd.read_csv(path, dtype={"source": str})
    except OSError as exc:
        raise ReportingError(f"Could not read metrics CSV {path}: {exc}") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise ReportingError(f"Malformed metrics CSV at {path}: {exc}") from exc

    required = ["source", "n", *METRIC_NAMES]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ReportingError(f"metrics CSV at {path} is missing required columns: {missing}")

    rows: list[SourceMetricRow] = []
    seen_sources: set[str] = set()
    for record in frame.to_dict(orient="records"):
        source = record.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ReportingError(
                f"metrics CSV at {path} has an empty or invalid source value: {source!r}"
            )
        source = source.strip()
        if source in seen_sources:
            raise ReportingError(f"metrics CSV at {path} has duplicate source rows: {source!r}")
        seen_sources.add(source)

        n_raw = record.get("n")
        if n_raw is None or bool(pd.isna(n_raw)):
            raise ReportingError(f"metrics CSV at {path} has a blank n for {source!r}")
        try:
            n_value = float(n_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ReportingError(
                f"metrics CSV at {path} has non-numeric n for {source!r}: {n_raw!r}"
            ) from exc
        if not n_value.is_integer() or n_value < 0:
            raise ReportingError(f"metrics CSV at {path} has invalid n for {source!r}: {n_raw!r}")

        metrics: dict[str, MetricValue] = {}
        for name in METRIC_NAMES:
            value = record.get(name)
            if value is None or bool(pd.isna(value)):
                metrics[name] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ReportingError(
                    f"metrics CSV at {path} has invalid metric {name!r} for {source!r}: {value!r}"
                )
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ReportingError(
                    f"metrics CSV at {path} has non-numeric metric {name!r} for "
                    f"{source!r}: {value!r}"
                ) from exc
            metrics[name] = _parse_metric_value(parsed, name, path)

        rows.append(SourceMetricRow(source=source, n=int(n_value), **metrics))

    rows.sort(key=lambda row: row.source)
    return tuple(rows)


# ------------------------------------------------------------- discovery / slots


def _expected_slots(config: AppConfig) -> tuple[tuple[str, int | None], ...]:
    """Deterministic run slots: baseline order is fixed; CLIP seeds ascend.

    clip_probe enabled -> one slot per configured seed
    clip_probe disabled -> one seed-less slot
    qwen_vl / assisted_qwen / npr -> one slot each
    """
    slots: list[tuple[str, int | None]] = []
    clip_cfg = config.baselines.clip_probe
    if clip_cfg.enabled:
        for seed in sorted(clip_cfg.seeds):
            slots.append(("clip_probe", seed))
    else:
        slots.append(("clip_probe", None))
    slots.append(("qwen_vl", None))
    slots.append(("assisted_qwen", None))
    slots.append(("npr", None))
    return tuple(slots)


def _missing_summary(baseline: str, seed: int | None) -> RunSummary:
    return RunSummary(
        baseline=baseline,
        seed=seed,
        run_id=None,
        status="missing",
        reason=None,
        run_dir=None,
        started_at=None,
        ended_at=None,
        total_records=None,
        overall=None,
        by_source=(),
        prediction_path=None,
    )


def _build_summary_for_run_dir(run_dir: Path, baseline: str, seed: int | None) -> RunSummary:
    """Materialize a RunSummary from one selected run directory.

    Completed runs require predictions.jsonl, metrics.json, and
    metrics_by_source.csv; failed/deferred runs never read metric artifacts so
    stale metrics cannot leak into comparisons.
    """
    status_path = run_dir / "status.json"
    status, _ended_at = _parse_status(status_path)

    prediction_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    by_source_path = run_dir / "metrics_by_source.csv"

    if status.status == "completed":
        if not prediction_path.is_file():
            raise ReportingError(
                f"Selected completed run {run_dir.name!r} is missing predictions.jsonl at {run_dir}"
            )
        if not metrics_path.is_file() or not by_source_path.is_file():
            raise ReportingError(
                f"Selected completed run {run_dir.name!r} has no metrics artifacts "
                f"at {run_dir}; run 'aiforensics evaluate --config <config>' first."
            )
        total_records, overall = _load_metrics_json(metrics_path)
        by_source = _load_metrics_by_source_csv(by_source_path)
        return RunSummary(
            baseline=baseline,
            seed=seed,
            run_id=run_dir.name,
            status="completed",
            reason=status.reason,
            run_dir=run_dir,
            started_at=status.started_at,
            ended_at=status.ended_at,
            total_records=total_records,
            overall=overall,
            by_source=by_source,
            prediction_path=prediction_path,
        )

    return RunSummary(
        baseline=baseline,
        seed=seed,
        run_id=run_dir.name,
        status=status.status,
        reason=status.reason,
        run_dir=run_dir,
        started_at=status.started_at,
        ended_at=status.ended_at,
        total_records=None,
        overall=None,
        by_source=(),
        prediction_path=None,
    )


def _current_run_scope(config: AppConfig):
    """Compute the current scope, reporting manifest problems as ReportingError.

    Scope computation reads evaluation manifests, so a corrupt manifest must
    surface as a reporting error with a clear message instead of escaping the
    CLI's error handling as a raw ManifestError.
    """
    from aiforensics.data.manifest import ManifestError

    try:
        return compute_run_scope(config)
    except ManifestError as exc:
        raise ReportingError(
            f"Could not determine the current run scope because an evaluation "
            f"manifest is invalid: {exc}"
        ) from exc


def discover_run_summaries(config: AppConfig) -> list[RunSummary]:
    """Discover run artifacts and select the latest run per expected slot.

    Selection policy: scan immediate child directories of ``output_root``,
    validate every ``status.json`` through the Task 6 ``RunStatus`` model,
    ignore unknown baselines, ignore runs whose ``run_scope.json`` does not
    match the current config's scope, map candidates onto expected slots (CLIP
    seeds via the ``_clip_probe_seed<N>`` run-id suffix), then pick the
    candidate with the greatest ``ended_at`` per slot, breaking exact timestamp
    ties by the lexicographically greatest directory name. Slots without
    candidates become reporting-only ``missing`` summaries. The latest run is
    the truthful result even when it failed, deferred, or lacks metrics.
    """
    output_root = config.paths.output_root
    expected_scope = _current_run_scope(config)
    candidates: dict[tuple[str, int | None], list[tuple[datetime, str, Path]]] = {}

    if output_root.is_dir():
        for entry in sorted(output_root.iterdir()):
            if not entry.is_dir():
                continue
            status_path = entry / "status.json"
            if not status_path.is_file():
                continue
            status, ended_at = _parse_status(status_path)
            if status.baseline not in _BASELINE_ORDER:
                continue
            if not scope_matches(entry, expected_scope):
                continue  # artifact belongs to a different config/dataset slice

            if status.baseline == "clip_probe" and config.baselines.clip_probe.enabled:
                seed = clip_seed_from_run_id(entry.name)
                if seed is None:
                    continue  # suffix-less CLIP runs only feed the disabled slot
                slot = ("clip_probe", seed)
            elif status.baseline == "clip_probe":
                if clip_seed_from_run_id(entry.name) is not None:
                    continue  # suffixed runs belong only to seed slots (rule 7)
                slot = ("clip_probe", None)
            else:
                slot = (status.baseline, None)

            candidates.setdefault(slot, []).append((ended_at, entry.name, entry))

    summaries: list[RunSummary] = []
    for slot in _expected_slots(config):
        slot_candidates = candidates.get(slot, [])
        if not slot_candidates:
            summaries.append(_missing_summary(*slot))
            continue
        _, _, latest_dir = max(slot_candidates, key=lambda item: (item[0], item[1]))
        summaries.append(_build_summary_for_run_dir(latest_dir, slot[0], slot[1]))
    return summaries


# ------------------------------------------------------------------ rendering


def _display_path(path: Path) -> str:
    """Prefer repository-relative display; fall back to the final component."""
    if path.is_absolute():
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return path.name
    return str(path)


def _seed_label(seed: int | None) -> str:
    return "N/A" if seed is None else str(seed)


def _slot_key(summary: RunSummary) -> tuple[str, int | None]:
    return (summary.baseline, summary.seed)


def _clip_stats(runs: Sequence[RunSummary]) -> tuple[list[RunSummary], int]:
    """Return (completed seed runs, expected seed slot count) for CLIP."""
    clip_runs = [r for r in runs if r.baseline == "clip_probe"]
    completed = [r for r in clip_runs if r.status == "completed"]
    return completed, len(clip_runs)


def _clip_metric_values(completed_seeds: Sequence[RunSummary], metric: str) -> list[float]:
    values: list[float] = []
    for run in completed_seeds:
        overall = run.overall
        if overall is None:
            continue
        value = overall.get(metric)
        if value is not None:
            values.append(float(value))
    return values


def _clip_per_source_cells(
    completed_seeds: Sequence[RunSummary],
) -> list[tuple[str, int, dict[str, list[MetricValue]]]]:
    """Aggregate per-source CLIP rows across completed seeds.

    All contributing seeds must agree on ``n`` for a source; otherwise seed
    aggregation would be scientifically misleading and a ReportingError is
    raised. Metric values stay as lists so cells can render mean +/- std.
    """
    by_source: dict[str, list[SourceMetricRow]] = {}
    for run in completed_seeds:
        for row in run.by_source:
            by_source.setdefault(row.source, []).append(row)

    aggregates: list[tuple[str, int, dict[str, list[MetricValue]]]] = []
    for source in sorted(by_source):
        contributions = by_source[source]
        distinct_n = {row.n for row in contributions}
        if len(distinct_n) > 1:
            raise ReportingError(
                f"CLIP per-source metric rows disagree on n for source "
                f"{source!r} across completed seeds: {sorted(distinct_n)}. "
                f"Seed aggregation would be misleading; investigate the runs."
            )
        values: dict[str, list[MetricValue]] = {
            name: [getattr(row, name) for row in contributions] for name in METRIC_NAMES
        }
        aggregates.append((source, contributions[0].n, values))
    return aggregates


def _configured_slots(config: AppConfig) -> dict[tuple[str, int | None], str]:
    """Map every expected slot to its configured enabled/disabled state."""
    clip_cfg = config.baselines.clip_probe
    slots: dict[tuple[str, int | None], str] = {}
    if clip_cfg.enabled:
        for seed in sorted(clip_cfg.seeds):
            slots[("clip_probe", seed)] = "enabled"
    else:
        slots[("clip_probe", None)] = "disabled"
    slots[("qwen_vl", None)] = "enabled" if config.baselines.qwen_vl.enabled else "disabled"
    slots[("assisted_qwen", None)] = (
        "enabled" if config.baselines.assisted_qwen.enabled else "disabled"
    )
    slots[("npr", None)] = "enabled" if config.baselines.npr.enabled else "disabled"
    return slots


def _resolved_runs(config: AppConfig, runs: Sequence[RunSummary]) -> list[RunSummary]:
    """Return one summary per expected slot in fixed order.

    Slots without a provided summary become reporting-only ``missing`` rows;
    historical or unexpected summaries never enter the report.
    """
    by_slot: dict[tuple[str, int | None], RunSummary] = {}
    for summary in runs:
        by_slot.setdefault(_slot_key(summary), summary)
    resolved: list[RunSummary] = []
    for slot, _state in _configured_slots(config).items():
        resolved.append(by_slot.get(slot) or _missing_summary(*slot))
    return resolved


def _coverage_signature(summary: RunSummary) -> tuple:
    sources = tuple(sorted((row.source, row.n) for row in summary.by_source))
    return (summary.total_records, sources)


def _render_project(config: AppConfig, lines: list[str]) -> None:
    lines.extend(
        [
            "## Project",
            "",
            f"- **Project:** {_escape_cell(config.project.name)}",
            f"- **Phase:** {_escape_cell(config.project.phase)}",
            f"- **Description:** {_escape_cell(config.project.description)}",
            f"- **Output root:** {_escape_cell(_display_path(config.paths.output_root))}",
            "",
        ]
    )


def _render_dataset_summary(
    config: AppConfig, resolved: list[RunSummary], lines: list[str]
) -> bool:
    lines.extend(["## Dataset Summary", "", "Configured datasets:", ""])
    lines.append("| Dataset | Enabled | Evaluation split/source | Manifest |")
    lines.append("| --- | --- | --- | --- |")

    datasets = config.datasets
    tiny = datasets.tiny_genimage
    unseen = datasets.genimage_unseen
    synth = datasets.synthbuster
    rows = [
        (
            "tiny_genimage",
            tiny.enabled,
            f"source={tiny.source}; evaluation split=dev (dev_manifest)",
            tiny.dev_manifest,
        ),
        (
            "genimage_unseen",
            unseen.enabled,
            f"preferred_generator={unseen.preferred_generator}; split={unseen.split}",
            unseen.manifest,
        ),
        (
            "synthbuster",
            synth.enabled,
            f"split={synth.split}",
            synth.manifest,
        ),
    ]
    for name, enabled, descriptor, manifest in rows:
        lines.append(
            f"| {name} | {'enabled' if enabled else 'disabled'} "
            f"| {_escape_cell(descriptor)} | {_escape_cell(_display_path(manifest))} |"
        )

    lines.extend(["", "Observed evaluation coverage (from selected completed runs):", ""])
    lines.append("| Baseline | Seed | Total records | Source counts |")
    lines.append("| --- | --- | --- | --- |")
    completed = [r for r in resolved if r.status == "completed"]
    for run in completed:
        counts = "; ".join(f"{row.source} (n={row.n})" for row in run.by_source) or "N/A"
        lines.append(
            f"| {_escape_cell(run.baseline)} | {_seed_label(run.seed)} "
            f"| {_escape_cell(run.total_records)} | {_escape_cell(counts)} |"
        )
    if not completed:
        lines.append("| N/A | N/A | N/A | N/A |")

    coverage_ok = len({(_coverage_signature(r)) for r in completed}) <= 1
    if not coverage_ok:
        lines.extend(
            [
                "",
                "**Warning:** Completed selected runs disagree on evaluation "
                "coverage (total records and/or per-source counts). Comparative "
                "recommendation is blocked until evaluation coverage is aligned.",
            ]
        )
    lines.append("")
    return coverage_ok


def _render_baseline_status(
    config: AppConfig, resolved: list[RunSummary], lines: list[str]
) -> None:
    configured = _configured_slots(config)
    lines.extend(
        [
            "## Baseline Status",
            "",
            "One row per expected run slot; `missing` means no run artifact "
            "exists under the configured output_root and is not a successful "
            "outcome.",
            "",
            "| Baseline | Seed | Configured | Status | Run ID | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for run in resolved:
        state = configured[_slot_key(run)]
        lines.append(
            f"| {_escape_cell(run.baseline)} | {_seed_label(run.seed)} | {state} "
            f"| {run.status} | {_escape_cell(run.run_id or 'N/A')} "
            f"| {_escape_cell(run.reason if run.reason is not None else 'N/A')} |"
        )
    lines.append("")


def _render_overall_metrics(
    resolved: list[RunSummary],
    clip_completed: list[RunSummary],
    clip_expected: int,
    lines: list[str],
) -> None:
    lines.extend(
        [
            "## Overall Metrics",
            "",
            "Values are rendered from artifacts produced by "
            "`aiforensics evaluate`; no metric is recomputed here.",
            "",
            "| Baseline | " + " | ".join(METRIC_NAMES) + " | Completion |",
            "| --- | " + " | ".join(["---"] * len(METRIC_NAMES)) + " | --- |",
        ]
    )
    clip_rendered = False
    for run in resolved:
        if run.baseline == "clip_probe":
            # CLIP is one aggregated row across all seed slots, not one row
            # per seed slot.
            if clip_rendered:
                continue
            clip_rendered = True
            if clip_completed:
                cells = [
                    _fmt_mean_std(_clip_metric_values(clip_completed, name))
                    for name in METRIC_NAMES
                ]
            else:
                cells = ["N/A"] * len(METRIC_NAMES)
            completion = f"{len(clip_completed)}/{clip_expected} seeds completed"
            lines.append("| clip_probe | " + " | ".join(cells) + f" | {completion} |")
            continue
        if run.status == "completed" and run.overall is not None:
            cells = [_fmt_metric(run.overall.get(name)) for name in METRIC_NAMES]
            completion = "completed"
        else:
            cells = ["N/A"] * len(METRIC_NAMES)
            completion = run.status
        lines.append(
            f"| {_escape_cell(run.baseline)} | " + " | ".join(cells) + f" | {completion} |"
        )
    lines.append("")


def _render_metric_cells(values: Sequence[MetricValue]) -> str:
    """Render one per-source metric cell (single value or mean +/- std)."""
    return _fmt_mean_std(list(values))


def _render_per_source(
    resolved: list[RunSummary], clip_completed: list[RunSummary], lines: list[str]
) -> None:
    lines.extend(
        [
            "## Per-Source Metrics",
            "",
            "| Baseline | Source | n | "
            + " | ".join(name.capitalize() for name in METRIC_NAMES)
            + " |",
            "| --- | --- | --- | " + " | ".join(["---"] * len(METRIC_NAMES)) + " |",
        ]
    )
    clip_cells = (
        {source: (n, values) for source, n, values in _clip_per_source_cells(clip_completed)}
        if clip_completed
        else {}
    )
    clip_rendered = False
    for run in resolved:
        if run.baseline == "clip_probe":
            # CLIP source rows are aggregated across seed slots; render once.
            if clip_rendered:
                continue
            clip_rendered = True
            for source in sorted(clip_cells):
                n, values = clip_cells[source]
                cells = [_render_metric_cells(values[name]) for name in METRIC_NAMES]
                lines.append(
                    f"| clip_probe | {_escape_cell(source)} | {n} | " + " | ".join(cells) + " |"
                )
        elif run.status == "completed":
            for row in sorted(run.by_source, key=lambda r: r.source):
                cells = [_fmt_metric(getattr(row, name)) for name in METRIC_NAMES]
                lines.append(
                    f"| {_escape_cell(run.baseline)} | {_escape_cell(row.source)} "
                    f"| {row.n} | " + " | ".join(cells) + " |"
                )
    lines.append("")


def _render_failure_notes(resolved: list[RunSummary], lines: list[str]) -> None:
    lines.extend(["## Failure and Deferred Notes", ""])
    for run in resolved:
        if run.status == "completed":
            continue
        if run.status == "missing":
            reason = "No run artifact found under the configured output_root."
        else:
            reason = run.reason or "N/A"
        lines.append(
            f"- {_escape_cell(run.baseline)} (seed {_seed_label(run.seed)}): "
            f"{run.status} - {_escape_cell(reason)}"
        )
    lines.append("")


def _render_explanations(config: AppConfig, resolved: list[RunSummary], lines: list[str]) -> None:
    from aiforensics.schemas.predictions import (
        PredictionError,
        PredictionRecord,
        load_predictions,
    )

    size = config.report.explanation_sample_size
    candidates: list[tuple[str, int, PredictionRecord]] = []
    for order, baseline in enumerate(("qwen_vl", "assisted_qwen")):
        run = next((r for r in resolved if r.baseline == baseline), None)
        if run is None or run.status != "completed":
            continue
        if run.prediction_path is None or not run.prediction_path.is_file():
            continue
        try:
            records = load_predictions(run.prediction_path)
        except PredictionError as exc:
            raise ReportingError(
                f"Could not read predictions for explanation sampling at "
                f"{run.prediction_path}: {exc}"
            ) from exc
        for record in records:
            if record.explanation:
                candidates.append((record.sample_id, order, record))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[:size] if size > 0 else []

    lines.extend(["## Explanation Samples", ""])
    if not selected:
        lines.append("No explanation samples available.")
        lines.append("")
        return

    lines.append("| Baseline | Sample ID | True | Predicted | Parse status | Explanation |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for sample_id, _order, record in selected:
        lines.append(
            f"| {_escape_cell(record.model_name)} | {_escape_cell(sample_id)} "
            f"| {_escape_cell(record.label_true)} | {_escape_cell(record.label_pred)} "
            f"| {_escape_cell(record.parse_status)} | {_escape_cell(record.explanation)} |"
        )
    lines.append("")


def _metric_mean(values: Sequence[MetricValue]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return statistics.mean(valid) if valid else None


def _enabled_baselines(config: AppConfig) -> set[str]:
    return {
        baseline
        for baseline, cfg in (
            ("clip_probe", config.baselines.clip_probe),
            ("qwen_vl", config.baselines.qwen_vl),
            ("assisted_qwen", config.baselines.assisted_qwen),
            ("npr", config.baselines.npr),
        )
        if cfg.enabled
    }


def _baseline_comparables(
    config: AppConfig,
    resolved: list[RunSummary],
    clip_completed: list[RunSummary],
) -> dict[str, dict[str, float | None]]:
    """Collect mean balanced_accuracy/f1/auroc per completed enabled baseline.

    Disabled baselines never enter the comparison even when historical run
    artifacts are completed: the recommendation reflects the current config.
    """
    enabled = _enabled_baselines(config)
    comparables: dict[str, dict[str, float | None]] = {}
    for run in resolved:
        if run.baseline not in enabled:
            continue
        if run.baseline == "clip_probe":
            if clip_completed and "clip_probe" not in comparables:
                comparables["clip_probe"] = {
                    name: _metric_mean(_clip_metric_values(clip_completed, name))
                    for name in ("balanced_accuracy", "f1", "auroc")
                }
            continue
        if run.status == "completed" and run.overall is not None:
            comparables[run.baseline] = {
                name: run.overall.get(name) for name in ("balanced_accuracy", "f1", "auroc")
            }
    return comparables


def _render_recommendation(
    config: AppConfig,
    resolved: list[RunSummary],
    clip_completed: list[RunSummary],
    clip_expected: int,
    coverage_ok: bool,
    lines: list[str],
) -> None:
    lines.extend(["## Next-Step Recommendation", ""])

    if config.project.phase == "phase_ab_smoke":
        lines.extend(
            [
                "This report was generated for the smoke phase. Smoke metrics "
                "validate pipeline behavior only; they are **not scientific "
                "evidence** and must not be used as a later-phase go/no-go "
                "decision.",
                "",
            ]
        )
        return

    configured = _configured_slots(config)
    enabled_slots = [slot for slot, state in configured.items() if state == "enabled"]
    unresolved = [
        run for run in resolved if _slot_key(run) in enabled_slots and run.status != "completed"
    ]

    if unresolved:
        details = ", ".join(
            f"{run.baseline}"
            + (f" (seed {_seed_label(run.seed)})" if run.seed is not None else "")
            + f" [{run.status}]"
            for run in unresolved
        )
        lines.extend(
            [
                "Comparative Phase A/B evidence is incomplete: the following "
                f"enabled slots are unresolved — {details}.",
                "No winner is selected while evidence is incomplete.",
                "",
            ]
        )
        return

    if not coverage_ok:
        lines.extend(
            [
                "Evaluation coverage must be aligned before comparing baselines: "
                "completed selected runs disagree on total records and/or "
                "per-source counts. No winner is selected under this condition.",
                "",
            ]
        )
        return

    comparables = _baseline_comparables(config, resolved, clip_completed)
    scored = [
        (baseline, values)
        for baseline, values in comparables.items()
        if values.get("balanced_accuracy") is not None
    ]
    if not scored:
        lines.extend(
            [
                "No completed baseline has a usable balanced_accuracy value; "
                "no comparative recommendation can be made.",
                "",
            ]
        )
        return

    order = {baseline: idx for idx, baseline in enumerate(_BASELINE_ORDER)}

    def _sort_key(item: tuple[str, dict[str, float | None]]) -> tuple:
        _baseline, values = item
        ba = values.get("balanced_accuracy")
        f1 = values.get("f1")
        auroc = values.get("auroc")
        # Ascending sort: higher metric = better, so a missing metric must sort
        # as the WORST value (+inf), never the best (-inf), or a baseline with
        # no F1/AUROC could win its tie-break unfairly.
        return (
            -float(ba) if ba is not None else float("inf"),
            -float(f1) if f1 is not None else float("inf"),
            -float(auroc) if auroc is not None else float("inf"),
            order.get(_baseline, len(_BASELINE_ORDER)),
        )

    scored.sort(key=_sort_key)
    best_baseline, best_values = scored[0]
    best_ba_value = best_values.get("balanced_accuracy")
    best_ba = float(best_ba_value) if best_ba_value is not None else float("nan")
    lines.extend(
        [
            f"Best observed baseline by balanced_accuracy: "
            f"{_escape_cell(best_baseline)} ({best_ba:.4f}).",
            "No claim of statistical significance is made.",
        ]
    )

    assisted = comparables.get("assisted_qwen")
    qwen = comparables.get("qwen_vl")
    assisted_ba = assisted.get("balanced_accuracy") if assisted else None
    qwen_ba = qwen.get("balanced_accuracy") if qwen else None
    # Spec: both Qwen-VL and Assisted Qwen must be enabled/completed with
    # non-null balanced accuracy before the delta is computed.
    if (
        config.baselines.qwen_vl.enabled
        and config.baselines.assisted_qwen.enabled
        and assisted_ba is not None
        and qwen_ba is not None
    ):
        delta = float(assisted_ba) - float(qwen_ba)
        lines.append("")
        if delta > 0:
            lines.extend(
                [
                    f"Assisted Qwen improved observed balanced_accuracy by "
                    f"{delta:+.4f} over Qwen-VL. The next experiment should "
                    "prioritize analysis of classifier-assisted fusion and "
                    "evidence transfer.",
                ]
            )
        else:
            lines.extend(
                [
                    f"Classifier assistance did not improve observed balanced "
                    f"accuracy ({delta:+.4f} vs Qwen-VL). Perform error analysis "
                    "before treating assistance/fusion as validated.",
                ]
            )

    lines.extend(
        [
            "",
            "Classification improvement must never be interpreted as proof of "
            "explanation faithfulness.",
            "",
        ]
    )


def build_phase_ab_report(config: AppConfig, runs: Sequence[RunSummary]) -> str:
    """Render the deterministic Phase A/B Markdown report.

    Pure function: consumes prepared run summaries and performs no filesystem
    discovery. Identical inputs produce byte-for-byte identical output.
    """
    resolved = _resolved_runs(config, runs)
    clip_completed, clip_expected = _clip_stats(resolved)

    lines: list[str] = ["# Phase A/B Baseline Report", ""]
    _render_project(config, lines)
    coverage_ok = _render_dataset_summary(config, resolved, lines)
    _render_baseline_status(config, resolved, lines)
    _render_overall_metrics(resolved, clip_completed, clip_expected, lines)
    _render_per_source(resolved, clip_completed, lines)
    if config.report.include_failure_notes:
        _render_failure_notes(resolved, lines)
    if config.report.include_explanations_sample:
        _render_explanations(config, resolved, lines)
    _render_recommendation(config, resolved, clip_completed, clip_expected, coverage_ok, lines)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_phase_ab_report(config: AppConfig, report_text: str) -> Path:
    """Validate the report filename and write the report under ``output_root``."""
    path = _resolve_report_path(config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text.rstrip("\n") + "\n", encoding="utf-8")
    except OSError as exc:
        raise ReportingError(f"Could not write report to {path}: {exc}") from exc
    return path
