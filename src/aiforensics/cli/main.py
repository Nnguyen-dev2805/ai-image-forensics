"""argparse-based CLI for the aiforensics Phase A/B baseline suite."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

from aiforensics.config import load_config

__all__ = ["build_parser", "main"]

SUPPORTED_BASELINES: tuple[str, ...] = (
    "clip_probe",
    "qwen_vl",
    "npr",
    "assisted_qwen",
)


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to a Phase A/B YAML config file.",
    )


def _build_prepare_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "prepare",
        help="Validate or build dataset manifests.",
    )
    parser.add_argument(
        "--build-manifests",
        action="store_true",
        help=(
            "Build manifests from a GenImage-layout data_root before validating. "
            "This overwrites the configured manifest CSV files."
        ),
    )
    _add_config_arg(parser)
    parser.set_defaults(handler=_cmd_prepare)


def _build_run_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run a single baseline against a manifest.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=SUPPORTED_BASELINES,
        help="Baseline adapter to execute.",
    )
    _add_config_arg(parser)
    parser.set_defaults(handler=_cmd_run)


def _build_evaluate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "evaluate",
        help="Compute metrics from prediction artifacts.",
    )
    _add_config_arg(parser)
    parser.set_defaults(handler=_cmd_evaluate)


def _build_report_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "report",
        help="Render a Markdown report from run artifacts.",
    )
    _add_config_arg(parser)
    parser.set_defaults(handler=_cmd_report)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="aiforensics",
        description=("Reproducible baseline suite for AI-generated image detection (Phase A/B)."),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        required=True,
    )
    _build_prepare_parser(subparsers)
    _build_run_parser(subparsers)
    _build_evaluate_parser(subparsers)
    _build_report_parser(subparsers)
    return parser


def _cmd_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    if config.project.phase == "phase_ab_smoke":
        from aiforensics.data.manifest import prepare_smoke_manifest

        result = prepare_smoke_manifest(config)
    else:
        from aiforensics.data.manifest import (
            ManifestError,
            load_manifest,
            validate_manifest,
        )

        # Building overwrites manifest CSVs, so it stays opt-in behind a flag
        # rather than happening implicitly whenever manifests are absent.
        if getattr(args, "build_manifests", False):
            from aiforensics.data.genimage import build_genimage_manifests

            build_result = build_genimage_manifests(config)
            for built in build_result.manifests:
                print(
                    f"[prepare] built {built.label} split={built.split} "
                    f"records={built.record_count} real={built.real_count} "
                    f"fake={built.fake_count} generators={','.join(built.generators)} "
                    f"path={built.path}"
                )
                skew = built.format_skew()
                if skew is not None:
                    print(f"[prepare] format warning: {skew}")
            if build_result.duplicate_checksums_skipped:
                print(
                    f"[prepare] skipped {build_result.duplicate_checksums_skipped} "
                    f"duplicate image(s) already claimed by an earlier split"
                )
            for warning in build_result.warnings:
                print(f"[prepare] warning: {warning}")

        manifest_paths: list[pathlib.Path] = []
        if config.datasets.tiny_genimage.enabled:
            manifest_paths.extend(
                [
                    config.datasets.tiny_genimage.train_manifest,
                    config.datasets.tiny_genimage.dev_manifest,
                ]
            )
        if config.datasets.genimage_unseen.enabled:
            manifest_paths.append(config.datasets.genimage_unseen.manifest)
        if config.datasets.synthbuster.enabled:
            manifest_paths.append(config.datasets.synthbuster.manifest)

        all_records = []
        found_any = False

        for path in manifest_paths:
            if path.exists():
                found_any = True
                records = load_manifest(path, data_root=config.paths.data_root)
                all_records.extend(records)

        if not found_any:
            raise ManifestError(
                "No configured manifest was found. Build manifests from a "
                "GenImage-layout data_root with 'aiforensics prepare "
                "--build-manifests --config <config>', or provision manifest "
                "CSV files at the configured paths."
            )

        result = validate_manifest(all_records)

    config.paths.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = config.paths.output_root / "manifest_validation.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2) + "\n")

    print(
        f"[prepare] project={config.project.name} phase={config.project.phase} "
        f"records={result.total_records} valid={result.is_valid} summary={summary_path}"
    )

    return 0 if result.is_valid else 1


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    import shutil
    from datetime import datetime, timezone

    from aiforensics.runs.artifacts import (
        RunStatus,
        create_run_dir,
        write_environment,
        write_status,
    )
    from aiforensics.runs.scope import SCOPE_FILENAME, compute_run_scope, write_run_scope

    # Computed once per invocation: the scope stamped into each run directory is
    # the identity later commands use to tell this experiment's artifacts apart
    # from unrelated history under the same output_root.
    run_scope = compute_run_scope(config)

    def _setup_run_dir(baseline: str, run_name: str | None = None) -> pathlib.Path:
        run_dir = create_run_dir(config.paths.output_root, baseline, run_name=run_name)
        (run_dir / "logs.txt").touch()
        shutil.copy2(args.config, run_dir / "config.yaml")
        write_run_scope(run_dir / SCOPE_FILENAME, run_scope)
        write_environment(run_dir / "environment.json")
        return run_dir

    completed = 0
    failed = 0
    deferred = 0

    if args.baseline == "clip_probe":
        from aiforensics.baselines.clip_probe.adapter import ClipProbeAdapter

        if not config.baselines.clip_probe.enabled:
            started_at = datetime.now(timezone.utc).isoformat()
            run_dir = _setup_run_dir("clip_probe", None)
            ended_at = datetime.now(timezone.utc).isoformat()

            write_status(
                run_dir / "status.json",
                RunStatus(
                    baseline="clip_probe",
                    status="deferred",
                    reason="clip_probe is disabled in config",
                    command=sys.argv,
                    started_at=started_at,
                    ended_at=ended_at,
                ),
            )
            deferred += 1
        else:
            adapter = ClipProbeAdapter()
            for seed in config.baselines.clip_probe.seeds:
                started_at = datetime.now(timezone.utc).isoformat()
                run_dir = _setup_run_dir("clip_probe", f"seed{seed}")

                try:
                    result = adapter.run(
                        config=config,
                        output_dir=run_dir,
                        run_id=run_dir.name,
                        seed=seed,
                    )
                    final_status = result.status
                    final_reason = result.reason
                except Exception as e:
                    with open(run_dir / "logs.txt", "a", encoding="utf-8") as f:
                        f.write(f"[FAILED] Unexpected error: {e}\n")
                    final_status = "failed"
                    final_reason = f"Adapter crashed: {e}"

                ended_at = datetime.now(timezone.utc).isoformat()
                write_status(
                    run_dir / "status.json",
                    RunStatus(
                        baseline="clip_probe",
                        status=final_status,
                        reason=final_reason,  # type: ignore
                        command=sys.argv,
                        started_at=started_at,
                        ended_at=ended_at,
                    ),
                )

                if final_status == "completed":
                    completed += 1
                elif final_status == "failed":
                    failed += 1
                elif final_status == "deferred":
                    deferred += 1

    elif args.baseline == "qwen_vl":
        from aiforensics.baselines.qwen_vl.adapter import QwenVLAdapter

        adapter = QwenVLAdapter()
        started_at = datetime.now(timezone.utc).isoformat()

        run_dir = _setup_run_dir("qwen_vl", None)

        try:
            result = adapter.run(
                config=config,
                output_dir=run_dir,
                run_id=run_dir.name,
            )
            final_status = result.status
            final_reason = getattr(result, "reason", "")
        except Exception as e:
            with open(run_dir / "logs.txt", "a", encoding="utf-8") as f:
                f.write(f"[FAILED] Unexpected error: {e}\n")
            final_status = "failed"
            final_reason = f"Adapter crashed: {e}"

        ended_at = datetime.now(timezone.utc).isoformat()
        write_status(
            run_dir / "status.json",
            RunStatus(
                baseline="qwen_vl",
                status=final_status,
                reason=final_reason,  # type: ignore
                command=sys.argv,
                started_at=started_at,
                ended_at=ended_at,
            ),
        )

        if final_status == "completed":
            completed += 1
        elif final_status == "failed":
            failed += 1
        elif final_status == "deferred":
            deferred += 1
    elif args.baseline == "assisted_qwen":
        from aiforensics.baselines.assisted_qwen.adapter import AssistedQwenAdapter

        adapter = AssistedQwenAdapter()
        started_at = datetime.now(timezone.utc).isoformat()

        run_dir = _setup_run_dir("assisted_qwen", None)

        try:
            result = adapter.run(
                config=config,
                output_dir=run_dir,
                run_id=run_dir.name,
            )
            final_status = result.status
            final_reason = getattr(result, "reason", "")
        except Exception as e:
            with open(run_dir / "logs.txt", "a", encoding="utf-8") as f:
                f.write(f"[FAILED] Unexpected error: {e}\n")
            final_status = "failed"
            final_reason = f"Adapter crashed: {e}"

        ended_at = datetime.now(timezone.utc).isoformat()
        write_status(
            run_dir / "status.json",
            RunStatus(
                baseline="assisted_qwen",
                status=final_status,
                reason=final_reason,  # type: ignore
                command=sys.argv,
                started_at=started_at,
                ended_at=ended_at,
            ),
        )

        if final_status == "completed":
            completed += 1
        elif final_status == "failed":
            failed += 1
        elif final_status == "deferred":
            deferred += 1

    elif args.baseline == "npr":
        from aiforensics.baselines.npr.adapter import NPRAdapter

        adapter = NPRAdapter()
        started_at = datetime.now(timezone.utc).isoformat()

        run_dir = _setup_run_dir("npr", None)

        try:
            result = adapter.run(
                config=config,
                output_dir=run_dir,
                run_id=run_dir.name,
            )
            final_status = result.status
            final_reason = getattr(result, "reason", "")
        except Exception as e:
            with open(run_dir / "logs.txt", "a", encoding="utf-8") as f:
                f.write(f"[FAILED] Unexpected error: {e}\n")
            final_status = "failed"
            final_reason = f"Adapter crashed: {e}"

        ended_at = datetime.now(timezone.utc).isoformat()
        write_status(
            run_dir / "status.json",
            RunStatus(
                baseline="npr",
                status=final_status,
                reason=final_reason,  # type: ignore
                command=sys.argv,
                started_at=started_at,
                ended_at=ended_at,
            ),
        )

        if final_status == "completed":
            completed += 1
        elif final_status == "failed":
            failed += 1
        elif final_status == "deferred":
            deferred += 1

    print(
        f"[run] baseline={args.baseline} runs={completed + failed + deferred} "
        f"completed={completed} failed={failed} deferred={deferred} "
        f"output_root={config.paths.output_root}"
    )

    return 1 if failed > 0 else 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    from aiforensics.data.manifest import ManifestError
    from aiforensics.data.selection import selected_evaluation_manifests
    from aiforensics.evaluation.metrics import (
        MetricsError,
        discover_scoped_prediction_files,
        evaluate_prediction_file,
    )
    from aiforensics.runs.scope import compute_run_scope

    # Selecting once serves both purposes: the manifest ids that predictions are
    # cross-checked against, and the scope digest that decides which runs belong
    # to this config.
    try:
        selection = selected_evaluation_manifests(config, strict=False)
    except ManifestError as exc:
        print(f"Error evaluating: invalid evaluation manifest: {exc}")
        return 1

    expected_scope = compute_run_scope(config, selection=selection)
    manifest_sample_ids = selection.sample_ids or None

    files, skipped = discover_scoped_prediction_files(config.paths.output_root, expected_scope)
    count = len(files)

    for p_file in skipped:
        print(f"[evaluate] skipped out-of-scope run: {p_file.parent.name}")

    for p_file in files:
        try:
            evaluate_prediction_file(p_file, manifest_sample_ids=manifest_sample_ids)
        except MetricsError as e:
            print(f"Error evaluating {p_file}: {e}")
            return 1

    print(
        f"[evaluate] project={config.project.name} "
        f"phase={config.project.phase} prediction_files={count} "
        f"skipped_out_of_scope={len(skipped)} scope={expected_scope.scope_id[:12]} "
        f"output_root={config.paths.output_root}"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    from aiforensics.reporting.markdown import (
        ReportingError,
        build_phase_ab_report,
        discover_run_summaries,
        write_phase_ab_report,
    )

    try:
        runs = discover_run_summaries(config)
        text = build_phase_ab_report(config, runs)
        report_path = write_phase_ab_report(config, text)
    except ReportingError as exc:
        print(f"Error generating report: {exc}")
        return 1

    counts = {"completed": 0, "failed": 0, "deferred": 0, "missing": 0}
    for run in runs:
        counts[run.status] += 1
    print(
        f"[report] project={config.project.name} phase={config.project.phase} "
        f"runs={len(runs)} completed={counts['completed']} "
        f"failed={counts['failed']} deferred={counts['deferred']} "
        f"missing={counts['missing']} path={report_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch a CLI invocation to the matching placeholder handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
