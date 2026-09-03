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
                "Real dataset manifest building is not implemented in Task 3. "
                "No configured manifests were found."
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

    def _setup_run_dir(baseline: str, run_name: str | None = None) -> pathlib.Path:
        run_dir = create_run_dir(config.paths.output_root, baseline, run_name=run_name)
        (run_dir / "logs.txt").touch()
        shutil.copy2(args.config, run_dir / "config.yaml")
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

    from aiforensics.evaluation.metrics import (
        MetricsError,
        discover_prediction_files,
        evaluate_prediction_file,
    )

    files = discover_prediction_files(config.paths.output_root)
    count = len(files)

    for p_file in files:
        try:
            evaluate_prediction_file(p_file)
        except MetricsError as e:
            print(f"Error evaluating {p_file}: {e}")
            return 1

    print(
        f"[evaluate] project={config.project.name} "
        f"phase={config.project.phase} prediction_files={count} "
        f"output_root={config.paths.output_root}"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(
        f"[report] placeholder: project={config.project.name} "
        f"phase={config.project.phase} config={args.config}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch a CLI invocation to the matching placeholder handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
