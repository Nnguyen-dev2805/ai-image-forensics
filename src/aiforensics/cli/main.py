"""argparse-based CLI for the aiforensics Phase A/B baseline suite."""

from __future__ import annotations

import argparse
import pathlib

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
        description=(
            "Reproducible baseline suite for AI-generated image detection (Phase A/B)."
        ),
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
    print(
        f"[run] placeholder: baseline={args.baseline} project={config.project.name} phase={config.project.phase} config={args.config}"
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    from aiforensics.evaluation.metrics import (
        discover_prediction_files,
        evaluate_prediction_file,
        MetricsError,
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
        f"[evaluate] project={config.project.name} phase={config.project.phase} prediction_files={count} output_root={config.paths.output_root}"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(
        f"[report] placeholder: project={config.project.name} phase={config.project.phase} config={args.config}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch a CLI invocation to the matching placeholder handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
