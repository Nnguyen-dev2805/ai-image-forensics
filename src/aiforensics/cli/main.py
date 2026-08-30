"""argparse-based CLI for the aiforensics Phase A/B baseline suite."""

from __future__ import annotations

import argparse
import pathlib

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


def _build_prepare_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare",
        help="Validate or build dataset manifests.",
    )
    _add_config_arg(parser)
    parser.set_defaults(handler=_cmd_prepare)


def _build_run_parser(subparsers: argparse._SubParsersAction) -> None:
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


def _build_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "evaluate",
        help="Compute metrics from prediction artifacts.",
    )
    _add_config_arg(parser)
    parser.set_defaults(handler=_cmd_evaluate)


def _build_report_parser(subparsers: argparse._SubParsersAction) -> None:
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
            "Reproducible baseline suite for AI-generated image detection "
            "(Phase A/B)."
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
    print(f"[prepare] placeholder: config={args.config}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    print(
        f"[run] placeholder: baseline={args.baseline} config={args.config}"
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    print(f"[evaluate] placeholder: config={args.config}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    print(f"[report] placeholder: config={args.config}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch a CLI invocation to the matching placeholder handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
