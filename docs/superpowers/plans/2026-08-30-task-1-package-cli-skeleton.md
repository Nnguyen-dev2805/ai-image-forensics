# Task 1: Package and CLI Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the minimal Python package skeleton for `aiforensics` with a working CLI entrypoint that recognizes the four Phase A/B subcommands (`prepare`, `run`, `evaluate`, `report`) and the four baseline names (`clip_probe`, `qwen_vl`, `npr`, `assisted_qwen`), with placeholders only — no real model, dataset, manifest, cache, or output logic.

**Architecture:** Standard setuptools-based `src/` layout package with a single `argparse` parser in `src/aiforensics/cli/main.py`. The CLI module exposes `build_parser()` (testable) and `main(argv)` (returns int). Placeholder handlers print the command/baseline/config path to stdout and return `0`. Tests invoke `main()` directly and assert on `capsys`-captured stdout and return code.

**Tech Stack:** Python 3.10 (per `pyproject.toml` `requires-python = ">=3.10,<3.11"`), setuptools ≥ 68, argparse (stdlib), pytest ≥ 8.0.0.

**Spec:** `docs/specs/task-1-package-cli-skeleton-spec.md`

## Global Constraints

- Python 3.10-compatible code (per spec, `pyproject.toml` pins `>=3.10,<3.11`).
- Use `argparse` only. No Typer, Click, or Hydra (per spec).
- No real model/dataset/manifest/cache/output logic in this task.
- All code goes under `src/aiforensics/`; tests under `tests/`.
- `main(argv)` returns `int` and must not call `sys.exit()` internally; the module-level `__main__` guard wraps it in `SystemExit`.
- `build_parser()` must be exposed for tests and reuse.
- Baseline names are exactly: `clip_probe`, `qwen_vl`, `npr`, `assisted_qwen`.
- Subcommands are exactly: `prepare`, `run`, `evaluate`, `report`.
- `--config` is required for every subcommand; `--baseline` is required for `run` only.
- Placeholder output must include the command name and the config path; `run` placeholder must additionally include the baseline name.
- `pyproject.toml` must include `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `pythonpath = ["src"]`.
- `pyproject.toml` must register console script `aiforensics = "aiforensics.cli.main:main"`.
- `requirements.txt` must pin: `pyyaml>=6.0.1`, `pydantic>=2.8.0`, `numpy>=1.26.0`, `pandas>=2.2.0`, `scikit-learn>=1.4.0`, `pillow>=10.0.0`, `pytest>=8.0.0`, `tqdm>=4.66.0`.
- Environment note: the host currently has Python 3.14, not 3.10. Use a 3.10 interpreter (e.g. `python3.10` or a `.venv` created from a 3.10 binary) when running the verification gate. If no 3.10 is available, record the exact command, interpreter path, and error in the final response and continue with `python3` only after confirming `argparse`/stdlib code is portable.

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, setuptools build, console script entrypoint, pytest config. |
| `requirements.txt` | Runtime dependencies for later tasks (empty in this task's logic but pinned per spec). |
| `src/aiforensics/__init__.py` | Package marker; exposes `__version__`. |
| `src/aiforensics/cli/__init__.py` | Subpackage marker for CLI. |
| `src/aiforensics/cli/main.py` | `argparse` parser, `build_parser()`, `main(argv)`, `__main__` guard, placeholder handlers. |
| `tests/test_cli_smoke.py` | Six pytest cases covering exit codes, placeholder text, and error paths. |
| `tests/__init__.py` | Empty marker so pytest treats `tests/` as a package (optional; spec does not require it, so omitted unless tests fail to import). |

The placeholder handlers live inside `main.py` as small private functions (`_cmd_prepare`, `_cmd_run`, `_cmd_evaluate`, `_cmd_report`) that print and return `0`. They are not yet separated into modules — Task 1 keeps a single file to avoid premature decomposition.

---

## Task 1.1: Package metadata and requirements

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`

**Interfaces:**
- Produces: editable-installable package named `aiforensics` with version `0.1.0`.
- Produces: console script `aiforensics` pointing to `aiforensics.cli.main:main`.

- [ ] **Step 1: Create `pyproject.toml`**

Write `pyproject.toml` with this exact content:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aiforensics"
version = "0.1.0"
requires-python = ">=3.10,<3.11"
dependencies = []

[project.scripts]
aiforensics = "aiforensics.cli.main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create `requirements.txt`**

Write `requirements.txt` with this exact content (one pin per line, no comments):

```text
pyyaml>=6.0.1
pydantic>=2.8.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
pillow>=10.0.0
pytest>=8.0.0
tqdm>=4.66.0
```

- [ ] **Step 3: Verify files exist and parse**

Run:

```bash
ls -la pyproject.toml requirements.txt
python3 -c "import tomllib; tomllib.loads(open('pyproject.toml').read()); print('pyproject.toml: ok')"
```

Expected: both files exist, and the tomllib check prints `pyproject.toml: ok`. If `tomllib` is missing (Python < 3.11), use `python3 -c "import tomli; tomli.loads(open('pyproject.toml').read())"` instead.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml requirements.txt
git commit -m "chore: add package metadata and runtime requirements pin list"
```

---

## Task 1.2: Package marker modules

**Files:**
- Create: `src/aiforensics/__init__.py`
- Create: `src/aiforensics/cli/__init__.py`

**Interfaces:**
- Produces: importable package `aiforensics` with attribute `__version__ == "0.1.0"`.

- [ ] **Step 1: Create `src/aiforensics/__init__.py`**

Write the file with this content:

```python
"""AI image forensics Phase A/B baseline suite."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Create `src/aiforensics/cli/__init__.py`**

Write the file with this content:

```python
"""Command-line interface for the aiforensics package."""
```

- [ ] **Step 3: Verify imports**

Run:

```bash
cd /Users/tnhatnguyendev2805/Documents/Projects/ai-image-forensics
PYTHONPATH=src python3 -c "import aiforensics; print(aiforensics.__version__)"
```

Expected output: `0.1.0`.

- [ ] **Step 4: Commit**

```bash
git add src/aiforensics/__init__.py src/aiforensics/cli/__init__.py
git commit -m "chore: add aiforensics package and cli subpackage markers"
```

---

## Task 1.3: Write failing CLI smoke tests

**Files:**
- Create: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `aiforensics.cli.main.build_parser`, `aiforensics.cli.main.main`.
- Produces: six pytest cases (placeholders `0.1.0` exit 0; `0.1.0` SystemExit on bad baseline; `--help` exits 0).

- [ ] **Step 1: Create `tests/test_cli_smoke.py`**

Write the file with this content:

```python
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
```

- [ ] **Step 2: Run tests and confirm they fail (no implementation yet)**

Run:

```bash
cd /Users/tnhatnguyendev2805/Documents/Projects/ai-image-forensics
PYTHONPATH=src python3 -m pytest tests/test_cli_smoke.py -v
```

Expected: every test errors with `ModuleNotFoundError: No module named 'aiforensics.cli.main'` (or `ImportError` for missing names). The point of this step is to confirm tests are wired correctly before implementation.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_cli_smoke.py
git commit -m "test: add CLI smoke tests for task 1 (expected to fail until impl lands)"
```

---

## Task 1.4: Implement CLI skeleton

**Files:**
- Create: `src/aiforensics/cli/main.py`

**Interfaces:**
- Produces: `build_parser() -> argparse.ArgumentParser` with subparsers for `prepare`, `run`, `evaluate`, `report`; `run` requires `--baseline` constrained to `{clip_probe, qwen_vl, npr, assisted_qwen}`.
- Produces: `main(argv: list[str] | None = None) -> int` dispatching to placeholder handlers and returning `0` on success.
- Produces: module-level guard calling `raise SystemExit(main())`.

- [ ] **Step 1: Create `src/aiforensics/cli/main.py`**

Write the file with this content:

```python
"""argparse-based CLI for the aiforensics Phase A/B baseline suite."""

from __future__ import annotations

import argparse
from typing import Sequence

__all__ = ["build_parser", "main"]

SUPPORTED_BASELINES: tuple[str, ...] = (
    "clip_probe",
    "qwen_vl",
    "npr",
    "assisted_qwen",
)

SUPPORTED_COMMANDS: tuple[str, ...] = (
    "prepare",
    "run",
    "evaluate",
    "report",
)


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
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


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a CLI invocation to the matching placeholder handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("no command handler registered")
        return 2
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run smoke tests and confirm they pass**

Run:

```bash
cd /Users/tnhatnguyendev2805/Documents/Projects/ai-image-forensics
PYTHONPATH=src python3 -m pytest tests/test_cli_smoke.py -v
```

Expected: all six tests pass.

- [ ] **Step 3: Run the manual `python -m` verification**

Run:

```bash
cd /Users/tnhatnguyendev2805/Documents/Projects/ai-image-forensics
PYTHONPATH=src python3 -m aiforensics.cli.main --help
PYTHONPATH=src python3 -m aiforensics.cli.main run --baseline clip_probe --config configs/phase_ab_smoke.yaml
```

Expected: `--help` prints a usage block and exits `0`. The `run` command prints a line containing `clip_probe` and the config path, then exits `0`.

- [ ] **Step 4: Commit the implementation**

```bash
git add src/aiforensics/cli/main.py
git commit -m "feat(cli): add argparse skeleton with prepare/run/evaluate/report subcommands"
```

---

## Task 1.5: Editable install and final verification

**Files:**
- (none — verification only)

- [ ] **Step 1: Editable-install the package**

Run (using whichever Python 3.10 interpreter is on PATH; on the current host, no 3.10 binary was found, so record this in the final response and continue with `python3` only if `setuptools` accepts it — the package itself has no Python-version-specific imports):

```bash
cd /Users/tnhatnguyendev2805/Documents/Projects/ai-image-forensics
python3 -m pip install -e .
```

Expected: install completes with `Successfully installed aiforensics-0.1.0`. If the host refuses because `requires-python` is `>=3.10,<3.11` and the interpreter is 3.14, stop here and record:

```text
Editable install skipped.
Reason: host Python is <version>; project requires >=3.10,<3.11.
Mitigation: invoke the CLI via `PYTHONPATH=src python3 -m aiforensics.cli.main ...`
until a 3.10 interpreter is available.
```

Then run the alternative verification below.

- [ ] **Step 2: Verify `aiforensics --help` works (only if Step 1 succeeded)**

```bash
aiforensics --help
aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
```

Expected: `--help` exits `0`; the `run` command prints the placeholder and exits `0`.

- [ ] **Step 3: Run the full pytest suite**

```bash
cd /Users/tnhatnguyendev2805/Documents/Projects/ai-image-forensics
PYTHONPATH=src python3 -m pytest -v
```

Expected: all six tests in `tests/test_cli_smoke.py` pass; no other tests exist yet.

- [ ] **Step 4: Final commit (only if Step 1 succeeded)**

```bash
git add -A
git commit --allow-empty -m "chore: task 1 verification gate (editable install + pytest)"
```

If Step 1 was skipped, do not create this commit; instead, record the skipped step in the final response per the verification-before-completion rule.

---

## Self-Review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| `pyproject.toml` content (build, project, scripts, pytest config) | 1.1 |
| `requirements.txt` with exact pins | 1.1 |
| `src/aiforensics/__init__.py` and `src/aiforensics/cli/__init__.py` | 1.2 |
| `src/aiforensics/cli/main.py` with `build_parser` and `main` | 1.4 |
| `tests/test_cli_smoke.py` with the six required cases | 1.3, 1.4 |
| `argparse`-only | 1.4 |
| Baseline names `{clip_probe, qwen_vl, npr, assisted_qwen}` enforced via `choices` | 1.4 |
| Placeholder output contains command + config path; `run` includes baseline | 1.4 |
| `main()` returns int, no `sys.exit()` inside | 1.4 |
| `__main__` guard uses `raise SystemExit(main())` | 1.4 |
| Verification gate commands from spec | 1.4 step 3 + 1.5 |
| Done criteria (files exist, `--help` works, tests pass, no real logic) | 1.5 |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later" anywhere.
- All test code is shown verbatim.
- All implementation code is shown verbatim.
- No "similar to Task N" hand-waves.

**3. Type consistency:**

- `main(argv: Sequence[str] | None = None) -> int` used consistently.
- `_cmd_*` handlers all take `argparse.Namespace` and return `int`.
- `args.handler` is set in every subparser via `parser.set_defaults`.
- `argparse._SubParsersAction` is private but stable across Python 3.10+; acceptable for an internal helper signature.

No issues.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-30-task-1-package-cli-skeleton.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
