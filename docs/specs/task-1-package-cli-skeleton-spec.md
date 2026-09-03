# Task 1 Spec: Package and CLI Skeleton

## Goal

Create the minimal Python package skeleton for `aiforensics` and expose a working CLI entrypoint. This task must not implement real dataset, model, evaluation, cache, or report logic yet. It only creates the stable command surface that later Phase A/B tasks will fill in.

## Required Reading

Before coding, read:

- `AGENTS.md`
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `configs/phase_ab_smoke.yaml`

## Files To Create

```text
pyproject.toml
requirements.txt
src/aiforensics/__init__.py
src/aiforensics/cli/__init__.py
src/aiforensics/cli/main.py
tests/test_cli_smoke.py
```

## CLI Contract

Implement one CLI command:

```bash
aiforensics
```

It must support these subcommands:

```bash
aiforensics prepare --config configs/phase_ab_smoke.yaml
aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
aiforensics run --baseline qwen_vl --config configs/phase_ab_smoke.yaml
aiforensics run --baseline npr --config configs/phase_ab_smoke.yaml
aiforensics run --baseline assisted_qwen --config configs/phase_ab_smoke.yaml
aiforensics evaluate --config configs/phase_ab_smoke.yaml
aiforensics report --config configs/phase_ab_smoke.yaml
```

For Task 1, each command may print a placeholder message. The important behavior is that valid commands parse successfully and return exit code `0`.

## Expected Behavior

- `prepare` accepts `--config`.
- `run` accepts `--baseline` and `--config`.
- `evaluate` accepts `--config`.
- `report` accepts `--config`.
- Unknown commands fail through normal `argparse` behavior.
- Unknown baselines fail with a clear parser error.
- Supported baseline names are `clip_probe`, `qwen_vl`, `npr`, and `assisted_qwen`.

## Implementation Requirements

Use `argparse`. Do not introduce Typer, Click, Hydra, or a larger CLI framework in Task 1.

Expose these functions from `src/aiforensics/cli/main.py`:

```python
def build_parser() -> argparse.ArgumentParser:
    ...

def main(argv: list[str] | None = None) -> int:
    ...
```

`main()` should return integer exit codes. It should not call `sys.exit()` internally. The module-level entrypoint may use:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Placeholder messages should include the command name and config path. The `run` placeholder should also include the baseline name.

## Package Metadata

Create `pyproject.toml` with setuptools and a console script:

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

Create `requirements.txt`:

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

## Tests

Create `tests/test_cli_smoke.py`.

Minimum tests:

1. `prepare --config configs/phase_ab_smoke.yaml` returns `0` and prints `prepare`.
2. `run --baseline clip_probe --config configs/phase_ab_smoke.yaml` returns `0` and prints `clip_probe`.
3. `evaluate --config configs/phase_ab_smoke.yaml` returns `0` and prints `evaluate`.
4. `report --config configs/phase_ab_smoke.yaml` returns `0` and prints `report`.
5. `run --baseline invalid --config configs/phase_ab_smoke.yaml` raises `SystemExit`.
6. `python -m aiforensics.cli.main --help` exits successfully.

Use `capsys` for tests that inspect printed placeholder output.

## Verification

Run:

```bash
pytest tests/test_cli_smoke.py -v
python -m aiforensics.cli.main --help
python -m aiforensics.cli.main run --baseline clip_probe --config configs/phase_ab_smoke.yaml
```

After editable install, also verify:

```bash
python -m pip install -e .
aiforensics --help
aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
```

## Done Criteria

Task 1 is complete when:

- all required files exist,
- `python -m aiforensics.cli.main --help` works,
- `aiforensics --help` works after editable install,
- all Task 1 tests pass,
- no real model, dataset, manifest, cache, or output logic is implemented.

