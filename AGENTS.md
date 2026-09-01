# AGENTS.md

## Mission

Build Phase A/B as a reproducible AI-image-forensics baseline suite. The goal is a trustworthy research baseline that can run on local machines, Google Colab, or Kaggle, and produce comparable artifacts for every baseline.

## Required Reading

Before implementing or changing behavior, read these files:

- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/schemas/manifest.md`
- `docs/schemas/predictions-jsonl.md`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`

Read `docs/research/literature-review.md` when changing research assumptions, baseline choices, datasets, or evaluation protocol.

## Phase Scope

- Implement only Phase A/B unless the user explicitly expands scope.
- Preserve the monorepo layout, but keep real implementation focused on Phase A/B.
- Phase A/B is not publication-ready training. It is a reproducible baseline suite for deciding whether to continue to later stages.

## Non-Negotiables

- Keep importable code under `src/aiforensics/`.
- Use YAML configs for paths, model choices, dataset slices, cache roots, and output roots.
- Do not hardcode local, Colab, or Kaggle paths.
- Keep notebooks thin. Notebooks install dependencies, mount storage, set config paths, and call the CLI.
- Do not commit datasets, model weights, downloaded repos, caches, generated outputs, or checkpoints.
- Use NPR only as an external pinned baseline. Do not copy NPR source into this repository.
- Every baseline must emit the shared `predictions.jsonl` schema.
- Every run must write inspectable artifacts under `outputs/<run_id>/`.
- If Qwen or NPR cannot run because of GPU, dependency, or environment limits, record `failed` or `deferred` with logs and continue the rest of the pipeline.

## CLI Contract

The intended user-facing CLI is:

```bash
aiforensics prepare --config configs/phase_ab.yaml
aiforensics run --baseline clip_probe --config configs/phase_ab.yaml
aiforensics run --baseline qwen_vl --config configs/phase_ab.yaml
aiforensics run --baseline npr --config configs/phase_ab.yaml
aiforensics run --baseline assisted_qwen --config configs/phase_ab.yaml
aiforensics evaluate --config configs/phase_ab.yaml
aiforensics report --config configs/phase_ab.yaml
```

Keep this contract stable. Internal modules may change, but callers should keep using these commands.

## Clean Code Rules

- Keep modules small and single-purpose.
- Separate orchestration, file I/O, model inference, validation, metrics, caching, and reporting.
- Use Python 3.10-compatible code.
- Use type hints for public functions, dataclasses, and config objects.
- Prefer `pathlib.Path` for filesystem paths.
- Use explicit typed structures for configs, manifests, predictions, metrics, and run metadata.
- Keep side effects at the edges. Validation and metrics should be pure where practical.
- Put model-specific loading, preprocessing, and inference inside baseline adapters.
- Put shared manifest, prediction, metric, cache, and report logic in common modules.
- Avoid hardcoded paths, implicit downloads, import-time downloads, global mutable state, and duplicated schema logic.
- Prefer clear code over clever abstractions. Add an abstraction only when at least two baselines need the same behavior.
- Use `logging` for runtime events and explicit exceptions for invalid inputs.
- Add comments only to explain non-obvious decisions.

## Testing Rules

- Add or update tests with every behavior change.
- Unit-test manifest parsing, manifest validation, prediction schema validation, metrics, cache keys, and run metadata.
- Use tiny synthetic images or fixtures for CPU smoke tests.
- Keep GPU/model-heavy tests optional and clearly marked.
- The smoke config must run without requiring Qwen, NPR, a real dataset, or full model downloads unless the test explicitly opts in.
- Before claiming completion, run the verification gate below.

## Linting and Formatting

- Use `ruff` for Python linting and formatting.
- Before claiming Python code is complete, run `uv run --extra dev ruff check src tests` and `uv run --extra dev ruff format --check src tests`.
- If formatting fails, run `uv run --extra dev ruff format .`, then re-run the verification commands.
- Keep ruff changes scoped to files touched by the current task unless the user asks for a broader cleanup.

## Verification Gate

Before claiming completion, run:

```bash
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
uv run pytest
uv run aiforensics prepare --config configs/phase_ab_smoke.yaml
uv run aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
uv run aiforensics evaluate --config configs/phase_ab_smoke.yaml
uv run aiforensics report --config configs/phase_ab_smoke.yaml
```

If a command cannot run in the current environment, record the exact command, error, and reason in the final response and in the run logs.
