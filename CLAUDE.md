# CLAUDE.md

## Purpose

This file is the Claude Code entrypoint for the AI Image Forensics project. Treat `AGENTS.md` as the main project rulebook and use this file only for Claude-specific workflow guidance.

## Required Reading

Before coding or changing behavior, read and follow:

- `AGENTS.md`
- `docs/architecture/phase-ab-architecture.md`
- `docs/plan/phase-ab-plan.md`
- `docs/schemas/manifest.md`
- `docs/schemas/predictions-jsonl.md`
- `configs/phase_ab.yaml`
- `configs/phase_ab_smoke.yaml`

Read `docs/research/literature-review.md` before changing research assumptions, baseline choices, datasets, or evaluation protocol.

## Claude Code Workflow

- Work task-by-task from `docs/plan/phase-ab-plan.md`.
- For task-specific work, read the matching spec under `docs/specs/` when one exists.
- Keep implementation focused on Phase A/B unless the user explicitly expands scope.
- Write or update tests before implementation when practical.
- Run the task-specific verification command before claiming completion.
- Report the exact commands run and whether they passed.

## Project Rules

- Keep importable code under `src/aiforensics/`.
- Keep notebooks thin; notebooks should call the CLI instead of duplicating pipeline logic.
- Use YAML configs for paths and experiment choices.
- Do not hardcode local, Colab, or Kaggle paths.
- Do not commit datasets, model weights, downloaded repos, caches, outputs, or checkpoints.
- Use NPR only as an external pinned baseline. Do not copy NPR source into this repo.
- If Qwen or NPR cannot run because of GPU or environment limits, write `failed` or `deferred` artifacts with logs and continue the rest of the pipeline.

## Verification Gate

Before claiming Phase A/B work is complete, run:

```bash
pytest
aiforensics prepare --config configs/phase_ab_smoke.yaml
aiforensics run --baseline clip_probe --config configs/phase_ab_smoke.yaml
aiforensics evaluate --config configs/phase_ab_smoke.yaml
aiforensics report --config configs/phase_ab_smoke.yaml
```

If a command cannot run in the current environment, record the command, error, and reason.

