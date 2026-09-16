# ADR 0003: The notebook boundary

**Status**: accepted 14 September 2026.

## Context

A public repository where the model lives in notebooks is unreviewable and untestable,
and is the most common reason data science projects cannot be written about credibly.
The boundary erodes under time pressure unless enforced by tooling.

## Decision

- Notebooks **consume** `plforecast`. They never define logic anything else depends on.
- Any function that survives a second use moves into `src/plforecast/` with a test, and
  the notebook is rewritten to import it.
- Notebooks are not run in CI and are not part of the pipeline.
- Outputs are stripped before commit by `nbstripout` in pre-commit.
- Every notebook opens with a Markdown cell stating its question and its conclusion. A
  notebook with no conclusion after a week is deleted.
- Naming: `notebooks/NN-MM-kebab-case-question.ipynb`, where the pair is
  stage-sequence, not a run order.

Enforcement is at pre-commit, not by discipline. The hooks run the project's own
toolchain via `uv run` so they cannot drift from `just check` (story A-03).

## Consequences

- There are currently no notebooks; the one empty notebook found at the repository
  root was removed (story A-19). `notebooks/README.md` restates these conventions when
  the first notebook is added.
- The fuzzy club-name helper `entities/clubs.py:suggest_matches` exists only for a
  reconciliation notebook and is never called by the pipeline.
