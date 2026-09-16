# Notebooks

Conventions (ADR 0003, design.md section 4):

- Notebooks **consume** `plforecast`; they never define logic anything else depends on.
  A function used twice moves into `src/plforecast/` with a test.
- Name `NN-MM-kebab-case-question.ipynb`: `NN` is the stage (01 eda, 02 cleaning,
  03 prototypes), `MM` the sequence within it.
- The first cell states the question and, once known, the conclusion. A notebook with no
  conclusion after a week is deleted.
- Outputs are stripped on commit by the `nbstripout` pre-commit hook. Rendered outputs
  worth keeping are exported to `docs/`.
- Run with the project virtualenv: `uv run jupyter lab`.

There are no notebooks yet.
