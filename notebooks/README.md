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

**Exception**: `03-prototypes/03-01-forecast-workbench.ipynb` is a standing multi-section
notebook rather than a single-conclusion one -- a local exploration surface (design.md
section 10.2) for tweaking the forecasting model's levers and visualizing the result,
meant to be revisited repeatedly rather than deleted once it reaches a conclusion. It
still obeys the "consume, never define logic" rule: every chart it renders comes from
`src/plforecast/viz.py`.

`03-prototypes/03-02-clubelo-workbench.ipynb` is a standard prototype, not another
exception: it defines a `ClubEloModel` and a ClubElo data fetcher inline because that
logic is genuinely new and unproven, exactly what this directory is for. It follows
`03-01`'s section structure for direct comparison but stays, moves into
`src/plforecast` once proven out, or gets deleted -- it does not get the standing-
exception status `03-01` has.
