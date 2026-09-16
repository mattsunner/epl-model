# Contributing

## Setup

```bash
uv sync
uv run pre-commit install
```

`just check` (lint, typecheck, test) is what CI runs; the pre-commit hooks run the same
commands via `uv run`, so there is one toolchain, not two that can drift apart.

## Where documentation lives (story C-12)

Docstrings carry the **contract and any gotcha specific to that function** -- roughly
10 lines. Project history, the reasoning behind a modelling or architectural choice,
and anything that would still be true if the function were rewritten belong elsewhere,
so they do not silently go stale next to code that has since changed:

| Kind of content | Goes in |
| --- | --- |
| A function's inputs, outputs, and a gotcha specific to it | Its own docstring |
| The maths behind a model or metric | `docs/methodology.md` |
| Intended use, limitations, known failure modes | `docs/model-card.md` |
| Why a significant decision was made, options considered, consequences | `docs/adr/` (one file per decision) |
| Source inventory, licensing, per-source gotchas | `docs/data-sources.md` |
| Architecture, layer boundaries, repo layout, milestones, risks | `docs/design.md` |
| A specific gap or improvement, tracked to closure | `stories.md` |

If you are about to write more than a couple of sentences explaining *why* something is
the way it is inside a docstring, it probably belongs in one of the files above instead,
with a one-line pointer left in the docstring.

## Conventions already documented elsewhere

- Notebook boundary: `docs/adr/0003-notebook-boundary.md`, `notebooks/README.md`.
- Data layer zones (`raw_*`/`dim_*`/`stg_*`/`mart_*`) and lineage: `docs/design.md`
  section 5.4, `src/plforecast/storage/curate.py`'s module docstring.
- Model ladder and the RPS gate each rung must clear: `docs/design.md` section 6.2.
- Git commit and PR conventions: none beyond a clear, honest message; this is a solo
  project at present.

## Tests

`uv run pytest`. `tests/unit/` for isolated behaviour, `tests/integration/` for the
full pipeline over a committed fixture dataset (no network), `tests/fixtures/` for
sample data. Live-source tests are marked `@pytest.mark.network` and excluded by
default; CI never has network access.
