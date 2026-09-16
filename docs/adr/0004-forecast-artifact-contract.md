# ADR 0004: Forecast artifact contract

**Status**: accepted 14 September 2026; `latest` semantics decided 16 September 2026.

## Context

The forecast artifact is the contract between the model and everything downstream: the
published page, the evaluation of published forecasts, and anyone who wants to check the
numbers independently. It is versioned, schema-validated and committed to git
(design.md section 9).

## Decision

**Two documents per published gameweek**, versioned independently:

| File | Contains | Consumer |
| --- | --- | --- |
| `forecast-gwNN.json` | Per-club position distribution, projected points, title / top-four / relegation probabilities | The published page |
| `fixtures-gwNN.json` | Per-fixture 1X2 probabilities, expected goals, top 10 scorelines | Evaluation and transparency |

**Publish what is needed to score the model, not what is needed to reconstruct it.**
The fixtures document does not carry the full scoreline matrix; the provenance block
(git SHA, model config hash, simulation count, seed, raw snapshot hashes, library
versions) is what makes the artifact reproducible.

**`latest` is both, not either** (design.md section 15, open question 1): every run
writes a gameweek-stamped file that git history preserves, and overwrites
`forecast-latest.json` and `fixtures-latest.json` in place. The site fetches the
`latest` file by a fixed URL; the stamped files are the public record and the input to
forecast-evolution charts. A pointer file was rejected as one more thing to keep
consistent for no reader benefit.

**Schemas** live in `src/plforecast/artifacts/schema.py` as pydantic models and are
exported to `artifacts/schema/*-v1.schema.json` on every forecast run for consumers
that do not import the package. Documents are strict (`extra="forbid"`), and the models
validate the invariants that matter: every `position_pmf` sums to one and has one entry
per club; every 1X2 triple sums to one.

**Governance**: breaking changes bump the major version and live alongside the old
version; the schema is locked before the first public forecast. A GitHub Actions job
validating every committed artifact against its declared schema is story A-09's
`artifact-validate.yml`.

## Consequences

- `plforecast forecast` writes four files per run plus the two schema files.
- The `artifacts/` directory is committed. Data under `data/` never is.
- A re-run of the same gameweek replaces its stamped file; the diff is visible in git,
  which is the intended audit trail.
