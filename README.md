# pl-forecast

Calibrated probability distributions over final Premier League table positions,
refreshed after each gameweek and evaluated against the market. See `design.md` for
the full architecture and rationale.

**Status**: early. Ingest (football-data.co.uk, FPL, Understat), club identity
resolution, curation into `stg_matches`/`mart_fixtures`, the first two rungs of the
model ladder (`models/poisson.py`, `models/dixon_coles.py`), the season simulation
engine (`simulate/`), the evaluation harness (`evaluate/`), and a promoted-club prior
(`features/priors.py`) are built and validated end to end against real data.
Publishing doesn't exist yet -- there is no forecast to link to.

Known gap: newly promoted clubs with no top-flight history in the backfill window
(Coventry, currently) can't be rated by either model directly. `features/priors.py`
gives such a club a real, wide-variance prior built from the empirical record of
similar clubs, but wiring that prior into the model layer itself is future work --
today it's constructed and validated standalone, not yet consumed by `PoissonModel` or
`DixonColesModel` (which don't yet have a mechanism to accept a prior; only the
not-yet-built hierarchical model, per design.md section 6.4, is meant to).

## Quickstart

```bash
uv sync
uv run pre-commit install  # hooks run the same ruff/mypy that `just check` runs
just bootstrap             # ingest all three sources, then curate (first run, ~2 min)
just evaluate              # walk-forward backtest against the market baseline
just check                 # lint, typecheck, test
```

`just bootstrap` is `just ingest-footballdata`, `just ingest-fpl`, `just ingest-understat`
and `just curate` in that order. Raw views are (re)created on every database connection,
so there is no separate migrate step; `just migrate` exists only to apply table
migrations explicitly.

## How it performs against the market

Measured. Full results and protocol in `docs/evaluation.md`; headline numbers (mean
RPS, walk-forward, 2015/16-2025/26, 4,066 test matches):

| Model | Mean RPS |
| --- | --- |
| `PoissonModel` (floor) | 0.2063 |
| `DixonColesModel` | 0.2010 |
| Market (Pinnacle closing, Shin de-vig) | 0.1935 |

Dixon-Coles beats the Poisson floor, as the model ladder requires. Neither beats the
market yet -- anticipated, not a surprise (Pinnacle is a sharp, liquid market, and
neither model has a promoted-club prior wired in yet or posterior uncertainty).
