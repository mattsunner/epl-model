# pl-forecast

Calibrated probability distributions over final Premier League table positions,
refreshed after each gameweek and evaluated against the market. See `design.md` for
the full architecture and rationale.

**Status**: early, but end to end. Ingest (football-data.co.uk, FPL, Understat), club
identity resolution, curation into `stg_matches`/`stg_odds`/`mart_fixtures`, the first
two rungs of the model ladder (`models/poisson.py`, `models/dixon_coles.py`), the season
simulation engine (`simulate/`), the evaluation harness (`evaluate/`), a promoted-club
prior (`features/priors.py`) and the forecast artifact (`artifacts/`) are built and run
against real data. The current forecast is `artifacts/2026-27/forecast-latest.json`,
produced by `just forecast`; it is not yet published to a page (ADR 0005 describes how it
will be).

Known gap, visible in the current forecast: a newly promoted club with almost no
top-flight history in the backfill window (Coventry, currently) is rated from a handful
of matches, so its projection is the model's failure mode rather than a prediction to
trust. `features/priors.py` builds a wide-variance prior for exactly this case, but the
two shipped models cannot consume it yet; wiring it in as weighted pseudo-observations is
story C-08 in `stories.md`.

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

Measured. Full results and protocol in `docs/evaluation.md` (rendered from
`docs/evaluation/metrics.json` by `just evaluate`); headline numbers: mean RPS,
walk-forward, 2015/16-2025/26, 4,066 matches scored on identical rows for every
model and the benchmark:

| Model | Mean RPS |
| --- | --- |
| `PoissonModel` (floor) | 0.2063 |
| `DixonColesModel` | 0.2010 |
| Market, Shin de-vig | 0.1939 |

Dixon-Coles beats the Poisson floor, as the model ladder requires. Neither beats the
market yet -- anticipated, not a surprise (the benchmark is mostly Pinnacle closing, a
sharp, liquid market, and neither model has a promoted-club prior wired in yet or
posterior uncertainty). The market benchmark is a fallback chain (ADR 0007) because
the data source stopped publishing Pinnacle closing prices in January 2026.
