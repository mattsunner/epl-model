# pl-forecast

Calibrated probability distributions over final Premier League table positions,
refreshed after each gameweek and evaluated against the market. See `docs/design.md` for
the full architecture and rationale.

**Status**: early, but end to end. Ingest (football-data.co.uk, FPL, Understat), club
identity resolution, curation into `stg_matches` (with xG and a benchmark price), `stg_odds`, `stg_fixtures` and `mart_team_match`, the first
three rungs of the model ladder (`models/poisson.py`, `models/dixon_coles.py`,
`models/xg_rates.py`), the season
simulation engine (`simulate/`), the evaluation harness (`evaluate/`), a promoted-club
prior (`features/priors.py`) and the forecast artifact (`artifacts/`) are built and run
against real data. The current forecast is `artifacts/2026-27/forecast-latest.json`,
produced by `just forecast`; it is not yet published to a page (ADR 0005 describes how it
will be).

Promoted clubs: a club with less than half a season of decay-weighted top-flight
evidence (Coventry, Hull and Ipswich for 2026/27) is shrunk toward a prior built from
the record of survival-zone clubs (ADR 0006), delivered to the models as
pseudo-observations. Before that landed, the first committed forecast rated Coventry
from four defeats alone; the prior is what stops a four-match sample deciding a
38-match season.

## Quickstart

```bash
uv sync
uv run pre-commit install  # hooks run the same ruff/mypy that `just check` runs
just bootstrap             # ingest all three sources, then curate (first run, ~2 min)
just validate              # data-quality invariants over the curated tables
just evaluate              # walk-forward backtest against the market baseline
just check                 # lint, typecheck, test
```

`just bootstrap` is `just ingest-footballdata`, `just ingest-fpl`, `just ingest-understat`
and `just curate` in that order. Raw views are (re)created on every database connection,
so there is no separate migrate step; `just migrate` exists only to apply table
migrations explicitly. `just curate` also writes `data/curate-manifest.json`: row counts
and raw-snapshot lineage for every curated table.

## How it performs against the market

Measured. Full results and protocol in `docs/evaluation.md` (rendered from
`docs/evaluation/metrics.json` by `just evaluate`); headline numbers: mean RPS,
walk-forward, 2015/16-2025/26, 4,066 matches scored on identical rows for every
model and the benchmark:

| Model | Mean RPS |
| --- | --- |
| `PoissonModel` (floor) | 0.2063 |
| `DixonColesModel` | 0.2010 |
| `XGRateModel` | 0.1980 |
| Market, Shin de-vig | 0.1939 |

The shipped model is whichever non-market model has the best held-out RPS, computed by
the report rather than chosen by hand; today that is `xg-rates`, +0.0041 RPS
from the Shin de-vigged market (design.md's stretch target is within 0.005). The
benchmark is mostly Pinnacle closing, a sharp, liquid market; it is a fallback chain
(ADR 0007) because the data source stopped publishing Pinnacle prices in January 2026.
`docs/evaluation.md` also carries per-outcome calibration, the season-level evaluation
(position distribution against the realised table at frozen cutoffs) and the tuning
grids behind every hyperparameter in `config.py`.

## Data sources and attribution

No third-party data is committed; `data/` is gitignored and `data/MANIFEST.md` says how to
rebuild it. Sources and their terms are in `docs/data-sources.md`:

- football-data.co.uk: match results and closing odds. Free for personal, non-commercial
  use; attribute the site when publishing derived results.
- Fantasy Premier League API: fixtures, kickoff times, team roster, player availability.
- Understat, via the `soccerdata` library: team-level expected goals. Scraped; the
  adapter reuses the cache for completed seasons and fetches only the current season live.

Code is MIT licensed (`LICENSE`). Forecast artifacts under `artifacts/` are original
output.
