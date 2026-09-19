# Design: Premier League Season Forecast

**Owner**: Matthew Sunner
**Last updated**: 16 September 2026
**Target**: first end-to-end forecast published before gameweek 19 (late December 2026)

---

## 1. Purpose

Produce a calibrated probability distribution over final 2026/27 Premier League table positions for all 20 clubs, refreshed after each gameweek, published publicly, and evaluated honestly against a market benchmark.

### 1.1 Goals

- A match-level model that outputs a full scoreline distribution for any fixture.
- A simulation engine that converts remaining fixtures into a distribution over final league position, points, and goal difference.
- An evaluation harness that scores the model with proper scoring rules against de-vigged closing odds, using walk-forward backtests across prior seasons.
- A reproducible pipeline: given a git SHA and a data snapshot, any run reproduces bit-for-bit.
- Documentation good enough that the repo is publishable and can be written about without a second pass.

### 1.2 Non-goals

- Live in-play forecasting. All inference is pre-match.
- Player-level projections or fantasy football optimisation.
- Betting execution or staking. Market odds are a benchmark, not a trading signal.
- Multi-league generalisation in v1. The architecture should not preclude it, but the Premier League is the only supported competition.
- Hosted, always-on inference. The pipeline runs locally on demand.

### 1.3 Success criteria

| Criterion | Target |
| --- | --- |
| Match model beats a naive Poisson baseline on ranked probability score | Required |
| Match model within 0.005 RPS of de-vigged closing odds | Stretch |
| Calibration: predicted vs realised outcome frequency in 10 buckets | Within confidence bands |
| Simulation determinism | Identical output for identical seed and config hash |
| Time from clean clone to reproduced forecast | Under 15 minutes |

---

## 2. Architecture

### 2.1 Two-stage design

The model does not regress on final points or position. It regresses on match outcomes and simulates the season.

```
sources  ->  raw landing  ->  curated tables  ->  features
                                                     |
                                                     v
                                              match model
                                        (scoreline distribution
                                         per remaining fixture)
                                                     |
                                                     v
                                          simulation engine
                                     (N seasons, PL tiebreak rules)
                                                     |
                                                     v
                                         forecast artifact (JSON)
                                            /              \
                                  static site           local dashboard
```

**Rationale** (see `docs/adr/0001-two-stage-architecture.md`): final position is a rank determined jointly across 20 clubs by a fixed fixture graph. Direct regression discards the fixture structure and has roughly 20 observations per season. Simulation preserves the structure and yields honest joint uncertainty, including correlations between clubs that share remaining opponents.

### 2.2 Layer boundaries

Each boundary is a contract with a schema. No layer reaches past its neighbour.

| Layer | Responsibility | Output |
| --- | --- | --- |
| `ingest` | Fetch from one source, no transformation beyond parsing | Immutable timestamped Parquet in `data/raw/` |
| `entities` | Resolve source-specific club names to canonical club IDs | Club dimension table |
| `storage` | Curate raw into typed, deduplicated fact and dimension tables | DuckDB database file |
| `features` | Derive model inputs from curated tables | Feature frames |
| `models` | Fit team strength, emit scoreline distributions | Fitted model object, scoreline matrices |
| `simulate` | Play out remaining fixtures, apply tiebreak rules | Position distribution arrays |
| `evaluate` | Score predictions, backtest, check calibration | Metric tables |
| `artifacts` | Serialise a versioned forecast document | `forecast.json` |

---

## 3. Repository layout

As built on 16 September 2026. Entries marked *planned* do not exist yet; entries marked *dropped* were superseded by a decision recorded in the ADR named.

```
pl-forecast/
├── README.md
├── LICENSE                        # MIT
├── CONTRIBUTING.md                 # setup, where documentation lives (story C-12)
├── pyproject.toml
├── justfile
├── .pre-commit-config.yaml        # local hooks running the uv toolchain (ADR 0003)
├── .gitignore
├── stories.md                     # review backlog with status lines
│
├── docs/
│   ├── design.md                  # this document
│   ├── data-sources.md            # source inventory, licensing, gotchas
│   ├── evaluation.md              # rendered by `plforecast evaluate`; never hand-edited
│   ├── evaluation/                # metrics.json, calibration.csv, tuning-*.json
│   ├── methodology.md             # the maths; docstrings carry only contract + gotchas (story C-12)
│   ├── model-card.md              # intended use, limitations, known failure modes
│   └── adr/                       # 0001-0009, all written
│
├── notebooks/                     # README.md only; conventions in ADR 0003
│
├── src/plforecast/
│   ├── config.py                  # pydantic-settings, single config surface, tuned hyperparameters
│   ├── logging.py                 # structlog, JSON lines on stderr
│   │
│   ├── ingest/
│   │   ├── base.py                # Source protocol, retry, rate limiting, TTL cache, snapshots
│   │   ├── footballdata.py        # results + every closing-odds set (ADR 0007)
│   │   ├── understat.py           # team-level xG via soccerdata; current season always live
│   │   ├── fpl.py                 # fixtures, kickoff times, roster (stable `code`), availability, gameweek calendar
│   │   ├── clubelo.py             # built (story C-16); ratings only, no match model consumes it
│   │   └── transfermarkt.py       # planned, v2 (story C-16)
│   │
│   ├── entities/
│   │   ├── clubs.py               # canonical IDs, strict alias resolution; fuzzy helper for notebooks only
│   │   ├── competitions.py        # club-season membership bridge
│   │   └── club_aliases.yaml      # hand-maintained, reviewed each season
│   │
│   ├── storage/
│   │   ├── db.py                  # DuckDB connection; raw views recreated on connect (ADR 0002)
│   │   ├── migrations/            # table migrations only; none yet
│   │   ├── curate.py              # raw -> dim_club, stg_*, mart_team_match; cross-source checks
│   │   └── validate.py            # data-quality invariants over the live curated tables
│   │
│   ├── features/
│   │   ├── strength.py            # decayed attack/defence rates, goals or xG
│   │   ├── schedule.py            # rest days from FPL kickoff times
│   │   └── priors.py              # promoted-club prior and its pseudo-observations (ADR 0006)
│   │
│   ├── models/
│   │   ├── base.py                # MatchModel protocol, UnknownClubError
│   │   ├── poisson.py             # rung 1, floor
│   │   ├── dixon_coles.py         # rung 2
│   │   ├── xg_rates.py            # rung 2.5 (ADR 0009)
│   │   ├── hierarchical.py        # planned, rung 3
│   │   └── market.py              # dropped: the benchmark is scored by evaluate/market.py, not a MatchModel
│   │
│   ├── simulate/
│   │   ├── engine.py              # vectorised Monte Carlo
│   │   ├── tiebreak.py            # TiebreakRules strategy, PL implementation
│   │   ├── standings.py           # points/GD/GF from played matches
│   │   └── competition.py         # league size, relegation and European spots
│   │
│   ├── evaluate/
│   │   ├── metrics.py             # RPS, log loss, Brier
│   │   ├── backtest.py            # walk-forward splitter with exact coverage accounting
│   │   ├── market.py              # de-vigged benchmark from the fallback chain
│   │   ├── calibration.py         # pooled and per-outcome reliability curves
│   │   ├── season.py              # season-level evaluation at frozen cutoffs
│   │   ├── tuning.py              # grid search with disjoint selection/report seasons
│   │   ├── report.py              # identical-rows comparison; the JSON report
│   │   └── render.py              # docs/evaluation.md from the report
│   │
│   ├── artifacts/
│   │   ├── schema.py              # pydantic forecast and fixtures documents (ADR 0004)
│   │   └── writer.py              # documents from a simulation; provenance; latest + stamped files
│   │
│   └── cli.py                     # typer: ingest, curate, evaluate, tune, forecast, validate, validate-artifacts
│
├── tests/
│   ├── unit/
│   ├── fixtures/                  # small committed sample data, observed name/code sets
│   ├── integration/               # tests/integration/test_pipeline.py: migrate/curate/validate/evaluate over tests/fixtures/integration_data/
│   └── golden/                    # planned; determinism is asserted directly in unit tests instead (story A-17)
│
├── data/                          # gitignored except .gitkeep and MANIFEST.md
│   ├── raw/                       # immutable timestamped snapshots per source
│   ├── cache/                     # TTL HTTP cache; soccerdata cache
│   └── pl.duckdb
│
├── artifacts/                     # committed
│   ├── 2026-27/                   # forecast-gwNN.json, fixtures-gwNN.json, *-latest.json
│   └── schema/                    # forecast-v1.schema.json, fixtures-v1.schema.json
│
└── .github/workflows/
    ├── ci.yml                     # lint, types, tests with coverage floor, artifact validation
    ├── artifact-validate.yml      # on changes under artifacts/
    └── docs.yml                   # planned
```

---

## 4. The notebook boundary

This is the single most important convention in the repo and the one most likely to erode.

### 4.1 Rules

- Notebooks **consume** `plforecast`. They never define logic that anything else depends on.
- Any function that survives a second use moves into `src/plforecast/` with a test, and the notebook is rewritten to import it.
- Notebooks are not run in CI and are not part of the pipeline. `just forecast` never touches a notebook.
- Notebook outputs are stripped before commit via `nbstripout` in pre-commit. Rendered outputs live in `docs/` as exported HTML if they are worth publishing.
- Every notebook opens with a markdown cell stating its question and its conclusion. If a notebook has no conclusion after a week, delete it.
- Naming: `NN-MM-kebab-case-question.ipynb`. The number pair is `stage-sequence`, not a run order.

### 4.2 Why

A public repo where the model lives in notebooks is unreviewable and untestable, and it is the single most common reason data science projects cannot be written about credibly. Enforcing the boundary at pre-commit rather than by discipline is the only version that survives contact with a busy month.

### 4.3 Tooling

- `jupyter` with `ipykernel` pointed at the project virtualenv, so notebooks and the package share one dependency set.
- `nbstripout` in `.pre-commit-config.yaml`.
- Optional: `jupytext` pairing to `.py:percent` if notebook diffs become painful. Defer until it hurts.

---

## 5. Data layer

### 5.1 Sources

Detailed inventory lives in `docs/data-sources.md`. Summary of what each source is responsible for:

| Source | Responsibility | Cadence | Status |
| --- | --- | --- | --- |
| football-data.co.uk | Match results and closing odds, E0 and E1, 1993 onward | Weekly | Built |
| Understat | Team and shot-level xG, 2014/15 onward | Weekly | Built (team-level) |
| FPL API | Fixture list, kickoff times, player availability and suspensions | Weekly | Built |
| ClubElo | Independent strength prior for the promoted-club prior | Weekly | Built and active (story C-16) |
| Transfermarkt | Squad market value, promoted-club prior, injury history | Per transfer window | Deferred to v2 (story C-16) |

FBref is explicitly excluded as a live source. Its Opta-derived advanced stats were removed in January 2026 and no longer update.

**ClubElo built, Transfermarkt deferred (story C-16, decided 16 September 2026, adapter
landed 18 September 2026).** `ingest/clubelo.py` unblocks `build_prior()`'s
`external_rating`/`external_weight` parameters (ADR 0006) -- the one remaining piece of
the *decided* promoted-club prior design (design.md section 6.3, decision 2 in section
14) that had no adapter yet. **Not a single CSV endpoint as originally assumed here**:
that classic API (`api.clubelo.com`) is gone; see `docs/data-sources.md`'s ClubElo
section for what the adapter actually does against the redesigned site, including two
real coverage gaps (history from ~2022/23 only, 32 of 35 clubs resolved). Active since
19 September 2026 (`config.clubelo_prior_weight = 0.5`) on live-forecast evidence, even
though the one dedicated evaluation run behind it
(`notebooks/03-prototypes/03-04-clubelo-prior-workbench.ipynb`) was inconclusive on a
small historical sample -- see `docs/adr/0006-promoted-club-priors.md` for the full
reasoning. Transfermarkt adds a
third club-naming scheme to maintain and is scraped (ToS-sensitive) for a signal
(squad market value) the prior does not strictly need: ClubElo's own rating already
blends recent form and squad quality. Revisit Transfermarkt if ClubElo alone proves
insufficient once there is enough evaluation evidence to trust it.

### 5.2 Ingestion contract

Every adapter implements the same protocol:

```python
class Source(Protocol):
    name: str

    def fetch(self, *, since: date | None = None) -> RawPayload: ...
    def parse(self, payload: RawPayload) -> pl.DataFrame: ...
```

Requirements on every adapter:

- **Immutability**: the parsed frame is written to `data/raw/{source}/{fetched_at}/` and never overwritten. Re-runs create new snapshots.
- **Provenance**: every raw file carries a sidecar `_meta.json` with source URL, HTTP status, fetch timestamp, row count, and a content hash.
- **Politeness**: a configured minimum inter-request delay. Understat and Transfermarkt are scraped, not API-served, and ToS applies.
- **Caching**: repeated fetches within a configurable TTL return the cached snapshot. Development should not hammer third-party sites.
- **Schema validation**: a `pandera` schema per source, enforced at parse time. Fail loudly on shape drift rather than silently producing a broken model.

### 5.3 Club identity

The hardest data problem in the project and the one that will bite silently.

- A canonical club dimension keyed by a stable internal ID, with `football_data_name`, `clubelo_name`, `understat_name`, `fpl_team_id`, `transfermarkt_id` columns.
- League membership lives in a separate club-season bridge table, not as a `league` column on the club. Clubs move between divisions, and the promoted-club priors already require Championship records for clubs that are now Premier League clubs. A column on the dimension would be wrong the moment a club is promoted.
- `club_aliases.yaml` is hand-maintained and reviewed at the start of every season when promoted clubs enter.
- Resolution is strict by default. Unmatched names raise rather than fall back to fuzzy matching, and fuzzy matching is a separate, explicit reconciliation helper used in a notebook, not in the pipeline.
- A test asserts that every club appearing in any curated fact table resolves to exactly one canonical ID.

### 5.4 Storage

DuckDB, single file at `data/pl.duckdb`, gitignored.

Zones:

- `raw_*`: views over the Parquet landing zone. Never mutated.
- `dim_*`: dimensions from hand-maintained sources. `dim_club` from `club_aliases.yaml`.
- `stg_*`: typed, deduplicated, club IDs resolved, one row per natural key, no cross-source derived features beyond the joins that define the row. `stg_matches` (results, xG, benchmark price), `stg_odds` (long, per bookmaker), `stg_fixtures` (current season, season derived from its own kickoffs), `stg_club_season` (membership bridge), `stg_gameweeks` (FPL's own gameweek calendar).
- `mart_*`: the feature grain. `mart_team_match`: one row per club per match with goals, xG, points and league rest days for and against. Further marts are added when a feature needs them.

Rationale for DuckDB over Postgres or plain Parquet: single-file, zero-service, reads Parquet natively, and the analytical query patterns here are exactly what it is built for. A local-only pipeline should not require a running database. See `docs/adr/0002-duckdb-as-analytical-store.md`.

`raw_*` views are recreated on every connection by `storage/db.py` because they embed the absolute path of `data/raw/`; a source with no snapshot yet simply has no view. Curated tables are rebuilt wholesale by `curate`, always from the most recently landed snapshot of each source. Numbered SQL migration files, applied once each at connection time, are reserved for tables whose history matters.

---

## 6. Model layer

Full specification in `docs/methodology.md`. Architectural points only here.

### 6.1 Interface

Every model, including the market baseline, implements:

```python
class MatchModel(Protocol):
    def fit(self, matches: pl.DataFrame) -> Self: ...
    def scoreline_matrix(self, home: ClubId, away: ClubId, max_goals: int = 10) -> np.ndarray: ...
    @property
    def config_hash(self) -> str: ...
```

`scoreline_matrix` returns an `(max_goals+1, max_goals+1)` array of joint probabilities. Everything downstream (1X2 probabilities, over/under, simulation sampling) derives from this single primitive. The market baseline satisfies the interface by fitting a scoreline distribution consistent with de-vigged 1X2 and totals prices, which makes benchmarking a one-line substitution rather than a special case.

### 6.2 Model ladder

Implement in this order. Each must beat the one before it on held-out RPS or it does not ship.

1. `poisson.PoissonModel`: independent Poisson, static team strengths. Floor.
2. `dixon_coles.DixonColesModel`: low-score dependence correction plus exponential time decay. Decay parameter `xi` tuned by backtest, not assumed (`plforecast tune`; the 1997 value turned out to be the interior optimum).
3. `xg_rates.XGRateModel` (rung 2.5, ADR 0009): attack, defence and home advantage fitted on a tuned blend of xG and goals by weighted least squares in log space, Poisson scorelines at the fitted rates with a low-score correction. Ships if it beats Dixon-Coles on identical rows; `docs/evaluation.md` records which model ships, computed from the report.
4. `hierarchical.HierarchicalModel`: Bayesian hierarchical Poisson with partial pooling across clubs. Gives posterior uncertainty on team strength and handles promoted clubs and thin early-season data properly. Not built.
5. The market benchmark: not a model to beat but the benchmark to measure against. Scored by `evaluate/market.py` from the de-vigged closing price (ADR 0007) rather than as a `MatchModel`.

Optional branch, only if 2 and 3 plateau: gradient boosting on engineered features, benchmarked against the Poisson family rather than replacing it.

### 6.3 Key modelling decisions to record as ADRs

- **xG rather than goals** drives strength estimation. Finishing variance is largely noise over a 38-game horizon.
- **Time decay is non-optional**. Squads turn over every window; a model treating a club as one entity across five seasons is wrong.
- **Promoted-club priors**: ClubElo rating at season start blended with squad market value, shrunk toward a recency-weighted promoted-club mean. **No Championship xG.** Understat does not cover the Championship, and deriving a league-strength conversion from goals would mean estimating a factor from roughly three clubs per season. ClubElo already performs continuous cross-league strength conversion and covers lower divisions by club-name lookup, so the conversion is both free and better estimated than a bespoke one. See `docs/adr/0006-promoted-club-priors.md`.
- **The promoted-club prior must carry real variance, not a point estimate.** The empirical record is unstable enough that a tight prior is indefensible. All six promoted clubs were relegated in each of 2023/24 and 2024/25, which had not happened once since 1997/98. Then 2025/26 broke it: Sunderland finished 7th on 54 points, the joint-best finish by a promoted side since Wolves in 2018/19, Leeds finished 14th, and Burnley was the only promoted club to go down. A prior fit on the two preceding seasons would have given Sunderland near-certain relegation. Anchor the prior mean on the longer record: the average points total for the club finishing 18th across the previous 22 completed seasons is 33.8.
- **Promoted clubs are not interchangeable.** For 2026/27 the promoted clubs are Coventry, Ipswich and Hull. Ipswich has 2024/25 top-flight data and keeps it; the prior gate (`needs_prior`) counts evidence with the model's own decay, so Ipswich is shrunk toward the prior only as far as its data has decayed, while Hull's 2016/17 season counts for almost nothing. The prior enters the shipped models as pseudo-observations against the real clubs in the fixture list (ADR 0006, story C-08).
- **Home advantage is time-varying**, not a constant. Fit it as a parameter over recent
  seasons. **Status (story C-14): satisfied by decay for rungs 1-3, not yet an explicit
  per-season term.** `PoissonModel` fits one home-advantage term with no decay (static,
  by rung 1's own definition); `DixonColesModel` and `XGRateModel` both fit one home
  term but over an exponentially time-decayed window, so it already tracks *recent*
  form more than a flat average would -- an approximation of time-varying, not the
  literal per-season parameter this bullet originally asked for. A genuine per-season
  home-advantage term (one coefficient per season rather than one continuously decayed
  scalar) is deferred to the hierarchical model (rung 4), where it fits naturally
  alongside the other partially-pooled parameters. Not treated as a defect: the decayed
  approximation is a real, if softer, version of the same idea, and no backtest to date
  has shown the flat decayed term to be a binding constraint on RPS.
- **European competition congestion is deferred.** Rest-day derivation from FPL `kickoff_time` is built in `features/schedule.py` from v1 because it is nearly free and belongs in the curated tables regardless, but it does not enter the model until the hierarchical baseline is established. The reason is confounding: clubs in European competition are also the strongest clubs, so a naive participation indicator partly re-encodes team strength and will appear predictive for the wrong reason. The genuine residual is a rest-day effect and it is small. Add it as a single additive term afterwards and keep it only if it improves held-out RPS on its own.

### 6.4 Uncertainty propagation

With the hierarchical model, the simulation draws a posterior sample of team strengths per simulated season rather than using point estimates. This is the difference between a forecast that says a club has a 12 percent title chance and one that is honest about not knowing how good that club is. Point-estimate simulation systematically understates tail probabilities.

---

## 7. Simulation engine

### 7.1 Approach

- Vectorised NumPy. Draw all remaining fixtures for all `N` simulated seasons in one pass rather than looping.
- Default `N = 50,000`. Configurable. Convergence check on title, top-four, and relegation probabilities.
- Seeded with an explicit `numpy.random.Generator`. The seed is recorded in the forecast artifact.

### 7.2 Tiebreak rules

Tiebreaks are a `TiebreakRules` strategy object, not hardcoded logic, with `PremierLeagueTiebreaks` as the v1 implementation. The v1 benefit is that the rules become independently testable against known historical cases without standing up a simulation; the v2 benefit is that La Liga (head-to-head first) and the Bundesliga (goal difference, then goals, then head-to-head) are new classes rather than a refactor.

`PremierLeagueTiebreaks` implements the ordering exactly:

1. Points
2. Goal difference
3. Goals scored
4. Head-to-head points between the tied clubs (applies only where title, European qualification, or relegation is at stake)
5. Away goals in those head-to-head matches
6. Playoff at a neutral ground

Steps 4 and 5 were introduced for 2019/20 and make a playoff far less likely than under the previous rules. Most public models stop at step 3, which is wrong and cheap to fix.

### 7.3 Invariants worth testing

Property-based tests using `hypothesis` on synthetic simulated seasons. Every invariant parameterises off `simulate/competition.py` rather than hardcoding league constants, so the assertions stay honest if the competition config ever changes:

- Exactly `n_clubs` clubs, `n_clubs * (n_clubs - 1)` matches, `2 * (n_clubs - 1)` per club. For the Premier League: 20, 380 and 38.
- Total goals scored across the league equals total goals conceded.
- League-wide goal difference sums to zero.
- Total points awarded is between `2 * n_matches` and `3 * n_matches` inclusive. For the Premier League: 760 to 1140.
- Position probabilities for each club sum to 1 across positions, and each position's probabilities sum to 1 across clubs.

That last pair is a doubly-stochastic check on the position matrix and it catches an entire class of indexing bug.

---

## 8. Evaluation

Protocol documented in `docs/evaluation.md` and results regenerated by `just evaluate`.

### 8.1 Metrics

- **Ranked probability score** on 1X2 outcomes. Primary metric. RPS is the accepted standard for football match forecasting because it rewards getting close on an ordered outcome, which accuracy and log loss do not.
- **Log loss** and **Brier score** as secondary.
- **Calibration curves** in 10 probability buckets with binomial confidence bands.
- **Season-level**: at frozen cutoffs (100, 190 and 280 matches played) in every completed season, the realised final position versus the predicted position distribution, scored with a ranked probability score over positions plus log loss of the realised title, top-four and relegation events (`evaluate/season.py`).

### 8.2 Protocol

- Walk-forward by date. Never random k-fold. Fit on everything before gameweek `k`, predict gameweek `k`, advance.
- Backtest window 2015/16 through 2025/26. Earlier seasons lack Understat xG.
- Every model evaluated on identical splits. The splitter is shared code, not reimplemented per model.
- The market baseline is scored on the same splits. If a model does not beat de-vigged closing odds, that is reported plainly in `docs/evaluation.md` rather than omitted.

### 8.3 Market baseline specification

- **Price source: Pinnacle closing** (`PSCH`, `PSCD`, `PSCA` in the football-data.co.uk CSVs) **where published, with a per-match fallback chain** to Betfair Exchange closing and then the site's average closing price (ADR 0007, amended 16 September 2026). Pinnacle was chosen first for three reasons: longer continuous history, which matters if the backtest is ever extended before 2015/16; consistent coverage across all fixtures, where Betfair has early-season liquidity gaps; and it is the convention in the football forecasting literature, so the RPS figures are comparable to published work. The chain exists because the site stopped publishing Pinnacle closing prices in January 2026; the source used is recorded per match and reported in `docs/evaluation.md`.
- **De-vig method: Shin.** Multiplicative normalisation systematically overstates favourites, which biases the benchmark in exactly the probability region where the comparison matters most. `penaltyblog.implied.calculate_implied` supports multiplicative, additive, power, Shin, differential margin weighting, odds ratio and logarithmic.
- **Report both.** `docs/evaluation.md` carries Shin as the headline and multiplicative alongside it, so the methodological choice is visible rather than buried in a config file.

### 8.4 Leakage guards

- A test asserts no feature frame contains data with a timestamp at or after the fixture kickoff it describes.
- Odds used as a benchmark are never used as a model feature in any model being compared against odds.

---

## 9. Forecast artifact

The contract between the model and everything downstream. Versioned, JSON Schema validated, committed to git.

### 9.1 Why commit it

Committing one artifact per gameweek gives, for free:

- A tamper-evident public record of what was forecast and when, which is the thing that makes writing about the project credible.
- A time series of forecast evolution across the season, which is the most interesting visualisation in the project.
- A reproducibility target: re-running the pipeline at a given SHA must reproduce that artifact.

### 9.2 Two documents, not one

Each published gameweek emits two files.

| File | Contains | Consumer |
| --- | --- | --- |
| `forecast-gwNN.json` | Table position distribution, projected points, title, top-four and relegation probabilities | The published page |
| `fixtures-gwNN.json` | Per-fixture 1X2 probabilities, expected goals, top 10 scorelines | Evaluation and transparency |

Rationale:

- **Separate files version independently.** A change to per-fixture output does not force a major bump on the headline document.
- **The site only loads what it renders.** The published page needs the position matrix, not 190 fixture predictions.
- **Size never constrains the headline document.** At gameweek 19 there are 190 remaining fixtures.

**Publish what is needed to score the model, not what is needed to reconstruct it.** The fixtures document stores 1X2 probabilities, home and away expected goals, and the top 10 scorelines by probability. It does not store the full scoreline matrix. The full matrix would be roughly 23,000 floats at gameweek 19, and it is reconstructible from the model given the git SHA, config hash and seed, which is what the provenance block exists for. The trimmed version is enough to compute RPS, log loss and calibration independently, which is the transparency that matters.

### 9.3 Shape

`forecast-gwNN.json`:

```json
{
  "schema_version": "1.0.0",
  "season": "2026-27",
  "as_of_gameweek": 4,
  "generated_at": "2026-09-14T18:00:00Z",
  "provenance": {
    "git_sha": "…",
    "model": "hierarchical",
    "model_config_hash": "…",
    "simulations": 50000,
    "random_seed": 20262027,
    "sources": [
      {"name": "football-data", "fetched_at": "…", "rows": 1140, "content_hash": "…"}
    ]
  },
  "clubs": [
    {
      "club_id": "arsenal",
      "display_name": "Arsenal",
      "current": {"played": 4, "points": 10, "gd": 5},
      "projected": {
        "points": {"mean": 74.2, "p10": 65, "p50": 74, "p90": 83},
        "position": {"mean": 3.1, "p10": 1, "p50": 3, "p90": 6},
        "position_pmf": [0.14, 0.19, 0.18, "…20 entries…"],
        "title": 0.14,
        "top_four": 0.71,
        "relegation": 0.00
      }
    }
  ],
  "benchmark": {
    "source": "clubelo",
    "note": "independent comparison, not an input"
  }
}
```

`position_pmf` is the core payload. Everything the front end renders derives from it.

`fixtures-gwNN.json`:

```json
{
  "schema_version": "1.0.0",
  "season": "2026-27",
  "as_of_gameweek": 4,
  "generated_at": "2026-09-14T18:00:00Z",
  "provenance": { "…": "identical block to the forecast document" },
  "fixtures": [
    {
      "fixture_id": "2026-27-gw05-ars-tot",
      "gameweek": 5,
      "kickoff": "2026-09-20T15:00:00Z",
      "home": "arsenal",
      "away": "tottenham",
      "outcome": {"home": 0.541, "draw": 0.243, "away": 0.216},
      "expected_goals": {"home": 1.82, "away": 1.04},
      "top_scorelines": [
        {"home": 1, "away": 0, "p": 0.118},
        {"home": 2, "away": 1, "p": 0.101}
      ]
    }
  ]
}
```

### 9.4 Governance

- Schemas at `artifacts/schema/forecast-v1.schema.json` and `artifacts/schema/fixtures-v1.schema.json`.
- A GitHub Actions workflow validates every committed artifact against its declared schema on push. This is the one place CI touches the artifacts, and it is cheap insurance against publishing a malformed forecast.
- Breaking schema changes bump the major version and live alongside the old version rather than rewriting history. The two documents version independently.
- Schema is locked before the first published forecast. Changing it after committed artifacts exist means either a migration or a break in the public record, and the public record is the point.

---

## 10. Front end

### 10.1 Recommendation: static page on mattsunner.com

Given a local-only pipeline and a public repo intended to support writing, publish as a static route on the existing Astro site rather than building an interactive app.

**Why this and not Streamlit or Dash:**

- An interactive app needs a running server. The pipeline is local and on-demand, so a hosted app would either sit stale or require the always-on infrastructure explicitly ruled out.
- The Astro, S3, CloudFront, Terraform and GitHub Actions OIDC stack already exists and already has a deployment path. Reusing it is close to zero marginal infrastructure.
- The forecast is a weekly snapshot, not a query surface. Nobody needs to filter it live. Precomputed is the correct shape.
- A static page on an owned domain is a better companion to written pieces than a link to a Streamlit instance that may be asleep.

**Delivery: pull and commit, driven from the site repo.**

- The forecast repo emits `artifacts/2026-27/forecast-latest.json` and commits it. Because the repo is public, GitHub already serves that file at a raw URL.
- A workflow in the **site** repo, triggered on `workflow_dispatch` plus a weekly schedule, fetches the file, compares content hashes, and commits it into `src/data/` if it changed. The existing Astro build fires on that commit.
- Astro imports a local JSON file. No build-time network dependency, so a third-party outage cannot break a site build.

**Rejected alternatives**, recorded in `docs/adr/0005-frontend-static-publishing.md`:

- **S3 plus build-time fetch.** Provisioning a bucket and an IAM role to serve a file GitHub already serves publicly is infrastructure for no gain, and it reintroduces a build-time network dependency.
- **Git submodule.** Pins the site to SHA bumps, complicates CI checkout, and delivers nothing the fetch does not.
- **`repository_dispatch` from the forecast repo.** Would give immediacy, but needs a fine-grained PAT or GitHub App with write access to the site repo. Since the pipeline is run manually, the operator is already at the keyboard and can trigger the site workflow in the same sitting. Not worth the cross-repo credential. Revisit only if the pipeline moves to scheduled execution.

**Side benefit**: the site repo's git history becomes a second, independent record of what was published and when.

**Charts**: rendered as static SVG at build time using Observable Plot or D3 in an Astro component. No client-side charting library, no runtime data fetching, no hydration. The page should be a few kilobytes of HTML and SVG.

**Visualisations, in priority order:**

1. **Position probability matrix**: 20 clubs by 20 positions, cells shaded by probability. This is the single best representation of the model's output and immediately communicates uncertainty in a way a ranked list cannot.
2. **Points distribution ridgeline**: per-club projected points density, ordered by mean.
3. **Forecast evolution**: title, top-four and relegation probabilities per club across gameweeks, drawn from the committed artifact history.
4. **Calibration chart**: from `docs/evaluation.md`. Publishing this is what separates a credible forecast from a confident one.

### 10.2 Local exploration surface

Separate concern, separate tool, not deployed.

- `marimo` or `streamlit` run locally via `just dashboard` for poking at model internals during development.
- It reads the same `forecast.json` plus the DuckDB file. It is a development convenience and is explicitly not the publishing path.
- Keep it in `tools/dashboard/` outside the package so it cannot become a dependency of the model.

### 10.3 Deferred

An interactive app becomes worth revisiting only if the pipeline moves to scheduled execution (homelab K3s CronJob or a GitHub Actions workflow) and there is a real query surface such as fixture-level what-ifs. Recorded as an open question, not a v1 commitment.

---

## 11. Engineering standards

Because the repo is public and intended to be written about, these are requirements rather than aspirations.

### 11.1 Toolchain

| Concern | Choice |
| --- | --- |
| Python | 3.12+ |
| Dependency and env management | `uv` with a committed `uv.lock` |
| Dataframes | Polars for pipeline, pandas only where a library forces it |
| Lint and format | `ruff` (both) |
| Type checking | `mypy` in strict mode on `src/`, relaxed on `tests/` |
| Testing | `pytest`, `hypothesis` for property tests |
| Data validation | `pandera` at ingest and curate boundaries |
| Config | `pydantic-settings`, single `config.py` surface |
| Logging | `structlog`, JSON output, no bare prints |
| Task runner | `justfile` |
| Pre-commit | `ruff`, `mypy`, `nbstripout`, `check-yaml`, `end-of-file-fixer` |

Polars over pandas is a real decision, not fashion: the pipeline is columnar transforms over match tables, the lazy API makes the transform graph explicit, and the strictness catches the schema drift that this project will actually suffer from. `soccerdata` and `penaltyblog` return pandas; convert at the adapter boundary.

### 11.2 CI

`.github/workflows/ci.yml` on every push and PR:

1. `uv sync --frozen`
2. `ruff check` and `ruff format --check`
3. `mypy src/`
4. `pytest` with coverage floor
5. Build docs and check internal links

No network access in CI. Ingest adapters are tested against committed fixture payloads in `tests/fixtures/`, never against live sources. Live-source tests are marked `@pytest.mark.network` and excluded by default.

### 11.3 Testing strategy

| Test type | Target |
| --- | --- |
| Unit | Tiebreak ordering, alias resolution, RPS calculation, de-vigging, scoreline matrix normalisation |
| Property | Simulation invariants from section 7.3 |
| Golden | Fixed seed and fixed input produce a byte-identical artifact |
| Integration | `tests/integration/test_pipeline.py`: migrate, curate, validate, evaluate over a committed two-season, real-club fixture dataset |
| Schema | Every committed artifact validates against its declared schema version |

Tiebreak logic and the position matrix deserve the most test coverage. Both are easy to get subtly wrong and neither will fail loudly.

### 11.4 Documentation

- `README.md`: what it is, one-command quickstart, current forecast link, honest statement of how it performs against market odds.
- `docs/methodology.md`: the maths, written to be readable by someone who has not read the code. Built.
- `docs/model-card.md`: intended use, limitations, known failure modes, the promoted-club weakness, what the model does not account for. Built.
- `docs/adr/`: one file per significant decision, with context, options considered, decision, and consequences. These are the raw material for anything written later, and writing them at decision time costs minutes while reconstructing them costs hours.
- Docstrings on every public function, but scoped narrowly (story C-12): contract plus
  any gotcha specific to that function, roughly 10 lines. Project history, the reasoning
  behind a modelling choice, and anything that would still be true if the function were
  rewritten belong in this file, the ADRs, `docs/methodology.md`, or `docs/model-card.md`
  instead, where they will not silently go stale next to code that has since changed.
  `mypy` strict means the signatures already carry most of the contract, so a docstring
  rarely needs to restate parameter types in prose.

### 11.5 Licensing and data ethics

Non-trivial for a public repo.

- Code under MIT or Apache 2.0.
- **No third-party data is committed.** `data/` is gitignored. `data/MANIFEST.md` documents what should be there and how to fetch it.
- `docs/data-sources.md` records each source's terms and attribution requirements. StatsBomb open data, if used at all, requires registration and explicit attribution.
- Ingest adapters rate-limit by default. The repo should not make it trivially easy for someone to hammer Understat.
- Forecast artifacts are original output and are committed freely.

---

## 12. Milestones

Roughly 15 gameweeks to the December midpoint.

| Phase | Deliverable | Exit criterion |
| --- | --- | --- |
| 1. Skeleton | Repo, toolchain, CI, package layout, DuckDB schema | `just test` passes on an empty pipeline |
| 2. Ingest | football-data.co.uk, Understat, FPL, ClubElo, club dimension, curated marts (Transfermarkt deferred to v2, story C-16) | Every club resolves; marts populated 2015/16 to date |
| 3. Baseline end to end | Poisson model plus simulation plus artifact plus published page | A forecast is live, however crude |
| 4. Evaluation | RPS, calibration, walk-forward backtest, market baseline | Published baseline numbers in `docs/evaluation.md` |
| 5. Dixon-Coles | Time decay, tuned `xi` | Beats Poisson on held-out RPS |
| 6. Hierarchical | Bayesian model, posterior propagation into simulation | Beats Dixon-Coles, or documented as not beating it |
| 7. Gameweek 19 refit | Refit on current-season data, compare to preseason forecast | Both forecasts archived and compared publicly |

Phase 3 before phase 4 and 5 is deliberate. A working end-to-end pipeline with a bad model is worth more than a good model with no plumbing, and it front-loads all the integration pain.

---

## 13. Risks

Kept live (story C-17): updated as risks materialise or are closed out, not written once
and left static. "Status" reflects what is actually known as of 16 September 2026, not
what was anticipated when the row was first written.

| Risk | Impact | Mitigation | Status |
| --- | --- | --- | --- |
| Understat scraper breaks mid-season | No current xG, model degrades to goals | Snapshot weekly and commit nothing but keep local history; goals-based fallback path in `features/strength.py` | Open, not yet materialised |
| Club alias drift on promoted clubs | Silent wrong joins | Strict resolution that raises; test asserting single resolution | Mitigated; also caught a related bug (FPL's per-season `id` vs stable `code`, story B-03) before it shipped |
| Overfitting the decay parameter to backtest | Flattering in-sample, poor live | Tune on a holdout period disjoint from the reporting period | Mitigated: `plforecast tune`, disjoint selection/report seasons (story C-03) |
| Promoted-club prior fit to a short recent window | Confidently wrong on a club like Sunderland in 2025/26 | Prior mean anchored on 22 seasons, wide variance, recency weighting capped; documented in `model-card.md` | Partly mitigated: the prior itself is built and wired in (story C-08); recency-weighting half-life is still an open question (section 15) |
| Model does not beat closing odds | Disappointment, or worse, quiet omission | Stated as a likely outcome up front in the README and model card | Materialised as anticipated: the shipped model (`xg-rates`) is 0.0041 RPS behind the market, inside the 0.005 stretch target but not beating it; stated in `docs/evaluation.md` |
| Scope creep into player-level modelling | Nothing ships by December | Non-goals section is binding | Open, not yet materialised |
| Notebook logic leaking into the pipeline | Repo becomes unreviewable | Pre-commit enforcement, not discipline | Mitigated; no notebooks exist yet (ADR 0003) |
| **Benchmark price source stops publishing mid-window** (materialised, not anticipated) | Market comparison has no price for the live season | football-data.co.uk stopped publishing Pinnacle closing prices after 8 January 2026 and dropped the columns from the 2026/27 file entirely. Fixed with a fallback chain (Pinnacle, then Betfair Exchange, then the site's average closing price), source recorded per match (ADR 0007) | Materialised and mitigated |
| **FPL's per-season team `id` used as a stable key** (materialised, not anticipated) | Silent wrong club resolution across a promotion/relegation boundary, since the same `id` value means a different club next season | `club_aliases.yaml` now keys on `teams[].code`, FPL's actually-stable identifier; fixtures map through the current snapshot's roster (story B-03) | Materialised and mitigated |

---

## 14. Decisions resolved

Closed 14 September 2026. Each has a corresponding ADR.

| # | Question | Decision | Section |
| --- | --- | --- | --- |
| 1 | Artifact delivery to the site | Pull and commit, driven from the site repo. S3, submodule and `repository_dispatch` rejected. | 10.1 |
| 2 | Promoted-club priors | ClubElo plus market value, shrunk to a recency-weighted mean anchored on 22 seasons. No Championship xG. | 6.3 |
| 3 | Market baseline | Pinnacle closing, de-vigged with Shin. Betfair Exchange as secondary check. | 8.3 |
| 4 | Per-fixture predictions | Yes, as a separate document. Score-level detail only, no full scoreline matrix. | 9.2 |
| 5 | European congestion feature | Deferred. Rest-day data captured in v1, not modelled until after the hierarchical baseline. | 6.3 |
| 6 | Multi-league generalisation | Three cheap structural choices now (tiebreak strategy, club-season bridge, competition config). Nothing else. | 5.3, 7.2, 7.3 |
| 7 | ClubElo and Transfermarkt scope | ClubElo is v1 (one CSV endpoint, unblocks the promoted-club prior's external-rating path). Transfermarkt deferred to v2. | 5.1 |

## 15. Remaining open questions

Questions 1, 3 and 4 were closed on 16 September 2026 (ADR 0004 and ADR 0005).

1. ~~Does `latest.json` point at the most recent gameweek or get overwritten in place?~~ **Both**: every run writes `forecast-gwNN.json` (git history is the record) and overwrites `forecast-latest.json` in place (the site's fixed URL). ADR 0004.
2. What is the recency-weighting half-life on the promoted-club prior mean? Needs a backtest across promoted cohorts, and the 2025/26 cohort is the most informative single test case. Depends on the season-level evaluation harness (stories C-07, C-08).
3. ~~Does the published page show the model's record against Pinnacle closing on the front page or only in `docs/evaluation.md`?~~ **Front page.** ADR 0005.
4. ~~Is `fixtures-latest.json` published from the first forecast or held back until the evaluation harness exists?~~ **Published from the first forecast**; the harness exists. ADR 0005.
