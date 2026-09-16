# Review stories: pl-forecast

Review date: 16 September 2026. Status lines under a story record what has since been done. Reviewed against the working tree (no commits exist yet),
the live `data/pl.duckdb`, and the raw snapshots under `data/raw/`. Every "verified" claim
below was checked by running the code or querying the database during the review, not
inferred from documentation.

## How to read this file

Each story has an ID, a priority, and a type.

| Priority | Meaning |
| --- | --- |
| P0 | Blocks a clean clone, a second run, or the credibility of a published number. Do first. |
| P1 | Needed before the first published forecast (design.md target: gameweek 19, late December 2026). |
| P2 | Needed for the repo to be publishable and written about without a second pass. |
| P3 | Worth doing; no deadline. |

| Type | Meaning |
| --- | --- |
| Correction | Something is wrong, stale, or contradicts design.md. |
| Recommendation | Something is missing or could be materially better. |
| Confirmation | An existing design decision the code embodies well. Keep it; the story records why. |

Story format is "as / I want / so that", followed by evidence and acceptance criteria.
File references use `path:line` against the tree as reviewed.

## Verified state at review time

| Check | Result |
| --- | --- |
| `uv run pytest` | 85 passed, 5 deprecation warnings from penaltyblog (NumPy 2.5 `.shape` assignment) |
| `uv run mypy src/` | clean, strict mode, 32 files |
| `uv run ruff check .` | clean |
| `uv run ruff format --check .` | fails: `design.md` fenced Python blocks would be reformatted |
| pre-commit hooks | not installed (`.git/hooks` has only samples) |
| git | zero commits, branch `master`, no remote |
| `stg_matches` | 4,220 rows, 12 seasons (2015/16 through 40 matches of 2026/27) |
| Pinnacle closing odds | present 2015/16 through 8 January 2026; null for the last 170 matches of 2025/26; columns absent from the 2026/27 file |
| `mart_fixtures` | 380 rows, 40 finished, all with kickoff and gameweek, `kickoff_time` is `TIMESTAMP WITH TIME ZONE` |
| `raw_understat_team_match` | 4,220 rows, no nulls in xG or PPDA, goals agree with football-data on every row that joins |
| Understat club names | 6 of 35 differ from football-data spelling; `understat_name` is null for every club in `club_aliases.yaml` |
| FPL vs football-data, 2026/27 | 40 finished fixtures on both sides, 0 scoreline disagreements, 0 date disagreements |
| One Dixon-Coles fit on 4,180 matches | 0.04 s; the walk-forward backtest has 1,252 refits per model |

---

# Part A: Solutions architect lens

Criteria: accuracy, conciseness, and clarity of architecture, documentation, and
engineering execution.

## A.1 Repository and delivery

### A-01 · P0 · Correction · Put the repository under version control

**Status**: Done 16 Sep 2026: initial commit on `main`.

**As** the maintainer, **I want** an initial commit on `main` with a remote, **so that** the
"tamper-evident public record" the design depends on (design.md section 9.1) can exist at
all, and so that the git SHA the artifact provenance block requires has a value.

Evidence: `git log` reports no commits; the branch is `master` while the tooling expects
`main`; `config.py:20` advertises a User-Agent pointing at a GitHub repo that has not been
pushed.

Acceptance criteria:
- Branch renamed to `main`; first commit contains the current tree.
- `git status` after commit shows nothing under `data/` except `.gitkeep` and `MANIFEST.md`.
- Remote configured; CI (A-09) runs on the first push.

### A-02 · P0 · Correction · The README quickstart fails on a clean clone

**Status**: Done 16 Sep 2026: raw views recreated on connect; `just bootstrap`.

**As** a new contributor, **I want** the documented command order to work from an empty
`data/` directory, **so that** the "under 15 minutes from clean clone" success criterion
(design.md section 1.3) is testable.

Evidence: README runs `just migrate` before any ingest. DuckDB binds a view at creation
time and raises `IO Error: No files found that match the pattern` for a `read_parquet`
glob with no files (verified on a fresh database). The migration log in the live database
confirms every migration was applied only after its source had been ingested, which is the
order the README does not describe.

Acceptance criteria:
- Either `connect()` (re)creates the `raw_*` views on every connection, tolerating missing
  directories (recommended, see A-21), or the README order becomes ingest, then migrate,
  then curate.
- A `just bootstrap` recipe runs the full sequence on an empty `data/`.
- A test creates a temp `data_dir`, runs migrate with no snapshots, and asserts no error.

### A-03 · P0 · Correction · `just check` fails today and pre-commit is not enforcing anything

**Status**: Done 16 Sep 2026: local hooks, design.md formatted, hooks installed.

**As** the maintainer, **I want** one toolchain that lint, pre-commit, and CI all share,
**so that** "enforced at pre-commit rather than by discipline" (design.md section 4.2) is
true.

Evidence: `ruff format --check .` flags `design.md` (ruff 0.16 formats fenced Python
blocks in Markdown). `.pre-commit-config.yaml` pins ruff v0.8.4 and mypy v1.13.0; `uv.lock`
resolves ruff 0.16.7 and mypy 2.3.1, so the hooks would run different tool versions with a
different dependency set than `just check`. Hooks are not installed.

Acceptance criteria:
- Decide: either accept the formatter's edits to the two code blocks in `design.md`
  (ruff's diff is trivial) or exclude `*.md` from `ruff format`. Recommended: accept them.
- Replace the remote ruff and mypy hooks with `repo: local` hooks that run `uv run ruff`
  and `uv run mypy src/`, so pre-commit and `just check` are the same command.
- `pre-commit install` documented in README; `just check` passes end to end.

### A-04 · P1 · Correction · One project has three names

**Status**: Done 16 Sep 2026: distribution `pl-forecast`, import `plforecast`; the local directory name is outside the repo's control.

**As** a reader, **I want** the distribution, import package, and prose to agree on a
name, **so that** search, citation, and `pip install` are unambiguous.

Evidence: `pyproject.toml` name is `epl-models`; the package is `plforecast`; README title
and design.md layout say `pl-forecast`; the working directory is `epl-models`.

Acceptance criteria:
- Distribution name `pl-forecast`, import name `plforecast`, repo name matches the
  distribution. README and design.md updated. Justfile and CLI already use `plforecast`.

### A-05 · P1 · Correction · design.md describes a repository that does not exist yet

**Status**: Done 16 Sep 2026: all nine ADRs exist; design.md section 3 is the as-built tree with planned/dropped markers.

**Status**: Partly done 16 Sep 2026: all eight ADRs exist. Section 3 status markers pending.

**As** a reader of design.md, **I want** to know which parts are built and which are
planned, **so that** the document is accurate rather than aspirational.

Evidence: section 3 lists, and other sections cite, files that are absent: `LICENSE`,
`docs/methodology.md`, `docs/model-card.md`, all eight `docs/adr/*` files (section 14
states "each has a corresponding ADR"; sections 2.1, 5.4, 6.3, 10.1 cite them by number),
`notebooks/`, `ingest/clubelo.py`, `ingest/transfermarkt.py`, `storage/schema.sql`,
`models/hierarchical.py`, `models/market.py`, `artifacts/`, `tests/integration/`,
`tests/golden/`, `.github/workflows/`. The implementation also diverged in a good way:
numbered migrations replaced `schema.sql`, and that should be reflected.

Acceptance criteria:
- Section 3 marks each entry as built, planned, or dropped, or is split into "as built" and
  "target" trees.
- The eight ADRs exist. Their content is already written inside design.md sections 2.1,
  5.4, 4, 9, 10.1, 6.3, 8.3, and 14; this is extraction, not authorship.
- Section 14's "each has a corresponding ADR" is true.

### A-06 · P1 · Correction · design.md lives in the wrong place

**Status**: Done 16 Sep 2026: moved to `docs/design.md`.

Evidence: design.md section 3 places it at `docs/design.md`; the file is at the repo
root. `README.md` and every module docstring link to `design.md` by bare name.

Acceptance criteria: move to `docs/design.md`; update links; or amend section 3.

### A-07 · P1 · Correction · Module docstrings carry stale project status

**Status**: Done 16 Sep 2026: all six stale docstrings corrected.

**As** a code reader, **I want** docstrings that describe the module's contract, **so
that** I am not misled about what exists.

Evidence, each verified false today:
- `storage/db.py:3-4`: "Only raw_* exists so far -- stg_* and mart_* land ... once club
  identity resolution is built." Both exist.
- `models/dixon_coles.py:3-4`: the held-out RPS check is "the evaluation harness's job, not
  yet built." It is built and has run.
- `features/strength.py:4-8`: "That needs Understat, which is not ingested yet." It is.
- `features/strength.py:77-78`: "features/priors.py (not yet built; needs ClubElo +
  Transfermarkt)". It is built.
- `evaluate/market.py:11-12`: "once the evaluation harness actually runs end to end". It has.
- `evaluate/market.py:14`: cites `docs/data-sources.md` for the Pinnacle coverage gap;
  data-sources.md does not mention the gap.

Acceptance criteria:
- All six fixed. Rule added to a CONTRIBUTING note: project status lives in README only;
  docstrings describe contracts and gotchas. See also C-12.

### A-08 · P1 · Recommendation · Add a LICENSE and data attribution

**Status**: Done 16 Sep 2026: MIT LICENSE; README attribution section.

Evidence: design.md section 11.5 requires MIT or Apache 2.0 and source attribution; no
`LICENSE` exists; README has no attribution section.

Acceptance criteria: `LICENSE` present; README has a "Data sources and attribution" section
listing football-data.co.uk, FPL, Understat with their terms as recorded in
`docs/data-sources.md`.

### A-09 · P1 · Recommendation · Add CI

**Status**: Done 16 Sep 2026: `ci.yml` (lint, types, tests with coverage floor, artifact validation) and `artifact-validate.yml`.

Evidence: design.md section 11.2 and milestone 1's exit criterion require CI; none exists.
`pytest-cov` is installed but not configured; there is no coverage floor.

Acceptance criteria:
- `.github/workflows/ci.yml`: `uv sync --frozen`, `ruff check`, `ruff format --check`,
  `mypy src/`, `pytest --cov=plforecast --cov-fail-under=<floor>` with `-m 'not network'`.
- No network access; ingest tests use committed fixtures (already true).
- Badge in README.

### A-10 · P1 · Correction · Phase 3 (end-to-end crude forecast) was skipped despite being "deliberate"

**Status**: Done 16 Sep 2026: `plforecast forecast`, `artifacts/` module and schemas, first artifact committed.

**As** the project owner, **I want** a `forecast` command that produces the artifact from
design.md section 9, **so that** the plumbing is proven before the model improves further.

Evidence: design.md section 12 orders phase 3 (Poisson + simulation + artifact + published
page) before phases 4 and 5, and calls that ordering deliberate. Phases 4 and 5 are done;
phase 3 has not started: no `forecast` CLI command, no `artifacts/` module, no schema, no
published page. README says the simulation engine was "validated end to end against real
data", but nothing in the repo reproduces that run (no command, script, or notebook).

Acceptance criteria:
- `plforecast forecast --model {poisson,dixon-coles} --xi <float> --seed <int>
  --simulations <int>` reads played matches and remaining fixtures (see B-07 for which
  table is authoritative), runs `simulate_season`, and writes `forecast-gwNN.json` and
  `fixtures-gwNN.json` matching section 9.3.
- `artifacts/schema.py` (pydantic) and `artifacts/schema/*.schema.json` exist; a test
  validates the emitted files against them.
- A golden test: fixed seed, fixed input, byte-identical artifact (see C-14 on tolerance).
- `just forecast` recipe.

## A.2 Architecture and boundaries

### A-11 · P2 · Recommendation · CLI ergonomics and the dead `since` path

**Status**: Partly done 16 Sep 2026: migrate runs inside connect(). Enum for `ingest` and `--refresh` flag pending.

Evidence: `cli.py:17` takes `source: str` and dispatches on string compare; typer would
render choices and autocomplete from an `Enum`. `Source.fetch(since=...)` is implemented in
all three adapters but no caller ever passes `since`, so the incremental-refresh path is
untested dead code and every ingest is a full backfill (see B-08 for the Understat
consequence).

Acceptance criteria:
- `ingest` takes an `Enum` and supports `all`.
- `--refresh-current` flag exercises `since`; a test covers each adapter's `since` branch.
- `db-migrate` runs automatically inside `curate`, `evaluate`, and `forecast` (design.md
  section 5.4: "applied idempotently at startup").

### A-12 · P2 · Correction · Logs and results share stdout

**Status**: Done 16 Sep 2026: JSON logs to stderr, resolved at write time.

Evidence: `logging.py:23` writes JSON logs to stdout; `cli.py:113` writes the results table
to stdout with `typer.echo`. `just evaluate > results.txt` captures interleaved JSON and
text.

Acceptance criteria: logs to stderr; results to stdout; `evaluate --json` emits the metrics
as one JSON document (C-07 builds on this).

### A-13 · P2 · Correction · Snapshot timestamps are local time labelled "Z"

**Status**: Done 16 Sep 2026: every adapter uses `datetime.now(UTC)`; `write_snapshot` rejects naive timestamps and stamps the directory from the UTC instant.

Evidence: `ingest/base.py:119` formats `payload.fetched_at` with a `Z` suffix, but every
adapter builds the payload with naive `datetime.now()` (`footballdata.py:79`,
`fpl.py:97`, `understat.py:108,111`). The live `_meta.json` for football-data shows
`fetched_at` 10:53:17 while its parts show 15:53:02+00:00: a five-hour discrepancy inside
one file.

Acceptance criteria: `datetime.now(UTC)` everywhere; `_meta.json` timestamps are all
offset-aware and agree; a test asserts the snapshot directory name matches the UTC instant.

### A-14 · P2 · Recommendation · Settings are a module-level singleton bound at import

Evidence: `config.py:71` instantiates `settings` at import; adapters default
`config: Settings = settings` at definition time; `curate.py:167` reads the global
directly. Environment overrides set after import are ignored; tests cannot substitute a
config without monkeypatching.

Acceptance criteria: a `get_settings()` accessor (cached) and explicit injection into
`curate_*`; no function reads the module global.

### A-15 · P2 · Recommendation · Catching bare `ValueError` in the backtest is too broad

**Status**: Done 16 Sep 2026: `UnknownClubError`.

Evidence: `evaluate/backtest.py:99` treats any `ValueError` from `scoreline_matrix` as
"unknown club" and skips the match. A genuinely broken matrix (negative cell, NaN) would be
silently skipped and counted as an unrateable club. Tests match on the substring
"training data", which is penaltyblog's message, not this project's.

Acceptance criteria: models raise a project-owned `UnknownClubError`; the backtest catches
only that; any other exception propagates.

### A-16 · P2 · Recommendation · Runs have no provenance manifest

**Status**: Partly done 16 Sep 2026: forecast artifacts carry git SHA, dirty flag, snapshot hashes and package versions. Evaluation runs do not yet.

Evidence: backtest and (future) forecast outputs record no git SHA, no data snapshot IDs,
no library versions. `config_hash` covers model hyperparameters only; the same hash can
yield different numbers after a penaltyblog upgrade.

Acceptance criteria: a `RunManifest` (git SHA, `uv.lock` hash, raw snapshot directory names,
model `config_hash`, seed, package versions of penaltyblog, numpy, scipy, polars) is written
next to every `evaluate` and `forecast` output and embedded in the artifact provenance block.

### A-17 · P2 · Recommendation · Untested plumbing

**Status**: Done 16 Sep 2026: `tests/unit/test_ingest_base.py` (TTL cache hit/miss, politeness delay, write_snapshot contracts), `tests/unit/test_db.py` (migration/view idempotence, pre-existing), `tests/unit/test_cli.py` (CliRunner, pre-existing). `tests/integration/test_pipeline.py` runs migrate/curate/validate/evaluate over a committed real-club two-season fixture dataset (`tests/fixtures/integration_data/`). Golden byte-identical-artifact test deferred: determinism is instead asserted directly (`test_documents_are_deterministic_given_seed` in test_artifacts.py, `test_simulate_season_is_deterministic_given_a_seed` in test_engine.py), which is the same guarantee without a brittle fixed-file comparison.

Evidence: no tests for `ingest/base.py` (TTL cache hit vs miss, politeness delay only on
live fetch, `_meta.json` contents, never-overwrite), `storage/db.py` (migration order,
idempotence, applied-set), or `cli.py` (typer `CliRunner`). design.md section 11.3 also
calls for an integration test over a committed two-season dataset and golden tests; neither
exists.

Acceptance criteria: unit tests for the three modules; one integration test that runs
migrate, curate, evaluate on a small fixture `data_dir`; golden test from A-10.

### A-18 · P2 · Recommendation · "Never overwrites" is not enforced

**Status**: Done 16 Sep 2026: `write_snapshot` uses `mkdir(exist_ok=False)` and raises a clear `FileExistsError` naming the collision.

Evidence: `ingest/base.py:120-121` uses `mkdir(exist_ok=True)` then `write_parquet`; two
runs within the same second overwrite each other silently.

Acceptance criteria: `mkdir(exist_ok=False)` and a clear error, or microsecond precision in
the directory name; a test for the collision.

### A-19 · P3 · Recommendation · Notebook hygiene

**Status**: Done 16 Sep 2026: empty notebook removed; `notebooks/README.md` states the conventions.

Evidence: `eda.ipynb` at the repo root has zero cells. design.md section 4 requires
`notebooks/NN-MM-question.ipynb`, an opening cell with question and conclusion, and
deletion after a week without a conclusion. There is no `notebooks/README.md`.

Acceptance criteria: delete `eda.ipynb` or move it to `notebooks/01-eda/01-01-...ipynb` with
content; `notebooks/README.md` states the conventions; nbstripout hook installed (A-03).

### A-20 · P2 · Confirmation · Layer boundaries are respected, with one dependency to note

The layering in design.md section 2.2 is honoured: ingest does no club resolution;
entities is the only resolver; models implement a protocol and are swapped by factory;
simulate consumes `MatchModel`; evaluate shares one splitter. Keep this.

One cross-layer import: `features/priors.py:56-57` imports `simulate.competition` and
`simulate.tiebreak` to rank historical tables. Acceptable, but the design diagram shows
features upstream of simulate. Either document the exception or move `ClubSeasonResult`
and standings computation into a small `standings` module both layers import.

### A-21 · P2 · Recommendation · The DuckDB file is not portable

**Status**: Done 16 Sep 2026 (with A-02).

Evidence: `storage/db.py:43` substitutes the absolute resolved `raw_dir` into the view SQL
and records the migration as applied. Moving the repository, or changing
`PLFORECAST_DATA_DIR`, leaves views pointing at the old path with no re-migration.

Acceptance criteria: `raw_*` views are recreated on every `connect()` (they are cheap and
idempotent); `schema_migrations` is reserved for tables. This also resolves A-02.

---

# Part B: Data architect and engineer lens

Criteria: schema and source correctness, and how data and features are ingested and
processed.

## B.1 Ingestion and refresh

### B-01 · P0 · Correction · The second FPL ingest breaks curate

**Status**: Done 16 Sep 2026: latest-snapshot selection in curate, two-snapshot test.

**As** the operator running weekly refreshes, **I want** curate to succeed after any
number of FPL ingests, **so that** the "refreshed after each gameweek" promise holds.

Evidence: `0002_raw_fpl.sql` unions every snapshot directory with no `filename` column.
`curate_fixtures` (`curate.py:160-163`) selects every row with no dedupe.
`MartFixtureSchema.fixture_id` is `unique=True` (`curate.py:77`). After a second
`just ingest-fpl`, every fixture appears twice and `just curate` raises a `SchemaError`.
Only one snapshot exists today, which is why it has not fired. The same applies to
`raw_fpl_teams` and `raw_fpl_player_availability`.

Acceptance criteria:
- Migration 0005 adds `filename = true` to the three FPL views (matching 0003 and 0004).
- A shared `latest_snapshot(view)` helper in curate selects rows from the most recent
  snapshot directory for full-state sources (FPL, football-data, Understat all are), then
  applies natural-key dedupe as a second guard.
- A test seeds two FPL snapshots and asserts one row per `fixture_id`, from the later one.

### B-02 · P0 · Correction · The market benchmark cannot be computed for the live season

**Status**: Done 16 Sep 2026: five closing sets landed, `stg_odds`, fallback chain, ADR 0007.

**As** the evaluator, **I want** a closing price for every match including the current
season, **so that** "evaluated against the market" is possible for the published forecast.

Evidence (verified from the cached CSVs):

| Season file | Closing-odds columns present |
| --- | --- |
| 2015/16 to 2018/19 | `PSC*` only |
| 2019/20 onward | adds `B365C*`, `BWC*`, `IWC*`, `WHC*`, `MaxC*`, `AvgC*` |
| 2024/25 onward | adds `BFEC*` (Betfair Exchange closing) |
| 2025/26 | `PSC*` present but null from 17 January 2026 (170 matches) |
| 2026/27 | `PSC*` absent entirely |

The adapter keeps only `PSC*`, so 210 of the last 550 matches have no benchmark.
design.md section 8.3 chose Pinnacle for continuity; the source no longer provides it.

Acceptance criteria:
- ADR 0007 amended: benchmark price is a documented fallback chain, Pinnacle closing, then
  Betfair Exchange closing, then `AvgC*`, with the source recorded per row.
- Adapter lands every `*C[HDA]` column present (long shape, see B-09's `stg_odds`), not
  three fixed columns.
- `docs/data-sources.md` documents per-season closing-odds coverage.
- `evaluate` reports benchmark n per price source.

### B-03 · P1 · Correction · `fpl_team_id` is a per-season index, not a stable key

**Status**: Done 16 Sep 2026: `fpl_code` landed and keyed; fixtures mapped via the roster.

**As** the maintainer of `club_aliases.yaml`, **I want** the FPL alias to survive
promotion and relegation, **so that** historical FPL snapshots do not resolve to the wrong
club next season.

Evidence: FPL `teams[].id` is 1 to 20, alphabetical, re-assigned every season. The
bootstrap payload also carries `teams[].code`, a stable multi-season identifier
(verified: Arsenal 3, Aston Villa 7, Bournemouth 91, Brentford 94, Brighton 36).
`club_aliases.yaml` stores `id`, and `_UNIQUE_COLUMNS` treats it as a permanent alias.
Fixtures reference teams by `id`, so the season-scoped mapping is still needed, but it
belongs in a bridge, not on the club dimension.

Acceptance criteria:
- `FPLTeamsSource` lands `code`; yaml column becomes `fpl_code`.
- `curate_fixtures` joins `team_h`/`team_a` to that snapshot's teams table to obtain
  `code`, then resolves. The season-scoped `id` is stored in `stg_club_season` (or a
  `stg_fpl_team_season` bridge), never on the dimension.
- `tests/fixtures/entities/fpl_team_ids.json` becomes codes.

### B-04 · P1 · Correction · Understat club names are unresolved

**Status**: Done 16 Sep 2026: all 35 `understat_name` values populated and tested.

Evidence: `understat_name` is null for all 35 clubs. Verified against the live data, 29
names match football-data exactly and 6 differ:

| Understat | football-data |
| --- | --- |
| Manchester City | Man City |
| Manchester United | Man United |
| Newcastle United | Newcastle |
| Nottingham Forest | Nott'm Forest |
| West Bromwich Albion | West Brom |
| Wolverhampton Wanderers | Wolves |

Acceptance criteria: yaml populated for all 35; a fixture file of observed Understat names
and a test that every one resolves 1:1 (mirroring `test_clubs.py`); B-05 unblocked.

### B-05 · P1 · Recommendation · Build the feature grain: `mart_team_match` with xG joined

**Status**: Done 16 Sep 2026: xG joined onto `stg_matches` with coverage and scoreline assertions; `mart_team_match` built; `build_club_strength` takes `metric`.

**As** the model author, **I want** one long table, one row per club per match, carrying
goals, xG, non-penalty xG, and the odds, **so that** every feature and model reads the same
grain and xG is actually usable.

Evidence: Understat is landed (4,220 rows, complete) but nothing joins it; `stg_matches`
has no xG columns; `features/strength.py` rebuilds a long appearances frame inline
(`strength.py:98-115`) that this table would replace. design.md section 5.4 names
`mart_team_match`. The join key (season, date, home club, away club) is sound: on the rows
whose names already agree, goals match on every joined row.

Acceptance criteria:
- `stg_matches` gains `home_xg`, `away_xg`, `home_np_xg`, `away_np_xg` via a resolved
  join; curate asserts 100 percent coverage (4,220 = 4,220) and goal agreement.
- `mart_team_match` materialised: `match_id, season, date, club_id, opponent_id, is_home,
  goals_for, goals_against, xg_for, xg_against, npxg_for, npxg_against, rest_days`.
- `build_club_strength` reads from it.

### B-06 · P1 · Correction · "Swap in xG later is a data-source change" is not true

**Status**: Done 16 Sep 2026: `models/xg_rates.py` (rung 2.5) per ADR 0009, in the evaluation ladder; shipped model chosen by primary RPS.

**Status**: Decided 16 Sep 2026: ADR 0009 chooses an xG-rate model as rung 2.5, gated by the backtest. Implementation is the next step.

Evidence: `strength.py:9-10` claims xG substitution is a data change. penaltyblog's
Poisson and Dixon-Coles models take integer goal counts; xG is continuous. The model layer
needs a decision, not a column swap.

Acceptance criteria: an ADR choosing one of:
1. Fit on goals; use xG only in priors and features. (Simplest; contradicts section 6.3.)
2. Fit attack, defence, and home terms on xG with weighted least squares or a Gamma GLM on
   the log scale, then sample goals from Poisson at those rates. Reuses the entire
   evaluation harness. Recommended as rung 2.5 with the same RPS gate.
3. Defer to the hierarchical model with an xG likelihood.

### B-07 · P1 · Correction · Current-season results exist in two tables with no declared authority

**Status**: Done 16 Sep 2026: `reconcile_current_season` in curate; FPL authoritative for the live season.

Evidence: `stg_matches` (football-data, weekly) and `mart_fixtures` (FPL, near real time)
both carry 2026/27 scorelines. They agree today (40 matches, 0 disagreements), but nothing
checks that, and the forecast command (A-10) needs to know which one is "played matches".

Acceptance criteria:
- Decision recorded: FPL is authoritative for current-season results and kickoff times;
  football-data for history and odds.
- Curate adds a reconciliation check: for finished FPL fixtures with a football-data row,
  scorelines must agree (fail loudly); count gaps are logged, not failed.
- `forecast` reads played matches from the declared source.

### B-08 · P1 · Correction · Understat refresh returns stale data for the current season

**Status**: Done 16 Sep 2026: current season always fetched live (`season_batches`).

Evidence: `cli.py:27-30` calls `ingest()` which calls `fetch()` with `since=None`;
`understat.py:80` then sets `no_cache=False`, so soccerdata reuses
`data/cache/understat-soccerdata/league_1_season_2026.json` indefinitely. Weekly runs land
the same 40 matches until someone deletes the cache. The docstring describes the
`since` path, but nothing invokes it (A-11).

Acceptance criteria: default ingest uses the cache for completed seasons and `no_cache`
for the current season (two scraper instances); `_meta.json` records cache state per
season; a test asserts the current season is fetched live.

## B.2 Schema and zones

### B-09 · P1 · Recommendation · Define the zones precisely and name tables accordingly

**Status**: Done 16 Sep 2026: zone criteria in design.md 5.4; `stg_fixtures`, `dim_club`, `stg_odds`, `mart_team_match`.

Evidence: design.md section 5.4 defines `stg_*` as typed and resolved and `mart_*` as
model-ready. `mart_fixtures` is typed and resolved only (staging grade). `stg_club_season`
matches the design's `mart_club_season`. There is no club dimension table in DuckDB
(`ClubDimension.frame` is held in memory only), so SQL cannot join to display names.
`stg_matches` carries three Pinnacle columns instead of a general odds table.

Acceptance criteria:
- Written zone criteria in design.md 5.4: raw (as landed, all snapshots); stg (typed,
  deduped, resolved, one row per natural key, no cross-source joins, no derived columns);
  mart (feature grain, cross-source joins, derived columns).
- Rename `mart_fixtures` to `stg_fixtures`. Add `dim_club` from the yaml. Add
  `stg_odds` (`match_id, bookmaker, price_type, home, draw, away`) from B-02. Add
  `mart_team_match` from B-05.
- `data/MANIFEST.md` and design.md section 5.4 updated.

### B-10 · P2 · Recommendation · Derive season and gameweek from the data, not the wall clock

**Status**: Done 16 Sep 2026: `stg_fixtures.season` derived from the fixture list's own earliest kickoff via `derive_season_from_kickoffs`, not the wall clock; `FPLEventsSource` lands the gameweek calendar as `raw_fpl_events`, curated into `stg_gameweeks`; `forecast`'s `as_of_gameweek` reads FPL's own `is_current` flag, falling back to the finished-fixture computation only when `stg_gameweeks` is absent.

Evidence: `curate.py:167` stamps `mart_fixtures.season` from today's date with a July
rollover. FPL's API rolls over on its own schedule; a June or early-July run would label
next season's fixtures with last season's label. The artifact needs `as_of_gameweek`
(section 9.3), which the bootstrap `events` array provides (`is_current`, `finished`,
`deadline_time`), but `FPLTeamsSource` and `FPLPlayersSource` discard it.

Acceptance criteria: season derived from the minimum fixture kickoff; a small
`fpl-events` raw table (or a parsed field) supplies the current gameweek; both covered by
tests.

### B-11 · P2 · Recommendation · Add lineage columns and a curate manifest

**Status**: Done 16 Sep 2026, with one deliberate deviation: every curated table carries a shared `curated_at` timestamp per `curate_all()` run, but full snapshot lineage is a manifest (`data/curate-manifest.json`, row counts plus every raw view each table read and its snapshot directory) rather than a per-row `source_snapshot` column, since most curated tables join two or more raw sources (stg_matches alone reads football-data and Understat) and one column would misrepresent that. `--snapshot` pin deferred: immutable raw snapshots plus the manifest already name exactly what a run used.

Evidence: `stg_*` and `mart_*` rows carry no snapshot identifier or load time; the design's
"given a git SHA and a data snapshot, any run reproduces" (section 1.1) has no way to name
the snapshot.

Acceptance criteria: every curated table has `source_snapshot` (raw directory name) and
`curated_at`; `curate` writes a manifest of row counts and snapshot IDs per table;
`curate --snapshot <name>` pins a raw snapshot.

### B-12 · P2 · Correction · `content_hash` hashes the Parquet file, not the content

**Status**: Done 16 Sep 2026: `content_hash()` hashes sorted per-row values via `hash_rows()`, independent of column order, row order, and Parquet/Arrow writer metadata.

Evidence: `ingest/base.py:122` hashes `data.parquet` bytes. Parquet output embeds writer
metadata and can differ across polars or arrow versions for identical rows, so the hash is
not a content identity. The site-repo workflow in design.md 10.1 compares content hashes
to decide whether to commit.

Acceptance criteria: hash a canonical serialisation (sorted rows, fixed column order, e.g.
an aggregate of `df.hash_rows()` or CSV bytes); a test shows the hash is stable across two
writes and changes when one value changes.

### B-13 · P2 · Recommendation · A `validate` command for data-quality invariants

**Status**: Done 16 Sep 2026: `plforecast validate` / `just validate` runs season round-robin shape, id uniqueness, no self-fixtures, result-vs-score, xG coverage, closing-odds coverage (report), club_id resolution against the live tables, and the B-07 reconciliation; exits non-zero naming every failed check. Also exercised in `tests/integration/test_pipeline.py`.

**As** the operator, **I want** a single command that checks the curated tables after every
curate, **so that** silent joins and coverage gaps are caught before a forecast is
published.

Checks, all cheap in DuckDB:
- 380 matches per completed season; 20 clubs per season; each club 19 home and 19 away.
- `match_id` and `fixture_id` unique; no self-fixtures; result agrees with score (already
  enforced in curate; re-check).
- xG coverage 100 percent; Understat goals equal football-data goals.
- FPL finished-fixture scores equal football-data scores (B-07).
- Closing-odds coverage per season and per price source, reported not failed.
- Every club in any fact table resolves to exactly one `club_id` (design.md 5.3's test,
  today only checked against a fixture snapshot of names, not the live tables).

Acceptance criteria: `plforecast validate` and `just validate`; non-zero exit on a failed
invariant; run as an integration lane in CI against a fixture `data_dir`.

### B-14 · P2 · Recommendation · Implement the leakage guards

**Status**: Done 16 Sep 2026: `models/base.py` adds `assert_no_odds_columns` (called first in every model's `fit()`) and `drop_odds_columns`; `run_backtest` and `evaluate_season_level` strip odds columns from the training frame before fitting, since their shared `matches` frame legitimately carries odds for market scoring. `tests/unit/test_leakage.py`: a hypothesis property test that `attach_decay_weights` never emits a row after `as_of`, a test combining a real `walk_forward_splits` split with `build_club_strength` to check the feature layer fed exactly what the backtest hands a model, and coverage for both odds-column guards.

Evidence: design.md section 8.4 asks for (a) a test that no feature frame contains data at
or after the kickoff it describes and (b) an assertion that odds are never a model feature.
`attach_decay_weights` filters `date <= as_of` (inclusive), which is correct for training
but the backtest uses `<` at the split; neither is asserted as a feature-frame invariant.
Nothing asserts (b).

Acceptance criteria: a `features` test that every emitted frame has `max(timestamp) <
as_of`; model `fit()` asserts no `*odds*` column is present in its input; the backtest
test already covering train dates is extended to the feature frames.

### B-15 · P2 · Recommendation · Understat provenance is weaker than the other adapters

**Status**: Done 16 Sep 2026: one `RawPart` per Understat season batch with `from_cache` reflecting that batch's cache permission and a pseudo-URL carrying the `soccerdata` version; rate-limiting arrangement documented in data-sources.md.

Evidence: `understat.py:104-110` fabricates a URL, hardcodes status 200, uses naive
timestamps, and records neither the soccerdata version nor whether each season came from
soccerdata's cache. The politeness delay in `Settings` is not applied (soccerdata manages
its own), which `docs/data-sources.md` should say.

Acceptance criteria: `_meta.json` records `via_library: soccerdata==<version>` and per-season
cache state; data-sources.md documents the rate-limiting arrangement.

### B-16 · P3 · Recommendation · Columns landed without a consumer

Evidence: `home_ppda`/`away_ppda` (max observed 193, a division artefact) and the whole
`fpl-players` table have no downstream reader. Landing is cheap and immutable, so keep
them, but say so.

Acceptance criteria: data-sources.md marks both "landed, unconsumed"; PPDA is excluded from
any feature until capped or winsorised; no `stg_player_availability` until a feature needs
it.

### B-17 · P3 · Recommendation · Raw retention and view cost

Evidence: every ingest lands a full snapshot and every `raw_*` view unions all of them.
Sizes are tiny today (`data/raw` is 256 KB), but view scans grow linearly with weekly runs
and B-01's dedupe becomes the only thing keeping counts right.

Acceptance criteria: `just prune-raw --keep N` with the latest snapshot always retained;
MANIFEST documents the policy.

### B-18 · P2 · Correction · Odds coercion silently nulls unparsable values

**Status**: Done 16 Sep 2026: `_coercion_failures()` distinguishes an unparsable odds cell from a blank one and logs a per-season count; corrupted cells still land as null, never crash the backfill.

Evidence: `footballdata.py:119` casts odds with `strict=False`, turning any unparsable
string into null with no count. Nullable is the right schema, but a coercion failure is a
different event from a missing value.

Acceptance criteria: per-season count of values nulled by coercion is logged; the count
must be zero where the column exists; a test with a corrupted odds cell.

### B-19 · P2 · Confirmation · Data-layer decisions to preserve

The following are correct and should be protected by the stories above rather than
reopened:
- Immutable timestamped Parquet snapshots with a `_meta.json` sidecar (`ingest/base.py`).
- pandera schemas at both parse and curate boundaries, `strict=True`, with an explicit
  timezone pin on `kickoff_time` (verified `TIMESTAMP WITH TIME ZONE` in DuckDB).
- Selecting the stable core columns from football-data at parse time rather than the
  drifting full column set (though B-02 widens "core" to all closing prices).
- Strict alias resolution that raises with every unmatched value in one error; fuzzy
  matching kept out of the pipeline (`entities/clubs.py`).
- Club-season membership as a bridge table, not a column on the club.
- Natural-key dedupe with deterministic latest-snapshot-wins (`curate_matches`).
- One canonical season label and one season code (`config.py`), and the pair-code fix for
  soccerdata's ambiguous integer seasons, which is documented well.
- Cross-source sanity checks recorded in data-sources.md (row counts and goal agreement).

---

# Part C: Outside both lenses

Modelling and statistics, product and publishing, and project management.

## C.1 Evaluation integrity

### C-01 · P0 · Correction · The evaluation coverage accounting does not reconcile

**Status**: Done 16 Sep 2026: `BacktestResult` accounting; evaluation.md regenerated.

**As** a reader of `docs/evaluation.md`, **I want** the numbers to add up, **so that** the
document is credible.

Evidence: evaluation.md states 4,180 matches in the window, 4,066 scored, and "14 matches
skipped, every one a round-1 fixture in 2015/16 or a later promoted club". 4,180 minus
4,066 is 114, not 14. `walk_forward_splits` (`backtest.py:65`) never yields a test split
until 100 training matches exist, so the first roughly 100 matches of 2015/16 are excluded
without being counted anywhere. The 14 skipped by `run_backtest` therefore cannot be
2015/16 round-1 fixtures (those fall inside the warm-up); they are the debut matches of
promoted clubs with no prior-window history. Counting from `club_aliases.yaml` and the
membership table: Burnley, Middlesbrough, Hull (2016); Brighton, Huddersfield (2017);
Wolves, Cardiff, Fulham (2018); Sheffield United (2019); Leeds (2020); Brentford (2021);
Nottingham Forest (2022); Luton (2023); Ipswich (2024). That is 14.

Acceptance criteria:
- `run_backtest` returns warm-up exclusions and unrateable skips as separate counts (not
  only a log line), with the skipped match IDs.
- evaluation.md reports both, with the correct explanation.

### C-02 · P0 · Correction · Models and the market are not scored on identical rows

**Status**: Done 16 Sep 2026: primary metrics on the intersection (4,066 rows).

Evidence: model rows are 4,066 (including 170 late-2025/26 matches with no odds). Market
rows are 4,010 (including the roughly 100 warm-up matches and the 14 debut matches the
models never predicted, excluding the 170). design.md section 8.2: "the market baseline is
scored on the same splits". The headline comparison in README and evaluation.md is
therefore between different match sets.

Acceptance criteria: primary metrics for every model and both de-vig methods are computed
on the intersection (rows with a model prediction and a benchmark price); one n is
reported; full-set numbers may be reported as secondary. README and evaluation.md updated
with the recomputed numbers, whichever direction they move.

### C-03 · P1 · Correction · `xi` is asserted to be tuned but is hardcoded to the 1997 value

**Status**: Done 16 Sep 2026: `plforecast tune` (selection 2015/16-2021/22, report 2022/23-2025/26); Dixon-Coles xi stays 0.0018 (interior optimum); xG-rates xi 0.005, blend 0.75, rho 0.05; grids rendered in evaluation.md; values live in config.py.

Evidence: `dixon_coles.py:7-12` refuses a default for `xi` because it must be "tuned by
backtest, never assumed"; `cli.py:83` then passes `0.0018`, the Dixon and Coles paper
value, and evaluation.md reports it as the shipped configuration. design.md's risk table
names overfitting `xi` and prescribes tuning on a holdout disjoint from the reporting
period. Cost is low: one fit is 0.04 s, so a full walk-forward per candidate is about
50 s per model.

Acceptance criteria:
- Grid over `xi` in {0, 0.0005, 0.001, 0.0018, 0.003, 0.005} selected on 2015/16 to
  2021/22, reported on 2022/23 to 2025/26.
- Selected value lives in config, not in `cli.py`; evaluation.md shows the grid.

### C-04 · P1 · Correction · The calibration claim is overstated and the pooled curve hides per-outcome error

**Status**: Done 16 Sep 2026: prose computed from data; per-outcome curves added.

Evidence: evaluation.md says mean predicted probability falls inside the empirical
confidence interval "in every one of the 10 buckets". Bucket 0.0-0.1: mean predicted
0.068, interval [0.070, 0.120]. It is outside, and the direction matters: outcomes the
model prices at about 7 percent occur about 9 percent of the time, so longshots are
under-priced. Pooling home, draw, and away into one curve (`calibration.py:5-12`) also hides
the draw miscalibration Poisson-family models are known for.

Acceptance criteria: prose corrected; per-outcome reliability curves added; a scalar
summary (expected calibration error or a calibration slope) reported alongside; if the
draw curve is off, that becomes a modelling story.

### C-05 · P1 · Correction · Uncontested ties in the simulator are broken alphabetically

**Status**: Done 16 Sep 2026: uncontested ties ordered by rng; symmetry test.

Evidence: `engine.py:65` sorts `club_ids` alphabetically; `_resolve_ties` builds standings
in that order; `PremierLeagueTiebreaks.rank` (`tiebreak.py:74`) uses a stable sort and
applies head-to-head only to contested positions. For an exact points, goal difference, and
goals-for tie in positions 8 to 17, the alphabetically earlier club always ranks higher.
Over 50,000 simulations that is a small but systematic bias in `position_pmf` (Arsenal
never loses a mid-table tie to Wolves). The Premier League awards such positions jointly.

Acceptance criteria: uncontested ties are ordered by the supplied `rng` (or the count is
split evenly across the tied positions); a test with two statistically identical clubs
asserts symmetric position distributions.

### C-06 · P2 · Recommendation · Evaluation outputs are hand-transcribed into docs

**Status**: Done 16 Sep 2026: `evaluate` writes metrics.json; `render-evaluation` renders the doc.

Evidence: `evaluate` prints a table; evaluation.md's calibration table came from "the
equivalent ad hoc script" that is not in the repo. The doc says it is a snapshot that should
be regenerated, but nothing regenerates it.

Acceptance criteria: `evaluate` writes `docs/evaluation/metrics.json` and
`calibration.csv`; a small renderer produces the Markdown tables; CI fails if the rendered
doc is out of date with the committed data.

### C-07 · P2 · Recommendation · Season-level evaluation is missing and is the metric the product claims

**Status**: Done 16 Sep 2026: `evaluate/season.py` scores position RPS and title/top-four/relegation log loss at cutoffs of 100, 190 and 280 matches for every completed season; rendered in evaluation.md.

Evidence: design.md section 8.1 asks for realised final position versus predicted
distribution at a frozen gameweek. Everything needed exists: `stg_matches` has every
completed season; remaining fixtures at any cutoff are the unplayed pairs of the round
robin; `simulate_season` and `PremierLeagueTiebreaks` are built. No FPL data is required.

Acceptance criteria: for each completed season and cutoffs at 10, 19, and 28 matches
played, run the simulation and score the realised position with a ranked probability
score over 20 positions, plus log loss of realised title, top-four, and relegation
outcomes; reported in evaluation.md; this becomes the gate for the hierarchical model too.

### C-08 · P2 · Recommendation · A bridge for the promoted-club prior before the hierarchical model

**Status**: Done 16 Sep 2026: prior enters as pseudo-observations against real opponents (a phantom opponent was not identifiable); `needs_prior` counts decayed evidence. Coventry, Hull and Ipswich covered in the live forecast. Not yet in the backtest.

**Correction, same day**: the gate initially reused the shipped model's own fitting `xi` (0.005 for xg-rates) as the decay rate, which flagged every established club -- Arsenal's 422 historical matches decay to an effective ~17 under that rate, below the 19-match threshold, so 20 of 20 current clubs got prior pseudo-matches injected. Fixed with a dedicated `PRIOR_GATE_XI` (~6-year half-life, ADR 0006) separate from any model's fitting decay; the live forecast now correctly flags only Coventry and Hull (Ipswich's one recent season carries enough weight at this rate to stand on its own).

Evidence: `features/priors.py` builds a prior nothing consumes; README says the two
shipped models "don't have a mechanism to accept a prior". They do, indirectly: penaltyblog
takes per-row `weights`, so a prior can enter as pseudo-observations.

Acceptance criteria: convert the prior (rate means and standard deviations) into K weighted
synthetic matches against a league-average opponent appended to the fit frame, with K
derived from the prior width; gate on RPS over promoted-club matches only; Ipswich-style
clubs with one recent season bypass it per `needs_prior`.

### C-09 · P2 · Recommendation · Backtest cadence should mirror the publishing cadence

Evidence: the backtest refits at every distinct match date (1,252 splits). The product
publishes once per gameweek. Per-date refits are a fair match-level protocol, but the
season-level evaluation (C-07) and the live process both operate per gameweek.

Acceptance criteria: a `--cadence {date,gameweek}` option; gameweek derived from FPL for
the current season and from date clustering for history; runtime recorded in
evaluation.md.

### C-10 · P3 · Recommendation · Success criterion progress is invisible

**Status**: Done 16 Sep 2026: evaluation.md states the gap to market and whether the stretch target is met.

Evidence: design.md 1.3's stretch target is within 0.005 RPS of the market. Current gap is
0.0075 on non-identical rows (C-02).

Acceptance criteria: after C-02, the evaluation table carries a "gap to market" column and
the target line, so progress is legible in the doc.

## C.2 Publishing and product

### C-11 · P1 · Recommendation · Resolve the open questions and start the publishing path

**Status**: Done 16 Sep 2026: ADRs 0004 and 0005; design.md section 15 updated. Site workflow not yet written.

Evidence: design.md section 15 leaves four questions open; section 10 has no
implementation; 14 weeks remain to the gameweek 19 target. Recommendations, to be recorded
as ADR decisions:
1. `latest.json`: overwrite in place (simplest for the site) and also commit
   `forecast-gwNN.json` (history). Both, not either.
2. Recency half-life for the prior mean: needs C-07's harness; defer with a default and a
   stated test plan.
3. Market record on the front page: yes.
4. `fixtures-latest.json`: publish from the first forecast, since the evaluation harness
   already exists and the scored record will accompany it.

Acceptance criteria: ADR 0004 and 0005 written with these decisions; the site-repo pull
workflow sketched; the first crude forecast (A-10) published to a private or unlisted route
within two weeks.

### C-12 · P2 · Recommendation · Move the "why" out of docstrings into docs

Evidence: several module docstrings run 20 to 40 lines and narrate project history ("the
season simulation validated in a prior session failed outright..."). That prose is valuable
but belongs in ADRs, `docs/methodology.md`, and the model card, where it will not rot with
the code and where design.md section 11.4 already expects it. This is also the root cause
of A-07.

Acceptance criteria: `docs/methodology.md` and `docs/model-card.md` written from existing
docstring material; docstrings reduced to contract plus gotchas (roughly 10 lines); a
CONTRIBUTING note states the split.

### C-13 · P3 · Recommendation · State the determinism claim precisely

**Status**: Done 16 Sep 2026 in spirit: artifacts carry git dirty flag and package versions; determinism tests compare simulation output exactly.

Evidence: design.md 1.3 promises bit-for-bit reproducibility. Simulation is seeded and
deterministic (verified by test). Model fitting uses scipy optimisers whose results can
differ at floating-point precision across BLAS builds and platforms.

Acceptance criteria: the claim is "identical on the same platform and lock file"; golden
tests compare fitted parameters with a tolerance and simulation outputs exactly.

### C-14 · P3 · Recommendation · Home advantage

Evidence: design.md 6.3 asks for a time-varying home advantage. Both shipped models fit one
home term over the decayed window, which approximates it. Not a defect; record it as
"satisfied by decay for rungs 1 and 2; explicit per-season term in rung 3".

## C.3 Project management and risk

### C-15 · P1 · Recommendation · Re-sequence the remaining 14 weeks

**Status**: Phase 1 and Phase 2 done 16 Sep 2026 (all listed stories except the site-repo workflow, which lives outside this repository). Phase 3 (hierarchical model, rung 3) not started; the shipped xG-rates model is within the 0.005 RPS stretch target, so the December deliverable stands on it if rung 3 does not beat it. Phase 4 is calendar-bound (gameweek 19).

Evidence: today is 16 September 2026; the target is a published forecast before gameweek
19 in late December. Done: ingest for three of five sources, entities, curate, Poisson,
Dixon-Coles, simulation engine, match-level evaluation, calibration, standalone prior.
Not done: end-to-end forecast and artifact (phase 3), publishing (section 10), xG
integration, promoted-club prior wiring, season-level evaluation, hierarchical model, ClubElo
and Transfermarkt adapters, CI, ADRs, license.

Recommended order, with the P0 corrections first in each window:
1. By 30 September: A-01 to A-03, B-01, B-02, C-01, C-02, C-05, then A-10 and C-11 (a
   crude Dixon-Coles forecast exists and is published somewhere).
2. By 31 October: B-03 to B-09 (feature grain with xG), C-03 (tuned `xi`), C-07
   (season-level evaluation), C-08 (prior bridge), A-09 (CI), A-05 (ADRs).
3. By 30 November: hierarchical model only if the above is complete; otherwise the December
   deliverable is a published, honest, calibrated Dixon-Coles forecast with the market
   record shown. The non-goals section already says scope creep is the main risk; the
   hierarchical model is the most likely place for it.
4. December: gameweek 19 refit and comparison (phase 7).

### C-16 · P2 · Recommendation · ClubElo and Transfermarkt: decide whether they are v1

Evidence: design.md lists five sources and milestone 2 requires "all five adapters". The
prior module already supports an external rating but nothing supplies one. Both adapters
are scraped, ToS-sensitive, and add club-alias maintenance for a second and third naming
scheme.

Acceptance criteria: an explicit decision. Recommended: ClubElo in v1 (a single CSV
endpoint, covers lower divisions, directly feeds the prior); Transfermarkt deferred to v2.
Update section 5.1, milestone 2, and the risk table.

### C-17 · P3 · Recommendation · Keep the risk table live

Evidence: design.md section 13's table is good but static. Two risks have already
materialised and are not recorded there: the benchmark price source disappeared mid-season
(B-02), and the FPL identifier is season-scoped (B-03).

Acceptance criteria: both added with their mitigations; the table gains a "status" column.

---

## Index by priority

| Priority | Stories |
| --- | --- |
| P0 | A-01, A-02, A-03, B-01, B-02, C-01, C-02 |
| P1 | A-04, A-05, A-06, A-07, A-08, A-09, A-10, B-03, B-04, B-05, B-06, B-07, B-08, B-09, C-03, C-04, C-05, C-11, C-15 |
| P2 | A-11 to A-18, A-20, A-21, B-10 to B-15, B-18, B-19, C-06 to C-09, C-12, C-16, C-17 |
| P3 | A-19, B-16, B-17, C-10, C-13, C-14 |
