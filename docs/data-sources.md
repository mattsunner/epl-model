# Data sources

Inventory of external sources the pipeline ingests. See `design.md` section 5 for the
ingestion contract every adapter implements.

## football-data.co.uk

- **Responsibility**: match results and closing 1X2 odds, E0 (Premier League) division.
  The market benchmark (design.md section 8.3, ADR 0007) is built from a fallback chain of
  closing prices, because the site's closing-price columns change by season:

  | Seasons | Closing-price sets in the file |
  | --- | --- |
  | 2015/16 to 2018/19 | Pinnacle (`PSC*`) only |
  | 2019/20 to 2023/24 | Pinnacle, Bet365 (`B365C*`), Max (`MaxC*`), Avg (`AvgC*`) |
  | 2024/25 | the above plus Betfair Exchange (`BFEC*`) |
  | 2025/26 | the above; Pinnacle null from 17 January 2026 (170 matches) |
  | 2026/27 | Bet365, Max, Avg, Betfair Exchange; Pinnacle columns absent |

  The adapter lands all five sets (nulls where absent); curate stores them long in
  `stg_odds` and picks `benchmark_*` per match as Pinnacle, then Betfair Exchange, then
  Avg. Coverage by source is logged by `just curate` and reported in
  `docs/evaluation.md`.
- **Access**: one CSV per season at
  `https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv`, where `season_code` is
  the two two-digit season-start/end years, e.g. `1516` for 2015/16. No auth, no rate
  limit published; the adapter applies a politeness delay and a TTL cache regardless.
- **Coverage used**: 2015/16 onward, matching the backtest window in design.md section
  8.2 (Understat xG does not exist before 2014/15). Full history back to 1993 is
  available at the source if the backtest window is ever extended earlier.
- **Column drift**: the core columns used here (date, teams, full-time score) are stable
  across seasons. The bookmaker-odds column set is not -- it changes shape every season as
  individual bookmakers stop or start reporting to the site, and that includes the
  closing prices (see the table above). The adapter selects the core plus the five
  closing-price sets at parse time rather than carrying the full raw column set into
  `data/raw/`.
- **Unplayed fixtures**: the season CSV only contains played matches. Remaining-fixture
  data (for the simulation engine) comes from the FPL API, not this source.
- **License / attribution**: free for personal/non-commercial use per the site's terms;
  attribute football-data.co.uk when publishing derived results.

## FPL API

- **Responsibility**: fixture list, kickoff times, player availability and
  suspensions, and the gameweek calendar. One physical source backs four raw tables
  (`src/plforecast/ingest/fpl.py` explains why one adapter module produces four
  `Source`-conforming classes rather than one).
- **Access**: no auth, no published rate limit.
  - `https://fantasy.premierleague.com/api/bootstrap-static/` -- team roster (`teams`),
    player availability (`elements`), and the gameweek calendar (`events`). ~1.7MB;
    `FPLTeamsSource`, `FPLPlayersSource` and `FPLEventsSource` all fetch it
    independently, but the TTL cache means only the first of the three is a live
    request.
  - `https://fantasy.premierleague.com/api/fixtures/` -- the full season's fixture list
    in one response, played and unplayed.
- **Gameweek calendar**: `events[]` carries each gameweek's deadline and FPL's own
  `is_previous`/`is_current`/`is_next`/`finished` flags, landed as `stg_gameweeks`.
  `as_of_gameweek` on the forecast artifact reads `is_current` directly rather than
  being re-derived from which fixtures happen to be finished (story B-10) -- note FPL
  keeps a gameweek "current" for a while after it finishes, not just while it is live.
  The season label stamped on `stg_fixtures` is likewise derived from the fixture
  list's own earliest kickoff, not the wall clock, so a curate run in June or early
  July does not mislabel next season's fixtures with last season's.
- **Team roster and identifiers**: `teams[].id` is a 1-20 index re-assigned every
  season (alphabetical), so it is *not* a stable club key. `teams[].code` is stable
  across seasons (Arsenal 3, Aston Villa 7, Manchester United 1, ...) and is what
  `club_aliases.yaml` stores as `fpl_code`. Fixtures and player rows reference teams by
  the per-season `id`; curate maps them through the same snapshot's roster to `code`
  before resolving to canonical club IDs.
- **Player status codes**: exactly five are documented and observed --
  `a` (available), `d` (doubtful), `i` (injured), `s` (suspended), `u` (unavailable).
  The schema fails loudly (`isin` check) on any other code rather than passing it
  through, since an unrecognised code is real schema drift, not noise.
- **Player availability: landed, unconsumed** (story B-16). `raw_fpl_player_availability`
  is typed and validated at parse time but has no `stg_*` table built from it and no
  feature reads it; nothing in the model layer accounts for missing players yet.
  Landing it costs nothing extra (same bootstrap-static response as the team roster),
  so it stays landed for when a future feature needs it, rather than being dropped and
  re-added later.
- **Timezone**: `kickoff_time` is UTC and must stay timezone-aware end to end --
  pandera's `coerce=True` silently strips timezone info back to a naive datetime
  unless the schema pins it explicitly via `dtype_kwargs={"time_zone": "UTC"}`. Caught
  by a unit test; would otherwise have been a silent, hard-to-notice bug the first time
  kickoff times were compared or joined against anything else timestamped in UTC.
- **Cross-source check**: `curate` reconciles every finished FPL fixture against
  football-data.co.uk for the current season and fails on any scoreline disagreement;
  count gaps in either direction are logged (football-data lags FPL by up to a week).
  FPL is authoritative for the live season's results and kickoff times; football-data
  for history and odds.

## Understat

- **Responsibility**: team-level xG (design.md section 5.1) -- the input section 6.3
  says should drive strength estimation instead of goals, once the model ladder moves
  past the Poisson baseline.
- **Access**: via the `soccerdata` library (design.md section 11.1's anticipated path
  for scraped sources), not a bespoke scraper. One bulk request per season.
- **Scope: team-level only, not shot-level.** Shot-level events need one HTTP request
  per match (~380/season); a full 2015/16-2026/27 backfill would be several thousand
  requests against a scraped, ToS-sensitive site, for data nothing in the pipeline
  consumes yet. Revisit if a future notebook or model genuinely needs shot data.
- **Season input format: `season_code()` pair-code strings only, never bare integers.**
  Confirmed by hand against the live site: passing the bare integer `2021` (intending
  the 2021/22 season) is silently misinterpreted by soccerdata as the 2020/21 season
  instead, because the string "2021" is self-referentially ambiguous -- it reads as
  both a literal year and a "20-21" pair-code, and soccerdata resolves that collision in
  favour of the pair-code reading. A real backfill run against this exact bug landed
  3,840 rows instead of 4,220, with all of 2021/22 silently missing. Passing the full
  pair-code string (e.g. "2122") up front sidesteps the ambiguity rather than fixing it
  -- it was never ambiguous once both halves of the pair are given explicitly.
- **Team names are Understat's own spelling**. 29 of the 35 clubs in the window match
  football-data's spelling exactly; the six that differ are Manchester City, Manchester
  United, Newcastle United, Nottingham Forest, West Bromwich Albion and Wolverhampton
  Wanderers. All 35 are mapped in `club_aliases.yaml` (`understat_name`) and a test
  asserts every observed name resolves.
- **Refresh**: completed seasons come from soccerdata's cache; the current season is
  always fetched live, because soccerdata's cache has no TTL and a cached in-progress
  season would otherwise be served stale on every weekly run.
- **Rate limiting**: `Settings.request_delay_seconds` (the politeness delay `cached_get`
  applies to football-data and FPL) does not apply here -- soccerdata makes its own HTTP
  requests internally and manages its own pacing; this adapter never calls `cached_get`.
  The on/off cache is the only lever this adapter has, which is why completed seasons
  are reused from it rather than re-scraped on every run (story B-15).
- **Provenance**: `_meta.json` records one part per season batch, each with `from_cache`
  reflecting whether soccerdata's cache was *permitted* for that batch (not necessarily
  a confirmed hit -- soccerdata does not report that) and a pseudo-URL carrying the
  exact `soccerdata` version, since the library and its version are this source's actual
  provenance rather than a fixed endpoint.
- **Cross-source check**: home/away goals from Understat's own match records agree with
  football-data.co.uk's for the same fixtures, and every season lands exactly 380 rows
  (40 for the in-progress 2026/27), matching football-data's counts exactly.
- **PPDA (passes per defensive action): landed, unconsumed** (story B-16). `home_ppda`/
  `away_ppda` are kept on `raw_understat_team_match` because they cost nothing extra to
  land alongside xG, but no feature reads them yet. The raw values are not
  winsorised or capped at ingest -- a small-sample-size PPDA can be extreme (max
  observed: 193, a division-by-a-handful-of-actions artefact) -- so any future feature
  built on them must cap or winsorise first rather than feed the raw column straight
  into a rate estimate.

## ClubElo, Transfermarkt

Not yet built. See design.md section 5.1 for their intended responsibility.
