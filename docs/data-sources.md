# Data sources

Inventory of external sources the pipeline ingests. See `design.md` section 5 for the
ingestion contract every adapter implements.

## football-data.co.uk

- **Responsibility**: match results and Pinnacle closing odds (`PSCH`/`PSCD`/`PSCA`), E0
  (Premier League) division.
- **Access**: one CSV per season at
  `https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv`, where `season_code` is
  the two two-digit season-start/end years, e.g. `1516` for 2015/16. No auth, no rate
  limit published; the adapter applies a politeness delay and a TTL cache regardless.
- **Coverage used**: 2015/16 onward, matching the backtest window in design.md section
  8.2 (Understat xG does not exist before 2014/15). Full history back to 1993 is
  available at the source if the backtest window is ever extended earlier.
- **Column drift**: the core columns used here (date, teams, full-time score, closing
  odds) are stable across seasons. The wider bookmaker-odds column set is not -- it
  changes shape every season as individual bookmakers stop or start reporting to the
  site. The adapter selects only the stable core at parse time rather than carrying the
  full raw column set into `data/raw/`.
- **Unplayed fixtures**: the season CSV only contains played matches. Remaining-fixture
  data (for the simulation engine) comes from the FPL API, not this source.
- **License / attribution**: free for personal/non-commercial use per the site's terms;
  attribute football-data.co.uk when publishing derived results.

## FPL API

- **Responsibility**: fixture list, kickoff times, player availability and suspensions.
  One physical source backs three raw tables (`src/plforecast/ingest/fpl.py` explains
  why one adapter module produces three `Source`-conforming classes rather than one).
- **Access**: no auth, no published rate limit.
  - `https://fantasy.premierleague.com/api/bootstrap-static/` -- team roster (`teams`)
    and player availability (`elements`). ~1.7MB; both `FPLTeamsSource` and
    `FPLPlayersSource` fetch it independently, but the TTL cache means only the first
    of the two is a live request.
  - `https://fantasy.premierleague.com/api/fixtures/` -- the full season's fixture list
    in one response, played and unplayed.
- **Team roster**: not one of the design's stated FPL responsibilities, but it is the
  source of the `fpl_team_id` column the club dimension needs (design.md section 5.3),
  and it is free to land alongside the players endpoint. Fixtures and player rows keep
  FPL's raw numeric team IDs; resolving them to canonical club IDs is the entities
  layer's job, not ingest's.
- **Player status codes**: exactly five are documented and observed --
  `a` (available), `d` (doubtful), `i` (injured), `s` (suspended), `u` (unavailable).
  The schema fails loudly (`isin` check) on any other code rather than passing it
  through, since an unrecognised code is real schema drift, not noise.
- **Timezone**: `kickoff_time` is UTC and must stay timezone-aware end to end --
  pandera's `coerce=True` silently strips timezone info back to a naive datetime
  unless the schema pins it explicitly via `dtype_kwargs={"time_zone": "UTC"}`. Caught
  by a unit test; would otherwise have been a silent, hard-to-notice bug the first time
  kickoff times were compared or joined against anything else timestamped in UTC.
- **Cross-source check**: at the time of writing, FPL reports 40 finished fixtures for
  2026/27 and football-data.co.uk reports 40 played E0 rows for the same season --
  a useful sanity signal that both sources agree on match count.

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
- **Team names are Understat's own spelling** (e.g. "Newcastle United" vs
  football-data's "Newcastle") -- resolving them to canonical club IDs is the entities
  layer's job, not ingest's. `club_aliases.yaml`'s `understat_name` column is still
  unpopulated.
- **Cross-source check**: home/away goals from Understat's own match records agree with
  football-data.co.uk's for the same fixtures, and every season lands exactly 380 rows
  (40 for the in-progress 2026/27), matching football-data's counts exactly.

## ClubElo, Transfermarkt

Not yet built. See design.md section 5.1 for their intended responsibility.
