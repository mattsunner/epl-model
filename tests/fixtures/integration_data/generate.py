"""Generator for tests/fixtures/integration_data/raw/. Run from the repo root as
`uv run python tests/fixtures/integration_data/generate.py`; its output is committed,
not generated at test time, so the fixture data is inspectable in the repo and the
integration test (tests/integration/test_pipeline.py) stays fast. Re-run this whenever
a parsed raw schema (MatchSchema, TeamMatchXGSchema, TeamSchema, FixtureSchema,
EventSchema) changes shape."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from plforecast.ingest.footballdata import MatchSchema
from plforecast.ingest.fpl import EventSchema, FixtureSchema, TeamSchema
from plforecast.ingest.understat import TeamMatchXGSchema

OUT = Path(__file__).parent / "raw"
CLUBS_FD = ["Arsenal", "Chelsea", "Liverpool", "Man City"]
CLUBS_US = ["Arsenal", "Chelsea", "Liverpool", "Manchester City"]  # Understat's own spelling
FPL_CODE = {"Arsenal": 3, "Chelsea": 8, "Liverpool": 14, "Man City": 43}
FPL_TEAM_ID = {"Arsenal": 1, "Chelsea": 2, "Liverpool": 3, "Man City": 4}


def _write(df: pl.DataFrame, source: str, snapshot: str) -> None:
    d = OUT / source / snapshot
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "data.parquet")
    print(f"wrote {d / 'data.parquet'} ({df.height} rows)")


def round_robin(clubs: list[str]) -> list[tuple[str, str]]:
    return [(h, a) for h in clubs for a in clubs if h != a]


def football_data_2020_21() -> pl.DataFrame:
    rows = []
    start = date(2020, 9, 12)
    for i, (home, away) in enumerate(round_robin(CLUBS_FD)):
        d = start + timedelta(days=7 * i)
        hg, ag = (2, 1) if i % 3 else (1, 1)
        row = {
            "season": "2020/21",
            "date": d,
            "home_team": home,
            "away_team": away,
            "fthg": hg,
            "ftag": ag,
            "ftr": "H" if hg > ag else ("A" if hg < ag else "D"),
        }
        for col in (
            "psch",
            "pscd",
            "psca",
            "bfech",
            "bfecd",
            "bfeca",
            "b365ch",
            "b365cd",
            "b365ca",
            "maxch",
            "maxcd",
            "maxca",
            "avgch",
            "avgcd",
            "avgca",
        ):
            row[col] = None
        row["psch"], row["pscd"], row["psca"] = 2.1, 3.4, 3.2
        rows.append(row)
    df = pl.DataFrame(rows)
    return MatchSchema.validate(df)


def understat_2020_21() -> pl.DataFrame:
    rows = []
    start = date(2020, 9, 12)
    fd_rows = round_robin(CLUBS_FD)
    us_rows = round_robin(CLUBS_US)
    for i, (_pair_fd, (home_us, away_us)) in enumerate(zip(fd_rows, us_rows, strict=True)):
        d = start + timedelta(days=7 * i)
        hg, ag = (2, 1) if i % 3 else (1, 1)
        rows.append(
            {
                "season": "2020/21",
                "date": d,
                "home_team": home_us,
                "away_team": away_us,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": float(hg) + 0.3,
                "away_xg": float(ag) + 0.2,
                "home_np_xg": float(hg),
                "away_np_xg": float(ag),
                "home_ppda": 9.5,
                "away_ppda": 11.2,
            }
        )
    df = pl.DataFrame(rows)
    return TeamMatchXGSchema.validate(df)


def fpl_teams() -> pl.DataFrame:
    rows = [
        {
            "fpl_team_id": FPL_TEAM_ID[c],
            "fpl_code": FPL_CODE[c],
            "name": c,
            "short_name": c[:3].upper(),
        }
        for c in CLUBS_FD
    ]
    return TeamSchema.validate(pl.DataFrame(rows))


def fpl_fixtures_2021_22() -> pl.DataFrame:
    rows = []
    start = datetime(2021, 8, 14, 15, 0, tzinfo=UTC)
    pairs = round_robin(CLUBS_FD)
    for i, (home, away) in enumerate(pairs):
        gw = i // 2 + 1
        kickoff = start + timedelta(days=7 * (gw - 1))
        finished = gw <= 2
        rows.append(
            {
                "fpl_fixture_id": i + 1,
                "gameweek": gw,
                "kickoff_time": kickoff,
                "home_team_id": FPL_TEAM_ID[home],
                "away_team_id": FPL_TEAM_ID[away],
                "home_score": 2 if finished else None,
                "away_score": 1 if finished else None,
                "finished": finished,
            }
        )
    df = pl.DataFrame(rows, schema_overrides={"home_score": pl.Int64, "away_score": pl.Int64})
    return FixtureSchema.validate(df)


def fpl_events_2021_22() -> pl.DataFrame:
    rows = []
    start = datetime(2021, 8, 20, 17, 30, tzinfo=UTC)
    for gw in range(1, 7):
        rows.append(
            {
                "gameweek": gw,
                "name": f"Gameweek {gw}",
                "deadline_time": start + timedelta(days=7 * (gw - 1)),
                "finished": gw <= 2,
                "is_previous": gw == 2,
                "is_current": gw == 3,
                "is_next": gw == 4,
            }
        )
    return EventSchema.validate(pl.DataFrame(rows))


football_data_2020_21_df = football_data_2020_21()
_write(football_data_2020_21_df, "football-data", "20200912T000000Z")
_write(understat_2020_21(), "understat", "20200912T000100Z")
_write(fpl_teams(), "fpl-teams", "20210814T000000Z")
_write(fpl_fixtures_2021_22(), "fpl-fixtures", "20210814T000100Z")
_write(fpl_events_2021_22(), "fpl-events", "20210814T000200Z")
print("done")
