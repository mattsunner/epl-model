import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import polars as pl
import pytest

from plforecast.entities.clubs import load_club_dimension
from plforecast.storage.curate import curate_fixtures, curate_matches

MINI_ALIASES = Path(__file__).parent.parent / "fixtures" / "entities" / "mini_club_aliases.yaml"


@pytest.fixture
def dimension():
    return load_club_dimension(MINI_ALIASES)


@pytest.fixture
def conn():
    connection = duckdb.connect(":memory:")
    yield connection
    connection.close()


def _seed_raw_footballdata_matches(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    df = pl.DataFrame(rows)
    conn.register("_seed", df.to_arrow())
    conn.execute("CREATE TABLE raw_footballdata_matches AS SELECT * FROM _seed")
    conn.unregister("_seed")


def _seed_raw_fpl_fixtures(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    df = pl.DataFrame(rows)
    conn.register("_seed", df.to_arrow())
    conn.execute("CREATE TABLE raw_fpl_fixtures AS SELECT * FROM _seed")
    conn.unregister("_seed")


def _base_match_row(**overrides) -> dict:
    row = {
        "season": "2015/16",
        "date": date(2015, 8, 8),
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "fthg": 2,
        "ftag": 1,
        "ftr": "H",
        "psch": 2.1,
        "pscd": 3.3,
        "psca": 3.7,
        "filename": "data/raw/football-data/20260101T000000Z/data.parquet",
    }
    row.update(overrides)
    return row


def test_curate_matches_keeps_most_recent_snapshot_per_natural_key(conn, dimension):
    """Re-ingesting football-data lands a brand new full-backfill snapshot every time,
    so the same match can appear more than once in raw_footballdata_matches. The most
    recently landed copy (by filename) must win, not an arbitrary one."""
    _seed_raw_footballdata_matches(
        conn,
        [
            _base_match_row(
                psch=2.1,
                filename="data/raw/football-data/20260101T000000Z/data.parquet",
            ),
            _base_match_row(
                psch=1.9,  # odds updated on the later re-ingest
                filename="data/raw/football-data/20260201T000000Z/data.parquet",
            ),
        ],
    )

    curate_matches(conn, dimension)
    result = conn.execute("SELECT * FROM stg_matches").pl()

    assert result.height == 1
    assert result["pinnacle_home_odds"].item() == pytest.approx(1.9)


def test_curate_matches_resolves_club_ids_and_builds_match_id(conn, dimension):
    _seed_raw_footballdata_matches(conn, [_base_match_row()])

    curate_matches(conn, dimension)
    result = conn.execute("SELECT * FROM stg_matches").pl()

    row = result.row(0, named=True)
    assert row["home_club_id"] == "arsenal"
    assert row["away_club_id"] == "chelsea"
    assert row["match_id"] == "2015-16-arsenal-chelsea"
    assert row["home_goals"] == 2
    assert row["away_goals"] == 1


def test_curate_matches_rejects_result_inconsistent_with_score(conn, dimension):
    _seed_raw_footballdata_matches(conn, [_base_match_row(ftr="A")])  # 2-1 is not an away win

    with pytest.raises(ValueError, match="disagrees with the scoreline"):
        curate_matches(conn, dimension)


def _base_fixture_row(**overrides) -> dict:
    row = {
        "fpl_fixture_id": 1,
        "gameweek": 1,
        "kickoff_time": datetime(2026, 8, 21, 19, 0, tzinfo=ZoneInfo("UTC")),
        "home_team_id": 1,
        "away_team_id": 2,
        "home_score": 3,
        "away_score": 0,
        "finished": True,
    }
    row.update(overrides)
    return row


def test_curate_fixtures_resolves_club_ids_and_stamps_season(conn, dimension):
    _seed_raw_fpl_fixtures(
        conn,
        [
            _base_fixture_row(),
            _base_fixture_row(
                fpl_fixture_id=41,
                gameweek=5,
                home_team_id=3,
                away_team_id=2,
                home_score=None,
                away_score=None,
                finished=False,
            ),
        ],
    )

    curate_fixtures(conn, dimension)
    result = conn.execute("SELECT * FROM mart_fixtures ORDER BY fixture_id").pl()

    assert result.height == 2
    played = result.row(0, named=True)
    assert played["home_club_id"] == "arsenal"
    assert played["away_club_id"] == "chelsea"
    assert played["home_goals"] == 3
    assert played["finished"] is True

    unplayed = result.row(1, named=True)
    assert unplayed["home_club_id"] == "leeds"
    assert unplayed["home_goals"] is None
    assert unplayed["finished"] is False

    # season is stamped from today's date, not fixed -- just check it's one consistent,
    # correctly-shaped value across every row.
    seasons = set(result["season"].to_list())
    assert len(seasons) == 1
    assert re.match(r"^\d{4}/\d{2}$", seasons.pop())
