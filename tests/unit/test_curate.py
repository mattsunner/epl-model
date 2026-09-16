import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import polars as pl
import pytest

from plforecast.entities.clubs import load_club_dimension
from plforecast.storage.curate import (
    curate_fixtures,
    curate_matches,
    reconcile_current_season,
)

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
    # The roster maps this season's 1-3 ids onto the mini dimension's codes (1-3).
    roster = pl.DataFrame(
        {
            "fpl_team_id": [1, 2, 3],
            "fpl_code": [1, 2, 3],
            "filename": ["data/raw/fpl-teams/20260101T000000Z/data.parquet"] * 3,
        }
    )
    conn.register("_roster", roster.to_arrow())
    conn.execute("CREATE TABLE raw_fpl_teams AS SELECT * FROM _roster")
    conn.unregister("_roster")


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
        "bfech": None,
        "bfecd": None,
        "bfeca": None,
        "b365ch": 2.0,
        "b365cd": 3.4,
        "b365ca": 3.8,
        "maxch": None,
        "maxcd": None,
        "maxca": None,
        "avgch": 2.05,
        "avgcd": 3.35,
        "avgca": 3.75,
        "filename": "data/raw/football-data/20260101T000000Z/data.parquet",
    }
    row.update(overrides)
    return row


def _seed_two_row_odds_scenario(conn) -> None:
    _seed_raw_footballdata_matches(
        conn,
        [
            _base_match_row(),  # Pinnacle present -> benchmark is Pinnacle
            _base_match_row(
                home_team="Leeds",
                away_team="Watford",
                psch=None,
                pscd=None,
                psca=None,
                bfech=2.2,
                bfecd=3.4,
                bfeca=3.6,
            ),  # Pinnacle gone -> falls back to Betfair Exchange, not Avg
        ],
    )


def test_curate_matches_applies_the_benchmark_fallback_chain(conn, dimension):
    _seed_two_row_odds_scenario(conn)

    curate_matches(conn, dimension)
    result = conn.execute("SELECT * FROM stg_matches ORDER BY match_id").pl()

    arsenal = result.row(0, named=True)
    assert arsenal["benchmark_source"] == "pinnacle"
    assert arsenal["benchmark_home_odds"] == pytest.approx(2.1)

    leeds = result.row(1, named=True)
    assert leeds["benchmark_source"] == "betfair_exchange"
    assert leeds["benchmark_home_odds"] == pytest.approx(2.2)


def test_curate_matches_builds_long_odds_table(conn, dimension):
    _seed_two_row_odds_scenario(conn)

    curate_matches(conn, dimension)
    odds = conn.execute("SELECT * FROM stg_odds ORDER BY match_id, bookmaker").pl()

    by_match = {
        m: sorted(g["bookmaker"].to_list())
        for m, g in odds.group_by("match_id", maintain_order=True)
    }
    assert by_match[("2015-16-arsenal-chelsea",)] == ["bet365", "market_avg", "pinnacle"]
    assert by_match[("2015-16-leeds-watford",)] == ["bet365", "betfair_exchange", "market_avg"]
    assert (odds["price_type"] == "closing").all()


def test_curate_matches_benchmark_is_null_when_no_closing_price_exists(conn, dimension):
    _seed_raw_footballdata_matches(
        conn,
        [
            _base_match_row(
                **{
                    c: None
                    for c in (
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
                    )
                }
            )
        ],
    )

    curate_matches(conn, dimension)
    row = conn.execute("SELECT * FROM stg_matches").pl().row(0, named=True)

    assert row["benchmark_source"] is None
    assert row["benchmark_home_odds"] is None
    assert conn.execute("SELECT count(*) FROM stg_odds").fetchone()[0] == 0


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
    assert result["benchmark_home_odds"].item() == pytest.approx(1.9)


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
        "filename": "data/raw/fpl-fixtures/20260101T000000Z/data.parquet",
    }
    row.update(overrides)
    return row


def test_curate_fixtures_uses_only_the_latest_snapshot(conn, dimension):
    """Every FPL ingest lands the full fixture list again. Without selecting the latest
    snapshot, the second ingest would double every fixture_id and fail the unique check
    -- the exact failure a weekly refresh would have hit."""
    _seed_raw_fpl_fixtures(
        conn,
        [
            _base_fixture_row(
                finished=False,
                home_score=None,
                away_score=None,
                filename="data/raw/fpl-fixtures/20260101T000000Z/data.parquet",
            ),
            _base_fixture_row(
                finished=True,
                home_score=3,
                away_score=0,
                filename="data/raw/fpl-fixtures/20260108T000000Z/data.parquet",
            ),
        ],
    )

    curate_fixtures(conn, dimension)
    result = conn.execute("SELECT * FROM mart_fixtures").pl()

    assert result.height == 1
    assert result["finished"].item() is True
    assert result["home_goals"].item() == 3


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


def _seed_reconciliation_scenario(conn, dimension, *, fd_home_goals: int) -> None:
    season = "2026/27"
    _seed_raw_footballdata_matches(
        conn,
        [
            _base_match_row(
                season=season, date=date(2026, 8, 21), fthg=fd_home_goals, ftag=0, ftr="H"
            )
        ],
    )
    _seed_raw_fpl_fixtures(
        conn,
        [
            _base_fixture_row(),  # arsenal 3-0 chelsea, finished
            _base_fixture_row(
                fpl_fixture_id=2, home_team_id=3, away_team_id=2, home_score=1, away_score=1
            ),  # leeds 1-1 chelsea, finished, football-data has not published it yet
        ],
    )
    curate_matches(conn, dimension)
    curate_fixtures(conn, dimension)
    conn.execute(
        "UPDATE mart_fixtures SET season = ?", [season]
    )  # curate stamps season from the wall clock (story B-10); pin it for the test


def test_reconcile_current_season_reports_count_gaps(conn, dimension):
    _seed_reconciliation_scenario(conn, dimension, fd_home_goals=3)

    counts = reconcile_current_season(conn)

    assert counts == {"finished_in_both": 1, "fpl_only": 1, "football_data_only": 0}


def test_reconcile_current_season_fails_on_scoreline_disagreement(conn, dimension):
    _seed_reconciliation_scenario(conn, dimension, fd_home_goals=2)  # FPL says 3-0

    with pytest.raises(ValueError, match="disagree on scorelines"):
        reconcile_current_season(conn)
