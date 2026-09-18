import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import polars as pl
import pytest

from plforecast.entities.clubs import load_club_dimension
from plforecast.storage.curate import (
    build_curate_manifest,
    build_team_match,
    curate_club_dimension,
    curate_clubelo,
    curate_fixtures,
    curate_gameweeks,
    curate_matches,
    derive_season_from_kickoffs,
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
    result = conn.execute("SELECT * FROM stg_fixtures").pl()

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
    result = conn.execute("SELECT * FROM stg_fixtures ORDER BY fixture_id").pl()

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
        "UPDATE stg_fixtures SET season = ?", [season]
    )  # curate stamps season from the wall clock (story B-10); pin it for the test


def test_reconcile_current_season_reports_count_gaps(conn, dimension):
    _seed_reconciliation_scenario(conn, dimension, fd_home_goals=3)

    counts = reconcile_current_season(conn)

    assert counts == {"finished_in_both": 1, "fpl_only": 1, "football_data_only": 0}


def test_reconcile_current_season_fails_on_scoreline_disagreement(conn, dimension):
    _seed_reconciliation_scenario(conn, dimension, fd_home_goals=2)  # FPL says 3-0

    with pytest.raises(ValueError, match="disagree on scorelines"):
        reconcile_current_season(conn)


def _seed_raw_understat(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    df = pl.DataFrame(rows)
    conn.register("_seed_us", df.to_arrow())
    conn.execute("CREATE TABLE raw_understat_team_match AS SELECT * FROM _seed_us")
    conn.unregister("_seed_us")


def _understat_row(**overrides) -> dict:
    row = {
        "season": "2015/16",
        "date": date(2015, 8, 8),
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_goals": 2,
        "away_goals": 1,
        "home_xg": 1.9,
        "away_xg": 0.7,
        "home_np_xg": 1.1,
        "away_np_xg": 0.7,
        "home_ppda": 9.0,
        "away_ppda": 12.0,
        "filename": "data/raw/understat/20260101T000000Z/data.parquet",
    }
    row.update(overrides)
    return row


def test_curate_matches_joins_understat_xg_by_resolved_key(conn, dimension):
    _seed_raw_footballdata_matches(conn, [_base_match_row()])
    _seed_raw_understat(conn, [_understat_row()])

    curate_matches(conn, dimension)
    row = conn.execute("SELECT * FROM stg_matches").pl().row(0, named=True)

    assert row["home_xg"] == pytest.approx(1.9)
    assert row["away_np_xg"] == pytest.approx(0.7)


def test_curate_matches_fails_when_understat_scoreline_disagrees(conn, dimension):
    _seed_raw_footballdata_matches(conn, [_base_match_row()])
    _seed_raw_understat(conn, [_understat_row(home_goals=3)])

    with pytest.raises(ValueError, match="disagree on scorelines"):
        curate_matches(conn, dimension)


def test_curate_matches_fails_when_a_completed_season_match_lacks_xg(conn, dimension):
    _seed_raw_footballdata_matches(
        conn,
        [
            _base_match_row(),
            _base_match_row(season="2016/17", date=date(2016, 8, 13)),  # newest season
        ],
    )
    _seed_raw_understat(conn, [_understat_row(season="2016/17", date=date(2016, 8, 13))])

    with pytest.raises(ValueError, match="completed-season matches have no Understat xG"):
        curate_matches(conn, dimension)


def test_curate_matches_tolerates_missing_xg_only_in_the_current_season(conn, dimension):
    _seed_raw_footballdata_matches(
        conn,
        [_base_match_row(), _base_match_row(season="2016/17", date=date(2016, 8, 13))],
    )
    _seed_raw_understat(conn, [_understat_row()])  # nothing yet for 2016/17

    curate_matches(conn, dimension)
    result = conn.execute("SELECT season, home_xg FROM stg_matches ORDER BY season").pl()

    assert result["home_xg"].to_list()[0] == pytest.approx(1.9)
    assert result["home_xg"].to_list()[1] is None


def test_curate_matches_without_understat_snapshot_leaves_xg_null(conn, dimension):
    _seed_raw_footballdata_matches(conn, [_base_match_row()])

    curate_matches(conn, dimension)

    assert conn.execute("SELECT home_xg FROM stg_matches").fetchone()[0] is None


def test_curate_club_dimension_materialises_dim_club(conn, dimension):
    curate_club_dimension(conn, dimension)
    rows = conn.execute("SELECT club_id, fpl_code FROM dim_club ORDER BY club_id").fetchall()
    assert rows == [("arsenal", 1), ("chelsea", 2), ("leeds", 3), ("watford", None)]


def test_build_team_match_has_two_rows_per_match_with_rest_days():
    matches = pl.DataFrame(
        [
            {
                "match_id": "2015-16-arsenal-chelsea",
                "season": "2015/16",
                "date": date(2015, 8, 8),
                "home_club_id": "arsenal",
                "away_club_id": "chelsea",
                "home_goals": 2,
                "away_goals": 1,
                "home_xg": 1.9,
                "away_xg": 0.7,
                "home_np_xg": 1.1,
                "away_np_xg": 0.7,
            },
            {
                "match_id": "2015-16-chelsea-arsenal",
                "season": "2015/16",
                "date": date(2015, 8, 15),
                "home_club_id": "chelsea",
                "away_club_id": "arsenal",
                "home_goals": 0,
                "away_goals": 0,
                "home_xg": 1.0,
                "away_xg": 1.2,
                "home_np_xg": 1.0,
                "away_np_xg": 1.2,
            },
        ]
    )

    team_match = build_team_match(matches)

    assert team_match.height == 4
    arsenal = team_match.filter(pl.col("club_id") == "arsenal").sort("date")
    assert arsenal["points"].to_list() == [3, 1]
    assert arsenal["result"].to_list() == ["W", "D"]
    assert arsenal["rest_days"].to_list() == [None, 7]
    assert arsenal["xg_for"].to_list() == pytest.approx([1.9, 1.2])
    assert arsenal["is_home"].to_list() == [True, False]


def test_curate_matches_joins_xg_across_a_one_day_timestamp_gap(conn, dimension):
    """Understat dates Monday-night matches in 2015/16 and 2016/17 on the next day."""
    _seed_raw_footballdata_matches(conn, [_base_match_row()])
    _seed_raw_understat(conn, [_understat_row(date=date(2015, 8, 9))])

    curate_matches(conn, dimension)

    assert conn.execute("SELECT home_xg FROM stg_matches").fetchone()[0] == pytest.approx(1.9)


def test_curate_matches_rejects_xg_with_a_large_date_gap(conn, dimension):
    _seed_raw_footballdata_matches(conn, [_base_match_row()])
    _seed_raw_understat(conn, [_understat_row(date=date(2015, 9, 1))])

    with pytest.raises(ValueError, match="differ from football-data by more than a day"):
        curate_matches(conn, dimension)


# ---- B-10: season derived from data, not the wall clock ----


def test_derive_season_from_kickoffs_uses_the_earliest_known_kickoff():
    fixtures = pl.DataFrame(
        {
            "kickoff_time": [
                datetime(2026, 8, 21, 15, 0, tzinfo=ZoneInfo("UTC")),
                datetime(2026, 9, 1, 15, 0, tzinfo=ZoneInfo("UTC")),
            ]
        }
    )
    assert derive_season_from_kickoffs(fixtures) == "2026/27"


def test_derive_season_from_kickoffs_handles_the_july_rollover_from_data_not_today():
    # A fixture list whose earliest known kickoff is in June still belongs to the
    # *previous* season by the July convention -- and this must not depend on what
    # today's wall-clock date happens to be when curate runs.
    fixtures = pl.DataFrame({"kickoff_time": [datetime(2026, 6, 1, 15, 0, tzinfo=ZoneInfo("UTC"))]})
    assert derive_season_from_kickoffs(fixtures) == "2025/26"


def test_derive_season_from_kickoffs_falls_back_when_every_kickoff_is_null():
    fixtures = pl.DataFrame(
        {"kickoff_time": pl.Series([None, None], dtype=pl.Datetime("us", "UTC"))}
    )
    assert derive_season_from_kickoffs(fixtures, fallback_today=date(2026, 9, 16)) == "2026/27"


def test_curate_fixtures_stamps_season_from_kickoffs_not_settings_wall_clock(conn, dimension):
    """Regression: settings.current_season_start_year() (the wall clock) must not be
    consulted at all once real kickoff data exists, however far it disagrees with it."""
    _seed_raw_fpl_fixtures(
        conn,
        [
            _base_fixture_row(
                kickoff_time=datetime(2019, 8, 9, 15, 0, tzinfo=ZoneInfo("UTC")),
            )
        ],
    )
    curate_fixtures(conn, dimension)
    result = conn.execute("SELECT season FROM stg_fixtures").pl()
    assert result["season"].to_list() == ["2019/20"]


# ---- B-10: FPL's own gameweek calendar ----


def _seed_raw_fpl_events(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    df = pl.DataFrame(rows)
    conn.register("_seed_events", df.to_arrow())
    conn.execute("CREATE TABLE raw_fpl_events AS SELECT * FROM _seed_events")
    conn.unregister("_seed_events")


def _event_row(**overrides) -> dict:
    row = {
        "gameweek": 1,
        "name": "Gameweek 1",
        "deadline_time": datetime(2026, 8, 21, 17, 30, tzinfo=ZoneInfo("UTC")),
        "finished": True,
        "is_previous": False,
        "is_current": False,
        "is_next": False,
        "filename": "data/raw/fpl-events/20260101T000000Z/data.parquet",
    }
    row.update(overrides)
    return row


def test_curate_gameweeks_materialises_stg_gameweeks(conn):
    _seed_raw_fpl_events(
        conn,
        [
            _event_row(gameweek=3, is_previous=True),
            _event_row(gameweek=4, is_current=True),
            _event_row(gameweek=5, finished=False, is_next=True),
        ],
    )

    curate_gameweeks(conn)
    result = conn.execute("SELECT * FROM stg_gameweeks ORDER BY gameweek").pl()

    assert result["gameweek"].to_list() == [3, 4, 5]
    assert result.filter(pl.col("is_current"))["gameweek"].to_list() == [4]


def test_curate_gameweeks_skips_without_raising_when_no_snapshot_exists(conn):
    curate_gameweeks(conn)  # no raw_fpl_events table at all
    tables = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'stg_gameweeks'"
    ).fetchone()
    assert tables[0] == 0


def _seed_raw_clubelo_ratings(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    df = pl.DataFrame(rows)
    conn.register("_seed_clubelo", df.to_arrow())
    conn.execute("CREATE TABLE raw_clubelo_ratings AS SELECT * FROM _seed_clubelo")
    conn.unregister("_seed_clubelo")


def _clubelo_row(**overrides) -> dict:
    row = {
        "club_id": "arsenal",
        "date": date(2026, 9, 1),
        "elo": 1900.0,
        "golo": 1.4,
        "filename": "data/raw/clubelo/20260918T031725Z/data.parquet",
    }
    row.update(overrides)
    return row


def test_curate_clubelo_materialises_stg_clubelo(conn):
    _seed_raw_clubelo_ratings(
        conn,
        [
            _clubelo_row(club_id="arsenal", date=date(2026, 9, 1), elo=1900.0),
            _clubelo_row(club_id="coventry", date=date(2026, 9, 5), elo=1650.0, golo=None),
        ],
    )

    curate_clubelo(conn)
    result = conn.execute("SELECT * FROM stg_clubelo ORDER BY club_id").pl()

    assert result["club_id"].to_list() == ["arsenal", "coventry"]
    assert result["elo"].to_list() == [1900.0, 1650.0]
    assert result.filter(pl.col("club_id") == "coventry")["golo"].item() is None


def test_curate_clubelo_skips_without_raising_when_no_snapshot_exists(conn):
    curate_clubelo(conn)  # no raw_clubelo_ratings table at all
    tables = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'stg_clubelo'"
    ).fetchone()
    assert tables[0] == 0


# ---- B-11: lineage (curated_at column, manifest) ----


def test_curated_at_is_stamped_on_every_curated_table_and_shared_across_a_run(conn, dimension):
    _seed_raw_footballdata_matches(conn, [_base_match_row()])
    _seed_raw_fpl_fixtures(conn, [_base_fixture_row()])
    _seed_raw_fpl_events(conn, [_event_row()])

    curated_at = datetime(2026, 9, 16, 12, 0, tzinfo=ZoneInfo("UTC"))
    curate_club_dimension(conn, dimension, curated_at=curated_at)
    curate_matches(conn, dimension, curated_at=curated_at)
    curate_fixtures(conn, dimension, curated_at=curated_at)
    curate_gameweeks(conn, curated_at=curated_at)

    for table in ("dim_club", "stg_matches", "stg_fixtures", "stg_gameweeks"):
        values = (
            conn.execute(f"SELECT DISTINCT curated_at FROM {table}").pl()["curated_at"].to_list()
        )
        assert values == [curated_at], f"{table} curated_at mismatch: {values}"


def test_materialize_defaults_curated_at_to_now_when_called_standalone(conn, dimension):
    curate_club_dimension(conn, dimension)  # no curated_at given
    value = conn.execute("SELECT curated_at FROM dim_club LIMIT 1").pl()["curated_at"].item()
    assert value is not None


def test_build_curate_manifest_names_every_table_its_row_count_and_its_raw_snapshots(
    conn, dimension
):
    _seed_raw_footballdata_matches(conn, [_base_match_row()])
    _seed_raw_fpl_fixtures(conn, [_base_fixture_row()])

    curated_at = datetime(2026, 9, 16, 12, 0, tzinfo=ZoneInfo("UTC"))
    curate_club_dimension(conn, dimension, curated_at=curated_at)
    curate_matches(conn, dimension, curated_at=curated_at)
    curate_fixtures(conn, dimension, curated_at=curated_at)

    manifest = build_curate_manifest(conn, curated_at=curated_at)

    assert manifest["curated_at"] == curated_at.isoformat()
    assert manifest["tables"]["dim_club"] == {"rows": 4, "sources": {"club_aliases.yaml": None}}
    stg_matches = manifest["tables"]["stg_matches"]
    assert stg_matches["rows"] == 1
    assert stg_matches["sources"]["raw_footballdata_matches"] == "20260101T000000Z"
    assert stg_matches["sources"]["raw_understat_team_match"] is None  # not ingested
    assert "stg_gameweeks" not in manifest["tables"]  # never curated in this test
