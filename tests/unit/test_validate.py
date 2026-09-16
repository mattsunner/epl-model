from datetime import date, timedelta

import duckdb
import polars as pl
import pytest

from plforecast.storage.validate import (
    ValidationFailed,
    check_closing_odds_coverage,
    check_club_ids_resolve,
    check_current_season_reconciliation,
    check_no_self_fixtures,
    check_result_matches_score,
    check_season_shape,
    check_unique_ids,
    check_xg_coverage,
    run_validations,
)


@pytest.fixture
def conn():
    connection = duckdb.connect(":memory:")
    yield connection
    connection.close()


def _seed(conn: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame) -> None:
    conn.register("_seed", df.to_arrow())
    conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _seed")
    conn.unregister("_seed")


def _round_robin(clubs: list[str], season: str, start: date) -> pl.DataFrame:
    rows = []
    day = 0
    for home in clubs:
        for away in clubs:
            if home == away:
                continue
            rows.append(
                {
                    "match_id": f"{season}-{home}-{away}",
                    "season": season,
                    "date": start + timedelta(days=day),
                    "home_club_id": home,
                    "away_club_id": away,
                    "home_goals": 1,
                    "away_goals": 0,
                    "result": "H",
                    "home_xg": 1.2,
                    "away_xg": 0.8,
                    "benchmark_source": "pinnacle",
                }
            )
            day += 1
    return pl.DataFrame(rows)


def _seed_stg_matches(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> None:
    _seed(conn, "stg_matches", df)


CLUBS = ["a", "b", "c", "d"]


def test_check_season_shape_passes_for_a_clean_round_robin(conn):
    completed = _round_robin(CLUBS, "2020/21", date(2020, 8, 1))
    current = _round_robin(CLUBS, "2021/22", date(2021, 8, 1)).head(2)  # in progress, incomplete
    _seed_stg_matches(conn, pl.concat([completed, current]))

    result = check_season_shape(conn)

    assert result.ok, result.detail
    assert "2020/21" not in result.detail.replace(
        "1 completed", ""
    )  # just sanity it's the summary form
    assert "clean round robins" in result.detail


def test_check_season_shape_fails_when_a_completed_season_is_missing_a_match(conn):
    completed = _round_robin(CLUBS, "2020/21", date(2020, 8, 1)).slice(1)  # drop the first match
    current = _round_robin(CLUBS, "2021/22", date(2021, 8, 1)).head(2)  # in progress, ignored
    _seed_stg_matches(conn, pl.concat([completed, current]))

    result = check_season_shape(conn)

    assert not result.ok
    assert "2020/21" in result.detail


def test_check_unique_ids_detects_a_duplicated_match_id(conn):
    df = _round_robin(CLUBS, "2020/21", date(2020, 8, 1))
    dup = df.head(1).with_columns(pl.col("home_goals").cast(pl.Int64) + 1)  # same match_id
    _seed_stg_matches(conn, pl.concat([df, dup]))

    result = check_unique_ids(conn)

    assert not result.ok
    assert "match_id" in result.detail


def test_check_no_self_fixtures_detects_a_club_playing_itself(conn):
    df = _round_robin(CLUBS, "2020/21", date(2020, 8, 1))
    bad = df.head(1).with_columns(pl.col("home_club_id").alias("away_club_id"))
    _seed_stg_matches(conn, pl.concat([df, bad]))

    result = check_no_self_fixtures(conn)

    assert not result.ok
    assert "stg_matches" in result.detail


def test_check_result_matches_score_detects_disagreement(conn):
    df = _round_robin(CLUBS, "2020/21", date(2020, 8, 1))
    bad = df.head(1).with_columns(pl.lit("A").alias("result"))  # home_goals=1 > away_goals=0, not A
    _seed_stg_matches(conn, pl.concat([df.slice(1), bad]))

    result = check_result_matches_score(conn)

    assert not result.ok
    assert "1 row" in result.detail


def test_check_xg_coverage_flags_a_missing_completed_season_value(conn):
    completed = _round_robin(CLUBS, "2020/21", date(2020, 8, 1))
    bad = completed.head(1).with_columns(pl.lit(None, dtype=pl.Float64).alias("home_xg"))
    current = _round_robin(CLUBS, "2021/22", date(2021, 8, 1)).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("home_xg")  # current season may lag, must not fail
    )
    _seed_stg_matches(conn, pl.concat([completed.slice(1), bad, current]))

    result = check_xg_coverage(conn)

    assert not result.ok
    assert "1 completed-season" in result.detail


def test_check_closing_odds_coverage_reports_but_never_fails(conn):
    df = _round_robin(CLUBS, "2020/21", date(2020, 8, 1))
    missing = df.head(1).with_columns(pl.lit(None, dtype=pl.Utf8).alias("benchmark_source"))
    _seed_stg_matches(conn, pl.concat([df.slice(1), missing]))

    result = check_closing_odds_coverage(conn)

    assert result.ok  # informational only
    assert "no benchmark price" in result.detail


def test_check_club_ids_resolve_detects_an_unknown_club(conn):
    _seed(conn, "dim_club", pl.DataFrame({"club_id": CLUBS}))
    df = _round_robin([*CLUBS, "ghost"], "2020/21", date(2020, 8, 1))
    _seed_stg_matches(conn, df)

    result = check_club_ids_resolve(conn)

    assert not result.ok
    assert "ghost" in result.detail


def test_check_club_ids_resolve_passes_when_every_id_is_known(conn):
    _seed(conn, "dim_club", pl.DataFrame({"club_id": CLUBS}))
    _seed_stg_matches(conn, _round_robin(CLUBS, "2020/21", date(2020, 8, 1)))

    result = check_club_ids_resolve(conn)

    assert result.ok


def test_check_current_season_reconciliation_reports_when_tables_absent(conn):
    result = check_current_season_reconciliation(conn)
    assert result.ok


def test_run_validations_raises_naming_every_failed_check(conn):
    df = _round_robin(CLUBS, "2020/21", date(2020, 8, 1)).slice(1)  # missing shape
    bad_result = df.head(1).with_columns(pl.lit("A").alias("result"))  # bad result too
    current = _round_robin(CLUBS, "2021/22", date(2021, 8, 1)).head(2)  # in progress, ignored
    _seed_stg_matches(conn, pl.concat([df.slice(1), bad_result, current]))

    with pytest.raises(ValidationFailed) as exc_info:
        run_validations(conn, raise_on_failure=True)

    message = str(exc_info.value)
    assert "season_shape" in message
    assert "result_matches_score" in message


def test_run_validations_does_not_raise_by_default(conn):
    df = _round_robin(CLUBS, "2020/21", date(2020, 8, 1)).slice(1)  # missing shape, no raise arg
    current = _round_robin(CLUBS, "2021/22", date(2021, 8, 1)).head(2)  # in progress, ignored
    _seed_stg_matches(conn, pl.concat([df, current]))

    results = run_validations(conn)  # raise_on_failure defaults to False

    assert any(not r.ok for r in results)
