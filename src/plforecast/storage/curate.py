"""Raw -> curated transforms (design.md section 5.4). Reads from the raw_* views,
resolves club identity, and materialises stg_*/mart_* tables.

- stg_club_season: which clubs were in the Premier League each season.
- stg_matches: played match results (football-data.co.uk), club IDs resolved, one row
  per natural key -- deduplicated across every re-ingested snapshot, keeping the most
  recently landed copy of each match.
- mart_fixtures: the full current-season fixture list (FPL), club IDs resolved. This is
  what the simulation engine reads remaining fixtures from.

mart_matches (a model-ready superset of stg_matches, per design.md's repo layout) and
mart_odds (a normalised, multi-bookmaker odds table) are deliberately not built yet --
there is only one odds source landed so far, and stg_matches already carries it at
match grain, which is enough until a second source makes a separate table worth it.
"""

from __future__ import annotations

import duckdb
import pandera.polars as pa
import polars as pl
import structlog
from pandera.typing.polars import Series

from plforecast.config import season_label, settings
from plforecast.entities.clubs import ClubDimension, load_club_dimension
from plforecast.entities.competitions import build_club_season_membership

log = structlog.get_logger()


class StgMatchSchema(pa.DataFrameModel):
    match_id: Series[str] = pa.Field(unique=True)
    season: Series[str] = pa.Field(str_matches=r"^\d{4}/\d{2}$")
    date: Series[pl.Date]
    home_club_id: Series[str]
    away_club_id: Series[str]
    home_goals: Series[int] = pa.Field(ge=0)
    away_goals: Series[int] = pa.Field(ge=0)
    result: Series[str] = pa.Field(isin=["H", "D", "A"])
    pinnacle_home_odds: Series[float] = pa.Field(nullable=True, ge=1.0)
    pinnacle_draw_odds: Series[float] = pa.Field(nullable=True, ge=1.0)
    pinnacle_away_odds: Series[float] = pa.Field(nullable=True, ge=1.0)

    class Config:
        strict = True
        coerce = True


def _assert_no_self_fixtures(df: pl.DataFrame) -> None:
    bad = df.filter(pl.col("home_club_id") == pl.col("away_club_id"))
    if bad.height:
        raise ValueError(f"a club cannot play itself, but found: {bad}")


def _assert_result_matches_score(df: pl.DataFrame) -> None:
    implied = (
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
    )
    bad = df.filter(implied != pl.col("result"))
    if bad.height:
        raise ValueError(f"result column disagrees with the scoreline for rows: {bad}")


def _materialize(conn: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame) -> None:
    staging_name = f"_incoming_{table}"
    conn.register(staging_name, df.to_arrow())
    conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM {staging_name}")
    conn.unregister(staging_name)


class MartFixtureSchema(pa.DataFrameModel):
    fixture_id: Series[int] = pa.Field(ge=1, unique=True)
    season: Series[str] = pa.Field(str_matches=r"^\d{4}/\d{2}$")
    gameweek: Series[int] = pa.Field(ge=1, nullable=True)
    kickoff_time: Series[pl.Datetime] = pa.Field(nullable=True, dtype_kwargs={"time_zone": "UTC"})
    home_club_id: Series[str]
    away_club_id: Series[str]
    home_goals: Series[int] = pa.Field(ge=0, nullable=True)
    away_goals: Series[int] = pa.Field(ge=0, nullable=True)
    finished: Series[bool]

    class Config:
        strict = True
        coerce = True


def curate_club_season_membership(conn: duckdb.DuckDBPyConnection) -> None:
    matches = conn.execute("SELECT season, home_team, away_team FROM raw_footballdata_matches").pl()
    dimension = load_club_dimension()
    membership = build_club_season_membership(matches, dimension)

    _materialize(conn, "stg_club_season", membership)
    log.info("curate.club_season_membership", rows=membership.height)


def curate_matches(conn: duckdb.DuckDBPyConnection, dimension: ClubDimension | None = None) -> None:
    """One row per (season, home team, away team) -- the natural key for a football-data
    row, since each pairing plays at a given ground exactly once a season. A re-ingest
    lands a brand new full-backfill snapshot every time (design.md section 5.2), so
    raw_footballdata_matches accumulates a duplicate copy of every already-played match
    on each re-run; picking the most recently landed snapshot per natural key is what
    keeps this a one-row-per-match table rather than a growing pile of duplicates."""
    dimension = dimension or load_club_dimension()
    deduped = conn.execute(
        """
        SELECT season, date, home_team, away_team, fthg, ftag, ftr, psch, pscd, psca
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY season, home_team, away_team ORDER BY filename DESC
            ) AS rn
            FROM raw_footballdata_matches
        )
        WHERE rn = 1
        """
    ).pl()

    home_club_ids = dimension.resolve(deduped["home_team"].to_list(), "football_data")
    away_club_ids = dimension.resolve(deduped["away_team"].to_list(), "football_data")

    curated = deduped.with_columns(
        pl.Series("home_club_id", home_club_ids),
        pl.Series("away_club_id", away_club_ids),
    ).select(
        (
            pl.col("season").str.replace("/", "-")
            + "-"
            + pl.col("home_club_id")
            + "-"
            + pl.col("away_club_id")
        ).alias("match_id"),
        "season",
        "date",
        "home_club_id",
        "away_club_id",
        pl.col("fthg").alias("home_goals"),
        pl.col("ftag").alias("away_goals"),
        pl.col("ftr").alias("result"),
        pl.col("psch").alias("pinnacle_home_odds"),
        pl.col("pscd").alias("pinnacle_draw_odds"),
        pl.col("psca").alias("pinnacle_away_odds"),
    )

    _assert_no_self_fixtures(curated)
    _assert_result_matches_score(curated)
    validated = StgMatchSchema.validate(curated)

    _materialize(conn, "stg_matches", validated)
    log.info("curate.matches", rows=validated.height)


def curate_fixtures(
    conn: duckdb.DuckDBPyConnection, dimension: ClubDimension | None = None
) -> None:
    dimension = dimension or load_club_dimension()
    fixtures = conn.execute(
        "SELECT fpl_fixture_id, gameweek, kickoff_time, home_team_id, away_team_id, "
        "home_score, away_score, finished FROM raw_fpl_fixtures"
    ).pl()

    home_club_ids = dimension.resolve(fixtures["home_team_id"].to_list(), "fpl")
    away_club_ids = dimension.resolve(fixtures["away_team_id"].to_list(), "fpl")
    season = season_label(settings.current_season_start_year())

    curated = fixtures.with_columns(
        pl.Series("home_club_id", home_club_ids),
        pl.Series("away_club_id", away_club_ids),
        pl.lit(season).alias("season"),
    ).select(
        pl.col("fpl_fixture_id").alias("fixture_id"),
        "season",
        "gameweek",
        "kickoff_time",
        "home_club_id",
        "away_club_id",
        pl.col("home_score").alias("home_goals"),
        pl.col("away_score").alias("away_goals"),
        "finished",
    )

    _assert_no_self_fixtures(curated)
    validated = MartFixtureSchema.validate(curated)

    _materialize(conn, "mart_fixtures", validated)
    log.info("curate.fixtures", rows=validated.height)


def curate_all(conn: duckdb.DuckDBPyConnection) -> None:
    dimension = load_club_dimension()
    curate_club_season_membership(conn)
    curate_matches(conn, dimension)
    curate_fixtures(conn, dimension)
