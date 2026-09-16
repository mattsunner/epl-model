"""Raw -> curated transforms (design.md section 5.4). Reads from the raw_* views,
resolves club identity, and materialises stg_*/mart_* tables.

- stg_club_season: which clubs were in the Premier League each season.
- stg_matches: played match results (football-data.co.uk), club IDs resolved, one row
  per natural key.
- stg_odds: one row per (match, bookmaker) closing 1X2 price, long shape, from every
  closing-price set football-data.co.uk publishes for that season.
- mart_fixtures: the full current-season fixture list (FPL), club IDs resolved. This is
  what the simulation engine reads remaining fixtures from.

The market benchmark price on stg_matches (`benchmark_*_odds`, `benchmark_source`) is
chosen per match by BENCHMARK_CHAIN: Pinnacle closing where the site still publishes
it, then Betfair Exchange closing, then the site's average closing price. Design.md
section 8.3 chose Pinnacle alone; the site dropped it mid-2025/26 (ADR 0007).

Every raw source is a full-state snapshot: each ingest run lands the complete current
view of that source, and the raw_* views union every snapshot ever landed. Curate
therefore always starts from the most recently landed snapshot of each view
(`latest_snapshot_sql`), and stg_matches additionally dedupes on its natural key as a
second guard.

mart_matches (a model-ready superset of stg_matches, per design.md's repo layout) is
not built yet; stg_matches carries everything the two shipped models need.
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
from plforecast.ingest.footballdata import CLOSING_ODDS_SETS, closing_odds_columns

log = structlog.get_logger()


# Order matters: the first set with a complete home/draw/away price for a match wins.
BENCHMARK_CHAIN: tuple[str, ...] = ("pinnacle", "betfair_exchange", "market_avg")
PRICE_TYPE_CLOSING = "closing"


class StgMatchSchema(pa.DataFrameModel):
    match_id: Series[str] = pa.Field(unique=True)
    season: Series[str] = pa.Field(str_matches=r"^\d{4}/\d{2}$")
    date: Series[pl.Date]
    home_club_id: Series[str]
    away_club_id: Series[str]
    home_goals: Series[int] = pa.Field(ge=0)
    away_goals: Series[int] = pa.Field(ge=0)
    result: Series[str] = pa.Field(isin=["H", "D", "A"])
    benchmark_home_odds: Series[float] = pa.Field(nullable=True, ge=1.0)
    benchmark_draw_odds: Series[float] = pa.Field(nullable=True, ge=1.0)
    benchmark_away_odds: Series[float] = pa.Field(nullable=True, ge=1.0)
    benchmark_source: Series[str] = pa.Field(nullable=True, isin=list(BENCHMARK_CHAIN))

    class Config:
        strict = True
        coerce = True


class StgOddsSchema(pa.DataFrameModel):
    match_id: Series[str]
    bookmaker: Series[str] = pa.Field(isin=list(CLOSING_ODDS_SETS.values()))
    price_type: Series[str] = pa.Field(isin=[PRICE_TYPE_CLOSING])
    home_odds: Series[float] = pa.Field(ge=1.0)
    draw_odds: Series[float] = pa.Field(ge=1.0)
    away_odds: Series[float] = pa.Field(ge=1.0)

    class Config:
        strict = True
        coerce = True


def _bookmaker_columns() -> dict[str, tuple[str, str, str]]:
    """bookmaker id -> (home, draw, away) parsed column names on the raw football-data frame."""
    return {
        bookmaker: closing_odds_columns(prefix) for prefix, bookmaker in CLOSING_ODDS_SETS.items()
    }


def _benchmark_exprs() -> list[pl.Expr]:
    """benchmark_{home,draw,away}_odds and benchmark_source from the first set in
    BENCHMARK_CHAIN with all three prices present for the row."""
    columns = _bookmaker_columns()
    complete = {
        bookmaker: pl.all_horizontal([pl.col(c).is_not_null() for c in columns[bookmaker]])
        for bookmaker in BENCHMARK_CHAIN
    }

    def chain(pick: dict[str, pl.Expr], dtype: pl.DataType) -> pl.Expr:
        expr: pl.Expr = pl.lit(None, dtype=dtype)
        for bookmaker in reversed(BENCHMARK_CHAIN):
            expr = pl.when(complete[bookmaker]).then(pick[bookmaker]).otherwise(expr)
        return expr

    return [
        chain({b: pl.col(columns[b][i]) for b in BENCHMARK_CHAIN}, pl.Float64()).alias(name)
        for i, name in enumerate(
            ("benchmark_home_odds", "benchmark_draw_odds", "benchmark_away_odds")
        )
    ] + [chain({b: pl.lit(b) for b in BENCHMARK_CHAIN}, pl.Utf8()).alias("benchmark_source")]


def build_odds_long(matches_with_odds: pl.DataFrame) -> pl.DataFrame:
    """`matches_with_odds` needs `match_id` plus the parsed closing-odds columns. One row
    per (match, bookmaker) where all three prices are present."""
    frames = []
    for bookmaker, (h, d, a) in _bookmaker_columns().items():
        frames.append(
            matches_with_odds.select(
                "match_id",
                pl.lit(bookmaker).alias("bookmaker"),
                pl.lit(PRICE_TYPE_CLOSING).alias("price_type"),
                pl.col(h).cast(pl.Float64).alias("home_odds"),
                pl.col(d).cast(pl.Float64).alias("draw_odds"),
                pl.col(a).cast(pl.Float64).alias("away_odds"),
            ).drop_nulls(["home_odds", "draw_odds", "away_odds"])
        )
    odds = pl.concat(frames).sort("match_id", "bookmaker")
    dupes = odds.filter(odds.select("match_id", "bookmaker").is_duplicated())
    if dupes.height:
        raise ValueError(f"stg_odds has duplicate (match_id, bookmaker) rows: {dupes}")
    return StgOddsSchema.validate(odds)


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


def latest_snapshot_sql(view: str) -> str:
    """Rows of `view` belonging to its most recently landed snapshot. Snapshot directories
    are named with a fixed-width UTC timestamp, so the lexicographic maximum of the
    `filename` column every raw view exposes is the latest one."""
    # max() over DISTINCT rather than max(filename) directly: DuckDB 1.5's optimizer tries
    # to answer a bare max(filename) from Parquet metadata and hits an internal error on
    # these union_by_name views. The DISTINCT subquery sidesteps that rewrite.
    return (
        f"SELECT * FROM {view} WHERE filename = "
        f"(SELECT max(f) FROM (SELECT DISTINCT filename AS f FROM {view}))"
    )


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
    matches = conn.execute(
        "SELECT season, home_team, away_team FROM "
        f"({latest_snapshot_sql('raw_footballdata_matches')})"
    ).pl()
    dimension = load_club_dimension()
    membership = build_club_season_membership(matches, dimension)

    _materialize(conn, "stg_club_season", membership)
    log.info("curate.club_season_membership", rows=membership.height)


def curate_matches(conn: duckdb.DuckDBPyConnection, dimension: ClubDimension | None = None) -> None:
    """One row per (season, home team, away team) -- the natural key for a football-data
    row, since each pairing plays at a given ground exactly once a season. Starts from
    the latest snapshot, then keeps the most recently landed row per natural key as a
    second guard, so the table is one row per match however many snapshots exist."""
    dimension = dimension or load_club_dimension()
    odds_cols = ", ".join(c for cols in _bookmaker_columns().values() for c in cols)
    # The natural-key dedupe runs in Polars, not as a SQL window: DuckDB 1.5 returns nulls
    # for union_by_name columns that only the newer snapshot has when a row_number()
    # window sits on top of the filename filter.
    deduped = (
        conn.execute(
            f"SELECT season, date, home_team, away_team, fthg, ftag, ftr, {odds_cols}, filename "
            f"FROM ({latest_snapshot_sql('raw_footballdata_matches')})"
        )
        .pl()
        .sort("filename", descending=True)
        .unique(subset=["season", "home_team", "away_team"], keep="first", maintain_order=True)
        .drop("filename")
        .sort("date", "home_team")
    )

    home_club_ids = dimension.resolve(deduped["home_team"].to_list(), "football_data")
    away_club_ids = dimension.resolve(deduped["away_team"].to_list(), "football_data")

    resolved = deduped.with_columns(
        pl.Series("home_club_id", home_club_ids),
        pl.Series("away_club_id", away_club_ids),
        (
            pl.col("season").str.replace("/", "-")
            + "-"
            + pl.Series("home_club_id", home_club_ids)
            + "-"
            + pl.Series("away_club_id", away_club_ids)
        ).alias("match_id"),
    )

    curated = resolved.select(
        "match_id",
        "season",
        "date",
        "home_club_id",
        "away_club_id",
        pl.col("fthg").alias("home_goals"),
        pl.col("ftag").alias("away_goals"),
        pl.col("ftr").alias("result"),
        *_benchmark_exprs(),
    )

    _assert_no_self_fixtures(curated)
    _assert_result_matches_score(curated)
    validated = StgMatchSchema.validate(curated)
    odds = build_odds_long(resolved)

    _materialize(conn, "stg_matches", validated)
    _materialize(conn, "stg_odds", odds)
    coverage = validated.group_by("benchmark_source").len().sort("benchmark_source").to_dicts()
    log.info("curate.matches", rows=validated.height, benchmark_coverage=coverage)
    log.info("curate.odds", rows=odds.height)


def curate_fixtures(
    conn: duckdb.DuckDBPyConnection, dimension: ClubDimension | None = None
) -> None:
    dimension = dimension or load_club_dimension()
    fixtures = conn.execute(
        "SELECT fpl_fixture_id, gameweek, kickoff_time, home_team_id, away_team_id, "
        f"home_score, away_score, finished FROM ({latest_snapshot_sql('raw_fpl_fixtures')})"
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
