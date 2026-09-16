"""Raw -> curated transforms (design.md section 5.4, ADR 0002). Reads from the raw_*
views, resolves club identity, and materialises the curated tables.

Zones (story B-09):
- dim_club: the club dimension from club_aliases.yaml, so SQL can join to names.
- stg_club_season: which clubs were in the Premier League each season.
- stg_matches: played match results (football-data.co.uk) with Understat xG joined,
  club IDs resolved, one row per natural key, benchmark closing price chosen.
- stg_odds: one row per (match, bookmaker) closing 1X2 price, long shape.
- stg_fixtures: the full current-season fixture list (FPL), club IDs resolved. What
  the simulation engine reads remaining fixtures from.
- mart_team_match: one row per club per match (the feature grain), with goals, xG,
  points and league rest days for and against.

Every raw source is a full-state snapshot: each ingest run lands the complete current
view of that source, and the raw_* views union every snapshot ever landed. Curate
therefore always starts from the most recently landed snapshot of each view
(`latest_snapshot_sql`), and stg_matches additionally dedupes on its natural key as a
second guard.

The market benchmark price on stg_matches (`benchmark_*_odds`, `benchmark_source`) is
chosen per match by BENCHMARK_CHAIN: Pinnacle closing where the site still publishes
it, then Betfair Exchange closing, then the site's average closing price (ADR 0007).
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
    # Nullable only for the in-progress season, where Understat can lag a result by a
    # day; curate raises if any completed-season match lacks xG.
    home_xg: Series[float] = pa.Field(nullable=True, ge=0)
    away_xg: Series[float] = pa.Field(nullable=True, ge=0)
    home_np_xg: Series[float] = pa.Field(nullable=True, ge=0)
    away_np_xg: Series[float] = pa.Field(nullable=True, ge=0)
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


class StgFixtureSchema(pa.DataFrameModel):
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


class DimClubSchema(pa.DataFrameModel):
    club_id: Series[str] = pa.Field(unique=True)
    display_name: Series[str]
    football_data_name: Series[str] = pa.Field(nullable=True)
    fpl_code: Series[int] = pa.Field(nullable=True)
    clubelo_name: Series[str] = pa.Field(nullable=True)
    understat_name: Series[str] = pa.Field(nullable=True)
    transfermarkt_id: Series[int] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


class MartTeamMatchSchema(pa.DataFrameModel):
    match_id: Series[str]
    season: Series[str] = pa.Field(str_matches=r"^\d{4}/\d{2}$")
    date: Series[pl.Date]
    club_id: Series[str]
    opponent_id: Series[str]
    is_home: Series[bool]
    goals_for: Series[int] = pa.Field(ge=0)
    goals_against: Series[int] = pa.Field(ge=0)
    xg_for: Series[float] = pa.Field(nullable=True, ge=0)
    xg_against: Series[float] = pa.Field(nullable=True, ge=0)
    npxg_for: Series[float] = pa.Field(nullable=True, ge=0)
    npxg_against: Series[float] = pa.Field(nullable=True, ge=0)
    result: Series[str] = pa.Field(isin=["W", "D", "L"])
    points: Series[int] = pa.Field(isin=[0, 1, 3])
    # Days since the club's previous *league* match this season; null for its first.
    # Cup and European fixtures are not in stg_matches, so this understates congestion.
    rest_days: Series[int] = pa.Field(nullable=True, ge=0)

    class Config:
        strict = True
        coerce = True


def curate_club_dimension(conn: duckdb.DuckDBPyConnection, dimension: ClubDimension) -> None:
    frame = dimension.frame.select(
        "club_id",
        "display_name",
        "football_data_name",
        pl.col("fpl_code").cast(pl.Int64),
        pl.col("clubelo_name").cast(pl.Utf8),
        pl.col("understat_name").cast(pl.Utf8),
        pl.col("transfermarkt_id").cast(pl.Int64),
    )
    validated = DimClubSchema.validate(frame)
    _materialize(conn, "dim_club", validated)
    log.info("curate.club_dimension", rows=validated.height)


def _has_relation(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row and row[0])


def _understat_xg(conn: duckdb.DuckDBPyConnection, dimension: ClubDimension) -> pl.DataFrame | None:
    """Latest Understat snapshot with club IDs resolved, keyed like stg_matches.
    None when Understat has not been ingested (a clean clone mid-bootstrap)."""
    if not _has_relation(conn, "raw_understat_team_match"):
        log.warning("curate.xg_skipped_no_understat_snapshot")
        return None
    raw = conn.execute(
        "SELECT season, date, home_team, away_team, home_goals, away_goals, home_xg, "
        "away_xg, home_np_xg, away_np_xg "
        f"FROM ({latest_snapshot_sql('raw_understat_team_match')})"
    ).pl()
    return raw.with_columns(
        pl.Series("home_club_id", dimension.resolve(raw["home_team"].to_list(), "understat")),
        pl.Series("away_club_id", dimension.resolve(raw["away_team"].to_list(), "understat")),
    ).drop("home_team", "away_team")


def _join_xg(matches: pl.DataFrame, xg: pl.DataFrame | None) -> pl.DataFrame:
    """Left-join Understat xG onto stg_matches rows by (season, home, away): each pairing
    plays at a given ground once per season, so that is the natural key. Date is *not*
    part of the key: Understat's timestamps for 2015/16 and 2016/17 Monday-night matches
    fall on the next calendar day. Instead the date gap is checked (a gap over one day
    means the join hit the wrong fixture). Goals must agree where both sources have the
    match; every completed-season match must have xG; the in-progress season may lag
    (warned, not failed)."""
    if xg is None:
        return matches.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in ("home_xg", "away_xg", "home_np_xg", "away_np_xg")
        )
    key = ["season", "home_club_id", "away_club_id"]
    joined = matches.join(
        xg.rename(
            {"home_goals": "us_home_goals", "away_goals": "us_away_goals", "date": "us_date"}
        ),
        on=key,
        how="left",
    )
    date_gap = joined.filter(
        pl.col("us_date").is_not_null()
        & ((pl.col("us_date") - pl.col("date")).dt.total_days().abs() > 1)
    )
    if date_gap.height:
        raise ValueError(
            f"Understat match dates differ from football-data by more than a day: {date_gap}"
        )
    disagree = joined.filter(
        pl.col("us_home_goals").is_not_null()
        & (
            (pl.col("us_home_goals") != pl.col("home_goals"))
            | (pl.col("us_away_goals") != pl.col("away_goals"))
        )
    )
    if disagree.height:
        raise ValueError(f"Understat and football-data disagree on scorelines: {disagree}")

    current_season = matches["season"].max()
    missing = joined.filter(pl.col("home_xg").is_null())
    missing_completed = missing.filter(pl.col("season") != current_season)
    if missing_completed.height:
        raise ValueError(
            f"{missing_completed.height} completed-season matches have no Understat xG: "
            f"{missing_completed.select(key)}"
        )
    if missing.height:
        log.warning("curate.xg_missing_current_season", count=missing.height)
    unmatched_xg = xg.join(matches.select(key), on=key, how="anti")
    if unmatched_xg.height:
        log.warning("curate.xg_rows_without_match", count=unmatched_xg.height)
    return joined.drop("us_home_goals", "us_away_goals", "us_date")


def build_team_match(matches: pl.DataFrame) -> pl.DataFrame:
    """One row per club per match from a stg_matches-shaped frame, with league rest
    days (days since the club's previous match in the same season, null for the first)."""
    home = matches.select(
        "match_id",
        "season",
        "date",
        pl.col("home_club_id").alias("club_id"),
        pl.col("away_club_id").alias("opponent_id"),
        pl.lit(True).alias("is_home"),
        pl.col("home_goals").alias("goals_for"),
        pl.col("away_goals").alias("goals_against"),
        pl.col("home_xg").alias("xg_for"),
        pl.col("away_xg").alias("xg_against"),
        pl.col("home_np_xg").alias("npxg_for"),
        pl.col("away_np_xg").alias("npxg_against"),
    )
    away = matches.select(
        "match_id",
        "season",
        "date",
        pl.col("away_club_id").alias("club_id"),
        pl.col("home_club_id").alias("opponent_id"),
        pl.lit(False).alias("is_home"),
        pl.col("away_goals").alias("goals_for"),
        pl.col("home_goals").alias("goals_against"),
        pl.col("away_xg").alias("xg_for"),
        pl.col("home_xg").alias("xg_against"),
        pl.col("away_np_xg").alias("npxg_for"),
        pl.col("home_np_xg").alias("npxg_against"),
    )
    team_match = (
        pl.concat([home, away])
        .with_columns(
            pl.when(pl.col("goals_for") > pl.col("goals_against"))
            .then(pl.lit("W"))
            .when(pl.col("goals_for") < pl.col("goals_against"))
            .then(pl.lit("L"))
            .otherwise(pl.lit("D"))
            .alias("result")
        )
        .with_columns(
            pl.when(pl.col("result") == "W")
            .then(3)
            .when(pl.col("result") == "D")
            .then(1)
            .otherwise(0)
            .cast(pl.Int64)
            .alias("points")
        )
        .sort("club_id", "season", "date", "match_id")
        .with_columns(
            (pl.col("date") - pl.col("date").shift(1).over("club_id", "season"))
            .dt.total_days()
            .cast(pl.Int64)
            .alias("rest_days")
        )
        .sort("date", "match_id", "is_home", descending=[False, False, True])
    )
    per_match = team_match.group_by("match_id").len()
    if (per_match["len"] != 2).any():
        raise ValueError("mart_team_match must have exactly two rows per match")
    return MartTeamMatchSchema.validate(team_match)


def curate_team_match(conn: duckdb.DuckDBPyConnection) -> None:
    matches = conn.execute("SELECT * FROM stg_matches").pl()
    team_match = build_team_match(matches)
    _materialize(conn, "mart_team_match", team_match)
    log.info("curate.team_match", rows=team_match.height)


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

    curated = _join_xg(
        resolved.select(
            "match_id",
            "season",
            "date",
            "home_club_id",
            "away_club_id",
            pl.col("fthg").alias("home_goals"),
            pl.col("ftag").alias("away_goals"),
            pl.col("ftr").alias("result"),
        ),
        _understat_xg(conn, dimension),
    ).join(resolved.select("match_id", *_benchmark_exprs()), on="match_id", how="left")

    _assert_no_self_fixtures(curated)
    _assert_result_matches_score(curated)
    validated = StgMatchSchema.validate(curated)
    odds = build_odds_long(resolved)

    _materialize(conn, "stg_matches", validated)
    _materialize(conn, "stg_odds", odds)
    coverage = validated.group_by("benchmark_source").len().sort("benchmark_source").to_dicts()
    log.info(
        "curate.matches",
        rows=validated.height,
        benchmark_coverage=coverage,
        xg_missing=validated["home_xg"].null_count(),
    )
    log.info("curate.odds", rows=odds.height)


def curate_fixtures(
    conn: duckdb.DuckDBPyConnection, dimension: ClubDimension | None = None
) -> None:
    dimension = dimension or load_club_dimension()
    fixtures = conn.execute(
        "SELECT fpl_fixture_id, gameweek, kickoff_time, home_team_id, away_team_id, "
        f"home_score, away_score, finished FROM ({latest_snapshot_sql('raw_fpl_fixtures')})"
    ).pl()
    # Fixtures reference teams by FPL's per-season `id`; the club dimension keys on the
    # stable `code`, so map through this snapshot's roster first.
    roster = conn.execute(
        f"SELECT fpl_team_id, fpl_code FROM ({latest_snapshot_sql('raw_fpl_teams')})"
    ).pl()
    id_to_code = dict(
        zip(roster["fpl_team_id"].to_list(), roster["fpl_code"].to_list(), strict=True)
    )
    referenced = set(fixtures["home_team_id"].to_list()) | set(fixtures["away_team_id"].to_list())
    unknown_ids = sorted(referenced - set(id_to_code))
    if unknown_ids:
        raise ValueError(f"FPL fixtures reference team ids missing from the roster: {unknown_ids}")

    home_club_ids = dimension.resolve(
        [id_to_code[i] for i in fixtures["home_team_id"].to_list()], "fpl"
    )
    away_club_ids = dimension.resolve(
        [id_to_code[i] for i in fixtures["away_team_id"].to_list()], "fpl"
    )
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
    validated = StgFixtureSchema.validate(curated)

    _materialize(conn, "stg_fixtures", validated)
    log.info("curate.fixtures", rows=validated.height)


def reconcile_current_season(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Cross-source check (story B-07): for the season stg_fixtures covers, every
    finished FPL fixture that football-data.co.uk has also published must carry the
    same scoreline. A disagreement raises; count gaps in either direction are returned
    and logged, since football-data lags FPL by up to a week."""
    joined = conn.execute(
        """
        SELECT f.fixture_id, f.home_club_id, f.away_club_id,
               f.home_goals AS fpl_home, f.away_goals AS fpl_away,
               m.home_goals AS fd_home, m.away_goals AS fd_away
        FROM stg_fixtures f
        LEFT JOIN stg_matches m
          ON m.season = f.season
         AND m.home_club_id = f.home_club_id
         AND m.away_club_id = f.away_club_id
        WHERE f.finished
        """
    ).pl()
    disagree = joined.filter(
        pl.col("fd_home").is_not_null()
        & ((pl.col("fpl_home") != pl.col("fd_home")) | (pl.col("fpl_away") != pl.col("fd_away")))
    )
    if disagree.height:
        raise ValueError(f"FPL and football-data disagree on scorelines: {disagree}")

    fpl_only = int(joined["fd_home"].null_count())
    fd_only = conn.execute(
        """
        SELECT count(*) FROM stg_matches m
        WHERE m.season = (SELECT min(season) FROM stg_fixtures)
          AND NOT EXISTS (
            SELECT 1 FROM stg_fixtures f
            WHERE f.finished AND f.home_club_id = m.home_club_id
              AND f.away_club_id = m.away_club_id
          )
        """
    ).fetchone()
    counts = {
        "finished_in_both": int(joined.height - fpl_only),
        "fpl_only": fpl_only,
        "football_data_only": int(fd_only[0]) if fd_only else 0,
    }
    log.info("curate.reconcile_current_season", **counts)
    return counts


def curate_all(conn: duckdb.DuckDBPyConnection) -> None:
    dimension = load_club_dimension()
    curate_club_dimension(conn, dimension)
    curate_club_season_membership(conn)
    curate_matches(conn, dimension)
    curate_fixtures(conn, dimension)
    curate_team_match(conn)
    reconcile_current_season(conn)
