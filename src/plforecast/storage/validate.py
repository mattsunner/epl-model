"""Data-quality invariants over the curated tables (story B-13): a standalone report
that catches silent joins and coverage gaps in whatever `pl.duckdb` currently holds,
independent of the curate run that produced it -- useful after copying the database,
or as a last check before publishing a forecast.

Several of these are already enforced destructively at curate time (a self-fixture, a
result that disagrees with its scoreline, an Understat/football-data or FPL/
football-data scoreline mismatch all raise inside `curate.py` and so can never reach a
materialised table). Re-deriving that exact logic here would duplicate it for no extra
coverage: if `stg_matches` exists, those checks already passed. What genuinely is new
here: the round-robin shape of every completed season (never checked anywhere else),
closing-odds coverage surfaced as a first-class report rather than a log line, and
every club_id in a fact table resolving to `dim_club` against the *live* tables rather
than only the fixture snapshot `tests/unit/test_clubs.py` checks against.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import polars as pl
import structlog

from plforecast.storage.curate import reconcile_current_season

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


class ValidationFailed(Exception):
    """Raised by `run_validations(..., raise_on_failure=True)` naming every failed
    check at once, not just the first -- the same reasoning `entities/clubs.py`'s
    alias resolution uses for batching its own errors."""


def _has_relation(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row and row[0])


def check_season_shape(conn: duckdb.DuckDBPyConnection) -> Check:
    """Every *completed* season (every season but the max, which is in progress) is a
    full round robin: `n` clubs, `n*(n-1)` matches, each club `n-1` home and `n-1`
    away. Derived from the data's own club count per season rather than a hardcoded
    20/380/19, so this stays correct if the league size ever changes."""
    matches = conn.execute("SELECT season, home_club_id, away_club_id FROM stg_matches").pl()
    if matches.height == 0:
        return Check("season_shape", True, "no matches curated yet")
    current_season = matches["season"].max()
    completed = matches.filter(pl.col("season") != current_season)

    problems = []
    for season, group in completed.group_by("season", maintain_order=True):
        season = season[0]
        clubs = sorted(set(group["home_club_id"]) | set(group["away_club_id"]))
        n = len(clubs)
        expected = n * (n - 1)
        if group.height != expected:
            problems.append(f"{season}: {group.height} matches, expected {expected} for {n} clubs")
        home_counts = group.group_by("home_club_id").len()
        away_counts = group.group_by("away_club_id").len()
        for club in clubs:
            home = home_counts.filter(pl.col("home_club_id") == club)["len"]
            away = away_counts.filter(pl.col("away_club_id") == club)["len"]
            home_n = home.item() if home.len() else 0
            away_n = away.item() if away.len() else 0
            if home_n != n - 1 or away_n != n - 1:
                problems.append(
                    f"{season}/{club}: {home_n} home, {away_n} away, expected {n - 1} each"
                )

    n_seasons = completed["season"].n_unique()
    detail = (
        "; ".join(problems)
        if problems
        else f"{n_seasons} completed season(s), all clean round robins"
    )
    return Check("season_shape", not problems, detail)


def check_unique_ids(conn: duckdb.DuckDBPyConnection) -> Check:
    problems = []
    for table, column in (("stg_matches", "match_id"), ("stg_fixtures", "fixture_id")):
        if not _has_relation(conn, table):
            continue
        dupes = conn.execute(
            f"SELECT {column}, count(*) FROM {table} GROUP BY {column} HAVING count(*) > 1"
        ).fetchall()
        if dupes:
            problems.append(f"{table}.{column}: {len(dupes)} duplicated value(s)")
    return Check(
        "unique_ids",
        not problems,
        "; ".join(problems) if problems else "match_id and fixture_id unique",
    )


def check_no_self_fixtures(conn: duckdb.DuckDBPyConnection) -> Check:
    problems = []
    for table in ("stg_matches", "stg_fixtures"):
        if not _has_relation(conn, table):
            continue
        n = conn.execute(
            f"SELECT count(*) FROM {table} WHERE home_club_id = away_club_id"
        ).fetchone()
        if n and n[0]:
            problems.append(f"{table}: {n[0]} row(s) with home_club_id = away_club_id")
    return Check(
        "no_self_fixtures",
        not problems,
        "; ".join(problems) if problems else "no club plays itself",
    )


def check_result_matches_score(conn: duckdb.DuckDBPyConnection) -> Check:
    if not _has_relation(conn, "stg_matches"):
        return Check("result_matches_score", True, "stg_matches not curated yet")
    row = conn.execute(
        """
        SELECT count(*) FROM stg_matches
        WHERE result != CASE
            WHEN home_goals > away_goals THEN 'H'
            WHEN home_goals < away_goals THEN 'A'
            ELSE 'D'
        END
        """
    ).fetchone()
    bad = int(row[0]) if row else 0
    ok = bad == 0
    return Check(
        "result_matches_score",
        ok,
        "consistent" if ok else f"{bad} row(s) disagree with their scoreline",
    )


def check_xg_coverage(conn: duckdb.DuckDBPyConnection) -> Check:
    """100% xG coverage on every completed season; the in-progress season may lag."""
    if not _has_relation(conn, "stg_matches"):
        return Check("xg_coverage", True, "stg_matches not curated yet")
    row = conn.execute(
        """
        SELECT count(*) FILTER (
            WHERE home_xg IS NULL AND season != (SELECT max(season) FROM stg_matches)
        )
        FROM stg_matches
        """
    ).fetchone()
    missing = row[0] if row else 0
    return Check(
        "xg_coverage",
        missing == 0,
        "complete" if missing == 0 else f"{missing} completed-season match(es) missing xG",
    )


def check_closing_odds_coverage(conn: duckdb.DuckDBPyConnection) -> Check:
    """Report only -- never fails on its own; a gap here is a known, logged fact
    (ADR 0007's fallback chain exists because the site itself has gaps), not a bug."""
    if not _has_relation(conn, "stg_matches"):
        return Check("closing_odds_coverage", True, "stg_matches not curated yet")
    by_season_source = conn.execute(
        """
        SELECT season, coalesce(benchmark_source, 'none') AS source, count(*) AS n
        FROM stg_matches GROUP BY season, source ORDER BY season, source
        """
    ).pl()
    missing = by_season_source.filter(pl.col("source") == "none")
    detail = (
        "every match has a benchmark price"
        if missing.height == 0
        else "no benchmark price for: "
        + ", ".join(f"{r['season']} ({r['n']})" for r in missing.to_dicts())
    )
    return Check("closing_odds_coverage", True, detail)


def check_club_ids_resolve(conn: duckdb.DuckDBPyConnection) -> Check:
    """Every club_id appearing in any fact table exists in dim_club (design.md 5.3's
    invariant, checked here against the live curated tables rather than only the
    fixture snapshot `tests/unit/test_clubs.py` checks alias resolution against)."""
    if not _has_relation(conn, "dim_club"):
        return Check("club_ids_resolve", True, "dim_club not curated yet")
    known = set(conn.execute("SELECT club_id FROM dim_club").pl()["club_id"].to_list())

    seen: set[str] = set()
    for table, columns in (
        ("stg_matches", ("home_club_id", "away_club_id")),
        ("stg_fixtures", ("home_club_id", "away_club_id")),
        ("stg_club_season", ("club_id",)),
        ("mart_team_match", ("club_id", "opponent_id")),
    ):
        if not _has_relation(conn, table):
            continue
        for column in columns:
            seen |= set(
                conn.execute(f"SELECT DISTINCT {column} FROM {table}").pl()[column].to_list()
            )

    unresolved = sorted(seen - known)
    ok = not unresolved
    detail = "every fact-table club_id resolves" if ok else f"unresolved club_id(s): {unresolved}"
    return Check("club_ids_resolve", ok, detail)


def check_current_season_reconciliation(conn: duckdb.DuckDBPyConnection) -> Check:
    """Re-runs the same FPL/football-data cross-check curate applies at write time
    (story B-07); surfaced here as a report rather than duplicated logic."""
    if not (_has_relation(conn, "stg_fixtures") and _has_relation(conn, "stg_matches")):
        return Check(
            "current_season_reconciliation", True, "stg_fixtures or stg_matches not curated yet"
        )
    try:
        counts = reconcile_current_season(conn)
    except ValueError as exc:
        return Check("current_season_reconciliation", False, str(exc))
    return Check("current_season_reconciliation", True, str(counts))


ALL_CHECKS = (
    check_season_shape,
    check_unique_ids,
    check_no_self_fixtures,
    check_result_matches_score,
    check_xg_coverage,
    check_closing_odds_coverage,
    check_club_ids_resolve,
    check_current_season_reconciliation,
)


def run_validations(
    conn: duckdb.DuckDBPyConnection, *, raise_on_failure: bool = False
) -> list[Check]:
    results = [check(conn) for check in ALL_CHECKS]
    for result in results:
        (log.info if result.ok else log.error)(
            "validate.check", name=result.name, ok=result.ok, detail=result.detail
        )
    if raise_on_failure:
        failed = [r for r in results if not r.ok]
        if failed:
            raise ValidationFailed("; ".join(f"{r.name}: {r.detail}" for r in failed))
    return results
