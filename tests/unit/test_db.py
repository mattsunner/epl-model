from pathlib import Path

import polars as pl

from plforecast.config import Settings
from plforecast.storage.curate import latest_snapshot_sql
from plforecast.storage.db import RAW_VIEWS, connect, ensure_raw_views


def _config(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def _views(conn) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
    ).fetchall()
    return {r[0] for r in rows}


def test_connect_on_empty_data_dir_creates_no_views_and_does_not_fail(tmp_path):
    """The clean-clone case: `connect()` before any ingest must succeed. DuckDB errors
    on a read_parquet glob with no matches, so sources without snapshots get no view."""
    conn = connect(_config(tmp_path))
    try:
        assert _views(conn) == set()
        assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 0
    finally:
        conn.close()


def test_ensure_raw_views_creates_a_view_once_a_snapshot_exists(tmp_path):
    config = _config(tmp_path)
    snapshot = config.raw_dir / "fpl-teams" / "20260101T000000Z"
    snapshot.mkdir(parents=True)
    pl.DataFrame({"fpl_team_id": [1], "name": ["Arsenal"], "short_name": ["ARS"]}).write_parquet(
        snapshot / "data.parquet"
    )

    conn = connect(config)
    try:
        assert _views(conn) == {"raw_fpl_teams"}
        row = conn.execute("SELECT fpl_team_id, filename FROM raw_fpl_teams").fetchone()
        assert row[0] == 1
        assert row[1].endswith("20260101T000000Z/data.parquet")  # filename exposed for dedupe
    finally:
        conn.close()


def test_ensure_raw_views_is_idempotent_and_drops_stale_views(tmp_path):
    config = _config(tmp_path)
    snapshot = config.raw_dir / "understat" / "20260101T000000Z"
    snapshot.mkdir(parents=True)
    pl.DataFrame({"season": ["2015/16"]}).write_parquet(snapshot / "data.parquet")

    conn = connect(config)
    try:
        assert ensure_raw_views(conn, config=config) == ["raw_understat_team_match"]
        assert ensure_raw_views(conn, config=config) == ["raw_understat_team_match"]

        (snapshot / "data.parquet").unlink()
        assert ensure_raw_views(conn, config=config) == []
        assert _views(conn) == set()
    finally:
        conn.close()


def test_every_raw_view_maps_to_a_distinct_source_directory():
    assert len(set(RAW_VIEWS.values())) == len(RAW_VIEWS)


def test_latest_snapshot_sql_selects_the_newest_parquet_snapshot(tmp_path):
    """Runs against a real Parquet-backed view, not a seeded table: the SQL form matters
    because DuckDB's metadata-aggregate optimisation errors on a bare max(filename)."""
    config = _config(tmp_path)
    for stamp, value in (("20260101T000000Z", 1), ("20260108T000000Z", 2)):
        snapshot = config.raw_dir / "fpl-teams" / stamp
        snapshot.mkdir(parents=True)
        pl.DataFrame({"fpl_team_id": [value], "name": ["x"], "short_name": ["X"]}).write_parquet(
            snapshot / "data.parquet"
        )

    conn = connect(config)
    try:
        assert conn.execute("SELECT count(*) FROM raw_fpl_teams").fetchone()[0] == 2
        latest = conn.execute(latest_snapshot_sql("raw_fpl_teams")).pl()
        assert latest["fpl_team_id"].to_list() == [2]
    finally:
        conn.close()
