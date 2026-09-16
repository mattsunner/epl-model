"""DuckDB connection, raw views, and table migrations (design.md section 5.4).

Zones: raw_* views over the immutable Parquet landing zone, stg_* typed and deduplicated
tables, mart_* model-ready tables.

raw_* views are not migrations. They embed the absolute path of `data/raw/`, so a view
created once would break silently the moment the repository or `PLFORECAST_DATA_DIR`
moved. They are cheap and idempotent, so `connect()` recreates every one of them on each
connection instead. A source with no landed snapshot yet gets no view: DuckDB binds a
`read_parquet` glob at view-creation time and errors when nothing matches, which is what
made the old "migrate before ingest" quickstart fail on a clean clone.

`schema_migrations` and the numbered SQL files in `migrations/` are reserved for tables,
where ordering and apply-once semantics matter. There are none yet; curated tables are
rebuilt wholesale by `curate` with CREATE OR REPLACE.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import structlog

from plforecast.config import Settings, settings

log = structlog.get_logger()

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# view name -> snapshot directory under data/raw/. Every view exposes `filename` so
# curate can pick the most recently landed snapshot deterministically.
RAW_VIEWS: dict[str, str] = {
    "raw_footballdata_matches": "football-data",
    "raw_fpl_teams": "fpl-teams",
    "raw_fpl_player_availability": "fpl-players",
    "raw_fpl_fixtures": "fpl-fixtures",
    "raw_understat_team_match": "understat",
}


def connect(config: Settings = settings) -> duckdb.DuckDBPyConnection:
    """Open the database, apply pending table migrations, and (re)create the raw views.
    Safe to call on an empty data directory."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(config.db_path))
    migrate(conn, config=config)
    ensure_raw_views(conn, config=config)
    return conn


def ensure_raw_views(conn: duckdb.DuckDBPyConnection, *, config: Settings = settings) -> list[str]:
    """Recreate every raw_* view over whatever snapshots currently exist. Returns the
    names of the views that were created; sources with no snapshot are skipped (and any
    stale view for them dropped) with a warning rather than an error."""
    created = []
    raw_dir = config.raw_dir.resolve()
    for view, source in RAW_VIEWS.items():
        if not any((raw_dir / source).glob("*/data.parquet")):
            conn.execute(f"DROP VIEW IF EXISTS {view}")
            log.warning("db.raw_view_skipped_no_snapshots", view=view, source=source)
            continue
        pattern = raw_dir / source / "*" / "data.parquet"
        conn.execute(
            f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet("
            f"'{pattern}', union_by_name = true, filename = true)"
        )
        created.append(view)
    return created


def applied_migrations(conn: duckdb.DuckDBPyConnection) -> set[str]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (name VARCHAR PRIMARY KEY, "
        "applied_at TIMESTAMP DEFAULT current_timestamp)"
    )
    return {row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()}


def migrate(conn: duckdb.DuckDBPyConnection, *, config: Settings = settings) -> None:
    """Apply every not-yet-applied table migration in MIGRATIONS_DIR, in filename order.
    Each migration is idempotent SQL so re-running is always safe."""
    already_applied = applied_migrations(conn)

    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in already_applied:
            continue
        conn.execute(path.read_text())
        conn.execute("INSERT INTO schema_migrations (name) VALUES (?)", [path.name])
        log.info("migrate.applied", migration=path.name)
