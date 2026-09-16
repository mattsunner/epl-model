"""DuckDB connection and migration runner. See design.md section 5.4 for the zone
convention: raw_* views over the Parquet landing zone (never mutated), stg_* typed and
deduplicated tables, mart_* model-ready tables. Only raw_* exists so far -- stg_* and
mart_* land with the entities/curate layer once club identity resolution is built.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import structlog

from plforecast.config import Settings, settings

log = structlog.get_logger()

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def connect(config: Settings = settings) -> duckdb.DuckDBPyConnection:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(config.db_path))


def applied_migrations(conn: duckdb.DuckDBPyConnection) -> set[str]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (name VARCHAR PRIMARY KEY, "
        "applied_at TIMESTAMP DEFAULT current_timestamp)"
    )
    return {row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()}


def migrate(conn: duckdb.DuckDBPyConnection, *, config: Settings = settings) -> None:
    """Apply every not-yet-applied migration in MIGRATIONS_DIR, in filename order.
    Each migration is idempotent SQL (CREATE OR REPLACE) so re-running is always safe."""
    already_applied = applied_migrations(conn)
    raw_dir = config.raw_dir.resolve()

    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in already_applied:
            continue
        sql = path.read_text().replace("{raw_dir}", str(raw_dir))
        conn.execute(sql)
        conn.execute("INSERT INTO schema_migrations (name) VALUES (?)", [path.name])
        log.info("migrate.applied", migration=path.name)
