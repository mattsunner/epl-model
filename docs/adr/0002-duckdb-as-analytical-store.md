# ADR 0002: DuckDB as the analytical store

**Status**: accepted 14 September 2026; raw-view handling revised 16 September 2026.

## Context

The pipeline is local and on demand. It lands immutable Parquet snapshots per source
and needs typed, deduplicated, joinable curated tables for features and evaluation.

## Options considered

1. **Postgres.** A running service for a pipeline that is explicitly local-only.
2. **Plain Parquet plus Polars.** Works, but every join and dedupe is application code
   and there is no single queryable surface for ad hoc inspection.
3. **DuckDB, single file.** Zero-service, reads Parquet natively, columnar and
   analytical, one file at `data/pl.duckdb`.

## Decision

Option 3. Zones:

- `raw_*`: views over the Parquet landing zone, never mutated. Recreated on every
  connection by `storage/db.py` because they embed the absolute path of `data/raw/`;
  a source with no snapshot yet simply has no view (this is what made "migrate before
  ingest" fail on a clean clone under the original one-shot migration approach).
- `stg_*`: typed, deduplicated, club IDs resolved, one row per natural key, built from
  the most recently landed snapshot of each source.
- `mart_*`: model-ready.

Numbered SQL migrations, applied once each, are reserved for tables whose history
matters. Curated tables are rebuilt wholesale by `curate` with `CREATE OR REPLACE`.

## Consequences

- The database file is a cache of `data/raw/` plus `club_aliases.yaml`; it is
  gitignored and fully reproducible.
- Two DuckDB 1.5 quirks on `union_by_name` views shaped the curate SQL: a bare
  `max(filename)` errors inside the optimizer, and a `row_number()` window over the
  filename filter returned nulls for columns only the newer snapshot had. Curate uses
  `max` over `DISTINCT` and does natural-key dedupe in Polars.
- Polars is the transform layer; DuckDB is storage and set-oriented joins.
