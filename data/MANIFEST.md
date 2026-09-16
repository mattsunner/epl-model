# data/ manifest

`data/` is gitignored except this file and `.gitkeep`. Nothing here is committed;
everything is reproducible from source given the adapters in `src/plforecast/ingest/`.

| Path | Contents | How to (re)populate |
| --- | --- | --- |
| `raw/football-data/{fetched_at}/data.parquet` | Immutable snapshot of E0 match results + Pinnacle closing odds, 2015/16 onward | `just ingest-footballdata` |
| `raw/fpl-teams/{fetched_at}/data.parquet` | Team roster: per-season `id` and stable `code` | `just ingest-fpl` |
| `raw/fpl-players/{fetched_at}/data.parquet` | Player availability and suspensions | `just ingest-fpl` |
| `raw/fpl-fixtures/{fetched_at}/data.parquet` | Full season fixture list, kickoff times | `just ingest-fpl` |
| `raw/understat/{fetched_at}/data.parquet` | Team-level xG, goals, PPDA, 2015/16 onward | `just ingest-understat` |
| `cache/football-data/*.bin`, `cache/fpl/*.bin` | TTL-cached raw HTTP responses, keyed by URL hash | Populated automatically by the adapters; safe to delete |
| `cache/understat-soccerdata/` | soccerdata's own scrape cache (no TTL, on/off only) | Populated automatically; safe to delete |
| `pl.duckdb` | Curated DuckDB database: `raw_*` views (recreated on every connection), plus `stg_club_season`, `stg_matches`, `stg_odds`, `mart_fixtures` | `just curate` (after the ingests above) |
