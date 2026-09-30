# data/ manifest

`data/` is gitignored except this file and `.gitkeep`. Nothing here is committed;
everything is reproducible from source given the adapters in `src/plforecast/ingest/`.

| Path | Contents | How to (re)populate |
| --- | --- | --- |
| `raw/football-data/{fetched_at}/data.parquet` | Immutable snapshot of E0 match results + Pinnacle closing odds, 2015/16 onward | `just ingest-footballdata` |
| `raw/fpl-teams/{fetched_at}/data.parquet` | Team roster: per-season `id` and stable `code` | `just ingest-fpl` |
| `raw/fpl-players/{fetched_at}/data.parquet` | Player availability and suspensions | `just ingest-fpl` |
| `raw/fpl-fixtures/{fetched_at}/data.parquet` | Full season fixture list, kickoff times | `just ingest-fpl` |
| `raw/fpl-events/{fetched_at}/data.parquet` | Gameweek calendar: deadlines, FPL's own current/next/previous/finished flags | `just ingest-fpl` |
| `raw/clubelo/{fetched_at}/data.parquet` | Per-club Elo history (`club_id`, `date`, `elo`, `golo`). **Also committed**, outside `data/`, as the fallback seed `seeds/clubelo/{fetched_at}/` -- see below | `just ingest-clubelo` (seed: `just refresh-clubelo-seed`) |
| `raw/understat/{fetched_at}/data.parquet` | Team-level xG, goals, PPDA, 2015/16 onward | `just ingest-understat` |
| `cache/football-data/*.bin`, `cache/fpl/*.bin` | TTL-cached raw HTTP responses, keyed by URL hash | Populated automatically by the adapters; safe to delete |
| `cache/understat-soccerdata/` | soccerdata's own scrape cache (no TTL, on/off only) | Populated automatically; safe to delete |
| `pl.duckdb` | Curated DuckDB database: `raw_*` views (recreated on every connection), `dim_club`, `stg_club_season`, `stg_matches`, `stg_odds`, `stg_fixtures`, `stg_gameweeks`, `mart_team_match` | `just curate` (after the ingests above) |
| `curate-manifest.json` | Row counts and raw-snapshot lineage for every curated table, one per `curate` run | `just curate` |

**Retention**: every ingest run lands a full immutable snapshot and every `raw_*` view unions all of them, so snapshot count is the only thing bounding view-scan cost over time (story B-17). `just prune-raw` (default: keep the 4 most recent per source, the single latest always kept regardless) deletes older snapshot directories; curate's own natural-key dedupe already treats the latest snapshot as authoritative, so pruning changes nothing about curated output, only how much history `data/raw/` retains.

**The one committed exception, `seeds/clubelo/`** (outside `data/`, so it is tracked): clubelo.com 504s GitHub Actions runners, so the scheduled pipeline falls back to this snapshot when a live fetch fails (`ingest clubelo --allow-stale`; the newer of it and a restored cache wins). It holds exactly one snapshot and is refreshed by hand with `just refresh-clubelo-seed`, then committed. `prune-raw` never touches it.
