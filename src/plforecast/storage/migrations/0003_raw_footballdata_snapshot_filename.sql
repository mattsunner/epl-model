-- Redefines raw_footballdata_matches (still the same raw_* convention: a view over the
-- immutable Parquet landing zone, never mutated) to also expose each row's source
-- filename. A re-run of the football-data ingest lands a brand new full-backfill
-- snapshot alongside every prior one (design.md section 5.2: "re-runs create new
-- snapshots"), so this view unions every snapshot ever landed -- including duplicate
-- copies of seasons that haven't changed. stg_matches needs the filename to pick the
-- most recently landed copy of each match deterministically, rather than an arbitrary
-- one, when it dedupes on to its natural key.
CREATE OR REPLACE VIEW raw_footballdata_matches AS
SELECT *
FROM read_parquet(
    '{raw_dir}/football-data/*/data.parquet',
    union_by_name = true,
    filename = true
);
