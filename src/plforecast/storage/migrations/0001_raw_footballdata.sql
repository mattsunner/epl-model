-- raw_* views are never mutated: they read straight from the immutable Parquet landing
-- zone. Every snapshot directory under data/raw/football-data/ is a full backfill or
-- refresh; this view always reflects the most recently landed snapshot.
CREATE OR REPLACE VIEW raw_footballdata_matches AS
SELECT *
FROM read_parquet('{raw_dir}/football-data/*/data.parquet', union_by_name = true);
