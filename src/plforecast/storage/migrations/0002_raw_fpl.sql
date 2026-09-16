-- Same raw_* convention as 0001: views over the immutable Parquet landing zone, never
-- mutated. Three views because one physical source (the FPL API) backs three logically
-- distinct raw tables -- see src/plforecast/ingest/fpl.py for why.
CREATE OR REPLACE VIEW raw_fpl_teams AS
SELECT *
FROM read_parquet('{raw_dir}/fpl-teams/*/data.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_fpl_player_availability AS
SELECT *
FROM read_parquet('{raw_dir}/fpl-players/*/data.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_fpl_fixtures AS
SELECT *
FROM read_parquet('{raw_dir}/fpl-fixtures/*/data.parquet', union_by_name = true);
