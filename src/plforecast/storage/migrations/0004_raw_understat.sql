-- Same raw_* convention as the others: a view over the immutable Parquet landing zone,
-- never mutated.
CREATE OR REPLACE VIEW raw_understat_team_match AS
SELECT *
FROM read_parquet(
    '{raw_dir}/understat/*/data.parquet',
    union_by_name = true,
    filename = true
);
