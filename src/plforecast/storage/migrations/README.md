# Table migrations

Numbered `NNNN_name.sql` files applied once each, in order, by `storage/db.py:migrate()`
and recorded in `schema_migrations`. Reserved for tables whose history matters.

`raw_*` views are **not** migrations: they are recreated on every `connect()` by
`ensure_raw_views()`, because they embed the absolute path of `data/raw/`. Curated
`stg_*`/`mart_*` tables are rebuilt wholesale by `curate` and need no migration either.
