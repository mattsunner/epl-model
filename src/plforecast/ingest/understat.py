"""Understat adapter: team-level xG (design.md section 5.1), via the `soccerdata`
library rather than a bespoke scraper -- section 11.1 anticipates exactly this
("soccerdata and penaltyblog return pandas; convert at the adapter boundary").

Scope: team-level match stats only (goals, xG, non-penalty xG, PPDA), not shot-level
events. Shot-level data needs one HTTP request per match against a scraped,
ToS-sensitive site -- a full 2015/16-2026/27 backfill would be several thousand
requests for data nothing in the pipeline consumes yet. Team-level stats come from one
bulk request per season and are everything an xG-driven strength model needs. Revisit
if a future notebook or model genuinely needs shot data.

Team names are Understat's own spelling (e.g. "Newcastle United" vs football-data's
"Newcastle") -- resolving them to canonical club IDs is the entities layer's job
(design.md section 5.3), not ingest's. `club_aliases.yaml`'s `understat_name` column is
still unpopulated; that is the next piece of work this adapter unblocks, not something
it does itself.

Caching: soccerdata has no TTL concept of its own, only an on/off cache. Completed
seasons are immutable, so their scrape is reused from the cache. The current season is
always fetched live (`no_cache=True`), whether the run is a full backfill or an
incremental refresh (`since` set): a cached copy of an in-progress season is stale by
definition, and reusing it would land last week's 40 matches every week (story B-08).

Provenance: one RawPart per season batch (matching the other adapters' one-part-per-
season convention as closely as soccerdata's bulk-fetch API allows), so `_meta.json`
records each batch's own cache state via `from_cache`, not one blanket value for the
whole fetch. The part's pseudo-URL also carries the exact `soccerdata` version, since
this adapter's actual provenance is "this scraper library, this version", unlike the
other two adapters where the URL alone identifies the source (story B-15).

Season input format: seasons are passed to soccerdata as `season_code()` pair-code
strings (e.g. "2122"), never bare integers. Confirmed by hand against the live site: a
bare integer season (e.g. `2021`, intending 2021/22) is silently misinterpreted by
soccerdata as 2020/21 instead, because the string "2021" is self-referentially
ambiguous -- see `season_code`'s docstring in config.py for the full explanation. A real
backfill run against this exact bug landed 3,840 rows instead of 4,220, with the entire
2021/22 season missing -- not a hypothetical edge case.
"""

from __future__ import annotations

import io
from datetime import UTC, date, datetime

import pandas as pd
import pandera.polars as pa
import polars as pl
import soccerdata as sd
import structlog
from pandera.typing.polars import Series

from plforecast.config import Settings, season_code, season_label, settings
from plforecast.ingest.base import RawPart, RawPayload, write_snapshot

log = structlog.get_logger()


class TeamMatchXGSchema(pa.DataFrameModel):
    season: Series[str] = pa.Field(str_matches=r"^\d{4}/\d{2}$")
    date: Series[pl.Date]
    home_team: Series[str]
    away_team: Series[str]
    home_goals: Series[int] = pa.Field(ge=0)
    away_goals: Series[int] = pa.Field(ge=0)
    home_xg: Series[float] = pa.Field(ge=0)
    away_xg: Series[float] = pa.Field(ge=0)
    home_np_xg: Series[float] = pa.Field(ge=0)
    away_np_xg: Series[float] = pa.Field(ge=0)
    home_ppda: Series[float] = pa.Field(ge=0)
    away_ppda: Series[float] = pa.Field(ge=0)

    class Config:
        strict = True
        coerce = True


def season_batches(start_years: list[int], current_start_year: int) -> list[tuple[list[int], bool]]:
    """Split requested seasons into (years, no_cache) batches: completed seasons from
    the cache, the current season always live."""
    completed = [y for y in start_years if y < current_start_year]
    current = [y for y in start_years if y >= current_start_year]
    batches: list[tuple[list[int], bool]] = []
    if completed:
        batches.append((completed, False))
    if current:
        batches.append((current, True))
    return batches


class UnderstatSource:
    name = "understat"

    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def fetch(self, *, since: date | None = None) -> RawPayload:
        current_start_year = self.config.current_season_start_year()
        if since is None:
            start_years = list(range(self.config.backfill_start_season, current_start_year + 1))
        else:
            start_years = [current_start_year]

        parts = []
        now = datetime.now(UTC)
        for years, no_cache in season_batches(start_years, current_start_year):
            scraper = sd.Understat(
                leagues=self.config.understat_league,
                seasons=[season_code(year) for year in years],
                data_dir=self.config.cache_dir / "understat-soccerdata",
                no_cache=no_cache,
            )
            # soccerdata prints a `Season id "2021" is ambiguous` UserWarning whenever the
            # 2020/21 season ("2021" as a pair-code) is in the requested range. Harmless
            # noise, already confirmed correct above for our pair-code-string input -- not
            # suppressed here because soccerdata fetches seasons from a worker thread, so a
            # filter registered in this thread does not reliably reach it.
            batch = scraper.read_team_match_stats().reset_index()
            label = season_label(years[0]) + (
                f"-{season_label(years[-1])}" if len(years) > 1 else ""
            )
            log.info(
                "understat.fetched", seasons=[season_label(y) for y in years], no_cache=no_cache
            )

            buffer = io.BytesIO()
            batch.to_parquet(buffer)
            parts.append(
                RawPart(
                    url=f"understat+soccerdata=={sd.__version__}:{self.config.understat_league}:{label}",
                    status_code=200,
                    content=buffer.getvalue(),
                    fetched_at=now,
                    label=label,
                    from_cache=not no_cache,
                )
            )

        return RawPayload(source=self.name, fetched_at=now, parts=parts)

    def parse(self, payload: RawPayload) -> pl.DataFrame:
        batches = [pd.read_parquet(io.BytesIO(part.content)) for part in payload.parts]
        raw = pl.from_pandas(pd.concat(batches, ignore_index=True))

        df = raw.select(
            pl.col("season_id").map_elements(season_label, return_dtype=pl.Utf8).alias("season"),
            pl.col("date").dt.date().alias("date"),
            pl.col("home_team"),
            pl.col("away_team"),
            pl.col("home_goals").cast(pl.Int64),
            pl.col("away_goals").cast(pl.Int64),
            pl.col("home_xg").cast(pl.Float64),
            pl.col("away_xg").cast(pl.Float64),
            pl.col("home_np_xg").cast(pl.Float64),
            pl.col("away_np_xg").cast(pl.Float64),
            pl.col("home_ppda").cast(pl.Float64),
            pl.col("away_ppda").cast(pl.Float64),
        )
        return TeamMatchXGSchema.validate(df)


def ingest(config: Settings = settings) -> None:
    """Full backfill: land every configured season as one immutable snapshot."""
    source = UnderstatSource(config)
    payload = source.fetch()
    df = source.parse(payload)
    write_snapshot(
        df,
        source=source.name,
        fetched_at=payload.fetched_at,
        raw_dir=config.raw_dir,
        parts=payload.parts,
    )
