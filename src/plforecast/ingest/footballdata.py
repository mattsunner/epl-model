"""football-data.co.uk adapter: E0 (Premier League) match results and Pinnacle closing odds.

Only the columns the pipeline actually depends on are kept at parse time (see
docs/data-sources.md): match identity, final score, and PSCH/PSCD/PSCA. The full
bookmaker-odds column set is dropped deliberately -- it changes shape every season as
bookmakers stop or start reporting, so concatenating raw files across seasons on the
full column set is not viable. That drift is exactly what schema validation below
guards against for the columns that do matter.
"""

from __future__ import annotations

from datetime import date, datetime

import httpx
import pandera.polars as pa
import polars as pl
import structlog
from pandera.typing.polars import Series

from plforecast.config import Settings, season_code, season_label, settings
from plforecast.ingest.base import RawPayload, cached_get, write_snapshot

log = structlog.get_logger()


class MatchSchema(pa.DataFrameModel):
    season: Series[str] = pa.Field(str_matches=r"^\d{4}/\d{2}$")
    date: Series[pl.Date]
    home_team: Series[str]
    away_team: Series[str]
    fthg: Series[int] = pa.Field(ge=0)
    ftag: Series[int] = pa.Field(ge=0)
    ftr: Series[str] = pa.Field(isin=["H", "D", "A"])
    # Closing odds are occasionally missing for early-season or thinly-covered fixtures.
    psch: Series[float] = pa.Field(nullable=True, ge=1.0)
    pscd: Series[float] = pa.Field(nullable=True, ge=1.0)
    psca: Series[float] = pa.Field(nullable=True, ge=1.0)

    class Config:
        strict = True
        coerce = True


class FootballDataSource:
    name = "football-data"

    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def fetch(self, *, since: date | None = None) -> RawPayload:
        """With since=None, fetch the full configured backfill range (start_season through
        the current season). With since set, only the current season's file is refetched --
        finished seasons are immutable, so there is nothing else that could have changed."""
        current_start_year = self.config.current_season_start_year()
        if since is None:
            start_years = range(self.config.backfill_start_season, current_start_year + 1)
        else:
            start_years = range(current_start_year, current_start_year + 1)

        parts = []
        headers = {"User-Agent": self.config.user_agent}
        with httpx.Client(
            follow_redirects=True, timeout=self.config.http_timeout_seconds, headers=headers
        ) as client:
            for start_year in start_years:
                url = f"{self.config.football_data_base_url}/{season_code(start_year)}/E0.csv"
                parts.append(
                    cached_get(
                        url,
                        label=season_label(start_year),
                        cache_dir=self.config.cache_dir / self.name,
                        ttl_hours=self.config.cache_ttl_hours,
                        delay_seconds=self.config.request_delay_seconds,
                        client=client,
                    )
                )

        return RawPayload(source=self.name, fetched_at=datetime.now(), parts=parts)

    def parse(self, payload: RawPayload) -> pl.DataFrame:
        frames = []
        for part in payload.parts:
            raw = pl.read_csv(part.content, encoding="utf8-lossy", infer_schema_length=0)

            # Pinnacle closing odds (PSCH/PSCD/PSCA) are not guaranteed to exist in every
            # season's file -- the site's bookmaker column set changes as individual books
            # stop or start reporting. Missing entirely is a stronger case of the same
            # "occasionally missing" nullability already modelled in MatchSchema, not an
            # error: surface it loudly via a log so it doesn't pass unnoticed, since the
            # market baseline in design.md section 8.3 depends on this price source.
            odds_cols = {"PSCH": "psch", "PSCD": "pscd", "PSCA": "psca"}
            missing = [c for c in odds_cols if c not in raw.columns]
            if missing:
                log.warning(
                    "footballdata.pinnacle_closing_odds_missing",
                    season=part.label,
                    missing_columns=missing,
                )

            frame = raw.select(
                pl.lit(part.label).alias("season"),
                # The site is inconsistent about year width across seasons: some files use
                # DD/MM/YYYY, others (e.g. 2016/17) use DD/MM/YY. A fixed "%d/%m/%Y" format
                # doesn't error on the short form, it silently parses "16" as the year 16 CE.
                # String length disambiguates unambiguously since every real match date is
                # exactly 8 or 10 characters in this "DD/MM/YY[YY]" shape.
                pl.when(pl.col("Date").str.strip_chars().str.len_chars() == 8)
                .then(pl.col("Date").str.strip_chars().str.to_date("%d/%m/%y"))
                .otherwise(pl.col("Date").str.strip_chars().str.to_date("%d/%m/%Y"))
                .alias("date"),
                pl.col("HomeTeam").alias("home_team"),
                pl.col("AwayTeam").alias("away_team"),
                pl.col("FTHG").cast(pl.Int64).alias("fthg"),
                pl.col("FTAG").cast(pl.Int64).alias("ftag"),
                pl.col("FTR").alias("ftr"),
                *(
                    (
                        pl.col(raw_col).cast(pl.Float64, strict=False).alias(alias)
                        if raw_col in raw.columns
                        else pl.lit(None, dtype=pl.Float64).alias(alias)
                    )
                    for raw_col, alias in odds_cols.items()
                ),
            )
            frames.append(frame)

        df = pl.concat(frames, how="vertical")
        return MatchSchema.validate(df)


def ingest() -> None:
    """Full backfill: land every configured season as one immutable snapshot."""
    source = FootballDataSource()
    payload = source.fetch()
    df = source.parse(payload)
    write_snapshot(
        df,
        source=source.name,
        fetched_at=payload.fetched_at,
        raw_dir=settings.raw_dir,
        parts=payload.parts,
    )
