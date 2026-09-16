"""football-data.co.uk adapter: E0 (Premier League) match results and closing odds.

Only the columns the pipeline depends on are kept at parse time (see
docs/data-sources.md): match identity, final score, and every *closing* 1X2 price the
site publishes that the market benchmark can fall back to. The wider bookmaker column
set is dropped deliberately -- it changes shape every season as bookmakers stop or start
reporting, so concatenating raw files across seasons on the full column set is not
viable. Closing prices are themselves subject to that drift (Pinnacle closing vanished
from the 2026/27 file), which is why several sets are landed and a fallback chain is
applied at curate time (design.md section 8.3, ADR 0007).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pandera.polars as pa
import polars as pl
import structlog
from pandera.typing.polars import Series

from plforecast.config import Settings, season_code, season_label, settings
from plforecast.ingest.base import RawPayload, cached_get, write_snapshot

log = structlog.get_logger()


# Closing 1X2 price sets landed from the site, keyed by the site's column prefix. The
# parsed column names are the site's own names lowercased (PSCH -> psch), so a row can
# always be traced back to the source column. Coverage differs by season: only PSC*
# exists before 2019/20; B365C*, MaxC*, AvgC* appear from 2019/20; BFEC* (Betfair
# Exchange) from 2024/25; PSC* is absent from 2026/27. See docs/data-sources.md.
CLOSING_ODDS_SETS: dict[str, str] = {
    "PSC": "pinnacle",
    "BFEC": "betfair_exchange",
    "B365C": "bet365",
    "MaxC": "market_max",
    "AvgC": "market_avg",
}


def closing_odds_columns(prefix: str) -> tuple[str, str, str]:
    """('psch', 'pscd', 'psca') for prefix 'PSC': the parsed home/draw/away column names."""
    return tuple(f"{prefix}{suffix}".lower() for suffix in ("H", "D", "A"))  # type: ignore[return-value]


class MatchSchema(pa.DataFrameModel):
    season: Series[str] = pa.Field(str_matches=r"^\d{4}/\d{2}$")
    date: Series[pl.Date]
    home_team: Series[str]
    away_team: Series[str]
    fthg: Series[int] = pa.Field(ge=0)
    ftag: Series[int] = pa.Field(ge=0)
    ftr: Series[str] = pa.Field(isin=["H", "D", "A"])
    # Every closing price is nullable: a set can be missing for a season (column absent)
    # or for individual thinly-covered fixtures (cell empty).
    psch: Series[float] = pa.Field(nullable=True, ge=1.0)
    pscd: Series[float] = pa.Field(nullable=True, ge=1.0)
    psca: Series[float] = pa.Field(nullable=True, ge=1.0)
    bfech: Series[float] = pa.Field(nullable=True, ge=1.0)
    bfecd: Series[float] = pa.Field(nullable=True, ge=1.0)
    bfeca: Series[float] = pa.Field(nullable=True, ge=1.0)
    b365ch: Series[float] = pa.Field(nullable=True, ge=1.0)
    b365cd: Series[float] = pa.Field(nullable=True, ge=1.0)
    b365ca: Series[float] = pa.Field(nullable=True, ge=1.0)
    maxch: Series[float] = pa.Field(nullable=True, ge=1.0)
    maxcd: Series[float] = pa.Field(nullable=True, ge=1.0)
    maxca: Series[float] = pa.Field(nullable=True, ge=1.0)
    avgch: Series[float] = pa.Field(nullable=True, ge=1.0)
    avgcd: Series[float] = pa.Field(nullable=True, ge=1.0)
    avgca: Series[float] = pa.Field(nullable=True, ge=1.0)

    class Config:
        strict = True
        coerce = True


def _closing_odds_exprs(present: set[str]) -> list[pl.Expr]:
    """One Float64 expression per closing-odds column: the parsed source column for
    sets in `present`, a null literal for the rest."""
    exprs = []
    for prefix in CLOSING_ODDS_SETS:
        for suffix, alias in zip(("H", "D", "A"), closing_odds_columns(prefix), strict=True):
            exprs.append(
                pl.col(f"{prefix}{suffix}").cast(pl.Float64, strict=False).alias(alias)
                if prefix in present
                else pl.lit(None, dtype=pl.Float64).alias(alias)
            )
    return exprs


def _coercion_failures(raw: pl.DataFrame, present: set[str]) -> dict[str, int]:
    """Column -> count of values that are non-blank in the raw string column but land
    null after `cast(Float64, strict=False)`: a genuinely unparsable value ("N/A", a
    stray comma), not the ordinary "cell was empty" case `MatchSchema`'s nullable odds
    fields already tolerate. The two are different events -- an empty cell is expected
    coverage gap, an unparsable one is the site's data changing shape under us -- so
    they must not be conflated into one silent null."""
    failures: dict[str, int] = {}
    for prefix in present:
        for suffix in ("H", "D", "A"):
            raw_col = f"{prefix}{suffix}"
            count = (
                raw.select(
                    (
                        pl.col(raw_col).str.strip_chars().ne("")
                        & pl.col(raw_col).cast(pl.Float64, strict=False).is_null()
                    ).sum()
                )
                .to_series()
                .item()
            )
            if count:
                failures[raw_col] = int(count)
    return failures


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

        return RawPayload(source=self.name, fetched_at=datetime.now(UTC), parts=parts)

    def parse(self, payload: RawPayload) -> pl.DataFrame:
        frames = []
        for part in payload.parts:
            raw = pl.read_csv(part.content, encoding="utf8-lossy", infer_schema_length=0)

            # Which closing-price sets this season's file carries. Missing sets land as
            # nulls, not errors: the market benchmark (design.md section 8.3) falls back
            # across sets at curate time, but a season with *no* closing prices at all
            # would leave the benchmark empty, so that case is logged loudly.
            present = {
                prefix
                for prefix in CLOSING_ODDS_SETS
                if all(f"{prefix}{suffix}" in raw.columns for suffix in ("H", "D", "A"))
            }
            missing = sorted(set(CLOSING_ODDS_SETS) - present)
            if missing:
                log.info(
                    "footballdata.closing_odds_sets_missing",
                    season=part.label,
                    missing=missing,
                    present=sorted(present),
                )
            if not present:
                log.warning("footballdata.no_closing_odds_at_all", season=part.label)

            failures = _coercion_failures(raw, present)
            if failures:
                log.warning(
                    "footballdata.odds_coercion_failed",
                    season=part.label,
                    failures=failures,
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
                *_closing_odds_exprs(present),
            )
            frames.append(frame)

        df = pl.concat(frames, how="vertical")
        return MatchSchema.validate(df)


def ingest(config: Settings = settings) -> None:
    """Full backfill: land every configured season as one immutable snapshot."""
    source = FootballDataSource(config)
    payload = source.fetch()
    df = source.parse(payload)
    write_snapshot(
        df,
        source=source.name,
        fetched_at=payload.fetched_at,
        raw_dir=config.raw_dir,
        parts=payload.parts,
    )
