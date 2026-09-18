"""ClubElo adapter: per-club rating history (design.md section 5.1, story C-16).

The classic `api.clubelo.com` CSV endpoint the original design assumed is gone (a real
finding from this pipeline's own build, not a hypothetical -- it now returns a bare
`502`). The current site, `clubelo.com`, serves one HTML page per club with its rating
history embedded as a Vega-Lite chart spec rather than exposed through any API. This
adapter fetches that page per club and extracts the embedded dataset directly (a JSON
decode from a known marker in the page source, not HTML scraping in the fragile
sense -- the data is a single well-formed JSON literal, just not served on its own).

**Coverage is narrower than the original design expected in two ways.** The site's
cached history only goes back to roughly September 2022, not the full 2015/16 backfill
window every other source covers. And only 32 of the 35 clubs in `club_aliases.yaml`
have a resolvable `clubelo_name` -- Huddersfield, Stoke and Watford could not be found
this way (the site has no directory API; slugs were found by probing its per-country
index pages), and all three have been out of the top flight since before ClubElo's own
cached window starts, so the gap costs nothing real. Clubs with no `clubelo_name` are
skipped, not an error.

Unlike football-data.co.uk's `since` (which narrows a fetch to just the current
season), ClubElo's per-club page always returns that club's full history in one
response -- there is no incremental endpoint. `since` is accepted for interface
symmetry with every other `Source` but does not change what is fetched.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import httpx
import pandera.polars as pa
import polars as pl
import structlog
from pandera.typing.polars import Series

from plforecast.config import Settings, settings
from plforecast.entities.clubs import load_club_dimension
from plforecast.ingest.base import RawPayload, cached_get, write_snapshot

log = structlog.get_logger()

CLUBELO_BASE_URL = "https://clubelo.com"


class ClubEloRatingSchema(pa.DataFrameModel):
    club_id: Series[str]
    date: Series[pl.Date]
    elo: Series[float] = pa.Field(gt=0)
    golo: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


def _extract_vega_json(html: str) -> dict[str, object]:
    """The page's rating-history chart data, embedded as `var vegaJson = {...};` in the
    page source. Raises `ValueError` (from `str.index`) if the marker is missing --
    loud failure on a page the site has restructured again, not a silently empty
    result."""
    marker = "vegaJson = "
    start = html.index(marker) + len(marker)
    obj, _ = json.JSONDecoder().raw_decode(html, start)
    return obj  # type: ignore[no-any-return]


def _rating_rows(html: str, club_id: str) -> list[dict[str, object]]:
    spec = _extract_vega_json(html)
    datasets = spec.get("datasets", {})
    for dataset in datasets.values() if isinstance(datasets, dict) else []:
        if dataset and "Elo" in dataset[0] and "Date" in dataset[0]:
            return [
                {
                    "club_id": club_id,
                    "date": row["Date"][:10],
                    "elo": row["Elo"],
                    "golo": row.get("Golo"),
                }
                for row in dataset
            ]
    return []


class ClubEloSource:
    name = "clubelo"

    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def fetch(self, *, since: date | None = None) -> RawPayload:
        dimension = load_club_dimension()
        clubs = [
            (row["club_id"], row["clubelo_name"])
            for row in dimension.frame.iter_rows(named=True)
            if row["clubelo_name"] is not None
        ]

        parts = []
        headers = {"User-Agent": self.config.user_agent}
        with httpx.Client(
            follow_redirects=True, timeout=self.config.http_timeout_seconds, headers=headers
        ) as client:
            for club_id, slug in clubs:
                parts.append(
                    cached_get(
                        f"{CLUBELO_BASE_URL}/{slug}",
                        label=club_id,
                        cache_dir=self.config.cache_dir / self.name,
                        ttl_hours=self.config.cache_ttl_hours,
                        delay_seconds=self.config.request_delay_seconds,
                        client=client,
                    )
                )

        return RawPayload(source=self.name, fetched_at=datetime.now(UTC), parts=parts)

    def parse(self, payload: RawPayload) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for part in payload.parts:
            club_rows = _rating_rows(part.content.decode("utf-8"), part.label)
            if not club_rows:
                log.warning("clubelo.no_rating_data", club_id=part.label)
            rows.extend(club_rows)

        df = pl.DataFrame(
            rows,
            schema={"club_id": pl.Utf8, "date": pl.Utf8, "elo": pl.Float64, "golo": pl.Float64},
        ).with_columns(pl.col("date").str.to_date())
        return ClubEloRatingSchema.validate(df)


def ingest(config: Settings = settings) -> None:
    """Full refresh: every club with a resolvable `clubelo_name` lands as one immutable
    snapshot. A page missing the expected `vegaJson` marker entirely (the site
    restructured again) raises and fails the whole ingest, the same fail-loud
    convention every other adapter follows -- one `RawPart` per club so a re-run after
    fixing the extraction only needs to re-fetch what changed, not the whole site."""
    source = ClubEloSource(config)
    payload = source.fetch()
    df = source.parse(payload)
    write_snapshot(
        df,
        source=source.name,
        fetched_at=payload.fetched_at,
        raw_dir=config.raw_dir,
        parts=payload.parts,
    )
