"""FPL API adapter: fixture list, kickoff times, and player availability/suspensions
(design.md section 5.1). One physical source (fantasy.premierleague.com) backs four
raw tables, so this module implements four independent Source-conforming classes
rather than one:

- FPLTeamsSource: the team roster mapping the per-season numeric `id` used everywhere
  else in the FPL API to the stable `code` the club dimension keys on (section 5.3). It
  is free to land alongside the players endpoint since both come from bootstrap-static.
- FPLPlayersSource: player availability and suspensions.
- FPLFixturesSource: the remaining/played fixture list and kickoff times.
- FPLEventsSource: the gameweek calendar (`bootstrap-static`'s `events` array) --
  deadlines and FPL's own `is_current`/`is_next`/`is_previous`/`finished` flags per
  gameweek. Story B-10: `as_of_gameweek` on the forecast artifact should come from
  FPL's own notion of the current gameweek, not be re-derived by counting finished
  fixtures, and the season label FPL fixtures get stamped with should come from the
  fixture list's own kickoff dates, not the wall clock.

Team and player data both come from the same bootstrap-static endpoint. Each class
fetches it independently (via cached_get) rather than sharing a fetch result, so each
stays testable in isolation per the Source protocol; the TTL cache means the second
class's "fetch" in the same run is a cache hit, not a second live request.

No club-name or club-ID resolution happens here. Fixtures and player rows keep FPL's
raw numeric team IDs; resolving them to canonical club IDs is the entities layer's job
(section 5.3), not ingest's.
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
from plforecast.ingest.base import RawPayload, cached_get, write_snapshot

log = structlog.get_logger()


class TeamSchema(pa.DataFrameModel):
    # `id` is a 1-20 index re-assigned every season; `code` is the stable club identifier
    # the club dimension keys on. Both are landed: fixtures reference teams by `id`.
    fpl_team_id: Series[int] = pa.Field(ge=1, unique=True)
    fpl_code: Series[int] = pa.Field(ge=1, unique=True)
    name: Series[str]
    short_name: Series[str]

    class Config:
        strict = True
        coerce = True


class FixtureSchema(pa.DataFrameModel):
    fpl_fixture_id: Series[int] = pa.Field(ge=1, unique=True)
    # Postponed fixtures can be pulled from a gameweek pending a reschedule, and their
    # kickoff time is unset until one is confirmed -- both nullable for the same reason
    # PSCH/PSCD/PSCA are nullable in the football-data schema.
    gameweek: Series[int] = pa.Field(ge=1, nullable=True)
    # dtype_kwargs pins the timezone: without it, pandera's coerce silently strips tz
    # info back to a naive datetime, which would make kickoff_time look right until it
    # is compared or joined against anything else timestamped in UTC.
    kickoff_time: Series[pl.Datetime] = pa.Field(nullable=True, dtype_kwargs={"time_zone": "UTC"})
    home_team_id: Series[int] = pa.Field(ge=1)
    away_team_id: Series[int] = pa.Field(ge=1)
    home_score: Series[int] = pa.Field(ge=0, nullable=True)
    away_score: Series[int] = pa.Field(ge=0, nullable=True)
    finished: Series[bool]

    class Config:
        strict = True
        coerce = True


class EventSchema(pa.DataFrameModel):
    gameweek: Series[int] = pa.Field(ge=1, unique=True)
    name: Series[str]
    deadline_time: Series[pl.Datetime] = pa.Field(dtype_kwargs={"time_zone": "UTC"})
    finished: Series[bool]
    is_previous: Series[bool]
    is_current: Series[bool]
    is_next: Series[bool]

    class Config:
        strict = True
        coerce = True


class PlayerAvailabilitySchema(pa.DataFrameModel):
    fpl_player_id: Series[int] = pa.Field(ge=1, unique=True)
    team_id: Series[int] = pa.Field(ge=1)
    web_name: Series[str]
    # The five status codes FPL documents: available, doubtful, injured, suspended,
    # unavailable. An unrecognised code should fail loudly rather than pass through --
    # that is what distinguishes a real new status from a silent schema drift.
    status: Series[str] = pa.Field(isin=["a", "d", "i", "s", "u"])
    chance_of_playing_this_round: Series[int] = pa.Field(ge=0, le=100, nullable=True)
    chance_of_playing_next_round: Series[int] = pa.Field(ge=0, le=100, nullable=True)
    news: Series[str]

    class Config:
        strict = True
        coerce = True


def _fetch_raw(url: str, *, label: str, config: Settings, client: httpx.Client) -> RawPayload:
    part = cached_get(
        url,
        label=label,
        cache_dir=config.cache_dir / "fpl",
        ttl_hours=config.cache_ttl_hours,
        delay_seconds=config.request_delay_seconds,
        client=client,
    )
    return RawPayload(source=f"fpl-{label}", fetched_at=datetime.now(UTC), parts=[part])


def _client(config: Settings) -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=config.http_timeout_seconds,
        headers={"User-Agent": config.user_agent},
    )


class FPLTeamsSource:
    name = "fpl-teams"

    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def fetch(self, *, since: date | None = None) -> RawPayload:
        with _client(self.config) as client:
            return _fetch_raw(
                self.config.fpl_bootstrap_url, label="bootstrap", config=self.config, client=client
            )

    def parse(self, payload: RawPayload) -> pl.DataFrame:
        body = json.loads(payload.parts[0].content)
        df = pl.DataFrame(body["teams"]).select(
            pl.col("id").alias("fpl_team_id"),
            pl.col("code").alias("fpl_code"),
            pl.col("name"),
            pl.col("short_name"),
        )
        return TeamSchema.validate(df)


class FPLPlayersSource:
    name = "fpl-players"

    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def fetch(self, *, since: date | None = None) -> RawPayload:
        with _client(self.config) as client:
            return _fetch_raw(
                self.config.fpl_bootstrap_url, label="bootstrap", config=self.config, client=client
            )

    def parse(self, payload: RawPayload) -> pl.DataFrame:
        body = json.loads(payload.parts[0].content)
        df = pl.DataFrame(body["elements"]).select(
            pl.col("id").alias("fpl_player_id"),
            pl.col("team").alias("team_id"),
            pl.col("web_name"),
            pl.col("status"),
            pl.col("chance_of_playing_this_round"),
            pl.col("chance_of_playing_next_round"),
            pl.col("news"),
        )
        return PlayerAvailabilitySchema.validate(df)


class FPLEventsSource:
    name = "fpl-events"

    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def fetch(self, *, since: date | None = None) -> RawPayload:
        with _client(self.config) as client:
            return _fetch_raw(
                self.config.fpl_bootstrap_url, label="bootstrap", config=self.config, client=client
            )

    def parse(self, payload: RawPayload) -> pl.DataFrame:
        body = json.loads(payload.parts[0].content)
        df = pl.DataFrame(body["events"], schema_overrides={"deadline_time": pl.Utf8}).select(
            pl.col("id").alias("gameweek"),
            pl.col("name"),
            pl.col("deadline_time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC"),
            pl.col("finished"),
            pl.col("is_previous"),
            pl.col("is_current"),
            pl.col("is_next"),
        )
        return EventSchema.validate(df)


class FPLFixturesSource:
    name = "fpl-fixtures"

    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def fetch(self, *, since: date | None = None) -> RawPayload:
        with _client(self.config) as client:
            return _fetch_raw(
                self.config.fpl_fixtures_url, label="fixtures", config=self.config, client=client
            )

    def parse(self, payload: RawPayload) -> pl.DataFrame:
        body = json.loads(payload.parts[0].content)
        df = pl.DataFrame(
            body,
            schema_overrides={"kickoff_time": pl.Utf8},
        ).select(
            pl.col("id").alias("fpl_fixture_id"),
            pl.col("event").alias("gameweek"),
            pl.col("kickoff_time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC"),
            pl.col("team_h").alias("home_team_id"),
            pl.col("team_a").alias("away_team_id"),
            pl.col("team_h_score").alias("home_score"),
            pl.col("team_a_score").alias("away_score"),
            pl.col("finished"),
        )
        return FixtureSchema.validate(df)


def ingest() -> None:
    """Land all four FPL-derived raw tables as immutable snapshots."""
    for source in (
        FPLTeamsSource(),
        FPLPlayersSource(),
        FPLFixturesSource(),
        FPLEventsSource(),
    ):
        payload = source.fetch()
        df = source.parse(payload)
        write_snapshot(
            df,
            source=source.name,
            fetched_at=payload.fetched_at,
            raw_dir=settings.raw_dir,
            parts=payload.parts,
        )
