"""Shared ingestion contract: every source adapter fetches raw bytes and parses them
into a typed frame, with no transformation beyond parsing. See docs/data-sources.md
and design.md section 5.2 for the requirements each adapter must satisfy:
immutability, provenance, politeness, caching, schema validation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import httpx
import polars as pl
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class RawPart:
    """One fetched HTTP resource that feeds into a source's payload (e.g. one season's file)."""

    url: str
    status_code: int
    content: bytes
    fetched_at: datetime
    label: str
    from_cache: bool = False


@dataclass(frozen=True, slots=True)
class RawPayload:
    source: str
    fetched_at: datetime
    parts: list[RawPart] = field(default_factory=list)


class Source(Protocol):
    name: str

    def fetch(self, *, since: date | None = None) -> RawPayload: ...

    def parse(self, payload: RawPayload) -> pl.DataFrame: ...


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def _get(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url)
    response.raise_for_status()
    return response


def cached_get(
    url: str,
    *,
    label: str,
    cache_dir: Path,
    ttl_hours: float,
    delay_seconds: float,
    client: httpx.Client,
) -> RawPart:
    """Fetch a URL through a TTL file cache. Politeness delay applies only on a live fetch,
    never on a cache hit, so repeated runs within the TTL do not hammer the source."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(url)}.bin"

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < ttl_hours:
            log.debug("ingest.cache_hit", url=url, age_hours=round(age_hours, 2))
            return RawPart(
                url=url,
                status_code=200,
                content=cache_file.read_bytes(),
                fetched_at=datetime.fromtimestamp(cache_file.stat().st_mtime, tz=UTC),
                label=label,
                from_cache=True,
            )

    response = _get(client, url)
    time.sleep(delay_seconds)
    cache_file.write_bytes(response.content)
    log.info("ingest.fetched", url=url, status=response.status_code, bytes=len(response.content))
    return RawPart(
        url=url,
        status_code=response.status_code,
        content=response.content,
        fetched_at=datetime.now(UTC),
        label=label,
    )


def write_snapshot(
    df: pl.DataFrame,
    *,
    source: str,
    fetched_at: datetime,
    raw_dir: Path,
    parts: list[RawPart],
) -> Path:
    """Land a parsed frame as an immutable, timestamped Parquet snapshot with a provenance
    sidecar. Never overwrites: every call creates a new snapshot directory."""
    snapshot_dir = raw_dir / source / fetched_at.strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    data_path = snapshot_dir / "data.parquet"
    df.write_parquet(data_path)
    content_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    meta = {
        "source": source,
        "fetched_at": fetched_at.isoformat(),
        "row_count": df.height,
        "content_hash": content_hash,
        "parts": [
            {
                "url": part.url,
                "status_code": part.status_code,
                "label": part.label,
                "from_cache": part.from_cache,
                "fetched_at": part.fetched_at.isoformat(),
            }
            for part in parts
        ],
    }
    (snapshot_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("ingest.snapshot_written", source=source, path=str(snapshot_dir), rows=df.height)
    return snapshot_dir
