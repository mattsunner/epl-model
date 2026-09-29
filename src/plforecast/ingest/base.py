"""Shared ingestion contract: every source adapter fetches raw bytes and parses them
into a typed frame, with no transformation beyond parsing. See docs/data-sources.md
and design.md section 5.2 for the requirements each adapter must satisfy:
immutability, provenance, politeness, caching, schema validation.

Every timestamp here is offset-aware UTC, never naive `datetime.now()`: `write_snapshot`
asserts this on `fetched_at` and raises rather than silently landing a timestamp that
looks right until it is compared against anything else in UTC (the same class of bug
`docs/data-sources.md` documents for FPL's `kickoff_time`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import httpx
import polars as pl
import structlog
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

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


def _is_transient(exc: BaseException) -> bool:
    """Network-level failures, plus 429 and 5xx responses (an overloaded or briefly
    unavailable source, e.g. clubelo.com's 504s). Other 4xx are permanent -- fail fast."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
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


def content_hash(df: pl.DataFrame) -> str:
    """A hash of `df`'s actual values, independent of column order, row order, or the
    Parquet/Arrow writer's own metadata -- which embeds a version-specific schema blob
    that can differ across polars or pyarrow releases for byte-for-byte identical rows,
    so hashing `data.parquet` itself (the previous approach) is not a content identity.
    The site-repo pull workflow (design.md 10.1) compares this hash to decide whether a
    fetched forecast actually changed, so it has to survive a dependency upgrade.

    Column order is fixed by sorting names; each row's own hash (`DataFrame.hash_rows`,
    a per-row value hash polars computes independent of any writer) is combined after
    sorting, so row order does not matter either."""
    ordered = df.select(sorted(df.columns))
    row_hashes = sorted(ordered.hash_rows().to_list())
    payload = ",".join(str(h) for h in row_hashes).encode()
    return hashlib.sha256(payload).hexdigest()


def prune_snapshots(
    raw_dir: Path, *, keep: int, sources: Iterable[str] | None = None
) -> dict[str, list[Path]]:
    """Delete all but the `keep` most recent snapshot directories under each source
    directory in `raw_dir` (story B-17). Snapshot directories sort lexicographically by
    their UTC timestamp name, so the last `keep` after a plain sort are the newest; the
    single most recent snapshot is always kept even when `keep < 1`, since every raw_*
    view must resolve to at least one snapshot (`storage.db.ensure_raw_views` drops a
    view entirely once its source has none). Returns source name -> the paths removed,
    so a caller can log or report what was pruned; removes nothing and returns an empty
    mapping for a source directory that does not exist.

    `sources`, when given, restricts pruning to just those source directory names
    (every other source is left untouched regardless of how many snapshots it has).
    **This is not just a filter, it's a safety requirement** for a source whose ingest
    can land a *partial* snapshot (story: weekly scheduled pipeline's
    `--current-season-only`, football-data and understat): older snapshots there are
    not simply redundant copies the way a full-refetch source's (fpl, clubelo) are --
    an old snapshot may be the only place a finished season's data still exists.
    Pruning one prematurely means the next curate run silently rebuilds its curated
    table with that history missing, since curate always reads from whatever raw
    snapshots currently exist. Callers must pass `sources` explicitly for anything
    that can produce a partial snapshot; `None` (prune everything) is only safe for
    sources that always land a complete dataset."""
    keep = max(keep, 1)
    removed: dict[str, list[Path]] = {}
    if not raw_dir.exists():
        return removed
    source_filter = set(sources) if sources is not None else None
    for source_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if source_filter is not None and source_dir.name not in source_filter:
            continue
        snapshots = sorted(p for p in source_dir.iterdir() if (p / "data.parquet").exists())
        stale = snapshots[:-keep] if keep < len(snapshots) else []
        if stale:
            for snapshot in stale:
                shutil.rmtree(snapshot)
            removed[source_dir.name] = stale
            log.info(
                "ingest.pruned",
                source=source_dir.name,
                removed=[p.name for p in stale],
                kept=[p.name for p in snapshots if p not in stale],
            )
    return removed


def write_snapshot(
    df: pl.DataFrame,
    *,
    source: str,
    fetched_at: datetime,
    raw_dir: Path,
    parts: list[RawPart],
) -> Path:
    """Land a parsed frame as an immutable, timestamped Parquet snapshot with a provenance
    sidecar. Never overwrites: `mkdir(exist_ok=False)` raises rather than silently
    clobbering a snapshot landed in the same second -- two ingests of the same source
    close together is a real case (a retry, a script re-run), not a hypothetical one."""
    if fetched_at.tzinfo is None:
        raise ValueError(
            f"write_snapshot requires an offset-aware fetched_at, got a naive {fetched_at!r}"
        )
    snapshot_dir = raw_dir / source / fetched_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"snapshot directory already exists: {snapshot_dir}. Raw snapshots are "
            "immutable and never overwritten; if this is a genuine same-second re-run, "
            "wait a second and retry."
        ) from exc

    data_path = snapshot_dir / "data.parquet"
    df.write_parquet(data_path)

    meta = {
        "source": source,
        "fetched_at": fetched_at.isoformat(),
        "row_count": df.height,
        "content_hash": content_hash(df),
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
