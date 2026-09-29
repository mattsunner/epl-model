import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from plforecast.ingest import base
from plforecast.ingest.base import (
    RawPart,
    cached_get,
    content_hash,
    prune_snapshots,
    write_snapshot,
)


class _StubTransport(httpx.BaseTransport):
    """Counts requests and always returns the same body, so a test can assert exactly
    how many live HTTP calls a TTL cache actually avoided."""

    def __init__(self, body: bytes = b"hello") -> None:
        self.body = body
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(200, content=self.body)


class _SequenceTransport(httpx.BaseTransport):
    """Replays a fixed list of status codes, one per request (the last repeats), so a
    test can script a source that fails a few times and then recovers."""

    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        return httpx.Response(status, content=b"ok" if status == 200 else b"")


def _client(transport: httpx.BaseTransport) -> httpx.Client:
    return httpx.Client(transport=transport)


def test_cached_get_is_a_live_fetch_on_a_cold_cache(tmp_path: Path):
    transport = _StubTransport()
    part = cached_get(
        "https://example.test/a",
        label="a",
        cache_dir=tmp_path,
        ttl_hours=24,
        delay_seconds=0,
        client=_client(transport),
    )
    assert transport.calls == 1
    assert part.from_cache is False
    assert part.content == b"hello"
    assert part.status_code == 200


def test_cached_get_within_ttl_is_a_cache_hit_and_makes_no_request(tmp_path: Path):
    transport = _StubTransport()
    client = _client(transport)
    cached_get(
        "https://example.test/a",
        label="a",
        cache_dir=tmp_path,
        ttl_hours=24,
        delay_seconds=0,
        client=client,
    )
    second = cached_get(
        "https://example.test/a",
        label="a",
        cache_dir=tmp_path,
        ttl_hours=24,
        delay_seconds=0,
        client=client,
    )
    assert transport.calls == 1  # only the first call hit the network
    assert second.from_cache is True
    assert second.content == b"hello"


def test_cached_get_past_ttl_refetches(tmp_path: Path):
    transport = _StubTransport()
    client = _client(transport)
    cached_get(
        "https://example.test/a",
        label="a",
        cache_dir=tmp_path,
        ttl_hours=24,
        delay_seconds=0,
        client=client,
    )
    # A TTL of 0 hours means the just-written cache file is already stale.
    second = cached_get(
        "https://example.test/a",
        label="a",
        cache_dir=tmp_path,
        ttl_hours=0,
        delay_seconds=0,
        client=client,
    )
    assert transport.calls == 2
    assert second.from_cache is False


def test_politeness_delay_applies_only_on_a_live_fetch_not_a_cache_hit(tmp_path: Path):
    transport = _StubTransport()
    client = _client(transport)
    start = time.monotonic()
    cached_get(
        "https://example.test/a",
        label="a",
        cache_dir=tmp_path,
        ttl_hours=24,
        delay_seconds=0.2,
        client=client,
    )
    after_live = time.monotonic()
    cached_get(
        "https://example.test/a",
        label="a",
        cache_dir=tmp_path,
        ttl_hours=24,
        delay_seconds=0.2,
        client=client,
    )
    after_cached = time.monotonic()

    assert after_live - start >= 0.2  # the live fetch paid the delay
    assert after_cached - after_live < 0.1  # the cache hit did not


def test_content_hash_is_stable_across_two_writes(tmp_path: Path):
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.write_parquet(tmp_path / "one.parquet")
    df.write_parquet(tmp_path / "two.parquet")

    # Re-reading from two separately-written files, not the same in-memory frame,
    # exercises exactly what write_snapshot hashes: values, not writer metadata.
    one = pl.read_parquet(tmp_path / "one.parquet")
    two = pl.read_parquet(tmp_path / "two.parquet")
    assert content_hash(one) == content_hash(two)


def test_content_hash_changes_when_a_value_changes():
    a = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    b = pl.DataFrame({"a": [1, 2], "b": ["x", "z"]})
    assert content_hash(a) != content_hash(b)


def test_content_hash_is_independent_of_column_and_row_order():
    a = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    reordered_cols = a.select("b", "a")
    reordered_rows = a.reverse()
    assert content_hash(a) == content_hash(reordered_cols)
    assert content_hash(a) == content_hash(reordered_rows)


def _parts() -> list[RawPart]:
    return [
        RawPart(
            url="https://example.test/a",
            status_code=200,
            content=b"x",
            fetched_at=datetime.now(UTC),
            label="a",
        )
    ]


def test_write_snapshot_rejects_a_naive_fetched_at(tmp_path: Path):
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="offset-aware"):
        write_snapshot(
            df, source="test", fetched_at=datetime.now(), raw_dir=tmp_path, parts=_parts()
        )


def test_write_snapshot_lands_a_snapshot_with_a_utc_directory_name_and_meta(tmp_path: Path):
    df = pl.DataFrame({"a": [1, 2]})
    fetched_at = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)

    snapshot_dir = write_snapshot(
        df, source="test", fetched_at=fetched_at, raw_dir=tmp_path, parts=_parts()
    )

    assert snapshot_dir.name == "20260916T120000Z"
    assert (snapshot_dir / "data.parquet").exists()
    import json

    meta = json.loads((snapshot_dir / "_meta.json").read_text())
    assert meta["row_count"] == 2
    assert meta["fetched_at"] == fetched_at.isoformat()
    assert meta["content_hash"] == content_hash(df)
    assert meta["parts"][0]["url"] == "https://example.test/a"


def test_write_snapshot_never_overwrites_an_existing_snapshot(tmp_path: Path):
    df = pl.DataFrame({"a": [1]})
    fetched_at = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)

    write_snapshot(df, source="test", fetched_at=fetched_at, raw_dir=tmp_path, parts=_parts())
    with pytest.raises(FileExistsError, match="already exists"):
        write_snapshot(df, source="test", fetched_at=fetched_at, raw_dir=tmp_path, parts=_parts())


def test_write_snapshot_a_second_apart_lands_two_snapshots(tmp_path: Path):
    df = pl.DataFrame({"a": [1]})
    first = write_snapshot(
        df,
        source="test",
        fetched_at=datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC),
        raw_dir=tmp_path,
        parts=_parts(),
    )
    second = write_snapshot(
        df,
        source="test",
        fetched_at=datetime(2026, 9, 16, 12, 0, 1, tzinfo=UTC),
        raw_dir=tmp_path,
        parts=_parts(),
    )
    assert first != second
    assert first.exists() and second.exists()


def _snapshot_names(raw_dir: Path, source: str) -> set[str]:
    return {p.name for p in (raw_dir / source).iterdir()}


def _land(raw_dir: Path, source: str, stamp: str) -> None:
    write_snapshot(
        pl.DataFrame({"a": [1]}),
        source=source,
        fetched_at=datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC),
        raw_dir=raw_dir,
        parts=_parts(),
    )


def test_prune_snapshots_keeps_only_the_n_most_recent_per_source(tmp_path: Path):
    for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
        _land(tmp_path, "football-data", stamp)

    removed = prune_snapshots(tmp_path, keep=2)

    assert [p.name for p in removed["football-data"]] == ["20260101T000000Z"]
    assert _snapshot_names(tmp_path, "football-data") == {"20260102T000000Z", "20260103T000000Z"}


def test_prune_snapshots_always_keeps_at_least_the_latest_even_with_keep_zero(tmp_path: Path):
    for stamp in ("20260101T000000Z", "20260102T000000Z"):
        _land(tmp_path, "football-data", stamp)

    prune_snapshots(tmp_path, keep=0)

    assert _snapshot_names(tmp_path, "football-data") == {"20260102T000000Z"}


def test_prune_snapshots_is_a_noop_when_within_the_keep_limit(tmp_path: Path):
    _land(tmp_path, "football-data", "20260101T000000Z")

    removed = prune_snapshots(tmp_path, keep=4)

    assert removed == {}
    assert _snapshot_names(tmp_path, "football-data") == {"20260101T000000Z"}


def test_prune_snapshots_handles_multiple_sources_independently(tmp_path: Path):
    for stamp in ("20260101T000000Z", "20260102T000000Z"):
        _land(tmp_path, "football-data", stamp)
    _land(tmp_path, "understat", "20260101T000000Z")

    removed = prune_snapshots(tmp_path, keep=1)

    assert set(removed) == {"football-data"}  # understat was already within the limit
    assert _snapshot_names(tmp_path, "understat") == {"20260101T000000Z"}


def test_prune_snapshots_on_a_missing_raw_dir_is_a_noop(tmp_path: Path):
    assert prune_snapshots(tmp_path / "does-not-exist", keep=4) == {}


def test_prune_snapshots_sources_filter_leaves_other_sources_untouched(tmp_path: Path):
    """The safety requirement for a source that can land a partial snapshot (story:
    weekly scheduled pipeline's --current-season-only): passing `sources` must prune
    only the named sources, not everything with more than `keep`."""
    for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
        _land(tmp_path, "football-data", stamp)
        _land(tmp_path, "clubelo", stamp)

    removed = prune_snapshots(tmp_path, keep=1, sources=["clubelo"])

    assert set(removed) == {"clubelo"}
    assert _snapshot_names(tmp_path, "football-data") == {
        "20260101T000000Z",
        "20260102T000000Z",
        "20260103T000000Z",
    }
    assert _snapshot_names(tmp_path, "clubelo") == {"20260103T000000Z"}


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip tenacity's real exponential-backoff sleeps so retry tests run instantly."""
    monkeypatch.setattr(base._get.retry, "sleep", lambda _seconds: None)  # type: ignore[attr-defined]


@pytest.mark.parametrize("status", [429, 500, 502, 504])
def test_get_retries_transient_status_then_succeeds(status: int, no_backoff: None):
    transport = _SequenceTransport([status, status, 200])
    response = base._get(_client(transport), "https://example.test/a")
    assert response.status_code == 200
    assert transport.calls == 3


def test_get_gives_up_after_five_attempts_and_raises_the_status_error(no_backoff: None):
    transport = _SequenceTransport([504])
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        base._get(_client(transport), "https://example.test/a")
    assert exc_info.value.response.status_code == 504
    assert transport.calls == 5


@pytest.mark.parametrize("status", [400, 403, 404])
def test_get_does_not_retry_permanent_client_errors(status: int, no_backoff: None):
    transport = _SequenceTransport([status])
    with pytest.raises(httpx.HTTPStatusError):
        base._get(_client(transport), "https://example.test/a")
    assert transport.calls == 1
