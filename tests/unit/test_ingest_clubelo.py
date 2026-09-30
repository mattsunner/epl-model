from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from plforecast.ingest.base import RawPart, RawPayload
from plforecast.ingest.clubelo import (
    ClubEloSource,
    StaleClubEloError,
    _extract_vega_json,
    ingest,
    refresh_seed,
    snapshot_age_days,
    use_fallback_snapshot,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "clubelo" / "sample_page.html"


def _payload(html: str, *, label: str = "arsenal") -> RawPayload:
    part = RawPart(
        url=f"https://clubelo.com/{label}",
        status_code=200,
        content=html.encode("utf-8"),
        fetched_at=datetime.now(),
        label=label,
    )
    return RawPayload(source="clubelo", fetched_at=datetime.now(), parts=[part])


def test_extract_vega_json_finds_the_embedded_dataset():
    spec = _extract_vega_json(FIXTURE.read_text())
    assert "data-sample" in spec["datasets"]
    assert len(spec["datasets"]["data-sample"]) == 3


def test_extract_vega_json_raises_loudly_when_marker_is_missing():
    with pytest.raises(ValueError):
        _extract_vega_json("<html><body>no chart here</body></html>")


def test_parse_lands_club_id_date_elo_golo():
    df = ClubEloSource().parse(_payload(FIXTURE.read_text(), label="arsenal"))

    assert df.height == 3
    assert df["club_id"].to_list() == ["arsenal"] * 3
    assert df["date"][0].isoformat() == "2022-09-18"
    assert df["elo"][0] == pytest.approx(1880.163985793919)
    assert df["golo"][0] == pytest.approx(1.4066594)


def test_parse_concatenates_every_club_part():
    payload = RawPayload(
        source="clubelo",
        fetched_at=datetime.now(),
        parts=[
            RawPart(
                url="https://clubelo.com/Arsenal",
                status_code=200,
                content=FIXTURE.read_bytes(),
                fetched_at=datetime.now(),
                label="arsenal",
            ),
            RawPart(
                url="https://clubelo.com/ManCity",
                status_code=200,
                content=FIXTURE.read_bytes(),
                fetched_at=datetime.now(),
                label="man-city",
            ),
        ],
    )
    df = ClubEloSource().parse(payload)

    assert df.height == 6
    assert set(df["club_id"]) == {"arsenal", "man-city"}


def test_parse_raises_on_a_page_missing_the_vega_marker():
    with pytest.raises(ValueError):
        ClubEloSource().parse(_payload("<html><body>restructured page</body></html>"))


def test_committed_seed_snapshot_is_a_valid_clubelo_snapshot():
    """The seed is what the weekly pipeline falls back to when clubelo.com is
    unreachable from CI, so a deleted or corrupted one must fail a test, not a Tuesday."""
    import polars as pl

    from plforecast.config import Settings
    from plforecast.ingest.clubelo import ClubEloRatingSchema

    seed_dir = Settings().clubelo_seed_dir
    snapshots = sorted(p for p in seed_dir.iterdir() if p.is_dir())
    assert snapshots, f"no seed snapshot committed under {seed_dir}"

    df = pl.read_parquet(snapshots[-1] / "data.parquet")
    ClubEloRatingSchema.validate(df)
    assert df["club_id"].n_unique() >= 30  # 32 of the 35 aliased clubs have a ClubElo page
    assert (snapshots[-1] / "_meta.json").exists()


# --- live-fetch fallback (allow_stale) ---------------------------------------------------


def _landed_snapshot(root: Path, name: str) -> Path:
    """A minimal snapshot directory: `_snapshot_dirs` only needs `data.parquet` to exist."""
    snapshot = root / name
    snapshot.mkdir(parents=True)
    (snapshot / "data.parquet").write_bytes(b"parquet")
    return snapshot


def _settings(tmp_path: Path):
    from plforecast.config import Settings

    return Settings(data_dir=tmp_path / "data", clubelo_seed_dir=tmp_path / "seed")


def _failing_fetch(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def boom(self, *, since=None):
        raise exc

    monkeypatch.setattr(ClubEloSource, "fetch", boom)


def _gateway_timeout() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://clubelo.com/Arsenal")
    return httpx.HTTPStatusError(
        "504", request=request, response=httpx.Response(504, request=request)
    )


def test_ingest_falls_back_to_the_seed_when_the_live_fetch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _settings(tmp_path)
    _landed_snapshot(config.clubelo_seed_dir, "20260929T202827Z")
    _failing_fetch(monkeypatch, _gateway_timeout())

    ingest(config, allow_stale=True)

    assert (config.raw_dir / "clubelo" / "20260929T202827Z" / "data.parquet").exists()


def test_ingest_falls_back_on_a_transport_error_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _settings(tmp_path)
    _landed_snapshot(config.clubelo_seed_dir, "20260929T202827Z")
    _failing_fetch(monkeypatch, httpx.ReadTimeout("read operation timed out"))

    ingest(config, allow_stale=True)

    assert (config.raw_dir / "clubelo" / "20260929T202827Z").exists()


def test_ingest_still_raises_without_allow_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _settings(tmp_path)
    _landed_snapshot(config.clubelo_seed_dir, "20260929T202827Z")
    _failing_fetch(monkeypatch, _gateway_timeout())

    with pytest.raises(httpx.HTTPStatusError):
        ingest(config)

    assert not (config.raw_dir / "clubelo").exists()


def test_ingest_raises_when_there_is_nothing_to_fall_back_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _settings(tmp_path)
    _failing_fetch(monkeypatch, _gateway_timeout())

    with pytest.raises(FileNotFoundError):
        ingest(config, allow_stale=True)


def test_a_parse_failure_is_never_masked_by_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The site restructuring is a different failure from the site being unreachable: a
    stale snapshot would hide it, so it must still fail loud even with allow_stale."""
    config = _settings(tmp_path)
    _landed_snapshot(config.clubelo_seed_dir, "20260929T202827Z")
    monkeypatch.setattr(
        ClubEloSource, "fetch", lambda self, *, since=None: _payload("<html>no chart</html>")
    )

    with pytest.raises(ValueError):
        ingest(config, allow_stale=True)


def test_fallback_prefers_a_newer_cached_snapshot_over_the_seed(tmp_path: Path):
    config = _settings(tmp_path)
    _landed_snapshot(config.clubelo_seed_dir, "20260901T000000Z")
    cached = _landed_snapshot(config.raw_dir / "clubelo", "20260920T000000Z")

    assert use_fallback_snapshot(config) == cached
    assert not (config.raw_dir / "clubelo" / "20260901T000000Z").exists()  # seed not copied


def test_fallback_prefers_a_newer_seed_over_an_older_cache(tmp_path: Path):
    """A seed refreshed by hand must win over an older snapshot in a restored Actions
    cache, or refreshing the seed would silently do nothing."""
    config = _settings(tmp_path)
    _landed_snapshot(config.raw_dir / "clubelo", "20260901T000000Z")
    _landed_snapshot(config.clubelo_seed_dir, "20260920T000000Z")

    restored = use_fallback_snapshot(config)

    assert restored.name == "20260920T000000Z"
    assert (restored / "data.parquet").exists()


def test_fallback_annotates_the_run_when_in_github_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    config = _settings(tmp_path)
    _landed_snapshot(config.clubelo_seed_dir, "20260929T202827Z")
    _failing_fetch(monkeypatch, _gateway_timeout())
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    ingest(config, allow_stale=True)

    assert "::warning title=ClubElo::" in capsys.readouterr().out


# --- staleness guard ---------------------------------------------------------------------


def _name_aged(days: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y%m%dT%H%M%SZ")


def test_snapshot_age_is_read_from_the_directory_name(tmp_path: Path):
    snapshot = _landed_snapshot(tmp_path, "20260901T120000Z")
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    assert snapshot_age_days(snapshot, now=now) == pytest.approx(10.0)


def test_fresh_fallback_snapshot_warns_once_without_the_refresh_nudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    config = _settings(tmp_path)
    _landed_snapshot(config.clubelo_seed_dir, _name_aged(5))
    _failing_fetch(monkeypatch, _gateway_timeout())
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    ingest(config, allow_stale=True)

    out = capsys.readouterr().out
    assert "live_fetch_failed_using_stale_snapshot" in out
    assert "refresh_the_seed" not in out


def test_snapshot_past_the_warn_threshold_adds_the_refresh_nudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    config = _settings(tmp_path)
    _landed_snapshot(config.clubelo_seed_dir, _name_aged(45))  # warn 30, fail 90
    _failing_fetch(monkeypatch, _gateway_timeout())
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    ingest(config, allow_stale=True)  # still succeeds

    out = capsys.readouterr().out
    assert "live_fetch_failed_using_stale_snapshot" in out
    assert "snapshot_getting_old_refresh_the_seed" in out
    assert "just refresh-clubelo-seed" in out


def test_snapshot_past_the_fail_threshold_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _settings(tmp_path)
    _landed_snapshot(config.clubelo_seed_dir, _name_aged(120))
    _failing_fetch(monkeypatch, _gateway_timeout())

    with pytest.raises(StaleClubEloError, match="refresh-clubelo-seed"):
        ingest(config, allow_stale=True)


def test_thresholds_come_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from plforecast.config import Settings

    config = Settings(
        data_dir=tmp_path / "data",
        clubelo_seed_dir=tmp_path / "seed",
        clubelo_stale_fail_days=10,
    )
    _landed_snapshot(config.clubelo_seed_dir, _name_aged(20))
    _failing_fetch(monkeypatch, _gateway_timeout())

    with pytest.raises(StaleClubEloError):
        ingest(config, allow_stale=True)


# --- seed refresh ------------------------------------------------------------------------


def _live_fetch_returns_the_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = RawPayload(
        source="clubelo",
        fetched_at=datetime.now(UTC),
        parts=[
            RawPart(
                url="https://clubelo.com/Arsenal",
                status_code=200,
                content=FIXTURE.read_bytes(),
                fetched_at=datetime.now(UTC),
                label="arsenal",
            )
        ],
    )
    monkeypatch.setattr(ClubEloSource, "fetch", lambda self, *, since=None: payload)


def test_refresh_seed_replaces_the_old_seed_with_a_fresh_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _settings(tmp_path)
    old = _landed_snapshot(config.clubelo_seed_dir, "20260101T000000Z")
    _live_fetch_returns_the_fixture(monkeypatch)

    seeded = refresh_seed(config)

    assert not old.exists()
    assert [p.name for p in config.clubelo_seed_dir.iterdir()] == [seeded.name]
    assert (seeded / "data.parquet").exists()
    assert (seeded / "_meta.json").exists()


def test_refresh_seed_creates_the_seed_dir_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _settings(tmp_path)
    _live_fetch_returns_the_fixture(monkeypatch)

    assert (refresh_seed(config) / "data.parquet").exists()


def test_a_failed_refresh_leaves_the_existing_seed_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _settings(tmp_path)
    old = _landed_snapshot(config.clubelo_seed_dir, "20260101T000000Z")
    _failing_fetch(monkeypatch, _gateway_timeout())

    with pytest.raises(httpx.HTTPStatusError):  # no fallback: refreshing must not go stale
        refresh_seed(config)

    assert old.exists()


# --- end to end: the committed seed -> a failed live fetch -> curate ----------------------


def test_the_real_seed_flows_through_a_failed_fetch_into_stg_clubelo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """What the scheduled run actually does on a cold runner: no cache, clubelo.com
    504ing. The committed seed must land in raw/, become the raw_clubelo_ratings view,
    and curate into a non-empty stg_clubelo the forecast's ClubElo prior can read."""
    from plforecast.config import Settings
    from plforecast.storage.curate import curate_clubelo
    from plforecast.storage.db import connect

    # Real clubelo_seed_dir; the age limit is lifted so this test does not start failing
    # the day the committed seed is 90 days old (that behaviour has its own tests above).
    config = Settings(data_dir=tmp_path / "data", clubelo_stale_fail_days=10_000)
    _failing_fetch(monkeypatch, _gateway_timeout())

    ingest(config, allow_stale=True)
    conn = connect(config)
    curate_clubelo(conn)

    rows, clubs = conn.execute(
        "SELECT count(*), count(DISTINCT club_id) FROM stg_clubelo"
    ).fetchone()
    assert rows > 1000
    assert clubs >= 30
    conn.close()
