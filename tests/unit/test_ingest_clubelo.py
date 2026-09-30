from datetime import datetime
from pathlib import Path

import httpx
import pytest

from plforecast.ingest.base import RawPart, RawPayload
from plforecast.ingest.clubelo import (
    ClubEloSource,
    _extract_vega_json,
    ingest,
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
