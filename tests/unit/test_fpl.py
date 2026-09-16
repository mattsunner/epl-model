from datetime import datetime
from pathlib import Path

import pandera.errors
import polars as pl
import pytest

from plforecast.ingest.base import RawPart, RawPayload
from plforecast.ingest.fpl import FPLFixturesSource, FPLPlayersSource, FPLTeamsSource

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "fpl"
BOOTSTRAP = FIXTURES_DIR / "bootstrap_sample.json"
FIXTURES = FIXTURES_DIR / "fixtures_sample.json"


def _payload(source: str, label: str, content: bytes) -> RawPayload:
    part = RawPart(
        url=f"https://fantasy.premierleague.com/api/{label}/",
        status_code=200,
        content=content,
        fetched_at=datetime.now(),
        label=label,
    )
    return RawPayload(source=source, fetched_at=datetime.now(), parts=[part])


def test_teams_parse():
    payload = _payload("fpl-teams", "bootstrap", BOOTSTRAP.read_bytes())
    df = FPLTeamsSource().parse(payload)

    assert df.height == 2
    assert set(df.columns) == {"fpl_team_id", "name", "short_name"}
    assert df.filter(pl.col("fpl_team_id") == 1)["name"].item() == "Arsenal"


def test_players_parse_preserves_availability_fields():
    payload = _payload("fpl-players", "bootstrap", BOOTSTRAP.read_bytes())
    df = FPLPlayersSource().parse(payload)

    assert df.height == 2
    saliba = df.filter(pl.col("fpl_player_id") == 6)
    assert saliba["status"].item() == "i"
    assert saliba["chance_of_playing_this_round"].item() == 0
    assert "injury" in saliba["news"].item().lower()

    raya = df.filter(pl.col("fpl_player_id") == 1)
    assert raya["status"].item() == "a"
    assert raya["chance_of_playing_this_round"].item() is None


def test_players_parse_rejects_unknown_status(tmp_path):
    corrupted = BOOTSTRAP.read_text().replace('"status": "a"', '"status": "x"')
    bad_file = tmp_path / "bad_bootstrap.json"
    bad_file.write_text(corrupted)

    payload = _payload("fpl-players", "bootstrap", bad_file.read_bytes())
    with pytest.raises(pandera.errors.SchemaError):
        FPLPlayersSource().parse(payload)


def test_fixtures_parse_handles_unplayed_fixtures():
    payload = _payload("fpl-fixtures", "fixtures", FIXTURES.read_bytes())
    df = FPLFixturesSource().parse(payload)

    assert df.height == 2
    unplayed = df.filter(pl.col("fpl_fixture_id") == 41)
    assert unplayed["finished"].item() is False
    assert unplayed["home_score"].item() is None
    assert unplayed["away_score"].item() is None
    assert (
        unplayed["kickoff_time"].item()
        == pl.Series(["2026-09-18T19:00:00Z"])
        .str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC")
        .item()
    )

    played = df.filter(pl.col("fpl_fixture_id") == 1)
    assert played["home_score"].item() == 3
    assert played["away_score"].item() == 0
