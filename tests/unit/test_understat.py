import io
from datetime import datetime

import pandas as pd
import pandera.errors
import polars as pl
import pytest

from plforecast.ingest.base import RawPart, RawPayload
from plforecast.ingest.understat import UnderstatSource, ingest, season_batches


def _payload(rows: list[dict]) -> RawPayload:
    """Builds a RawPayload the same shape UnderstatSource.fetch() produces: parquet
    bytes of soccerdata's raw (reset_index()'d) team-match-stats frame."""
    raw_pd = pd.DataFrame(rows)
    buffer = io.BytesIO()
    raw_pd.to_parquet(buffer)
    part = RawPart(
        url="understat:ENG-Premier League:2015/16",
        status_code=200,
        content=buffer.getvalue(),
        fetched_at=datetime.now(),
        label="2015/16",
    )
    return RawPayload(source="understat", fetched_at=datetime.now(), parts=[part])


def _row(**overrides) -> dict:
    row = {
        "season_id": 2015,
        "date": datetime(2015, 8, 8, 19, 0),
        "home_team": "Bournemouth",
        "away_team": "Aston Villa",
        "home_goals": 0,
        "away_goals": 1,
        "home_xg": 1.23,
        "away_xg": 0.87,
        "home_np_xg": 1.10,
        "away_np_xg": 0.87,
        "home_ppda": 8.76,
        "away_ppda": 11.58,
    }
    row.update(overrides)
    return row


def test_parse_maps_season_id_to_canonical_label():
    df = UnderstatSource().parse(_payload([_row(season_id=2015)]))
    assert df["season"].to_list() == ["2015/16"]


def test_parse_selects_and_types_xg_columns():
    df = UnderstatSource().parse(_payload([_row()]))

    row = df.row(0, named=True)
    assert set(df.columns) == {
        "season",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_np_xg",
        "away_np_xg",
        "home_ppda",
        "away_ppda",
    }
    assert row["home_team"] == "Bournemouth"
    assert row["home_goals"] == 0
    assert row["date"] == pl.Series(["2015-08-08"]).str.to_date().item()
    assert df["home_xg"].dtype == pl.Float64


def test_parse_rejects_negative_xg(tmp_path):
    with pytest.raises(pandera.errors.SchemaError):
        UnderstatSource().parse(_payload([_row(home_xg=-1.0)]))


def test_parse_handles_multiple_matches_and_seasons():
    df = UnderstatSource().parse(
        _payload(
            [
                _row(season_id=2015, home_team="Chelsea", away_team="Swansea"),
                _row(season_id=2016, home_team="Arsenal", away_team="Leicester"),
            ]
        )
    )

    assert df["season"].to_list() == ["2015/16", "2016/17"]


def test_season_batches_keep_completed_cached_and_current_live():
    assert season_batches([2015, 2016, 2026], 2026) == [([2015, 2016], False), ([2026], True)]
    assert season_batches([2026], 2026) == [([2026], True)]
    assert season_batches([2015], 2026) == [([2015], False)]


# ---- ingest(): current_season_only (story: weekly scheduled pipeline) ----


def test_ingest_full_backfill_by_default_calls_fetch_with_since_none(monkeypatch, tmp_path):
    captured = {}

    def fake_fetch(self, *, since=None):
        captured["since"] = since
        return _payload([_row()])

    monkeypatch.setattr(UnderstatSource, "fetch", fake_fetch)
    monkeypatch.setattr(
        "plforecast.ingest.understat.write_snapshot", lambda *a, **k: tmp_path / "snapshot"
    )

    ingest()

    assert captured["since"] is None


def test_ingest_current_season_only_calls_fetch_with_a_since_date(monkeypatch, tmp_path):
    captured = {}

    def fake_fetch(self, *, since=None):
        captured["since"] = since
        return _payload([_row()])

    monkeypatch.setattr(UnderstatSource, "fetch", fake_fetch)
    monkeypatch.setattr(
        "plforecast.ingest.understat.write_snapshot", lambda *a, **k: tmp_path / "snapshot"
    )

    ingest(current_season_only=True)

    assert captured["since"] is not None
