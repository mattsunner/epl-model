from datetime import datetime
from pathlib import Path

import pandera.errors
import polars as pl
import pytest

from plforecast.ingest.base import RawPart, RawPayload
from plforecast.ingest.footballdata import (
    CLOSING_ODDS_SETS,
    FootballDataSource,
    closing_odds_columns,
    season_code,
    season_label,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "footballdata" / "E0_1516_sample.csv"
TWO_DIGIT_YEAR_FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "footballdata" / "E0_1617_two_digit_year_sample.csv"
)


def _payload(content: bytes, *, label: str = "2015/16", url: str | None = None) -> RawPayload:
    part = RawPart(
        url=url or "https://www.football-data.co.uk/mmz4281/1516/E0.csv",
        status_code=200,
        content=content,
        fetched_at=datetime.now(),
        label=label,
    )
    return RawPayload(source="football-data", fetched_at=datetime.now(), parts=[part])


def test_season_code_and_label_roundtrip():
    assert season_code(2015) == "1516"
    assert season_code(2026) == "2627"
    assert season_label(2015) == "2015/16"


def test_parse_selects_and_types_core_columns():
    df = FootballDataSource().parse(_payload(FIXTURE.read_bytes()))

    assert df.height == 5
    assert {"season", "date", "home_team", "away_team", "fthg", "ftag", "ftr"} < set(df.columns)
    for prefix in CLOSING_ODDS_SETS:
        assert set(closing_odds_columns(prefix)) <= set(df.columns)
    assert df["season"].to_list() == ["2015/16"] * 5
    assert df["date"][0] == pl.Series(["2015-08-08"]).str.to_date().item()
    assert df["home_team"][0] == "Bournemouth"
    assert df["ftr"].is_in(["H", "D", "A"]).all()
    assert df["psch"].dtype == pl.Float64
    # 2015/16 carries Pinnacle closing only; the later sets land as nulls.
    assert df["psch"].null_count() == 0
    assert df["avgch"].null_count() == 5


def test_parse_rejects_bad_ftr(tmp_path):
    corrupted = FIXTURE.read_text().replace(",A,0,0,D,", ",Z,0,0,D,")
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text(corrupted)

    with pytest.raises(pandera.errors.SchemaError):
        FootballDataSource().parse(_payload(bad_file.read_bytes()))


def test_parse_handles_two_digit_year_dates():
    """football-data.co.uk is inconsistent about year width across seasons -- 2016/17
    uses DD/MM/YY while most seasons use DD/MM/YYYY. A naive fixed format silently
    parses "16" as the year 16 CE instead of 2016; this must not regress."""
    payload = _payload(
        TWO_DIGIT_YEAR_FIXTURE.read_bytes(),
        label="2016/17",
        url="https://www.football-data.co.uk/mmz4281/1617/E0.csv",
    )
    df = FootballDataSource().parse(payload)

    assert df["date"].to_list() == [
        pl.Series(["2016-08-13"]).str.to_date().item(),
        pl.Series(["2016-08-15"]).str.to_date().item(),
    ]


def test_parse_lands_every_closing_set_the_file_carries_and_nulls_the_rest():
    """The 2026/27 file dropped Pinnacle closing entirely but carries Bet365, Betfair
    Exchange, Max and Avg closing prices. Absent sets land as nulls, never as a crash."""
    csv = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365CH,B365CD,B365CA,"
        "MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,BFECH,BFECD,BFECA\n"
        "E0,15/08/2026,20:00,Liverpool,Bournemouth,4,2,H,1.4,4.8,7.5,1.45,5.0,8.0,"
        "1.41,4.7,7.4,1.46,5.1,8.2\n"
    )
    payload = _payload(csv.encode(), label="2026/27")

    df = FootballDataSource().parse(payload)

    assert df.height == 1
    assert df["psch"].to_list() == [None]
    assert df["b365ch"].to_list() == [1.4]
    assert df["avgca"].to_list() == [7.4]
    assert df["bfech"].to_list() == [1.46]
