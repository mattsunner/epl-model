from datetime import datetime
from pathlib import Path

import pytest

from plforecast.ingest.base import RawPart, RawPayload
from plforecast.ingest.clubelo import ClubEloSource, _extract_vega_json

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
