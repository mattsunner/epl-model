from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from plforecast.entities.clubs import load_club_dimension
from plforecast.entities.competitions import PREMIER_LEAGUE, build_club_season_membership
from plforecast.ingest.base import RawPart, RawPayload
from plforecast.ingest.footballdata import FootballDataSource

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "entities"
MINI_ALIASES = FIXTURES_DIR / "mini_club_aliases.yaml"
FOOTBALLDATA_SAMPLE = (
    Path(__file__).parent.parent / "fixtures" / "footballdata" / "E0_1516_sample.csv"
)


def _matches(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["season", "home_team", "away_team"], orient="row")


def test_membership_reflects_promotion_and_relegation():
    dimension = load_club_dimension(MINI_ALIASES)
    matches = _matches(
        [
            ("2015/16", "Arsenal", "Chelsea"),
            ("2015/16", "Leeds", "Watford"),
            ("2016/17", "Arsenal", "Chelsea"),
        ]
    )

    membership = build_club_season_membership(matches, dimension)

    season_1516 = set(membership.filter(pl.col("season") == "2015/16")["club_id"].to_list())
    season_1617 = set(membership.filter(pl.col("season") == "2016/17")["club_id"].to_list())
    assert season_1516 == {"arsenal", "chelsea", "leeds", "watford"}
    assert season_1617 == {"arsenal", "chelsea"}, "Leeds/Watford relegated, must drop out"


def test_membership_dedupes_repeat_fixtures_within_a_season():
    dimension = load_club_dimension(MINI_ALIASES)
    matches = _matches(
        [
            ("2015/16", "Arsenal", "Chelsea"),
            ("2015/16", "Chelsea", "Arsenal"),  # reverse fixture, same season
        ]
    )

    membership = build_club_season_membership(matches, dimension)

    assert membership.height == 2  # one row each for Arsenal and Chelsea, not four


def test_membership_columns_and_defaults():
    dimension = load_club_dimension(MINI_ALIASES)
    matches = _matches([("2015/16", "Arsenal", "Chelsea")])

    membership = build_club_season_membership(matches, dimension)

    assert set(membership.columns) == {"club_id", "season", "competition", "division_tier"}
    assert (membership["competition"] == PREMIER_LEAGUE).all()
    assert (membership["division_tier"] == 1).all()


def test_membership_against_real_ingested_sample():
    """Same football-data.co.uk sample used to validate the ingest adapter itself, run
    through the real production club dimension end to end."""
    dimension = load_club_dimension()
    part = RawPart(
        url="https://www.football-data.co.uk/mmz4281/1516/E0.csv",
        status_code=200,
        content=FOOTBALLDATA_SAMPLE.read_bytes(),
        fetched_at=datetime.now(),
        label="2015/16",
    )
    payload = RawPayload(source="football-data", fetched_at=datetime.now(), parts=[part])
    matches = FootballDataSource().parse(payload)

    membership = build_club_season_membership(matches, dimension)

    assert set(membership["club_id"].to_list()) == {
        "bournemouth",
        "aston-villa",
        "chelsea",
        "swansea",
        "everton",
        "watford",
        "leicester",
        "sunderland",
        "man-united",
        "tottenham",
    }


def test_unresolvable_club_name_raises():
    dimension = load_club_dimension(MINI_ALIASES)
    matches = _matches([("2015/16", "Arsenal", "Not A Real Club")])

    with pytest.raises(ValueError, match="Not A Real Club"):
        build_club_season_membership(matches, dimension)
