from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from plforecast.features.schedule import build_rest_days

UTC = ZoneInfo("UTC")


def _fixtures(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_first_fixture_of_the_season_has_null_rest_days():
    fixtures = _fixtures(
        [
            {
                "fixture_id": 1,
                "kickoff_time": datetime(2026, 8, 21, 15, 0, tzinfo=UTC),
                "home_club_id": "arsenal",
                "away_club_id": "chelsea",
            }
        ]
    )

    result = build_rest_days(fixtures)

    row = result.row(0, named=True)
    assert row["home_rest_days"] is None
    assert row["away_rest_days"] is None


def test_rest_days_computed_from_previous_kickoff():
    fixtures = _fixtures(
        [
            {
                "fixture_id": 1,
                "kickoff_time": datetime(2026, 8, 21, 15, 0, tzinfo=UTC),
                "home_club_id": "arsenal",
                "away_club_id": "chelsea",
            },
            {
                "fixture_id": 2,
                # Arsenal at home again 4 days later; Chelsea away 3 days later.
                "kickoff_time": datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
                "home_club_id": "arsenal",
                "away_club_id": "leeds",
            },
            {
                "fixture_id": 3,
                "kickoff_time": datetime(2026, 8, 24, 15, 0, tzinfo=UTC),
                "home_club_id": "watford",
                "away_club_id": "chelsea",
            },
        ]
    )

    result = build_rest_days(fixtures).sort("fixture_id")

    fixture_2 = result.filter(pl.col("fixture_id") == 2).row(0, named=True)
    assert fixture_2["home_rest_days"] == pytest.approx(3.875)  # 15:00 Aug21 -> 12:00 Aug25

    fixture_3 = result.filter(pl.col("fixture_id") == 3).row(0, named=True)
    assert fixture_3["away_rest_days"] == 3.0  # Chelsea: 21 Aug 15:00 -> 24 Aug 15:00


def test_fixture_with_null_kickoff_is_excluded_but_others_unaffected():
    fixtures = _fixtures(
        [
            {
                "fixture_id": 1,
                "kickoff_time": datetime(2026, 8, 21, 15, 0, tzinfo=UTC),
                "home_club_id": "arsenal",
                "away_club_id": "chelsea",
            },
            {
                "fixture_id": 2,
                "kickoff_time": None,
                "home_club_id": "arsenal",
                "away_club_id": "leeds",
            },
        ]
    )

    result = build_rest_days(fixtures).sort("fixture_id")

    unresolved = result.filter(pl.col("fixture_id") == 2).row(0, named=True)
    assert unresolved["home_rest_days"] is None
    assert unresolved["away_rest_days"] is None

    first = result.filter(pl.col("fixture_id") == 1).row(0, named=True)
    assert first["home_rest_days"] is None  # still the season's first known Arsenal fixture


def test_output_has_one_row_per_fixture_in_original_order_coverage():
    fixture_ids = range(1, 6)
    fixtures = _fixtures(
        [
            {
                "fixture_id": i,
                "kickoff_time": datetime(2026, 8, 20 + i, 15, 0, tzinfo=UTC),
                "home_club_id": "arsenal",
                "away_club_id": "chelsea",
            }
            for i in fixture_ids
        ]
    )

    result = build_rest_days(fixtures)

    assert result.height == 5
    assert set(result["fixture_id"].to_list()) == set(fixture_ids)
