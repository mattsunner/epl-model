import math
from datetime import date

import polars as pl
import pytest

from plforecast.features.strength import attach_decay_weights, build_club_strength


def _match(match_id: str, day: date, home: str, away: str, hg: int, ag: int) -> dict:
    return {
        "match_id": match_id,
        "date": day,
        "home_club_id": home,
        "away_club_id": away,
        "home_goals": hg,
        "away_goals": ag,
    }


def test_attach_decay_weights_excludes_future_matches():
    matches = pl.DataFrame(
        [
            _match("m1", date(2026, 1, 1), "arsenal", "chelsea", 2, 1),
            _match("m2", date(2026, 6, 1), "arsenal", "chelsea", 1, 1),  # after as_of
        ]
    )

    result = attach_decay_weights(matches, as_of=date(2026, 3, 1), xi=0.01)

    assert result["match_id"].to_list() == ["m1"]


def test_attach_decay_weights_computes_exponential_weight():
    matches = pl.DataFrame([_match("m1", date(2026, 1, 1), "arsenal", "chelsea", 2, 1)])
    as_of = date(2026, 1, 31)  # 30 days later

    result = attach_decay_weights(matches, as_of=as_of, xi=0.01)

    row = result.row(0, named=True)
    assert row["days_since"] == 30
    assert row["weight"] == pytest.approx(math.exp(-0.01 * 30))


def test_attach_decay_weights_zero_xi_disables_decay():
    matches = pl.DataFrame(
        [
            _match("m1", date(2025, 1, 1), "arsenal", "chelsea", 2, 1),
            _match("m2", date(2026, 1, 1), "arsenal", "chelsea", 0, 0),
        ]
    )

    result = attach_decay_weights(matches, as_of=date(2026, 1, 1), xi=0.0)

    assert result["weight"].to_list() == [1.0, 1.0]


def test_build_club_strength_weighted_average_by_venue():
    # Arsenal at home: 2 matches, most recent one heavily upweighted by decay.
    matches = pl.DataFrame(
        [
            _match("m1", date(2020, 1, 1), "arsenal", "watford", 0, 0),  # far in the past
            _match("m2", date(2026, 1, 1), "arsenal", "leeds", 4, 0),  # recent
            _match("m3", date(2026, 1, 1), "chelsea", "arsenal", 1, 2),  # arsenal away
        ]
    )

    result = build_club_strength(matches, as_of=date(2026, 1, 1), xi=0.05)
    arsenal = result.filter(pl.col("club_id") == "arsenal").row(0, named=True)

    # m1 is 2192 days old at xi=0.05 -> weight ~ 0, so home_attack_rate should sit very
    # close to m2's 4 goals, not the midpoint between 0 and 4.
    assert arsenal["home_attack_rate"] == pytest.approx(4.0, abs=0.01)
    assert arsenal["home_defence_rate"] == pytest.approx(0.0, abs=0.01)
    assert arsenal["away_attack_rate"] == pytest.approx(2.0)
    assert arsenal["away_defence_rate"] == pytest.approx(1.0)


def test_build_club_strength_omits_clubs_with_no_matches_before_as_of():
    matches = pl.DataFrame(
        [_match("m1", date(2026, 6, 1), "arsenal", "chelsea", 2, 1)]  # after as_of
    )

    result = build_club_strength(matches, as_of=date(2026, 1, 1), xi=0.01)

    assert result.height == 0
