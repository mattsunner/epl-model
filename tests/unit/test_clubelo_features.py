from datetime import date

import polars as pl

from plforecast.features.clubelo import RATE_FIELDS, external_rating_for_club, fit_elo_to_rate

AS_OF = date(2026, 9, 1)


def _match(match_id: str, day: date, home: str, away: str, hg: int, ag: int) -> dict:
    return {
        "match_id": match_id,
        "date": day,
        "home_club_id": home,
        "away_club_id": away,
        "home_goals": hg,
        "away_goals": ag,
    }


def _elo_row(club_id: str, day: date, elo: float) -> dict:
    return {"club_id": club_id, "date": day, "elo": elo}


def _established_matches() -> pl.DataFrame:
    # Three clubs, round-robin home and away, each with real rate variation: "strong"
    # scores a lot and concedes little, "weak" the reverse, "mid" in between -- enough
    # spread for a regression against Elo to find a real, non-degenerate slope.
    rows = [
        _match("m1", date(2026, 8, 1), "strong", "weak", 4, 0),
        _match("m2", date(2026, 8, 2), "weak", "mid", 0, 2),
        _match("m3", date(2026, 8, 3), "mid", "strong", 1, 3),
        _match("m4", date(2026, 8, 8), "weak", "strong", 0, 3),
        _match("m5", date(2026, 8, 9), "mid", "weak", 2, 0),
        _match("m6", date(2026, 8, 10), "strong", "mid", 3, 1),
    ]
    return pl.DataFrame(rows)


def _established_elo() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _elo_row("strong", date(2026, 7, 1), 1900.0),
            _elo_row("mid", date(2026, 7, 1), 1600.0),
            _elo_row("weak", date(2026, 7, 1), 1300.0),
        ]
    )


def test_fit_elo_to_rate_returns_a_slope_and_intercept_for_every_rate_field():
    coefficients = fit_elo_to_rate(_established_matches(), _established_elo(), as_of=AS_OF)

    assert set(coefficients) == set(RATE_FIELDS)
    for slope, intercept in coefficients.values():
        assert isinstance(slope, float)
        assert isinstance(intercept, float)


def test_fit_elo_to_rate_is_empty_when_no_elo_data_exists_by_as_of():
    """The real bug found in 03-04: fitting on a date before ClubElo's own cached
    history begins must return {}, not raise on an empty frame."""
    future_elo = pl.DataFrame([_elo_row("strong", date(2027, 1, 1), 1900.0)])

    coefficients = fit_elo_to_rate(_established_matches(), future_elo, as_of=AS_OF)

    assert coefficients == {}


def test_fit_elo_to_rate_is_empty_with_fewer_than_two_calibration_points():
    one_club_elo = pl.DataFrame([_elo_row("strong", date(2026, 7, 1), 1900.0)])

    coefficients = fit_elo_to_rate(_established_matches(), one_club_elo, as_of=AS_OF)

    assert coefficients == {}


def test_external_rating_for_club_uses_a_promoted_clubs_own_elo():
    coefficients = fit_elo_to_rate(_established_matches(), _established_elo(), as_of=AS_OF)
    promoted_elo = pl.concat(
        [_established_elo(), pl.DataFrame([_elo_row("promoted", date(2026, 7, 1), 1700.0)])]
    )

    rating = external_rating_for_club("promoted", coefficients, promoted_elo, as_of=AS_OF)

    assert rating is not None
    assert set(rating) == {"home_attack", "home_defence", "away_attack", "away_defence"}
    assert all(value > 0 for value in rating.values())


def test_external_rating_for_club_is_none_without_any_elo_rating():
    coefficients = fit_elo_to_rate(_established_matches(), _established_elo(), as_of=AS_OF)

    rating = external_rating_for_club("no-elo-club", coefficients, _established_elo(), as_of=AS_OF)

    assert rating is None


def test_external_rating_for_club_is_none_when_coefficients_are_empty():
    rating = external_rating_for_club("strong", {}, _established_elo(), as_of=AS_OF)

    assert rating is None


def test_external_rating_for_club_reflects_relative_elo_ordering():
    """A club with a higher Elo than 'weak' should get a higher predicted attack rate,
    the whole point of using Elo as a strength signal rather than a fixed anchor."""
    coefficients = fit_elo_to_rate(_established_matches(), _established_elo(), as_of=AS_OF)
    high_elo = pl.concat(
        [_established_elo(), pl.DataFrame([_elo_row("newcomer", date(2026, 7, 1), 2000.0)])]
    )

    rating = external_rating_for_club("newcomer", coefficients, high_elo, as_of=AS_OF)
    weak_rating = external_rating_for_club("weak", coefficients, _established_elo(), as_of=AS_OF)

    assert rating is not None and weak_rating is not None
    assert rating["home_attack"] > weak_rating["home_attack"]
