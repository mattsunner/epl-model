import polars as pl
import pytest

from plforecast.evaluate.market import market_probabilities


def _matches(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_market_probabilities_sums_to_one_and_removes_the_margin():
    matches = _matches(
        [{"pinnacle_home_odds": 2.0, "pinnacle_draw_odds": 3.5, "pinnacle_away_odds": 4.0}]
    )

    result = market_probabilities(matches, method="shin")

    total = result["p_home"][0] + result["p_draw"][0] + result["p_away"][0]
    assert total == pytest.approx(1.0, abs=1e-6)
    # Naive 1/odds would sum to > 1 (the bookmaker's margin); de-vigged must not.
    naive_total = 1 / 2.0 + 1 / 3.5 + 1 / 4.0
    assert naive_total > 1.0


def test_market_probabilities_drops_rows_with_missing_odds():
    matches = _matches(
        [
            {"pinnacle_home_odds": 2.0, "pinnacle_draw_odds": 3.5, "pinnacle_away_odds": 4.0},
            {"pinnacle_home_odds": None, "pinnacle_draw_odds": 3.5, "pinnacle_away_odds": 4.0},
        ]
    )

    result = market_probabilities(matches)

    assert result.height == 1


def test_market_probabilities_empty_when_all_odds_missing():
    matches = _matches(
        [{"pinnacle_home_odds": None, "pinnacle_draw_odds": None, "pinnacle_away_odds": None}]
    )

    result = market_probabilities(matches)

    assert result.height == 0
    assert {"p_home", "p_draw", "p_away"} <= set(result.columns)


def test_market_probabilities_shin_and_multiplicative_both_sum_to_one_but_differ():
    matches = _matches(
        [{"pinnacle_home_odds": 1.5, "pinnacle_draw_odds": 4.0, "pinnacle_away_odds": 7.0}]
    )

    shin = market_probabilities(matches, method="shin")
    multiplicative = market_probabilities(matches, method="multiplicative")

    assert shin["p_home"][0] + shin["p_draw"][0] + shin["p_away"][0] == pytest.approx(1.0)
    assert multiplicative["p_home"][0] + multiplicative["p_draw"][0] + multiplicative["p_away"][
        0
    ] == pytest.approx(1.0)
    # Different de-vig methods should generally disagree on the exact split, especially
    # with a big favourite like 1.5 -- design.md section 8.3's whole reason for
    # preferring Shin over multiplicative.
    assert shin["p_home"][0] != pytest.approx(multiplicative["p_home"][0])
