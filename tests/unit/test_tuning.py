from datetime import date, timedelta

import numpy as np
import polars as pl

from plforecast.evaluate.tuning import tune_parameter


def _matches() -> pl.DataFrame:
    rows = []
    for season, start in (("2020/21", date(2020, 8, 1)), ("2021/22", date(2021, 8, 1))):
        for r in range(6):
            d = start + timedelta(days=7 * r)
            rows.append((season, d, "a", "b", 2, 1, "H"))
            rows.append((season, d, "c", "d", 0, 0, "D"))
    return pl.DataFrame(
        rows,
        schema=[
            "season",
            "date",
            "home_club_id",
            "away_club_id",
            "home_goals",
            "away_goals",
            "result",
        ],
        orient="row",
    )


class _ParamModel:
    """Predicts P(home) = value; results are all H or D, so the best value is clear."""

    def __init__(self, value: float) -> None:
        self.value = value

    def fit(self, matches):
        return self

    def scoreline_matrix(self, home, away, max_goals=10):
        m = np.zeros((max_goals + 1, max_goals + 1))
        m[1, 0] = self.value
        m[0, 0] = 1 - self.value
        return m

    @property
    def config_hash(self):
        return f"param-{self.value}"


def test_tune_selects_on_early_seasons_and_reports_on_later_ones():
    result = tune_parameter(
        _matches(),
        lambda v: lambda: _ParamModel(v),
        [0.1, 0.5, 0.9],
        parameter="p_home",
        select_through_season="2020/21",
        min_train_matches=2,
    )

    assert result["parameter"] == "p_home"
    assert result["selection_seasons"] == ["2020/21"]
    assert result["report_seasons"] == ["2021/22"]
    assert result["selected"] == 0.5  # half the results are H, half D
    grid = {row["value"]: row for row in result["grid"]}
    assert grid[0.5]["select_rps"] < grid[0.9]["select_rps"]
    assert all(row["report_n"] == 12 for row in result["grid"])
    assert all(row["select_n"] == 10 for row in result["grid"])  # 2-match warm-up excluded


def test_parsimony_rule_prefers_the_null_value_on_a_flat_grid():
    result = tune_parameter(
        _matches(),
        lambda v: lambda: _ParamModel(v),
        [0.5, 0.52],  # 0.52 is fractionally better on a near-flat surface
        parameter="p_home",
        select_through_season="2020/21",
        min_train_matches=2,
        null_value=0.5,
        tolerance=0.01,
    )
    assert result["best_by_selection_rps"] in (0.5, 0.52)
    assert result["selected"] == 0.5
    assert result["parsimony_applied"] == (result["best_by_selection_rps"] != 0.5)
