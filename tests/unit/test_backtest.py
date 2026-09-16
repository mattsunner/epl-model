from datetime import date, timedelta

import numpy as np
import polars as pl

from plforecast.evaluate.backtest import run_backtest, walk_forward_splits


def _matches(rows: list[tuple]) -> pl.DataFrame:
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


def _season_rows(season: str, start: date, n_rounds: int) -> list[tuple]:
    rows = []
    for r in range(n_rounds):
        d = start + timedelta(days=7 * r)
        rows.append((season, d, "a", "b", 2, 1, "H"))
        rows.append((season, d, "c", "d", 0, 0, "D"))
    return rows


def test_walk_forward_never_trains_on_future_matches():
    matches = _matches(
        _season_rows("2020/21", date(2020, 8, 1), 5) + _season_rows("2021/22", date(2021, 8, 1), 5)
    )

    for split in walk_forward_splits(matches, min_train_matches=1):
        # Every train row from the split's own season must be strictly before
        # as_of_date; rows from a prior season are always fully in the past anyway.
        same_season_train = split.train.filter(pl.col("season") == split.season)
        assert (same_season_train["date"] < split.as_of_date).all()
        assert (split.test["date"] == split.as_of_date).all()


def test_walk_forward_accumulates_prior_seasons_into_training_data():
    matches = _matches(
        _season_rows("2020/21", date(2020, 8, 1), 3) + _season_rows("2021/22", date(2021, 8, 1), 3)
    )

    splits = list(walk_forward_splits(matches, min_train_matches=1))
    first_split_of_second_season = next(s for s in splits if s.season == "2021/22")

    assert first_split_of_second_season.train.filter(pl.col("season") == "2020/21").height == 6


def test_min_train_matches_skips_earliest_rounds():
    matches = _matches(_season_rows("2020/21", date(2020, 8, 1), 5))

    splits = list(walk_forward_splits(matches, min_train_matches=3))
    assert all(s.train.height >= 3 for s in splits)
    assert len(splits) < 5  # the very first round(s), with too little training data, are skipped


class _StubModel:
    """Predicts a fixed [P(home), P(draw), P(away)] regardless of matchup or training
    data -- enough to exercise run_backtest's plumbing without needing a real fit."""

    def fit(self, matches: pl.DataFrame) -> "_StubModel":
        return self

    def scoreline_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        matrix[1, 0] = 0.5  # home win
        matrix[0, 0] = 0.3  # draw
        matrix[0, 1] = 0.2  # away win
        return matrix

    @property
    def config_hash(self) -> str:
        return "stub"


def test_run_backtest_produces_one_row_per_test_match_with_scores():
    matches = _matches(_season_rows("2020/21", date(2020, 8, 1), 5))

    results = run_backtest(matches, _StubModel, min_train_matches=2)

    assert results.height > 0
    assert {"p_home", "p_draw", "p_away", "rps", "log_loss", "brier"} <= set(results.columns)
    assert (results["p_home"] == 0.5).all()
    assert (results["rps"] >= 0).all()


class _RejectsUnknownClubModel:
    """Raises exactly like PoissonModel/DixonColesModel do for a club absent from
    training data -- every round-1 fixture of the backtest window's first season hits
    this in practice, since no club has any prior-window history at that point."""

    def fit(self, matches: pl.DataFrame) -> "_RejectsUnknownClubModel":
        self._known = set(matches["home_club_id"]) | set(matches["away_club_id"])
        return self

    def scoreline_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        if home not in self._known or away not in self._known:
            raise ValueError("Both teams must have been in the training data.")
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        matrix[0, 0] = 1.0
        return matrix

    @property
    def config_hash(self) -> str:
        return "rejects-unknown"


def test_run_backtest_skips_matches_with_a_club_unseen_in_training():
    # Round 1 of the very first season: nothing has been trained on yet, so every one
    # of its own fixtures involves a club absent from training data at prediction time.
    matches = _matches(_season_rows("2020/21", date(2020, 8, 1), 5))

    results = run_backtest(matches, _RejectsUnknownClubModel, min_train_matches=0)

    # Round 1 (2 matches) is unrateable and must be skipped, not raise; rounds 2-5
    # (2 matches each) are trainable from round 1 onward.
    assert results.height == 8
