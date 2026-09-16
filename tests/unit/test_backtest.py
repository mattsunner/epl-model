from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from plforecast.evaluate.backtest import run_backtest, walk_forward_splits
from plforecast.models.base import UnknownClubError


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


def _matches_with_ids(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=[
            "match_id",
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


def _round_robin_season_with_ids(season: str, start: date, clubs: list[str]) -> list[tuple]:
    """A full round robin among `clubs`, one match per calendar day, in a fixed
    generation order -- realistic gameweek clustering needs date order plus a
    tiebreak, and a fixed match_id order is that tiebreak here."""
    rows = []
    pairs = [(h, a) for h in clubs for a in clubs if h != a]
    for i, (home, away) in enumerate(pairs):
        d = start + timedelta(days=i)
        rows.append((f"{season}-{i}", season, d, home, away, 1, 0, "H"))
    return rows


def test_gameweek_cadence_groups_n_clubs_over_two_matches_per_round():
    # 4 clubs -> 2 matches per round; the round-robin above lands one match per day,
    # so consecutive pairs of days must be clustered into the same round.
    matches = _matches_with_ids(
        _round_robin_season_with_ids("2020/21", date(2020, 8, 1), ["a", "b", "c", "d"])
    )

    splits = list(walk_forward_splits(matches, min_train_matches=0, cadence="gameweek"))

    assert all(split.test.height == 2 for split in splits)
    for split in splits:
        assert split.as_of_date == split.test["date"].min()


def test_gameweek_cadence_never_trains_on_the_rounds_own_matches():
    matches = _matches_with_ids(
        _round_robin_season_with_ids("2020/21", date(2020, 8, 1), ["a", "b", "c", "d"])
    )

    for split in walk_forward_splits(matches, min_train_matches=0, cadence="gameweek"):
        same_season_train = split.train.filter(pl.col("season") == split.season)
        assert (same_season_train["date"] < split.as_of_date).all()
        # The round itself may span more than one date; every one of its matches must
        # still be excluded from training, not just the ones dated as_of_date.
        assert split.train.join(split.test, on="match_id", how="inner").height == 0


def test_gameweek_cadence_scales_matches_per_round_with_club_count_not_a_hardcoded_ten():
    six_clubs = _matches_with_ids(
        _round_robin_season_with_ids("2020/21", date(2020, 8, 1), ["a", "b", "c", "d", "e", "f"])
    )
    splits = list(walk_forward_splits(six_clubs, min_train_matches=0, cadence="gameweek"))
    assert all(split.test.height == 3 for split in splits)  # 6 clubs -> 3 matches/round


def test_gameweek_cadence_covers_every_match_exactly_once_across_all_splits():
    matches = _matches_with_ids(
        _round_robin_season_with_ids("2020/21", date(2020, 8, 1), ["a", "b", "c", "d"])
    )
    splits = list(walk_forward_splits(matches, min_train_matches=0, cadence="gameweek"))
    tested_ids = pl.concat([s.test for s in splits])["match_id"].to_list()
    assert sorted(tested_ids) == sorted(matches["match_id"].to_list())


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

    result = run_backtest(matches, _StubModel, min_train_matches=2)
    results = result.scores

    assert results.height > 0
    assert {"p_home", "p_draw", "p_away", "rps", "log_loss", "brier"} <= set(results.columns)
    assert (results["p_home"] == 0.5).all()
    assert (results["rps"] >= 0).all()
    # Exact accounting: every input match is scored, warm-up excluded, or unrateable.
    assert result.n_scored + result.warmup_excluded + result.unrateable.height == matches.height
    assert result.warmup_excluded == 2  # round 1 (2 matches) precedes the 2-match warm-up
    assert result.unrateable.height == 0


class _RejectsUnknownClubModel:
    """Raises exactly like PoissonModel/DixonColesModel do for a club absent from
    training data -- every round-1 fixture of the backtest window's first season hits
    this in practice, since no club has any prior-window history at that point."""

    def fit(self, matches: pl.DataFrame) -> "_RejectsUnknownClubModel":
        self._known = set(matches["home_club_id"]) | set(matches["away_club_id"])
        return self

    def scoreline_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        if home not in self._known or away not in self._known:
            raise UnknownClubError("Both teams must have been in the training data.")
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

    result = run_backtest(matches, _RejectsUnknownClubModel, min_train_matches=0)

    # Round 1 (2 matches) is unrateable and must be recorded, not raise; rounds 2-5
    # (2 matches each) are trainable from round 1 onward.
    assert result.n_scored == 8
    assert result.unrateable.height == 2
    assert set(result.unrateable.columns) == {"season", "date", "home_club_id", "away_club_id"}
    assert result.warmup_excluded == 0


class _BrokenModel(_StubModel):
    def scoreline_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        raise ValueError("a genuine bug, not an unknown club")


def test_run_backtest_gameweek_cadence_covers_every_match_exactly_once():
    matches = _matches_with_ids(
        _round_robin_season_with_ids("2020/21", date(2020, 8, 1), ["a", "b", "c", "d"])
    )

    result = run_backtest(matches, _StubModel, min_train_matches=0, cadence="gameweek")

    assert result.n_scored + result.warmup_excluded + result.unrateable.height == matches.height
    assert result.n_scored == matches.height  # min_train_matches=0: nothing warms up


def test_run_backtest_does_not_swallow_other_value_errors():
    matches = _matches(_season_rows("2020/21", date(2020, 8, 1), 3))
    with pytest.raises(ValueError, match="genuine bug"):
        run_backtest(matches, _BrokenModel, min_train_matches=1)
