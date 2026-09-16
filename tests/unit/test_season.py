from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.stats import poisson

from plforecast.evaluate.season import (
    evaluate_season_level,
    position_rps,
    remaining_pairs,
    score_season,
    split_at_cutoff,
    summarise_season_level,
)
from plforecast.simulate.competition import CompetitionConfig
from plforecast.simulate.tiebreak import PremierLeagueTiebreaks

CLUBS = ["a", "b", "c", "d"]
SMALL = CompetitionConfig(n_clubs=4, relegation_spots=1, european_spots=1)


class _FixedPoissonModel:
    def fit(self, matches):
        return self

    def scoreline_matrix(self, home, away, max_goals=10):
        goals = np.arange(max_goals + 1)
        m = np.outer(poisson.pmf(goals, 1.4), poisson.pmf(goals, 1.1))
        return m / m.sum()

    @property
    def config_hash(self):
        return "fixed"


def _season(season: str, start: date, rng) -> pl.DataFrame:
    rows = []
    day = 0
    for h in CLUBS:
        for a in CLUBS:
            if h == a:
                continue
            rows.append(
                {
                    "match_id": f"{season}-{h}-{a}",
                    "season": season,
                    "date": start + timedelta(days=day // 2),  # two matches per date
                    "home_club_id": h,
                    "away_club_id": a,
                    "home_goals": int(rng.integers(0, 4)),
                    "away_goals": int(rng.integers(0, 4)),
                }
            )
            day += 1
    return pl.DataFrame(rows)


def test_split_at_cutoff_never_splits_a_match_date():
    season = _season("2020/21", date(2020, 8, 1), np.random.default_rng(0))
    played = split_at_cutoff(season, 3)  # third match shares a date with the fourth
    assert played.height == 4


def test_remaining_pairs_are_the_unplayed_ordered_pairs():
    played = pl.DataFrame({"home_club_id": ["a", "b"], "away_club_id": ["b", "a"]})
    remaining = remaining_pairs(CLUBS, played)
    assert remaining.height == 12 - 2
    assert ("a", "b") not in set(
        zip(remaining["home_club_id"], remaining["away_club_id"], strict=True)
    )


def test_position_rps_rewards_near_misses():
    certain_first = np.array([1.0, 0, 0, 0])
    assert position_rps(certain_first, 0) == 0.0
    assert position_rps(certain_first, 1) < position_rps(certain_first, 3)


def test_score_season_scores_realised_events():
    pmf = np.array(
        [
            [0.7, 0.2, 0.1, 0.0],
            [0.2, 0.5, 0.2, 0.1],
            [0.1, 0.2, 0.5, 0.2],
            [0.0, 0.1, 0.2, 0.7],
        ]
    )
    scores = score_season(pmf, CLUBS, ["a", "b", "c", "d"], competition=SMALL)
    assert scores["title_log_loss"] == pytest.approx(-np.log(0.7))
    assert 0 < scores["position_rps"] < 0.2
    assert scores["relegation_log_loss"] < 0.3


def test_evaluate_season_level_scores_every_model_season_and_cutoff():
    rng = np.random.default_rng(1)
    matches = pl.concat(
        [_season("2020/21", date(2020, 8, 1), rng), _season("2021/22", date(2021, 8, 1), rng)]
    )

    scores = evaluate_season_level(
        matches,
        {"fixed": _FixedPoissonModel},
        cutoffs=(4, 8),
        n_simulations=200,
        competition=SMALL,
        tiebreak_rules=PremierLeagueTiebreaks(SMALL),
        max_goals=6,
    )

    assert scores.height == 4  # 2 seasons x 2 cutoffs
    assert set(scores["model"]) == {"fixed"}
    assert (scores["position_rps"] >= 0).all()
    summary = summarise_season_level(scores)
    assert [(row["cutoff"], row["n_seasons"]) for row in summary] == [(4, 2), (8, 2)]
