from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from plforecast.features.priors import (
    SurvivalZoneReference,
    build_prior,
    build_survival_zone_reference,
    needs_prior,
)
from plforecast.simulate.competition import CompetitionConfig


def _matches(rows: list[tuple]) -> pl.DataFrame:
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
        ],
        orient="row",
    )


def test_needs_prior_true_below_threshold():
    matches = _matches([("m1", "2024/25", date(2024, 8, 10), "coventry", "arsenal", 1, 2)])
    assert needs_prior("coventry", matches, min_matches=38) is True


def test_needs_prior_false_above_threshold():
    rows = [
        (f"m{i}", "2024/25", date(2024, 8, 10) + timedelta(days=i), "ipswich", "arsenal", 1, 1)
        for i in range(38)
    ]
    matches = _matches(rows)
    assert needs_prior("ipswich", matches, min_matches=38) is False


def test_needs_prior_false_for_a_club_never_mentioned():
    matches = _matches([("m1", "2024/25", date(2024, 8, 10), "arsenal", "chelsea", 1, 1)])
    assert needs_prior("some-other-club", matches, min_matches=1) is True


def _small_round_robin_season(season: str, start: date, clubs: list[str], rng) -> list[tuple]:
    rows = []
    day = 0
    for home in clubs:
        for away in clubs:
            if home == away:
                continue
            hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            rows.append(
                (f"{season}-{home}-{away}", season, start + timedelta(days=day), home, away, hg, ag)
            )
            day += 1
    return rows


def test_survival_zone_reference_reflects_only_the_target_positions():
    rng = np.random.default_rng(0)
    clubs = [f"club-{i}" for i in range(8)]
    matches = _matches(_small_round_robin_season("2020/21", date(2020, 8, 1), clubs, rng))

    competition = CompetitionConfig(n_clubs=8, relegation_spots=1, european_spots=1)
    reference = build_survival_zone_reference(
        matches, competition=competition, positions=range(5, 7)
    )

    assert reference.n_observations == 2  # two clubs, one season, positions 5-6
    assert reference.home_attack_mean > 0
    assert reference.home_attack_std >= 0


def test_build_prior_pure_anchor_when_no_external_rating():
    reference = SurvivalZoneReference(
        n_observations=10,
        mean_points=34.0,
        home_attack_mean=1.1,
        home_attack_std=0.3,
        home_defence_mean=1.3,
        home_defence_std=0.4,
        away_attack_mean=0.9,
        away_attack_std=0.25,
        away_defence_mean=1.5,
        away_defence_std=0.35,
    )

    prior = build_prior("coventry", reference)

    assert prior.club_id == "coventry"
    assert prior.home_attack_mean == reference.home_attack_mean
    assert prior.home_attack_std == reference.home_attack_std


def test_build_prior_shrinks_toward_external_rating_when_given():
    reference = SurvivalZoneReference(
        n_observations=10,
        mean_points=34.0,
        home_attack_mean=1.0,
        home_attack_std=0.4,
        home_defence_mean=1.3,
        home_defence_std=0.4,
        away_attack_mean=0.9,
        away_attack_std=0.25,
        away_defence_mean=1.5,
        away_defence_std=0.35,
    )

    prior = build_prior(
        "hull",
        reference,
        external_rating={"home_attack": 2.0},
        external_weight=0.5,
    )

    assert prior.home_attack_mean == pytest.approx(1.5)  # halfway between 1.0 and 2.0
    assert prior.home_attack_std < reference.home_attack_std  # narrower once trusted at all
    # Fields with no external rating are untouched.
    assert prior.home_defence_mean == reference.home_defence_mean


def test_build_prior_rejects_out_of_range_weight():
    reference = SurvivalZoneReference(
        n_observations=1,
        mean_points=34.0,
        home_attack_mean=1.0,
        home_attack_std=0.4,
        home_defence_mean=1.3,
        home_defence_std=0.4,
        away_attack_mean=0.9,
        away_attack_std=0.25,
        away_defence_mean=1.5,
        away_defence_std=0.35,
    )
    with pytest.raises(ValueError, match="external_weight"):
        build_prior("x", reference, external_rating={"home_attack": 2.0}, external_weight=1.5)
