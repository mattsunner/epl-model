from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from plforecast.features.priors import (
    PRIOR_GATE_XI,
    PromotedClubPrior,
    SurvivalZoneReference,
    build_prior,
    build_survival_zone_reference,
    effective_sample_size,
    needs_prior,
    prior_pseudo_matches,
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


def test_needs_prior_counts_decayed_evidence_when_given_xi():
    # A full season played ten years ago is worth nothing under decay.
    rows = [
        (f"m{i}", "2016/17", date(2016, 8, 10) + timedelta(days=i), "hull", "arsenal", 1, 1)
        for i in range(38)
    ]
    matches = _matches(rows)
    assert needs_prior("hull", matches, min_matches=19) is False  # raw count: 38
    assert needs_prior("hull", matches, min_matches=19, as_of=date(2026, 9, 16), xi=0.0018)
    # The same season played last year still counts for most of its matches.
    assert needs_prior("hull", matches, min_matches=19, as_of=date(2017, 6, 1), xi=0.0018) is False


def test_prior_gate_xi_does_not_flag_a_club_with_a_decade_of_continuous_history():
    """Regression: a club present in every season of a long backfill window must not
    be flagged just because the gate applies decay. This is the exact failure mode a
    fitting-model xi (0.005) produced in the live forecast -- an established club's
    hundreds of historical matches decayed to an effective count near the threshold
    the moment a season was only a few gameweeks old."""
    rows = [
        (
            f"m{season}-{i}",
            f"{2015 + season}/{16 + season:02d}",
            date(2015 + season, 8, 10) + timedelta(days=7 * i),
            "arsenal",
            "opponent",
            2,
            1,
        )
        for season in range(10)
        for i in range(38)
    ]
    matches = _matches(rows)

    assert needs_prior("arsenal", matches, as_of=date(2026, 9, 16), xi=PRIOR_GATE_XI) is False


def test_prior_gate_xi_still_flags_a_club_whose_only_season_is_a_decade_old():
    rows = [
        (
            f"m0-{i}",
            "2016/17",
            date(2016, 8, 13) + timedelta(days=7 * i),
            "hull",
            "opponent",
            1,
            1,
        )
        for i in range(38)
    ]
    matches = _matches(rows)

    assert needs_prior("hull", matches, as_of=date(2026, 9, 16), xi=PRIOR_GATE_XI) is True


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


def _prior(std: float = 0.3) -> PromotedClubPrior:
    return PromotedClubPrior(
        club_id="coventry",
        home_attack_mean=1.1,
        home_attack_std=std,
        home_defence_mean=1.4,
        home_defence_std=std,
        away_attack_mean=0.8,
        away_attack_std=std,
        away_defence_mean=1.7,
        away_defence_std=std,
    )


def test_effective_sample_size_shrinks_with_prior_width_and_is_clamped():
    assert effective_sample_size(_prior(std=0.3)) > effective_sample_size(_prior(std=0.6))
    assert effective_sample_size(_prior(std=10.0)) == 4.0  # floor
    assert effective_sample_size(_prior(std=0.01)) == 38.0  # cap


def test_prior_pseudo_matches_alternate_venue_and_carry_the_prior_as_xg():
    opponents = ["arsenal", "chelsea", "leeds", "coventry"]  # own id must be skipped
    rows = prior_pseudo_matches(
        _prior(), opponents=opponents, season="2026/27", as_of=date(2026, 9, 16), seed=1
    )

    assert rows.height == round(effective_sample_size(_prior()))
    home = rows.filter(pl.col("home_club_id") == "coventry")
    away = rows.filter(pl.col("away_club_id") == "coventry")
    assert abs(home.height - away.height) <= 1
    assert set(home["away_club_id"].to_list()) <= {"arsenal", "chelsea", "leeds"}
    assert home["home_xg"].to_list() == pytest.approx([1.1] * home.height)
    assert home["away_xg"].to_list() == pytest.approx([1.4] * home.height)
    assert away["away_xg"].to_list() == pytest.approx([0.8] * away.height)
    assert rows["home_goals"].dtype == pl.Int64
    assert set(rows["result"].to_list()) <= {"H", "D", "A"}
    assert rows["match_id"].n_unique() == rows.height


def test_prior_pseudo_matches_are_reproducible_by_seed():
    kwargs = dict(opponents=["arsenal", "chelsea"], season="2026/27", as_of=date(2026, 9, 16))
    a = prior_pseudo_matches(_prior(), seed=7, **kwargs)
    b = prior_pseudo_matches(_prior(), seed=7, **kwargs)
    assert a.equals(b)


def test_prior_pseudo_matches_need_a_real_opponent():
    with pytest.raises(ValueError, match="real opponent"):
        prior_pseudo_matches(
            _prior(), opponents=["coventry"], season="2026/27", as_of=date(2026, 9, 16)
        )
