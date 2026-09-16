from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from plforecast.models.base import UnknownClubError
from plforecast.models.xg_rates import XGRateModel

CLUBS = ["arsenal", "chelsea", "leeds", "watford", "everton", "fulham"]


def _synthetic(seed: int = 0, seasons: int = 2) -> pl.DataFrame:
    """Round robin with a known strength ordering; xG is the underlying rate plus
    noise, goals are Poisson draws from it."""
    rng = np.random.default_rng(seed)
    strength = {c: 0.35 - 0.14 * i for i, c in enumerate(CLUBS)}  # arsenal strongest
    rows = []
    day = 0
    for _ in range(seasons):
        for h in CLUBS:
            for a in CLUBS:
                if h == a:
                    continue
                lam_h = np.exp(0.25 + 0.2 + strength[h] - strength[a])
                lam_a = np.exp(0.25 + strength[a] - strength[h])
                rows.append(
                    {
                        "home_club_id": h,
                        "away_club_id": a,
                        "home_goals": int(rng.poisson(lam_h)),
                        "away_goals": int(rng.poisson(lam_a)),
                        "home_xg": float(max(lam_h + rng.normal(0, 0.2), 0.05)),
                        "away_xg": float(max(lam_a + rng.normal(0, 0.2), 0.05)),
                        "date": date(2024, 8, 1) + timedelta(days=day),
                    }
                )
                day += 1
    return pl.DataFrame(rows)


def test_fit_recovers_strength_ordering_from_xg():
    model = XGRateModel(xi=0.0).fit(_synthetic())
    lam_arsenal, _ = model.rates("arsenal", "leeds")
    lam_fulham, _ = model.rates("fulham", "leeds")
    assert lam_arsenal > lam_fulham
    assert model._home_advantage > 0


def test_scoreline_matrix_is_a_distribution():
    model = XGRateModel(xi=0.001).fit(_synthetic())
    matrix = model.scoreline_matrix("arsenal", "chelsea", max_goals=8)
    assert matrix.shape == (9, 9)
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()


def test_rho_correction_keeps_a_distribution_and_changes_low_scores():
    frame = _synthetic()
    plain = XGRateModel(xi=0.0).fit(frame).scoreline_matrix("arsenal", "chelsea")
    corrected = XGRateModel(xi=0.0, rho=-0.1).fit(frame).scoreline_matrix("arsenal", "chelsea")
    assert corrected.sum() == pytest.approx(1.0)
    assert corrected[1, 1] != pytest.approx(plain[1, 1])


def test_unknown_club_raises_the_shared_error():
    model = XGRateModel(xi=0.0).fit(_synthetic())
    with pytest.raises(UnknownClubError):
        model.scoreline_matrix("arsenal", "not-a-club")


def test_scoreline_before_fit_raises():
    with pytest.raises(ValueError, match="call fit"):
        XGRateModel(xi=0.0).scoreline_matrix("a", "b")


def test_blend_zero_uses_goals_and_ignores_null_xg():
    frame = _synthetic().with_columns(pl.lit(None, dtype=pl.Float64).alias("home_xg"))
    model = XGRateModel(xi=0.0, blend=0.0).fit(frame)
    assert model.scoreline_matrix("arsenal", "chelsea").sum() == pytest.approx(1.0)


def test_weight_column_and_decay_move_the_fit():
    frame = _synthetic()
    base = XGRateModel(xi=0.0).fit(frame).rates("arsenal", "fulham")
    # Up-weighting the second season only (same generating process) should change
    # the estimates slightly but not their direction.
    weighted = frame.with_columns(
        pl.when(pl.col("date") > date(2024, 9, 1)).then(5.0).otherwise(1.0).alias("weight")
    )
    shifted = XGRateModel(xi=0.0).fit(weighted).rates("arsenal", "fulham")
    assert shifted != base
    assert shifted[0] > shifted[1]


def test_config_hash_reflects_every_knob():
    hashes = {
        XGRateModel(xi=0.001).config_hash,
        XGRateModel(xi=0.002).config_hash,
        XGRateModel(xi=0.001, blend=0.5).config_hash,
        XGRateModel(xi=0.001, rho=-0.1).config_hash,
    }
    assert len(hashes) == 4


def test_rejects_blend_out_of_range():
    with pytest.raises(ValueError, match="blend"):
        XGRateModel(xi=0.0, blend=1.5)
