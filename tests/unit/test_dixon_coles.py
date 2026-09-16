from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from plforecast.models.dixon_coles import DixonColesModel

CLUBS = ["arsenal", "chelsea", "leeds", "watford", "everton", "fulham", "brentford", "sunderland"]


def _synthetic_matches() -> pl.DataFrame:
    """A full round robin among 8 clubs, repeated across 3 seasons with randomised
    (but seeded) scorelines. Dixon-Coles fits one extra parameter (rho, the low-score
    correlation correction) beyond Poisson, and needs enough data -- especially enough
    low-scoring matches -- to fit it stably; a handful of fixtures among 4 clubs isn't
    enough and can produce a mathematically invalid (negative) grid cell."""
    rng = np.random.default_rng(0)
    fixtures = [(h, a) for h in CLUBS for a in CLUBS if h != a]
    rows = []
    start = date(2023, 8, 1)
    day = 0
    for _season in range(3):
        for home, away in fixtures:
            hg, ag = int(rng.poisson(1.4)), int(rng.poisson(1.1))
            rows.append((home, away, hg, ag, start + timedelta(days=day)))
            day += 1
    return pl.DataFrame(
        rows,
        schema=["home_club_id", "away_club_id", "home_goals", "away_goals", "date"],
        orient="row",
    )


def _reversing_form_matches() -> pl.DataFrame:
    """Arsenal dominates early (old matches), Watford dominates late (recent matches).
    A model that actually applies decay should end up favouring Watford; a model that
    ignores it (xi=0) sees a wash across the full history."""
    rows = []
    early = date(2015, 8, 1)
    recent = date(2026, 8, 1)
    for i in range(20):
        rows.append(("arsenal", "watford", 4, 0, early + timedelta(days=i)))
        rows.append(("watford", "arsenal", 3, 0, early + timedelta(days=i)))
    for i in range(20):
        rows.append(("arsenal", "watford", 0, 4, recent + timedelta(days=i)))
        rows.append(("watford", "arsenal", 0, 3, recent + timedelta(days=i)))
    return pl.DataFrame(
        rows,
        schema=["home_club_id", "away_club_id", "home_goals", "away_goals", "date"],
        orient="row",
    )


def test_fit_returns_self():
    model = DixonColesModel(xi=0.0018)
    result = model.fit(_synthetic_matches())
    assert result is model


def test_scoreline_matrix_before_fit_raises():
    with pytest.raises(ValueError, match="call fit"):
        DixonColesModel(xi=0.0018).scoreline_matrix("arsenal", "chelsea")


def test_scoreline_matrix_shape_and_normalisation():
    model = DixonColesModel(xi=0.0018).fit(_synthetic_matches())
    matrix = model.scoreline_matrix("arsenal", "watford", max_goals=10)

    assert matrix.shape == (11, 11)
    assert np.isclose(matrix.sum(), 1.0, atol=1e-6)
    assert (matrix >= 0).all()


def test_scoreline_matrix_rejects_unknown_club():
    model = DixonColesModel(xi=0.0018).fit(_synthetic_matches())
    with pytest.raises(ValueError, match="training data"):
        model.scoreline_matrix("arsenal", "not-a-real-club")


def test_config_hash_reflects_xi():
    a = DixonColesModel(xi=0.001)
    b = DixonColesModel(xi=0.001)
    c = DixonColesModel(xi=0.01)

    assert a.config_hash == b.config_hash
    assert a.config_hash != c.config_hash


def test_decay_actually_shifts_the_fit_toward_recent_form():
    matches = _reversing_form_matches()

    undecayed = DixonColesModel(xi=0.0).fit(matches)
    decayed = DixonColesModel(xi=0.01).fit(matches)

    def home_win_prob(model: DixonColesModel) -> float:
        m = model.scoreline_matrix("arsenal", "watford", max_goals=8)
        return float(np.tril(m, k=-1).sum())

    # With no decay, 20 early Arsenal blowouts and 20 recent Watford blowouts roughly
    # cancel out. With real decay, only the recent (Watford-dominant) matches carry
    # weight, so Arsenal's home-win probability should be materially lower.
    assert home_win_prob(decayed) < home_win_prob(undecayed) - 0.05
