import numpy as np
import polars as pl
import pytest

from plforecast.models.poisson import PoissonModel


def _synthetic_matches() -> pl.DataFrame:
    """Two seasons of a 4-club round robin with a deliberate skill gap (arsenal and
    chelsea score/concede more favourably than leeds/watford) so the fit has a real
    signal to recover, not just noise."""
    rows = []
    fixtures = [
        ("arsenal", "chelsea", 2, 1),
        ("chelsea", "leeds", 3, 0),
        ("leeds", "watford", 1, 1),
        ("watford", "arsenal", 0, 3),
        ("arsenal", "leeds", 4, 0),
        ("chelsea", "watford", 2, 0),
        ("leeds", "chelsea", 0, 2),
        ("watford", "leeds", 1, 2),
        ("arsenal", "watford", 3, 0),
        ("chelsea", "arsenal", 1, 1),
        ("leeds", "arsenal", 0, 2),
        ("watford", "chelsea", 0, 3),
    ]
    for _ in range(3):  # repeat to give the optimizer enough signal to converge cleanly
        rows.extend(fixtures)
    return pl.DataFrame(
        rows, schema=["home_club_id", "away_club_id", "home_goals", "away_goals"], orient="row"
    )


def test_fit_returns_self():
    model = PoissonModel()
    result = model.fit(_synthetic_matches())
    assert result is model


def test_scoreline_matrix_before_fit_raises():
    with pytest.raises(ValueError, match="call fit"):
        PoissonModel().scoreline_matrix("arsenal", "chelsea")


def test_scoreline_matrix_shape_and_normalisation():
    model = PoissonModel().fit(_synthetic_matches())

    matrix = model.scoreline_matrix("arsenal", "watford", max_goals=10)

    assert matrix.shape == (11, 11)
    assert np.isclose(matrix.sum(), 1.0, atol=1e-6)
    assert (matrix >= 0).all()


def test_scoreline_matrix_reflects_relative_strength():
    """Arsenal beat Watford in every synthetic fixture -- the fitted model should give
    Arsenal a materially higher chance of winning than losing."""
    model = PoissonModel().fit(_synthetic_matches())
    matrix = model.scoreline_matrix("arsenal", "watford", max_goals=10)

    home_win = np.tril(matrix, k=-1).sum()
    away_win = np.triu(matrix, k=1).sum()
    assert home_win > away_win


def test_scoreline_matrix_rejects_unknown_club():
    model = PoissonModel().fit(_synthetic_matches())
    with pytest.raises(ValueError, match="training data"):
        model.scoreline_matrix("arsenal", "not-a-real-club")


def test_config_hash_stable_and_data_independent():
    a = PoissonModel().fit(_synthetic_matches())
    b = PoissonModel()  # never fit

    assert a.config_hash == b.config_hash
    assert isinstance(a.config_hash, str) and len(a.config_hash) == 64
