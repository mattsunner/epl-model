import numpy as np
import pytest

from plforecast.evaluate.metrics import (
    brier_score,
    log_loss,
    outcome_index,
    outcome_probabilities,
    rps,
)


def test_outcome_index_maps_result_codes():
    assert outcome_index(["H", "D", "A"]).tolist() == [0, 1, 2]


def test_outcome_probabilities_from_scoreline_matrix():
    # 3x3 grid: home wins on (1,0),(2,0),(2,1); draws on diagonal; away wins elsewhere.
    matrix = np.array(
        [
            [0.10, 0.05, 0.05],
            [0.20, 0.10, 0.05],
            [0.20, 0.15, 0.10],
        ]
    )
    probs = outcome_probabilities(matrix)
    assert probs == pytest.approx([0.20 + 0.20 + 0.15, 0.10 + 0.10 + 0.10, 0.05 + 0.05 + 0.05])


def test_rps_is_zero_for_a_perfect_certain_forecast():
    probs = np.array([[1.0, 0.0, 0.0]])
    outcomes = outcome_index(["H"])
    assert rps(probs, outcomes)[0] == pytest.approx(0.0)


def test_rps_penalises_a_miss_to_the_adjacent_class_less_than_the_far_class():
    outcomes = outcome_index(["H"])
    predicted_draw = rps(np.array([[0.0, 1.0, 0.0]]), outcomes)[0]
    predicted_away = rps(np.array([[0.0, 0.0, 1.0]]), outcomes)[0]
    # Confidently predicting a draw when it's a home win is a smaller ranked miss than
    # confidently predicting an away win -- the entire point of using RPS over an
    # unordered metric like Brier for a 1X2 market.
    assert predicted_draw < predicted_away


def test_log_loss_is_near_zero_for_a_confident_correct_call():
    probs = np.array([[0.99, 0.005, 0.005]])
    outcomes = outcome_index(["H"])
    assert log_loss(probs, outcomes)[0] < 0.02


def test_log_loss_does_not_blow_up_on_a_confident_wrong_call():
    probs = np.array([[1.0, 0.0, 0.0]])
    outcomes = outcome_index(["A"])
    assert np.isfinite(log_loss(probs, outcomes)[0])


def test_brier_score_is_unordered_unlike_rps():
    outcomes = outcome_index(["H"])
    predicted_draw = brier_score(np.array([[0.0, 1.0, 0.0]]), outcomes)[0]
    predicted_away = brier_score(np.array([[0.0, 0.0, 1.0]]), outcomes)[0]
    # Unlike RPS, Brier treats "wrong" as equally wrong regardless of ordinal distance.
    assert predicted_draw == pytest.approx(predicted_away)
