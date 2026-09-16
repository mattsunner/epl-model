"""Scoring rules for 1X2 outcome forecasts (design.md section 8.1).

Every function here takes a `(n, 3)` array of outcome probabilities -- columns ordered
[home win, draw, away win], each row summing to 1 -- and an `(n,)` array of outcome
indices in that same order (0=home win, 1=draw, 2=away win), and returns an `(n,)` array
of per-match scores. Aggregation (mean across matches, across a season, across the
whole backtest) is deliberately left to the caller: evaluate/backtest.py needs those
means sliced different ways (per model, per season, per split), and a function that
already collapsed to a single scalar would force recomputation to get any other view.
"""

from __future__ import annotations

import numpy as np

RESULT_TO_OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def outcome_index(results: list[str]) -> np.ndarray:
    """Maps stg_matches' `result` column ("H"/"D"/"A") to the 0/1/2 outcome index used
    by every function below."""
    return np.array([RESULT_TO_OUTCOME_INDEX[r] for r in results])


def outcome_probabilities(scoreline_matrix: np.ndarray) -> np.ndarray:
    """Collapses a joint scoreline matrix (design.md section 6.1's MatchModel output)
    into [P(home win), P(draw), P(away win)] -- the lower triangle, diagonal, and
    upper triangle of the matrix respectively."""
    home_win = np.tril(scoreline_matrix, k=-1).sum()
    draw = np.trace(scoreline_matrix)
    away_win = np.triu(scoreline_matrix, k=1).sum()
    return np.array([home_win, draw, away_win])


def _one_hot(outcomes: np.ndarray, n_classes: int) -> np.ndarray:
    one_hot = np.zeros((len(outcomes), n_classes))
    one_hot[np.arange(len(outcomes)), outcomes] = 1.0
    return one_hot


def rps(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Ranked probability score (design.md section 8.1: the primary metric, "the
    accepted standard for football match forecasting because it rewards getting close
    on an ordered outcome, which accuracy and log loss do not"). Outcomes are treated
    as ordered [home win, draw, away win] -- predicting a draw when the result is a
    home win is scored as closer than predicting an away win, which is the entire
    point of using RPS over log loss or Brier here. Lower is better; 0 is a perfect
    forecast, 1 is maximally wrong."""
    n_classes = probs.shape[1]
    one_hot = _one_hot(outcomes, n_classes)
    cum_probs = np.cumsum(probs, axis=1)
    cum_outcomes = np.cumsum(one_hot, axis=1)
    return np.asarray(((cum_probs - cum_outcomes) ** 2).sum(axis=1) / (n_classes - 1))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Negative log likelihood of the realised outcome. Lower is better. Clipped away
    from exactly 0 so a model that was (wrongly) certain doesn't score infinitely bad
    on one match and dominate an aggregate mean."""
    eps = 1e-15
    p_actual = probs[np.arange(len(outcomes)), outcomes]
    return np.asarray(-np.log(np.clip(p_actual, eps, 1 - eps)))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Multi-class Brier score: summed squared error between predicted probabilities
    and the one-hot realised outcome, unordered (unlike RPS, missing a draw by
    predicting a home win or an away win are scored identically here). Lower is
    better."""
    n_classes = probs.shape[1]
    one_hot = _one_hot(outcomes, n_classes)
    return np.asarray(((probs - one_hot) ** 2).sum(axis=1))
