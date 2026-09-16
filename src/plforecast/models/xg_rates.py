"""xG-rate match model (ADR 0009, rung 2.5 of the ladder).

Attack, defence and home-advantage terms are fitted on expected goals rather than
goals, by weighted least squares in log space:

    log(target_home) = mu + h + att[home] - def[away]
    log(target_away) = mu     + att[away] - def[home]

with sum-to-zero constraints on `att` and `def`, exponential time decay
(`exp(-xi * days_since)`) and an optional per-row `weight` column (the promoted-club
prior enters as weighted pseudo-observations, story C-08). `target` is xG by default;
`blend` mixes goals back in (`blend * xg + (1 - blend) * goals`) so the goals/xG choice
can be tuned by backtest rather than assumed.

Scorelines are independent Poissons at the fitted rates, with Dixon and Coles' low-score
correction applied when `rho` is non-zero. Fitting is a closed-form solve, so the
walk-forward backtest costs milliseconds per refit.
"""

from __future__ import annotations

import hashlib
import json
from typing import Self

import numpy as np
import polars as pl
from scipy.stats import poisson

from plforecast.models.base import ClubId, UnknownClubError

CONSTRAINT_WEIGHT = 1e3


class XGRateModel:
    """`fit()` expects `matches` shaped like stg_matches: `home_club_id`, `away_club_id`,
    `home_goals`, `away_goals`, `home_xg`, `away_xg`, `date`, and optionally `weight`.
    Rows with null xG are dropped when `blend > 0`."""

    def __init__(
        self, xi: float, *, blend: float = 1.0, rho: float = 0.0, floor: float = 0.05
    ) -> None:
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {blend}")
        self.xi = xi
        self.blend = blend
        self.rho = rho
        self.floor = floor
        self._clubs: list[ClubId] = []
        self._index: dict[ClubId, int] = {}
        self._mu = 0.0
        self._home_advantage = 0.0
        self._attack: np.ndarray = np.zeros(0)
        self._defence: np.ndarray = np.zeros(0)

    def fit(self, matches: pl.DataFrame) -> Self:
        frame = matches
        if self.blend > 0:
            frame = frame.drop_nulls(["home_xg", "away_xg"])
        if frame.height == 0:
            raise ValueError("no rows with xG to fit on")

        clubs = sorted(set(frame["home_club_id"]) | set(frame["away_club_id"]))
        index = {club: i for i, club in enumerate(clubs)}
        n = len(clubs)

        home_target = self._target(frame, "home")
        away_target = self._target(frame, "away")
        weights = self._weights(frame)

        # Two rows per match: [mu, h, att..., def...]
        n_rows = 2 * frame.height
        design = np.zeros((n_rows, 2 + 2 * n))
        target = np.empty(n_rows)
        row_weights = np.empty(n_rows)
        home_idx = np.array([index[c] for c in frame["home_club_id"].to_list()])
        away_idx = np.array([index[c] for c in frame["away_club_id"].to_list()])
        rows = np.arange(frame.height)

        design[2 * rows, 0] = 1.0
        design[2 * rows, 1] = 1.0
        design[2 * rows, 2 + home_idx] = 1.0
        design[2 * rows, 2 + n + away_idx] = -1.0
        target[2 * rows] = np.log(home_target)
        row_weights[2 * rows] = weights

        design[2 * rows + 1, 0] = 1.0
        design[2 * rows + 1, 2 + away_idx] = 1.0
        design[2 * rows + 1, 2 + n + home_idx] = -1.0
        target[2 * rows + 1] = np.log(away_target)
        row_weights[2 * rows + 1] = weights

        # Sum-to-zero constraints as heavily weighted pseudo-rows.
        constraints = np.zeros((2, 2 + 2 * n))
        constraints[0, 2 : 2 + n] = 1.0
        constraints[1, 2 + n :] = 1.0
        design = np.vstack([design, constraints])
        target = np.concatenate([target, [0.0, 0.0]])
        row_weights = np.concatenate([row_weights, [CONSTRAINT_WEIGHT, CONSTRAINT_WEIGHT]])

        sqrt_w = np.sqrt(row_weights)[:, None]
        solution, *_ = np.linalg.lstsq(design * sqrt_w, target * sqrt_w[:, 0], rcond=None)

        self._clubs = clubs
        self._index = index
        self._mu = float(solution[0])
        self._home_advantage = float(solution[1])
        self._attack = solution[2 : 2 + n]
        self._defence = solution[2 + n :]
        return self

    def _target(self, frame: pl.DataFrame, side: str) -> np.ndarray:
        goals = frame[f"{side}_goals"].cast(pl.Float64).to_numpy()
        if self.blend == 0:
            values = goals
        else:
            xg = frame[f"{side}_xg"].cast(pl.Float64).to_numpy()
            values = self.blend * xg + (1 - self.blend) * goals
        return np.asarray(np.maximum(values, self.floor), dtype=float)

    def _weights(self, frame: pl.DataFrame) -> np.ndarray:
        days_since = (
            frame.select((pl.col("date").max() - pl.col("date")).dt.total_days().cast(pl.Float64))
            .to_series()
            .to_numpy()
        )
        decay = np.exp(-self.xi * days_since)
        if "weight" in frame.columns:
            decay = decay * frame["weight"].cast(pl.Float64).fill_null(1.0).to_numpy()
        return np.asarray(decay, dtype=float)

    def rates(self, home: ClubId, away: ClubId) -> tuple[float, float]:
        unknown = [club for club in (home, away) if club not in self._index]
        if unknown:
            raise UnknownClubError(f"club(s) not in training data: {unknown!r}")
        i, j = self._index[home], self._index[away]
        lam_home = np.exp(self._mu + self._home_advantage + self._attack[i] - self._defence[j])
        lam_away = np.exp(self._mu + self._attack[j] - self._defence[i])
        return float(lam_home), float(lam_away)

    def scoreline_matrix(self, home: ClubId, away: ClubId, max_goals: int = 10) -> np.ndarray:
        if not self._clubs:
            raise ValueError("call fit() before scoreline_matrix()")
        lam_home, lam_away = self.rates(home, away)
        goals = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
        if self.rho:
            matrix[0, 0] *= 1 - lam_home * lam_away * self.rho
            matrix[0, 1] *= 1 + lam_home * self.rho
            matrix[1, 0] *= 1 + lam_away * self.rho
            matrix[1, 1] *= 1 - self.rho
            matrix = np.clip(matrix, 0.0, None)
        return np.asarray(matrix / matrix.sum())

    @property
    def config_hash(self) -> str:
        config = {
            "model": "xg-rates",
            "version": 1,
            "xi": self.xi,
            "blend": self.blend,
            "rho": self.rho,
            "floor": self.floor,
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
