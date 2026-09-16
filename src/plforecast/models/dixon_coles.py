"""Dixon-Coles match model (design.md section 6.2, rung 2): low-score dependence
correction plus exponential time decay on top of the independent-Poisson floor. Must
beat PoissonModel on held-out RPS or it does not ship; `evaluate/report.py` makes that
check and docs/evaluation.md records the result.

Time decay is non-optional (design.md section 6.3): a squad from three seasons ago is a
different team. `xi` -- how fast old matches lose influence -- is a genuine model
hyperparameter tuned by backtest, never assumed here, so it is a required constructor
argument with no default: a caller has to make an explicit choice, not silently inherit
one. (penaltyblog defaults its own decay helper to 0.0018, Dixon and Coles' original
1997 value; passing it here would still be a visible choice, just one this module
doesn't make on the caller's behalf.)

Low-score dependence: Dixon and Coles' rho correction adjusts the independent-Poisson
joint probability for the four low-scoring outcomes (0-0, 1-0, 0-1, 1-1), where real
match data shows systematic dependence between the two teams' goal counts that an
independent Poisson can't capture. penaltyblog's DixonColesGoalModel fits this jointly
alongside attack/defence/home-advantage -- same delegation rationale as PoissonModel:
a well-tested implementation of a well-established method, not something worth
re-deriving by hand.
"""

from __future__ import annotations

import hashlib
import json
from typing import Self

import numpy as np
import penaltyblog as pb
import polars as pl

from plforecast.models.base import ClubId, UnknownClubError


class DixonColesModel:
    """`fit()` expects `matches` shaped like stg_matches: at minimum `home_club_id`,
    `away_club_id`, `home_goals`, `away_goals`, `date`. Every row contributes, but
    weighted by `exp(-xi * days_since_match)` relative to the most recent match date in
    `matches` -- unlike PoissonModel, which weights every row equally."""

    def __init__(self, xi: float) -> None:
        self.xi = xi
        self._model: pb.models.DixonColesGoalModel | None = None
        self._clubs: set[ClubId] = set()

    def fit(self, matches: pl.DataFrame) -> Self:
        weights = pb.models.dixon_coles_weights(matches["date"].to_list(), xi=self.xi)
        self._model = pb.models.DixonColesGoalModel(
            goals_home=matches["home_goals"].to_list(),
            goals_away=matches["away_goals"].to_list(),
            teams_home=matches["home_club_id"].to_list(),
            teams_away=matches["away_club_id"].to_list(),
            weights=weights,
        )
        self._model.fit()
        self._clubs = set(matches["home_club_id"]) | set(matches["away_club_id"])
        return self

    def scoreline_matrix(self, home: ClubId, away: ClubId, max_goals: int = 10) -> np.ndarray:
        if self._model is None:
            raise ValueError("call fit() before scoreline_matrix()")
        unknown = [club for club in (home, away) if club not in self._clubs]
        if unknown:
            raise UnknownClubError(f"club(s) not in training data: {unknown!r}")
        # Same off-by-one translation as PoissonModel: penaltyblog's max_goals is an
        # exclusive upper bound, design.md's protocol wants an inclusive one.
        grid = self._model.predict(home, away, max_goals=max_goals + 1).grid
        return np.asarray(grid)

    @property
    def config_hash(self) -> str:
        """Unlike PoissonModel, this model has a real hyperparameter to capture: two
        DixonColesModel instances with different `xi` are different configurations and
        must hash differently."""
        config = {"model": "dixon-coles", "version": 1, "xi": self.xi}
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
