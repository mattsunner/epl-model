"""Independent Poisson match model (design.md section 6.2, rung 1: "independent
Poisson, static team strengths. Floor."). Every model further up the ladder must beat
this one on held-out RPS or it does not ship.

Static team strengths: unlike Dixon-Coles (rung 2), this model applies no time decay
and no low-score dependence correction -- every match `fit()` is given is weighted
equally, whichever season it is from. That is deliberate, not an oversight: this is the
floor the rest of the ladder is measured against, and decay is introduced one rung up.

Fitting -- the joint maximum-likelihood estimation of an attack and defence strength
per club plus one shared home-advantage term, solved together so each club's rating
already accounts for the strength of whoever it played -- is delegated to
`penaltyblog`, which design.md section 11.1 already anticipates using (for de-vigging)
and which ships exactly this model (the classic Maher 1982 formulation). Reimplementing
that fit by hand would just be re-deriving a well-tested library's Poisson regression.
"""

from __future__ import annotations

import hashlib
import json
from typing import Self

import numpy as np
import penaltyblog as pb
import polars as pl

from plforecast.models.base import ClubId, UnknownClubError


class PoissonModel:
    """`fit()` expects `matches` shaped like stg_matches: at minimum `home_club_id`,
    `away_club_id`, `home_goals`, `away_goals`. Every row is used, unweighted -- callers
    decide the training window (e.g. everything up to some as-of date) by choosing
    which rows to pass in; this model does not filter by date itself."""

    def __init__(self) -> None:
        self._model: pb.models.PoissonGoalsModel | None = None
        self._clubs: set[ClubId] = set()

    def fit(self, matches: pl.DataFrame) -> Self:
        self._model = pb.models.PoissonGoalsModel(
            goals_home=matches["home_goals"].to_list(),
            goals_away=matches["away_goals"].to_list(),
            teams_home=matches["home_club_id"].to_list(),
            teams_away=matches["away_club_id"].to_list(),
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
        # penaltyblog's own max_goals is an exclusive upper bound -- predict(max_goals=N)
        # returns an N x N grid for scores 0..N-1 -- while design.md's protocol wants an
        # inclusive (max_goals+1, max_goals+1) matrix for scores 0..max_goals.
        grid = self._model.predict(home, away, max_goals=max_goals + 1).grid
        return np.asarray(grid)

    @property
    def config_hash(self) -> str:
        """Hashes the model's *configuration* (class identity plus hyperparameters),
        not its fitted weights or training data -- data identity is already covered
        separately by the forecast artifact's source content hashes (design.md section
        9.3). Plain Poisson has no tunable hyperparameters, so this is currently just a
        stable identity hash; it starts doing real work once a model has real knobs
        (Dixon-Coles' decay `xi`, for instance)."""
        config = {"model": "poisson", "version": 1}
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
