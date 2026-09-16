"""Shared model-layer contract (design.md section 6.1). Every match model, including
the market baseline once it exists, implements this protocol. `scoreline_matrix` is the
single primitive everything downstream derives from -- 1X2 probabilities, over/under,
simulation sampling -- which is what makes benchmarking a one-line substitution rather
than a special case.
"""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
import polars as pl

ClubId = str


class UnknownClubError(ValueError):
    """Raised by `scoreline_matrix` when either club was absent from the frame passed to
    `fit()`. Its own type so callers (the backtest) can skip exactly this case without
    swallowing every other ValueError a model might raise."""


class MatchModel(Protocol):
    def fit(self, matches: pl.DataFrame) -> Self: ...

    def scoreline_matrix(self, home: ClubId, away: ClubId, max_goals: int = 10) -> np.ndarray: ...

    @property
    def config_hash(self) -> str: ...
