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


class MatchModel(Protocol):
    def fit(self, matches: pl.DataFrame) -> Self: ...

    def scoreline_matrix(self, home: ClubId, away: ClubId, max_goals: int = 10) -> np.ndarray: ...

    @property
    def config_hash(self) -> str: ...
