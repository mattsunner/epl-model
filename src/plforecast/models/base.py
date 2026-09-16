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


def _odds_columns(matches: pl.DataFrame) -> list[str]:
    return [c for c in matches.columns if "odds" in c.lower()]


def assert_no_odds_columns(matches: pl.DataFrame) -> None:
    """Design.md section 8.4's leakage guard: odds are the market benchmark every model
    is scored against, never a model input. Every `fit()` in this package calls this
    first, so passing a raw `stg_matches`-shaped frame (which carries
    `benchmark_*_odds`) straight to a model fails loudly instead of quietly training on
    the thing it is meant to be compared against."""
    leaked = _odds_columns(matches)
    if leaked:
        raise ValueError(
            f"matches passed to fit() must not carry odds columns (found {leaked}); "
            "odds are the market benchmark, never a model input (design.md 8.4)"
        )


def drop_odds_columns(matches: pl.DataFrame) -> pl.DataFrame:
    """The other half of the same guard: a caller (the backtest) that legitimately
    needs odds columns on its own frame -- to score the market baseline on the same
    rows a model was tested on -- drops them before handing that frame to a model's
    `fit()`, rather than the model ever seeing them."""
    return matches.drop(_odds_columns(matches))


class MatchModel(Protocol):
    def fit(self, matches: pl.DataFrame) -> Self: ...

    def scoreline_matrix(self, home: ClubId, away: ClubId, max_goals: int = 10) -> np.ndarray: ...

    @property
    def config_hash(self) -> str: ...
