"""Walk-forward backtest splitter and orchestration (design.md section 8.2).

Never random k-fold: fit on everything strictly before a cutoff date, predict the next
round of fixtures (matches sharing the next distinct match date), advance. Historical
seasons have no gameweek numbers in our curated data -- FPL's gameweek numbering only
covers the current season -- so "the next round" is defined as the next distinct match
date within a season. That is walk-forward *by date*, exactly as design.md section 8.2
specifies, not an approximation of gameweek-based splitting.

Every model is evaluated on identical splits: this module is shared code, never
reimplemented per model, so a difference in reported RPS between two models is never
explainable by a difference in what data each one saw (design.md section 8.2).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date

import polars as pl
import structlog

from plforecast.evaluate.metrics import (
    brier_score,
    log_loss,
    outcome_index,
    outcome_probabilities,
    rps,
)
from plforecast.models.base import MatchModel, UnknownClubError, drop_odds_columns

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """`scores`: one row per scored test match with the model's 1X2 probabilities and
    per-match RPS, log loss and Brier score. `warmup_excluded`: matches that were never
    a test row because fewer than `min_train_matches` preceded them (the start of the
    first season). `unrateable`: test matches the model could not price because a club
    had no training history yet (promoted-club debuts), one row each. Both exclusions
    are reported separately so evaluation coverage reconciles exactly:
    total = scored + warmup_excluded + len(unrateable)."""

    scores: pl.DataFrame
    warmup_excluded: int
    unrateable: pl.DataFrame

    @property
    def n_scored(self) -> int:
        return self.scores.height


@dataclass(frozen=True, slots=True)
class BacktestSplit:
    season: str
    as_of_date: date
    train: pl.DataFrame
    test: pl.DataFrame


def walk_forward_splits(
    matches: pl.DataFrame, *, min_train_matches: int = 100
) -> Iterator[BacktestSplit]:
    """`matches` shaped like stg_matches, one or more seasons, sorted by date. Splits
    *within* each season: a model is never trained on a season's future to predict its
    own past, and training data accumulates across every prior season plus the current
    season's earlier rounds -- "fit on everything before gameweek k" (design.md section
    8.2), just without a literal gameweek number to key off. `min_train_matches` skips
    the earliest rounds of the very first season in the window, where there isn't
    enough history yet for a meaningful fit.
    """
    matches = matches.sort("date")
    seasons = matches["season"].unique(maintain_order=True).to_list()

    for season_idx, season in enumerate(seasons):
        season_matches = matches.filter(pl.col("season") == season)
        season_dates = season_matches["date"].unique(maintain_order=True).sort()
        prior_seasons = matches.filter(pl.col("season").is_in(seasons[:season_idx]))

        for as_of_date in season_dates:
            train = pl.concat([prior_seasons, season_matches.filter(pl.col("date") < as_of_date)])
            if train.height < min_train_matches:
                continue
            test = season_matches.filter(pl.col("date") == as_of_date)
            yield BacktestSplit(season=season, as_of_date=as_of_date, train=train, test=test)


def run_backtest(
    matches: pl.DataFrame,
    model_factory: Callable[[], MatchModel],
    *,
    min_train_matches: int = 100,
    max_goals: int = 10,
) -> BacktestResult:
    """Runs `model_factory()` (a fresh, unfit model each split -- walk-forward means
    refitting at every step, never reusing a fit across splits) through every split
    from `walk_forward_splits`, scoring each test match with every metric in
    evaluate/metrics.py.

    A test match involving a club with zero appearances in accumulated training data is
    recorded in `unrateable`, not fabricated: every promoted club with no prior-window
    history hits this on its debut. The same real gap features/priors.py exists to fill
    for the model layer; the harness only reports it."""
    rows = []
    unrateable_rows = []
    n_tested = 0
    for split in walk_forward_splits(matches, min_train_matches=min_train_matches):
        # `matches` may legitimately carry odds columns (the caller also scores the
        # market baseline on it); a model's fit() must never see them (design.md 8.4).
        model = model_factory().fit(drop_odds_columns(split.train))
        n_tested += split.test.height
        for row in split.test.iter_rows(named=True):
            key = {
                "season": split.season,
                "date": split.as_of_date,
                "home_club_id": row["home_club_id"],
                "away_club_id": row["away_club_id"],
            }
            try:
                matrix = model.scoreline_matrix(row["home_club_id"], row["away_club_id"], max_goals)
            except UnknownClubError:
                unrateable_rows.append(key)
                continue
            probs = outcome_probabilities(matrix)
            rows.append(
                {
                    **key,
                    "result": row["result"],
                    "p_home": probs[0],
                    "p_draw": probs[1],
                    "p_away": probs[2],
                }
            )

    key_schema = {
        "season": pl.Utf8,
        "date": pl.Date,
        "home_club_id": pl.Utf8,
        "away_club_id": pl.Utf8,
    }
    unrateable = pl.DataFrame(unrateable_rows, schema=key_schema)
    warmup_excluded = matches.height - n_tested
    if unrateable.height:
        log.warning("backtest.unrateable_matches", count=unrateable.height)

    results = pl.DataFrame(
        rows,
        schema={
            **key_schema,
            "result": pl.Utf8,
            "p_home": pl.Float64,
            "p_draw": pl.Float64,
            "p_away": pl.Float64,
        },
    )
    if results.height == 0:
        return BacktestResult(results, warmup_excluded, unrateable)

    probs = results.select("p_home", "p_draw", "p_away").to_numpy()
    outcomes = outcome_index(results["result"].to_list())
    scored = results.with_columns(
        pl.Series("rps", rps(probs, outcomes)),
        pl.Series("log_loss", log_loss(probs, outcomes)),
        pl.Series("brier", brier_score(probs, outcomes)),
    )
    return BacktestResult(scored, warmup_excluded, unrateable)
