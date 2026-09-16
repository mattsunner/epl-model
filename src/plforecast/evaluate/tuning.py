"""Hyperparameter tuning by walk-forward backtest with disjoint selection and reporting
seasons (design.md section 13: "tune on a holdout period disjoint from the reporting
period"; story C-03).

Every grid value runs through the same splitter on the same matches, so the scored rows
are identical across the grid and the comparison is like-for-like. The value with the
lowest mean RPS on the selection seasons is chosen; its score on the later reporting
seasons is what gets published, so the reported number is never the one that was
optimised.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import polars as pl
import structlog

from plforecast.evaluate.backtest import run_backtest
from plforecast.models.base import MatchModel

log = structlog.get_logger()


def _mean(frame: pl.DataFrame, column: str) -> float:
    if frame.height == 0:
        return float("nan")
    return float(frame.select(pl.col(column).mean()).item())


def tune_parameter(
    matches: pl.DataFrame,
    make_factory: Callable[[float], Callable[[], MatchModel]],
    grid: Sequence[float],
    *,
    parameter: str,
    select_through_season: str,
    min_train_matches: int = 100,
    null_value: float | None = None,
    tolerance: float = 0.0005,
) -> dict[str, Any]:
    """`make_factory(value)` returns a model factory with `parameter` set to `value`.
    Seasons up to and including `select_through_season` select; later seasons report.

    Parsimony rule: when `null_value` (the value that switches the parameter off, e.g.
    0 for a correlation correction) is in the grid and its selection RPS is within
    `tolerance` of the best, the null value is selected. A flat grid should not pick a
    boundary value on noise."""
    rows = []
    for value in grid:
        result = run_backtest(matches, make_factory(value), min_train_matches=min_train_matches)
        scores = result.scores
        select = scores.filter(pl.col("season") <= select_through_season)
        report = scores.filter(pl.col("season") > select_through_season)
        rows.append(
            {
                "value": float(value),
                "select_n": select.height,
                "select_rps": _mean(select, "rps"),
                "report_n": report.height,
                "report_rps": _mean(report, "rps"),
            }
        )
        log.info("tuning.grid_point", parameter=parameter, **rows[-1])

    best = min(rows, key=lambda r: r["select_rps"])
    selected = best["value"]
    parsimony_applied = False
    if null_value is not None:
        null_row = next((r for r in rows if r["value"] == float(null_value)), None)
        if null_row is not None and null_row["select_rps"] - best["select_rps"] <= tolerance:
            selected = null_row["value"]
            parsimony_applied = selected != best["value"]
    seasons = sorted(matches["season"].unique().to_list())
    return {
        "parameter": parameter,
        "grid": rows,
        "selected": selected,
        "best_by_selection_rps": best["value"],
        "null_value": null_value,
        "tolerance": tolerance,
        "parsimony_applied": parsimony_applied,
        "selection_seasons": [s for s in seasons if s <= select_through_season],
        "report_seasons": [s for s in seasons if s > select_through_season],
    }
