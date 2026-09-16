"""Evaluation report: every model and the market benchmark scored on identical rows
(design.md section 8.2), with exact coverage accounting and calibration curves.

Why an intersection: a walk-forward model cannot price the warm-up rounds of the first
season or a promoted club's debut, and the market has no price where the site
published no closing odds. Scoring each on whatever rows it happens to cover compares
different match sets. The primary numbers here are computed only on rows every model
*and* the benchmark covered; the full-set numbers are kept as a secondary view.

The report is a plain dict so it can be written as JSON and rendered into
docs/evaluation.md without hand-transcription.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from plforecast.evaluate.backtest import BacktestResult, Cadence, run_backtest
from plforecast.evaluate.calibration import calibration_by_outcome
from plforecast.evaluate.market import market_probabilities
from plforecast.evaluate.metrics import brier_score, log_loss, outcome_index, rps
from plforecast.evaluate.season import evaluate_season_level, summarise_season_level
from plforecast.models.base import MatchModel

MATCH_KEY = ["season", "date", "home_club_id", "away_club_id"]
DEVIG_METHODS = ("shin", "multiplicative")


def _score(frame: pl.DataFrame) -> dict[str, float | int]:
    """Mean RPS, log loss and Brier over a frame with p_home/p_draw/p_away and result."""
    if frame.height == 0:
        return {"n": 0, "rps": float("nan"), "log_loss": float("nan"), "brier": float("nan")}
    probs = frame.select("p_home", "p_draw", "p_away").to_numpy()
    outcomes = outcome_index(frame["result"].to_list())
    return {
        "n": frame.height,
        "rps": float(rps(probs, outcomes).mean()),
        "log_loss": float(log_loss(probs, outcomes).mean()),
        "brier": float(brier_score(probs, outcomes).mean()),
    }


def _restrict(frame: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    return frame.join(keys, on=MATCH_KEY, how="semi")


def _by_season(frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (season,), group in frame.group_by("season", maintain_order=True):
        rows.append({"season": season, **_score(group)})
    return sorted(rows, key=lambda r: str(r["season"]))


def shipped_model(primary: list[dict[str, Any]]) -> str:
    """The non-market model with the lowest primary RPS: what the pipeline ships
    (design.md 6.2's gate), computed rather than asserted."""
    candidates = [row for row in primary if not str(row["model"]).startswith("market")]
    return str(min(candidates, key=lambda row: row["rps"])["model"])


def build_report(
    matches: pl.DataFrame,
    model_factories: Mapping[str, Callable[[], MatchModel]],
    *,
    min_train_matches: int = 100,
    calibration_model: str | None = None,
    season_level: bool = True,
    season_level_simulations: int = 5_000,
    tuning: Mapping[str, Any] | None = None,
    cadence: Cadence = "date",
) -> dict[str, Any]:
    """`matches` shaped like stg_matches (benchmark_* columns included), already
    restricted to the evaluation window. Runs every model through the shared
    walk-forward splitter, de-vigs the benchmark both ways, and scores everything on
    the intersection of rows all of them covered.

    `cadence` (story C-09) is the backtest's own refit cadence -- "date" (the finest
    fair match-level protocol) or "gameweek" (refit once per round, mirroring how the
    product actually publishes). Runtime is recorded either way, since "gameweek"
    means far fewer refits and the difference is worth seeing in the rendered report.
    """
    started = time.perf_counter()
    runs: dict[str, BacktestResult] = {
        name: run_backtest(matches, factory, min_train_matches=min_train_matches, cadence=cadence)
        for name, factory in model_factories.items()
    }
    backtest_seconds = time.perf_counter() - started
    market: dict[str, pl.DataFrame] = {
        method: market_probabilities(matches, method=method) for method in DEVIG_METHODS
    }

    keys = matches.select(MATCH_KEY).unique()
    for frame in [*(run.scores for run in runs.values()), *market.values()]:
        keys = keys.join(frame.select(MATCH_KEY).unique(), on=MATCH_KEY, how="inner")

    benchmark_sources = (
        matches.filter(pl.col("benchmark_source").is_not_null())
        .group_by("benchmark_source")
        .len()
        .sort("benchmark_source")
    )

    primary = []
    secondary = []
    by_season: list[dict[str, Any]] = []
    for name, run in runs.items():
        on_intersection = _restrict(run.scores, keys)
        primary.append({"model": name, **_score(on_intersection)})
        secondary.append({"model": name, **_score(run.scores)})
        by_season.extend({"model": name, **row} for row in _by_season(on_intersection))
    for method, frame in market.items():
        name = f"market ({method})"
        on_intersection = _restrict(frame, keys)
        primary.append({"model": name, **_score(on_intersection)})
        secondary.append({"model": name, **_score(frame)})
        by_season.extend({"model": name, **row} for row in _by_season(on_intersection))

    calibration: dict[str, list[dict[str, Any]]] = {}
    calibration_targets = [calibration_model] if calibration_model else list(runs)
    for name in calibration_targets:
        frame = _restrict(runs[name].scores, keys)
        probs = frame.select("p_home", "p_draw", "p_away").to_numpy()
        outcomes = outcome_index(frame["result"].to_list())
        calibration[name] = calibration_by_outcome(probs, outcomes).to_dicts()

    season_scores = (
        evaluate_season_level(matches, model_factories, n_simulations=season_level_simulations)
        if season_level
        else None
    )

    seasons = sorted(matches["season"].unique().to_list())
    return {
        "shipped_model": shipped_model(primary),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": {
            "first_season": seasons[0],
            "last_season": seasons[-1],
            "n_seasons": len(seasons),
            "matches": matches.height,
        },
        "coverage": {
            "warmup_excluded": {name: run.warmup_excluded for name, run in runs.items()},
            "unrateable": {name: run.unrateable.height for name, run in runs.items()},
            "unrateable_matches": {
                name: run.unrateable.with_columns(pl.col("date").cast(pl.Utf8)).to_dicts()
                for name, run in runs.items()
            },
            "benchmark_by_source": {
                str(row["benchmark_source"]): int(row["len"])
                for row in benchmark_sources.to_dicts()
            },
            "benchmark_missing": int(matches["benchmark_source"].null_count()),
            "intersection": keys.height,
        },
        "min_train_matches": min_train_matches,
        "cadence": cadence,
        "backtest_seconds": round(backtest_seconds, 2),
        "primary": primary,
        "secondary_full_set": secondary,
        "by_season": by_season,
        "calibration": calibration,
        "season_level": {
            "simulations": season_level_simulations,
            "summary": summarise_season_level(season_scores) if season_scores is not None else [],
            "scores": season_scores.to_dicts() if season_scores is not None else [],
        },
        "tuning": dict(tuning or {}),
    }


def write_report(report: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Writes metrics.json (the whole report) and calibration.csv (long form) into
    `out_dir`. Returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")

    calibration_path = out_dir / "calibration.csv"
    fieldnames = [
        "model",
        "outcome",
        "bucket_low",
        "bucket_high",
        "mean_predicted",
        "empirical_frequency",
        "n",
        "ci_low",
        "ci_high",
    ]
    with calibration_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model, rows in report["calibration"].items():
            for row in rows:
                writer.writerow({"model": model, **row})
    return [metrics_path, calibration_path]


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def format_table(rows: list[dict[str, Any]]) -> str:
    """Fixed-width text table of model/n/rps/log_loss/brier rows for the terminal."""
    lines = [f"{'model':24s} {'n':>6s} {'rps':>8s} {'log_loss':>9s} {'brier':>8s}"]
    for row in rows:
        lines.append(
            f"{row['model']:24s} {row['n']:6d} {row['rps']:8.4f} "
            f"{row['log_loss']:9.4f} {row['brier']:8.4f}"
        )
    return "\n".join(lines)
