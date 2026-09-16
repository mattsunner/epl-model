"""Attack/defence strength inputs, time-decayed (design.md section 6.3 and 6.2).

Rates here are a naive decayed average of a per-match quantity scored and conceded by
venue, nothing more: not a fitted Poisson/Dixon-Coles parameter (those are solved
jointly across every club, net of opponent strength). `metric` selects goals (the
fallback path design.md 13 names for when Understat is unavailable) or xG (design.md
6.3's intended input; requires the xG columns stg_matches carries).

Time decay is non-optional (section 6.3): a squad from three seasons ago is a different
team. `xi`, the decay rate, is a model-fitting decision tuned by backtest (section 6.2)
and is never assumed here -- callers pass it in.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series


class DecayWeightedMatchSchema(pa.DataFrameModel):
    match_id: Series[str] = pa.Field(unique=True)
    date: Series[pl.Date]
    home_club_id: Series[str]
    away_club_id: Series[str]
    home_goals: Series[float] = pa.Field(ge=0)  # goals, or xG when metric="xg"
    away_goals: Series[float] = pa.Field(ge=0)
    days_since: Series[int] = pa.Field(ge=0)
    weight: Series[float] = pa.Field(gt=0, le=1.0)

    class Config:
        strict = False  # extra source columns (result, odds, ...) pass through untouched
        coerce = True


class ClubStrengthSchema(pa.DataFrameModel):
    club_id: Series[str] = pa.Field(unique=True)
    home_attack_rate: Series[float] = pa.Field(nullable=True, ge=0)
    home_defence_rate: Series[float] = pa.Field(nullable=True, ge=0)
    away_attack_rate: Series[float] = pa.Field(nullable=True, ge=0)
    away_defence_rate: Series[float] = pa.Field(nullable=True, ge=0)

    class Config:
        strict = True
        coerce = True


def attach_decay_weights(matches: pl.DataFrame, *, as_of: date, xi: float) -> pl.DataFrame:
    """`matches` shaped like stg_matches (at minimum: match_id, date, home_club_id,
    away_club_id, home_goals, away_goals). Keeps only matches on or before `as_of` --
    a model can never be trained on a result from the future -- and adds `days_since`
    and an exponential-decay `weight = exp(-xi * days_since)`. `xi = 0` disables decay
    (every kept match weighted equally)."""
    return (
        matches.filter(pl.col("date") <= as_of)
        .with_columns((pl.lit(as_of) - pl.col("date")).dt.total_days().alias("days_since"))
        .with_columns((-xi * pl.col("days_since")).exp().alias("weight"))
        .pipe(DecayWeightedMatchSchema.validate)
    )


def build_club_strength(
    matches: pl.DataFrame, *, as_of: date, xi: float, metric: Literal["goals", "xg"] = "goals"
) -> pl.DataFrame:
    """One row per club with at least one match on or before `as_of`: decayed average
    goals scored (`*_attack_rate`) and conceded (`*_defence_rate`), split by venue.

    A club with no rows here has no matches at or before `as_of` in this data --
    typically a newly promoted club with no top-flight history. That gap is exactly
    what features/priors.py exists to fill. This function does not paper over it with
    an assumed league-average rate.

    `metric="xg"` uses `home_xg`/`away_xg` in place of goals (rows with null xG are
    dropped first).
    """
    if metric == "xg":
        matches = matches.drop_nulls(["home_xg", "away_xg"]).with_columns(
            pl.col("home_xg").alias("home_goals"), pl.col("away_xg").alias("away_goals")
        )
    weighted = attach_decay_weights(matches, as_of=as_of, xi=xi)

    if weighted.height == 0:
        # pivot() has no venue values to create columns from on an empty frame, so it
        # can't produce the expected shape -- short-circuit rather than let a case with
        # literally nothing to compute (no matches at all on or before as_of) blow up.
        empty = pl.DataFrame(
            schema={
                "club_id": pl.Utf8,
                "home_attack_rate": pl.Float64,
                "home_defence_rate": pl.Float64,
                "away_attack_rate": pl.Float64,
                "away_defence_rate": pl.Float64,
            }
        )
        return ClubStrengthSchema.validate(empty)

    appearances = pl.concat(
        [
            weighted.select(
                pl.col("home_club_id").alias("club_id"),
                pl.lit("home").alias("venue"),
                pl.col("home_goals").alias("goals_for"),
                pl.col("away_goals").alias("goals_against"),
                "weight",
            ),
            weighted.select(
                pl.col("away_club_id").alias("club_id"),
                pl.lit("away").alias("venue"),
                pl.col("away_goals").alias("goals_for"),
                pl.col("home_goals").alias("goals_against"),
                "weight",
            ),
        ]
    )

    rates = appearances.group_by("club_id", "venue").agg(
        ((pl.col("goals_for") * pl.col("weight")).sum() / pl.col("weight").sum()).alias(
            "attack_rate"
        ),
        ((pl.col("goals_against") * pl.col("weight")).sum() / pl.col("weight").sum()).alias(
            "defence_rate"
        ),
    )

    pivoted = rates.pivot(on="venue", index="club_id", values=["attack_rate", "defence_rate"])
    result = pivoted.rename(
        {
            "attack_rate_home": "home_attack_rate",
            "defence_rate_home": "home_defence_rate",
            "attack_rate_away": "away_attack_rate",
            "defence_rate_away": "away_defence_rate",
        }
    )
    return ClubStrengthSchema.validate(result)
