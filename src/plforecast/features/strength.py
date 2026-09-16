"""Attack/defence strength inputs, time-decayed (design.md section 6.3 and 6.2).

Design intent is xG-driven strength estimation -- finishing variance is largely noise
over a 38-game horizon, and goals alone would flatter or punish teams for shot luck.
That needs Understat, which is not ingested yet (design.md section 13's risk table
names exactly this as a real, anticipated contingency: "Understat scraper breaks
mid-season -> No current xG, model degrades to goals; goals-based fallback path in
features/strength.py"). This module *is* that fallback path, built first because it is
the one available now, not a placeholder -- swapping in xG later is a data-source
change to the functions below, not a redesign.

Time decay is non-optional (section 6.3): a squad from three seasons ago is a different
team. `xi`, the decay rate, is a model-fitting decision tuned by backtest (section 6.2)
and is never assumed here -- callers pass it in.

What this deliberately is not: a fitted Poisson/Dixon-Coles attack-defence parameter.
Those are solved jointly across every club at once, net of opponent strength. The rates
here are a naive decayed average of goals scored/conceded by venue, nothing more -- a
feature the model layer's Poisson fit consumes, not a substitute for that fit.
"""

from __future__ import annotations

from datetime import date

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series


class DecayWeightedMatchSchema(pa.DataFrameModel):
    match_id: Series[str] = pa.Field(unique=True)
    date: Series[pl.Date]
    home_club_id: Series[str]
    away_club_id: Series[str]
    home_goals: Series[int] = pa.Field(ge=0)
    away_goals: Series[int] = pa.Field(ge=0)
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


def build_club_strength(matches: pl.DataFrame, *, as_of: date, xi: float) -> pl.DataFrame:
    """One row per club with at least one match on or before `as_of`: decayed average
    goals scored (`*_attack_rate`) and conceded (`*_defence_rate`), split by venue.

    A club with no rows here has no matches at or before `as_of` in this data --
    typically a newly promoted club with no top-flight history. That gap is exactly
    what features/priors.py (not yet built; needs ClubElo + Transfermarkt) exists to
    fill. This function does not paper over it with an assumed league-average rate.
    """
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
