"""Rest days between a club's fixtures (design.md section 6.3): built from v1 because
it is nearly free and belongs in the curated tables regardless, even though it does not
enter the model until the hierarchical baseline is established (the genuine residual is
a small rest-day effect; a naive European-competition indicator would just re-encode
team strength, since the strongest clubs are also the ones playing in Europe).

European congestion itself is deferred entirely -- there is no ingested source for it --
so this module only ever computes rest days, not the fuller "schedule congestion"
scope its docstring name implies down the line.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series


class RestDaysSchema(pa.DataFrameModel):
    fixture_id: Series[int] = pa.Field(ge=1, unique=True)
    home_rest_days: Series[float] = pa.Field(nullable=True, ge=0)
    away_rest_days: Series[float] = pa.Field(nullable=True, ge=0)

    class Config:
        strict = True
        coerce = True


def build_rest_days(fixtures: pl.DataFrame) -> pl.DataFrame:
    """`fixtures` needs `fixture_id`, `kickoff_time`, `home_club_id`, `away_club_id` --
    exactly mart_fixtures' shape. One row per fixture: how many days each side had
    since its previous fixture *in this same season's schedule*. Null for a club's
    first fixture of the season -- mart_fixtures is FPL's current-season fixture list,
    so there is no earlier kickoff within it to measure from, and a summer break is not
    a rest-days signal worth inventing a number for.

    A fixture with a null kickoff_time (an unrescheduled postponement) cannot be placed
    in chronological order and is excluded from the rest-day chain entirely -- both its
    own rest days and its neighbours' are affected. None of the currently ingested data
    has this case; if it arises, treat any nearby rest-day values as unreliable.
    """
    appearances = pl.concat(
        [
            fixtures.select(
                pl.col("fixture_id"),
                pl.col("home_club_id").alias("club_id"),
                pl.col("kickoff_time"),
                pl.lit(True).alias("is_home"),
            ),
            fixtures.select(
                pl.col("fixture_id"),
                pl.col("away_club_id").alias("club_id"),
                pl.col("kickoff_time"),
                pl.lit(False).alias("is_home"),
            ),
        ]
    ).filter(pl.col("kickoff_time").is_not_null())

    with_rest = appearances.sort("club_id", "kickoff_time").with_columns(
        (
            (
                pl.col("kickoff_time") - pl.col("kickoff_time").shift(1).over("club_id")
            ).dt.total_minutes()
            / (24 * 60)
        ).alias("rest_days")
    )

    home_rest = with_rest.filter(pl.col("is_home")).select(
        "fixture_id", pl.col("rest_days").alias("home_rest_days")
    )
    away_rest = with_rest.filter(~pl.col("is_home")).select(
        "fixture_id", pl.col("rest_days").alias("away_rest_days")
    )

    result = (
        fixtures.select("fixture_id")
        .join(home_rest, on="fixture_id", how="left")
        .join(away_rest, on="fixture_id", how="left")
    )
    return RestDaysSchema.validate(result)
