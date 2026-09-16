"""Club-season membership bridge (design.md section 5.3): which clubs were in which
competition each season. A separate table rather than a `league` column on the club
dimension, because clubs move between divisions -- a column would be wrong the moment
a club is promoted or relegated, and the promoted-club priors already need Championship
history for clubs that are now Premier League clubs.

v1 only ever populates "Premier League" rows. Membership is derived from appearance:
whichever clubs actually show up in a given season's football-data.co.uk file were, by
construction, in the Premier League that season (E0 *is* the Premier League division).
There is no independent membership list to cross-check against and none is needed.

`competition` and `division_tier` exist as columns from the start anyway -- one of the
three cheap, deliberate structural allowances for multi-league generalisation noted in
design.md section 14, decision 6 -- so a second division later is a new row shape, not
a schema migration.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series

from plforecast.entities.clubs import ClubDimension

PREMIER_LEAGUE = "Premier League"


class ClubSeasonMembershipSchema(pa.DataFrameModel):
    club_id: Series[str]
    season: Series[str] = pa.Field(str_matches=r"^\d{4}/\d{2}$")
    competition: Series[str]
    division_tier: Series[int] = pa.Field(ge=1)

    class Config:
        strict = True
        coerce = True


def _assert_unique(df: pl.DataFrame, columns: list[str]) -> None:
    dupes = df.filter(df.select(columns).is_duplicated())
    if dupes.height:
        raise ValueError(f"club-season membership has duplicate {columns} rows: {dupes}")


def build_club_season_membership(
    matches: pl.DataFrame,
    dimension: ClubDimension,
    *,
    competition: str = PREMIER_LEAGUE,
    division_tier: int = 1,
) -> pl.DataFrame:
    """`matches` needs only `season`, `home_team`, `away_team` columns holding raw
    football-data-shaped club names -- exactly the shape FootballDataSource.parse()
    produces. One row per club per season it actually appeared in."""
    appearances = pl.concat(
        [
            matches.select(pl.col("season"), pl.col("home_team").alias("raw_name")),
            matches.select(pl.col("season"), pl.col("away_team").alias("raw_name")),
        ]
    ).unique()

    club_ids = dimension.resolve(appearances["raw_name"].to_list(), "football_data")
    membership = (
        appearances.with_columns(pl.Series("club_id", club_ids))
        .select("club_id", "season")
        .unique()
        .with_columns(
            pl.lit(competition).alias("competition"),
            pl.lit(division_tier).alias("division_tier"),
        )
        .sort("season", "club_id")
    )
    _assert_unique(membership, ["club_id", "season", "competition"])
    return ClubSeasonMembershipSchema.validate(membership)
