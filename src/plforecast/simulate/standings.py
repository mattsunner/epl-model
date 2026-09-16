"""Points, goal difference and goals scored per club from a frame of played matches.
Shared by the promoted-club prior (ranking historical tables) and the season-level
evaluation (the realised final table), so the two never disagree on arithmetic."""

from __future__ import annotations

import polars as pl

from plforecast.simulate.tiebreak import ClubSeasonResult


def season_standings(matches: pl.DataFrame) -> list[ClubSeasonResult]:
    """`matches`: home_club_id, away_club_id, home_goals, away_goals."""
    home = (
        matches.group_by("home_club_id")
        .agg(
            pl.col("home_goals").sum().alias("gf"),
            pl.col("away_goals").sum().alias("ga"),
            pl.when(pl.col("home_goals") > pl.col("away_goals"))
            .then(3)
            .when(pl.col("home_goals") < pl.col("away_goals"))
            .then(0)
            .otherwise(1)
            .sum()
            .alias("points"),
        )
        .rename({"home_club_id": "club_id"})
    )
    away = (
        matches.group_by("away_club_id")
        .agg(
            pl.col("away_goals").sum().alias("gf"),
            pl.col("home_goals").sum().alias("ga"),
            pl.when(pl.col("away_goals") > pl.col("home_goals"))
            .then(3)
            .when(pl.col("away_goals") < pl.col("home_goals"))
            .then(0)
            .otherwise(1)
            .sum()
            .alias("points"),
        )
        .rename({"away_club_id": "club_id"})
    )
    combined = (
        pl.concat([home, away])
        .group_by("club_id")
        .agg(pl.col("gf").sum(), pl.col("ga").sum(), pl.col("points").sum())
        .with_columns((pl.col("gf") - pl.col("ga")).alias("gd"))
        .sort("club_id")
    )
    return [
        ClubSeasonResult(
            club_id=row["club_id"],
            points=int(row["points"]),
            goal_difference=int(row["gd"]),
            goals_for=int(row["gf"]),
        )
        for row in combined.iter_rows(named=True)
    ]
