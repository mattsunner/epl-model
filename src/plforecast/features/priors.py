"""Promoted-club prior construction (design.md section 6.3). For a club with no
usable top-flight history in the backfill window -- a genuinely new promotion, e.g.
Coventry for 2026/27 -- features/strength.py's build_club_strength() has nothing to
compute from at all: the club simply doesn't appear in its output. Confirmed for real
in this project: the season simulation validated in a prior session failed outright the
moment a remaining fixture involved Coventry, and the evaluation harness's backtest
independently hits the same wall for every round-1 fixture of the very first season in
the window (design.md section 8.2's backtest window starts at 2015/16, where by
definition no club has any prior-window history yet). This module is what fills that
gap: a prior distribution over attack/defence strength, anchored on the historical
record rather than assumed from nothing.

A club with SOME recent top-flight history (Ipswich, which played 2024/25) should be
modelled from that history directly via features/strength.py, not pooled into this
no-data prior alongside genuinely blank clubs -- design.md section 6.3 is explicit
about this. `needs_prior()` is the gate: it decides which clubs need a prior at all,
separately from constructing one correctly.

**The anchor, in rate-space, not points-space.** Design.md's own cited anchor -- the
average points total for the club finishing 18th across the 22 Premier League seasons
preceding this project's 2015/16 backfill window is 33.8 -- predates every season this
pipeline has ingested, so it cannot be recomputed from our data, and it is expressed in
points, not the attack/defence goal rates the model layer actually consumes. Rather
than inventing an unverified points-to-rates conversion, this module builds the closest
real, directly comparable anchor from data we actually have: the empirical attack and
defence rate profile of clubs that finished in the "survival zone" (15th-18th, i.e.
competitive enough to stay up but not comfortably) in every completed season in the
window. `build_survival_zone_reference()` also reports the mean points of that same
cohort as a cross-check against design.md's 33.8 -- not required to match exactly,
since it is a different (more recent, differently composed) set of seasons, but a
useful sanity signal that the two anchors are in the same ballpark.

**Real variance, not a point estimate** (design.md section 6.3: "the empirical record
is unstable enough that a tight prior is indefensible" -- all six promoted clubs
relegated in both 2023/24 and 2024/25, then Sunderland finishing 7th on 54 points in
2025/26). The survival-zone reference reports the standard deviation of each rate
across every observed club-season, not just its mean, and `build_prior()` propagates
that width forward rather than collapsing it.

**Blending an external rating** (ClubElo + squad market value, design.md section 6.3)
is structurally supported by `build_prior()`'s `external_rating`/`external_weight`
parameters, but not exercisable with real data yet -- neither source is ingested. Every
club needing a prior today (Coventry) gets the survival-zone anchor alone, unshrunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import cast

import numpy as np
import polars as pl

from plforecast.features.strength import build_club_strength
from plforecast.simulate.competition import PREMIER_LEAGUE, CompetitionConfig
from plforecast.simulate.tiebreak import ClubSeasonResult, PremierLeagueTiebreaks, TiebreakRules

_RATE_FIELDS = ("home_attack", "home_defence", "away_attack", "away_defence")


@dataclass(frozen=True, slots=True)
class SurvivalZoneReference:
    n_observations: int
    mean_points: float  # cross-check against design.md's documented 33.8
    home_attack_mean: float
    home_attack_std: float
    home_defence_mean: float
    home_defence_std: float
    away_attack_mean: float
    away_attack_std: float
    away_defence_mean: float
    away_defence_std: float


@dataclass(frozen=True, slots=True)
class PromotedClubPrior:
    club_id: str
    home_attack_mean: float
    home_attack_std: float
    home_defence_mean: float
    home_defence_std: float
    away_attack_mean: float
    away_attack_std: float
    away_defence_mean: float
    away_defence_std: float


def needs_prior(club_id: str, historical_matches: pl.DataFrame, *, min_matches: int = 38) -> bool:
    """True if `club_id` has fewer than one season's worth of matches (`min_matches`,
    default 38 -- a full round-robin season, design.md section 6.3's own example of
    "enough": Ipswich's 2024/25 season) in `historical_matches` (shaped like
    stg_matches). A club below the threshold should get a prior from this module
    rather than features/strength.py's fallback, which has nothing to compute from at
    all once matches drop to zero, and too little to be stable well before that."""
    appearances = historical_matches.filter(
        (pl.col("home_club_id") == club_id) | (pl.col("away_club_id") == club_id)
    )
    return appearances.height < min_matches


def _club_season_standings(season_matches: pl.DataFrame) -> list[ClubSeasonResult]:
    home = (
        season_matches.group_by("home_club_id")
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
        season_matches.group_by("away_club_id")
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
    )
    return [
        ClubSeasonResult(
            club_id=row["club_id"],
            points=row["points"],
            goal_difference=row["gd"],
            goals_for=row["gf"],
        )
        for row in combined.iter_rows(named=True)
    ]


def build_survival_zone_reference(
    completed_seasons: pl.DataFrame,
    *,
    competition: CompetitionConfig = PREMIER_LEAGUE,
    tiebreak_rules: TiebreakRules | None = None,
    positions: range = range(15, 19),
    seed: int = 0,
) -> SurvivalZoneReference:
    """`completed_seasons` shaped like stg_matches, containing only *finished* seasons
    -- the caller's job to exclude any in-progress season, since this function has no
    way to tell a completed table from a partial one just by looking at match rows.

    For every season in `completed_seasons`, ranks the final table (design.md section
    7.2's real tiebreak rules -- reused as-is, not reimplemented, since final standings
    from real completed matches need exactly the same ordering logic as a simulated
    one) and takes the clubs finishing in `positions` (default 15th-18th: competitive
    enough to survive, not comfortably). Returns the mean and standard deviation of
    each attack/defence rate across every such (club, season) observation.
    """
    tiebreak_rules = tiebreak_rules or PremierLeagueTiebreaks(competition)
    rng = np.random.default_rng(seed)

    rate_rows = []
    points_by_observation = []
    for season in completed_seasons["season"].unique(maintain_order=True).to_list():
        season_matches = completed_seasons.filter(pl.col("season") == season)
        standings = _club_season_standings(season_matches)
        standings_by_id = {s.club_id: s for s in standings}
        ranked = tiebreak_rules.rank(standings, season_matches, rng=rng)

        zone_clubs = [ranked[p - 1] for p in positions if p - 1 < len(ranked)]
        last_date = cast(date, season_matches["date"].max())
        as_of = last_date + timedelta(days=1)
        season_rates = build_club_strength(season_matches, as_of=as_of, xi=0.0)

        for club_id in zone_clubs:
            row = season_rates.filter(pl.col("club_id") == club_id)
            if row.height == 0:
                continue
            rate_rows.append(row.row(0, named=True))
            points_by_observation.append(standings_by_id[club_id].points)

    rates = pl.DataFrame(rate_rows)

    means = {field: cast(float, rates[f"{field}_rate"].mean()) for field in _RATE_FIELDS}
    stds = {field: cast(float, rates[f"{field}_rate"].std()) for field in _RATE_FIELDS}

    return SurvivalZoneReference(
        n_observations=len(rate_rows),
        mean_points=float(np.mean(points_by_observation)),
        home_attack_mean=means["home_attack"],
        home_attack_std=stds["home_attack"],
        home_defence_mean=means["home_defence"],
        home_defence_std=stds["home_defence"],
        away_attack_mean=means["away_attack"],
        away_attack_std=stds["away_attack"],
        away_defence_mean=means["away_defence"],
        away_defence_std=stds["away_defence"],
    )


def build_prior(
    club_id: str,
    reference: SurvivalZoneReference,
    *,
    external_rating: dict[str, float] | None = None,
    external_weight: float = 0.0,
) -> PromotedClubPrior:
    """`external_rating`, when given, provides an already-blended ClubElo + squad
    market-value signal for one or more of the four rate fields (keys from
    `home_attack`, `home_defence`, `away_attack`, `away_defence`) -- the caller's job to
    construct, not this function's, since neither ClubElo nor Transfermarkt is ingested
    yet. `external_weight` in [0, 1] controls how far each provided field's mean shrinks
    toward it; fields absent from `external_rating` are unaffected. Defaults to 0
    (ignore `external_rating` entirely), which is the only path this pipeline can
    exercise for real today: every club currently needing a prior (Coventry) has no
    external rating of any kind, so it gets the survival-zone anchor alone, unshrunk,
    carrying its full width.
    """
    if not 0.0 <= external_weight <= 1.0:
        raise ValueError(f"external_weight must be in [0, 1], got {external_weight}")

    external_rating = external_rating or {}
    means = {}
    stds = {}
    for field in _RATE_FIELDS:
        anchor_mean = getattr(reference, f"{field}_mean")
        anchor_std = getattr(reference, f"{field}_std")
        if field in external_rating:
            means[field] = (1 - external_weight) * anchor_mean + external_weight * external_rating[
                field
            ]
            # Shrinking toward an external signal narrows the prior -- trusting the
            # signal at all means trusting it more than blank ignorance -- but never
            # below half the anchor's width: design.md's objection is to a *tight*
            # prior, and a fully-trusted external rating is still a rating for a club
            # that has never played in this league, not a measurement of it.
            stds[field] = anchor_std * max(1 - external_weight, 0.5)
        else:
            means[field] = anchor_mean
            stds[field] = anchor_std

    return PromotedClubPrior(
        club_id=club_id,
        home_attack_mean=means["home_attack"],
        home_attack_std=stds["home_attack"],
        home_defence_mean=means["home_defence"],
        home_defence_std=stds["home_defence"],
        away_attack_mean=means["away_attack"],
        away_attack_std=stds["away_attack"],
        away_defence_mean=means["away_defence"],
        away_defence_std=stds["away_defence"],
    )
