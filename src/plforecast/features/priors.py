"""Promoted-club prior construction (design.md section 6.3, ADR 0006).

A club with no usable top-flight history in the backfill window (Coventry for 2026/27)
has nothing for the match model to fit. This module builds a prior over its attack and
defence rates from the empirical record of "survival zone" clubs (15th to 18th in every
completed season in the window), with the observed variance across those club-seasons
rather than a point estimate, and delivers it to the model layer as pseudo-observations
(`prior_pseudo_matches`, story C-08): synthetic matches against the real clubs in the
fixture list, as many as the prior's width is worth in evidence.

A club with SOME recent top-flight history (Ipswich, 2024/25) is modelled from that
history; `needs_prior()` is the gate.

The anchor is in rate space, not points space. Design.md's cited figure (33.8 points
for 18th place over 22 seasons) predates the ingested window and is in points; the
survival-zone reference reports its own mean points as a cross-check.

Blending an external rating (ClubElo, squad value) is supported by `build_prior()`'s
`external_rating`/`external_weight` but has no data source yet (story C-16).

**Layer boundary note (story A-20).** design.md section 2.2 draws `features` upstream
of `simulate`; this module imports `simulate.tiebreak` and `simulate.competition`
anyway, to rank historical final tables when building the survival-zone reference
(15th-18th needs the *real* final ordering, ties included, not an approximation). The
standings arithmetic itself (points, goal difference, goals for) was factored out to
`simulate.standings`, shared with `evaluate.season`, so this module no longer
duplicates it; the tiebreak-rules dependency is the one piece that cannot be factored
out the same way, since ranking IS what `TiebreakRules` is for. Accepted as a
documented exception rather than inverting the dependency, since a features-layer
"final table ranker" that isn't the actual competition's own tiebreak rules would be a
second, divergent implementation of design.md section 7.2's logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import cast

import numpy as np
import polars as pl

from plforecast.features.strength import build_club_strength
from plforecast.simulate.competition import PREMIER_LEAGUE, CompetitionConfig
from plforecast.simulate.standings import season_standings
from plforecast.simulate.tiebreak import PremierLeagueTiebreaks, TiebreakRules

_RATE_FIELDS = ("home_attack", "home_defence", "away_attack", "away_defence")

# Decay rate for needs_prior's evidence gate -- deliberately much gentler than any
# match model's own fitting xi (0.0018-0.005/day). A fitting xi is tuned to smooth
# *rate estimation* and is aggressive enough that even a club with hundreds of matches
# spread across the whole backfill window has an effective count near a single
# season's worth once decayed -- using it as the sufficiency threshold's own decay rate
# was tried and flagged essentially every established club as prior-needing the moment
# a season is only a few gameweeks old, which is wrong (verified against real data: at
# xi=0.005, Arsenal's 422 historical matches decay to an effective ~17, under the
# default 19-match threshold). PRIOR_GATE_XI's ~6-year half-life instead separates a
# stale one-off season (Hull's 2016/17: effective ~17 today) from a recent one
# (Ipswich's 2024/25: effective ~36) while leaving every continuously-active club's
# effective count in the hundreds.
PRIOR_GATE_XI = 0.0003


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


def needs_prior(
    club_id: str,
    historical_matches: pl.DataFrame,
    *,
    min_matches: float = 19,
    as_of: date | None = None,
    xi: float = 0.0,
) -> bool:
    """True if `club_id` has less than `min_matches` worth of evidence in
    `historical_matches` (shaped like stg_matches).

    Evidence is counted the way the models weight it: with `xi` and `as_of` given, each
    match counts `exp(-xi * days_since)`, so a promoted club whose only top-flight
    season was years ago (Hull, 2016/17) is recognised as nearly data-free, and a club
    with one recent season (Ipswich, 2024/25) is shrunk toward the prior in proportion
    to how much its evidence has decayed rather than pooled with blank clubs. With
    `xi = 0` every match counts once. The default threshold is half a season."""
    appearances = historical_matches.filter(
        (pl.col("home_club_id") == club_id) | (pl.col("away_club_id") == club_id)
    )
    if xi <= 0 or as_of is None:
        return appearances.height < min_matches
    weights = (
        appearances.select(
            ((pl.lit(as_of) - pl.col("date")).dt.total_days().cast(pl.Float64) * -xi).exp()
        )
        .to_series()
        .sum()
    )
    return float(weights) < min_matches


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
        standings = season_standings(season_matches)
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


def effective_sample_size(
    prior: PromotedClubPrior, *, floor: float = 4.0, cap: float = 38.0
) -> float:
    """How many matches of evidence the prior is worth. For a Poisson rate with mean m
    observed over n matches the standard error is sqrt(m / n); setting that equal to the
    prior's spread s gives n = m / s^2. Averaged over the four rate fields and clamped so
    a degenerate reference can neither swamp real results nor vanish."""
    values = []
    for field in _RATE_FIELDS:
        mean = getattr(prior, f"{field}_mean")
        std = getattr(prior, f"{field}_std")
        if std and std > 0 and mean > 0:
            values.append(mean / std**2)
    if not values:
        return floor
    return float(min(max(sum(values) / len(values), floor), cap))


def prior_pseudo_matches(
    prior: PromotedClubPrior,
    *,
    opponents: Sequence[str],
    season: str,
    as_of: date,
    seed: int = 0,
) -> pl.DataFrame:
    """Synthetic stg_matches-shaped rows expressing `prior` as evidence, spread over
    real `opponents` (alternating home and away, opponents cycled in seeded order).

    Real opponents matter: a phantom opponent played only by this club is not
    identifiable from it, and a joint fit can put every pseudo-match into the phantom's
    parameters while leaving the club's untouched. Real clubs' ratings are pinned by
    their own results, so the club's fitted attack and defence land at the prior.

    The club scores at its prior attack rate and concedes at its prior defence rate
    regardless of opponent, which is what "prior relative to an average opponent"
    means. xG is the prior mean exactly (no noise); goals are seeded Poisson draws from
    it, since the goals-based models need integers. Row count is
    `effective_sample_size(prior)` rounded, so a wide prior contributes little; with 19
    opponents that is under one pseudo-match each, so no opponent's own rating moves.
    """
    others = [club for club in opponents if club != prior.club_id]
    if not others:
        raise ValueError("prior_pseudo_matches needs at least one real opponent")
    rng = np.random.default_rng(seed)
    order = [others[i] for i in rng.permutation(len(others))]
    n = round(effective_sample_size(prior))
    rows = []
    for k in range(n):
        opponent = order[k % len(order)]
        at_home = k % 2 == 0
        if at_home:
            home_id, away_id = prior.club_id, opponent
            home_xg, away_xg = prior.home_attack_mean, prior.home_defence_mean
        else:
            home_id, away_id = opponent, prior.club_id
            home_xg, away_xg = prior.away_defence_mean, prior.away_attack_mean
        home_goals = int(rng.poisson(home_xg))
        away_goals = int(rng.poisson(away_xg))
        rows.append(
            {
                "match_id": f"prior-{prior.club_id}-{k:02d}",
                "season": season,
                "date": as_of,
                "home_club_id": home_id,
                "away_club_id": away_id,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": float(home_xg),
                "away_xg": float(away_xg),
                "result": "H"
                if home_goals > away_goals
                else "A"
                if home_goals < away_goals
                else "D",
            }
        )
    return pl.DataFrame(rows)


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
