"""ClubElo-to-rate calibration (ADR 0006, story C-16): bridges ClubElo's Elo scale to
the goals-per-match rate fields `features/priors.py.build_prior()`'s `external_rating`
expects (`home_attack`, `home_defence`, `away_attack`, `away_defence`).

A club needing the promoted-club prior has a ClubElo rating (it may have no top-flight
history in this pipeline's own data, but ClubElo covers the Championship) and nothing
else. This module fits four small log-linear regressions, `log(rate) = intercept +
slope * elo`, on *established* clubs where both a `features/strength.py` rate snapshot
and a ClubElo rating exist at the same point in time, then applies that fit to a
promoted club's own current Elo. Same log-space regression shape every other rung
already uses (`models/xg_rates.py`).

Prototyped in `notebooks/03-prototypes/03-04-clubelo-prior-workbench.ipynb` before
being ported here, including a real bug found there: fitting on the 2022/23 season's
own start finds zero usable data points, since ClubElo's cached history does not begin
until partway through that season. `fit_elo_to_rate` returns `{}` for that case, not an
error -- callers fall back to no external rating at all, the same as a club ClubElo
simply doesn't cover.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from plforecast.features.strength import build_club_strength

RATE_FIELDS = ("home_attack_rate", "home_defence_rate", "away_attack_rate", "away_defence_rate")

_EXTERNAL_KEYS = {
    "home_attack_rate": "home_attack",
    "home_defence_rate": "home_defence",
    "away_attack_rate": "away_attack",
    "away_defence_rate": "away_defence",
}


def fit_elo_to_rate(
    matches: pl.DataFrame,
    elo_ratings: pl.DataFrame,
    *,
    as_of: date,
    strength_xi: float = 0.0018,
) -> dict[str, tuple[float, float]]:
    """`(slope, intercept)` per rate field, fit on established clubs with both a
    decayed rate snapshot (`matches`, shaped like `stg_matches`) and a ClubElo rating
    (`elo_ratings`, shaped like `stg_clubelo`: `club_id`, `date`, `elo`) as of `as_of`.
    Refit at every call site rather than cached across dates -- there is no lookahead-
    safe way to reuse a fit from a later date."""
    strength = build_club_strength(matches, as_of=as_of, xi=strength_xi)
    latest_elo = (
        elo_ratings.filter(pl.col("date") <= as_of)
        .sort("date", descending=True)
        .group_by("club_id", maintain_order=True)
        .agg(pl.col("elo").first())
    )
    calib = strength.join(latest_elo, on="club_id", how="inner").drop_nulls(list(RATE_FIELDS))

    coefficients: dict[str, tuple[float, float]] = {}
    if calib.height < 2:
        return coefficients
    elo_values = calib["elo"].to_numpy()
    for field in RATE_FIELDS:
        rates = calib[field].to_numpy()
        valid = rates > 0
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(elo_values[valid], np.log(rates[valid]), 1)
            coefficients[field] = (float(slope), float(intercept))
    return coefficients


def external_rating_for_club(
    club_id: str,
    coefficients: dict[str, tuple[float, float]],
    elo_ratings: pl.DataFrame,
    *,
    as_of: date,
) -> dict[str, float] | None:
    """`club_id`'s ClubElo-derived `external_rating` (the shape `build_prior()`
    expects), or `None` when unavailable: an empty `coefficients` (see
    `fit_elo_to_rate`), or no Elo rating for this club by `as_of` (an unresolved
    `clubelo_name`, or a club ClubElo has not rated yet)."""
    if not coefficients:
        return None
    rows = elo_ratings.filter((pl.col("club_id") == club_id) & (pl.col("date") <= as_of))
    if rows.height == 0:
        return None
    elo = float(rows.sort("date")[-1, "elo"])
    return {
        _EXTERNAL_KEYS[field]: float(np.exp(intercept + slope * elo))
        for field, (slope, intercept) in coefficients.items()
    }
