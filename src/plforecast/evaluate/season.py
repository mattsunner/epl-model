"""Season-level evaluation (design.md section 8.1, story C-07): at frozen points in each
completed season, simulate the rest of the season from what had been played and score
the predicted position distribution against the realised final table.

This is the metric the product actually claims. Match-level RPS says whether the model
prices fixtures well; this says whether the simulated table is honest about where clubs
finish. Remaining fixtures at a cutoff are the unplayed ordered pairs of the round robin,
so no fixture list beyond stg_matches is needed for historical seasons.

Scores per (model, season, cutoff):
- `position_rps`: ranked probability score of each club's position distribution against
  its realised position (positions are ordered, so an off-by-one miss costs less than an
  off-by-ten miss), averaged over clubs.
- `title_log_loss`: negative log probability the model gave the actual champion.
- `top_four_log_loss`, `relegation_log_loss`: mean binary log loss over clubs of the
  realised top-four and relegation events.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
import structlog

from plforecast.models.base import MatchModel, UnknownClubError, drop_odds_columns
from plforecast.simulate.competition import PREMIER_LEAGUE, CompetitionConfig
from plforecast.simulate.engine import simulate_season
from plforecast.simulate.standings import season_standings
from plforecast.simulate.tiebreak import PremierLeagueTiebreaks, TiebreakRules

log = structlog.get_logger()

EPS = 1e-12


def split_at_cutoff(season_matches: pl.DataFrame, cutoff: int) -> pl.DataFrame:
    """The first `cutoff` matches of a season by date, extended to the end of the match
    date the cutoff falls on so a round is never split."""
    ordered = season_matches.sort("date", "match_id")
    if cutoff >= ordered.height:
        return ordered
    cutoff_date = ordered["date"][cutoff - 1]
    return ordered.filter(pl.col("date") <= cutoff_date)


def remaining_pairs(club_ids: Sequence[str], played: pl.DataFrame) -> pl.DataFrame:
    """Every ordered (home, away) pair of the round robin not yet in `played`."""
    done = set(zip(played["home_club_id"].to_list(), played["away_club_id"].to_list(), strict=True))
    rows = [(h, a) for h in club_ids for a in club_ids if h != a and (h, a) not in done]
    return pl.DataFrame(rows, schema=["home_club_id", "away_club_id"], orient="row")


def position_rps(pmf: np.ndarray, realised_index: int) -> float:
    n = len(pmf)
    cum_pmf = np.cumsum(pmf)
    cum_actual = (np.arange(n) >= realised_index).astype(float)
    return float(((cum_pmf - cum_actual) ** 2).sum() / (n - 1))


def _binary_log_loss(p: np.ndarray, actual: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(actual * np.log(p) + (1 - actual) * np.log(1 - p)).mean())


def score_season(
    pmf: np.ndarray,
    club_ids: Sequence[str],
    realised_order: Sequence[str],
    *,
    competition: CompetitionConfig,
) -> dict[str, float]:
    realised_index = {club: pos for pos, club in enumerate(realised_order)}
    positions = np.array([realised_index[c] for c in club_ids])
    n = len(club_ids)

    rps = float(np.mean([position_rps(pmf[i], positions[i]) for i in range(n)]))
    champion = int(np.argmax(positions == 0))
    title_ll = float(-np.log(max(pmf[champion, 0], EPS)))
    top_four_p = pmf[:, :4].sum(axis=1)
    top_four_actual = (positions < 4).astype(float)
    relegation_p = pmf[:, n - competition.relegation_spots :].sum(axis=1)
    relegation_actual = (positions >= n - competition.relegation_spots).astype(float)
    return {
        "position_rps": rps,
        "title_log_loss": title_ll,
        "top_four_log_loss": _binary_log_loss(top_four_p, top_four_actual),
        "relegation_log_loss": _binary_log_loss(relegation_p, relegation_actual),
    }


def evaluate_season_level(
    matches: pl.DataFrame,
    model_factories: Mapping[str, Callable[[], MatchModel]],
    *,
    cutoffs: Sequence[int] = (100, 190, 280),
    n_simulations: int = 5_000,
    seed: int = 0,
    competition: CompetitionConfig = PREMIER_LEAGUE,
    tiebreak_rules: TiebreakRules | None = None,
    max_goals: int = 10,
) -> pl.DataFrame:
    """`matches` shaped like stg_matches, *completed seasons only*. One row per
    (model, season, cutoff); a (season, cutoff) a model cannot price (a club with no
    history at all) is skipped and logged."""
    rules = tiebreak_rules or PremierLeagueTiebreaks(competition)
    seasons = sorted(matches["season"].unique().to_list())
    rows: list[dict[str, Any]] = []
    for season in seasons:
        season_matches = matches.filter(pl.col("season") == season)
        prior = matches.filter(pl.col("season") < season)
        club_ids = sorted(set(season_matches["home_club_id"]) | set(season_matches["away_club_id"]))
        realised = rules.rank(
            season_standings(season_matches), season_matches, rng=np.random.default_rng(seed)
        )
        for cutoff in cutoffs:
            played = split_at_cutoff(season_matches, cutoff)
            remaining = remaining_pairs(club_ids, played)
            train = pl.concat([prior, played], how="vertical_relaxed")
            for name, factory in model_factories.items():
                try:
                    model = factory().fit(drop_odds_columns(train))
                    result = simulate_season(
                        played.select("home_club_id", "away_club_id", "home_goals", "away_goals"),
                        remaining,
                        model,
                        rules,
                        n_simulations=n_simulations,
                        max_goals=max_goals,
                        seed=seed,
                    )
                except UnknownClubError as exc:
                    log.warning(
                        "season_eval.skipped",
                        model=name,
                        season=season,
                        cutoff=cutoff,
                        reason=str(exc),
                    )
                    continue
                scores = score_season(
                    result.position_pmf, result.club_ids, realised, competition=competition
                )
                rows.append(
                    {
                        "model": name,
                        "season": season,
                        "cutoff": cutoff,
                        "played": played.height,
                        **scores,
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "model": pl.Utf8,
            "season": pl.Utf8,
            "cutoff": pl.Int64,
            "played": pl.Int64,
            "position_rps": pl.Float64,
            "title_log_loss": pl.Float64,
            "top_four_log_loss": pl.Float64,
            "relegation_log_loss": pl.Float64,
        },
    )


def summarise_season_level(scores: pl.DataFrame) -> list[dict[str, Any]]:
    """Mean of each score per (model, cutoff), with the season count."""
    if scores.height == 0:
        return []
    summary = (
        scores.group_by("model", "cutoff")
        .agg(
            pl.len().alias("n_seasons"),
            pl.col("position_rps").mean(),
            pl.col("title_log_loss").mean(),
            pl.col("top_four_log_loss").mean(),
            pl.col("relegation_log_loss").mean(),
        )
        .sort("cutoff", "model")
    )
    return summary.to_dicts()
