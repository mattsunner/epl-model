"""Vectorised Monte Carlo season simulation (design.md section 7.1). Every remaining
fixture's scoreline is drawn for every simulated season in one vectorised call per
fixture -- not a Python loop over simulated seasons. The one place this module drops
into a per-season loop is resolving the exact points/goal-difference/goals-scored ties
the vectorised primary sort can't finish on its own (design.md section 7.2); those are
rare enough in a real table that the loop only ever runs over a small fraction of
`n_simulations`.

Seeded with an explicit `numpy.random.Generator` (never global numpy random state), so
the same seed reproduces the same simulation bit-for-bit -- design.md section 1.3's
"identical output for identical seed and config hash" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from plforecast.models.base import ClubId, MatchModel
from plforecast.simulate.tiebreak import ClubSeasonResult, TiebreakRules


@dataclass(frozen=True, slots=True)
class SimulationResult:
    club_ids: list[ClubId]
    # position_counts[i, p] = number of simulated seasons where club_ids[i] finished in
    # position p (0-indexed: column 0 is 1st place, column -1 is last place).
    position_counts: np.ndarray
    # points/goal_difference/goals_for: shape (n_simulations, len(club_ids)), full final
    # season stats per simulation -- not just the final rank. The forecast artifact
    # (design.md section 9.3) needs points percentiles alongside position percentiles,
    # not position alone.
    points: np.ndarray
    goal_difference: np.ndarray
    goals_for: np.ndarray
    n_simulations: int

    @property
    def position_pmf(self) -> np.ndarray:
        """Row i: probability distribution over final positions for club_ids[i]. Rows
        sum to 1 across positions; columns also sum to 1 across clubs (design.md
        section 7.3's doubly-stochastic invariant)."""
        return self.position_counts / self.n_simulations


def simulate_season(
    played_matches: pl.DataFrame,
    remaining_fixtures: pl.DataFrame,
    model: MatchModel,
    tiebreak_rules: TiebreakRules,
    *,
    n_simulations: int = 50_000,
    max_goals: int = 10,
    seed: int,
) -> SimulationResult:
    """`played_matches`: home_club_id, away_club_id, home_goals, away_goals -- results
    already known this season, identical across every simulated season.
    `remaining_fixtures`: home_club_id, away_club_id -- not yet played; a scoreline is
    drawn independently for each, for every simulated season, from `model`.
    """
    rng = np.random.default_rng(seed)

    club_ids = sorted(
        set(played_matches["home_club_id"])
        | set(played_matches["away_club_id"])
        | set(remaining_fixtures["home_club_id"])
        | set(remaining_fixtures["away_club_id"])
    )
    club_index = {club: i for i, club in enumerate(club_ids)}
    n_clubs = len(club_ids)

    sim_home_goals, sim_away_goals = _draw_remaining_fixtures(
        remaining_fixtures, model, n_simulations=n_simulations, max_goals=max_goals, rng=rng
    )

    points, goal_diff, goals_for = _accumulate_standings(
        played_matches,
        remaining_fixtures,
        sim_home_goals,
        sim_away_goals,
        club_index,
        n_simulations,
    )

    order = np.lexsort((-goals_for, -goal_diff, -points), axis=-1)
    needs_tiebreak = _detect_exact_ties(points, goal_diff, goals_for, order)

    position_counts = np.zeros((n_clubs, n_clubs), dtype=np.int64)

    fast = ~needs_tiebreak
    for rank_idx in range(n_clubs):
        np.add.at(position_counts, (order[fast, rank_idx], rank_idx), 1)

    if needs_tiebreak.any():
        _resolve_ties(
            np.nonzero(needs_tiebreak)[0],
            played_matches,
            remaining_fixtures,
            sim_home_goals,
            sim_away_goals,
            points,
            goal_diff,
            goals_for,
            club_ids,
            club_index,
            tiebreak_rules,
            position_counts,
            rng,
        )

    return SimulationResult(
        club_ids=club_ids,
        position_counts=position_counts,
        points=points,
        goal_difference=goal_diff,
        goals_for=goals_for,
        n_simulations=n_simulations,
    )


def _draw_remaining_fixtures(
    remaining_fixtures: pl.DataFrame,
    model: MatchModel,
    *,
    n_simulations: int,
    max_goals: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_remaining = remaining_fixtures.height
    sim_home_goals = np.zeros((n_simulations, n_remaining), dtype=np.int64)
    sim_away_goals = np.zeros((n_simulations, n_remaining), dtype=np.int64)

    for f, row in enumerate(remaining_fixtures.iter_rows(named=True)):
        matrix = model.scoreline_matrix(row["home_club_id"], row["away_club_id"], max_goals)
        flat_probs = matrix.ravel()
        draws = rng.choice(flat_probs.size, size=n_simulations, p=flat_probs)
        home_goals, away_goals = np.divmod(draws, matrix.shape[1])
        sim_home_goals[:, f] = home_goals
        sim_away_goals[:, f] = away_goals

    return sim_home_goals, sim_away_goals


def _accumulate_standings(
    played_matches: pl.DataFrame,
    remaining_fixtures: pl.DataFrame,
    sim_home_goals: np.ndarray,
    sim_away_goals: np.ndarray,
    club_index: dict[ClubId, int],
    n_simulations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_clubs = len(club_index)
    points = np.zeros((n_simulations, n_clubs), dtype=np.int64)
    goal_diff = np.zeros((n_simulations, n_clubs), dtype=np.int64)
    goals_for = np.zeros((n_simulations, n_clubs), dtype=np.int64)

    # Already-played matches are identical across every simulation, so each one is a
    # single scalar increment broadcast across the whole simulation axis at once.
    for row in played_matches.iter_rows(named=True):
        h, a = club_index[row["home_club_id"]], club_index[row["away_club_id"]]
        hg, ag = row["home_goals"], row["away_goals"]
        goals_for[:, h] += hg
        goals_for[:, a] += ag
        goal_diff[:, h] += hg - ag
        goal_diff[:, a] += ag - hg
        if hg > ag:
            points[:, h] += 3
        elif hg < ag:
            points[:, a] += 3
        else:
            points[:, h] += 1
            points[:, a] += 1

    for f, row in enumerate(remaining_fixtures.iter_rows(named=True)):
        h, a = club_index[row["home_club_id"]], club_index[row["away_club_id"]]
        hg, ag = sim_home_goals[:, f], sim_away_goals[:, f]
        goals_for[:, h] += hg
        goals_for[:, a] += ag
        goal_diff[:, h] += hg - ag
        goal_diff[:, a] += ag - hg
        home_win = hg > ag
        away_win = hg < ag
        draw = ~home_win & ~away_win
        points[:, h] += np.where(home_win, 3, np.where(draw, 1, 0))
        points[:, a] += np.where(away_win, 3, np.where(draw, 1, 0))

    return points, goal_diff, goals_for


def _detect_exact_ties(
    points: np.ndarray, goal_diff: np.ndarray, goals_for: np.ndarray, order: np.ndarray
) -> np.ndarray:
    sorted_points = np.take_along_axis(points, order, axis=-1)
    sorted_gd = np.take_along_axis(goal_diff, order, axis=-1)
    sorted_gf = np.take_along_axis(goals_for, order, axis=-1)
    exact_tie = (
        (sorted_points[:, :-1] == sorted_points[:, 1:])
        & (sorted_gd[:, :-1] == sorted_gd[:, 1:])
        & (sorted_gf[:, :-1] == sorted_gf[:, 1:])
    )
    return np.asarray(exact_tie.any(axis=1))


def _resolve_ties(
    tied_simulations: np.ndarray,
    played_matches: pl.DataFrame,
    remaining_fixtures: pl.DataFrame,
    sim_home_goals: np.ndarray,
    sim_away_goals: np.ndarray,
    points: np.ndarray,
    goal_diff: np.ndarray,
    goals_for: np.ndarray,
    club_ids: list[ClubId],
    club_index: dict[ClubId, int],
    tiebreak_rules: TiebreakRules,
    position_counts: np.ndarray,
    rng: np.random.Generator,
) -> None:
    played_slim = played_matches.select("home_club_id", "away_club_id", "home_goals", "away_goals")
    remaining_slim = remaining_fixtures.select("home_club_id", "away_club_id")

    for s in tied_simulations:
        standings = [
            ClubSeasonResult(
                club_id=club,
                points=int(points[s, i]),
                goal_difference=int(goal_diff[s, i]),
                goals_for=int(goals_for[s, i]),
            )
            for i, club in enumerate(club_ids)
        ]
        season_matches = pl.concat(
            [
                played_slim,
                remaining_slim.with_columns(
                    pl.Series("home_goals", sim_home_goals[s]),
                    pl.Series("away_goals", sim_away_goals[s]),
                ),
            ]
        )
        ranked = tiebreak_rules.rank(standings, season_matches, rng=rng)
        for rank_idx, club in enumerate(ranked):
            position_counts[club_index[club], rank_idx] += 1
