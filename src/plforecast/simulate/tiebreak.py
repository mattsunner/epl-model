"""Final-table tiebreak resolution (design.md section 7.2). A TiebreakRules strategy
object, not hardcoded logic: a second competition (La Liga: head-to-head first;
Bundesliga: goal difference, then goals, then head-to-head) is a new class rather than
a refactor -- one of the three cheap structural allowances for multi-league
generalisation named in section 14, decision 6.

PremierLeagueTiebreaks implements the exact post-2019/20 ordering:
1. Points
2. Goal difference
3. Goals scored
4. Head-to-head points between the tied clubs, applied only where the tie affects the
   title, European qualification, or relegation
5. Away goals in those head-to-head matches
6. A playoff at a neutral ground

Steps 4 and 5 make a genuine playoff (step 6) far less likely than under the previous
rules. Most public models stop at step 3, which is wrong and cheap to fix -- this
module is that fix. It is independently testable against known historical cases
without standing up a simulation, which is the whole point of keeping it a separate
strategy object rather than inlining the logic into the simulation engine.

A real neutral-ground playoff is not simulated match-by-match: if steps 1-5 leave clubs
still level -- vanishingly rare, since points, goal difference, goals scored,
head-to-head points, and head-to-head away goals would all have to match exactly --
the order between them is settled by drawing on the supplied random generator. That is
an honest representation of "unpredictable, decided on the day" for a case this rare,
not a simulated match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl

from plforecast.simulate.competition import PREMIER_LEAGUE, CompetitionConfig


@dataclass(frozen=True, slots=True)
class ClubSeasonResult:
    club_id: str
    points: int
    goal_difference: int
    goals_for: int


class TiebreakRules(Protocol):
    def rank(
        self,
        standings: list[ClubSeasonResult],
        matches: pl.DataFrame,
        *,
        rng: np.random.Generator,
    ) -> list[str]:
        """Returns club_ids in final league order, first to last. `matches` is every
        match played this season (real, simulated, or both): home_club_id,
        away_club_id, home_goals, away_goals -- needed to resolve head-to-head ties."""
        ...


class PremierLeagueTiebreaks:
    def __init__(self, competition: CompetitionConfig = PREMIER_LEAGUE) -> None:
        self.competition = competition

    def rank(
        self,
        standings: list[ClubSeasonResult],
        matches: pl.DataFrame,
        *,
        rng: np.random.Generator,
    ) -> list[str]:
        ordered = sorted(
            standings, key=lambda r: (r.points, r.goal_difference, r.goals_for), reverse=True
        )
        contested = self._contested_positions()

        result: list[str] = []
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and self._tied(ordered[i], ordered[j + 1]):
                j += 1

            group = ordered[i : j + 1]
            positions = set(range(i + 1, j + 2))  # 1-indexed positions this group occupies
            if len(group) > 1 and positions & contested:
                group = self._break_tie(group, matches, rng=rng)
            result.extend(club.club_id for club in group)
            i = j + 1

        return result

    def _contested_positions(self) -> set[int]:
        c = self.competition
        return set(range(1, c.european_spots + 1)) | set(
            range(c.n_clubs - c.relegation_spots + 1, c.n_clubs + 1)
        )

    @staticmethod
    def _tied(a: ClubSeasonResult, b: ClubSeasonResult) -> bool:
        return (
            a.points == b.points
            and a.goal_difference == b.goal_difference
            and a.goals_for == b.goals_for
        )

    @staticmethod
    def _break_tie(
        group: list[ClubSeasonResult], matches: pl.DataFrame, *, rng: np.random.Generator
    ) -> list[ClubSeasonResult]:
        club_ids = {c.club_id for c in group}
        among_group = matches.filter(
            pl.col("home_club_id").is_in(club_ids) & pl.col("away_club_id").is_in(club_ids)
        )

        h2h_points = dict.fromkeys(club_ids, 0)
        h2h_away_goals = dict.fromkeys(club_ids, 0)
        for row in among_group.iter_rows(named=True):
            home, away = row["home_club_id"], row["away_club_id"]
            home_goals, away_goals = row["home_goals"], row["away_goals"]
            if home_goals > away_goals:
                h2h_points[home] += 3
            elif home_goals < away_goals:
                h2h_points[away] += 3
            else:
                h2h_points[home] += 1
                h2h_points[away] += 1
            h2h_away_goals[away] += away_goals

        # The playoff stand-in: a random-but-deterministic-given-the-seed key so ties
        # surviving every real criterion don't just fall back to input order.
        coin = {c.club_id: rng.random() for c in group}

        return sorted(
            group,
            key=lambda c: (h2h_points[c.club_id], h2h_away_goals[c.club_id], coin[c.club_id]),
            reverse=True,
        )
