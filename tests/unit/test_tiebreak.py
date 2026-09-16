import numpy as np
import polars as pl

from plforecast.simulate.competition import CompetitionConfig
from plforecast.simulate.tiebreak import ClubSeasonResult, PremierLeagueTiebreaks

# A small 4-club competition: only position 1 (title) and position 4 (relegation) are
# contested. Positions 2-3 are deliberately left uncontested so tests can check that
# head-to-head is *not* applied there (design.md section 7.2: "applies only where the
# title, European qualification or relegation is at stake").
SMALL = CompetitionConfig(n_clubs=4, relegation_spots=1, european_spots=1)


def _matches(rows: list[tuple[str, str, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["home_club_id", "away_club_id", "home_goals", "away_goals"],
        orient="row",
    )


def test_ranks_by_points_when_no_ties():
    standings = [
        ClubSeasonResult("a", points=10, goal_difference=5, goals_for=15),
        ClubSeasonResult("b", points=15, goal_difference=2, goals_for=10),
        ClubSeasonResult("c", points=5, goal_difference=-3, goals_for=8),
    ]
    ranked = PremierLeagueTiebreaks(SMALL).rank(
        standings, _matches([]), rng=np.random.default_rng(0)
    )
    assert ranked == ["b", "a", "c"]


def test_contested_tie_broken_by_head_to_head_points():
    # a and b tied for the title (position 1) on points/GD/GF; a won both h2h meetings.
    standings = [
        ClubSeasonResult("a", points=20, goal_difference=10, goals_for=20),
        ClubSeasonResult("b", points=20, goal_difference=10, goals_for=20),
        ClubSeasonResult("c", points=5, goal_difference=-5, goals_for=5),
        ClubSeasonResult("d", points=3, goal_difference=-5, goals_for=4),
    ]
    matches = _matches([("a", "b", 2, 0), ("b", "a", 0, 1)])

    ranked = PremierLeagueTiebreaks(SMALL).rank(standings, matches, rng=np.random.default_rng(0))
    assert ranked[0] == "a"
    assert ranked[1] == "b"


def test_contested_tie_falls_through_to_away_goals_when_h2h_points_level():
    # a and b split their two meetings 1-1 on h2h points, but b scored more away.
    standings = [
        ClubSeasonResult("a", points=20, goal_difference=10, goals_for=20),
        ClubSeasonResult("b", points=20, goal_difference=10, goals_for=20),
        ClubSeasonResult("c", points=5, goal_difference=-5, goals_for=5),
        ClubSeasonResult("d", points=3, goal_difference=-5, goals_for=4),
    ]
    matches = _matches([("a", "b", 2, 2), ("b", "a", 1, 1)])  # b scored 2 away, a scored 1 away

    ranked = PremierLeagueTiebreaks(SMALL).rank(standings, matches, rng=np.random.default_rng(0))
    assert ranked[0] == "b"
    assert ranked[1] == "a"


def test_uncontested_tie_is_not_resolved_by_head_to_head():
    # b and c are tied for 2nd/3rd -- not contested in this 4-club config -- even though
    # c actually beat b head-to-head. Head-to-head must NOT be applied here.
    standings = [
        ClubSeasonResult("a", points=30, goal_difference=20, goals_for=30),
        ClubSeasonResult("b", points=15, goal_difference=0, goals_for=15),
        ClubSeasonResult("c", points=15, goal_difference=0, goals_for=15),
        ClubSeasonResult("d", points=3, goal_difference=-5, goals_for=4),
    ]
    matches = _matches([("c", "b", 3, 0)])  # c thrashed b, but it must not matter here

    ranked = PremierLeagueTiebreaks(SMALL).rank(standings, matches, rng=np.random.default_rng(0))
    # Stable sort preserves b before c (their order in `standings`), unaffected by h2h.
    assert ranked[1:3] == ["b", "c"]


def test_relegation_tie_is_contested():
    standings = [
        ClubSeasonResult("a", points=30, goal_difference=20, goals_for=30),
        ClubSeasonResult("b", points=20, goal_difference=5, goals_for=20),
        ClubSeasonResult("c", points=10, goal_difference=-10, goals_for=10),
        ClubSeasonResult("d", points=10, goal_difference=-10, goals_for=10),
    ]
    matches = _matches([("c", "d", 2, 0), ("d", "c", 0, 0)])  # c: 4 h2h points, d: 1

    ranked = PremierLeagueTiebreaks(SMALL).rank(standings, matches, rng=np.random.default_rng(0))
    assert ranked[2] == "c"
    assert ranked[3] == "d"


def test_full_deadlock_is_broken_deterministically_by_seed():
    standings = [
        ClubSeasonResult("a", points=20, goal_difference=10, goals_for=20),
        ClubSeasonResult("b", points=20, goal_difference=10, goals_for=20),
        ClubSeasonResult("c", points=5, goal_difference=-5, goals_for=5),
        ClubSeasonResult("d", points=3, goal_difference=-5, goals_for=4),
    ]
    matches = _matches([("a", "b", 1, 1), ("b", "a", 1, 1)])  # identical in every respect

    first = PremierLeagueTiebreaks(SMALL).rank(standings, matches, rng=np.random.default_rng(42))
    second = PremierLeagueTiebreaks(SMALL).rank(standings, matches, rng=np.random.default_rng(42))
    assert {first[0], first[1]} == {"a", "b"}
    assert first == second  # same seed -> same resolution
