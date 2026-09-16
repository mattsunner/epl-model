import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import poisson

from plforecast.simulate.competition import CompetitionConfig
from plforecast.simulate.engine import simulate_season
from plforecast.simulate.tiebreak import PremierLeagueTiebreaks


class _FixedPoissonModel:
    """Deterministic MatchModel stand-in for testing the simulation engine in
    isolation from any real fitted model -- home/away scoring rates are fixed
    constants, not derived from data."""

    def __init__(self, home_lambda: float = 1.4, away_lambda: float = 1.1) -> None:
        self.home_lambda = home_lambda
        self.away_lambda = away_lambda

    def scoreline_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        goals = np.arange(max_goals + 1)
        home_pmf = poisson.pmf(goals, self.home_lambda)
        away_pmf = poisson.pmf(goals, self.away_lambda)
        matrix = np.outer(home_pmf, away_pmf)
        return matrix / matrix.sum()

    @property
    def config_hash(self) -> str:
        return "fixed-poisson-test-double"


def _round_robin(club_ids: list[str]) -> list[tuple[str, str]]:
    return [(h, a) for h in club_ids for a in club_ids if h != a]


def _matches_frame(rows: list[tuple], columns: list[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 if "club_id" in c else pl.Int64 for c in columns})
    return pl.DataFrame(rows, schema=columns, orient="row")


@given(
    n_clubs=st.integers(min_value=3, max_value=6),
    played_fraction=st.floats(min_value=0.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=15, deadline=None)
def test_simulation_invariants(n_clubs: int, played_fraction: float, seed: int) -> None:
    """design.md section 7.3's invariants, parameterised off CompetitionConfig rather
    than hardcoded league constants, so they stay honest if the competition config
    ever changes."""
    club_ids = [f"club-{i}" for i in range(n_clubs)]
    fixtures = _round_robin(club_ids)
    n_played = int(len(fixtures) * played_fraction)

    score_rng = np.random.default_rng(seed)
    played_rows = [
        (h, a, int(score_rng.integers(0, 4)), int(score_rng.integers(0, 4)))
        for h, a in fixtures[:n_played]
    ]
    remaining_rows = [(h, a) for h, a in fixtures[n_played:]]

    played_matches = _matches_frame(
        played_rows, ["home_club_id", "away_club_id", "home_goals", "away_goals"]
    )
    remaining_fixtures = _matches_frame(remaining_rows, ["home_club_id", "away_club_id"])

    competition = CompetitionConfig(n_clubs=n_clubs, relegation_spots=1, european_spots=1)
    n_simulations = 300
    result = simulate_season(
        played_matches,
        remaining_fixtures,
        _FixedPoissonModel(),
        PremierLeagueTiebreaks(competition),
        n_simulations=n_simulations,
        max_goals=8,
        seed=seed,
    )

    assert len(result.club_ids) == n_clubs
    assert competition.total_matches == n_clubs * (n_clubs - 1)
    assert competition.matches_per_club == 2 * (n_clubs - 1)

    # Doubly-stochastic position matrix: each club's positions sum to n_simulations
    # across positions, and each position's clubs sum to n_simulations across clubs.
    assert (result.position_counts.sum(axis=1) == n_simulations).all()
    assert (result.position_counts.sum(axis=0) == n_simulations).all()

    # League-wide goal difference sums to zero, every simulation.
    assert (result.goal_difference.sum(axis=1) == 0).all()

    # Total goals scored across the league equals total goals conceded, every simulation.
    goals_against = result.goals_for - result.goal_difference
    assert np.array_equal(result.goals_for.sum(axis=1), goals_against.sum(axis=1))

    # Total points awarded is between 2*n_matches and 3*n_matches inclusive.
    total_matches = competition.total_matches
    total_points = result.points.sum(axis=1)
    assert (total_points >= 2 * total_matches).all()
    assert (total_points <= 3 * total_matches).all()


def test_simulate_season_is_deterministic_given_a_seed() -> None:
    club_ids = ["a", "b", "c", "d"]
    fixtures = _round_robin(club_ids)
    played_matches = _matches_frame(
        [(h, a, 1, 1) for h, a in fixtures[:6]],
        ["home_club_id", "away_club_id", "home_goals", "away_goals"],
    )
    remaining_fixtures = _matches_frame(fixtures[6:], ["home_club_id", "away_club_id"])

    competition = CompetitionConfig(n_clubs=4, relegation_spots=1, european_spots=1)
    kwargs = dict(
        played_matches=played_matches,
        remaining_fixtures=remaining_fixtures,
        model=_FixedPoissonModel(),
        tiebreak_rules=PremierLeagueTiebreaks(competition),
        n_simulations=200,
        max_goals=6,
        seed=123,
    )

    first = simulate_season(**kwargs)
    second = simulate_season(**kwargs)

    assert np.array_equal(first.position_counts, second.position_counts)
    assert np.array_equal(first.points, second.points)


def test_no_remaining_fixtures_gives_certain_final_table() -> None:
    """With nothing left to play, every simulation must produce the exact same table --
    there is no uncertainty left to simulate."""
    club_ids = ["a", "b", "c", "d"]
    fixtures = _round_robin(club_ids)
    rng = np.random.default_rng(0)
    played_matches = _matches_frame(
        [(h, a, int(rng.integers(0, 4)), int(rng.integers(0, 4))) for h, a in fixtures],
        ["home_club_id", "away_club_id", "home_goals", "away_goals"],
    )
    remaining_fixtures = _matches_frame([], ["home_club_id", "away_club_id"])

    competition = CompetitionConfig(n_clubs=4, relegation_spots=1, european_spots=1)
    result = simulate_season(
        played_matches,
        remaining_fixtures,
        _FixedPoissonModel(),
        PremierLeagueTiebreaks(competition),
        n_simulations=500,
        seed=7,
    )

    # Every simulation agrees on who finished where: one position per club has all the
    # probability mass.
    assert ((result.position_counts == 0) | (result.position_counts == 500)).all()
