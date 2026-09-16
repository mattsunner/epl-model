"""League size, fixture count, and promotion/relegation config (design.md section 3's
repo layout). A `CompetitionConfig` instance parameterises the property-based invariants
in section 7.3 and the tiebreak rules in section 7.2 -- so a second competition (a
different league size, different European/relegation spot counts) is a new config
instance, not a change to the simulation or tiebreak logic itself. One of the three
cheap structural allowances for multi-league generalisation named in section 14,
decision 6.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompetitionConfig:
    n_clubs: int = 20
    relegation_spots: int = 3
    # Generous default: Champions League (top 4-5 depending on coefficient) plus
    # Europa League and Conference League berths can put as many as 7 clubs into
    # European competition in a given season. Contested-position tiebreak logic
    # (design.md section 7.2) treats this as "European qualification is at stake" for
    # every position within it, so erring generous is the safer default.
    european_spots: int = 7

    @property
    def matches_per_club(self) -> int:
        """2*(n_clubs-1): a full home-and-away round robin."""
        return 2 * (self.n_clubs - 1)

    @property
    def total_matches(self) -> int:
        """n_clubs*(n_clubs-1): one match per ordered pair of clubs."""
        return self.n_clubs * (self.n_clubs - 1)


PREMIER_LEAGUE = CompetitionConfig()
