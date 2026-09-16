# ADR 0008: Multi-league structural allowances

**Status**: accepted 14 September 2026.

## Context

Multi-league generalisation is a v1 non-goal (design.md 1.2). The architecture should
not preclude it, but every allowance made now is paid for in complexity now.

## Decision

Exactly three cheap structural choices, and nothing else:

1. **Tiebreak rules as a strategy object.** `simulate/tiebreak.py` defines a
   `TiebreakRules` protocol with `PremierLeagueTiebreaks` as the only implementation.
   La Liga (head-to-head first) or the Bundesliga (goal difference, goals, then
   head-to-head) would be new classes, not a refactor. The v1 benefit is that the rules
   are independently testable against known cases without standing up a simulation.
2. **Club-season membership as a bridge table**, not a `league` column on the club
   dimension (`entities/competitions.py`, `stg_club_season`). Clubs move between
   divisions, and the promoted-club priors already need Championship records for
   clubs that are now Premier League clubs. `competition` and `division_tier` columns
   exist from the start so a second division is a new row shape, not a migration.
3. **Competition config** (`simulate/competition.py`): league size, relegation spots,
   European spots. The property-based simulation invariants parameterise off it rather
   than hardcoding 20, 380 and 38.

## Consequences

- No abstraction over sources, alias tables or feature builders for other leagues.
- `club_aliases.yaml` is Premier League clubs only; a second league would add rows with
  the same columns.
- The `european_spots` default of 7 is deliberately generous so head-to-head is applied
  wherever European qualification could plausibly be at stake.
