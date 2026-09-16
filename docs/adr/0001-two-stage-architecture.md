# ADR 0001: Two-stage architecture (match model, then simulation)

**Status**: accepted 14 September 2026.

## Context

The product is a probability distribution over each club's final league position. The
obvious shortcut is to regress directly on final points or position from club-level
features.

## Options considered

1. **Direct regression on final points or position.** Roughly 20 observations per
   season; discards the fixture graph; gives no joint distribution across clubs.
2. **Match model plus season simulation.** A model emits a scoreline distribution per
   remaining fixture; a Monte Carlo engine plays out the season under the competition's
   tiebreak rules and tabulates final positions.

## Decision

Option 2. Final position is a rank determined jointly across 20 clubs by a fixed fixture
graph. Simulation preserves that structure and yields honest joint uncertainty,
including correlations between clubs that share remaining opponents.

The match model's single primitive is `scoreline_matrix(home, away)`, an
`(max_goals+1, max_goals+1)` joint distribution (design.md 6.1). Everything downstream,
including 1X2 probabilities for evaluation and sampling for simulation, derives from it,
so every model, and the market benchmark, is a one-line substitution.

## Consequences

- Match-level evaluation (RPS on 1X2) and season-level evaluation (position
  distribution vs realised table) are both possible and both needed.
- Improving the forecast means improving the match model or the inputs to it; the
  simulation engine is stable infrastructure.
- Posterior uncertainty in team strength (the hierarchical model, design.md 6.4) enters
  by drawing strengths per simulated season, not by changing the engine's contract.
