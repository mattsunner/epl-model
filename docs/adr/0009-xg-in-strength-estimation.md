# ADR 0009: How xG enters strength estimation

**Status**: accepted 16 September 2026.

## Context

design.md section 6.3 says xG rather than goals should drive strength estimation, because
finishing variance is largely noise over a 38-game horizon. `features/strength.py`
described the switch as "a data-source change". It is not: the two shipped models
(`PoissonModel`, `DixonColesModel`) delegate fitting to penaltyblog, whose likelihood is a
Poisson on integer goal counts. xG is continuous. Something in the model layer has to
change, and the choice affects the evaluation harness and the simulation engine.

Understat team-level xG is now landed and joined onto every match in the window
(`stg_matches.home_xg` and friends, story B-05), so the input exists.

## Options considered

1. **Fit on goals; use xG only in features and priors.** No model change. Contradicts
   the design's stated intent and leaves the main modelling lever unpulled.
2. **Fit attack, defence and home advantage on xG by weighted least squares in log
   space**, then produce the scoreline distribution from independent Poissons at the
   fitted rates (optionally with a Dixon-Coles low-score correction whose `rho` is
   taken from the goals fit). The model is a `MatchModel` like any other, so the
   walk-forward harness, the identical-rows comparison and the simulation engine all
   apply unchanged, and the ladder's RPS gate decides whether it ships.
3. **Defer to the hierarchical model with an xG likelihood** (a Gamma or log-normal
   observation model on xG, Poisson on goals for scoring). The right long-run home for
   xG, but it lands with rung 3 and nothing is learned about xG's value until then.

## Decision

Option 2, as rung 2.5 of the ladder (`models/xg_rates.py`, `XGRateModel`), with the
same gate as every other rung: it ships only if it beats Dixon-Coles on held-out RPS on
identical rows. Blending goals and xG (a convex combination of the two per-match
targets) is a one-parameter extension of the same model and is tuned by the same
backtest if the pure-xG version does not beat goals.

Option 3 remains the plan for the hierarchical model; the xG rates fitted here are its
natural starting values.

## Consequences

- `features/strength.py` gains a `metric` switch (goals or xG) for the decayed-rate
  features and the prior's survival-zone anchor; that part *is* a data change.
- The evaluation report and the rendered `docs/evaluation.md` list the xG model next
  to the others; the shipped model is whichever non-market model has the best primary
  RPS, computed, not asserted.
- If xG does not beat goals on this data, that result is published with the rest.
