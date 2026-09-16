# ADR 0006: Promoted-club priors

**Status**: accepted 14 September 2026; implementation path revised 16 September 2026.

## Context

A club promoted with no top-flight history in the backfill window has no data for the
match model. The empirical record for promoted clubs is unstable: all six promoted clubs
were relegated in each of 2023/24 and 2024/25, which had not happened since 1997/98;
then 2025/26 broke it, with Sunderland finishing 7th on 54 points, Leeds 14th and only
Burnley going down. A prior fit on the two preceding seasons would have given
Sunderland near-certain relegation.

## Options considered

1. **Championship xG.** Understat does not cover the Championship, and a league-strength
   conversion estimated from goals would rest on roughly three clubs per season.
2. **ClubElo rating at season start blended with squad market value**, shrunk toward a
   recency-weighted promoted-club mean. ClubElo already performs continuous
   cross-league conversion and covers lower divisions.
3. **Empirical survival-zone anchor.** The attack and defence rate profile of clubs
   finishing 15th to 18th in every completed season in the window, with its full
   observed variance.

## Decision

- The prior **must carry real variance**, never a point estimate. The prior mean is
  anchored on the long record, not the last two seasons.
- Option 3 is what is built today (`features/priors.py`): the survival-zone reference,
  reporting mean points alongside as a cross-check against the 22-season figure of 33.8
  points for 18th place. Option 2 is structurally supported (`external_rating`,
  `external_weight`) and becomes live when ClubElo is ingested (story C-16).
- **Promoted clubs are not interchangeable.** A club with a recent top-flight season
  (Ipswich, 2024/25) is modelled from that history; `needs_prior` is the gate.
- **Delivery into the model layer**: as weighted pseudo-observations appended to the
  fit frame, which penaltyblog's `weights` already supports, rather than waiting for the
  hierarchical model (story C-08). The number and weight of pseudo-matches is derived
  from the prior's width.

## Consequences

- Until C-08 lands, a promoted club is rated only from whatever matches it has played,
  and its projection is the model's failure mode. The README says so and the first
  committed forecast shows it (Coventry, 4.8 expected points after four matches).
- The recency half-life on the prior mean (design.md 15, question 2) needs the
  season-level evaluation harness (story C-07) to be tuned honestly.
