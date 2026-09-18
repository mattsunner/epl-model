# ADR 0006: Promoted-club priors

**Status**: accepted 14 September 2026; implementation path revised 16 September 2026;
option 2 (ClubElo) built 18 September 2026 (`ingest/clubelo.py`, `features/clubelo.py`)
but shipped inactive -- `config.clubelo_prior_weight` defaults to `0`. The one
evaluation run behind it (`notebooks/03-prototypes/03-04-clubelo-prior-workbench
.ipynb`) was inconclusive on 18 historical debut-era matches (RPS differences across a
full weight sweep stayed within noise for that sample size), even though the live
forecast shifted materially for at least one promoted club. Revisit the default once
there is more evidence than that.

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
- Option 3 (`features/priors.py`) is the anchor every prior uses: the survival-zone
  reference, reporting mean points alongside as a cross-check against the 22-season
  figure of 33.8 points for 18th place. Option 2 (ClubElo, `features/clubelo.py`,
  `ingest/clubelo.py`) is built and wired through `build_prior()`'s `external_rating`/
  `external_weight` (story C-16), but `config.clubelo_prior_weight` defaults to `0` --
  see the status line above for why.
- **Promoted clubs are not interchangeable.** `needs_prior` is the gate, and it counts
  evidence the way the models weight it: each past match counts `exp(-xi * days)`. A
  club with a recent top-flight season (Ipswich, 2024/25) keeps its data and is shrunk
  toward the prior only in proportion to how much that data has decayed; a club whose
  only top-flight season was years ago (Hull, 2016/17) is treated as nearly data-free,
  which it is under decay. The threshold is half a season of effective matches.
  **The gate's decay rate is its own constant (`PRIOR_GATE_XI`), never a fitted
  model's own `xi`.** Tried the obvious version first -- pass the shipped model's
  fitting decay straight through -- and it flagged every established club, not just
  the genuinely promoted ones: a fitting `xi` (0.0018-0.005/day) is tuned for
  match-level rate smoothing, aggressive enough that even a club with hundreds of
  matches across the whole backfill window has an effective count near a single
  season's worth once decayed, so early in any season the gate caught Arsenal and
  Chelsea alongside Coventry. `PRIOR_GATE_XI` (~6-year half-life) instead separates a
  stale one-off season from a fresh one while leaving every continuously-active club
  comfortably above the threshold.
- **Delivery into the model layer**: as pseudo-observations appended to the fit frame
  (story C-08), rather than waiting for the hierarchical model. The count is the
  prior's effective sample size, `mean / std^2` averaged over the four rate fields and
  clamped to 4 to 38 matches. Pseudo-matches are played against the *real* clubs in the
  fixture list, not a phantom opponent: a phantom played only by the promoted club is
  not identifiable from it, and in practice the fit put the whole prior into the
  phantom's parameters and left the club's untouched.

## Consequences

- Before C-08 landed, a promoted club was rated only from whatever matches it had
  played: the first committed forecast gave Coventry 4.8 expected points after four
  defeats. With the prior it is a survival-zone team until its results say otherwise.
- The recency half-life on the prior mean (design.md 15, question 2) needs the
  season-level evaluation harness (story C-07) to be tuned honestly.
