# Model card

Intended use, limitations and known failure modes for the pl-forecast pipeline, per
design.md section 11.4 and story C-12. Written for a reader deciding whether, and how
much, to trust a published forecast -- not for someone extending the code, which
`docs/methodology.md` and the source itself are for. Figures below are current as of
16 September 2026 (`docs/evaluation.md`'s own generation timestamp); re-check
`docs/evaluation.md` for the live numbers, since this document is not regenerated
automatically.

## Intended use

A calibrated probability distribution over each Premier League club's final table
position for the current season, refreshed after each gameweek, published alongside an
honest comparison against the closing betting market. It is meant to be read the way a
weather forecast is read: "Arsenal has about a 50% chance of the title" is a claim about
the model's uncertainty given what it has seen, not a prediction that Arsenal will or
will not win.

**What it is not.** Not betting advice, not a staking recommendation, and not built to
find an edge against the market -- design.md's non-goals section rules out betting
execution entirely; the market is a benchmark to be honest about losing to, not a
target to beat for profit. Not a player-level tool: no fantasy football optimisation,
no injury-adjusted lineup projection (player availability data is landed but
unconsumed, see `docs/data-sources.md`). Not a live, in-play system: every forecast is
pre-match, generated from results known up to the moment it was run.

## Current performance

From `docs/evaluation.md`'s walk-forward backtest, 2015/16 through 2025/26, 4,066
matches scored on identical rows for every model and the market:

| Model | Mean RPS |
| --- | --- |
| `PoissonModel` (floor) | 0.2063 |
| `DixonColesModel` | 0.2010 |
| `XGRateModel` (shipped) | 0.1980 |
| Market, Shin de-vigged | 0.1939 |

The shipped model beats the Poisson floor and Dixon-Coles, satisfying the model
ladder's own gate (each rung must beat the one before it). **It does not beat the
market.** The gap (0.0041 RPS) is inside design.md's stretch target of 0.005 but is a
real, acknowledged gap, not a rounding difference. This was anticipated, not a
surprise: Pinnacle (the primary benchmark price source) is a sharp, liquid market, and
the shipped model has no posterior uncertainty over team strength (that is what the
planned hierarchical model, rung 4, is for) and only recently gained a promoted-club
prior.

**Calibration** is good but not perfect. Pooled across all three outcomes in 10
probability buckets, most buckets' mean predicted probability falls inside the
empirical frequency's 95% confidence interval. Two buckets do not: the model is
somewhat under-confident in the 0-10% range (predicts about 6.6%, happens about 9% of
the time) and somewhat over-confident in the 60-70% range. `docs/evaluation.md` carries
the full table, including the per-outcome breakdown (pooled calibration can hide a
class-specific miss, and Poisson-family models are known to be prone to mispricing
draws specifically).

## Known limitations

- **Home advantage is not yet time-varying in the literal sense design.md originally
  asked for.** Every shipped model except the Poisson floor approximates it via
  exponential time decay on the whole fit (recent matches, including recent home
  performances, count more), rather than fitting a separate coefficient per season.
  An explicit per-season term is deferred to the hierarchical model.
- **European competition congestion is not modelled.** Rest-day data is captured
  (`features/schedule.py`) because it is nearly free to compute from FPL kickoff times,
  but no feature uses it yet: a naive European-participation indicator would mostly
  re-encode team strength (the clubs in Europe are also the strongest ones), and the
  genuine residual -- a fatigue effect from a shorter rest gap -- is expected to be
  small. It will be added as a single term once the hierarchical baseline exists, and
  kept only if it improves held-out RPS on its own.
- **No posterior uncertainty over team strength.** Every shipped model produces a point
  estimate of each club's attack and defence rates; the simulation draws scorelines
  from that point estimate, not from a distribution over plausible ratings. This
  systematically understates tail probabilities (very good or very bad seasons) -- the
  documented reason the hierarchical model (rung 4) is the next planned step, not an
  optional extra.
- **PPDA (passes per defensive action) is landed but not used**, and would need
  winsorising before any feature used it -- the raw column has extreme values (an
  observed maximum of 193) that a small-sample-size match can produce.
- **The promoted-club prior is not part of the backtest.** It is wired into live
  forecasting (`plforecast forecast`) but `docs/evaluation.md`'s walk-forward numbers do
  not exercise it, so the backtest's own RPS figures do not reflect whatever
  improvement (or regression) the prior contributes for a promoted club's early
  matches. This is a real gap in the evaluation, not a claim that the prior helps.

## Known failure mode: promoted clubs, and how it was found

A club with little or no top-flight history in the backfill window is the model's
hardest case by construction: there is no learned attack/defence signal for it at all
until `features/priors.py`'s survival-zone prior is wired in (`docs/methodology.md`
section 4). Two real, concrete failures during this project's own build illustrate the
shape of the risk:

1. **Before the prior existed at all** (early forecast runs), a genuinely blank
   promoted club (Coventry, 2026/27) was rated from whatever handful of matches it had
   played that season alone -- four matches, all defeats, giving an implausibly low
   expected points total. A four-match sample deciding a 38-match season's forecast is
   exactly the failure mode the prior exists to prevent.
2. **After the prior was added, the sufficiency gate itself had a bug**: it initially
   reused a fitted model's own time-decay rate to decide which clubs needed the prior.
   That decay rate is tuned for smoothing match-level rate estimation and is
   aggressive enough that *every* established club's effective evidence saturates near
   the gate's threshold early in a season, regardless of how much real history it has.
   The live forecast briefly injected prior pseudo-matches into all 20 current clubs,
   not just the genuinely data-poor ones, diluting every club's rating toward a
   generic mid-table profile. Fixed with a decay rate dedicated to the gate itself,
   verified against real club histories before being shipped.

The general lesson, stated plainly rather than left implicit: **any club-specific
adjustment mechanism in this pipeline needs to be checked against real data for every
club it could plausibly apply to, not just the club it was designed for.** A gate or
correction that behaves correctly for the one motivating example (a genuinely blank
promoted club) can still behave wrong for every other club sharing the same code path.

The current promoted-club prior mitigates known failure 1. Its width (real variance,
not a point estimate) is a deliberate response to how unstable the empirical record
for promoted clubs actually is: all six promoted clubs were relegated in both 2023/24
and 2024/25, which had not happened since 1997/98, and then Sunderland finished 7th on
54 points in 2025/26, the joint-best finish by a promoted side since 2018/19. A
forecast for a promoted club should be read with correspondingly wider uncertainty in
mind even with the prior in place.

## What the model does not account for

- Transfers, loans, or squad changes within a season (no roster-change feature exists).
- Managerial changes (a well-documented driver of short-term form swings in football
  forecasting generally; not modelled here).
- Player-level injuries or suspensions, despite the data being landed
  (`raw_fpl_player_availability`) -- see "Known limitations" above.
- Match importance or motivation effects (a dead rubber late in a season, a cup
  distraction) -- every match is treated as equally meaningful to both sides.
- Weather, pitch conditions, or any factor not implicit in a club's decayed historical
  scoring rates.

## Reproducibility

Simulation is exactly deterministic given a seed (tested directly, not just asserted).
Model fitting uses scipy/numpy optimisers whose results can differ at floating-point
precision across BLAS builds and platforms; the reproducibility claim is therefore
"identical on the same platform and lock file," not literally bit-for-bit on any
machine. Every forecast artifact records its git SHA (and whether the working tree was
clean), the model's config hash, the random seed, every raw data snapshot's content
hash, and the installed versions of every numerically-relevant package
(`artifacts/schema.py`'s `Provenance` block) -- enough to know exactly what produced a
given forecast, even if re-running it on different hardware would not reproduce it to
the last bit.
