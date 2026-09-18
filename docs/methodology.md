# Methodology

The maths behind the forecast, written to be readable without reading the code. See
`docs/design.md` for architecture and rationale, `docs/data-sources.md` for the inputs,
`docs/evaluation.md` for current results, and `docs/model-card.md` for what the model
does and does not account for. Per story C-12, this document (and the model card) is
where the *why* behind a modelling choice lives; module docstrings in the code carry
only the contract and any gotcha specific to that function.

## 1. The two-stage design

The forecast never regresses on a club's final points or position directly. Final
position is a rank determined jointly across 20 clubs by a fixed fixture graph, and a
season gives only about 20 final-position observations to learn from. Instead:

1. A **match model** produces a full joint scoreline distribution for any fixture:
   `P(home scores h, away scores a)` for every `(h, a)` pair up to some cap.
2. A **simulation engine** draws a scoreline from that distribution for every remaining
   fixture, many thousand times over, and tabulates where each club finishes under the
   competition's real tiebreak rules.

This preserves the fixture structure (two clubs' remaining schedules are correlated
when they share opponents) and gives an honest *joint* distribution over the whole
table, not independent per-club estimates.

## 2. Match models

Every model implements one contract: `scoreline_matrix(home, away, max_goals)` returns
an `(max_goals+1, max_goals+1)` array of joint probabilities, rows indexed by home
goals and columns by away goals. Everything else -- 1X2 probabilities, expected goals,
simulation sampling -- is a projection of this one object, computed by shared code
(`evaluate/metrics.py`, `simulate/engine.py`), never reimplemented per model.

### 2.1 `PoissonModel` (rung 1, the floor)

The classic Maher (1982) independent-Poisson model. Each club `i` gets an attack
strength `α_i` and defence strength `β_i`; there is one shared home-advantage term `γ`.
For a fixture between home club `i` and away club `j`:

```
home goals ~ Poisson(α_i · β_j · γ)
away goals ~ Poisson(α_j · β_i)
```

fit jointly by maximum likelihood over every match given, so each club's rating already
accounts for the strength of whoever it played. Every match is weighted equally
regardless of season -- no time decay, deliberately, since this is the floor every
other rung must beat. Delegated to `penaltyblog.models.PoissonGoalsModel`: a well-tested
implementation of a standard method, not worth re-deriving by hand.

The two goal counts are drawn independently given the two rates, which is the model's
known weakness: real match data shows systematic dependence between the two teams'
scores at low totals (0-0 and 1-1 in particular are more common than independence
predicts, 1-0 and 0-1 less common). Rung 2 exists to correct exactly this.

### 2.2 `DixonColesModel` (rung 2)

Adds two things to the Poisson floor:

- **The Dixon and Coles (1997) low-score correction.** A multiplicative adjustment
  `τ(h, a; ρ)` applied to the four low-scoring cells (0-0, 1-0, 0-1, 1-1) of the
  independent-Poisson joint distribution, parameterised by a single correlation term
  `ρ` fit alongside the attack/defence/home parameters. Corrects exactly the dependence
  rung 1 cannot capture.
- **Exponential time decay.** Each match's contribution to the log-likelihood is
  weighted by `exp(-ξ · days_since_match)`, so recent matches count more than old ones.
  A squad from three seasons ago is materially different from today's; unweighted
  fitting would treat them as equally informative. `ξ` is a genuine hyperparameter
  tuned by backtest (`plforecast tune`), never assumed.

Delegated to `penaltyblog.models.DixonColesGoalModel`.

### 2.3 `XGRateModel` (rung 2.5)

Design.md's stated intent is that expected goals (xG), not raw goals, should drive
strength estimation: a club's finishing can run hot or cold for a whole season while
its underlying chance quality stays roughly constant, so goals alone are noisier than
xG at a 38-match horizon. penaltyblog's Poisson and Dixon-Coles models take integer
goal counts as their likelihood, so xG cannot be substituted in directly; `XGRateModel`
is a separate rung built to actually use it (see `docs/adr/0009-xg-in-strength-estimation.md`
for why this needed a new model rather than a data swap).

Attack, defence and home-advantage terms are fit by weighted least squares on the log
scale:

```
log(target_home) = μ + h + attack_i - defence_j
log(target_away) = μ     + attack_j - defence_i
```

with a sum-to-zero constraint on both the attack and defence vectors (so ratings are
relative, not absolute), the same `exp(-ξ · days)` time decay as rung 2, and an optional
per-row `weight` column -- this is how the promoted-club prior enters the fit (section 4
below), as extra pseudo-rows rather than a separate mechanism.

`target` is a tunable blend of xG and actual goals, `blend · xG + (1 - blend) · goals`,
so the goals-vs-xG question is answered by backtest rather than assumed; the tuned
value (`docs/evaluation.md`, "Hyperparameter tuning") currently favours mostly xG with
some goals mixed in. Scorelines are then independent Poissons at the fitted rates, with
the same Dixon-Coles low-score correction available via a `rho` parameter.

### 2.4 The market benchmark

Not a `MatchModel` -- it needs no fitting, since de-vigging is a per-match computation
on that match's own closing price. `evaluate/market.py` converts bookmaker odds to
probabilities by removing the overround (the margin built into odds that makes the
implied probabilities sum to more than 1). Two de-vig methods are computed and
reported: **Shin's method** (the headline; accounts for the fact that a small fraction
of bettors have private information, which multiplicative de-vigging ignores and which
systematically overstates favourites) and plain **multiplicative** normalisation
(divide every implied probability by their sum). See `docs/adr/0007-market-baseline-and-devig.md`
for the price-source fallback chain.

### 2.5 The hierarchical model (rung 4, not built)

The planned next rung: a Bayesian hierarchical Poisson with partial pooling across
clubs. Two things the current rungs cannot do: give a full posterior over team
strength (rather than a point estimate) for honest uncertainty propagation into the
simulation, and handle a club with almost no data (a genuinely blank promoted club)
by shrinking toward a league-wide or division-wide prior automatically, rather than the
hand-built survival-zone prior described in section 4.

## 3. Which model ships

`evaluate/report.py`'s `shipped_model()` picks whichever non-market model has the
lowest primary RPS on the shared evaluation window -- computed from the evaluation
report every time `plforecast evaluate` runs, never hand-asserted. `docs/evaluation.md`
states which model that currently is and by how much it beats the next rung down.
`plforecast forecast --model shipped` (the default) reads this from
`docs/evaluation/metrics.json` and falls back to `dixon-coles` if no evaluation has
been run yet.

## 4. The promoted-club prior

A club with little or no top-flight history in the backfill window (a fresh promotion)
has nothing, or almost nothing, for a match model to fit against. `features/priors.py`
builds a prior from the empirical record instead of leaving such a club unrated.

**The gate.** `needs_prior(club, matches, as_of, xi)` decides whether a club needs the
prior at all, by counting its historical evidence with exponential decay: each past
match counts `exp(-xi * days_since)`, and a club falls under the gate once its
decay-weighted total drops below half a season's worth (19 matches). This distinguishes
a club with a recent season of data (shrunk only a little, since its evidence has
barely decayed) from a club whose only top-flight history is a decade old (treated as
effectively blank). The gate's own decay rate, `PRIOR_GATE_XI`, is *not* a fitted
model's own decay rate -- a fitted model's `xi` (0.0018-0.005/day) is tuned for
smoothing rate estimation and is aggressive enough that even a club with hundreds of
matches across the whole backfill window has an effective count near the threshold
early in any season. `PRIOR_GATE_XI` (a roughly six-year half-life) is tuned
specifically so that continuously-active clubs never trip the gate while a stale
single season does.

**The anchor.** For a club that needs the prior, `build_survival_zone_reference()`
computes the empirical attack and defence rate profile of clubs that finished in the
"survival zone" (15th-18th place: competitive enough to stay up, not comfortably) in
every completed season in the backfill window, real tiebreak rules applied. It reports
both the mean and the standard deviation of each rate across every such observation --
the prior must carry real variance, not a point estimate, because the record is
genuinely unstable (all six promoted clubs were relegated in both 2023/24 and 2024/25,
which had not happened since 1997/98, and then Sunderland finished 7th in 2025/26).

**Delivery into the model.** The prior enters a model's fit as synthetic
pseudo-observations (`prior_pseudo_matches()`): the promoted club plays a number of
fixtures against the real clubs already in the season's fixture list (never a
synthetic "average opponent" -- a club never actually played is not identifiable from
the fit, so its ratings absorb the whole prior and the promoted club's own rating is
left untouched), scoring and conceding at the prior's rates. The number of
pseudo-matches is the prior's own *effective sample size* (`mean / std²`, averaged
across the four rate fields and clamped to 4-38), so a wide, uncertain prior
contributes only a little evidence and a tighter one contributes more.

**Shrinking the anchor toward ClubElo (ADR 0006, story C-16).** `features/clubelo.py`
bridges ClubElo's Elo scale to the prior's rate-field vocabulary: four small
log-linear regressions, `log(rate) = intercept + slope * elo`, fit on established
clubs with both a decayed rate snapshot and a ClubElo rating at the same point in
time, refit at every call site rather than cached (no lookahead bias). Applying that
fit to a promoted club's own current Elo -- available even with zero top-flight
history, since ClubElo covers the Championship -- gives `build_prior()`'s
`external_rating`; `config.clubelo_prior_weight` is `build_prior()`'s own
`external_weight`, in `[0, 1]`. Defaults to `0` (the survival-zone anchor alone,
unshrunk): the only evaluation run so far
(`notebooks/03-prototypes/03-04-clubelo-prior-workbench.ipynb`) reconstructed every
historical promoted-club-like debut ClubElo's cached history could reach and found no
clear RPS improvement on that small sample, even though the live forecast shifted
materially for at least one promoted club. A club with no resolvable ClubElo rating
falls back to the anchor alone regardless of the weight setting.

## 5. Simulation

`simulate/engine.py` draws every remaining fixture's scoreline for every simulated
season in one vectorised pass (not a Python loop over seasons): the default 50,000
simulated seasons, `max_goals` scoreline classes per fixture, sampled from each
fixture's `scoreline_matrix` in a single `numpy.random.Generator.choice` call per
fixture. Already-played matches are identical across every simulation (broadcast once,
not re-drawn); only unplayed fixtures vary.

Final standings are resolved by `simulate/tiebreak.py`'s `PremierLeagueTiebreaks`,
implementing the competition's actual post-2019/20 tiebreak order: points, goal
difference, goals scored, then -- only where the tie affects the title, European
qualification or relegation -- head-to-head points and head-to-head away goals between
the tied clubs, and finally (vanishingly rarely) a coin flip standing in for a neutral-
ground playoff. Most public forecasting models stop at goals scored; the head-to-head
steps make a genuine tie far rarer than that shortcut implies. Ties in *uncontested*
positions (nothing at stake) are resolved by the simulation's own random generator, not
by input order, so no club is systematically favoured by how the code happens to sort
club IDs.

The result is a position count matrix: for each club, how many of the `N` simulations
placed it in each of the `n_clubs` positions. Divided by `N`, this is the club's
`position_pmf`. Summed across clubs for a fixed position, or across positions for a
fixed club, it must equal `N` exactly (the doubly-stochastic invariant checked by a
property-based test).

## 6. Evaluation

### 6.1 Match-level metrics

- **Ranked probability score (RPS)**, the primary metric. Unlike accuracy or log loss,
  RPS treats the three outcomes (home win, draw, away win) as *ordered*: predicting a
  draw when the actual result is a home win is scored as a smaller miss than predicting
  an away win. This is the right notion of "close" for a market where the draw sits
  between the two win outcomes, and it is the accepted standard in football forecasting
  literature.
- **Log loss** and **Brier score**, secondary, unordered metrics: reported for
  comparability with other work, but RPS is what model-ladder promotion decisions are
  made on.
- **Calibration**: for 10 predicted-probability buckets, the mean predicted probability
  versus the empirical frequency the event actually happened, with a Wilson score 95%
  confidence interval (well-behaved at small sample counts and near 0/1, unlike the
  normal approximation). Computed pooled across all three outcome classes (the headline
  reliability diagram) and per class (since pooling can hide, for instance, systematic
  draw mispricing that a Poisson-family model is prone to).

### 6.2 Protocol

Walk-forward, never random k-fold: fit on everything strictly before a cutoff, predict
the next round, advance, refit. This is the only protocol that respects match order --
a random split would let a model "predict" a March result having been trained on
results from the following August. Every model, and the market benchmark, is scored on
the same splits and, for the headline numbers, restricted further to the exact set of
matches every contender actually covered (a walk-forward model cannot price a season's
opening round or a promoted club's debut; the market has no price for matches the
closing-odds source never published) -- so a reported RPS difference between two
models is never an artefact of one seeing an easier subset of matches than the other.

Two refit cadences are available (`--cadence`): **date** (refit after every distinct
match date, the finest fair protocol) and **gameweek** (refit once per full round of
fixtures, reconstructed by clustering `n_clubs // 2` matches at a time in date order,
since historical seasons carry no gameweek number). Gameweek cadence mirrors how the
product actually publishes -- once a week, before that week's fixtures kick off -- and
is stricter: a real Premier League round often spans two to four calendar dates, and
date cadence can use part of a round's own results to help predict the rest of it,
which the live process never does.

### 6.3 Season-level evaluation

Match-level RPS says whether a model prices individual fixtures well. It does not say
whether the *simulated table* -- the actual product -- is an honest picture of where
clubs will finish. `evaluate/season.py` closes that gap: at fixed cutoffs (10, 19 and
28 gameweeks' worth of matches played) in every completed season, the rest of the
season is simulated from a model fit on everything up to that cutoff, and the resulting
position distribution is scored against the club's *realised* final position with the
same ranked-probability logic as match-level RPS, extended to positions instead of
1X2 outcomes. Title, top-four and relegation probabilities are additionally scored with
binary log loss against whether each event actually happened.

### 6.4 Hyperparameter tuning

Every tunable model parameter (`xi` for Dixon-Coles and XGRateModel; `blend` and `rho`
for XGRateModel) is chosen by grid search (`plforecast tune`) with **disjoint
selection and reporting seasons**: the grid is scored on an earlier block of seasons,
the value with the best score there is selected, and *that* value's score on a later,
untouched block of seasons is what gets reported. Selecting and reporting on the same
data would flatter the chosen value with information it should not have had. A
parsimony rule additionally prefers a "null" value (e.g. `rho = 0`, switching a
correction off) when it scores within a small tolerance of the best value on the
selection seasons, so a flat grid does not pick a boundary value on noise.
