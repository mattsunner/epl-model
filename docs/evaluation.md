# Evaluation

Results from the walk-forward backtest (`evaluate/backtest.py`), run for real against
the full curated `stg_matches` table. Regenerate with `just evaluate` (or the
equivalent ad hoc script) whenever a model or the training data changes -- this file is
a snapshot, not something to hand-edit.

## Protocol

- **Window**: 2015/16 through 2025/26 (design.md section 8.2) -- the in-progress
  2026/27 season is excluded, since its final results aren't known yet.
- **Split**: walk-forward by date (`evaluate/backtest.py`), never random k-fold. Fit on
  everything strictly before a cutoff date, predict every match sharing the next
  distinct match date, advance. Every model sees identical splits.
- **Coverage**: 4,180 matches in the window; 4,066 scored (97.3%). 14 matches skipped --
  every one is a round-1 fixture in 2015/16 (the very first season in the window, where
  by definition no club has any prior-window history yet) or a later promoted club with
  no top-flight history since 2015/16. This is the same real gap `features/priors.py`
  exists to eventually fill for the model layer; the harness's job here is only to
  report the gap, not paper over it.

## Results

| Model | n | Mean RPS | Mean log loss | Mean Brier |
| --- | --- | --- | --- | --- |
| `PoissonModel` (floor) | 4,066 | 0.2063 | 0.9935 | 0.5907 |
| `DixonColesModel` (`xi=0.0018`) | 4,066 | 0.2010 | 0.9789 | 0.5804 |
| Market, Pinnacle closing, Shin de-vig | 4,010 | 0.1935 | 0.9523 | 0.5637 |
| Market, Pinnacle closing, multiplicative de-vig | 4,010 | 0.1935 | 0.9523 | 0.5637 |

**Dixon-Coles beats the Poisson floor on held-out RPS (0.2010 < 0.2063) and ships**,
per the model ladder's own gate (design.md section 6.2: "each must beat the one before
it on held-out RPS or it does not ship"). The improvement is consistent across log loss
and Brier too, not just RPS.

**Neither model beats the market yet.** Stated plainly, as design.md's own risk table
requires (section 13: "Model does not beat closing odds -> Disappointment, or worse,
quiet omission -> Stated as a likely outcome up front"). This was anticipated, not a
surprise -- Pinnacle is a sharp, liquid market, and Dixon-Coles is still a
two-parameter-per-club model with no promoted-club prior, no European-congestion term,
and no posterior uncertainty.

**Shin and multiplicative de-vigging are nearly identical here** (both methods agree
to four decimal places on every metric). Design.md section 8.3's stated reason to
prefer Shin -- multiplicative systematically overstates favourites -- is a real effect
in general, but a small one for this specific market: Pinnacle is priced sharply enough
that the two methods rarely disagree by much. Both are reported, per design.md's
instruction, so the methodological choice stays visible rather than buried in a config
file, even though it turns out not to matter much for this backtest.

## Calibration

`DixonColesModel`'s calibration curve (`evaluate/calibration.py`), pooling all three
1X2 outcomes into one reliability diagram (10 buckets, Wilson score 95% confidence
intervals):

| Predicted range | Mean predicted | Empirical frequency | n | 95% CI |
| --- | --- | --- | --- | --- |
| 0.0-0.1 | 0.068 | 0.092 | 523 | [0.070, 0.120] |
| 0.1-0.2 | 0.159 | 0.154 | 1,840 | [0.138, 0.171] |
| 0.2-0.3 | 0.251 | 0.259 | 4,454 | [0.247, 0.272] |
| 0.3-0.4 | 0.345 | 0.340 | 1,909 | [0.319, 0.362] |
| 0.4-0.5 | 0.449 | 0.429 | 1,297 | [0.403, 0.457] |
| 0.5-0.6 | 0.545 | 0.530 | 961 | [0.498, 0.561] |
| 0.6-0.7 | 0.646 | 0.653 | 668 | [0.616, 0.688] |
| 0.7-0.8 | 0.747 | 0.764 | 382 | [0.719, 0.804] |
| 0.8-0.9 | 0.837 | 0.826 | 144 | [0.756, 0.880] |
| 0.9-1.0 | 0.931 | 0.900 | 20 | [0.699, 0.972] |

**Well calibrated in practice**: the mean predicted probability falls inside (or right
at the edge of) the empirical frequency's confidence interval in every one of the 10
buckets, across the full 0-1 range. This is the more useful, checkable claim publishing
this chart is meant to support (design.md section 10.1: "publishing this is what
separates a credible forecast from a confident one") -- a model can have a middling RPS
and still be honest about its own uncertainty, and that is what this curve is actually
evidence for, distinct from the ranking comparison above.

## What this doesn't cover yet

- **Season-level evaluation** (realised final position vs predicted position
  distribution, design.md section 8.1) needs the simulation engine run at a frozen
  gameweek across historical seasons -- not yet wired into this harness.
- **Betfair Exchange** as the secondary sanity check (design.md section 8.3) -- no
  adapter for it exists.
- **The hierarchical model** (design.md section 6.2, rung 3) isn't built yet, so it
  isn't in this table.
