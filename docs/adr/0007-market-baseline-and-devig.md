# ADR 0007: Market baseline price source and de-vig method

**Status**: accepted 14 September 2026; amended 16 September 2026 (fallback chain).

## Context

The evaluation harness scores every model against de-vigged closing odds (design.md
section 8). The benchmark needs a price for every match in the backtest window and for
every match of the live season, from one consistent source where possible.

Bookmaker odds carry a margin (overround). Removing it to obtain probabilities is a
methodological choice that changes the benchmark's RPS in the probability region where
the comparison matters most.

## Options considered

Price source:
1. **Pinnacle closing** (`PSCH/PSCD/PSCA` on football-data.co.uk). Sharp, liquid, the
   convention in the forecasting literature, continuous since before 2015/16.
2. **Betfair Exchange closing** (`BFECH/BFECD/BFECA`). Near-zero margin, but only present
   in the site's files from 2024/25 and known for early-season liquidity gaps.
3. **The site's average closing price** (`AvgCH/AvgCD/AvgCA`). Present from 2019/20,
   never missing while any bookmaker reports, but a blend rather than one sharp book.

De-vig method: multiplicative, additive, power, Shin, differential margin weighting, odds
ratio, logarithmic (all supported by `penaltyblog.implied.calculate_implied`).

## Decision

**Price source is a fallback chain, applied per match at curate time**, in this order:
Pinnacle closing, then Betfair Exchange closing, then the site's average closing price.
The chosen source is recorded on every row (`stg_matches.benchmark_source`) so results can
be sliced by it. Every closing-price set the file carries is also landed in long form in
`stg_odds` so the chain can be changed without re-ingesting.

**De-vig method is Shin**, with multiplicative reported alongside it in
`docs/evaluation.md` so the choice stays visible.

## Why the amendment

The original decision was Pinnacle alone. On 16 September 2026 the ingested data showed
that football-data.co.uk stopped populating Pinnacle closing prices after 8 January 2026
(the last 170 matches of 2025/26 are null) and dropped the columns entirely from the
2026/27 file. A single-source benchmark would therefore have no price for the live
season, which defeats the purpose of publishing a market comparison.

Verified column coverage by season file:

| Seasons | Closing-price sets present |
| --- | --- |
| 2015/16 to 2018/19 | Pinnacle only |
| 2019/20 to 2023/24 | Pinnacle, Bet365, Max, Avg |
| 2024/25 | Pinnacle, Bet365, Max, Avg, Betfair Exchange |
| 2025/26 | as 2024/25, Pinnacle null from 17 January 2026 |
| 2026/27 | Bet365, Max, Avg, Betfair Exchange |

Betfair Exchange is second in the chain because it is the closest thing to a sharp
price after Pinnacle; the average is last because it is a blend. A benchmark whose source
changes mid-window is imperfect, and the evaluation reports the per-source split so a
reader can see where the switch happens.

## Consequences

- Every match in the backtest window and the live season has a benchmark price.
- Metrics are comparable to published Pinnacle-based figures only for the seasons where
  the source is Pinnacle; `docs/evaluation.md` states the split.
- Adding a further source is a change to `CLOSING_ODDS_SETS` in the adapter and, if it
  should be a fallback, to `BENCHMARK_CHAIN` in curate.
