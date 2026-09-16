"""Market baseline scoring (design.md section 8.3): de-vigged Pinnacle closing odds,
scored with the exact same metrics as any fitted model. Unlike a model, the market
baseline needs no walk-forward fit -- de-vigging is a per-match computation on that
match's own odds, so it can be scored directly on any subset of stg_matches, including
the same test rows a `run_backtest` call produces, for a like-for-like comparison.

De-vig method: Shin is the headline (design.md section 8.3: "multiplicative
normalisation systematically overstates favourites, which biases the benchmark in
exactly the probability region where the comparison matters most"), with
multiplicative reported alongside so the methodological choice stays visible rather
than buried in a config file. `docs/evaluation.md` is where both get reported together
once the evaluation harness actually runs end to end and has numbers to publish.

Price source: `benchmark_*_odds` on stg_matches, chosen per match by curate's fallback
chain (Pinnacle closing, then Betfair Exchange closing, then the site's average closing
price -- ADR 0007). `benchmark_source` is carried through so results can be sliced by
it. Matches with no complete closing price from any set are dropped here, not imputed.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from penaltyblog.implied import calculate_implied


def market_probabilities(matches: pl.DataFrame, *, method: str = "shin") -> pl.DataFrame:
    """`matches` needs `benchmark_home_odds`, `benchmark_draw_odds`, `benchmark_away_odds`
    (stg_matches' shape). Drops rows with any null odds column. Returns the input rows
    (odds-complete subset) with `p_home`, `p_draw`, `p_away` columns added."""
    complete = matches.filter(
        pl.col("benchmark_home_odds").is_not_null()
        & pl.col("benchmark_draw_odds").is_not_null()
        & pl.col("benchmark_away_odds").is_not_null()
    )

    probs = np.array(
        [
            calculate_implied(
                [
                    row["benchmark_home_odds"],
                    row["benchmark_draw_odds"],
                    row["benchmark_away_odds"],
                ],
                method=method,
            ).probabilities
            for row in complete.iter_rows(named=True)
        ]
    )

    if complete.height == 0:
        return complete.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(c) for c in ("p_home", "p_draw", "p_away")
        )

    return complete.with_columns(
        pl.Series("p_home", probs[:, 0]),
        pl.Series("p_draw", probs[:, 1]),
        pl.Series("p_away", probs[:, 2]),
    )
