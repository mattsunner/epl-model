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

Coverage gap (docs/data-sources.md): Pinnacle closing odds are missing for the entire
2026/27 season and roughly the second half of 2025/26 in the data this pipeline has
ingested. Matches with a null odds column are dropped here, not imputed -- an honest
gap in the benchmark, not something to paper over with a guessed price.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from penaltyblog.implied import calculate_implied


def market_probabilities(matches: pl.DataFrame, *, method: str = "shin") -> pl.DataFrame:
    """`matches` needs `pinnacle_home_odds`, `pinnacle_draw_odds`, `pinnacle_away_odds`
    (stg_matches' shape). Drops rows with any null odds column. Returns the input rows
    (odds-complete subset) with `p_home`, `p_draw`, `p_away` columns added."""
    complete = matches.filter(
        pl.col("pinnacle_home_odds").is_not_null()
        & pl.col("pinnacle_draw_odds").is_not_null()
        & pl.col("pinnacle_away_odds").is_not_null()
    )

    probs = np.array(
        [
            calculate_implied(
                [row["pinnacle_home_odds"], row["pinnacle_draw_odds"], row["pinnacle_away_odds"]],
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
