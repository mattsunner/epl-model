"""Calibration curves (design.md section 8.1): predicted vs realised outcome frequency
in probability buckets, with binomial confidence bands. "Publishing this is what
separates a credible forecast from a confident one" (section 10.1).

Every (match, candidate outcome) pair -- home win, draw, away win -- is flattened into
one binary calibration point: the model's predicted probability for that specific
outcome, and whether it actually happened. Pooling all three outcome classes into one
curve (rather than three separate per-class curves) is a deliberate simplification: it
is the standard one-vs-rest reliability-diagram construction, and keeps a single chart
answering "when this model says X%, does X% of the time happen" across the whole
1X2 market rather than three harder-to-read panels.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


def calibration_curve(
    probs: np.ndarray, outcomes: np.ndarray, *, n_buckets: int = 10, confidence: float = 0.95
) -> pl.DataFrame:
    """`probs`: (n, 3) array of [P(home win), P(draw), P(away win)]. `outcomes`: (n,)
    array of realised outcome indices (0/1/2). Returns one row per non-empty bucket:
    the bucket's probability range, the mean predicted probability within it, the
    empirical frequency the event actually happened, the sample count, and a Wilson
    score confidence interval on that frequency (well-behaved near 0/1 and at the
    small sample counts individual buckets can end up with, unlike the normal
    approximation)."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(outcomes)), outcomes] = 1.0

    flat_probs = probs.ravel()
    flat_actual = one_hot.ravel()

    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_idx = np.clip(np.digitize(flat_probs, bucket_edges[1:-1]), 0, n_buckets - 1)

    z = stats.norm.ppf(0.5 + confidence / 2)
    rows = []
    for b in range(n_buckets):
        mask = bucket_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        p_hat = float(flat_actual[mask].mean())
        # Wilson score interval.
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        half_width = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        rows.append(
            {
                "bucket_low": float(bucket_edges[b]),
                "bucket_high": float(bucket_edges[b + 1]),
                "mean_predicted": float(flat_probs[mask].mean()),
                "empirical_frequency": p_hat,
                "n": n,
                "ci_low": max(0.0, center - half_width),
                "ci_high": min(1.0, center + half_width),
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "bucket_low": pl.Float64,
            "bucket_high": pl.Float64,
            "mean_predicted": pl.Float64,
            "empirical_frequency": pl.Float64,
            "n": pl.Int64,
            "ci_low": pl.Float64,
            "ci_high": pl.Float64,
        },
    )
