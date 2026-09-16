"""Render docs/evaluation.md from the evaluation report (metrics.json), so the published
numbers are never hand-transcribed and the prose that depends on them (which model
ships, whether the market is beaten, how many calibration buckets miss) is computed
rather than asserted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plforecast.evaluate.calibration import OUTCOME_NAMES

FLOOR_MODEL = "poisson"
SHIPPED_MODEL = "dixon-coles"
HEADLINE_MARKET = "market (shin)"
MARKET_STRETCH_GAP = 0.005  # design.md 1.3


def _metrics_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | n | Mean RPS | Mean log loss | Mean Brier |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {_label(row['model'])} | {row['n']:,} | {row['rps']:.4f} | "
            f"{row['log_loss']:.4f} | {row['brier']:.4f} |"
        )
    return "\n".join(lines)


def _label(model: str) -> str:
    return {
        "poisson": "`PoissonModel` (floor)",
        "dixon-coles": "`DixonColesModel`",
        "market (shin)": "Market, Shin de-vig",
        "market (multiplicative)": "Market, multiplicative de-vig",
    }.get(model, f"`{model}`")


def _by_season_table(rows: list[dict[str, Any]], models: list[str]) -> str:
    seasons = sorted({r["season"] for r in rows})
    lookup = {(r["model"], r["season"]): r for r in rows}
    header = "| Season | n | " + " | ".join(_label(m) for m in models) + " |"
    lines = [header, "| --- | --- | " + " | ".join("---" for _ in models) + " |"]
    for season in seasons:
        n = lookup[(models[0], season)]["n"]
        cells = " | ".join(f"{lookup[(m, season)]['rps']:.4f}" for m in models)
        lines.append(f"| {season} | {n:,} | {cells} |")
    return "\n".join(lines)


def _calibration_table(rows: list[dict[str, Any]]) -> tuple[str, int]:
    """Markdown table plus the number of buckets whose mean prediction falls outside
    the empirical confidence interval."""
    lines = [
        "| Predicted range | Mean predicted | Empirical frequency | n | 95% CI | |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    misses = 0
    for row in rows:
        inside = row["ci_low"] <= row["mean_predicted"] <= row["ci_high"]
        misses += 0 if inside else 1
        lines.append(
            f"| {row['bucket_low']:.1f}-{row['bucket_high']:.1f} | {row['mean_predicted']:.3f} | "
            f"{row['empirical_frequency']:.3f} | {row['n']:,} | "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {'' if inside else 'outside'} |"
        )
    return "\n".join(lines), misses


def render_markdown(report: Mapping[str, Any]) -> str:
    window = report["window"]
    coverage = report["coverage"]
    primary = {r["model"]: r for r in report["primary"]}
    models = [r["model"] for r in report["primary"]]

    floor = primary[FLOOR_MODEL]
    shipped = primary[SHIPPED_MODEL]
    market = primary[HEADLINE_MARKET]
    beats_floor = shipped["rps"] < floor["rps"]
    gap = shipped["rps"] - market["rps"]

    unrateable = coverage["unrateable"][SHIPPED_MODEL]
    warmup = coverage["warmup_excluded"][SHIPPED_MODEL]
    n_intersection = coverage["intersection"]
    sources = coverage["benchmark_by_source"]
    source_text = ", ".join(f"{name} {n:,}" for name, n in sources.items())

    debut_pairs = ", ".join(
        sorted(
            {
                m["home_club_id"] + "/" + m["away_club_id"]
                for m in coverage["unrateable_matches"][SHIPPED_MODEL]
            }
        )
    )
    min_train = report["min_train_matches"]
    first_season = window["first_season"]
    coverage_rows = "\n".join(
        [
            "| Bucket | Matches | Why |",
            "| --- | --- | --- |",
            f"| Scored (primary) | {n_intersection:,} | Priced by every model and the benchmark |",
            f"| Warm-up | {warmup:,} | Fewer than {min_train} training matches preceded them "
            f"(start of {first_season}) |",
            f"| Unrateable | {unrateable:,} | A club with no prior-window history: "
            "promoted-club debuts (see below) |",
            f"| No benchmark price | {coverage['benchmark_missing']:,} | No closing price from "
            "any source in the chain |",
        ]
    )

    calibration_sections = []
    total_misses = {}
    for outcome in ("pooled", *OUTCOME_NAMES):
        rows = [r for r in report["calibration"][SHIPPED_MODEL] if r["outcome"] == outcome]
        table, misses = _calibration_table(rows)
        total_misses[outcome] = (misses, len(rows))
        title = "All outcomes pooled" if outcome == "pooled" else f"{outcome.title()} only"
        calibration_sections.append(f"#### {title}\n\n{table}")

    miss_text = "; ".join(
        f"{name}: {misses} of {n} buckets outside" for name, (misses, n) in total_misses.items()
    )

    verdict = (
        f"**{_label(SHIPPED_MODEL)} beats the Poisson floor on held-out RPS "
        f"({shipped['rps']:.4f} < {floor['rps']:.4f}) and ships**, per the model ladder's "
        "gate (design.md section 6.2)."
        if beats_floor
        else f"**{_label(SHIPPED_MODEL)} does not beat the Poisson floor "
        f"({shipped['rps']:.4f} >= {floor['rps']:.4f}) and does not ship.**"
    )
    market_verdict = (
        f"**Neither model beats the market.** The shipped model's RPS gap to the Shin "
        f"de-vigged benchmark is {gap:+.4f}; design.md section 1.3's stretch target is "
        f"within {MARKET_STRETCH_GAP}."
        if gap > 0
        else f"**The shipped model beats the market** by {-gap:.4f} RPS."
    )
    stretch = "met" if gap <= MARKET_STRETCH_GAP else "not met"

    return f"""# Evaluation

Generated by `plforecast evaluate` on {report["generated_at"]} from
`docs/evaluation/metrics.json`. Do not hand-edit: re-run `just evaluate` whenever a
model or the training data changes, then `just render-evaluation`.

## Protocol

- **Window**: {window["first_season"]} through {window["last_season"]}
  ({window["n_seasons"]} seasons, {window["matches"]:,} matches; design.md section 8.2).
  The in-progress season is excluded because its results are not final.
- **Split**: walk-forward by date (`evaluate/backtest.py`), never random k-fold. Fit on
  everything strictly before a cutoff date, predict every match on the next distinct
  match date, advance. Every model sees identical splits.
- **Benchmark**: closing odds de-vigged with Shin (headline) and multiplicative
  (reported alongside), from the fallback chain in ADR 0007. Source split for the
  window: {source_text}.
- **Identical rows**: the primary table scores every model and the benchmark on the
  same {n_intersection:,} matches: those every model could price and the benchmark
  covered. Numbers on each contender's own full coverage are in the secondary table.

## Coverage

Of {window["matches"]:,} matches in the window:

{coverage_rows}

Unrateable matches are the first appearance of a club that had not played in the
window before: {debut_pairs}.
This is the gap `features/priors.py` exists to fill for the model layer; the harness
reports it rather than papering over it.

## Results

Primary, on identical rows:

{_metrics_table(report["primary"])}

{verdict}

{market_verdict} Stretch target {stretch}.

Secondary, each contender on every row it covered (not like-for-like; shown so the
effect of restricting to identical rows is visible):

{_metrics_table(report["secondary_full_set"])}

Per season, primary rows, mean RPS:

{_by_season_table(report["by_season"], models)}

## Calibration

`{SHIPPED_MODEL}` reliability curves on the primary rows (`evaluate/calibration.py`,
10 buckets, Wilson score 95% confidence intervals). A bucket is marked *outside* when
the mean predicted probability falls outside the empirical frequency's interval.

Summary: {miss_text}.

{(chr(10) * 2).join(calibration_sections)}

The pooled curve is the headline reliability diagram (design.md section 8.1). The
per-outcome curves exist because pooling hides class-specific error: Poisson-family
models are known to misprice draws, and the draw curve is where to look for it.

## What this doesn't cover yet

- **Season-level evaluation** (realised final position vs predicted position
  distribution, design.md section 8.1): story C-07.
- **The hierarchical model** (design.md section 6.2, rung 3) is not built.
- **`xi` tuning** (story C-03): the shipped Dixon-Coles uses the paper's 0.0018 until
  the grid search lands; this file will then show the grid.
"""


def render_readme_table(report: Mapping[str, Any]) -> str:
    """The short table the README carries: primary RPS for the floor, the shipped
    model and the headline market."""
    primary = {r["model"]: r for r in report["primary"]}
    lines = ["| Model | Mean RPS |", "| --- | --- |"]
    for model in (FLOOR_MODEL, SHIPPED_MODEL, HEADLINE_MARKET):
        lines.append(f"| {_label(model)} | {primary[model]['rps']:.4f} |")
    return "\n".join(lines)
