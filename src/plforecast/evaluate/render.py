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
        "xg-rates": "`XGRateModel`",
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


def _season_level_table(summary: list[dict[str, Any]]) -> str:
    lines = [
        "| Cutoff (matches played) | Model | Seasons | Position RPS | Title log loss "
        "| Top-four log loss | Relegation log loss |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['cutoff']} | {_label(row['model'])} | {row['n_seasons']} | "
            f"{row['position_rps']:.4f} | {row['title_log_loss']:.3f} | "
            f"{row['top_four_log_loss']:.3f} | {row['relegation_log_loss']:.3f} |"
        )
    return "\n".join(lines)


def _tuning_section(tuning: Mapping[str, Any]) -> str:
    if not tuning:
        return (
            "No tuning run has been recorded yet; the shipped hyperparameters are the "
            "config defaults (`config.py`)."
        )
    parts = []
    for key, run in tuning.items():
        parsimony = (
            f" The lowest selection RPS was at {run['best_by_selection_rps']}, within "
            f"{run['tolerance']} of the null value {run['null_value']}, so the null value "
            "is selected (parsimony rule)."
            if run.get("parsimony_applied")
            else ""
        )
        lines = [
            f"**{key}** (`{run['parameter']}`): selected on "
            f"{run['selection_seasons'][0]} to {run['selection_seasons'][-1]}, reported on "
            f"{run['report_seasons'][0]} to {run['report_seasons'][-1]}. "
            f"Selected value: **{run['selected']}**.{parsimony}",
            "",
            "| Value | Selection RPS | Selection n | Report RPS | Report n |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in run["grid"]:
            mark = " (selected)" if row["value"] == run["selected"] else ""
            lines.append(
                f"| {row['value']}{mark} | {row['select_rps']:.4f} | {row['select_n']:,} | "
                f"{row['report_rps']:.4f} | {row['report_n']:,} |"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def render_markdown(report: Mapping[str, Any]) -> str:
    window = report["window"]
    cadence = report.get("cadence", "date")
    cadence_label = {
        "date": "date (next distinct match date)",
        "gameweek": "gameweek (next full round, mirroring the publishing cadence)",
    }.get(cadence, cadence)
    backtest_seconds = report.get("backtest_seconds", 0.0)
    coverage = report["coverage"]
    primary = {r["model"]: r for r in report["primary"]}
    models = [r["model"] for r in report["primary"]]
    shipped_name = report["shipped_model"]

    floor = primary[FLOOR_MODEL]
    shipped = primary[shipped_name]
    market = primary[HEADLINE_MARKET]
    beats_floor = shipped["rps"] < floor["rps"]
    gap = shipped["rps"] - market["rps"]

    unrateable = coverage["unrateable"][shipped_name]
    warmup = coverage["warmup_excluded"][shipped_name]
    n_intersection = coverage["intersection"]
    sources = coverage["benchmark_by_source"]
    source_text = ", ".join(f"{name} {n:,}" for name, n in sources.items())

    debut_pairs = ", ".join(
        sorted(
            {
                m["home_club_id"] + "/" + m["away_club_id"]
                for m in coverage["unrateable_matches"][shipped_name]
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
        rows = [r for r in report["calibration"][shipped_name] if r["outcome"] == outcome]
        table, misses = _calibration_table(rows)
        total_misses[outcome] = (misses, len(rows))
        title = "All outcomes pooled" if outcome == "pooled" else f"{outcome.title()} only"
        calibration_sections.append(f"#### {title}\n\n{table}")

    miss_text = "; ".join(
        f"{name}: {misses} of {n} buckets outside" for name, (misses, n) in total_misses.items()
    )

    ranking = ", ".join(
        f"{_label(m)} {primary[m]['rps']:.4f}"
        for m in sorted(
            (m for m in models if not m.startswith("market")), key=lambda m: primary[m]["rps"]
        )
    )
    verdict = (
        f"**{_label(shipped_name)} has the best held-out RPS of the non-market models "
        f"({ranking}) and ships**, per the model ladder's gate (design.md section 6.2): "
        "each rung must beat the one before it on identical rows."
        if beats_floor
        else f"**No model beats the Poisson floor ({ranking}); nothing above the floor ships.**"
    )
    market_verdict = (
        f"**Neither model beats the market.** The shipped model's RPS gap to the Shin "
        f"de-vigged benchmark is {gap:+.4f}; design.md section 1.3's stretch target is "
        f"within {MARKET_STRETCH_GAP}."
        if gap > 0
        else f"**The shipped model beats the market** by {-gap:.4f} RPS."
    )
    stretch = "met" if gap <= MARKET_STRETCH_GAP else "not met"

    season_table = (
        _season_level_table(report["season_level"]["summary"])
        if report["season_level"]["summary"]
        else "Not run in this report (`plforecast evaluate --no-season-level`)."
    )

    return f"""# Evaluation

Generated by `plforecast evaluate` on {report["generated_at"]} from
`docs/evaluation/metrics.json`. Do not hand-edit: re-run `just evaluate` whenever a
model or the training data changes, then `just render-evaluation`.

## Protocol

- **Window**: {window["first_season"]} through {window["last_season"]}
  ({window["n_seasons"]} seasons, {window["matches"]:,} matches; design.md section 8.2).
  The in-progress season is excluded because its results are not final.
- **Split**: walk-forward, cadence **{cadence_label}** (`evaluate/backtest.py`, story
  C-09), never random k-fold. Fit on everything strictly before a cutoff, predict the
  next round, advance. Every model sees identical splits. This run took
  {backtest_seconds:.1f}s across every model's refits.
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

`{shipped_name}` reliability curves on the primary rows (`evaluate/calibration.py`,
10 buckets, Wilson score 95% confidence intervals). A bucket is marked *outside* when
the mean predicted probability falls outside the empirical frequency's interval.

Summary: {miss_text}.

{(chr(10) * 2).join(calibration_sections)}

The pooled curve is the headline reliability diagram (design.md section 8.1). The
per-outcome curves exist because pooling hides class-specific error: Poisson-family
models are known to misprice draws, and the draw curve is where to look for it.

## Season-level evaluation

Design.md section 8.1's product-level check (`evaluate/season.py`): for every completed
season in the window, the rest of the season is simulated from a frozen cutoff
({report["season_level"]["simulations"]:,} simulations, the realised results up to the cutoff
as played matches, the unplayed round-robin pairs as remaining fixtures) and the position
distribution is scored against the realised final table. Position RPS treats positions
as ordered, so an off-by-one miss costs less than an off-by-ten miss; lower is better for
every column.

{season_table}

## Hyperparameter tuning

{_tuning_section(report["tuning"])}

## What this doesn't cover yet

- **The hierarchical model** (design.md section 6.2, rung 3) is not built.
- **The promoted-club prior** is not yet part of the backtest; its effect is only
  visible in the live forecast (story C-08).
"""


def render_readme_table(report: Mapping[str, Any]) -> str:
    """The short table the README carries: primary RPS for the floor, the shipped
    model and the headline market."""
    primary = {r["model"]: r for r in report["primary"]}
    lines = ["| Model | Mean RPS |", "| --- | --- |"]
    non_market = [m for m in primary if not m.startswith("market")]
    for model in (*non_market, HEADLINE_MARKET):
        lines.append(f"| {_label(model)} | {primary[model]['rps']:.4f} |")
    return "\n".join(lines)
