import json
from pathlib import Path

from plforecast.evaluate.render import render_markdown, render_readme_table


def _report():
    cal = [
        {
            "outcome": o,
            "bucket_low": 0.0,
            "bucket_high": 0.1,
            "mean_predicted": 0.05,
            "empirical_frequency": 0.06,
            "n": 10,
            "ci_low": 0.01,
            "ci_high": 0.2,
        }
        for o in ("pooled", "home", "draw", "away")
    ]
    return {
        "generated_at": "2026-09-16T00:00:00+00:00",
        "window": {
            "first_season": "2015/16",
            "last_season": "2025/26",
            "n_seasons": 11,
            "matches": 100,
        },
        "coverage": {
            "warmup_excluded": {"poisson": 10, "dixon-coles": 10},
            "unrateable": {"poisson": 1, "dixon-coles": 1},
            "unrateable_matches": {"dixon-coles": [{"home_club_id": "x", "away_club_id": "y"}]},
            "benchmark_by_source": {"pinnacle": 90},
            "benchmark_missing": 0,
            "intersection": 89,
        },
        "min_train_matches": 10,
        "primary": [
            {"model": "poisson", "n": 89, "rps": 0.21, "log_loss": 1.0, "brier": 0.6},
            {"model": "dixon-coles", "n": 89, "rps": 0.20, "log_loss": 0.98, "brier": 0.58},
            {"model": "market (shin)", "n": 89, "rps": 0.19, "log_loss": 0.95, "brier": 0.56},
            {
                "model": "market (multiplicative)",
                "n": 89,
                "rps": 0.19,
                "log_loss": 0.95,
                "brier": 0.56,
            },
        ],
        "secondary_full_set": [
            {"model": "poisson", "n": 89, "rps": 0.21, "log_loss": 1.0, "brier": 0.6},
            {"model": "dixon-coles", "n": 89, "rps": 0.20, "log_loss": 0.98, "brier": 0.58},
            {"model": "market (shin)", "n": 100, "rps": 0.19, "log_loss": 0.95, "brier": 0.56},
            {
                "model": "market (multiplicative)",
                "n": 100,
                "rps": 0.19,
                "log_loss": 0.95,
                "brier": 0.56,
            },
        ],
        "by_season": [
            {"model": m, "season": "2015/16", "n": 89, "rps": 0.2, "log_loss": 1, "brier": 0.6}
            for m in ("poisson", "dixon-coles", "market (shin)", "market (multiplicative)")
        ],
        "calibration": {"dixon-coles": cal},
        "shipped_model": "dixon-coles",
        "season_level": {"simulations": 100, "summary": [], "scores": []},
        "tuning": {},
    }


def test_render_markdown_computes_the_verdicts_from_the_numbers():
    md = render_markdown(_report())
    assert "`DixonColesModel` has the best held-out RPS of the non-market models" in md
    assert "and ships" in md
    assert "No tuning run has been recorded" in md
    assert "Neither model beats the market" in md
    assert "gap to the Shin de-vigged benchmark is +0.0100" in md
    assert "Stretch target not met" in md
    assert "| Scored (primary) | 89 |" in md
    assert "pooled: 0 of 1 buckets outside" in md


def test_render_readme_table_lists_floor_shipped_and_market():
    table = render_readme_table(_report())
    assert table.splitlines()[2].startswith("| `PoissonModel` (floor) | 0.2100 |")
    assert len(table.splitlines()) == 5  # header, rule, two models, market


def test_render_round_trips_through_json(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_report()))
    assert render_markdown(json.loads(path.read_text())).startswith("# Evaluation")
