import json
from datetime import date, timedelta

import numpy as np
import polars as pl

from plforecast.evaluate.report import MATCH_KEY, build_report, format_table, write_report
from plforecast.models.base import UnknownClubError


def _matches(n_rounds: int, *, odds_from_round: int = 0) -> pl.DataFrame:
    rows = []
    start = date(2020, 8, 1)
    for r in range(n_rounds):
        d = start + timedelta(days=7 * r)
        has_odds = r >= odds_from_round
        for home, away, hg, ag, res in (("a", "b", 2, 1, "H"), ("c", "d", 0, 0, "D")):
            rows.append(
                {
                    "season": "2020/21",
                    "date": d,
                    "home_club_id": home,
                    "away_club_id": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "result": res,
                    "benchmark_home_odds": 2.0 if has_odds else None,
                    "benchmark_draw_odds": 3.5 if has_odds else None,
                    "benchmark_away_odds": 4.0 if has_odds else None,
                    "benchmark_source": "pinnacle" if has_odds else None,
                }
            )
    return pl.DataFrame(rows)


class _StubModel:
    def fit(self, matches):
        self._known = set(matches["home_club_id"]) | set(matches["away_club_id"])
        return self

    def scoreline_matrix(self, home, away, max_goals=10):
        if home not in self._known or away not in self._known:
            raise UnknownClubError("unknown")
        m = np.zeros((max_goals + 1, max_goals + 1))
        m[1, 0], m[0, 0], m[0, 1] = 0.5, 0.3, 0.2
        return m

    @property
    def config_hash(self):
        return "stub"


def test_primary_metrics_use_identical_rows_for_models_and_market():
    # 6 rounds; the benchmark only exists from round 3; the model warm-up is 2 matches
    # (round 1) so round 1 is warm-up-excluded. Intersection = rounds 3-6 = 8 matches.
    matches = _matches(6, odds_from_round=2)

    report = build_report(matches, {"stub": _StubModel}, min_train_matches=2, season_level=False)

    primary = {row["model"]: row for row in report["primary"]}
    assert report["shipped_model"] == "stub"
    assert report["coverage"]["intersection"] == 8
    assert {row["n"] for row in primary.values()} == {8}
    secondary = {row["model"]: row for row in report["secondary_full_set"]}
    assert secondary["stub"]["n"] == 10  # rounds 2-6 on its own
    assert secondary["market (shin)"]["n"] == 8
    assert report["coverage"]["warmup_excluded"] == {"stub": 2}
    assert report["coverage"]["benchmark_missing"] == 4
    assert report["coverage"]["benchmark_by_source"] == {"pinnacle": 8}


def test_report_reconciles_exactly_and_serialises(tmp_path):
    matches = _matches(4)
    report = build_report(matches, {"stub": _StubModel}, min_train_matches=2, season_level=False)

    scored = report["secondary_full_set"][0]["n"]
    assert (
        scored
        + report["coverage"]["warmup_excluded"]["stub"]
        + report["coverage"]["unrateable"]["stub"]
        == matches.height
    )
    assert set(report["calibration"]) == {"stub"}
    assert {row["outcome"] for row in report["calibration"]["stub"]} == {
        "pooled",
        "home",
        "draw",
        "away",
    }

    written = write_report(report, tmp_path)
    assert [p.name for p in written] == ["metrics.json", "calibration.csv"]
    reloaded = json.loads((tmp_path / "metrics.json").read_text())
    assert reloaded["primary"][0]["n"] == report["primary"][0]["n"]
    assert "model,outcome,bucket_low" in (tmp_path / "calibration.csv").read_text().splitlines()[0]
    assert MATCH_KEY == ["season", "date", "home_club_id", "away_club_id"]


def test_format_table_has_one_line_per_row_plus_header():
    rows = [{"model": "x", "n": 3, "rps": 0.2, "log_loss": 1.0, "brier": 0.6}]
    assert len(format_table(rows).splitlines()) == 2
