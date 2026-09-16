"""Full pipeline integration test (story A-17, story B-13): migrate/curate/validate/
evaluate over a small, committed, real-club fixture dataset instead of live sources --
CI has no network access, so this is the only place the whole pipeline runs end to end
rather than one layer at a time.

`tests/fixtures/integration_data/raw/` holds four real, currently-top-flight clubs
(Arsenal, Chelsea, Liverpool, Man City -- chosen so `dimension.resolve()` succeeds
against the real `club_aliases.yaml`, not a mini fixture): a complete 2020/21 round
robin from football-data and Understat, and a 2021/22 FPL fixture list (partly
finished) with its own gameweek calendar. Regenerate with
`uv run python tests/fixtures/integration_data/generate.py` if a parsed raw schema
ever changes shape.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from plforecast.config import Settings
from plforecast.evaluate.report import build_report
from plforecast.models.poisson import PoissonModel
from plforecast.storage.curate import curate_all
from plforecast.storage.db import connect
from plforecast.storage.validate import run_validations

FIXTURE_RAW = Path(__file__).parent.parent / "fixtures" / "integration_data" / "raw"


@pytest.fixture
def config(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    shutil.copytree(FIXTURE_RAW, data_dir / "raw")
    return Settings(data_dir=data_dir)


def test_migrate_curate_validate_evaluate_over_a_real_fixture_dataset(config: Settings):
    # migrate (connect() applies table migrations and (re)creates the raw_* views)
    conn = connect(config)

    # curate
    manifest = curate_all(conn)
    assert manifest["tables"]["stg_matches"]["rows"] == 12
    assert manifest["tables"]["stg_fixtures"]["rows"] == 12
    assert manifest["tables"]["stg_gameweeks"]["rows"] == 6
    assert manifest["tables"]["dim_club"]["rows"] > 30  # the real club dimension, not a fixture one
    assert manifest["tables"]["mart_team_match"]["rows"] == 24  # 2 rows per match

    resolved_clubs = conn.execute("SELECT DISTINCT home_club_id FROM stg_matches").pl()
    assert set(resolved_clubs["home_club_id"].to_list()) == {
        "arsenal",
        "chelsea",
        "liverpool",
        "man-city",
    }
    # Understat's "Manchester City" resolved to the same club_id as football-data's
    # "Man City" -- the cross-source alias difference this fixture set exists to cover.
    xg_coverage = conn.execute("SELECT count(*) FROM stg_matches WHERE home_xg IS NULL").fetchone()
    assert xg_coverage[0] == 0

    current_gw = conn.execute("SELECT gameweek FROM stg_gameweeks WHERE is_current").fetchone()
    assert current_gw[0] == 3

    # validate
    results = run_validations(conn, raise_on_failure=True)
    assert all(r.ok for r in results), [r for r in results if not r.ok]

    # evaluate (a tiny walk-forward, not the full production window, to stay fast)
    matches = conn.execute(
        "SELECT match_id, season, date, home_club_id, away_club_id, home_goals, away_goals, "
        "home_xg, away_xg, result, benchmark_home_odds, benchmark_draw_odds, "
        "benchmark_away_odds, benchmark_source FROM stg_matches ORDER BY date"
    ).pl()
    conn.close()

    report = build_report(
        matches, {"poisson": PoissonModel}, min_train_matches=4, season_level=False
    )
    assert report["primary"][0]["n"] > 0
    assert report["shipped_model"] == "poisson"
