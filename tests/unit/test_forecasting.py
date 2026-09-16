import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from plforecast.config import Settings
from plforecast.forecasting import (
    build_model_factory,
    fit_and_simulate,
    resolve_model_name,
)
from plforecast.storage.db import connect

# Synthetic club ids, not real ones: fit_and_simulate reads already-curated stg_*
# tables directly, after entity resolution has already happened, so nothing here
# needs to resolve against the real club_aliases.yaml (unlike
# tests/integration/test_pipeline.py, which exercises curate itself and does need
# real, resolvable names).
HISTORY_CLUBS = [f"club-{i:02d}" for i in range(19)]  # one full season's round robin
PROMOTED = "club-promoted"  # in the current fixture list only, never in stg_matches
CURRENT_SEASON_CLUBS = [*HISTORY_CLUBS, PROMOTED]


def _round_robin(clubs: list[str]) -> list[tuple[str, str]]:
    return [(h, a) for h in clubs for a in clubs if h != a]


def _seed(conn, table: str, df: pl.DataFrame) -> None:
    conn.register("_seed", df.to_arrow())
    conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _seed")
    conn.unregister("_seed")


def _seed_curated_db(
    config: Settings,
    *,
    current_season_clubs: list[str] = CURRENT_SEASON_CLUBS,
    current_season: str = "2026/27",
) -> None:
    """One completed season's round robin among `HISTORY_CLUBS` in `stg_matches` (so
    every established club clears the promoted-club prior gate on its own -- a full
    season, 38 matches per club, comfortably above the 19-match threshold even
    decayed), a partial current-season round robin among `current_season_clubs` in
    `stg_fixtures`, and a small `stg_gameweeks` calendar -- exactly the three tables
    `fit_and_simulate` reads. `PROMOTED`, when included in `current_season_clubs`, has
    no rows in `stg_matches` at all, mirroring a genuinely blank promoted club."""
    conn = connect(config)

    match_rows = []
    start = date(2025, 8, 1)
    for i, (home, away) in enumerate(_round_robin(HISTORY_CLUBS)):
        d = start + timedelta(days=i // 4)  # a few matches per date, like real rounds
        match_rows.append(
            {
                "match_id": f"2025/26-{home}-{away}",
                "season": "2025/26",
                "date": d,
                "home_club_id": home,
                "away_club_id": away,
                "home_goals": 2,
                "away_goals": 1,
                "result": "H",
                "home_xg": 1.6,
                "away_xg": 0.9,
            }
        )
    _seed(conn, "stg_matches", pl.DataFrame(match_rows))

    fixture_rows = []
    fixture_start = datetime(2026, 8, 14, 15, 0, tzinfo=UTC)
    pairs = _round_robin(current_season_clubs)
    matches_per_round = len(current_season_clubs) // 2
    for i, (home, away) in enumerate(pairs):
        gw = i // matches_per_round + 1
        finished = gw <= 2
        fixture_rows.append(
            {
                "fixture_id": i + 1,
                "season": current_season,
                "gameweek": gw,
                "kickoff_time": fixture_start + timedelta(days=7 * (gw - 1)),
                "home_club_id": home,
                "away_club_id": away,
                "home_goals": 2 if finished else None,
                "away_goals": 1 if finished else None,
                "finished": finished,
            }
        )
    _seed(
        conn,
        "stg_fixtures",
        pl.DataFrame(
            fixture_rows, schema_overrides={"home_goals": pl.Int64, "away_goals": pl.Int64}
        ),
    )

    gw_rows = [
        {
            "gameweek": gw,
            "name": f"Gameweek {gw}",
            "deadline_time": fixture_start + timedelta(days=7 * (gw - 1)),
            "finished": gw <= 2,
            "is_previous": gw == 2,
            "is_current": gw == 3,
            "is_next": gw == 4,
        }
        for gw in range(1, 7)
    ]
    _seed(conn, "stg_gameweeks", pl.DataFrame(gw_rows))
    conn.close()


@pytest.fixture
def config(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


# ---- resolve_model_name ----


def test_resolve_model_name_falls_back_to_dixon_coles_when_no_metrics_file(tmp_path: Path):
    assert resolve_model_name("shipped", metrics_path=tmp_path / "missing.json") == "dixon-coles"


def test_resolve_model_name_reads_the_shipped_model_from_metrics(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"shipped_model": "xg-rates"}))
    assert resolve_model_name("shipped", metrics_path=metrics_path) == "xg-rates"


def test_resolve_model_name_passes_through_a_concrete_name():
    assert resolve_model_name("poisson") == "poisson"


# ---- build_model_factory ----


def test_build_model_factory_poisson_ignores_every_lever():
    factory = build_model_factory("poisson", xi=0.5, blend=0.5, rho=0.5)
    assert factory().config_hash  # constructs without error; no lever to check


def test_build_model_factory_dixon_coles_uses_the_override_xi():
    model = build_model_factory("dixon-coles", xi=0.01)()
    assert model.xi == 0.01


def test_build_model_factory_dixon_coles_falls_back_to_settings_default():
    from plforecast.config import settings

    model = build_model_factory("dixon-coles")()
    assert model.xi == settings.dixon_coles_xi


def test_build_model_factory_xg_rates_uses_every_override():
    model = build_model_factory("xg-rates", xi=0.01, blend=0.5, rho=-0.1)()
    assert (model.xi, model.blend, model.rho) == (0.01, 0.5, -0.1)


def test_build_model_factory_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        build_model_factory("not-a-real-model")


# ---- fit_and_simulate ----


def test_fit_and_simulate_happy_path(config: Settings):
    _seed_curated_db(config, current_season_clubs=HISTORY_CLUBS)  # no promoted club here

    run = fit_and_simulate(
        "poisson", simulations=200, seed=1, as_of=date(2026, 9, 1), config=config
    )

    assert run.model_name == "poisson"
    assert run.season_label == "2026/27"
    assert run.as_of_gameweek == 3  # from the seeded stg_gameweeks is_current row
    assert run.promoted_clubs == []  # every club has a full season of recent history
    assert set(run.result.club_ids) == set(HISTORY_CLUBS)
    n = len(HISTORY_CLUBS)
    assert run.result.position_pmf.shape == (n, n)
    assert run.result.n_simulations == 200
    # doubly-stochastic invariant (design.md 7.3), spot-checked here too
    assert (run.result.position_counts.sum(axis=1) == 200).all()
    assert (run.result.position_counts.sum(axis=0) == 200).all()


def test_fit_and_simulate_is_deterministic_given_a_seed(config: Settings):
    _seed_curated_db(config, current_season_clubs=HISTORY_CLUBS)
    kwargs = dict(simulations=200, seed=7, as_of=date(2026, 9, 1), config=config)

    first = fit_and_simulate("poisson", **kwargs)
    second = fit_and_simulate("poisson", **kwargs)

    assert (first.result.position_counts == second.result.position_counts).all()


def test_fit_and_simulate_injects_the_prior_for_a_club_with_no_history(config: Settings):
    """Regression for the bug found in this project's own build: a genuinely blank
    club (no rows in stg_matches at all) must be covered by the promoted-club prior,
    not left to raise an error, and every established club must be left alone."""
    _seed_curated_db(config)  # PROMOTED is in the current fixture list, not in history

    run = fit_and_simulate(
        "poisson", simulations=200, seed=1, as_of=date(2026, 9, 1), config=config
    )

    assert run.promoted_clubs == [PROMOTED]
    assert PROMOTED in run.result.club_ids
    assert set(run.result.club_ids) == set(CURRENT_SEASON_CLUBS)


def test_fit_and_simulate_falls_back_to_computed_gameweek_without_stg_gameweeks(config: Settings):
    _seed_curated_db(config, current_season_clubs=HISTORY_CLUBS)
    conn = connect(config)
    conn.execute("DROP TABLE stg_gameweeks")
    conn.close()

    run = fit_and_simulate("poisson", simulations=50, seed=1, as_of=date(2026, 9, 1), config=config)

    # Gameweeks 1-2 are seeded finished, 3+ are not: the fallback ("last gameweek
    # whose fixtures are all finished") must agree with stg_gameweeks' own answer.
    assert run.as_of_gameweek == 2
