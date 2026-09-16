import json
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from scipy.stats import poisson

from plforecast.artifacts.schema import (
    ClubProjection,
    FixturesDocument,
    ForecastDocument,
    Provenance,
    Quantiles,
    export_json_schemas,
)
from plforecast.artifacts.writer import (
    build_fixtures_document,
    build_forecast_document,
    git_provenance,
    package_versions,
    snapshot_provenance,
    write_documents,
)
from plforecast.simulate.competition import CompetitionConfig
from plforecast.simulate.engine import simulate_season
from plforecast.simulate.tiebreak import PremierLeagueTiebreaks

CLUBS = ["a", "b", "c", "d"]


class _FixedPoissonModel:
    def scoreline_matrix(self, home, away, max_goals=10):
        goals = np.arange(max_goals + 1)
        m = np.outer(poisson.pmf(goals, 1.4), poisson.pmf(goals, 1.1))
        return m / m.sum()

    @property
    def config_hash(self):
        return "fixed"


def _provenance() -> Provenance:
    return Provenance(
        git_sha="abc123",
        git_dirty=False,
        model="fixed",
        model_config_hash="fixed",
        simulations=300,
        random_seed=1,
        sources=[],
        package_versions={"numpy": "x"},
    )


def _fixtures():
    rows = [(h, a) for h in CLUBS for a in CLUBS if h != a]
    played = pl.DataFrame(
        [(h, a, 1, 0) for h, a in rows[:4]],
        schema=["home_club_id", "away_club_id", "home_goals", "away_goals"],
        orient="row",
    )
    remaining = pl.DataFrame(
        [
            (i, 2, datetime(2026, 9, 20, 15, 0, tzinfo=UTC), h, a)
            for i, (h, a) in enumerate(rows[4:], start=5)
        ],
        schema=["fixture_id", "gameweek", "kickoff_time", "home_club_id", "away_club_id"],
        orient="row",
    )
    return played, remaining


def _documents(seed: int = 1):
    played, remaining = _fixtures()
    competition = CompetitionConfig(n_clubs=4, relegation_spots=1, european_spots=1)
    model = _FixedPoissonModel()
    result = simulate_season(
        played,
        remaining,
        model,
        PremierLeagueTiebreaks(competition),
        n_simulations=300,
        max_goals=6,
        seed=seed,
    )
    generated_at = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
    forecast = build_forecast_document(
        result,
        played=played,
        display_names={"a": "Club A"},
        season="2026-27",
        as_of_gameweek=1,
        generated_at=generated_at,
        provenance=_provenance(),
        competition=competition,
    )
    fixtures = build_fixtures_document(
        remaining,
        model,
        season="2026-27",
        as_of_gameweek=1,
        generated_at=generated_at,
        provenance=_provenance(),
        max_goals=6,
        top_n_scorelines=3,
    )
    return forecast, fixtures


def test_forecast_document_is_a_valid_doubly_stochastic_table():
    forecast, _ = _documents()

    assert len(forecast.clubs) == 4
    pmf = np.array([c.projected.position_pmf for c in forecast.clubs])
    assert np.allclose(pmf.sum(axis=1), 1.0)
    assert np.allclose(pmf.sum(axis=0), 1.0)
    club_a = next(c for c in forecast.clubs if c.club_id == "a")
    assert club_a.display_name == "Club A"
    # rows[:4] = a-b, a-c, a-d, b-a, all 1-0 home wins: a played 4, won 3, lost 1.
    assert club_a.current.played == 4
    assert club_a.current.points == 9
    assert club_a.current.gd == 2
    assert sum(c.projected.title for c in forecast.clubs) == pytest.approx(1.0)
    assert sum(c.projected.relegation for c in forecast.clubs) == pytest.approx(1.0)
    # Clubs are ordered by mean projected position.
    means = [c.projected.position.mean for c in forecast.clubs]
    assert means == sorted(means)


def test_fixtures_document_carries_scoreable_detail_only():
    _, fixtures = _documents()

    assert len(fixtures.fixtures) == 8
    first = fixtures.fixtures[0]
    assert first.fixture_id == "5"
    assert first.gameweek == 2
    assert first.outcome.home + first.outcome.draw + first.outcome.away == pytest.approx(1.0)
    assert first.expected_goals.home == pytest.approx(1.4, abs=0.05)
    assert len(first.top_scorelines) == 3
    assert first.top_scorelines[0].p >= first.top_scorelines[1].p


def test_documents_are_deterministic_given_seed():
    a, _ = _documents(seed=7)
    b, _ = _documents(seed=7)
    assert a.model_dump_json() == b.model_dump_json()


def test_write_documents_emits_gameweek_and_latest_copies(tmp_path):
    forecast, fixtures = _documents()

    written = write_documents(forecast, fixtures, artifacts_dir=tmp_path)

    names = sorted(p.name for p in written)
    assert names == [
        "fixtures-gw01.json",
        "fixtures-latest.json",
        "forecast-gw01.json",
        "forecast-latest.json",
    ]
    assert (tmp_path / "2026-27" / "forecast-gw01.json").read_text() == (
        tmp_path / "2026-27" / "forecast-latest.json"
    ).read_text()
    reloaded = ForecastDocument.model_validate_json(
        (tmp_path / "2026-27" / "forecast-latest.json").read_text()
    )
    assert reloaded == forecast
    FixturesDocument.model_validate_json(
        (tmp_path / "2026-27" / "fixtures-latest.json").read_text()
    )


def test_schema_rejects_a_pmf_that_is_not_a_distribution():
    q = Quantiles(mean=1, p10=1, p50=1, p90=1)
    with pytest.raises(ValidationError, match="sums to"):
        ClubProjection(
            points=q, position=q, position_pmf=[0.5, 0.4], title=0.5, top_four=1, relegation=0
        )


def test_export_json_schemas_writes_both_files(tmp_path):
    written = export_json_schemas(tmp_path)
    assert sorted(p.name for p in written) == ["fixtures-v1.schema.json", "forecast-v1.schema.json"]
    schema = json.loads((tmp_path / "forecast-v1.schema.json").read_text())
    assert "position_pmf" in json.dumps(schema)


def test_snapshot_provenance_reads_latest_meta_per_source(tmp_path):
    for stamp, rows in (("20260101T000000Z", 5), ("20260108T000000Z", 7)):
        d = tmp_path / "football-data" / stamp
        d.mkdir(parents=True)
        (d / "_meta.json").write_text(
            json.dumps(
                {
                    "source": "football-data",
                    "fetched_at": "2026-01-08T00:00:00+00:00",
                    "row_count": rows,
                    "content_hash": stamp,
                }
            )
        )
    (tmp_path / "cache").mkdir()  # not a source dir; must be ignored

    sources = snapshot_provenance(tmp_path)

    assert [(s.name, s.rows, s.content_hash) for s in sources] == [
        ("football-data", 7, "20260108T000000Z")
    ]


def test_git_and_package_provenance_never_raise():
    sha, dirty = git_provenance()
    assert isinstance(sha, str) and isinstance(dirty, bool)
    versions = package_versions(("numpy", "definitely-not-a-package"))
    assert versions["definitely-not-a-package"] == "unknown"
    assert versions["numpy"] != "unknown"
