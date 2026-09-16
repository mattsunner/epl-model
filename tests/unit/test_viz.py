from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from plforecast.artifacts.schema import (
    ClubCurrent,
    ClubForecast,
    ClubProjection,
    ForecastDocument,
    Provenance,
    Quantiles,
)
from plforecast.viz import (
    calibration_chart,
    forecast_evolution,
    lever_comparison,
    points_ridgeline,
    position_matrix,
    scoreline_heatmap,
)

CLUBS = ["a", "b", "c"]
NAMES = {"a": "Alpha", "b": "Beta", "c": "Gamma"}


# ---- position_matrix ----


def test_position_matrix_orders_clubs_by_expected_position_best_first():
    # club "b" is nearly always 1st, "a" nearly always 3rd, "c" always 2nd.
    pmf = np.array(
        [
            [0.05, 0.05, 0.9],  # a: mostly 3rd
            [0.9, 0.05, 0.05],  # b: mostly 1st
            [0.05, 0.9, 0.05],  # c: mostly 2nd
        ]
    )
    fig = position_matrix(CLUBS, pmf, NAMES)

    assert fig.data[0].z.shape == (3, 3)
    assert list(fig.data[0].y) == ["Beta", "Gamma", "Alpha"]
    assert list(fig.data[0].x) == ["1", "2", "3"]


def test_position_matrix_falls_back_to_club_id_without_display_names():
    pmf = np.eye(3)
    fig = position_matrix(CLUBS, pmf)
    assert set(fig.data[0].y) == set(CLUBS)


# ---- points_ridgeline ----


def test_points_ridgeline_has_one_trace_per_club_ordered_by_mean_descending():
    rng = np.random.default_rng(0)
    points = np.column_stack(
        [rng.normal(50, 5, 500), rng.normal(70, 5, 500), rng.normal(60, 5, 500)]
    )  # a=50, b=70, c=60

    fig = points_ridgeline(CLUBS, points, NAMES)

    assert len(fig.data) == 3
    assert [trace.name for trace in fig.data] == ["Beta", "Gamma", "Alpha"]


# ---- forecast_evolution ----


def _projection(title: float, top_four: float, relegation: float) -> ClubProjection:
    q = Quantiles(mean=50, p10=40, p50=50, p90=60)
    return ClubProjection(
        points=q,
        position=q,
        position_pmf=[1.0],
        title=title,
        top_four=top_four,
        relegation=relegation,
    )


def _provenance() -> Provenance:
    return Provenance(
        git_sha="abc",
        git_dirty=False,
        model="poisson",
        model_config_hash="hash",
        simulations=100,
        random_seed=1,
        sources=[],
        package_versions={},
    )


def _write_forecast(path: Path, *, gameweek: int, club_id: str, title: float) -> None:
    doc = ForecastDocument(
        season="2026-27",
        as_of_gameweek=gameweek,
        generated_at=datetime(2026, 9, 16, tzinfo=UTC),
        provenance=_provenance(),
        clubs=[
            ClubForecast(
                club_id=club_id,
                display_name=club_id.title(),
                current=ClubCurrent(played=gameweek, points=gameweek * 3, gd=1),
                projected=_projection(title, top_four=0.5, relegation=0.0),
            )
        ],
    )
    path.write_text(doc.model_dump_json())


def test_forecast_evolution_reads_every_gw_file_in_gameweek_order(tmp_path: Path):
    season_dir = tmp_path / "2026-27"
    season_dir.mkdir()
    _write_forecast(season_dir / "forecast-gw04.json", gameweek=4, club_id="arsenal", title=0.4)
    _write_forecast(season_dir / "forecast-gw01.json", gameweek=1, club_id="arsenal", title=0.1)
    (season_dir / "forecast-latest.json").write_text(
        (season_dir / "forecast-gw04.json").read_text()
    )  # must be ignored, not double-counted

    fig = forecast_evolution(tmp_path, "2026-27", "arsenal")

    title_trace = next(t for t in fig.data if t.name == "Title")
    assert list(title_trace.x) == [1, 4]
    assert list(title_trace.y) == [0.1, 0.4]


def test_forecast_evolution_is_empty_for_a_club_never_in_any_document(tmp_path: Path):
    season_dir = tmp_path / "2026-27"
    season_dir.mkdir()
    _write_forecast(season_dir / "forecast-gw01.json", gameweek=1, club_id="arsenal", title=0.1)

    fig = forecast_evolution(tmp_path, "2026-27", "not-a-real-club")

    assert len(fig.data) == 0


# ---- calibration_chart ----


def _bucket(
    bucket_low: float, mean_predicted: float, empirical: float, outcome: str = "pooled"
) -> dict:
    return {
        "outcome": outcome,
        "bucket_low": bucket_low,
        "bucket_high": bucket_low + 0.1,
        "mean_predicted": mean_predicted,
        "empirical_frequency": empirical,
        "n": 100,
        "ci_low": empirical - 0.05,
        "ci_high": empirical + 0.05,
    }


def test_calibration_chart_plots_the_reference_line_and_pooled_points_by_default():
    rows = [
        _bucket(0.0, 0.05, 0.06),
        _bucket(0.1, 0.15, 0.12),
        _bucket(0.0, 0.05, 0.9, outcome="home"),
    ]

    fig = calibration_chart(rows)

    assert len(fig.data) == 2  # reference line + pooled points
    points_trace = fig.data[1]
    assert list(points_trace.x) == [0.05, 0.15]
    assert list(points_trace.y) == [0.06, 0.12]


def test_calibration_chart_can_select_a_single_outcome():
    rows = [_bucket(0.0, 0.05, 0.06, outcome="home"), _bucket(0.0, 0.05, 0.9, outcome="away")]

    fig = calibration_chart(rows, outcome="away")

    assert list(fig.data[1].y) == [0.9]


# ---- scoreline_heatmap ----


def test_scoreline_heatmap_crops_to_max_goals():
    matrix = np.random.default_rng(0).random((11, 11))
    matrix /= matrix.sum()

    fig = scoreline_heatmap(matrix, "Arsenal", "Chelsea", max_goals=3)

    assert fig.data[0].z.shape == (4, 4)
    assert list(fig.data[0].x) == [0, 1, 2, 3]


# ---- lever_comparison ----


def test_lever_comparison_plots_selection_and_report_series_for_tuning_grid_rows():
    rows = [
        {"value": 0.001, "select_rps": 0.20, "report_rps": 0.21},
        {"value": 0.002, "select_rps": 0.19, "report_rps": 0.20},
    ]

    fig = lever_comparison(rows, metric="rps")

    assert len(fig.data) == 2
    assert [t.name for t in fig.data] == ["Selection seasons", "Report seasons"]
    assert list(fig.data[0].x) == [0.001, 0.002]


def test_lever_comparison_plots_a_single_series_for_ad_hoc_rows():
    rows = [{"value": 0.5, "rps": 0.19}, {"value": 0.75, "rps": 0.18}]

    fig = lever_comparison(rows, metric="rps")

    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [0.19, 0.18]
