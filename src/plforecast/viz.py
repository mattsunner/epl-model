"""Chart-building functions for exploring a forecast (notebook workbench). Each
function takes plain arrays/frames the rest of the package already produces --
`SimulationResult`, a loaded `ForecastDocument`, `evaluate/calibration.py`'s bucket
rows, `evaluate/tuning.py`'s grid rows, a `MatchModel.scoreline_matrix()` -- and
returns a `plotly.graph_objects.Figure`. Nothing here computes a model output; it only
shapes one for display. Kept in `src/plforecast` rather than inline in a notebook per
ADR 0003 ("notebooks consume `plforecast`, never define logic"), so the shaping is
tested and reusable if the published site's own charts are ever built from it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from plforecast.artifacts.schema import ForecastDocument

OUTCOME_COLORS = {"home": "#2b6cb0", "draw": "#a0aec0", "away": "#c05621"}


def _label(club_id: str, display_names: Mapping[str, str]) -> str:
    return display_names.get(club_id, club_id)


def _order_by(values: Sequence[float] | np.ndarray, *, descending: bool = False) -> np.ndarray:
    """Indices that sort `values` ascending (or descending), stable on ties."""
    order: np.ndarray = np.argsort(np.asarray(values), kind="stable")
    return order[::-1] if descending else order


def position_matrix(
    club_ids: Sequence[str],
    position_pmf: np.ndarray,
    display_names: Mapping[str, str] | None = None,
) -> go.Figure:
    """A club-by-position heatmap: row i, column p is `P(club i finishes position
    p+1)`. `position_pmf` is `(n_clubs, n_clubs)`, exactly `SimulationResult
    .position_pmf` or a `ForecastDocument`'s per-club `position_pmf` lists stacked in
    the same club order as `club_ids`. Clubs are ordered by expected position (best
    first), the single best representation of the model's output per design.md 10.1."""
    display_names = display_names or {}
    n = len(club_ids)
    positions = np.arange(1, n + 1)
    expected_position = position_pmf @ positions
    order = _order_by(expected_position)

    labels = [_label(club_ids[i], display_names) for i in order]
    z = position_pmf[order]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[str(p) for p in positions],
            y=labels,
            colorscale="Blues",
            colorbar={"title": "Probability"},
            hovertemplate="%{y} finishes %{x}: %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Position probability matrix",
        xaxis_title="Final position",
        yaxis={"autorange": "reversed"},
        height=max(320, 22 * n),
    )
    return fig


def points_ridgeline(
    club_ids: Sequence[str],
    points: np.ndarray,
    display_names: Mapping[str, str] | None = None,
    *,
    bandwidth: Any = "scott",
) -> go.Figure:
    """A ridgeline of each club's projected-points density, ordered by mean (highest
    at the top). `points` is `(n_simulations, n_clubs)` -- the *full* distribution from
    a live `SimulationResult.points`, not the four-quantile summary the committed
    forecast artifact carries (design.md 9.2 deliberately does not publish enough to
    reconstruct this chart), so this only ever runs from a fresh simulation."""
    display_names = display_names or {}
    n_clubs = points.shape[1]
    means = points.mean(axis=0)
    order = _order_by(means, descending=True)  # highest mean drawn first, at the top

    grid = np.linspace(points.min() - 5, points.max() + 5, 200)
    row_height = 1.0
    fig = go.Figure()
    for rank, i in enumerate(order):
        kde: Any = gaussian_kde(points[:, i], bw_method=bandwidth)
        density = kde(grid)
        baseline = (n_clubs - 1 - rank) * row_height
        scaled = density / density.max() * row_height * 0.9
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=scaled + baseline,
                mode="lines",
                fill="tonexty" if rank == 0 else "tozeroy",
                fillcolor="rgba(43,108,176,0.5)",
                line={"color": "#2b6cb0", "width": 1},
                name=_label(club_ids[i], display_names),
                hovertemplate=f"{_label(club_ids[i], display_names)}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Projected points distribution",
        xaxis_title="Points",
        yaxis={
            "tickvals": [(n_clubs - 1 - r) * row_height for r in range(n_clubs)],
            "ticktext": [_label(club_ids[i], display_names) for i in order],
        },
        showlegend=False,
        height=max(320, 26 * n_clubs),
    )
    return fig


def forecast_evolution(artifacts_dir: Path, season: str, club_id: str) -> go.Figure:
    """Title/top-four/relegation probability for `club_id` across every committed,
    gameweek-stamped forecast document for `season` (design.md 10.1's third chart).
    Reads `forecast-gwNN.json` files directly (never `-latest.json`, which is
    overwritten in place and carries no history) via the existing pydantic schema."""
    season_dir = Path(artifacts_dir) / season
    points: list[tuple[int, float, float, float]] = []
    for path in sorted(season_dir.glob("forecast-gw*.json")):
        doc = ForecastDocument.model_validate_json(path.read_text())
        club = next((c for c in doc.clubs if c.club_id == club_id), None)
        if club is None:
            continue
        points.append(
            (
                doc.as_of_gameweek,
                club.projected.title,
                club.projected.top_four,
                club.projected.relegation,
            )
        )
    points.sort(key=lambda row: row[0])

    fig = go.Figure()
    if points:
        gws, title, top4, rel = zip(*points, strict=True)
        for name, series, color in (
            ("Title", title, "#d69e2e"),
            ("Top four", top4, "#2b6cb0"),
            ("Relegation", rel, "#c53030"),
        ):
            fig.add_trace(
                go.Scatter(x=gws, y=series, mode="lines+markers", name=name, line={"color": color})
            )
    fig.update_layout(
        title=f"Forecast evolution: {club_id}",
        xaxis_title="Gameweek",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
    )
    return fig


def calibration_chart(
    calibration_rows: Sequence[Mapping[str, Any]], *, outcome: str = "pooled"
) -> go.Figure:
    """A reliability diagram from `evaluate/calibration.py`'s bucket rows (as written
    to `docs/evaluation/metrics.json["calibration"][model]`): mean predicted
    probability vs. empirical frequency, with the Wilson-interval error bars already
    computed, against the y=x line a perfectly calibrated model would sit on."""
    rows = [r for r in calibration_rows if r.get("outcome", "pooled") == outcome]
    rows = sorted(rows, key=lambda r: r["bucket_low"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#a0aec0"},
            name="Perfect calibration",
        )
    )
    if rows:
        x = [r["mean_predicted"] for r in rows]
        y = [r["empirical_frequency"] for r in rows]
        err_high = [r["ci_high"] - r["empirical_frequency"] for r in rows]
        err_low = [r["empirical_frequency"] - r["ci_low"] for r in rows]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={"size": [max(6, min(24, r["n"] ** 0.4)) for r in rows], "color": "#2b6cb0"},
                error_y={"type": "data", "array": err_high, "arrayminus": err_low},
                name=f"{outcome} calibration",
                hovertemplate="predicted %{x:.2f}, actual %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Calibration ({outcome})",
        xaxis_title="Mean predicted probability",
        yaxis_title="Empirical frequency",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    return fig


def scoreline_heatmap(
    matrix: np.ndarray, home_label: str, away_label: str, *, max_goals: int = 6
) -> go.Figure:
    """A single fixture's `scoreline_matrix()` output as a small heatmap, cropped to
    `max_goals` a side for readability. The most direct "change a lever, see the model
    change" visual: refit with a different `xi`/`blend`/`rho` and re-render."""
    cropped = matrix[: max_goals + 1, : max_goals + 1]
    goals = list(range(cropped.shape[0]))
    fig = go.Figure(
        data=go.Heatmap(
            z=cropped,
            x=goals,
            y=goals,
            colorscale="Purples",
            colorbar={"title": "Probability"},
            text=[[f"{p:.1%}" for p in row] for row in cropped],
            texttemplate="%{text}",
            hovertemplate=f"{home_label} %{{y}} - %{{x}} {away_label}: %{{z:.1%}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{home_label} vs {away_label}",
        xaxis_title=f"{away_label} goals",
        yaxis_title=f"{home_label} goals",
    )
    return fig


def lever_comparison(
    rows: Sequence[Mapping[str, Any]], *, metric: str = "rps", x_key: str = "value"
) -> go.Figure:
    """A line chart of `metric` against a swept lever's values. Accepts either
    `evaluate/tuning.py`'s grid rows directly (each with `select_{metric}` and
    `report_{metric}` -- both are plotted, so overfitting to the selection seasons is
    visible) or a plain list of `{x_key: ..., metric: ...}` dicts for an ad hoc
    hand-picked comparison."""
    rows = sorted(rows, key=lambda r: r[x_key])
    x = [r[x_key] for r in rows]

    fig = go.Figure()
    select_key, report_key = f"select_{metric}", f"report_{metric}"
    if rows and select_key in rows[0]:
        fig.add_trace(
            go.Scatter(
                x=x, y=[r[select_key] for r in rows], mode="lines+markers", name="Selection seasons"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=[r[report_key] for r in rows], mode="lines+markers", name="Report seasons"
            )
        )
    else:
        fig.add_trace(
            go.Scatter(x=x, y=[r[metric] for r in rows], mode="lines+markers", name=metric)
        )
    fig.update_layout(title=f"{metric.upper()} vs {x_key}", xaxis_title=x_key, yaxis_title=metric)
    return fig
