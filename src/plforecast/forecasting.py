"""The live-forecast pipeline as an importable function (story: notebook workbench).

`fit_and_simulate()` is everything between "read the curated tables" and "build the
forecast artifact documents": assembling the current season's training data (history
plus any FPL result football-data.co.uk hasn't published yet, story B-07), the
promoted-club prior gate and injection (ADR 0006, story C-08), fitting the chosen
model, and running the season simulation. `cli.py`'s `forecast` command calls this and
handles only its own concerns (building the artifact documents, writing files,
printing the table) -- the same thin-wrapper-over-an-importable-function shape every
other CLI command already has. A notebook exploring lever settings imports this
directly rather than re-deriving any of the above, per ADR 0003's "notebooks consume
`plforecast`, never define logic."
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import structlog

from plforecast.config import Settings, settings
from plforecast.entities.clubs import load_club_dimension
from plforecast.features.priors import (
    PRIOR_GATE_XI,
    build_prior,
    build_survival_zone_reference,
    needs_prior,
    prior_pseudo_matches,
)
from plforecast.models.base import MatchModel, UnknownClubError
from plforecast.models.dixon_coles import DixonColesModel
from plforecast.models.poisson import PoissonModel
from plforecast.models.xg_rates import XGRateModel
from plforecast.simulate.competition import PREMIER_LEAGUE
from plforecast.simulate.engine import SimulationResult, simulate_season
from plforecast.simulate.tiebreak import PremierLeagueTiebreaks
from plforecast.storage.db import connect

log = structlog.get_logger()

DEFAULT_METRICS_PATH = Path("docs/evaluation/metrics.json")


def resolve_model_name(model_name: str, *, metrics_path: Path = DEFAULT_METRICS_PATH) -> str:
    """'shipped' -> whichever model `docs/evaluation/metrics.json` names as shipped,
    falling back to 'dixon-coles' if no evaluation has been run yet. Any other name
    passes through unchanged."""
    if model_name != "shipped":
        return model_name
    if metrics_path.exists():
        return str(json.loads(metrics_path.read_text()).get("shipped_model", "dixon-coles"))
    return "dixon-coles"


def build_model_factory(
    model_name: str,
    *,
    xi: float | None = None,
    blend: float | None = None,
    rho: float | None = None,
) -> Callable[[], MatchModel]:
    """A zero-arg factory for `model_name`, with `xi`/`blend`/`rho` overriding
    `config.py`'s tuned defaults when given (the lever-tweaking entry point: pass a
    value to see the model with that lever changed, leave it None to use the shipped
    default). `blend` and `rho` are only meaningful for `xg-rates`; a value given for
    a model that doesn't use them is silently ignored, matching how a caller who only
    wants to vary `xi` shouldn't have to know which models take the other two."""
    if model_name == "poisson":
        return PoissonModel
    if model_name == "dixon-coles":
        resolved_xi = settings.dixon_coles_xi if xi is None else xi
        return lambda: DixonColesModel(xi=resolved_xi)
    if model_name == "xg-rates":
        resolved_xi = settings.xg_rates_xi if xi is None else xi
        resolved_blend = settings.xg_rates_blend if blend is None else blend
        resolved_rho = settings.xg_rates_rho if rho is None else rho
        return lambda: XGRateModel(xi=resolved_xi, blend=resolved_blend, rho=resolved_rho)
    raise ValueError(f"unknown model {model_name!r}; choose from poisson, dixon-coles, xg-rates")


@dataclass(frozen=True, slots=True)
class ForecastRun:
    """Everything a caller needs to either build the forecast artifact documents
    (`cli.py forecast`) or visualize the result directly (the notebook workbench)."""

    model_name: str
    fitted: MatchModel
    result: SimulationResult
    played: pl.DataFrame
    remaining: pl.DataFrame
    fixtures: pl.DataFrame
    display_names: dict[str, str]
    season_label: str
    as_of_gameweek: int
    training: pl.DataFrame
    promoted_clubs: list[str]


def _current_gameweek(conn) -> int | None:  # type: ignore[no-untyped-def]
    """FPL's own current gameweek (story B-10), when `stg_gameweeks` exists and has a
    row flagged current; None on a database that predates `ingest fpl` landing it."""
    has_gameweeks = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'stg_gameweeks'"
    ).fetchone()
    if not (has_gameweeks and has_gameweeks[0]):
        return None
    current_rows = conn.execute("SELECT gameweek FROM stg_gameweeks WHERE is_current").fetchall()
    return int(current_rows[0][0]) if current_rows else None


def _fallback_gameweek(fixtures: pl.DataFrame, played: pl.DataFrame) -> int:
    """The last gameweek every one of whose fixtures is finished -- used only when
    `stg_gameweeks` is absent."""
    finished_by_gw = played.group_by("gameweek").len().rename({"len": "finished"})
    all_by_gw = fixtures.group_by("gameweek").len().rename({"len": "total"})
    complete_gws = (
        all_by_gw.join(finished_by_gw, on="gameweek", how="left")
        .fill_null(0)
        .filter(pl.col("finished") == pl.col("total"))
        .drop_nulls("gameweek")
    )
    return max((int(g) for g in complete_gws["gameweek"].to_list()), default=0)


def _assemble_training_data(
    history: pl.DataFrame, fixtures: pl.DataFrame, played: pl.DataFrame, season_label: str
) -> tuple[pl.DataFrame, int]:
    """History plus current-season results FPL has that football-data.co.uk hasn't
    published yet (story B-07: FPL is authoritative for the live season). Returns the
    combined frame and the count of FPL-only rows it added, for logging."""
    fpl_results = played.select(
        (
            pl.lit(season_label.replace("/", "-") + "-")
            + pl.col("home_club_id")
            + "-"
            + pl.col("away_club_id")
        ).alias("match_id"),
        pl.lit(season_label).alias("season"),
        pl.col("kickoff_time").dt.date().alias("date"),
        "home_club_id",
        "away_club_id",
        "home_goals",
        "away_goals",
        pl.lit(None, dtype=pl.Float64).alias("home_xg"),  # Understat lags FPL by a day
        pl.lit(None, dtype=pl.Float64).alias("away_xg"),
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
        .alias("result"),
    )
    missing_from_history = fpl_results.join(
        history, on=["season", "home_club_id", "away_club_id"], how="anti"
    )
    training = pl.concat([history, missing_from_history]).sort("date")
    return training, missing_from_history.height


def _inject_promoted_club_prior(
    training: pl.DataFrame,
    fixtures: pl.DataFrame,
    *,
    season_label: str,
    as_of: date,
    seed: int,
) -> tuple[pl.DataFrame, list[str]]:
    """The promoted-club prior (ADR 0006, story C-08): a club in the fixture list with
    less than half a season of decay-weighted evidence gets pseudo-observations built
    from the survival-zone reference, appended to `training`. Returns the (possibly
    unchanged) training frame and the list of clubs the prior covered."""
    completed = training.filter(pl.col("season") != season_label)
    fixture_clubs = sorted(set(fixtures["home_club_id"]) | set(fixtures["away_club_id"]))
    needing_prior = [
        club for club in fixture_clubs if needs_prior(club, training, as_of=as_of, xi=PRIOR_GATE_XI)
    ]
    if not needing_prior:
        return training, []

    reference = build_survival_zone_reference(completed)
    pseudo = [
        prior_pseudo_matches(
            build_prior(club, reference),
            opponents=fixture_clubs,
            season=season_label,
            as_of=as_of,
            seed=seed,
        )
        for club in needing_prior
    ]
    training = pl.concat([training, *pseudo], how="vertical_relaxed").sort("date")
    log.info(
        "forecasting.promoted_club_prior",
        clubs=needing_prior,
        pseudo_matches=sum(p.height for p in pseudo),
        reference_observations=reference.n_observations,
        reference_mean_points=round(reference.mean_points, 1),
    )
    return training, needing_prior


def fit_and_simulate(
    model_name: str = "shipped",
    *,
    xi: float | None = None,
    blend: float | None = None,
    rho: float | None = None,
    simulations: int = 50_000,
    seed: int = 20262027,
    max_goals: int = 10,
    as_of: date | None = None,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    config: Settings = settings,
) -> ForecastRun:
    """Fit `model_name` (or 'shipped', resolved via `metrics_path`) on the current
    season's training data and simulate the rest of the season. `xi`/`blend`/`rho`
    override the tuned defaults for the lever it applies to; `as_of` (default: today)
    is the moment the promoted-club prior gate evaluates evidence as of. `config`
    (story A-14's injection convention) lets a test point `connect()` at a temporary
    database instead of the module-global default."""
    resolved_name = resolve_model_name(model_name, metrics_path=metrics_path)
    factory = build_model_factory(resolved_name, xi=xi, blend=blend, rho=rho)
    as_of = as_of or date.today()

    conn = connect(config)
    fixtures = conn.execute(
        "SELECT fixture_id, season, gameweek, kickoff_time, home_club_id, away_club_id, "
        "home_goals, away_goals, finished FROM stg_fixtures ORDER BY kickoff_time, fixture_id"
    ).pl()
    history = conn.execute(
        "SELECT match_id, season, date, home_club_id, away_club_id, home_goals, away_goals, "
        "home_xg, away_xg, result FROM stg_matches ORDER BY date"
    ).pl()
    fpl_current_gameweek = _current_gameweek(conn)
    conn.close()

    season_label = str(fixtures["season"][0])
    played = fixtures.filter(pl.col("finished"))
    remaining = fixtures.filter(~pl.col("finished"))

    training, fpl_only_count = _assemble_training_data(history, fixtures, played, season_label)
    training, promoted_clubs = _inject_promoted_club_prior(
        training, fixtures, season_label=season_label, as_of=as_of, seed=seed
    )
    log.info(
        "forecasting.training",
        rows=training.height,
        fpl_only_results=fpl_only_count,
        season=season_label,
    )

    fitted = factory().fit(training)
    try:
        for row in remaining.select("home_club_id", "away_club_id").unique().iter_rows():
            fitted.scoreline_matrix(row[0], row[1], max_goals)
    except UnknownClubError as exc:
        raise ValueError(
            f"{exc}. The promoted-club prior should have covered this club; check "
            "needs_prior() and the fixture list."
        ) from exc

    as_of_gameweek = (
        fpl_current_gameweek
        if fpl_current_gameweek is not None
        else _fallback_gameweek(fixtures, played)
    )

    result = simulate_season(
        played.select("home_club_id", "away_club_id", "home_goals", "away_goals"),
        remaining.select("home_club_id", "away_club_id"),
        fitted,
        PremierLeagueTiebreaks(PREMIER_LEAGUE),
        n_simulations=simulations,
        max_goals=max_goals,
        seed=seed,
    )

    dimension = load_club_dimension()
    display_names = dict(
        zip(
            dimension.frame["club_id"].to_list(),
            dimension.frame["display_name"].to_list(),
            strict=True,
        )
    )

    return ForecastRun(
        model_name=resolved_name,
        fitted=fitted,
        result=result,
        played=played,
        remaining=remaining,
        fixtures=fixtures,
        display_names=display_names,
        season_label=season_label,
        as_of_gameweek=as_of_gameweek,
        training=training,
        promoted_clubs=promoted_clubs,
    )
