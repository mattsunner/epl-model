from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from plforecast.logging import configure_logging
from plforecast.storage.db import connect, migrate

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    configure_logging()


@app.command()
def ingest(source: str) -> None:
    """Fetch and land a raw snapshot from one source."""
    if source == "football-data":
        from plforecast.ingest.footballdata import ingest as ingest_footballdata

        ingest_footballdata()
    elif source == "fpl":
        from plforecast.ingest.fpl import ingest as ingest_fpl

        ingest_fpl()
    elif source == "understat":
        from plforecast.ingest.understat import ingest as ingest_understat

        ingest_understat()
    else:
        raise typer.BadParameter(f"unknown source: {source!r}")


@app.command()
def db_migrate() -> None:
    """Apply any pending DuckDB migrations."""
    conn = connect()
    migrate(conn)
    conn.close()


@app.command()
def curate() -> None:
    """Materialise stg_*/mart_* tables from the raw_* views."""
    from plforecast.storage.curate import curate_all

    conn = connect()
    curate_all(conn)
    conn.close()


@app.command()
def evaluate(
    out_dir: Annotated[
        Path, typer.Option(help="Where metrics.json and calibration.csv are written.")
    ] = Path("docs/evaluation"),
    xi: Annotated[
        float, typer.Option(help="Dixon-Coles decay rate (story C-03 tunes this).")
    ] = 0.0018,
    min_train_matches: Annotated[
        int, typer.Option(help="Warm-up before the first test split.")
    ] = 100,
) -> None:
    """Walk-forward backtest Poisson and Dixon-Coles against the market baseline over
    stg_matches (design.md section 8), scoring everything on identical rows. Prints the
    primary table and writes the full report; docs/evaluation.md is rendered from it."""
    import structlog

    from plforecast.evaluate.report import build_report, format_table, write_report
    from plforecast.models.dixon_coles import DixonColesModel
    from plforecast.models.poisson import PoissonModel

    log = structlog.get_logger()
    conn = connect()
    matches = conn.execute(
        "SELECT match_id, season, date, home_club_id, away_club_id, home_goals, "
        "away_goals, result, benchmark_home_odds, benchmark_draw_odds, benchmark_away_odds, "
        "benchmark_source "
        "FROM stg_matches WHERE season != (SELECT max(season) FROM stg_matches) "
        "ORDER BY date"
    ).pl()
    conn.close()

    log.info("evaluate.window", seasons=matches["season"].n_unique(), matches=matches.height)

    report = build_report(
        matches,
        {"poisson": PoissonModel, "dixon-coles": lambda: DixonColesModel(xi=xi)},
        min_train_matches=min_train_matches,
    )
    written = write_report(report, out_dir)

    typer.echo("Primary (identical rows for every model and the benchmark):")
    typer.echo(format_table(report["primary"]))
    typer.echo("")
    typer.echo("Secondary (each on every row it covered):")
    typer.echo(format_table(report["secondary_full_set"]))
    typer.echo("")
    typer.echo(f"coverage: {json.dumps(report['coverage'], default=str)}")
    written.append(_render_evaluation(out_dir))
    typer.echo(f"written: {', '.join(str(path) for path in written)}")


def _render_evaluation(report_dir: Path, target: Path = Path("docs/evaluation.md")) -> Path:
    from plforecast.evaluate.render import render_markdown

    report = json.loads((report_dir / "metrics.json").read_text())
    target.write_text(render_markdown(report))
    return target


@app.command()
def render_evaluation(
    report_dir: Annotated[Path, typer.Option(help="Directory holding metrics.json.")] = Path(
        "docs/evaluation"
    ),
    target: Annotated[Path, typer.Option(help="Markdown file to write.")] = Path(
        "docs/evaluation.md"
    ),
) -> None:
    """Re-render docs/evaluation.md from an existing metrics.json without re-running the
    backtest."""
    typer.echo(f"written: {_render_evaluation(report_dir, target)}")


@app.command()
def forecast(
    model: Annotated[str, typer.Option(help="poisson or dixon-coles")] = "dixon-coles",
    xi: Annotated[float, typer.Option(help="Dixon-Coles decay rate.")] = 0.0018,
    simulations: Annotated[int, typer.Option(help="Simulated seasons.")] = 50_000,
    seed: Annotated[int, typer.Option(help="Random seed, recorded in the artifact.")] = 20262027,
    artifacts_dir: Annotated[Path, typer.Option(help="Where the documents go.")] = Path(
        "artifacts"
    ),
    max_goals: Annotated[int, typer.Option(help="Scoreline matrix size.")] = 10,
) -> None:
    """Fit the match model on every curated result, simulate the rest of the current
    season, and write the forecast and fixtures documents (design.md section 9).

    Played matches for the current season come from the FPL fixture list, which
    updates within hours of a result; football-data.co.uk lags by up to a week, so the
    training set is stg_matches plus any FPL-finished fixture not yet in it."""
    import structlog

    from plforecast.artifacts.schema import Provenance, export_json_schemas
    from plforecast.artifacts.writer import (
        build_fixtures_document,
        build_forecast_document,
        git_provenance,
        now_utc,
        package_versions,
        snapshot_provenance,
        write_documents,
    )
    from plforecast.config import settings
    from plforecast.entities.clubs import load_club_dimension
    from plforecast.models.base import MatchModel, UnknownClubError
    from plforecast.models.dixon_coles import DixonColesModel
    from plforecast.models.poisson import PoissonModel
    from plforecast.simulate.competition import PREMIER_LEAGUE
    from plforecast.simulate.engine import simulate_season
    from plforecast.simulate.tiebreak import PremierLeagueTiebreaks

    log = structlog.get_logger()
    factories: dict[str, Callable[[], MatchModel]] = {
        "poisson": PoissonModel,
        "dixon-coles": lambda: DixonColesModel(xi=xi),
    }
    if model not in factories:
        raise typer.BadParameter(f"unknown model {model!r}; choose from {sorted(factories)}")

    conn = connect()
    fixtures = conn.execute(
        "SELECT fixture_id, season, gameweek, kickoff_time, home_club_id, away_club_id, "
        "home_goals, away_goals, finished FROM mart_fixtures ORDER BY kickoff_time, fixture_id"
    ).pl()
    history = conn.execute(
        "SELECT season, date, home_club_id, away_club_id, home_goals, away_goals "
        "FROM stg_matches ORDER BY date"
    ).pl()
    conn.close()

    season_label_ = str(fixtures["season"][0])
    played = fixtures.filter(pl.col("finished"))
    remaining = fixtures.filter(~pl.col("finished"))

    # Training data: history plus current-season results FPL has that football-data
    # has not published yet (story B-07: FPL is authoritative for the live season).
    fpl_results = played.select(
        pl.lit(season_label_).alias("season"),
        pl.col("kickoff_time").dt.date().alias("date"),
        "home_club_id",
        "away_club_id",
        "home_goals",
        "away_goals",
    )
    missing_from_history = fpl_results.join(
        history, on=["season", "home_club_id", "away_club_id"], how="anti"
    )
    training = pl.concat([history, missing_from_history]).sort("date")
    log.info(
        "forecast.training",
        rows=training.height,
        fpl_only_results=missing_from_history.height,
        season=season_label_,
    )

    fitted = factories[model]().fit(training)
    try:
        for row in remaining.select("home_club_id", "away_club_id").unique().iter_rows():
            fitted.scoreline_matrix(row[0], row[1], max_goals)
    except UnknownClubError as exc:
        raise typer.Exit(code=1) from typer.BadParameter(
            f"{exc}. A club with no results at all cannot be rated yet (story C-08 wires "
            "the promoted-club prior in); re-run after its first match."
        )

    finished_by_gw = played.group_by("gameweek").len().rename({"len": "finished"})
    all_by_gw = fixtures.group_by("gameweek").len().rename({"len": "total"})
    complete_gws = (
        all_by_gw.join(finished_by_gw, on="gameweek", how="left")
        .fill_null(0)
        .filter(pl.col("finished") == pl.col("total"))
        .drop_nulls("gameweek")
    )
    as_of_gameweek = max((int(g) for g in complete_gws["gameweek"].to_list()), default=0)

    result = simulate_season(
        played.select("home_club_id", "away_club_id", "home_goals", "away_goals"),
        remaining.select("home_club_id", "away_club_id"),
        fitted,
        PremierLeagueTiebreaks(PREMIER_LEAGUE),
        n_simulations=simulations,
        max_goals=max_goals,
        seed=seed,
    )

    sha, dirty = git_provenance()
    provenance = Provenance(
        git_sha=sha,
        git_dirty=dirty,
        model=model,
        model_config_hash=fitted.config_hash,
        simulations=simulations,
        random_seed=seed,
        sources=snapshot_provenance(settings.raw_dir),
        package_versions=package_versions(),
    )
    generated_at = now_utc()
    dimension = load_club_dimension()
    display_names = dict(
        zip(
            dimension.frame["club_id"].to_list(),
            dimension.frame["display_name"].to_list(),
            strict=True,
        )
    )
    season_slug = season_label_.replace("/", "-")
    forecast_doc = build_forecast_document(
        result,
        played=played,
        display_names=display_names,
        season=season_slug,
        as_of_gameweek=as_of_gameweek,
        generated_at=generated_at,
        provenance=provenance,
    )
    fixtures_doc = build_fixtures_document(
        remaining,
        fitted,
        season=season_slug,
        as_of_gameweek=as_of_gameweek,
        generated_at=generated_at,
        provenance=provenance,
        max_goals=max_goals,
    )
    written = write_documents(forecast_doc, fixtures_doc, artifacts_dir=artifacts_dir)
    written += export_json_schemas(artifacts_dir / "schema")

    typer.echo(
        f"{season_label_} after gameweek {as_of_gameweek}: {played.height} played, "
        f"{remaining.height} remaining, {simulations:,} simulations, model={model}"
    )
    header = ("club", "pld", "pts", "E[pts]", "title", "top4", "rel")
    typer.echo(
        f"{header[0]:24s} {header[1]:>3s} {header[2]:>3s} {header[3]:>7s} "
        f"{header[4]:>6s} {header[5]:>6s} {header[6]:>6s}"
    )
    for club in forecast_doc.clubs:
        pr = club.projected
        typer.echo(
            f"{club.display_name:24s} {club.current.played:3d} {club.current.points:3d} "
            f"{pr.points.mean:7.1f} {pr.title:6.1%} {pr.top_four:6.1%} {pr.relegation:6.1%}"
        )
    typer.echo(f"written: {', '.join(str(path) for path in written)}")


@app.command()
def validate_artifacts(
    artifacts_dir: Annotated[Path, typer.Option(help="Directory of committed forecasts.")] = Path(
        "artifacts"
    ),
) -> None:
    """Validate every committed forecast and fixtures document against its schema
    (design.md 9.4). Exit code 1 on the first invalid file."""
    from pydantic import ValidationError

    from plforecast.artifacts.schema import FixturesDocument, ForecastDocument

    checked = 0
    season_dirs = sorted(
        p for p in artifacts_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]") if p.is_dir()
    )
    for path in sorted(path for d in season_dirs for path in d.glob("*.json")):
        model = ForecastDocument if path.name.startswith("forecast-") else FixturesDocument
        try:
            model.model_validate_json(path.read_text())
        except ValidationError as exc:
            typer.echo(f"INVALID {path}: {exc}", err=True)
            raise typer.Exit(code=1) from None
        checked += 1
    typer.echo(f"validated {checked} artifact document(s) under {artifacts_dir}")


if __name__ == "__main__":
    app()
