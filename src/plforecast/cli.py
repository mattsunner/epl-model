from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from plforecast.logging import configure_logging
from plforecast.models.base import MatchModel
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
def curate(
    manifest_path: Annotated[
        Path, typer.Option(help="Where the lineage manifest is written (story B-11).")
    ] = Path("data/curate-manifest.json"),
) -> None:
    """Materialise dim_*/stg_*/mart_* tables from the raw_* views and write a manifest
    of row counts and raw-snapshot lineage next to the database."""
    import json

    from plforecast.storage.curate import curate_all

    conn = connect()
    manifest = curate_all(conn)
    conn.close()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    typer.echo(f"written: {manifest_path}")


@app.command()
def evaluate(
    out_dir: Annotated[
        Path, typer.Option(help="Where metrics.json and calibration.csv are written.")
    ] = Path("docs/evaluation"),
    min_train_matches: Annotated[
        int, typer.Option(help="Warm-up before the first test split.")
    ] = 100,
    season_level: Annotated[
        bool, typer.Option(help="Also run the season-level evaluation (adds ~1 min).")
    ] = True,
    cadence: Annotated[
        str,
        typer.Option(
            help="Backtest refit cadence: 'date' (finest fair protocol) or "
            "'gameweek' (once per round, matching the publishing cadence, story C-09)."
        ),
    ] = "date",
) -> None:
    """Walk-forward backtest Poisson and Dixon-Coles against the market baseline over
    stg_matches (design.md section 8), scoring everything on identical rows. Prints the
    primary table and writes the full report; docs/evaluation.md is rendered from it."""
    import structlog

    from plforecast.evaluate.report import build_report, format_table, write_report

    if cadence not in ("date", "gameweek"):
        raise typer.BadParameter(f"cadence must be 'date' or 'gameweek', got {cadence!r}")

    log = structlog.get_logger()
    matches = _completed_season_matches()
    log.info("evaluate.window", seasons=matches["season"].n_unique(), matches=matches.height)

    tuning = {}
    for path in sorted(out_dir.glob("tuning-*.json")):
        tuning[path.stem.removeprefix("tuning-")] = json.loads(path.read_text())

    report = build_report(
        matches,
        _model_factories(),
        min_train_matches=min_train_matches,
        season_level=season_level,
        tuning=tuning,
        cadence=cadence,  # type: ignore[arg-type]
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


def _completed_season_matches() -> pl.DataFrame:
    """Every stg_matches column the models and the benchmark need, completed seasons
    only (the in-progress season's results are not final)."""
    conn = connect()
    matches = conn.execute(
        "SELECT match_id, season, date, home_club_id, away_club_id, home_goals, away_goals, "
        "home_xg, away_xg, result, benchmark_home_odds, benchmark_draw_odds, "
        "benchmark_away_odds, benchmark_source "
        "FROM stg_matches WHERE season != (SELECT max(season) FROM stg_matches) "
        "ORDER BY date"
    ).pl()
    conn.close()
    return matches


def _model_factories() -> dict[str, Callable[[], MatchModel]]:
    """The model ladder as configured (config.py holds the tuned hyperparameters)."""
    from plforecast.config import settings
    from plforecast.models.dixon_coles import DixonColesModel
    from plforecast.models.poisson import PoissonModel
    from plforecast.models.xg_rates import XGRateModel

    return {
        "poisson": PoissonModel,
        "dixon-coles": lambda: DixonColesModel(xi=settings.dixon_coles_xi),
        "xg-rates": lambda: XGRateModel(
            xi=settings.xg_rates_xi, blend=settings.xg_rates_blend, rho=settings.xg_rates_rho
        ),
    }


@app.command()
def tune(
    model: Annotated[str, typer.Option(help="dixon-coles or xg-rates")] = "dixon-coles",
    parameter: Annotated[str, typer.Option(help="xi, blend or rho")] = "xi",
    grid: Annotated[
        str, typer.Option(help="Comma-separated values to try.")
    ] = "0,0.0005,0.001,0.0018,0.003,0.005",
    select_through_season: Annotated[
        str, typer.Option(help="Last season used for selection; later seasons report.")
    ] = "2021/22",
    out_dir: Annotated[Path, typer.Option(help="Where tuning-<model>-<param>.json goes.")] = Path(
        "docs/evaluation"
    ),
    null_value: Annotated[
        float | None,
        typer.Option(
            help="Value that switches the parameter off; preferred when within tolerance."
        ),
    ] = None,
    tolerance: Annotated[
        float, typer.Option(help="RPS tolerance for the parsimony rule.")
    ] = 0.0005,
) -> None:
    """Grid-search one hyperparameter by walk-forward backtest with disjoint selection
    and reporting seasons (story C-03). Writes the grid; set the chosen value in
    config.py (or the PLFORECAST_* env var) and re-run `evaluate`."""
    from plforecast.config import settings
    from plforecast.evaluate.tuning import tune_parameter
    from plforecast.models.dixon_coles import DixonColesModel
    from plforecast.models.xg_rates import XGRateModel

    values = [float(v) for v in grid.split(",")]

    def make_factory(value: float) -> Callable[[], MatchModel]:
        if model == "dixon-coles" and parameter == "xi":
            return lambda: DixonColesModel(xi=value)
        if model == "xg-rates":
            kwargs = {
                "xi": settings.xg_rates_xi,
                "blend": settings.xg_rates_blend,
                "rho": settings.xg_rates_rho,
            }
            if parameter not in kwargs:
                raise typer.BadParameter(f"xg-rates has no parameter {parameter!r}")
            kwargs[parameter] = value
            return lambda: XGRateModel(**kwargs)
        raise typer.BadParameter(f"cannot tune {parameter!r} on {model!r}")

    matches = _completed_season_matches()
    result = tune_parameter(
        matches,
        make_factory,
        values,
        parameter=parameter,
        select_through_season=select_through_season,
        null_value=null_value,
        tolerance=tolerance,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tuning-{model}-{parameter}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["grid"]:
        typer.echo(
            f"{parameter}={row['value']:<8} select rps={row['select_rps']:.4f} "
            f"(n={row['select_n']})  report rps={row['report_rps']:.4f} (n={row['report_n']})"
        )
    note = " (parsimony rule)" if result["parsimony_applied"] else ""
    typer.echo(f"selected {parameter}={result['selected']}{note}; written {path}")


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
    model: Annotated[
        str,
        typer.Option(
            help="poisson, dixon-coles, xg-rates, or 'shipped' (the best model in "
            "docs/evaluation/metrics.json, falling back to dixon-coles)."
        ),
    ] = "shipped",
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
    training set is stg_matches plus any FPL-finished fixture not yet in it. The
    pipeline itself (training-data assembly, promoted-club prior, fit, simulate) is
    `forecasting.fit_and_simulate`, importable directly for anything other than
    writing artifact files -- the notebook workbench in particular."""
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
    from plforecast.forecasting import fit_and_simulate

    generated_at = now_utc()
    try:
        run = fit_and_simulate(
            model,
            simulations=simulations,
            seed=seed,
            max_goals=max_goals,
            as_of=generated_at.date(),
        )
    except ValueError as exc:
        raise typer.Exit(code=1) from typer.BadParameter(str(exc))

    sha, dirty = git_provenance()
    provenance = Provenance(
        git_sha=sha,
        git_dirty=dirty,
        model=run.model_name,
        model_config_hash=run.fitted.config_hash,
        simulations=simulations,
        random_seed=seed,
        sources=snapshot_provenance(settings.raw_dir),
        package_versions=package_versions(),
    )
    season_slug = run.season_label.replace("/", "-")
    forecast_doc = build_forecast_document(
        run.result,
        played=run.played,
        display_names=run.display_names,
        season=season_slug,
        as_of_gameweek=run.as_of_gameweek,
        generated_at=generated_at,
        provenance=provenance,
    )
    fixtures_doc = build_fixtures_document(
        run.remaining,
        run.fitted,
        season=season_slug,
        as_of_gameweek=run.as_of_gameweek,
        generated_at=generated_at,
        provenance=provenance,
        max_goals=max_goals,
    )
    written = write_documents(forecast_doc, fixtures_doc, artifacts_dir=artifacts_dir)
    written += export_json_schemas(artifacts_dir / "schema")

    typer.echo(
        f"{run.season_label} after gameweek {run.as_of_gameweek}: {run.played.height} played, "
        f"{run.remaining.height} remaining, {simulations:,} simulations, model={run.model_name}"
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
def validate() -> None:
    """Run data-quality invariants over the curated tables (story B-13): round-robin
    shape per completed season, id uniqueness, no self-fixtures, result-vs-score
    agreement, xG coverage, closing-odds coverage (report only), every fact-table
    club_id resolving to dim_club, and the FPL/football-data reconciliation. Exits
    non-zero and prints every failed check if any invariant does not hold."""
    from plforecast.storage.validate import ValidationFailed, run_validations

    conn = connect()
    try:
        results = run_validations(conn, raise_on_failure=False)
    finally:
        conn.close()

    for result in results:
        mark = "OK  " if result.ok else "FAIL"
        typer.echo(f"{mark} {result.name}: {result.detail}")

    failed = [r for r in results if not r.ok]
    if failed:
        raise typer.Exit(code=1) from ValidationFailed(
            "; ".join(f"{r.name}: {r.detail}" for r in failed)
        )


@app.command()
def prune_raw(
    keep: Annotated[
        int, typer.Option(help="Snapshots to keep per source; the latest is always kept.")
    ] = 4,
) -> None:
    """Delete all but the `keep` most recent raw snapshots per source (story B-17).
    Every ingest run lands a full snapshot and every raw_* view unions all of them, so
    retained snapshot count is the only thing keeping view-scan cost from growing
    without bound on a long-running install; curate's own natural-key dedupe makes old
    snapshots redundant once a newer one exists."""
    from plforecast.config import settings
    from plforecast.ingest.base import prune_snapshots

    removed = prune_snapshots(settings.raw_dir, keep=keep)
    if not removed:
        typer.echo(f"nothing to prune (every source has {keep} or fewer snapshots)")
        return
    for source, paths in removed.items():
        typer.echo(f"{source}: removed {len(paths)} snapshot(s)")


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
