from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

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


if __name__ == "__main__":
    app()
