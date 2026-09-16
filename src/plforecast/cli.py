from __future__ import annotations

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
def evaluate() -> None:
    """Walk-forward backtest Poisson and Dixon-Coles against the market baseline over
    stg_matches (design.md section 8). Prints a summary table; see docs/evaluation.md
    for the last full write-up."""
    from collections.abc import Callable

    import structlog

    from plforecast.evaluate.backtest import run_backtest
    from plforecast.evaluate.market import market_probabilities
    from plforecast.evaluate.metrics import brier_score, log_loss, outcome_index, rps
    from plforecast.models.base import MatchModel
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

    model_factories: list[tuple[str, Callable[[], MatchModel]]] = [
        ("poisson", PoissonModel),
        ("dixon-coles", lambda: DixonColesModel(xi=0.0018)),
    ]
    rows = []
    for name, factory in model_factories:
        result = run_backtest(matches, factory)
        rows.append(
            {
                "model": name,
                "n": result.height,
                "mean_rps": result["rps"].mean(),
                "mean_log_loss": result["log_loss"].mean(),
                "mean_brier": result["brier"].mean(),
            }
        )

    for method in ("shin", "multiplicative"):
        market = market_probabilities(matches, method=method)
        outcomes = outcome_index(market["result"].to_list())
        probs = market.select("p_home", "p_draw", "p_away").to_numpy()
        rows.append(
            {
                "model": f"market ({method})",
                "n": market.height,
                "mean_rps": rps(probs, outcomes).mean(),
                "mean_log_loss": log_loss(probs, outcomes).mean(),
                "mean_brier": brier_score(probs, outcomes).mean(),
            }
        )

    for row in rows:
        typer.echo(
            f"{row['model']:20s} n={row['n']:5d}  "
            f"rps={row['mean_rps']:.4f}  log_loss={row['mean_log_loss']:.4f}  "
            f"brier={row['mean_brier']:.4f}"
        )


if __name__ == "__main__":
    app()
