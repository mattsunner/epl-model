"""Build and write the forecast artifact documents from a simulation result.

Publishing rule (design.md 15, open question 1, decided in ADR 0004): both a
gameweek-stamped copy (`forecast-gwNN.json`) that is never overwritten, and a
`forecast-latest.json` that is overwritten in place for the site. Same for fixtures.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import polars as pl

from plforecast.artifacts.schema import (
    Benchmark,
    ClubCurrent,
    ClubForecast,
    ClubProjection,
    ExpectedGoals,
    FixtureForecast,
    FixturesDocument,
    ForecastDocument,
    Outcome,
    Provenance,
    Quantiles,
    Scoreline,
    SourceProvenance,
)
from plforecast.evaluate.metrics import outcome_probabilities
from plforecast.models.base import MatchModel
from plforecast.simulate.competition import PREMIER_LEAGUE, CompetitionConfig
from plforecast.simulate.engine import SimulationResult

PROVENANCE_PACKAGES = ("plforecast", "penaltyblog", "numpy", "scipy", "polars", "duckdb")


def git_provenance(repo_dir: Path | None = None) -> tuple[str, bool]:
    """(sha, dirty). 'unknown' and dirty=True when git is unavailable, so an artifact
    can never silently claim a clean provenance it does not have."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def package_versions(packages: tuple[str, ...] = PROVENANCE_PACKAGES) -> dict[str, str]:
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def snapshot_provenance(raw_dir: Path) -> list[SourceProvenance]:
    """One entry per raw source, from the `_meta.json` of its most recent snapshot."""
    sources = []
    for source_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        snapshots = sorted(p for p in source_dir.iterdir() if (p / "_meta.json").exists())
        if not snapshots:
            continue
        meta = json.loads((snapshots[-1] / "_meta.json").read_text())
        sources.append(
            SourceProvenance(
                name=meta["source"],
                fetched_at=datetime.fromisoformat(meta["fetched_at"]),
                rows=meta["row_count"],
                content_hash=meta["content_hash"],
            )
        )
    return sources


def _quantiles(values: np.ndarray) -> Quantiles:
    return Quantiles(
        mean=float(values.mean()),
        p10=float(np.percentile(values, 10)),
        p50=float(np.percentile(values, 50)),
        p90=float(np.percentile(values, 90)),
    )


def _position_quantiles(pmf: np.ndarray) -> Quantiles:
    positions = np.arange(1, len(pmf) + 1)
    cdf = np.cumsum(pmf)

    def quantile(q: float) -> float:
        return float(positions[int(np.searchsorted(cdf, q, side="left"))])

    return Quantiles(
        mean=float((pmf * positions).sum()),
        p10=quantile(0.10),
        p50=quantile(0.50),
        p90=quantile(0.90),
    )


def _current_table(played: pl.DataFrame, club_ids: list[str]) -> dict[str, ClubCurrent]:
    played_count = dict.fromkeys(club_ids, 0)
    points = dict.fromkeys(club_ids, 0)
    gd = dict.fromkeys(club_ids, 0)
    for row in played.iter_rows(named=True):
        h, a = row["home_club_id"], row["away_club_id"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        played_count[h] += 1
        played_count[a] += 1
        gd[h] += hg - ag
        gd[a] += ag - hg
        if hg > ag:
            points[h] += 3
        elif hg < ag:
            points[a] += 3
        else:
            points[h] += 1
            points[a] += 1
    return {c: ClubCurrent(played=played_count[c], points=points[c], gd=gd[c]) for c in club_ids}


def build_forecast_document(
    result: SimulationResult,
    *,
    played: pl.DataFrame,
    display_names: dict[str, str],
    season: str,
    as_of_gameweek: int,
    generated_at: datetime,
    provenance: Provenance,
    competition: CompetitionConfig = PREMIER_LEAGUE,
    benchmark: Benchmark | None = None,
) -> ForecastDocument:
    """`played`: this season's finished matches (home_club_id, away_club_id, home_goals,
    away_goals), the same frame the simulation started from."""
    pmf = result.position_pmf
    current = _current_table(played, result.club_ids)
    n = len(result.club_ids)
    first_relegation_index = n - competition.relegation_spots

    clubs = []
    for i, club_id in enumerate(result.club_ids):
        club_pmf = pmf[i]
        clubs.append(
            ClubForecast(
                club_id=club_id,
                display_name=display_names.get(club_id, club_id),
                current=current[club_id],
                projected=ClubProjection(
                    points=_quantiles(result.points[:, i]),
                    position=_position_quantiles(club_pmf),
                    position_pmf=[float(p) for p in club_pmf],
                    title=float(club_pmf[0]),
                    top_four=float(club_pmf[:4].sum()),
                    relegation=float(club_pmf[first_relegation_index:].sum()),
                ),
            )
        )
    clubs.sort(key=lambda c: (c.projected.position.mean, c.club_id))

    return ForecastDocument(
        season=season,
        as_of_gameweek=as_of_gameweek,
        generated_at=generated_at,
        provenance=provenance,
        clubs=clubs,
        benchmark=benchmark,
    )


def build_fixtures_document(
    remaining: pl.DataFrame,
    model: MatchModel,
    *,
    season: str,
    as_of_gameweek: int,
    generated_at: datetime,
    provenance: Provenance,
    max_goals: int = 10,
    top_n_scorelines: int = 10,
) -> FixturesDocument:
    """`remaining`: fixture_id, gameweek (nullable), kickoff_time (nullable),
    home_club_id, away_club_id. One entry per remaining fixture, from the same fitted
    `model` the simulation used."""
    fixtures = []
    for row in remaining.iter_rows(named=True):
        matrix = model.scoreline_matrix(row["home_club_id"], row["away_club_id"], max_goals)
        probs = outcome_probabilities(matrix)
        goals = np.arange(max_goals + 1)
        flat = matrix.ravel()
        top = np.argsort(-flat)[:top_n_scorelines]
        fixtures.append(
            FixtureForecast(
                fixture_id=str(row["fixture_id"]),
                gameweek=row.get("gameweek"),
                kickoff=row.get("kickoff_time"),
                home=row["home_club_id"],
                away=row["away_club_id"],
                outcome=Outcome(home=float(probs[0]), draw=float(probs[1]), away=float(probs[2])),
                expected_goals=ExpectedGoals(
                    home=float((matrix.sum(axis=1) * goals).sum()),
                    away=float((matrix.sum(axis=0) * goals).sum()),
                ),
                top_scorelines=[
                    Scoreline(
                        home=int(idx // matrix.shape[1]),
                        away=int(idx % matrix.shape[1]),
                        p=float(flat[idx]),
                    )
                    for idx in top
                ],
            )
        )
    return FixturesDocument(
        season=season,
        as_of_gameweek=as_of_gameweek,
        generated_at=generated_at,
        provenance=provenance,
        fixtures=fixtures,
    )


def write_documents(
    forecast: ForecastDocument,
    fixtures: FixturesDocument,
    *,
    artifacts_dir: Path,
) -> list[Path]:
    """Writes forecast-gwNN.json and fixtures-gwNN.json (never overwritten once
    committed: a re-run of the same gameweek replaces the file, and git shows the
    diff) plus forecast-latest.json and fixtures-latest.json, overwritten in place."""
    season_dir = artifacts_dir / forecast.season
    season_dir.mkdir(parents=True, exist_ok=True)
    gw = f"gw{forecast.as_of_gameweek:02d}"
    written = []
    for name, document in (("forecast", forecast), ("fixtures", fixtures)):
        payload = document.model_dump_json(indent=2) + "\n"
        for suffix in (gw, "latest"):
            path = season_dir / f"{name}-{suffix}.json"
            path.write_text(payload)
            written.append(path)
    return written


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)
