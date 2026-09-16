"""Forecast artifact documents (design.md section 9): the contract between the model
and everything downstream. Two documents per published gameweek, versioned
independently:

- `ForecastDocument` (forecast-gwNN.json): per-club position distribution, projected
  points, title / top-four / relegation probabilities. What the published page renders.
- `FixturesDocument` (fixtures-gwNN.json): per-fixture 1X2 probabilities, expected goals
  and the top scorelines. What is needed to score the model independently, not what is
  needed to reconstruct it (no full scoreline matrix).

Both carry an identical provenance block: git SHA, model config hash, simulation count,
seed, raw-snapshot identities and library versions, so a run at that SHA reproduces the
artifact (design.md sections 1.3 and 9.1; story A-16).

JSON Schema files under artifacts/schema/ are exported from these models
(`export_json_schemas`) for consumers that do not import this package.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FORECAST_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
FIXTURES_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

Probability = Field(ge=0.0, le=1.0)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceProvenance(_Strict):
    name: str
    fetched_at: datetime
    rows: int = Field(ge=0)
    content_hash: str


class Provenance(_Strict):
    git_sha: str
    git_dirty: bool
    model: str
    model_config_hash: str
    simulations: int = Field(ge=1)
    random_seed: int
    sources: list[SourceProvenance]
    package_versions: dict[str, str]


class Quantiles(_Strict):
    mean: float
    p10: float
    p50: float
    p90: float


class ClubCurrent(_Strict):
    played: int = Field(ge=0)
    points: int = Field(ge=0)
    gd: int


class ClubProjection(_Strict):
    points: Quantiles
    position: Quantiles
    position_pmf: list[float]
    title: float = Probability
    top_four: float = Probability
    relegation: float = Probability

    @model_validator(mode="after")
    def _pmf_is_a_distribution(self) -> ClubProjection:
        if any(p < 0 for p in self.position_pmf):
            raise ValueError("position_pmf has a negative entry")
        if not math.isclose(sum(self.position_pmf), 1.0, abs_tol=1e-6):
            raise ValueError(f"position_pmf sums to {sum(self.position_pmf)}, not 1")
        return self


class ClubForecast(_Strict):
    club_id: str
    display_name: str
    current: ClubCurrent
    projected: ClubProjection


class Benchmark(_Strict):
    source: str
    note: str


class ForecastDocument(_Strict):
    schema_version: Literal["1.0.0"] = FORECAST_SCHEMA_VERSION
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    as_of_gameweek: int = Field(ge=0)
    generated_at: datetime
    provenance: Provenance
    clubs: list[ClubForecast]
    benchmark: Benchmark | None = None

    @model_validator(mode="after")
    def _position_matrix_is_square(self) -> ForecastDocument:
        n = len(self.clubs)
        bad = [c.club_id for c in self.clubs if len(c.projected.position_pmf) != n]
        if bad:
            raise ValueError(f"position_pmf length != number of clubs ({n}) for {bad}")
        return self


class Outcome(_Strict):
    home: float = Probability
    draw: float = Probability
    away: float = Probability

    @model_validator(mode="after")
    def _sums_to_one(self) -> Outcome:
        if not math.isclose(self.home + self.draw + self.away, 1.0, abs_tol=1e-6):
            raise ValueError("1X2 probabilities must sum to 1")
        return self


class ExpectedGoals(_Strict):
    home: float = Field(ge=0)
    away: float = Field(ge=0)


class Scoreline(_Strict):
    home: int = Field(ge=0)
    away: int = Field(ge=0)
    p: float = Probability


class FixtureForecast(_Strict):
    fixture_id: str
    gameweek: int | None = Field(default=None, ge=1)
    kickoff: datetime | None
    home: str
    away: str
    outcome: Outcome
    expected_goals: ExpectedGoals
    top_scorelines: list[Scoreline]


class FixturesDocument(_Strict):
    schema_version: Literal["1.0.0"] = FIXTURES_SCHEMA_VERSION
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    as_of_gameweek: int = Field(ge=0)
    generated_at: datetime
    provenance: Provenance
    fixtures: list[FixtureForecast]


def export_json_schemas(schema_dir: Path) -> list[Path]:
    """Write forecast-v1.schema.json and fixtures-v1.schema.json (design.md 9.4)."""
    schema_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in (("forecast", ForecastDocument), ("fixtures", FixturesDocument)):
        path = schema_dir / f"{name}-v1.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
        written.append(path)
    return written
