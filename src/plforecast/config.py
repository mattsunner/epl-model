"""Single configuration surface for the pipeline. Env vars use the PLFORECAST_ prefix."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLFORECAST_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    db_filename: str = "pl.duckdb"

    request_delay_seconds: float = 1.0
    cache_ttl_hours: float = 24.0
    http_timeout_seconds: float = 30.0
    user_agent: str = "plforecast/0.1 (+https://github.com/matthewsunner/epl-models)"

    backfill_start_season: int = 2015  # first season's start year, i.e. 2015/16 --
    # shared by every source's full-history backfill, since the backtest window
    # (design.md section 8.2) fixes how far back the pipeline needs to reach.

    football_data_base_url: str = "https://www.football-data.co.uk/mmz4281"

    fpl_bootstrap_url: str = "https://fantasy.premierleague.com/api/bootstrap-static/"
    fpl_fixtures_url: str = "https://fantasy.premierleague.com/api/fixtures/"

    understat_league: str = "ENG-Premier League"

    # Model hyperparameters, set by `plforecast tune` (story C-03) and recorded in
    # docs/evaluation/tuning-*.json. Never assumed in model code: every model takes
    # them as explicit constructor arguments; these are only the pipeline's defaults.
    # Tuned 16 September 2026 on 2015/16-2021/22, reported on 2022/23-2025/26
    # (docs/evaluation.md, "Hyperparameter tuning"). Dixon and Coles' 1997 value for xi
    # turned out to be the interior optimum for the goals model.
    dixon_coles_xi: float = 0.0018
    xg_rates_xi: float = 0.005
    xg_rates_blend: float = 0.75
    xg_rates_rho: float = 0.0  # grid flat within 0.0004 RPS; parsimony rule keeps it off

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    def current_season_start_year(self, today: date | None = None) -> int:
        """Premier League seasons run August-May; treat July as the season-year rollover."""
        today = today or date.today()
        return today.year if today.month >= 7 else today.year - 1


def season_label(start_year: int) -> str:
    """2015 -> '2015/16'. The one canonical season-string format used everywhere in the
    pipeline, so every layer that needs to stamp a season onto a curated table agrees
    on its shape without depending on each other's internals."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def season_code(start_year: int) -> str:
    """2015 -> '1516'. Shared by football-data.co.uk's URL scheme and, as it turns out,
    the only unambiguous way to ask soccerdata's Understat scraper for a given season:
    passing a bare integer start year (e.g. 2021, meaning 2021/22) gets silently
    misinterpreted by soccerdata as the 2020/21 season instead, because the string
    "2021" is self-referentially ambiguous -- it reads as both a literal year and a
    "20-21" pair-code, and soccerdata's parser resolves that specific collision in
    favour of the pair-code reading. Passing the full pair-code string up front (e.g.
    "2122" for 2021/22) sidesteps the ambiguity entirely rather than fixing it -- it was
    never ambiguous to begin with once both halves of the pair are given explicitly."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


settings = Settings()
