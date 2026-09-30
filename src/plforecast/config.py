"""Single configuration surface for the pipeline. Env vars use the PLFORECAST_ prefix.

`settings` (module-level, built once at import) is the process-wide default and the
right thing for a CLI command to read at its own entry point -- that is the injection
boundary. Everything below that boundary (ingest Source classes, storage.db.connect,
storage.curate's functions, derive_season_from_kickoffs) takes an explicit
`config: Settings = settings` parameter instead of reaching for the global itself, so a
caller (a test, or a future multi-tenant use) can override it (story A-14).
`get_settings()` exists for the same reason `settings = Settings()` alone does not
fully cover: an env var set after import (a test using monkeypatch.setenv then wanting
a fresh read) is otherwise invisible, since the module-level `settings` was already
built.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
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

    # ClubElo as the promoted-club prior's external_rating (ADR 0006, story C-16),
    # in [0, 1] -- build_prior()'s own shrinkage weight. Activated 19 September 2026
    # at 0.5 (features/clubelo.py, forecasting._inject_promoted_club_prior). The one
    # evaluation run behind this (notebooks/03-prototypes/03-04-clubelo-prior-
    # workbench.ipynb) was inconclusive on RPS across an 18-match historical sample
    # -- differences across the whole 0-1 weight sweep stayed within noise for that
    # sample size -- but the live forecast shifted materially and plausibly for the
    # promoted clubs it applies to (Hull City, Coventry City), and that live-forecast
    # evidence was judged sufficient to activate. A club with no resolvable ClubElo
    # rating still falls back to the survival-zone anchor alone, unaffected by this
    # value. Revisit toward 0 if a wider evaluation contradicts the live shift.
    clubelo_prior_weight: float = 0.5

    # ClubElo's live fetch fails from GitHub Actions runners (clubelo.com returns 504s to
    # them, though it is fine from a home connection), so the weekly pipeline falls back
    # to the newest committed snapshot in `clubelo_seed_dir` (ingest/clubelo.py). Elo
    # moves slowly and only shapes the promoted-club priors, so a stale snapshot is a
    # small error -- but it must not rot unnoticed: past `warn_days` the fallback says
    # to refresh the seed (`just refresh-clubelo-seed`), past `fail_days` it refuses.
    clubelo_seed_dir: Path = Path("seeds/clubelo")
    clubelo_stale_warn_days: int = 30
    clubelo_stale_fail_days: int = 90

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """A cached, explicitly-constructed Settings -- for a call site that wants the
    current settings without importing the eagerly-built module singleton (and so
    that a test can call `get_settings.cache_clear()` after changing the environment
    and get a fresh read, which mutating or reassigning the `settings` singleton
    cannot offer since other modules already hold their own reference to it)."""
    return Settings()
