from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).parent.parent.parent / ".github" / "workflows"


def _run_commands(workflow: str) -> list[str]:
    doc = yaml.safe_load((WORKFLOWS / workflow).read_text())
    return [step["run"] for job in doc["jobs"].values() for step in job["steps"] if "run" in step]


def test_weekly_refresh_ingests_clubelo_with_the_stale_fallback():
    """clubelo.com 504s Actions runners; without --allow-stale the whole weekly run dies
    on it (and, since actions/cache only saves on success, never seeds its cache)."""
    clubelo = [c for c in _run_commands("weekly-refresh.yml") if "ingest clubelo" in c]
    assert clubelo, "weekly-refresh.yml has no ClubElo ingest step"
    assert all("--allow-stale" in c for c in clubelo)
