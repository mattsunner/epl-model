from pathlib import Path

from typer.testing import CliRunner

from plforecast.artifacts.schema import export_json_schemas
from plforecast.cli import app

runner = CliRunner()


def test_validate_artifacts_checks_season_documents_and_skips_schema_dir(tmp_path: Path):
    season = tmp_path / "2026-27"
    season.mkdir()
    export_json_schemas(tmp_path / "schema")  # must be ignored, not validated as a document
    (season / "forecast-gw01.json").write_text("{}")  # invalid: missing every field

    result = runner.invoke(app, ["validate-artifacts", "--artifacts-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "INVALID" in result.output


def test_validate_artifacts_passes_on_an_empty_directory(tmp_path: Path):
    result = runner.invoke(app, ["validate-artifacts", "--artifacts-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "validated 0" in result.output


def test_ingest_rejects_unknown_source():
    result = runner.invoke(app, ["ingest", "nope"])
    assert result.exit_code != 0


def test_ingest_accepts_allow_stale_for_clubelo(monkeypatch):
    seen: dict[str, bool] = {}
    monkeypatch.setattr(
        "plforecast.ingest.clubelo.ingest",
        lambda **kwargs: seen.update(kwargs),
    )

    result = runner.invoke(app, ["ingest", "clubelo", "--allow-stale"])

    assert result.exit_code == 0
    assert seen["allow_stale"] is True
