from pathlib import Path

from plforecast.config import Settings, get_settings, season_code, season_label


def test_get_settings_is_cached_and_returns_a_real_settings_instance():
    a = get_settings()
    b = get_settings()
    assert a is b
    assert isinstance(a, Settings)


def test_get_settings_reflects_env_vars_set_after_a_cache_clear(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("PLFORECAST_DATA_DIR", "/tmp/some-other-data-dir")
    try:
        fresh = get_settings()
        assert fresh.data_dir == Path("/tmp/some-other-data-dir")
    finally:
        get_settings.cache_clear()  # do not leak the override into later tests


def test_settings_can_be_constructed_directly_without_touching_the_global(tmp_path):
    # The injection pattern every ingest Source, connect(), and curate function relies
    # on: a caller builds its own Settings and passes it explicitly.
    custom = Settings(data_dir=tmp_path)
    assert custom.raw_dir == tmp_path / "raw"
    assert custom.db_path == tmp_path / "pl.duckdb"


def test_season_label_and_code_are_inverse_shapes_of_the_same_year():
    assert season_label(2015) == "2015/16"
    assert season_code(2015) == "1516"
    assert season_label(1999) == "1999/00"
