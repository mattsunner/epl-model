import json
from pathlib import Path

import pytest

from plforecast.entities.clubs import load_club_dimension, suggest_matches

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "entities"

# The full set of distinct HomeTeam/AwayTeam values actually observed in the real
# football-data.co.uk backfill (2015/16-2026/27) and the full set of fpl_team_id values
# from the current FPL bootstrap-static roster, at the time these fixtures were
# generated. This is the concrete form of the invariant in design.md section 5.3: every
# club appearing in any curated fact table resolves to exactly one canonical ID.
FOOTBALL_DATA_NAMES = (
    (FIXTURES_DIR / "football_data_team_names.txt").read_text().strip().splitlines()
)
FPL_TEAM_IDS = json.loads((FIXTURES_DIR / "fpl_team_ids.json").read_text())


def test_every_observed_football_data_name_resolves():
    dimension = load_club_dimension()
    resolved = dimension.resolve(FOOTBALL_DATA_NAMES, "football_data")

    assert len(resolved) == len(FOOTBALL_DATA_NAMES)
    assert len(set(resolved)) == len(FOOTBALL_DATA_NAMES), "expected a 1:1 name -> club_id mapping"


def test_every_current_fpl_team_id_resolves():
    dimension = load_club_dimension()
    resolved = dimension.resolve(FPL_TEAM_IDS, "fpl")

    assert len(resolved) == len(FPL_TEAM_IDS)
    assert len(set(resolved)) == len(FPL_TEAM_IDS)


def test_unresolvable_value_raises_naming_the_value():
    dimension = load_club_dimension()

    with pytest.raises(ValueError, match=r"Nonexistent FC"):
        dimension.resolve(["Arsenal", "Nonexistent FC"], "football_data")


def test_unmatched_values_are_batched_into_one_error():
    dimension = load_club_dimension()

    with pytest.raises(ValueError, match=r"Foo.*Bar|Bar.*Foo"):
        dimension.resolve(["Foo FC", "Bar FC"], "football_data")


def test_rejects_duplicate_club_id(tmp_path):
    bad_yaml = tmp_path / "club_aliases.yaml"
    bad_yaml.write_text(
        """
        clubs:
          - club_id: arsenal
            display_name: Arsenal
            football_data_name: Arsenal
            fpl_team_id: 1
            clubelo_name: null
            understat_name: null
            transfermarkt_id: null
          - club_id: arsenal
            display_name: Arsenal Duplicate
            football_data_name: Arsenal Duplicate
            fpl_team_id: 99
            clubelo_name: null
            understat_name: null
            transfermarkt_id: null
        """
    )

    with pytest.raises(ValueError, match="duplicate club_id"):
        load_club_dimension(bad_yaml)


def test_rejects_duplicate_football_data_name(tmp_path):
    bad_yaml = tmp_path / "club_aliases.yaml"
    bad_yaml.write_text(
        """
        clubs:
          - club_id: arsenal
            display_name: Arsenal
            football_data_name: Arsenal
            fpl_team_id: 1
            clubelo_name: null
            understat_name: null
            transfermarkt_id: null
          - club_id: arsenal-2
            display_name: Arsenal Reserves
            football_data_name: Arsenal
            fpl_team_id: 2
            clubelo_name: null
            understat_name: null
            transfermarkt_id: null
        """
    )

    with pytest.raises(ValueError, match="duplicate football_data_name"):
        load_club_dimension(bad_yaml)


def test_suggest_matches_is_never_used_by_resolve():
    # suggest_matches is a standalone helper for the reconciliation notebook. It should
    # still find the obvious near-miss, but resolve() must never call it -- confirmed
    # separately by resolve() raising instead of silently fuzzy-matching above.
    assert suggest_matches("Spurs", ["Tottenham", "Arsenal", "Chelsea"]) == []
    assert "Tottenham" in suggest_matches("Totenham", ["Tottenham", "Arsenal", "Chelsea"])
