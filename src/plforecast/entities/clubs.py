"""Canonical club dimension and strict alias resolution.

The hardest data problem in the project (design.md section 5.3) and the one that will
bite silently: every source names clubs differently, and a club's absence from a
source -- a relegated club no longer appearing in the FPL API's current roster, say --
is expected, not an error. Resolution is strict by default: an unresolvable value
raises rather than falling back to fuzzy matching. `suggest_matches` exists only as an
explicit, separate helper for a club-name reconciliation notebook; it is never called
by `resolve` and must never be wired into the pipeline.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import yaml

ALIASES_PATH = Path(__file__).parent / "club_aliases.yaml"

SourceName = Literal["football_data", "fpl", "clubelo", "understat", "transfermarkt"]

_ALIAS_COLUMNS: dict[SourceName, str] = {
    "football_data": "football_data_name",
    "fpl": "fpl_code",
    "clubelo": "clubelo_name",
    "understat": "understat_name",
    "transfermarkt": "transfermarkt_id",
}

_UNIQUE_COLUMNS = ("club_id", *_ALIAS_COLUMNS.values())


@dataclass(frozen=True, slots=True)
class ClubDimension:
    """The canonical club dimension plus a per-source lookup index."""

    frame: pl.DataFrame
    _index: dict[SourceName, dict[object, str]]

    def resolve(self, values: Sequence[object], source: SourceName) -> list[str]:
        """Map raw source values to canonical club_id, in order. Raises ValueError
        naming every unmatched value at once -- alias gaps tend to arrive in batches
        (a whole season's worth of a newly promoted club), not one row at a time."""
        index = self._index[source]
        unmatched = sorted({v for v in values if v not in index}, key=str)
        if unmatched:
            raise ValueError(
                f"no club_aliases.yaml entry for {source} value(s): {unmatched!r}. "
                f"Add them to {ALIASES_PATH}."
            )
        return [index[v] for v in values]


def _duplicates(rows: list[dict[str, object]], column: str) -> list[object]:
    seen: dict[object, int] = {}
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        seen[value] = seen.get(value, 0) + 1
    return [value for value, count in seen.items() if count > 1]


def _validate_uniqueness(rows: list[dict[str, object]]) -> None:
    for column in _UNIQUE_COLUMNS:
        dupes = _duplicates(rows, column)
        if dupes:
            raise ValueError(f"club_aliases.yaml has duplicate {column} value(s): {dupes!r}")


def load_club_dimension(path: Path = ALIASES_PATH) -> ClubDimension:
    document = yaml.safe_load(path.read_text())
    rows: list[dict[str, object]] = document["clubs"]
    _validate_uniqueness(rows)

    frame = pl.DataFrame(rows)
    index: dict[SourceName, dict[object, str]] = {
        source: {row[column]: str(row["club_id"]) for row in rows if row.get(column) is not None}
        for source, column in _ALIAS_COLUMNS.items()
    }
    return ClubDimension(frame=frame, _index=index)


def suggest_matches(
    name: str, candidates: Sequence[str], *, n: int = 3, cutoff: float = 0.6
) -> list[str]:
    """Fuzzy-match helper for the club-name reconciliation notebook only. Never used by
    `resolve` -- design.md section 5.3 requires fuzzy matching to stay explicit and
    outside the pipeline."""
    return difflib.get_close_matches(name, candidates, n=n, cutoff=cutoff)
