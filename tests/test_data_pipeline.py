"""
The data pipeline, and the two CSV traps that fail silently.

Both traps produce a plausible-looking dataset with wrong contents, which is
worse than a crash — see docs/DECISIONS.md.
"""
import json

import pandas as pd

from config import CLUB_NAME_MAP, CURRENT_PL_CLUBS, FEATURE_COLUMNS
from conftest import CSV, needs_csv, needs_db
from ingest import COLUMN_MAP

OUT_OF_SCOPE_IN_2024_25 = {"Leicester City", "Southampton", "West Ham", "Wolves"}


@needs_csv
def test_column_map_reads_season_totals_not_the_per90_duplicates():
    """
    Regression: the CSV repeats ten stat names as per-90 versions, which pandas
    de-duplicates to 'Gls.1', 'npxG.1'. Reading those instead of the totals
    divides by minutes twice — quietly wrong numbers, no error.
    """
    df = pd.read_csv(CSV)
    for source in COLUMN_MAP.values():
        assert not source.endswith(".1"), f"COLUMN_MAP points at a per-90 duplicate: {source}"
    for total in ("Gls", "npxG", "xAG"):
        assert df[total].sum() > df[f"{total}.1"].sum(), f"{total} is not a season total"


@needs_csv
def test_club_name_map_catches_every_in_scope_fbref_short_name():
    """
    Regression: the CSV uses 'Brighton', 'Manchester Utd', "Nott'ham Forest".
    Filtering without normalising silently dropped five clubs and still
    produced a dataset that looked fine.
    """
    raw = set(pd.read_csv(CSV)["Squad"].dropna().unique())
    unmapped = {c for c in raw if CLUB_NAME_MAP.get(c, c) not in CURRENT_PL_CLUBS}
    assert unmapped == OUT_OF_SCOPE_IN_2024_25, \
        f"unexpected unmapped clubs — likely a spelling miss: {unmapped - OUT_OF_SCOPE_IN_2024_25}"


@needs_csv
def test_club_name_map_has_no_stale_entries():
    raw = set(pd.read_csv(CSV)["Squad"].dropna().unique())
    dead = set(CLUB_NAME_MAP) - raw
    assert not dead, f"CLUB_NAME_MAP maps names absent from the CSV: {dead}"


@needs_db
def test_every_vector_carries_all_three_representations(db):
    rows = db.execute("SELECT feature_json, raw_per90_json, percentiles_json "
                      "FROM player_features").fetchall()
    assert rows, "no feature vectors built"
    for feats, raw, pcts in rows:
        for blob in (feats, raw, pcts):
            assert set(json.loads(blob)) == set(FEATURE_COLUMNS)


@needs_db
def test_percentiles_are_within_range(db):
    for (blob,) in db.execute("SELECT percentiles_json FROM player_features"):
        for feature, pct in json.loads(blob).items():
            assert 0 <= pct <= 100, f"{feature} percentile out of range: {pct}"


@needs_db
def test_percentiles_are_computed_against_positional_peers(db):
    """A defender's shooting percentile must be judged among defenders — 0.12
    npxG/90 is nothing for a forward and elite for a centre-back."""
    row = db.execute(
        """SELECT percentiles_json FROM player_features f
           JOIN players p USING (player_id)
           WHERE p.name = 'Joško Gvardiol'""").fetchone()
    assert row, "expected Gvardiol in the dataset"
    assert json.loads(row[0])["npxg_p90"] >= 90


@needs_db
def test_only_current_pl_clubs_are_ranked(db):
    clubs = {c for (c,) in db.execute(
        "SELECT DISTINCT current_club FROM players JOIN player_features USING (player_id)")}
    assert clubs <= set(CURRENT_PL_CLUBS), f"out of scope: {clubs - set(CURRENT_PL_CLUBS)}"


@needs_db
def test_the_minutes_floor_is_enforced(db):
    low = db.execute("SELECT COUNT(*) FROM player_features WHERE total_minutes < 450").fetchone()[0]
    assert low == 0, f"{low} players below the 450-minute floor were ranked"


@needs_db
def test_market_value_table_is_gone(db):
    """Decided 2026-08-25: out of scope, removed rather than left empty."""
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "player_market_value" not in tables


@needs_db
def test_unavailable_stats_are_null_not_zero(db):
    """
    Zero would mean "this player made no tackles", which is a claim.
    NULL means "we do not know", which is the truth.
    """
    row = db.execute("SELECT tackles, interceptions, pressures FROM player_season_stats "
                     "LIMIT 1").fetchone()
    assert all(v is None for v in row)
