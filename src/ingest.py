"""
Loads the Kaggle FBref Premier League player-stats CSV into SQLite per
schema.sql, filtered to CURRENT_PL_CLUBS.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CLUB_NAME_MAP,
    CURRENT_PL_CLUBS,
    DATA_SEASON,
    DB_PATH,
    PLAYER_STATS_CSV,
)

COLUMN_MAP = {
    "minutes": "Min",
    "goals": "Gls",
    "assists": "Ast",
    "npxg": "npxG",
    "xa": "xAG",             
    "prog_passes": "PrgP",
    "prog_carries": "PrgC",
    "prog_passes_received": "PrgR",
}



def get_connection():
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    with open(Path(__file__).parent / "schema.sql") as f:
        conn.executescript(f.read())
    return conn


def _assert_season_totals(df: pd.DataFrame):
    """
    Guard against silently reading the per-90 duplicate columns. Season totals
    must be >= their per-90 twins across the squad (a player with >90 mins has
    total >= per-90). If this trips, COLUMN_MAP grabbed the wrong half.
    """
    for total, per90 in (("Gls", "Gls.1"), ("npxG", "npxG.1"), ("xAG", "xAG.1")):
        if per90 in df.columns and df[total].sum() < df[per90].sum():
            raise RuntimeError(
                f"Column sanity check failed: '{total}' looks like a per-90 column, "
                f"not a season total. Check COLUMN_MAP against the CSV header."
            )


def _clean_nation(value) -> str | None:
    """FBref nation cells look like 'eng ENG' — keep the 3-letter code."""
    if pd.isna(value):
        return None
    return str(value).split()[-1]


def _clean_age(value) -> int | None:
    if pd.isna(value):
        return None
    return int(float(value))


def load_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        print(f"CSV not found: {path}\n"
            f"Download it from Kaggle first — see the module docstring.")
        sys.exit(1)

    df = pd.read_csv(path)
    _assert_season_totals(df)

    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV is missing expected columns: {missing}")

    df["Squad"] = df["Squad"].replace(CLUB_NAME_MAP)

    before = len(df)
    unknown = sorted(set(df["Squad"]) - set(CURRENT_PL_CLUBS))
    df = df[df["Squad"].isin(CURRENT_PL_CLUBS)].copy()
    print(f"Filtered {before} -> {len(df)} rows")
    if unknown:
        print(f"  [warn] clubs not in CURRENT_PL_CLUBS — check CLUB_NAME_MAP: "
            f"{', '.join(unknown)}")

    return df


def load_into_db(conn: sqlite3.Connection, df: pd.DataFrame, season: str) -> int:
    stat_cols = list(COLUMN_MAP)
    placeholders = ", ".join("?" * (3 + len(stat_cols)))
    insert_stats = (
        f"INSERT OR REPLACE INTO player_season_stats "
        f"(player_id, season, club, {', '.join(stat_cols)}) VALUES ({placeholders})"
    )

    inserted = 0
    for _, row in df.iterrows():
        club = row["Squad"]
        player_id = f"{row['Player']}_{club}".replace(" ", "_")
        position = row["Pos"] if pd.notna(row["Pos"]) else None

        conn.execute(
            """INSERT OR REPLACE INTO players
            (player_id, name, club, position, detailed_position, age, nationality)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (player_id, row["Player"], club,
            position,                                    # full 'DF,MF' so LIKE matches either
            position.split(",")[0] if position else None,  # primary position
            _clean_age(row["Age"]), _clean_nation(row["Nation"])),
        )

        values = [player_id, season, club]
        values += [row[COLUMN_MAP[c]] for c in COLUMN_MAP]
        conn.execute(insert_stats, values)
        inserted += 1

    conn.commit()
    return inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=PLAYER_STATS_CSV)
    parser.add_argument("--season", default=DATA_SEASON)
    args = parser.parse_args()

    print(f"Loading {args.csv} (season {args.season})...")
    df = load_csv(args.csv)

    conn = get_connection()
    inserted = load_into_db(conn, df, args.season)
    print(f"Loaded {inserted} player-rows across {df['Squad'].nunique()} clubs")

    absent = [c for c in CURRENT_PL_CLUBS if c not in set(df["Squad"])]
    if absent:
        print(f"\n[warn] in-scope clubs with zero rows: {', '.join(absent)}")

    print(f"\nNote: no defensive stats in this source (tackles/interceptions/"
        f"pressures/pass%) — those columns are NULL by design.")
    conn.close()
    print(f"Done. DB at {DB_PATH}")


if __name__ == "__main__":
    main()
