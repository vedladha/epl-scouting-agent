"""
Scans every CSV under data/kaggle_raw and classifies it as:
  - PLAYER-LEVEL  (what we actually need)
  - TEAM-LEVEL    (league tables, club stats — not useful for player scouting)
  - UNCLEAR       (couldn't confidently tell — inspect manually)

Classification is based on column name signatures + row count heuristics:
  - Team-level tables usually have ~20-25 rows (one per club) and columns
    like games_played/points/goal_difference.
  - Player-level tables usually have hundreds of rows and columns like
    player/name + age/nation/position/minutes/goals/assists.

Run: python src/classify_kaggle_csvs.py
"""
import pandas as pd
from pathlib import Path

raw_dir = Path("data/kaggle_raw")

TEAM_SIGNAL_COLS = {
    "games_played", "games_won", "games_drawn", "games_lost",
    "goal_difference", "points", "position", "club_stats",
}
PLAYER_SIGNAL_COLS = {
    "player", "player_name", "name", "age", "nation", "nationality",
    "pos", "position", "squad", "minutes", "min", "goals", "assists",
    "xg", "npxg", "xa", "shots", "tackles", "interceptions", "pressures",
    "born", "dob",
}

# Columns that strongly indicate TEAM level even if a couple player-ish
# columns also happen to be present (e.g. both have "name" and "position")
STRONG_TEAM_COLS = {"games_played", "games_won", "goal_difference", "points"}


def classify(path: Path):
    try:
        df = pd.read_csv(path, nrows=50)
    except Exception as e:
        return "UNREADABLE", str(e), 0

    cols = {c.strip().lower() for c in df.columns}
    team_hits = cols & TEAM_SIGNAL_COLS
    player_hits = cols & PLAYER_SIGNAL_COLS
    strong_team_hits = cols & STRONG_TEAM_COLS

    # Get real row count (not just the 50-row sample)
    try:
        row_count = sum(1 for _ in open(path)) - 1  # minus header
    except Exception:
        row_count = -1

    if strong_team_hits:
        return "TEAM-LEVEL", f"matched {strong_team_hits}", row_count

    if player_hits and row_count > 30:
        return "PLAYER-LEVEL", f"matched {player_hits}", row_count

    if player_hits and row_count <= 30:
        return "UNCLEAR", f"has player-ish cols {player_hits} but only {row_count} rows (could be one team's squad, or team-level)", row_count

    return "UNCLEAR", f"no strong signal — columns: {sorted(cols)}", row_count


def main():
    csvs = sorted(raw_dir.rglob("*.csv"))
    if not csvs:
        print(f"No CSVs found under {raw_dir}")
        return

    results = {"PLAYER-LEVEL": [], "TEAM-LEVEL": [], "UNCLEAR": [], "UNREADABLE": []}

    for path in csvs:
        label, reason, row_count = classify(path)
        results[label].append((path, reason, row_count))

    for label in ["PLAYER-LEVEL", "TEAM-LEVEL", "UNCLEAR", "UNREADABLE"]:
        items = results[label]
        print(f"\n{'='*60}\n{label}: {len(items)} file(s)\n{'='*60}")
        for path, reason, row_count in items[:15]:  # cap noisy output
            print(f"  [{row_count} rows] {path}")
            print(f"    -> {reason}")
        if len(items) > 15:
            print(f"  ... and {len(items) - 15} more")

    print(f"\n\nSUMMARY: {len(results['PLAYER-LEVEL'])} player-level, "
          f"{len(results['TEAM-LEVEL'])} team-level, "
          f"{len(results['UNCLEAR'])} unclear, "
          f"{len(results['UNREADABLE'])} unreadable, "
          f"out of {len(csvs)} total CSVs.")


if __name__ == "__main__":
    main()