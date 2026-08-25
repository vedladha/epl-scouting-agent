"""
Day 2: Feature engineering.

Reads player_season_stats, aggregates across available seasons per player
(weighted toward recent seasons), computes per-90 stats, normalizes, and
writes a feature vector JSON blob per player into player_features.

Run: python src/build_features.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DB_PATH, HISTORICAL_SEASONS

# Recency weights — most recent season counts most. Index 0 = oldest.
# Simple linear ramp; tune later if testers say old form is over-weighted.
SEASON_WEIGHTS = {season: i + 1 for i, season in enumerate(HISTORICAL_SEASONS)}

# Trimmed to the stats the source CSV actually provides. The schema still
# carries the wider set (shots, tackles, pressures, ...) for a richer future
# data source, but those columns are NULL today — including them here would
# produce all-zero features that dilute every similarity score.
# These seven ARE the feature space; config.ROLE_ARCHETYPES must stay in sync.
COUNTING_STATS = [
    "goals", "assists", "npxg", "xa",
    "prog_passes", "prog_carries", "prog_passes_received",
]


def load_stats(conn) -> pd.DataFrame:
    return pd.read_sql(
        """SELECT s.*, p.detailed_position
           FROM player_season_stats s
           JOIN players p ON s.player_id = p.player_id
           WHERE s.minutes IS NOT NULL AND s.minutes > 0""",
        conn,
    )


def weighted_per90(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple seasons per player into one recency-weighted per-90 row."""
    df = df.copy()
    df["weight"] = df["season"].map(SEASON_WEIGHTS).fillna(1)

    rows = []
    for player_id, g in df.groupby("player_id"):
        total_minutes = g["minutes"].sum()
        if total_minutes < 450:  # ~5 full matches — floor to avoid noisy small samples
            continue
        w = g["weight"] * g["minutes"]  # weight by both recency AND minutes played
        w_sum = w.sum()
        row = {"player_id": player_id, "total_minutes": total_minutes,
               "season_span": f"{g['season'].min()}:{g['season'].max()}",
               "position": (g["detailed_position"].dropna().iloc[0]
                            if g["detailed_position"].notna().any() else "UNK")}
        for stat in COUNTING_STATS:
            if stat not in g.columns:
                continue
            per90 = (g[stat].fillna(0) / g["minutes"].replace(0, np.nan)) * 90
            row[f"{stat}_p90"] = float((per90.fillna(0) * w).sum() / w_sum) if w_sum else 0.0
        # NOTE: pass_completion_pct / aerials_won_pct used to be derived here.
        # Removed — the source CSV has no passes_completed/attempted or aerials
        # data, so they were always 0.0 and added a fake shared axis to every
        # player's vector. Restore them if a richer data source lands.
        rows.append(row)

    return pd.DataFrame(rows)


META_COLS = ("player_id", "total_minutes", "season_span", "position")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def positional_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile rank for each stat against players in the SAME broad position.

    This is for EXPLAINING a player, not for ranking them — ranking uses
    league-wide z-scores, because a trait's value to a team is absolute. But
    "5.2 progressive carries per 90" tells a scout nothing without a reference
    class, and the useful reference class is positional: that number is
    unremarkable for a winger and extraordinary for a centre-back.
    """
    pct = df.copy()
    for col in feature_columns(df):
        pct[col] = (df.groupby("position")[col]
                      .rank(pct=True, method="average") * 100).round(0)
    return pct


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each feature column so no single stat dominates ranking."""
    normed = df.copy()
    for col in feature_columns(df):
        mean, std = df[col].mean(), df[col].std()
        normed[col] = (df[col] - mean) / std if std > 0 else 0.0
    return normed


def main():
    conn = sqlite3.connect(DB_PATH)
    raw = load_stats(conn)
    if raw.empty:
        print("No stats found — run ingest.py first.")
        return

    per90 = weighted_per90(raw)
    normed = normalize(per90)
    pcts = positional_percentiles(per90).set_index("player_id")
    feature_cols = feature_columns(per90)

    # Store BOTH representations: z-scores drive ranking (comparable across
    # stats), raw per-90 values let the agent cite real numbers in its
    # justifications instead of quoting a similarity score at the user.
    raw_by_id = per90.set_index("player_id")

    for _, row in normed.iterrows():
        player_id = row["player_id"]
        conn.execute(
            """INSERT OR REPLACE INTO player_features
               (player_id, season_span, total_minutes, feature_json,
                raw_per90_json, percentiles_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (player_id, row["season_span"], int(row["total_minutes"]),
             json.dumps({c: float(row[c]) for c in feature_cols}),
             # 2dp, not 3 — "5.204 prog carries/90" is false precision that
             # makes the output look more scientific than the data supports.
             json.dumps({c: round(float(raw_by_id.at[player_id, c]), 2) for c in feature_cols}),
             json.dumps({c: int(pcts.at[player_id, c]) for c in feature_cols})),
        )
    conn.commit()
    conn.close()
    print(f"Built feature vectors for {len(normed)} players.")


if __name__ == "__main__":
    main()
