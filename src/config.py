"""
Central config for the EPL Scouting Agent.

Single source of truth for the seasons loaded, the clubs in scope, and the
feature vocabulary the agent composes style profiles from.
"""

# --- Season scope ---
# Seasons loaded, oldest first. build_features.py derives recency weights from
# the order, so add new seasons in chronological order. With one season the
# weight is a constant and cancels out of the weighted average.
HISTORICAL_SEASONS = [
    "2024-2025",
]

# --- Data source ---
# Kaggle: siddhrajthakor/fbref-premier-league-202425-player-stats-dataset
# Single season of FBref standard stats. No scraping — FBref is Cloudflare-blocked.
PLAYER_STATS_CSV = "data/kaggle_raw/fbref_PL_2024-25.csv"
DATA_SEASON = "2024-2025"

# --- Clubs in scope ---
# The 20 clubs that played the 2024-25 Premier League — i.e. every club in the
# dataset. Deliberately NOT the current 2026-27 squads.
#
# Filtering by present-day squads sounds more useful for scouting and is
# actually wrong in both directions, because the dataset has no transfer
# information: a player's club here is the club he played for in 2024-25, not
# where he is now. Scoping to 2026-27 clubs dropped Matheus Cunha and Mohammed
# Kudus — who have since moved TO Premier League clubs — while keeping players
# who have since left the league entirely. 127 players were excluded to buy a
# guarantee the data cannot provide.
#
# So the scope claim this list makes is one the data can support: every player
# who appeared in the 2024-25 Premier League. Add a season's clubs here when
# you add that season's data.
IN_SCOPE_CLUBS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Ipswich Town",
    "Leicester City",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Southampton",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]

# FBref abbreviates club names in the CSV's Squad column. Normalise them onto
# the canonical spellings BEFORE filtering, or the abbreviated clubs silently
# drop out of the dataset.
CLUB_NAME_MAP = {
    "Brighton": "Brighton & Hove Albion",
    "Manchester Utd": "Manchester United",
    "Newcastle Utd": "Newcastle United",
    "Nott'ham Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}

# --- Feature space ---
# The seven per-90 stats build_features.py produces. This IS the entire
# vocabulary available for style matching — the source CSV has no defensive
# data (no tackles, interceptions, pressures, or pass completion %).
#
# The glossary text is fed to the agent so it can translate a plain-English
# tactical ask into weights over these columns. Keep it tactical, not
# statistical — it describes what a stat MEANS on a pitch.
FEATURE_GLOSSARY = {
    "goals_p90": "Goals scored per 90. Actual output, not chance quality.",
    "assists_p90": "Assists per 90. Actual output, not chance quality.",
    "npxg_p90": "Non-penalty expected goals per 90 — how often the player gets "
                "into good shooting positions. The best single measure of goal "
                "threat, and more stable than raw goals.",
    "xa_p90": "Expected assisted goals per 90 — the quality of chances the "
              "player creates for others. The best measure of creativity.",
    "prog_passes_p90": "Progressive passes played per 90 — passes that move the "
                       "ball meaningfully upfield. High for deep playmakers, "
                       "passing centre-backs, and midfielders who dictate from "
                       "deep.",
    "prog_carries_p90": "Progressive carries per 90 — carrying the ball upfield "
                        "at the feet. High for dribblers, ball-carrying "
                        "midfielders, and wingers who drive inside.",
    "prog_passes_received_p90": "Progressive passes RECEIVED per 90 — how often "
                                "the player gets on the end of forward passes. "
                                "High for forwards and wingers who run in behind "
                                "or hold advanced positions; low for deep-lying "
                                "players who play the pass rather than receive it.",
}

FEATURE_COLUMNS = list(FEATURE_GLOSSARY)

# Plain-English name for each stat, for talking to people who do not read
# football analytics. "5.2 prog carries/90" means nothing to most fans, scouts,
# or recruiters; "carries the ball forward — 94th percentile among PL forwards"
# means something to everyone.
FEATURE_LABELS = {
    "goals_p90": "scoring",
    "assists_p90": "setting up goals",
    "npxg_p90": "getting into shooting positions",
    "xa_p90": "creating chances for others",
    "prog_passes_p90": "passing the ball forward",
    "prog_carries_p90": "carrying the ball forward",
    "prog_passes_received_p90": "getting on the end of forward passes",
}

# Percentile -> word. Gives the agent one consistent vocabulary instead of
# inventing a new adjective for every player.
PERCENTILE_BANDS = [
    (95, "elite"),
    (85, "excellent"),
    (70, "strong"),
    (40, "average"),
    (20, "below average"),
    (0, "poor"),
]


def percentile_band(pct: float) -> str:
    for floor, word in PERCENTILE_BANDS:
        if pct >= floor:
            return word
    return "poor"


# --- Style presets ---
# STARTING POINTS, NOT A FIXED MENU. The agent composes its own weight vectors
# over FEATURE_COLUMNS for whatever a user actually asks for — these exist as
# worked examples and as shortcuts for common asks.
#
# Weights run roughly -2..+2 over z-scored features, so a negative weight means
# "this style is defined partly by NOT doing this" (a deep playmaker plays
# progressive passes rather than receiving them).
STYLE_PRESETS = {
    "inverted_fullback": {
        "prog_passes_p90": 1.3,
        "prog_carries_p90": 0.9,
        "prog_passes_received_p90": 0.3,
        "npxg_p90": -0.4,
    },
    "deep_lying_playmaker": {
        "prog_passes_p90": 1.5,
        "xa_p90": 0.5,
        "prog_carries_p90": 0.4,
        "prog_passes_received_p90": -0.6,
        "npxg_p90": -0.5,
    },
    "box_crasher_10": {
        "npxg_p90": 1.2,
        "goals_p90": 1.0,
        "prog_passes_received_p90": 0.9,
        "xa_p90": 0.6,
        "prog_carries_p90": 0.4,
    },
    "creative_winger": {
        "xa_p90": 1.3,
        "prog_carries_p90": 1.1,
        "assists_p90": 1.0,
        "prog_passes_received_p90": 0.9,
        "npxg_p90": 0.4,
    },
    "goal_threat_forward": {
        "npxg_p90": 1.5,
        "goals_p90": 1.3,
        "prog_passes_received_p90": 0.7,
        "xa_p90": 0.2,
        "prog_passes_p90": -0.5,
    },
    "ball_carrier": {
        "prog_carries_p90": 1.5,
        "prog_passes_received_p90": 0.9,
        "xa_p90": 0.4,
        "prog_passes_p90": 0.2,
    },
}

# Backwards-compatible alias — STYLE_PRESETS is the name to use going forward.
ROLE_ARCHETYPES = STYLE_PRESETS

DB_PATH = "data/epl_scout.db"
