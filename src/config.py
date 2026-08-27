"""
Central config for the EPL Scouting Agent.
Edit CURRENT_PL_CLUBS each season — this is the single source of truth
for which players are "in scope" for recommendations.
"""

#Season in scope
HISTORICAL_SEASONS = [
    "2024-2025",
]

PLAYER_STATS_CSV = "data/kaggle_raw/fbref_PL_2024-25.csv"
DATA_SEASON = "2024-2025"

CURRENT_PL_CLUBS = [
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

CLUB_NAME_MAP = {
    "Brighton": "Brighton & Hove Albion",
    "Manchester Utd": "Manchester United",
    "Newcastle Utd": "Newcastle United",
    "Nott'ham Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",          
    "Wolves": "Wolverhampton Wanderers",  
}

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

# Plain-English name for each stat, for folks who do not have in-depth footballing knowledge 
FEATURE_LABELS = {
    "goals_p90": "scoring",
    "assists_p90": "setting up goals",
    "npxg_p90": "getting into shooting positions",
    "xa_p90": "creating chances for others",
    "prog_passes_p90": "passing the ball forward",
    "prog_carries_p90": "carrying the ball forward",
    "prog_passes_received_p90": "getting on the end of forward passes",
}

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


# Style presets 
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

DB_PATH = "data/epl_scout.db"
