-- EPL Scouting Agent schema
-- One row per player-season-team (handles mid-season transfers)

CREATE TABLE IF NOT EXISTS players (
    player_id       TEXT PRIMARY KEY,   -- fbref id or normalized name+dob hash
    name            TEXT NOT NULL,
    club            TEXT NOT NULL,      -- club played for in the loaded season,
                                        -- NOT necessarily where they are now
    position        TEXT,               -- e.g. 'DF', 'MF', 'FW', 'GK'
    detailed_position TEXT,             -- e.g. 'LB', 'CB', 'CM', 'ST'
    age             INTEGER,
    nationality     TEXT
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id       TEXT NOT NULL,
    season          TEXT NOT NULL,      -- e.g. '2025-2026'
    club            TEXT NOT NULL,      -- club they played for THAT season
    minutes         INTEGER,
    -- raw counting stats (per-90 computed in build_features.py)
    goals           REAL,
    assists         REAL,
    npxg            REAL,
    xa              REAL,
    shots           REAL,
    key_passes      REAL,
    passes_completed REAL,
    passes_attempted REAL,
    prog_passes     REAL,
    prog_carries    REAL,
    prog_passes_received REAL,   -- FBref PrgR: progressive passes RECEIVED
    passes_into_final_third REAL,
    touches_mid3    REAL,
    touches_att_pen REAL,
    tackles         REAL,
    tackles_att3    REAL,
    interceptions   REAL,
    pressures       REAL,
    pressures_att3  REAL,
    aerials_won     REAL,
    aerials_total   REAL,
    long_passes_completed REAL,
    PRIMARY KEY (player_id, season, club),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- Precomputed per-90 feature vectors, one row per player using their most
-- recent season(s) — populated by build_features.py, consumed by agent_tools.py
CREATE TABLE IF NOT EXISTS player_features (
    player_id       TEXT PRIMARY KEY,
    season_span     TEXT,   -- e.g. '2021-2022:2025-2026' — which seasons fed this vector
    total_minutes   INTEGER,-- sample size behind the vector; also a filter
    feature_json    TEXT,   -- JSON blob of z-scored per-90 stats (for ranking)
    raw_per90_json  TEXT,   -- JSON blob of RAW per-90 stats (for citing real numbers)
    percentiles_json TEXT,  -- percentile rank vs same-position PL players (for explaining them)
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
