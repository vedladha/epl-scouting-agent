-- EPL Scouting Agent schema

CREATE TABLE IF NOT EXISTS players (
    player_id       TEXT PRIMARY KEY,  
    name            TEXT NOT NULL,
    club            TEXT NOT NULL,      -- club played for in the loaded season,
                                        
    position        TEXT,               
    detailed_position TEXT,            
    age             INTEGER,
    nationality     TEXT
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id       TEXT NOT NULL,
    season          TEXT NOT NULL,      
    club            TEXT NOT NULL,      
    minutes         INTEGER,
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
    prog_passes_received REAL,   
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


CREATE TABLE IF NOT EXISTS player_features (
    player_id       TEXT PRIMARY KEY,
    season_span     TEXT,   
    total_minutes   INTEGER,
    feature_json    TEXT,   
    raw_per90_json  TEXT,   
    percentiles_json TEXT,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
