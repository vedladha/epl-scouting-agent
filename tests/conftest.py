"""
Shared test setup.

The whole suite is deterministic: no network, no API calls, no cost. It runs
in under a second so there is no excuse not to run it on every change.
Behavioural evaluation of the agent's writing is a separate concern.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DB_PATH, PLAYER_STATS_CSV  # noqa: E402

CSV = ROOT / PLAYER_STATS_CSV
DB = ROOT / DB_PATH

needs_csv = pytest.mark.skipif(not CSV.exists(), reason="Kaggle CSV not downloaded")
needs_db = pytest.mark.skipif(not DB.exists(), reason="run ingest.py then build_features.py")


@pytest.fixture
def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
