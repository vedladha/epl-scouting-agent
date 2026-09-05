"""The labeled eval cases must stay loadable and must name real players."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from cases import CaseError, EvalCase, load_cases, referenced_player_ids  # noqa: E402
from conftest import needs_db  # noqa: E402


@pytest.fixture(scope="module")
def cases() -> list[EvalCase]:
    return load_cases()


def test_the_shipped_case_file_loads(cases):
    assert cases


def test_every_mode_is_covered(cases):
    modes = {case.mode for case in cases}
    assert {"rank", "refuse", "similar"} <= modes


def test_refusal_cases_say_what_the_answer_must_mention(cases):
    for case in cases:
        if case.is_refusal:
            assert case.mentions, f"{case.id} has nothing to check the refusal against"


@needs_db
def test_every_referenced_player_exists_and_is_ranked(db, cases):
    ranked = {row["player_id"] for row in
              db.execute("SELECT player_id FROM player_features")}
    missing = sorted(referenced_player_ids(cases) - ranked)
    assert not missing, f"cases name players with no feature vector: {missing}"


@needs_db
def test_shortlisted_players_satisfy_their_own_case_filters(db, cases):
    for case in cases:
        position = case.filters.get("position")
        max_age = case.filters.get("max_age")
        for player_id in case.shortlist:
            row = db.execute(
                """SELECT p.position, p.age, f.total_minutes
                   FROM players p JOIN player_features f USING (player_id)
                   WHERE p.player_id = ?""", (player_id,)).fetchone()
            if position:
                assert position in row["position"], \
                    f"{case.id}: {player_id} is {row['position']}, not {position}"
            if max_age:
                assert row["age"] <= max_age, \
                    f"{case.id}: {player_id} is {row['age']}, over max_age {max_age}"
            min_minutes = case.filters.get("min_minutes")
            if min_minutes:
                assert row["total_minutes"] >= min_minutes, \
                    f"{case.id}: {player_id} played {row['total_minutes']} minutes"


@pytest.mark.parametrize("broken,message", [
    ({"id": "x", "ask": "a", "mode": "rank",
      "weights": {"tackles_p90": 1.0}}, "do not exist"),
    ({"id": "x", "ask": "a", "mode": "rank", "weights": {"goals_p90": 0}}, "zero"),
    ({"id": "x", "ask": "a", "mode": "rank",
      "weights": {"goals_p90": 1}, "filters": {"position": "LB"}}, "position must be"),
    ({"id": "x", "ask": "a", "mode": "refuse"}, "needs mentions"),
    ({"id": "x", "ask": "a", "mode": "similar"}, "needs a similar_to"),
    ({"id": "x", "ask": "a", "mode": "guess", "weights": {"goals_p90": 1}}, "mode must be"),
])
def test_a_malformed_case_is_rejected_with_a_useful_message(broken, message, tmp_path):
    import yaml
    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump({"cases": [broken]}))
    with pytest.raises(CaseError, match=message):
        load_cases(path)


def test_duplicate_ids_are_rejected(tmp_path):
    import yaml
    case = {"id": "same", "ask": "a", "mode": "rank", "weights": {"goals_p90": 1}}
    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump({"cases": [case, dict(case)]}))
    with pytest.raises(CaseError, match="duplicate case id"):
        load_cases(path)
