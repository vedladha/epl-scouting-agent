"""Calibration bookkeeping, checked without calling the judge."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from calibrate import Fixture, calibrate, load_fixtures, report  # noqa: E402
from judge import RULES  # noqa: E402


class ScriptedClient:
    def __init__(self, verdicts: dict):
        self.verdicts = verdicts

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        rules = next((v for k, v in self.verdicts.items() if k in prompt), [])
        payload = json.dumps({"violations": [
            {"rule": rule, "quote": "q", "why": "w"} for rule in rules]})
        return SimpleNamespace(stop_reason="end_turn", model=kwargs.get("model"),
                               content=[SimpleNamespace(type="text", text=payload)])


PLANTED = Fixture("planted", "an ask", "He is naturally left-footed.",
                  ("player_side_or_foot",))
CLEAN = Fixture("clean", "an ask", "He is elite at carrying the ball forward.", ())


def test_the_shipped_fixtures_load_and_name_only_real_rules():
    fixtures = load_fixtures()
    assert fixtures
    for fixture in fixtures:
        assert set(fixture.expect) <= set(RULES)


def test_the_shipped_fixtures_include_clean_answers():
    assert any(f.is_clean for f in load_fixtures())


def test_the_shipped_fixtures_cover_every_rule():
    covered = {rule for fixture in load_fixtures() for rule in fixture.expect}
    assert covered == set(RULES)


def test_a_fixture_expecting_an_unknown_rule_is_rejected(tmp_path):
    import yaml
    path = tmp_path / "f.yaml"
    path.write_text(yaml.safe_dump({"fixtures": [
        {"id": "x", "ask": "a", "answer": "b", "expect": ["vibes"]}]}))
    with pytest.raises(ValueError, match="unknown rules"):
        load_fixtures(path)


def test_fixtures_without_a_clean_answer_are_rejected(tmp_path):
    import yaml
    path = tmp_path / "f.yaml"
    path.write_text(yaml.safe_dump({"fixtures": [
        {"id": "x", "ask": "a", "answer": "b", "expect": ["raw_stat"]}]}))
    with pytest.raises(ValueError, match="clean answers"):
        load_fixtures(path)


def test_a_perfect_judge_scores_perfectly():
    client = ScriptedClient({"naturally left-footed": ["player_side_or_foot"]})
    c = calibrate([PLANTED, CLEAN], client)
    assert c.recall == 1.0 and c.false_alarm_rate == 0.0 and c.with_extras == []


def test_a_judge_that_finds_nothing_scores_zero_recall_not_a_pass():
    c = calibrate([PLANTED, CLEAN], ScriptedClient({}))
    assert c.recall == 0.0 and c.false_alarm_rate == 0.0
    assert [r.fixture.id for r in c.missed] == ["planted"]


def test_a_judge_that_flags_everything_is_caught_by_the_false_alarm_rate():
    client = ScriptedClient({"an ask": ["outside_the_data"]})
    c = calibrate([PLANTED, CLEAN], client)
    assert c.false_alarm_rate == 1.0


def test_an_extra_finding_alongside_the_planted_one_still_counts_as_caught():
    client = ScriptedClient({"naturally left-footed": ["player_side_or_foot", "raw_stat"]})
    c = calibrate([PLANTED, CLEAN], client)
    assert c.recall == 1.0
    assert [r.fixture.id for r in c.with_extras] == ["planted"]
    assert c.results[0].extras == ("raw_stat",)


def test_finding_only_a_different_rule_is_a_miss_not_a_catch():
    client = ScriptedClient({"naturally left-footed": ["raw_stat"]})
    c = calibrate([PLANTED, CLEAN], client)
    assert c.recall == 0.0
    assert [r.fixture.id for r in c.missed] == ["planted"]


def test_a_judge_error_is_recorded_rather_than_counted_as_clean():
    class Broken(ScriptedClient):
        def create(self, **kwargs):
            return SimpleNamespace(stop_reason="refusal", model="m", content=[])

    c = calibrate([PLANTED], Broken({}))
    assert c.results[0].error and c.recall == 0.0


def test_the_report_names_what_was_missed():
    text = report(calibrate([PLANTED, CLEAN], ScriptedClient({})), "some-model")
    assert "MISSED" in text and "planted" in text and "some-model" in text


def test_the_report_marks_extras_without_calling_them_failures():
    client = ScriptedClient({"naturally left-footed": ["player_side_or_foot", "raw_stat"]})
    text = report(calibrate([PLANTED, CLEAN], client), "some-model")
    assert "+extra" in text and "MISSED" not in text
