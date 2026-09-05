"""The judge's prompt construction and response handling, with no API calls."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from judge import (DEFAULT_JUDGE_MODEL, JUDGE_MODEL_ENV, OUTPUT_SCHEMA,  # noqa: E402
                   RULES, JudgeError, build_prompt, judge_answer,
                   parse_verdict, resolve_model)


class FakeClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.calls = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            model=kwargs.get("model"),
            content=[SimpleNamespace(type="text", text=self.payload)],
        )


def payload(*violations) -> str:
    return json.dumps({"violations": list(violations)})


def test_a_clean_answer_passes():
    verdict = parse_verdict(payload())
    assert verdict.passed and verdict.violations == ()


def test_a_violation_carries_the_rule_the_quote_and_the_reason():
    verdict = parse_verdict(payload({
        "rule": "player_side_or_foot",
        "quote": "He favours his stronger side when cutting inside.",
        "why": "Attributes a side preference to a named player.",
    }))
    assert not verdict.passed
    assert verdict.rules_broken == ("player_side_or_foot",)
    assert verdict.violations[0].quote.startswith("He favours")


def test_an_invented_rule_name_is_rejected_rather_than_recorded():
    with pytest.raises(JudgeError, match="invented a rule name"):
        parse_verdict(payload({"rule": "vibes", "quote": "x", "why": "y"}))


def test_unparseable_output_is_rejected():
    with pytest.raises(JudgeError, match="unparseable JSON"):
        parse_verdict("I think the answer looks fine, actually.")


def test_a_missing_violations_list_is_rejected():
    with pytest.raises(JudgeError, match="violations"):
        parse_verdict(json.dumps({"result": "clean"}))


def test_a_refusal_from_the_judge_is_surfaced_not_read_as_a_pass():
    client = FakeClient(payload(), stop_reason="refusal")
    with pytest.raises(JudgeError, match="declined"):
        judge_answer("an ask", "an answer", client=client)


def test_an_empty_answer_costs_no_api_call():
    client = FakeClient(payload())
    assert judge_answer("an ask", "   ", client=client).passed
    assert client.calls == []


def test_the_request_carries_the_output_schema():
    client = FakeClient(payload())
    judge_answer("an ask", "an answer", client=client)
    assert client.calls[0]["output_config"]["format"]["schema"] == OUTPUT_SCHEMA


def test_any_model_can_be_named_per_call():
    client = FakeClient(payload())
    verdict = judge_answer("an ask", "an answer", client=client, model="some-other-model")
    assert client.calls[0]["model"] == "some-other-model"
    assert verdict.model == "some-other-model"


def test_the_environment_can_choose_the_judge_model(monkeypatch):
    monkeypatch.setenv(JUDGE_MODEL_ENV, "model-from-the-environment")
    client = FakeClient(payload())
    judge_answer("an ask", "an answer", client=client)
    assert client.calls[0]["model"] == "model-from-the-environment"


def test_an_explicit_model_outranks_the_environment(monkeypatch):
    monkeypatch.setenv(JUDGE_MODEL_ENV, "model-from-the-environment")
    assert resolve_model("named-in-code") == "named-in-code"


def test_the_default_applies_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    assert resolve_model() == DEFAULT_JUDGE_MODEL


def test_the_schema_only_admits_the_documented_rules():
    rule_enum = OUTPUT_SCHEMA["properties"]["violations"]["items"]["properties"]["rule"]
    assert rule_enum["enum"] == sorted(RULES)


def test_the_prompt_carries_the_rules_the_ask_and_the_answer():
    prompt = build_prompt("find me a left-back", "Here is a shortlist.")
    assert "find me a left-back" in prompt and "Here is a shortlist." in prompt
    for rule in RULES:
        assert rule in prompt


def test_evidence_is_only_included_when_supplied():
    assert "traceable" not in build_prompt("ask", "answer")
    assert "traceable" in build_prompt("ask", "answer", evidence='{"percentile": 96}')
