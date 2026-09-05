"""The runner: what it extracts from a turn, what it scores, what it fails."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from cases import EvalCase  # noqa: E402
from conftest import needs_db  # noqa: E402
from run_eval import (AGENT_MODEL_ENV, extract_calls, final_text,  # noqa: E402
                      report, resolve_agent_model, run_case_live,
                      run_case_offline, summarize)

RANK_CASE = EvalCase(
    id="goalscorer", ask="the most dangerous goalscorer", mode="rank",
    weights={"npxg_p90": 1.5, "goals_p90": 1.3},
    filters={"position": "FW"},
    shortlist=("Mohamed_Salah_Liverpool",),
)

REFUSAL_CASE = EvalCase(id="ball_winner", ask="a ball-winner", mode="refuse",
                        mentions=("tackles",))


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(tool_input, name="find_players"):
    return SimpleNamespace(type="tool_use", name=name, id="tu_1", input=tool_input)


class ScriptedAgent:
    """Replays a fixed sequence of model responses through run_agent_turn."""

    def __init__(self, *responses):
        self.responses = list(responses)

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        content, stop = self.responses.pop(0)
        return SimpleNamespace(content=content, stop_reason=stop)


def ranking_agent(tool_input, answer="Salah leads, elite at scoring."):
    return ScriptedAgent(
        ([tool_block(tool_input)], "tool_use"),
        ([text_block(answer)], "end_turn"),
    )


def test_tool_calls_are_pulled_out_of_the_turn():
    messages = [
        {"role": "user", "content": "an ask"},
        {"role": "assistant", "content": [tool_block({"style_weights": {"goals_p90": 1}})]},
        {"role": "assistant", "content": [tool_block({"x": 1}, name="query_players")]},
    ]
    assert extract_calls(messages) == [{"style_weights": {"goals_p90": 1}}]


def test_the_final_text_is_read_from_the_last_message():
    messages = [{"role": "assistant", "content": [text_block("a"), text_block("b")]}]
    assert final_text(messages) == "a\nb"


def test_final_text_of_an_empty_turn_is_empty():
    assert final_text([]) == ""


@needs_db
def test_the_offline_tier_scores_the_labeled_weights_without_a_model():
    result = run_case_offline(RANK_CASE)
    assert result.ranked and result.passed
    assert "Mohamed_Salah_Liverpool" in result.returned_ids


@needs_db
def test_the_offline_tier_cannot_run_a_refusal_case():
    assert "live tier" in run_case_offline(REFUSAL_CASE).error


@needs_db
def test_a_faithful_translation_passes():
    agent = ranking_agent({"style_weights": {"npxg_p90": 1.4, "goals_p90": 1.1},
                           "position": "FW"})
    result = run_case_live(RANK_CASE, agent)
    assert result.passed
    assert result.weights.direction_accuracy == 1.0 and result.filters.is_exact


@needs_db
def test_an_inverted_weight_fails_the_case():
    agent = ranking_agent({"style_weights": {"npxg_p90": -1.5, "goals_p90": 1.3},
                           "position": "FW"})
    result = run_case_live(RANK_CASE, agent)
    assert not result.passed and result.weights.sign_flips == ("npxg_p90",)


@needs_db
def test_a_dropped_filter_fails_the_case():
    agent = ranking_agent({"style_weights": {"npxg_p90": 1.5, "goals_p90": 1.3}})
    result = run_case_live(RANK_CASE, agent)
    assert not result.passed
    assert result.filters.mismatched == (("position", "FW", None),)


@needs_db
def test_a_leak_in_the_answer_fails_an_otherwise_perfect_case():
    agent = ranking_agent({"style_weights": {"npxg_p90": 1.5, "goals_p90": 1.3},
                           "position": "FW"},
                          answer="Salah is elite, at 0.81 npxG per 90.")
    result = run_case_live(RANK_CASE, agent)
    assert not result.passed
    assert result.leaks and result.leaks[0].kind == "per_90_rate"


def test_ranking_when_the_case_expects_a_refusal_fails():
    agent = ranking_agent({"style_weights": {"goals_p90": 1.0}},
                          answer="No defensive data, but here are some names.")
    result = run_case_live(REFUSAL_CASE, agent)
    assert not result.passed and result.ranked


def test_a_refusal_that_names_the_missing_data_passes():
    agent = ScriptedAgent(
        ([text_block("This data has no tackles at all, so I cannot rank that.")],
         "end_turn"))
    result = run_case_live(REFUSAL_CASE, agent)
    assert result.passed and not result.ranked


def test_a_model_error_is_recorded_against_the_case():
    class Broken(ScriptedAgent):
        def create(self, **kwargs):
            raise RuntimeError("connection reset")

    result = run_case_live(RANK_CASE, Broken())
    assert not result.passed and "connection reset" in result.error


@needs_db
def test_the_summary_averages_only_over_cases_it_can_score():
    results = [run_case_offline(RANK_CASE)]
    summary = summarize(results)
    assert summary["weight_direction_accuracy"] is None
    assert summary["shortlist_recall"] == 1.0
    assert summary["cases"] == 1


@needs_db
def test_the_report_names_the_failure_rather_than_only_counting_it():
    agent = ranking_agent({"style_weights": {"npxg_p90": -1.5, "goals_p90": 1.3},
                           "position": "FW"})
    results = [run_case_live(RANK_CASE, agent)]
    text = report(results, summarize(results), "live")
    assert "FAIL" in text and "wrong direction: npxg_p90" in text


def test_the_agent_model_comes_from_the_environment_when_not_named(monkeypatch):
    monkeypatch.setenv(AGENT_MODEL_ENV, "model-from-the-environment")
    assert resolve_agent_model() == "model-from-the-environment"
    assert resolve_agent_model("named-in-code") == "named-in-code"


def test_no_model_is_forced_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv(AGENT_MODEL_ENV, raising=False)
    assert resolve_agent_model() is None
