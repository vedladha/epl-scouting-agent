"""The eval scorers themselves, which have to be right before any score means anything."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from score import (find_leaks, score_filters, score_refusal,  # noqa: E402
                   score_shortlist, score_weights)

EXPECTED = {"xa_p90": 1.5, "prog_passes_p90": 1.2,
            "prog_passes_received_p90": -0.8, "npxg_p90": -0.5}


def test_an_exact_translation_scores_perfectly():
    s = score_weights(EXPECTED, dict(EXPECTED))
    assert s.direction_accuracy == 1.0 and s.cosine == pytest.approx(1.0)
    assert s.is_clean


def test_magnitudes_may_differ_as_long_as_the_directions_hold():
    loose = {k: v * 0.4 for k, v in EXPECTED.items()}
    s = score_weights(EXPECTED, loose)
    assert s.direction_accuracy == 1.0 and s.cosine == pytest.approx(1.0)


def test_a_flipped_sign_is_caught_because_it_inverts_the_ask():
    flipped = dict(EXPECTED, prog_passes_received_p90=0.8)
    s = score_weights(EXPECTED, flipped)
    assert s.sign_flips == ("prog_passes_received_p90",)
    assert s.direction_accuracy == 0.75 and not s.is_clean


def test_missing_and_extra_features_are_reported_separately():
    actual = {"xa_p90": 1.5, "prog_passes_p90": 1.2, "goals_p90": 1.0}
    s = score_weights(EXPECTED, actual)
    assert s.missing == ("npxg_p90", "prog_passes_received_p90")
    assert s.extra == ("goals_p90",)


def test_a_zero_weight_counts_as_absent():
    s = score_weights({"xa_p90": 1.0}, {"xa_p90": 1.0, "goals_p90": 0.0})
    assert s.extra == () and s.is_clean


def test_filters_must_match_exactly():
    s = score_filters({"position": "FW", "max_age": 25},
                      {"position": "FW", "max_age": 23})
    assert s.matched == ("position",)
    assert s.mismatched == (("max_age", 25, 23),)
    assert not s.is_exact


def test_a_missing_filter_is_a_mismatch_against_none():
    s = score_filters({"position": "DF"}, {})
    assert s.mismatched == (("position", "DF", None),)


def test_shortlist_recall_counts_expected_players_that_appeared():
    s = score_shortlist(["a", "b", "c"], ["c", "z", "a"])
    assert s.hits == ("a", "c") and s.misses == ("b",)
    assert s.recall == pytest.approx(2 / 3)


def test_a_forbidden_player_appearing_is_recorded():
    s = score_shortlist(["a"], ["a", "haaland"], forbidden=["haaland"])
    assert s.forbidden_hits == ("haaland",) and s.recall == 1.0


def test_a_refusal_passes_only_when_it_ranks_nothing_and_says_why():
    text = "This dataset has no tackles or defensive actions at all."
    assert score_refusal(text, ["tackles", "defensive"], ranked=False).passed


def test_ranking_anyway_fails_the_refusal_even_with_the_right_words():
    text = "There is no defensive data, but here are some tackles-adjacent options."
    assert not score_refusal(text, ["defensive"], ranked=True).passed


def test_a_refusal_that_omits_the_reason_fails():
    s = score_refusal("I cannot help with that.", ["tackles"], ranked=False)
    assert s.refused and s.unmentioned == ("tackles",) and not s.passed


@pytest.mark.parametrize("text,kind", [
    ("He posts 5.2 progressive carries per 90.", "per_90_rate"),
    ("Ranked on npxg_p90 and prog_carries_p90.", "column_name"),
    ("He is 2.75 standard deviations above average.", "statistical_jargon"),
    ("His fit score is the highest in the pool.", "statistical_jargon"),
    ("A naturally left-footed defender.", "footedness"),
    ("He is more of a right-sided player.", "footedness"),
    ("City's price tag will be significant.", "transfer_or_money"),
    ("He is reportedly set to leave for Real Madrid.", "transfer_or_money"),
    ("An injury-disrupted season explains the minutes.", "injury_inference"),
])
def test_each_documented_guardrail_failure_is_detected(text, kind):
    assert kind in {leak.kind for leak in find_leaks(text)}


@pytest.mark.parametrize("text", [
    "The data cannot distinguish left from right, so rule out anyone on the wrong side.",
    "I have no market value or transfer information for these players.",
    "This dataset does not measure tackles, interceptions or pressures.",
    "Low minutes are not evidence of injury here.",
])
def test_stating_a_limitation_is_not_a_leak(text):
    assert find_leaks(text) == []


def test_a_leak_reports_the_sentence_so_a_human_can_judge_it():
    leaks = find_leaks("Elite at carrying. He averages 12.7 per 90.")
    assert len(leaks) == 1
    assert leaks[0].quote == "He averages 12.7 per 90."


def test_clean_scouting_prose_produces_no_leaks():
    text = ("Elite at carrying the ball forward, 96th percentile among Premier "
            "League forwards. Strong at creating chances for others. He played "
            "a full season, so the rates are reliable.")
    assert find_leaks(text) == []
