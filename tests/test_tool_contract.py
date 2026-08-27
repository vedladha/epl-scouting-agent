"""
The contract `find_players` offers the agent.

Every error path here exists so a confused model gets a message it can act on
instead of a silent zero. If these regress, the agent fails quietly — which is
the failure mode this project keeps having.
"""
import json

import pytest

from agent_tools import _validate_weights, find_players, get_player_report
from config import FEATURE_COLUMNS, FEATURE_LABELS, STYLE_PRESETS
from conftest import needs_db


# ---------- weight validation ----------

def test_unknown_feature_is_rejected_and_the_agent_is_told_what_exists():
    err = _validate_weights({"tackles_p90": 1.0})
    assert err and "tackles_p90" in err["error"]
    assert err["valid_features"] == FEATURE_COLUMNS
    assert "glossary" in err, "the agent needs the glossary to correct itself"


def test_valid_weights_pass_including_negative_ones():
    assert _validate_weights({"npxg_p90": 1.0, "xa_p90": -0.5}) is None


@pytest.mark.parametrize("bad", [{}, None, "npxg_p90", {"npxg_p90": "high"}, {"npxg_p90": 0}])
def test_unusable_weight_payloads_are_rejected(bad):
    assert _validate_weights(bad) is not None


# ---------- mode selection ----------

@needs_db
def test_exactly_one_scoring_mode_is_required():
    for kwargs in (
        {},                                                            # none
        {"style_weights": {"npxg_p90": 1}, "preset": "ball_carrier"},   # two
        {"preset": "ball_carrier", "similar_to_player_id": "x"},        # two
    ):
        assert "error" in find_players(**kwargs)


@needs_db
def test_unknown_preset_lists_the_real_ones():
    err = find_players(preset="libero")
    assert "error" in err and set(err["available_presets"]) == set(STYLE_PRESETS)


@needs_db
def test_unknown_player_explains_the_two_likely_causes():
    err = find_players(similar_to_player_id="Lionel_Messi_Arsenal")
    assert "error" in err and "450" in err["error"]


@needs_db
def test_fine_grained_position_is_rejected_with_guidance():
    """
    'LB' is the intuitive thing for a model to pass and the dataset has no such
    concept. The error has to explain how to express it instead.
    """
    err = find_players(style_weights={"npxg_p90": 1}, position="LB")
    assert "error" in err and "hint" in err
    assert "style_weights" in err["hint"]


# ---------- result shape ----------

@needs_db
def test_results_speak_plain_english_with_a_named_comparison_group():
    res = find_players(style_weights={"prog_carries_p90": 1.5}, position="FW", k=3)
    assert res["results"], "expected candidates"
    top = res["results"][0]
    assert top["compared_against"] == "Premier League forwards"
    stat = top["stats"][0]
    assert set(stat) == {"stat", "per_90", "percentile", "rating"}
    assert "_p90" not in stat["stat"], "the user-facing label leaked a column name"
    assert 0 <= stat["percentile"] <= 100


@needs_db
def test_stats_are_ordered_by_how_much_they_mattered_to_the_query():
    res = find_players(style_weights={"xa_p90": 0.2, "prog_carries_p90": 1.9},
                       position="FW", k=1)
    assert res["results"][0]["stats"][0]["stat"] == FEATURE_LABELS["prog_carries_p90"]


@needs_db
def test_zero_weight_features_are_left_out_of_the_explanation():
    res = find_players(style_weights={"prog_carries_p90": 1.0, "xa_p90": 0.0},
                       position="FW", k=1)
    labels = [s["stat"] for s in res["results"][0]["stats"]]
    assert labels == [FEATURE_LABELS["prog_carries_p90"]]


@needs_db
def test_the_interpretation_is_echoed_back_for_auditing():
    weights = {"prog_carries_p90": 1.5, "xa_p90": -0.4}
    res = find_players(style_weights=weights, position="FW", k=1)
    assert res["interpreted_as"] == weights


@needs_db
def test_filters_actually_filter():
    res = find_players(style_weights={"prog_carries_p90": 1.0},
                       position="DF", max_age=23, min_minutes=1200, k=20)
    for r in res["results"]:
        assert "DF" in r["position"] and r["age"] <= 23 and r["minutes"] >= 1200


@needs_db
def test_a_reference_player_is_never_returned_as_their_own_match():
    ref = "Bukayo_Saka_Arsenal"
    res = find_players(similar_to_player_id=ref, k=20)
    assert ref not in [r["player_id"] for r in res["results"]]


@needs_db
def test_similarity_results_describe_closeness_in_words():
    style = find_players(style_weights={"prog_carries_p90": 1.0}, k=1)
    similar = find_players(similar_to_player_id="Bukayo_Saka_Arsenal", k=1)
    assert "resemblance" not in style["results"][0]
    assert similar["results"][0]["resemblance"] in {"very close", "close", "loose"}


@needs_db
def test_an_impossible_filter_explains_itself_rather_than_returning_nothing():
    res = find_players(style_weights={"npxg_p90": 1}, club="Coventry City")
    assert "error" in res and "2024-25" in res["hint"]


@needs_db
def test_results_are_ranked_best_first():
    res = find_players(style_weights={"prog_carries_p90": 1.0}, k=10)
    assert [r["rank"] for r in res["results"]] == list(range(1, 11))
    # Ordering IS the ranking signal — no score is exposed to check it against.
    tops = res["results"][0]["stats"][0]["percentile"]
    lasts = res["results"][-1]["stats"][0]["percentile"]
    assert tops >= lasts


# ---------- player report ----------

@needs_db
def test_player_report_covers_every_feature_in_plain_english():
    rep = get_player_report("Bukayo_Saka_Arsenal")
    assert rep["name"] == "Bukayo Saka"
    assert len(rep["stats"]) == len(FEATURE_COLUMNS)
    assert all("_p90" not in s["stat"] for s in rep["stats"])
    assert rep["compared_against"].startswith("Premier League")


@needs_db
def test_missing_player_report_points_at_the_lookup_tool():
    rep = get_player_report("Nobody_Anywhere")
    assert "error" in rep and "query_players" in rep["hint"]


# ---------- what the model is NOT allowed to see ----------

INTERNAL_FIELDS = {"fit_score", "raw_fit_score", "sample_reliability",
                   "style_match", "score_type", "score_meaning", "feature"}


@needs_db
def test_no_internal_scoring_fields_reach_the_model():
    """
    The payload must not carry anything the user should never hear.

    These fields used to be sent and the system prompt had to forbid quoting
    each one — a rule the model broke by printing `goals_p90` tables to users.
    Not sending a value is stronger than instructing against repeating it, so
    this guards the payload rather than the prompt.
    """
    res = find_players(style_weights={"prog_carries_p90": 1.5, "npxg_p90": -0.5},
                       position="FW", k=3)
    assert not INTERNAL_FIELDS & set(res), f"envelope leaks: {INTERNAL_FIELDS & set(res)}"
    for entry in res["results"]:
        assert not INTERNAL_FIELDS & set(entry), f"result leaks: {INTERNAL_FIELDS & set(entry)}"
        for stat in entry["stats"]:
            assert not INTERNAL_FIELDS & set(stat), f"stat leaks: {INTERNAL_FIELDS & set(stat)}"


@needs_db
def test_no_column_names_appear_anywhere_in_a_result():
    """A column name the model never receives is a column name it cannot print."""
    res = find_players(style_weights={"prog_carries_p90": 1.5}, position="FW", k=3)
    blob = json.dumps(res["results"])
    for feature in FEATURE_COLUMNS:
        assert feature not in blob, f"{feature} reached the model in a result"


@needs_db
def test_sample_size_is_expressed_in_words():
    res = find_players(style_weights={"prog_carries_p90": 1.0}, position="FW", k=20)
    assert all(isinstance(r["sample"], str) for r in res["results"])
