"""
Consistency between the config, the feature space, and the agent's prompt.

Everything here is a regression: each one is a bug that shipped, produced no
exception, and quietly degraded the output.
"""
import pytest

from agent_tools import SYSTEM_PROMPT, TOOL_SCHEMAS, describe_interpretation
from config import (FEATURE_COLUMNS, FEATURE_GLOSSARY, FEATURE_LABELS,
                    STYLE_PRESETS, percentile_band)


def test_every_preset_references_only_real_features():
    """
    Regression: the original archetypes were written against column names
    build_features.py never produced, so every archetype scored exactly 0.0
    for every player and nobody noticed.
    """
    for name, weights in STYLE_PRESETS.items():
        unknown = set(weights) - set(FEATURE_COLUMNS)
        assert not unknown, f"preset {name!r} references features that do not exist: {unknown}"


def test_every_feature_has_both_a_glossary_entry_and_a_plain_english_label():
    """Missing glossary -> the agent cannot map words onto it.
    Missing label -> the column name reaches the user."""
    assert set(FEATURE_GLOSSARY) == set(FEATURE_COLUMNS)
    assert set(FEATURE_LABELS) == set(FEATURE_COLUMNS)


def test_labels_do_not_read_like_column_names():
    for feature, label in FEATURE_LABELS.items():
        assert "p90" not in label.lower() and "_" not in label, \
            f"{feature} label is not plain English: {label!r}"


def test_the_tool_description_teaches_the_whole_vocabulary():
    """The agent composes weights from the tool description; a feature absent
    from it is a feature the agent will never use."""
    description = next(t for t in TOOL_SCHEMAS if t["name"] == "find_players")["description"]
    for feature in FEATURE_COLUMNS:
        assert feature in description, f"{feature} is missing from the tool description"


# The prompt is hand-wrapped prose, so search it with whitespace collapsed —
# rewrapping a paragraph must not fail these.
FLAT_PROMPT = " ".join(SYSTEM_PROMPT.lower().split())


@pytest.mark.parametrize("topic,needle", [
    ("defensive data", "no defensive"),
    ("transfers and money", "market values"),
    ("contracts", "contracts"),
    ("injuries", "injur"),
    ("footedness and side", "preferred foot"),
    ("which flank a player uses", "which flank"),
    ("standard deviations", "standard deviations"),
    ("internal score names", "fit_score"),
    ("internal column names", "_p90"),
])
def test_the_prompt_still_carries_each_honesty_rule(topic, needle):
    """
    Every rule here was added after the agent got it wrong in live testing.
    This catches a prompt edit that quietly drops one.
    """
    assert needle in FLAT_PROMPT, f"prompt no longer covers {topic}"


def test_percentile_language_is_preferred_over_standard_deviations():
    assert "percentile" in FLAT_PROMPT


# ---------- the UI's explanation of itself ----------

def test_interpretation_renders_weights_with_a_direction():
    d = describe_interpretation("find_players", {
        "style_weights": {"prog_carries_p90": 1.5, "npxg_p90": -0.5},
        "position": "FW", "max_age": 23})
    labels = {w["label"]: w["direction"] for w in d["weights"]}
    assert labels["carrying the ball forward"] == "more"
    assert labels["getting into shooting positions"] == "less"
    assert "position forwards" in d["filters"] and "age at most 23" in d["filters"]


def test_interpretation_orders_by_importance():
    d = describe_interpretation("find_players", {
        "style_weights": {"xa_p90": 0.3, "prog_carries_p90": 1.8}})
    assert d["weights"][0]["label"] == "carrying the ball forward"


def test_a_preset_is_expanded_so_the_user_sees_real_priorities():
    d = describe_interpretation("find_players", {"preset": "creative_winger"})
    assert d["preset"] == "creative_winger" and d["weights"]


def test_similarity_mode_is_explained_as_a_reference_player():
    d = describe_interpretation("find_players", {"similar_to_player_id": "Bukayo_Saka_Arsenal"})
    assert d["mode"] == "similar" and d["reference"] == "Bukayo Saka Arsenal"


def test_other_tools_produce_no_interpretation_panel():
    assert describe_interpretation("get_player_report", {"player_id": "x"}) is None
