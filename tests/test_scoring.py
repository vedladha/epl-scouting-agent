"""
Scoring maths. These pin the two decisions the ranking rests on:
weighted-mean-z instead of cosine, and sample-size shrinkage.
"""
import pytest

from agent_tools import _cosine_sim, _fit_score, _reliability
from config import percentile_band


# ---------- fit score ----------

def test_fit_score_is_a_weighted_mean_of_z_scores():
    z = {"npxg_p90": 2.0, "goals_p90": 1.0}
    assert _fit_score({"npxg_p90": 1.0, "goals_p90": 1.0}, z) == pytest.approx(1.5)


def test_relative_weight_changes_the_result():
    z = {"npxg_p90": 2.0, "goals_p90": 0.0}
    # npxg carries three quarters of the weight -> 0.75 * 2.0
    assert _fit_score({"npxg_p90": 3.0, "goals_p90": 1.0}, z) == pytest.approx(1.5)


def test_negative_weight_rewards_the_player_who_abstains():
    """'Drops deep RATHER THAN poaching' is only expressible via negative weights."""
    weights = {"prog_passes_received_p90": -1.0}
    deep = _fit_score(weights, {"prog_passes_received_p90": -2.0})
    advanced = _fit_score(weights, {"prog_passes_received_p90": 2.0})
    assert deep > advanced and deep == pytest.approx(2.0)


def test_a_missing_feature_counts_as_league_average_rather_than_erroring():
    assert _fit_score({"npxg_p90": 1.0, "xa_p90": 1.0}, {"npxg_p90": 2.0}) == pytest.approx(1.0)


def test_all_zero_weights_return_zero_not_a_zero_division():
    assert _fit_score({"npxg_p90": 0.0}, {"npxg_p90": 5.0}) == 0.0


def test_scoring_is_scale_sensitive_unlike_cosine():
    """
    The reason cosine was replaced: it called these two profiles identical,
    which put Darwin Núñez above Erling Haaland on goal threat.
    """
    weights = {"npxg_p90": 1.0, "goals_p90": 1.0}
    modest = {"npxg_p90": 0.5, "goals_p90": 0.5}
    extreme = {"npxg_p90": 3.0, "goals_p90": 3.0}
    assert _fit_score(weights, extreme) > _fit_score(weights, modest)
    assert _cosine_sim(modest, extreme) == pytest.approx(1.0)


# ---------- sample-size shrinkage ----------

def test_reliability_rises_with_minutes():
    assert _reliability(300) < _reliability(1500) < _reliability(3000)


def test_reliability_stays_between_zero_and_one():
    assert 0 < _reliability(1) and _reliability(100_000) < 1.0


@pytest.mark.parametrize("minutes", [0, None])
def test_no_minutes_means_no_confidence(minutes):
    assert _reliability(minutes) == 0.0


def test_a_cameo_cannot_outrank_a_full_season_on_a_similar_raw_score():
    """
    Regression: a 638-minute striker beat Haaland and Salah on goal threat
    before shrinkage existed. A hot streak over seven games produces a wilder
    per-90 than a full season does.
    """
    assert 2.6 * _reliability(2736) > 3.3 * _reliability(638)


# ---------- cosine, used only for player-to-player similarity ----------

def test_a_profile_is_identical_to_itself():
    v = {"a": 1.0, "b": 2.0, "c": -1.0}
    assert _cosine_sim(v, v) == pytest.approx(1.0)


def test_opposite_profiles_score_minus_one():
    assert _cosine_sim({"a": 1.0, "b": 2.0}, {"a": -1.0, "b": -2.0}) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude_which_is_why_similarity_uses_it():
    assert _cosine_sim({"a": 1.0, "b": 1.0}, {"a": 10.0, "b": 10.0}) == pytest.approx(1.0)


def test_vectors_sharing_no_features_score_zero_rather_than_erroring():
    assert _cosine_sim({"a": 1.0}, {"b": 1.0}) == 0.0


# ---------- percentile wording ----------

@pytest.mark.parametrize("pct,word", [
    (100, "elite"), (95, "elite"), (94, "excellent"), (85, "excellent"),
    (84, "strong"), (70, "strong"), (69, "average"), (40, "average"),
    (39, "below average"), (20, "below average"), (19, "poor"), (0, "poor"),
])
def test_percentile_band_boundaries(pct, word):
    assert percentile_band(pct) == word


def test_bands_never_go_backwards_as_percentile_rises():
    order = ["poor", "below average", "average", "strong", "excellent", "elite"]
    scores = [order.index(percentile_band(p)) for p in range(101)]
    assert scores == sorted(scores)
