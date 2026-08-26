"""
Agent tools + Claude tool-use orchestration.

The core idea: there is NO fixed menu of roles. A user can ask for any style
in plain English ("a striker who drops deep and creates", "someone who beats
his man and gets crosses in", "a midfielder who breaks lines by carrying").
The agent translates that ask into a WEIGHT VECTOR over the seven per-90
features that actually exist (see config.FEATURE_GLOSSARY), and the scoring
here ranks players against it. config.STYLE_PRESETS are worked examples and
shortcuts, not the vocabulary.

Tools:
  - query_players(filters)        -> roster filtering, no style involved
  - find_players(style_weights|preset|similar_to_player_id, filters, k)
  - get_player_report(player_id)  -> full stat line for one player

Wired to Claude via tool use, including the "challenge/re-justify" loop: the
conversation history is preserved, so a follow-up "why not X instead?" lets
the model re-call tools and respond with fresh justification.
"""
import json
import sqlite3
from contextlib import contextmanager
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (DB_PATH, FEATURE_COLUMNS, FEATURE_GLOSSARY, FEATURE_LABELS,
                    STYLE_PRESETS, percentile_band)

try:
    import anthropic
except ImportError:
    anthropic = None

VALID_POSITIONS = ("GK", "DF", "MF", "FW")


@contextmanager
def _conn():
    """
    Read-only helper. NOTE: `with sqlite3.connect(...)` commits but does NOT
    close the connection — in a long-lived Streamlit session that leaks a
    handle per query, so close it explicitly here.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------- Tool implementations ----------

def query_players(position: str = None, min_age: int = None, max_age: int = None,
                  club: str = None, exclude_club: str = None,
                  min_minutes: int = None, limit: int = 50) -> list[dict]:
    """Filter the current-squad player pool by basic criteria. No style scoring."""
    sql = ["""SELECT p.player_id, p.name, p.club, p.position, p.age,
                     f.total_minutes
              FROM players p
              LEFT JOIN player_features f ON p.player_id = f.player_id
              WHERE 1=1"""]
    params = []
    if position:
        sql.append("AND p.position LIKE ?")
        params.append(f"%{position.upper()}%")
    if min_age:
        sql.append("AND p.age >= ?")
        params.append(min_age)
    if max_age:
        sql.append("AND p.age <= ?")
        params.append(max_age)
    if club:
        sql.append("AND p.club = ?")
        params.append(club)
    if exclude_club:
        sql.append("AND p.club != ?")
        params.append(exclude_club)
    if min_minutes:
        sql.append("AND f.total_minutes >= ?")
        params.append(min_minutes)
    sql.append("ORDER BY f.total_minutes DESC NULLS LAST LIMIT ?")
    params.append(limit)

    with _conn() as conn:
        return [dict(r) for r in conn.execute(" ".join(sql), params)]


def _validate_weights(style_weights: dict) -> dict | None:
    """Return an error payload if the weights are unusable, else None."""
    if not isinstance(style_weights, dict) or not style_weights:
        return {"error": "style_weights must be a non-empty object mapping "
                         "feature names to numeric weights.",
                "valid_features": FEATURE_COLUMNS}

    unknown = [k for k in style_weights if k not in FEATURE_COLUMNS]
    if unknown:
        return {"error": f"Unknown feature(s): {unknown}. These are the ONLY "
                         f"stats available — the dataset has no defensive or "
                         f"passing-accuracy data. Re-express the style using "
                         f"the valid features.",
                "valid_features": FEATURE_COLUMNS,
                "glossary": FEATURE_GLOSSARY}

    non_numeric = [k for k, v in style_weights.items() if not isinstance(v, (int, float))]
    if non_numeric:
        return {"error": f"Weights must be numbers. Non-numeric: {non_numeric}"}

    if all(float(v) == 0 for v in style_weights.values()):
        return {"error": "All weights are zero — nothing to rank by."}

    return None


def _load_pool(position=None, min_age=None, max_age=None, club=None,
               exclude_club=None, min_minutes=None) -> list[sqlite3.Row]:
    sql = ["""SELECT p.player_id, p.name, p.club, p.position, p.age,
                     p.detailed_position, f.total_minutes, f.feature_json,
                     f.raw_per90_json, f.percentiles_json
              FROM players p JOIN player_features f ON p.player_id = f.player_id
              WHERE 1=1"""]
    params = []
    if position:
        sql.append("AND p.position LIKE ?")
        params.append(f"%{position.upper()}%")
    if min_age:
        sql.append("AND p.age >= ?")
        params.append(min_age)
    if max_age:
        sql.append("AND p.age <= ?")
        params.append(max_age)
    if club:
        sql.append("AND p.club = ?")
        params.append(club)
    if exclude_club:
        sql.append("AND p.club != ?")
        params.append(exclude_club)
    if min_minutes:
        sql.append("AND f.total_minutes >= ?")
        params.append(min_minutes)

    with _conn() as conn:
        return list(conn.execute(" ".join(sql), params))


# Minutes at which a player's stats are considered half-reliable. A per-90 rate
# from 500 minutes is mostly noise; from 3000 it is mostly signal. Roughly ten
# full matches.
RELIABILITY_MINUTES = 900


def _reliability(minutes: int | None) -> float:
    """
    Shrinkage factor in (0, 1): minutes / (minutes + RELIABILITY_MINUTES).

    Without this, every query surfaces cameo players — a striker with 638
    minutes outscored Haaland and Salah on "most dangerous goalscorer" purely
    because a hot streak over seven matches produces a wilder per-90 than a
    full season does. Shrinking the score toward zero in proportion to sample
    size is the standard fix and keeps small-sample players visible but honest.

    Applied only to fit_score. It scales a player's whole z-vector uniformly,
    so it would cancel out of a cosine similarity anyway.
    """
    if not minutes or minutes <= 0:
        return 0.0
    return minutes / (minutes + RELIABILITY_MINUTES)


def _fit_score(weights: dict, z_scores: dict) -> float:
    """
    Weighted average z-score across the requested features.

    Deliberately NOT cosine similarity. Cosine measures the SHAPE of a profile
    and ignores magnitude, so it ranks a modest player whose profile points the
    right way above an extreme one — asking for a goal threat returned Darwin
    Núñez above Erling Haaland. A weighted mean z-score rewards being far in
    the direction asked for, which is what "find me the best X" means.

    Scale is interpretable: +1.5 means "on average 1.5 standard deviations
    above a Premier League regular in the traits you asked for".
    """
    total_weight = sum(abs(float(w)) for w in weights.values())
    if not total_weight:
        return 0.0
    score = sum(float(w) * z_scores.get(feat, 0.0) for feat, w in weights.items())
    return score / total_weight


def _cosine_sim(a: dict, b: dict) -> float:
    """Direction-only similarity — the right metric for player-to-player style."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0
    va = np.array([a[k] for k in keys])
    vb = np.array([b[k] for k in keys])
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


POSITION_NAMES = {"GK": "goalkeepers", "DF": "defenders",
                  "MF": "midfielders", "FW": "forwards"}


def _stat_lines(row: sqlite3.Row, features: list[str]) -> list[dict]:
    """
    Explain each relevant stat in terms a non-analyst can act on.

    A raw per-90 rate is meaningless without a reference class — "5.2
    progressive carries per 90" is unremarkable for a winger and extraordinary
    for a centre-back, and nobody outside analytics knows which. So every stat
    ships as: a plain-English name, the number, its percentile against
    same-position PL players, and a word for that percentile.
    """
    raw = json.loads(row["raw_per90_json"])
    pcts = json.loads(row["percentiles_json"])
    lines = []
    for f in features:
        pct = pcts.get(f, 50)
        lines.append({
            "stat": FEATURE_LABELS.get(f, f),
            "feature": f,                     # machine name, for follow-up calls
            "per_90": raw.get(f),
            "percentile": pct,
            "rating": percentile_band(pct),
        })
    return lines


def _result_row(row: sqlite3.Row, score_name: str, score: float,
                features: list[str], extra: dict = None) -> dict:
    """One ranked player, carrying the real numbers behind the score."""
    group = (row["detailed_position"] or "UNK").upper()
    return {
        "player_id": row["player_id"],
        "name": row["name"],
        "club": row["club"],
        "position": row["position"],
        "age": row["age"],
        "minutes": row["total_minutes"],
        score_name: round(score, 3),
        # Percentiles are vs this group, so the agent can name the comparison.
        "compared_against": f"Premier League {POSITION_NAMES.get(group, 'players')}",
        "stats": _stat_lines(row, features),
        **(extra or {}),
    }


def find_players(style_weights: dict = None, preset: str = None,
                 similar_to_player_id: str = None, position: str = None,
                 min_age: int = None, max_age: int = None, club: str = None,
                 exclude_club: str = None, min_minutes: int = None,
                 exclude_ids: list[str] = None, k: int = 8) -> dict:
    """
    Rank players by style fit. Exactly one of style_weights / preset /
    similar_to_player_id defines what "fit" means.

    style_weights is the general case: any plain-English ask, expressed as
    weights over config.FEATURE_COLUMNS. preset looks up a worked example from
    config.STYLE_PRESETS. similar_to_player_id finds players whose overall
    profile resembles a named player.
    """
    modes = [m for m in (style_weights, preset, similar_to_player_id) if m]
    if len(modes) != 1:
        return {"error": "Provide exactly one of style_weights, preset, or "
                         "similar_to_player_id.",
                "available_presets": list(STYLE_PRESETS),
                "valid_features": FEATURE_COLUMNS}

    if position and position.upper() not in VALID_POSITIONS:
        return {"error": f"position must be one of {list(VALID_POSITIONS)} "
                         f"(the dataset has no finer detail than that — there "
                         f"is no 'LB' or 'CM'). Got {position!r}.",
                "hint": "For a left-back ask, filter position='DF' and express "
                        "the left-back part through style_weights."}

    exclude = set(exclude_ids or [])
    reference = None

    if preset:
        if preset not in STYLE_PRESETS:
            return {"error": f"Unknown preset {preset!r}.",
                    "available_presets": list(STYLE_PRESETS),
                    "hint": "Presets are only shortcuts — for anything else, "
                            "build style_weights yourself."}
        style_weights = STYLE_PRESETS[preset]

    if similar_to_player_id:
        with _conn() as conn:
            ref = conn.execute(
                "SELECT feature_json, raw_per90_json FROM player_features WHERE player_id = ?",
                (similar_to_player_id,),
            ).fetchone()
        if ref is None:
            return {"error": f"No feature vector for {similar_to_player_id!r}. "
                             f"Either the player isn't in a current PL squad, or "
                             f"they played under 450 minutes in 2024-25.",
                    "hint": "Use query_players to find the exact player_id."}
        reference = json.loads(ref["feature_json"])
        exclude.add(similar_to_player_id)
        features = FEATURE_COLUMNS
        score_name = "style_match"
    else:
        err = _validate_weights(style_weights)
        if err:
            return err
        features = sorted((f for f, w in style_weights.items() if float(w) != 0),
                          key=lambda f: -abs(float(style_weights[f])))
        score_name = "fit_score"

    pool = _load_pool(position, min_age, max_age, club, exclude_club, min_minutes)
    if not pool:
        return {"error": "No players matched those filters.",
                "hint": "Loosen the filters. Note that club names must match "
                        "the 2024-25 Premier League — a club promoted since "
                        "then has no rows.",
                "results": []}

    scored = []
    for row in pool:
        if row["player_id"] in exclude:
            continue
        z = json.loads(row["feature_json"])
        if reference:
            scored.append(_result_row(row, score_name, _cosine_sim(reference, z), features))
            continue
        unshrunk = _fit_score(style_weights, z)
        reliability = _reliability(row["total_minutes"])
        scored.append(_result_row(
            row, score_name, unshrunk * reliability, features,
            extra={"raw_fit_score": round(unshrunk, 3),
                   "sample_reliability": round(reliability, 2)},
        ))

    scored.sort(key=lambda r: -r[score_name])

    return {
        # Echo the interpretation back so the agent can explain its reasoning
        # and the user can see what the plain-English ask was turned into.
        "interpreted_as": ({"similar_to": similar_to_player_id}
                           if reference else dict(style_weights)),
        "preset_used": preset,
        "filters": {"position": position, "min_age": min_age, "max_age": max_age,
                    "club": club, "exclude_club": exclude_club,
                    "min_minutes": min_minutes},
        "pool_size": len(scored),
        "score_type": score_name,
        "score_meaning": ("cosine similarity of full style profile, -1 to 1"
                          if reference else
                          "weighted mean z-score vs Premier League regulars, "
                          "shrunk toward 0 by sample size. raw_fit_score is the "
                          "unshrunk value and sample_reliability the factor "
                          "applied; a low reliability means the per90 numbers "
                          "come from a small number of minutes and should be "
                          "quoted with that caveat"),
        "results": scored[:k],
    }


def get_player_report(player_id: str) -> dict:
    """Full stat line for one player — season totals, per-90s, and z-scores."""
    with _conn() as conn:
        player = conn.execute(
            """SELECT name, club, position, age, nationality
               FROM players WHERE player_id = ?""", (player_id,)).fetchone()
        if player is None:
            return {"error": f"player_id {player_id!r} not found",
                    "hint": "Use query_players to look up the exact player_id."}
        stats = conn.execute(
            """SELECT season, club, minutes, goals, assists, npxg, xa,
                      prog_passes, prog_carries, prog_passes_received
               FROM player_season_stats WHERE player_id = ? ORDER BY season""",
            (player_id,)).fetchall()
        feat = conn.execute(
            """SELECT p.detailed_position, f.total_minutes, f.season_span,
                      f.feature_json, f.raw_per90_json, f.percentiles_json
               FROM player_features f JOIN players p ON p.player_id = f.player_id
               WHERE f.player_id = ?""", (player_id,)).fetchone()

    report = {
        "player_id": player_id,
        "name": player["name"], "club": player["club"],
        "position": player["position"], "age": player["age"],
        "nationality": player["nationality"],
        "season_totals": [dict(s) for s in stats],
    }
    if feat is None:
        report["note"] = ("No feature vector — under the 450-minute floor, so "
                          "this player cannot be style-ranked. Season totals "
                          "above are still real.")
    else:
        group = (feat["detailed_position"] or "UNK").upper()
        report["minutes"] = feat["total_minutes"]
        report["season_span"] = feat["season_span"]
        report["compared_against"] = f"Premier League {POSITION_NAMES.get(group, 'players')}"
        report["stats"] = _stat_lines(feat, FEATURE_COLUMNS)
    return report


def describe_interpretation(tool_name: str, tool_input: dict) -> dict | None:
    """
    Turn a find_players call into something a human can read.

    The whole premise of this tool is that a plain-English ask becomes a set of
    measurable weights. That translation is the part a user most needs to see
    and argue with — if it stays inside the tool call, the ranking is just
    another black box that happens to be worded nicely.
    """
    if tool_name != "find_players":
        return None

    if tool_input.get("similar_to_player_id"):
        return {"mode": "similar",
                "reference": tool_input["similar_to_player_id"].replace("_", " "),
                "filters": _readable_filters(tool_input)}

    weights = tool_input.get("style_weights")
    preset = tool_input.get("preset")
    if preset and not weights:
        weights = STYLE_PRESETS.get(preset, {})
    if not weights:
        return None

    rows = []
    for feat, weight in sorted(weights.items(), key=lambda kv: -abs(float(kv[1]))):
        weight = float(weight)
        if weight == 0:
            continue
        rows.append({
            "label": FEATURE_LABELS.get(feat, feat),
            "weight": weight,
            # A bar is easier to read at a glance than a signed decimal.
            "bar": ("▇" * max(1, round(abs(weight) * 2)))[:6],
            "direction": "more" if weight > 0 else "less",
        })
    return {"mode": "preset" if preset else "style", "preset": preset,
            "weights": rows, "filters": _readable_filters(tool_input)}


def _readable_filters(tool_input: dict) -> list[str]:
    labels = {"position": "position", "min_age": "age at least",
              "max_age": "age at most", "club": "club",
              "exclude_club": "excluding", "min_minutes": "minutes played at least"}
    out = []
    for key, label in labels.items():
        value = tool_input.get(key)
        if value:
            if key == "position":
                value = POSITION_NAMES.get(str(value).upper(), value)
            out.append(f"{label} {value}")
    return out


# ---------- Claude tool-use wiring ----------

_GLOSSARY_TEXT = "\n".join(f"  - {feat}: {desc}" for feat, desc in FEATURE_GLOSSARY.items())
_PRESET_TEXT = "\n".join(f"  - {name}: {weights}" for name, weights in STYLE_PRESETS.items())

TOOL_SCHEMAS = [
    {
        "name": "query_players",
        "description": (
            "Filter the current Premier League player pool by position, age, "
            "club, or minutes played. No style scoring — use this to look up a "
            "player_id, check who is available, or size a candidate pool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {"type": "string", "enum": list(VALID_POSITIONS),
                             "description": "Broad position only — the dataset "
                                            "has no LB/CB/CM detail."},
                "min_age": {"type": "integer"},
                "max_age": {"type": "integer"},
                "club": {"type": "string"},
                "exclude_club": {"type": "string"},
                "min_minutes": {"type": "integer"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "find_players",
        "description": (
            "Rank players by how well they fit a playing style. This is the "
            "main tool.\n\n"
            "THERE IS NO FIXED LIST OF ROLES. Translate whatever the user asks "
            "for into `style_weights`: an object mapping feature names to "
            "weights, roughly -2 to +2 over these seven stats (the only ones "
            "that exist):\n\n"
            f"{_GLOSSARY_TEXT}\n\n"
            "Include only the features you actually mean. A negative weight "
            "means the style is defined partly by NOT doing that thing.\n\n"
            "Worked examples of the same idea:\n"
            f"{_PRESET_TEXT}\n\n"
            "Use `preset` as a shortcut when one of those matches exactly. Use "
            "`similar_to_player_id` for 'find me another X' asks. Combine any "
            "of them with `position` and the other filters.\n\n"
            "IMPORTANT: this dataset has NO defensive stats (no tackles, "
            "interceptions, pressures, duels) and no pass-completion data. If "
            "the user asks for a destroyer, a no-nonsense centre-back, or a "
            "presser, say plainly that the data cannot rank that, rather than "
            "substituting a proxy and implying it measures defending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "style_weights": {
                    "type": "object",
                    "description": "Feature name -> numeric weight. The general "
                                   "case; build this from the user's words.",
                    "additionalProperties": {"type": "number"},
                },
                "preset": {"type": "string", "enum": list(STYLE_PRESETS),
                           "description": "Shortcut for a common ask. Optional."},
                "similar_to_player_id": {
                    "type": "string",
                    "description": "Rank by resemblance to this player's overall profile."},
                "position": {"type": "string", "enum": list(VALID_POSITIONS)},
                "min_age": {"type": "integer"},
                "max_age": {"type": "integer"},
                "club": {"type": "string"},
                "exclude_club": {"type": "string"},
                "min_minutes": {"type": "integer",
                                "description": "Raise above the 450 default to "
                                               "demand a bigger sample."},
                "exclude_ids": {"type": "array", "items": {"type": "string"}},
                "k": {"type": "integer", "default": 8},
            },
        },
    },
    {
        "name": "get_player_report",
        "description": ("Full stat line for one player: season totals, per-90 "
                        "rates, and how far above/below league average each "
                        "stat is. Use this to justify or challenge a pick."),
        "input_schema": {
            "type": "object",
            "properties": {"player_id": {"type": "string"}},
            "required": ["player_id"],
        },
    },
]

DISPATCH = {
    "query_players": query_players,
    "find_players": find_players,
    "get_player_report": get_player_report,
}

SYSTEM_PROMPT = f"""You are an autonomous football scouting analyst for Premier League clubs.

A user describes a tactical need in their own words. Your job is to turn that
description into a measurable style profile, rank real players against it, and
justify the shortlist with real numbers.

## How to translate a request

There is no menu of roles. Read what the user actually asked for and express it
as weights over these seven per-90 stats — the only ones in the dataset:

{_GLOSSARY_TEXT}

Then call find_players with those style_weights. Weight roughly -2 to +2,
include only what you mean, and use negative weights for traits the style is
defined by NOT having.

Examples of the reasoning:
- "a striker who drops deep and creates" -> the creating and receiving matter,
  the pure poaching does not: xa_p90 high, prog_passes_p90 positive,
  prog_passes_received_p90 negative (they come short rather than run beyond),
  npxg_p90 mildly positive. Filter position='FW'.
- "a left-back who inverts into midfield" -> position='DF' plus
  prog_passes_p90 and prog_carries_p90 high, npxg_p90 negative.
- "someone to replace Saka" -> use similar_to_player_id.

Always pass a position filter when the ask implies one. The dataset only knows
GK/DF/MF/FW, so express the finer detail (left-back, number 8, false nine)
through the weights.

## Honesty rules

- Never invent a number. Every stat you quote must come from a tool result.
- Stick to what the data says. You have NO information about transfers, market
  values, contracts, availability, injuries, or price tags — do not speculate
  about them even in passing, and do not use football knowledge from memory as
  if it were evidence. In particular, LOW MINUTES ARE NOT EVIDENCE OF INJURY.
  A player may have been rotated, benched, suspended, or signed mid-season.
  Report the minutes and stop there.
- NEVER state a player's preferred foot, or which flank or side they play. The
  dataset has no footedness and no left/right information — only GK/DF/MF/FW.
  Saying "naturally left-sided" or "right-footed" is recalled trivia dressed as
  analysis, and it is wrong often enough to be dangerous. Do not hedge it
  either ("likely left-footed"); simply do not make the claim.
  This holds in EVERY framing, not just direct description. Do not smuggle it
  into an aside, a caveat, a footnote, an "also worth noting", an honourable
  mention, or advice to the reader ("he is more of a right-sided player, flag
  him to yourself accordingly"). If a sentence would tell the reader which side
  or foot a specific player favours, do not write it.
  When the request depends on side — a LEFT-back, a LEFT winger — say so
  plainly ONCE, near the shortlist: the data cannot distinguish left from
  right, so the list is every defender/winger who plays this WAY, and the user
  should rule out anyone on the wrong side. Say it about the DATASET, never
  about a named player. That is a real limitation of the tool; surface it
  rather than papering over it with guesses. Saying "player X is reportedly leaving" or "he'll be
  expensive" inside a data-backed report is exactly the failure this tool
  exists to avoid. If availability matters to the user, say it's outside what
  you can see and let them supply it.
- Ages come from the 2024-25 season and are now two years stale. Write them as
  "24 (2024-25)" or similar — never present them as the player's current age.
- The dataset covers the 2024-25 Premier League season only, and has NO
  defensive data: no tackles, interceptions, pressures, duels, or pass
  completion. If the user asks for a ball-winner, a dominant defender, or a
  pressing monster, say directly that this data cannot measure that. Do not
  substitute an attacking proxy and let it pass as defending.
- The data covers everyone who played in the 2024-25 Premier League, and a
  player's club is where he played THAT season — not necessarily where he is
  now. There is no transfer information, so never imply a listed club is
  current. If the user asks about a club that was not in the 2024-25 PL, say
  it is not in the data rather than returning nothing.
- Write for a fan, scout, or recruiter — NOT for an analyst. Most people cannot
  read a raw per-90 rate: "5.2 progressive carries per 90" tells them nothing,
  because they have no idea whether that is good. Every stat in a tool result
  therefore arrives with a plain-English name, a percentile against
  same-position PL players, and a rating word. Use them.
  - Lead with the plain-English name and the rating, name the comparison group,
    and put the number last if at all:
      GOOD: "elite at carrying the ball forward — 96th percentile among PL
             forwards (5.2 progressive carries per 90)"
      BAD:  "5.204 prog carries/90"
      BAD:  "2.75 standard deviations above average"
  - NEVER mention standard deviations, z-scores, fit_score, raw_fit_score,
    sample_reliability, style_weights, or shrinkage to the user. Not the
    numbers, not the names, not an explanation of how they work. They are
    internal ranking machinery; percentiles do the same job in language
    everyone already understands. If a player ranks low because of a small
    sample, say "only 900 minutes last season, so treat these numbers as
    provisional" — describe the minutes, never the mechanism.
  - Round numbers sensibly. Two decimals at most, usually one.
  - NEVER print the internal column names — `goals_p90`, `npxg_p90`,
    `prog_passes_received_p90` and the rest are database identifiers, not
    English. This applies when listing what the data does and does not cover
    too: say "chance creation" and "ball progression", never a table of
    `xa_p90` and `prog_carries_p90`. If you would not say it out loud to a
    coach, do not put it on the screen.
- Sample size is already handled for you: fit_score is shrunk toward zero for
  players with few minutes, and each result carries sample_reliability plus the
  unshrunk raw_fit_score. When you recommend anyone below about 1000 minutes,
  say so explicitly — their per90 numbers are real but volatile.

## Output

A ranked shortlist, each with 1-2 sentences of justification citing specific
per-90 numbers. Be concise — this is a scouting note, not an essay. Everything
you write must trace back to a tool result. When challenged ("why not X instead?"), call
get_player_report or find_players again to examine the alternative on the data,
and genuinely update your ranking if the numbers support it.
"""


def run_agent_turn(client: "anthropic.Anthropic", messages: list[dict],
                   model: str = "claude-sonnet-4-6") -> list[dict]:
    """
    Runs one full agent turn: calls Claude, executes any tool calls, feeds
    results back, and loops until Claude returns a final text response.
    Returns the updated messages list (append this to your conversation state).
    """
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = DISPATCH.get(block.name)
                try:
                    result = fn(**block.input) if fn else {"error": f"unknown tool {block.name}"}
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
        # All tool_results for one assistant turn must go back in ONE user
        # message, or Claude stops making parallel calls.
        messages.append({"role": "user", "content": tool_results})

    return messages
