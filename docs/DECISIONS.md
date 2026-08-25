# Design decisions

Why this project is built the way it is: the constraints that shaped it,
the approaches that were tried and abandoned, and the traps in the data
that fail silently. Read alongside the README.

## Goal
An autonomous scouting agent: given a tactical need in plain English
(e.g. "we need a left-back who can invert into midfield"), it queries player
stats, ranks candidates by style fit, generates a data-backed comparison
report, and can be challenged ("why not X instead?") to re-justify with data.

Portfolio project, 3-day build timeline. Validation plan: test with soccer
fans/recruiters/coaches via a Listen Labs contact.

## Scope decisions (final, do not re-litigate unless asked)
- **League**: Premier League only.
- **Squad scope**: current 2026-27 PL clubs only (20 teams) — NOT every
  player who's ever played in the PL. See `src/config.py` `CURRENT_PL_CLUBS`
  for the exact 20 (17 retained from 2025-26 + Coventry City, Ipswich Town,
  Hull City promoted; West Ham, Burnley, Wolves relegated out).
- **Historical depth**: originally planned 5 seasons, recency-weighted.
  **Reality check**: currently single-season only (2024-25) due to data
  source constraints — see "Data source" below. `build_features.py` still has
  the recency-weighting logic in place (harmless with one season — the weight
  is constant and cancels out of the weighted average), ready if more seasons
  get added later.
- **Build order**: single-season MVP first — this is DONE and verified against
  real data. 5-season extension remains a stretch goal only.
- **2026-27 season note**: the working dataset is 2024-25, two seasons behind
  the squad scope. Accepted tradeoff, not an error.

## Tech stack (decided, already scaffolded)
- Python + pandas, SQLite storage
- Per-90 stat feature vectors + cosine similarity for style matching
  (deliberately NOT using learned embeddings — explainability > sophistication
  for a 3-day build)
- Claude (Anthropic API, tool use / function calling) as the agent orchestrator
- Streamlit for the chat UI

## Data source — RESOLVED
Original plan was to scrape FBref via `soccerdata`. **Confirmed blocked**:
FBref is behind Cloudflare, returns 403 "Just a moment..." to all scraping
attempts. Do not revisit scraping FBref directly.

Went through two Kaggle dead-ends before landing on a working dataset:
1. `danielijezie/premier-league-data-from-2016-to-2024` — team/league-table
   level only, no player data. Deleted.
2. `furkanark/premier-league-2024-2025-data`'s `Squad_PlayerStats__stats_standard.csv`
   — real player-level data (574 rows) but only goals/assists/minutes/cards,
   no xG or progression stats. Superseded by #3.
3. **CURRENT / IN USE**: `siddhrajthakor/fbref-premier-league-202425-player-stats-dataset`
   → `data/kaggle_raw/fbref_PL_2024-25.csv`. 574 players, single season
   (2024-25 only). Has goals, assists, xG, npxG, xAG, and — critically —
   progressive carries/passes/passes-received (PrgC/PrgP/PrgR), which is what
   makes style-matching for positional/creative roles possible.

### Two gotchas in this CSV that will bite you silently
1. **Duplicate column names.** The header repeats ten stat names at the end as
   their per-90 versions, so pandas de-duplicates them to `Gls.1`, `xG.1`,
   `npxG.1`, `xAG.1` etc. `ingest.py`'s `COLUMN_MAP` deliberately uses the
   UNSUFFIXED names (season totals) because `build_features.py` divides by
   minutes itself. Reading the `.1` columns would per-90 the value twice and
   produce quietly-wrong numbers rather than an error.
   `_assert_season_totals()` in `ingest.py` guards this.
2. **FBref short club names.** The `Squad` column says `Brighton`,
   `Manchester Utd`, `Newcastle Utd`, `Nott'ham Forest`, `Tottenham` — none of
   which match `CURRENT_PL_CLUBS`. `config.CLUB_NAME_MAP` normalizes them
   BEFORE filtering. Without it, five clubs silently vanish from the dataset.

### Clubs with zero rows (expected, not a bug)
Four in-scope clubs weren't in the PL in 2024-25 — see
`config.CLUBS_MISSING_FROM_DATA`:
- **Coventry City, Hull City** — promoted for 2026-27, Championship in 2024-25
- **Leeds United, Sunderland** — promoted for 2025-26, Championship in 2024-25

Note **Ipswich Town DOES have data** (32 players) — they were in the PL in
2024-25 and relegated, then promoted again for 2026-27.

### Stats this source does NOT have
No tackles, interceptions, pressures, or pass completion %. Those columns exist
in `schema.sql` and are written as NULL. `ROLE_ARCHETYPES` is scoped to avoid
them entirely, and `build_features.py`'s `COUNTING_STATS` excludes them (an
all-NULL stat becomes an all-zero feature that dilutes every similarity score).

## Files already built
- `src/config.py` — `CURRENT_PL_CLUBS`, `CLUB_NAME_MAP`, `CLUBS_MISSING_FROM_DATA`,
  `PLAYER_STATS_CSV`, `DATA_SEASON`, `FEATURE_GLOSSARY` / `FEATURE_COLUMNS`
  (the style vocabulary, fed to the agent), and `STYLE_PRESETS` — six worked
  examples, NOT a fixed menu (`ROLE_ARCHETYPES` is a backwards-compatible alias).
- `src/schema.sql` — SQLite schema: players, player_season_stats (incl.
  `prog_passes_received` for PrgR), player_features (incl. `total_minutes`,
  `feature_json` z-scores, `raw_per90_json`, `percentiles_json`).
- `src/ingest.py` — loads the CSV directly (no scraping), normalizes club
  names, filters to `CURRENT_PL_CLUBS`, loads into SQLite. **Verified against
  the real CSV.**
- `src/build_features.py` — recency-weighted per-90 stats, z-score
  normalization. **Verified against the real data.**
- `src/agent_tools.py` — three tools (`query_players`, `find_players`,
  `get_player_report`) + Claude tool-use loop with challenge/re-justify support.
  See "Open-vocabulary style matching" below — this is the heart of the project.
- `src/app.py` — Streamlit chat UI. **Not yet run live** (needs API key).
- `src/classify_kaggle_csvs.py`, `src/inspect_kaggle.py`,
  `src/inspect_player_stats.py` — reusable utilities for evaluating another
  dataset later.

## Open-vocabulary style matching — the core design
**There is no fixed list of roles.** A user asks for anything in plain English;
the agent translates it into a WEIGHT VECTOR over the seven features that
exist, and `find_players` ranks against that vector. `STYLE_PRESETS` are worked
examples in the tool description (few-shot material), plus shortcuts — they are
not the vocabulary.

The seven feature keys. Anything else silently scores 0.0, so `find_players`
rejects unknown keys with an error that lists the valid ones and the glossary,
letting the agent self-correct:
```
goals_p90, assists_p90, npxg_p90, xa_p90,
prog_passes_p90, prog_carries_p90, prog_passes_received_p90
```
Features are z-scored, so a negative weight means "this style is defined partly
by NOT doing this" (a deep playmaker plays progressive passes rather than
receiving them). `config.FEATURE_GLOSSARY` describes each one tactically and is
injected into both the tool description and the system prompt — one source of
truth, no drift.

### Scoring: weighted mean z-score, not cosine
`fit_score = Σ(w·z) / Σ|w|`, then multiplied by a sample-size shrinkage factor.

Cosine similarity was the original choice and is **wrong for "find me the best
X"** — it matches the SHAPE of a profile and ignores magnitude, so it ranked
Darwin Núñez above Erling Haaland on goal threat. The weighted mean z-score
rewards being far in the direction asked for, and is directly interpretable:
+1.5 means "on average 1.5 SD above a PL regular in the traits requested".

Cosine is still used, correctly, for `similar_to_player_id` — player-to-player
resemblance genuinely is a question about profile shape, not magnitude.

### Sample-size shrinkage
`reliability = minutes / (minutes + 900)`, multiplied into `fit_score`.

Without it every query surfaced cameo players — a striker with 638 minutes beat
Haaland and Salah on "most dangerous goalscorer", because a hot streak over
seven matches produces a wilder per-90 than a full season does. Results carry
`raw_fit_score` and `sample_reliability` so the agent can be explicit about it.
Shrinkage scales a player's whole z-vector uniformly, so it correctly cancels
out of the cosine path and is applied only to `fit_score`.

### Communicating results to non-analysts
User feedback: "5.204 prog carries/90 | 11.48 prog passes received/90" is
meaningless to the fans/scouts/recruiters who are the actual audience — they
have no idea whether a number is good. Three changes:

1. **Positional percentiles.** `build_features.py` computes each stat's
   percentile against players with the same primary position (stored in
   `player_features.percentiles_json`). Ranking still uses league-wide
   z-scores — a trait's value to a team is absolute — but explanation uses
   positional percentiles, which is the reference class a scout thinks in.
2. **Plain-English labels.** `config.FEATURE_LABELS` maps `prog_carries_p90` →
   "carrying the ball forward". `config.PERCENTILE_BANDS` / `percentile_band()`
   give one consistent vocabulary (elite / excellent / strong / average /
   below average / poor) instead of a fresh adjective per player.
3. **Jargon is banned from output.** The system prompt forbids mentioning
   standard deviations, z-scores, fit_score, sample_reliability, style_weights,
   or shrinkage — not the numbers, not the names, not explanations of the
   mechanism. Small samples are described as minutes, never as machinery.

Tool results now ship a `stats` list per player, ordered by how much each stat
mattered to the query: `{stat, feature, per_90, percentile, rating}`, plus a
`compared_against` string naming the reference group. Raw per-90s are rounded
to 2dp — `5.204` was false precision that made the output look more scientific
than the data supports.

### Showing the interpretation in the UI
`agent_tools.describe_interpretation()` turns a `find_players` call back into
readable priorities, and `app.py` renders them in a "How your request was
interpreted" expander under each answer. Without this the translation — the
project's whole thesis — was computed and discarded, leaving a black box that
merely worded itself nicely.

### Position filtering
`find_players(position=...)` accepts GK/DF/MF/FW only — the dataset has no
finer detail. Finer positional meaning (left-back, number 8, false nine) is
expressed through the weights, not the filter. Passing `position="LB"` returns
an error saying exactly that.

## Pipeline```
python src/ingest.py          -> Filtered 574 -> 447 rows, 16 clubs
python src/build_features.py  -> 315 feature vectors (450-minute floor)
```
447 of 574 players are in scope; 315 clear the 450-minute sample floor.
Excluded as out of scope: Leicester City, Southampton, West Ham, Wolves.

Raw-data sanity check passed — top scorers match reality for 2024-25:
Salah 29, Isak 23, Haaland 22, Mbeumo 20, Wood 20. (This also confirms the
season-total vs per-90 column trap above was avoided.)

Free-form style queries sanity-checked against the tool layer (hand-written
weight vectors standing in for what the model emits):
- "left-back who inverts into midfield" (DF + progression, negative npxG)
  → Alexander-Arnold, Gvardiol, Trippier, van Hecke, Akanji
- "most dangerous pure goalscorer" (FW) → Salah, Haaland, Isak, Wissa, Watkins
- "young winger who beats his man" (FW, u23) → Doku, Sávio, Saka, Madueke, Garnacho
- "creator behind the striker, 2000+ mins" (MF) → Bruno Fernandes, Damsgaard,
  Ødegaard, Palmer, Szoboszlai
- "someone like Bukayo Saka" (similarity) → Son, Trossard, Murphy, Rogers

Guardrails verified: unknown feature names, `position="LB"`, two scoring modes
at once, and an unknown player_id all return actionable errors rather than
silently scoring zero.

## Agent behaviourRun against the real Anthropic API with `claude-sonnet-4-6`. The NL→weights
translation works; this was the last untested piece.

- **"a striker who drops deep and creates rather than poaching, under 26"** →
  the model composed, unprompted:
  `{xa_p90: 1.5, assists_p90: 1.0, prog_passes_p90: 1.2, prog_carries_p90: 0.8,
    prog_passes_received_p90: -0.8, npxg_p90: -0.5}` with `position="FW"`,
  `max_age=25`. The negative weights are the right encoding of "drops deep
  rather than poaching" — this is the core mechanic working as designed.
- **"a left-back who can invert into midfield"** → the model chose the
  `inverted_fullback` preset rather than composing weights. Correct behaviour
  (it matched), but note the preset path can mask the compose path in testing.
- **"a ball-winning midfielder who breaks up play"** → correctly REFUSED,
  listed the seven available stats, explained that ranking on them would
  surface a creative midfielder rather than a ball-winner, and pointed at
  FBref/StatsBomb/Opta/Wyscout. No attacking proxy was substituted.
- **Challenge loop** ("why not Gvardiol instead?") → re-called
  `get_player_report`, argued from the numbers, and genuinely conceded that
  Gvardiol is the better pick if the brief emphasises carrying over passing.

### Output constraints and why they exist
1. **Footedness ban had a loophole.** After being told not to state a player's
   foot or side, the model reframed the claim as an aside: "Trent ranked top
   overall but is more of a specialist right-sided player — flag him to
   yourself accordingly." The ban now explicitly covers asides, caveats,
   footnotes, honourable mentions, and advice to the reader, and says the side
   limitation must be stated about the DATASET, never a named player.
2. **Internal column names leaked to the user.** The refusal path printed a
   markdown table of `goals_p90`, `npxg_p90`, `prog_passes_received_p90` — the
   exact jargon the plain-English work removed everywhere else. The prompt now
   bars printing column names anywhere, including when listing what the data
   does and does not cover.

### Prohibited claims
The first live run asserted **"Trent is almost certainly leaving Liverpool and
going to Real Madrid"** and **"Man City's price tag will be significant"** —
model world-knowledge presented as fact inside a data-backed scouting report.
That is precisely the failure this project exists to avoid.

Fixed by adding explicit rules to `SYSTEM_PROMPT`: no transfers, market values,
contracts, availability, injuries, or price tags, and no football knowledge from
memory used as evidence. Also fixed age presentation — the data is 2024-25, so
ages must be written "22 (2024-25)" rather than as current ages. Re-verified:
both behaviours are now clean, and the model volunteers the availability caveat
on its own.

**Footedness and side.** The model was stating
preferred foot and natural side ("naturally left-sided", "right-footed and
right-sided by nature"). Correct often enough to be tempting — it once produced
a genuinely useful note that Trippier doesn't fit a left-back brief — but it is
recalled trivia dressed as analysis, and "right this time" is not a property
you can rely on. Same class of error as the transfer speculation.

Now barred outright in `SYSTEM_PROMPT`, including hedged forms ("likely
left-footed"). Instead the agent must surface the real limitation: when a
request depends on side, it states once that the data cannot distinguish left
from right and that the user should rule out anyone on the wrong side.
The left-back query now opens with exactly that caveat and
contains no footedness claims.

**Also prohibited:** the model quoted
"fit score" to the user AND misexplained it (blamed "a single season's sample"
when shrinkage is minutes-based), and inferred "an injury-disrupted season"
from low minutes. Both now explicitly barred in `SYSTEM_PROMPT` — low minutes
can mean rotation, suspension, or a mid-season signing.

## Known limitations to state honestly on the resume/demo
1. **No defensive data.** Style-matching covers attacking output, chance
   creation, and ball progression only — no tackles/interceptions/pressures/
   duels/pass completion. The system prompt instructs the agent to say so
   plainly rather than substitute an attacking proxy for defending. Demo
   scenarios should still lean into what the data supports.
2. **Position granularity is GK/DF/MF/FW only.** "Left-back" is expressed
   through the weights, not a filter, so a left-back query will also surface
   ball-playing centre-backs (Gvardiol, van Hecke, Akanji rank alongside
   Alexander-Arnold and Trippier). Defensible — they genuinely do the thing
   asked for — but worth naming rather than pretending it's precise.
3. **The NL→weights translation is the model's judgement, not a validated
   mapping.** `interpreted_as` is echoed back in every result so the user can
   see what their words became, which makes it auditable but not guaranteed
   correct. Good demo material: show the interpretation, invite disagreement.
4. **Single season (2024-25).** No form trend, no age curve, no injury data.

## Immediate next steps
1. **Get `src/app.py` running** (`streamlit run src/app.py`, with
   `ANTHROPIC_API_KEY` exported). The agent loop is verified; the Streamlit
   wrapper around it is the only untested layer left.
2. Decide the footedness/outside-the-data question above.
3. Consider the model ID in `agent_tools.py`: currently `claude-sonnet-4-6`,
   which is valid and performed well in live testing, but `claude-opus-5` is
   the stronger default for multi-step tool-use reasoning.
   `run_agent_turn(..., model=...)` takes an override.
4. Demo script worth rehearsing, since all four are verified working: the
   left-back ask → the challenge → the deep-lying creator (shows open
   vocabulary) → the ball-winner ask (shows honest refusal). That last one is
   the most impressive part of the demo and the least obvious.
4. Market values / transfer data are OUT OF SCOPE — decided 2026-08-25. The
   `player_market_value` table was removed from `schema.sql` rather than left
   as an empty promise. Do not reintroduce it.
5. If time allows: find a second single-season CSV in the same
   `siddhrajthakor` format for an earlier season. Union it before
   `build_features.py` — no schema changes needed if column names match, and
   `ingest.py --csv <path> --season <season>` already supports loading one.

## Test suite
`python -m pytest tests/ -q` — 79 tests, ~0.5s, fully deterministic. No network,
no API calls, no cost, so there is no excuse not to run it on every change.

- `tests/test_scoring.py` — fit score, shrinkage, cosine, percentile bands
- `tests/test_tool_contract.py` — `find_players` error paths and result shape
- `tests/test_config_integrity.py` — config/prompt/vocabulary consistency
- `tests/test_data_pipeline.py` — the silent CSV traps and DB invariants

Most tests are regressions for bugs that actually shipped in this project and
produced no exception. The ones worth not deleting:
- presets referencing features that do not exist (scored 0.0 silently)
- `COLUMN_MAP` reading the per-90 duplicate columns instead of season totals
- `CLUB_NAME_MAP` missing an FBref short name (drops a club, dataset still
  looks fine)
- the prompt losing an honesty rule — one parametrised test asserts each ban
  is still present, searched with whitespace collapsed so rewrapping a
  paragraph does not fail it

NOT covered: whether the shortlist is any GOOD. That needs scouts — Layer 3,
still being designed. Do not let a green suite imply the scouting is validated.

## Environment notes
- macOS, Python 3.14, venv at `venv/` in the project root (created because of a
  Homebrew pip conflict — always `source venv/bin/activate` before running)
- `ANTHROPIC_API_KEY` must be exported in the shell before running the agent
- `KAGGLE_API_TOKEN` was set via `export` (newer Kaggle CLI token flow, not the
  older `kaggle.json` file method) to download datasets
- The `diagnose*.py` scripts used to debug the FBref Cloudflare block have been
  deleted — that investigation is finished, don't recreate them
- `data/epl_scout.db` is regenerable from the CSV at any time. If `schema.sql`
  changes, DELETE the DB first — `CREATE TABLE IF NOT EXISTS` will not add a
  column to an existing table.
