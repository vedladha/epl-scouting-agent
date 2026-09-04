# EPL Autonomous Scouting Agent

Describe the player you need in plain English. The agent works out which stats
express that idea, ranks every player from the 2024-25 Premier League against
it, and justifies the shortlist with percentiles — then defends its picks when
you push back.

## How it works

1. **`ingest.py`** loads a season of FBref Premier League player stats from CSV
   into SQLite, filtered to the 20 clubs of that season.
2. **`build_features.py`** turns season totals into per-90 rates, z-scores them
   against the league, and ranks each stat against same-position players. It
   stores the z-scores (for ranking) and the percentiles (for explaining a
   pick); goalkeepers are skipped entirely.
3. **`agent_tools.py`** exposes three tools to Claude — `query_players`,
   `find_players`, `get_player_report` — and runs the tool-use loop.
4. **`app.py`** is a Streamlit chat UI over that loop.

The agent gets a glossary describing what each stat *means on a pitch*, not
what it measures statistically, and composes a weight vector from it. Unknown
feature names come back as an error listing the valid ones, so the model
self-corrects rather than silently scoring zero.

### Scoring

`fit_score = Σ(w·z) / Σ|w|`, shrunk by sample size.

Cosine similarity is the obvious choice and is **wrong for "find me the best
X"** — it matches the *shape* of a profile and ignores magnitude, which put
Darwin Núñez above Erling Haaland on goal threat. A weighted mean z-score
rewards being far in the direction asked for, and reads directly: `+1.5` means
one and a half standard deviations above a Premier League regular in the traits
requested. Cosine is still used for "find me another Bukayo Saka", where
profile shape genuinely is the question.

Shrinkage is `minutes / (minutes + 900)`. Without it, cameo players win every
query — a striker with 638 minutes outscored Haaland and Salah, because a hot
streak over seven matches makes a wilder per-90 than a full season does.

## Making the numbers mean something

A raw per-90 rate is useless to anyone who isn't already an analyst. "5.2
progressive carries per 90" is unremarkable for a winger and extraordinary for
a centre-back, and nobody outside the field knows which. So every stat reaches
the user as a plain-English name, a percentile against **same-position** PL
players, and a word for that percentile:

> **Elite at carrying the ball forward** — 100th percentile among Premier
> League forwards

Raw per-90 rates, standard deviations, z-scores and the fit score itself never
leave the tool layer — they are not shown, and not sent to the agent either, so
there is nothing for it to quote. Percentiles do the same job in language
everyone already has.

Ranking uses league-wide z-scores, because a trait's value to a team is
absolute. Explanation uses positional percentiles, because that's the reference
class a scout actually thinks in. Gvardiol reads as "99th percentile among
defenders for getting into shooting positions", off a raw rate that would be
unremarkable for a forward.

## Data

**Source:** [`siddhrajthakor/fbref-premier-league-202425-player-stats-dataset`](https://www.kaggle.com/datasets/siddhrajthakor/fbref-premier-league-202425-player-stats-dataset)
(Kaggle) → `data/kaggle_raw/fbref_PL_2024-25.csv`

Scraping FBref directly is blocked — it sits behind Cloudflare and 403s every
attempt — so this uses a static Kaggle export instead.

**Pipeline output:** 574 players across all 20 clubs → 367 feature vectors
(the rest are goalkeepers, who are excluded entirely, or fall below a
450-minute sample floor).

The seven available features, and the entire style vocabulary:

```
goals_p90  assists_p90  npxg_p90  xa_p90
prog_passes_p90  prog_carries_p90  prog_passes_received_p90
```

### Known limitations

- **One season (2024-25)**, two behind the 2026-27 squad scope. No form trend,
  no age curve, no injury data. Ages are reported as 2024-25 ages and labelled
  as such.
- **No defensive data** — see above.
- **Position granularity is DF/MF/FW only** (goalkeepers are not ranked at
  all — none of the seven stats mean anything for them). "Left-back" is
  expressed through the weights, so a left-back query also surfaces
  ball-playing centre-backs. Defensible, but not precise.
- **A player's club is where he played in 2024-25**, not where he is now.
  There is no transfer data, so the scope is "everyone who played in the
  2024-25 Premier League" rather than any present-day squad.
- **The NL→weights translation is the model's judgement**, not a validated
  mapping. Every result echoes `interpreted_as` so it's auditable.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download the Kaggle CSV to data/kaggle_raw/fbref_PL_2024-25.csv, then:
python src/ingest.py
python src/build_features.py

export ANTHROPIC_API_KEY=sk-ant-...
streamlit run src/app.py
```

`data/epl_scout.db` is regenerable from the CSV at any time. If `schema.sql`
changes, DELETE the DB before rebuilding — `CREATE TABLE IF NOT EXISTS` will
not add a column to an existing table.

