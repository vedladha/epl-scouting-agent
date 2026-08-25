# EPL Autonomous Scouting Agent

Describe the player you need in plain English. The agent works out which stats
express that idea, ranks every current Premier League player against it, and
justifies the shortlist with real numbers — then defends its picks when you
push back.

```
"a striker who drops deep and creates rather than just poaching, under 26"

  -> style_weights = {
       xa_p90:  1.5,  assists_p90: 1.0,  prog_passes_p90: 1.2,
       prog_carries_p90: 0.8,  prog_passes_received_p90: -0.8,
       npxg_p90: -0.5,
     },  position = "FW",  max_age = 25
```

That translation is the whole project. There is **no fixed menu of roles** — the
negative weights above are the model's own encoding of "drops deep rather than
poaching," composed at query time from the user's words.

## How it works

1. **`ingest.py`** loads a season of FBref Premier League player stats from CSV
   into SQLite, filtered to current PL squads.
2. **`build_features.py`** turns season totals into per-90 rates, z-scores them
   against the league, and stores both the z-scores (for ranking) and the raw
   per-90s (for citing real numbers).
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

> **Elite at carrying the ball forward** — 100th percentile among PL forwards
> (12.7 progressive carries per 90)

Standard deviations, z-scores, and the fit score itself are never shown. They
are internal ranking machinery, and percentiles do the same job in language
everyone already has.

Ranking uses league-wide z-scores, because a trait's value to a team is
absolute. Explanation uses positional percentiles, because that's the reference
class a scout actually thinks in. Gvardiol reads as "98th percentile among
defenders for getting into shooting positions" at 0.12 npxG/90 — a number that
would be nothing for a forward.

### Showing the translation

The UI surfaces how each request was read, so the ranking isn't a black box
that merely words itself nicely:

```
🔍 How your request was interpreted
   🟩 ▇▇▇    more  carrying the ball forward
   🟩 ▇▇▇    more  creating chances for others
   🟩 ▇▇     more  getting on the end of forward passes
   🟥 ▇      less  getting into shooting positions
   Filters: position forwards · age at most 22
```

If the reading is wrong, the user can say so and the agent re-runs it.

## What it deliberately refuses to do

The dataset has **no defensive stats** — no tackles, interceptions, pressures,
duels, or pass completion. Asked for a ball-winning midfielder, the agent says
so and names better data sources, rather than ranking on attacking stats and
implying the result measures defending:

> I have to be straightforward with you here: this dataset cannot measure
> ball-winning or defensive ability at all. […] If I ranked midfielders on
> these stats and called the result a "ball-winner," I'd just be finding the
> most creative or progressive midfielder — a completely different profile.

It is likewise barred from transfers, market values, contracts, availability,
injuries, and — because the data has no left/right information — a player's
preferred foot or which flank they play. Asked for a *left*-back, it says
plainly that it cannot distinguish side and that the user must rule out anyone
on the wrong one, rather than guessing from memory. An earlier version volunteered "Trent is almost certainly leaving
Liverpool for Real Madrid" inside a data-backed report — model world-knowledge
dressed as evidence, which is the exact failure this tool exists to avoid.

## Data

**Source:** [`siddhrajthakor/fbref-premier-league-202425-player-stats-dataset`](https://www.kaggle.com/datasets/siddhrajthakor/fbref-premier-league-202425-player-stats-dataset)
(Kaggle) → `data/kaggle_raw/fbref_PL_2024-25.csv`

Scraping FBref directly is blocked — it sits behind Cloudflare and 403s every
attempt — so this uses a static Kaggle export instead.

**Pipeline output:** 574 rows → 447 players across 16 clubs → 315 feature
vectors (450-minute floor).

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
- **Position granularity is GK/DF/MF/FW only.** "Left-back" is expressed
  through the weights, so a left-back query also surfaces ball-playing
  centre-backs. Defensible, but not precise.
- **Four in-scope clubs have zero rows** — Coventry, Hull, Leeds, and
  Sunderland were in the Championship in 2024-25.
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

[`docs/DECISIONS.md`](docs/DECISIONS.md) covers why the project is shaped this
way — two abandoned data sources, the scoring choices, and the traps in this
CSV that fail silently.

## Tech

Python · pandas · SQLite · Claude (tool use) · Streamlit

Style matching is per-90 feature vectors and explicit weights, not learned
embeddings — for a build this size, being able to show a coach exactly why a
player ranked where they did is worth more than sophistication.

## Deploying a shareable demo

Streamlit Community Cloud hosts this for free from a GitHub repo.

**Before you share the URL:** the app runs on one API key — yours — and every
query bills it. Community Cloud apps are publicly reachable, so treat the link
as a spending surface, not just a demo.

1. Set a spend limit in the [Anthropic Console](https://console.anthropic.com)
   (Billing → Limits). This is the only hard stop; everything else is a
   deterrent.
2. Set `APP_PASSWORD` in the app's secrets. The app then asks for a passphrase
   before it will talk to the API. It is a spend guard, not authentication —
   there are no user accounts — but it keeps a stray link from costing money.
3. Use a **separate** API key for the deployment, so you can revoke it without
   touching your local setup.

```bash
git init && git add . && git commit -m "EPL scouting agent"
gh repo create epl-scout-agent --private --source=. --push
```

Then at [share.streamlit.io](https://share.streamlit.io): New app → pick the
repo → main file `src/app.py` → **Advanced settings → Python 3.12** (the code
needs 3.10+ for `X | None` annotations) → paste your secrets:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
APP_PASSWORD = "something-you-send-your-tester"
```

`data/kaggle_raw/fbref_PL_2024-25.csv` (88KB) must be committed. The SQLite
database is rebuilt from it automatically on first load, so it does not need
to be — a fresh container takes a few seconds longer on its first request.

## Tests

```bash
python -m pytest tests/ -q      # 79 tests, ~0.5s, no network, no API cost
```

Deterministic only — the suite never calls the model, so it runs on every
change. Four areas:

- **Scoring** — the weighted-mean-z ranking, sample-size shrinkage, and cosine
  similarity, including the two properties the design turns on: that scoring is
  scale-sensitive where cosine is not, and that a cameo cannot outrank a full
  season on a similar raw score.
- **Tool contract** — every error path `find_players` offers the agent
  (unknown feature, unknown preset, `position="LB"`, two scoring modes at
  once), plus the result shape: plain-English labels, percentiles in range,
  stats ordered by how much they mattered, filters actually filtering.
- **Config integrity** — that presets only reference features that exist, every
  feature has both a glossary entry and a plain-English label, the tool
  description teaches the whole vocabulary, and the prompt still carries each
  honesty rule.
- **Data pipeline** — the two CSV traps that fail silently (per-90 duplicate
  columns, FBref short club names), and the built database's invariants.

Most of these are regressions: each corresponds to a real bug that shipped,
raised no exception, and quietly degraded the output.

Evaluating the *quality* of the scouting itself is a separate problem that
needs domain experts — see the validation plan below.

## Validation plan

Testing with soccer fans, recruiters, and coaches via a Listen Labs contact:

- Give 3-4 real tactical asks with an expected answer, and measure whether the
  top-5 shortlist matches expert judgement
- Have testers rate whether the justification would be useful in a real
  scouting conversation
- Check that the challenge loop re-justifies from data rather than restating

**Status:** built and verified end to end; external testing not yet run.
