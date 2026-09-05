"""
Day 3: Minimal Streamlit chat UI for the EPL Scouting Agent.

Run: streamlit run src/app.py
Requires ANTHROPIC_API_KEY set in your environment (or a .env file).
"""
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from agent_tools import describe_interpretation, run_agent_turn
from config import DB_PATH, PLAYER_STATS_CSV

SRC = Path(__file__).parent
ROOT = SRC.parent


def _secret(name: str) -> str | None:
    """Streamlit Cloud puts config in st.secrets; locally it's the environment."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass                      # no secrets.toml at all — fine locally
    return os.environ.get(name)


@st.cache_resource(show_spinner="Building the player database...")
def ensure_database() -> bool:
    """
    Build the SQLite database from the committed CSV if it isn't there.

    Streamlit Cloud has no build step, so a fresh container would otherwise
    start with no data. Cached, so this runs once per container, not per user.
    """
    if (ROOT / DB_PATH).exists():
        return True
    if not (ROOT / PLAYER_STATS_CSV).exists():
        return False
    for script in ("ingest.py", "build_features.py"):
        subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT, check=True)
    return True


def require_password():
    """
    Shared passphrase for the deployed demo.

    This is a spend guard, not authentication — the app has no user accounts
    and every query bills one API key. It exists so a public URL doesn't hand
    strangers your credit. Skipped entirely when APP_PASSWORD is unset.
    """
    expected = _secret("APP_PASSWORD")
    if not expected or st.session_state.get("authed"):
        return
    st.title("⚽ EPL Autonomous Scouting Agent")
    st.caption("This demo runs on a metered API key, so it's passphrase-protected.")
    entered = st.text_input("Passphrase", type="password")
    if entered and entered == expected:
        st.session_state.authed = True
        st.rerun()
    elif entered:
        st.error("Not quite — check the passphrase you were sent.")
    st.stop()

try:
    import anthropic
except ImportError:
    st.error("Run: pip install anthropic --break-system-packages")
    st.stop()

st.set_page_config(page_title="EPL Scouting Agent", page_icon="⚽")

# Streamlit shows a "Running..." status widget in the top-right corner on every
# rerun. A query takes ~20s, so that animation sits there spinning for the whole
# wait and reads as the app being stuck rather than working.
st.markdown(
    """
    <style>
        [data-testid="stStatusWidget"] { display: none !important; }
        [data-testid="stToolbarActions"] { display: none !important; }
        [data-testid="stMainMenu"] { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }

    </style>
    """,
    unsafe_allow_html=True,
)

require_password()

if not ensure_database():
    st.error(f"No player data found. Expected {PLAYER_STATS_CSV} in the repo.")
    st.stop()

st.title("⚽ EPL Autonomous Scouting Agent")
st.caption("Scouting that argues its case.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # UI-facing history (text only)
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []  # full Claude message history incl. tool calls

def get_api_key() -> str | None:
    """
    Prefer a configured key; otherwise let the viewer supply their own.

    Bring-your-own-key is what makes a public demo genuinely free to run: the
    person testing pays for their own queries, and the deployer risks nothing.
    The key lives in this browser session only — never written to disk, never
    logged, gone when the tab closes.
    """
    configured = _secret("ANTHROPIC_API_KEY")
    if configured:
        return configured
    if st.session_state.get("user_api_key"):
        return st.session_state.user_api_key

    st.info(
        "**This demo runs on your own Anthropic API key.**\n\n"
        "Paste one below to try it. It stays in your browser session — it is "
        "not stored, logged, or sent anywhere except Anthropic. A few queries "
        "cost a fraction of a cent.\n\n"
        "Get one at [console.anthropic.com](https://console.anthropic.com/settings/keys)."
    )
    entered = st.text_input("Anthropic API key", type="password",
                            placeholder="sk-ant-...")
    if entered:
        if not entered.startswith("sk-ant-"):
            st.error("That does not look like an Anthropic key — they start with `sk-ant-`.")
            st.stop()
        st.session_state.user_api_key = entered
        st.rerun()
    st.stop()


api_key = get_api_key()

client = anthropic.Anthropic(api_key=api_key)

def render_interpretation(interpretations):
    """Show how the plain-English request became something measurable."""
    if not interpretations:
        return
    with st.expander("🔍 How your request was interpreted"):
        for i, d in enumerate(interpretations):
            if i:
                st.divider()
            if d["mode"] == "similar":
                st.markdown(f"Matched against **{d['reference']}**'s overall profile.")
            else:
                if d.get("preset"):
                    st.caption(f"Matched a saved profile: `{d['preset']}`")
                st.markdown("Your words were turned into these priorities:")
                for w in d["weights"]:
                    arrow = "🟩" if w["direction"] == "more" else "🟥"
                    st.markdown(
                        f"{arrow} `{w['bar']:<6}` **{w['direction']}** {w['label']}")
            if d["filters"]:
                st.caption("Filters: " + " · ".join(d["filters"]))
        st.caption(
            "Players are then ranked on these priorities and scored against "
            "others in the same position. Disagree with the reading? Say so — "
            "the agent will re-run it."
        )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_interpretation(msg.get("interpretations"))

prompt = st.chat_input("Describe the player you need, in plain English")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.agent_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Reading the request, then querying the player database..."):
            start = len(st.session_state.agent_messages)
            try:
                st.session_state.agent_messages = run_agent_turn(
                    client, st.session_state.agent_messages)
            except anthropic.APIStatusError as e:
                # Without this, any API hiccup renders a raw traceback in the
                # chat window — which is what a live demo audience would see.
                st.error(f"The model API returned an error ({e.status_code}). "
                         f"Try again in a moment.")
                st.session_state.agent_messages = st.session_state.agent_messages[:start - 1]
                st.stop()
            except anthropic.APIConnectionError:
                st.error("Couldn't reach the model API. Check your connection.")
                st.session_state.agent_messages = st.session_state.agent_messages[:start - 1]
                st.stop()

            new_messages = st.session_state.agent_messages[start:]
            last = st.session_state.agent_messages[-1]
            text = "\n".join(b.text for b in last["content"]
                              if getattr(b, "type", None) == "text")
            interpretations = [
                d for msg in new_messages if not isinstance(msg["content"], str)
                for b in msg["content"]
                if getattr(b, "type", None) == "tool_use"
                for d in [describe_interpretation(b.name, b.input)] if d
            ]

        st.markdown(text)
        render_interpretation(interpretations)
        st.session_state.messages.append(
            {"role": "assistant", "content": text, "interpretations": interpretations})

with st.sidebar:
    st.link_button("GitHub", "https://github.com/vedladha/epl-scouting-agent")
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.agent_messages = []
        st.rerun()
