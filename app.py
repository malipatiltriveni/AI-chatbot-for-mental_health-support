"""
app.py

Main Streamlit app tying together:
- Risk classification (every message checked before anything else)
- Crisis path (fixed responses, no LLM)
- CBT exercise suggestions (retrieved, not generated)
- Journaling with sentiment trend
- Gated supportive chat (LLM, only for normal/distress messages)

Run with: streamlit run app.py
"""

import json
import os
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from crisis_responses import get_crisis_response, log_crisis_event
from risk_classifier import classify_risk, load_model, MODEL_PATH
from sentiment_tracker import score_entry

DB_PATH = "data/app.db"
HAS_API_KEY = bool(os.environ.get("GEMINI_API_KEY"))

st.set_page_config(page_title="Mind Companion (Student Project)", page_icon="🌱")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mood_score INTEGER,
            entry TEXT,
            sentiment REAL
        )
    """)
    conn.commit()
    conn.close()


@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        st.error(
            f"No trained model found at {MODEL_PATH}. "
            "Run `python risk_classifier.py` first to train it."
        )
        st.stop()


@st.cache_data
def load_cbt_library():
    with open("cbt_library.json") as f:
        return json.load(f)


init_db()
model = get_model()
cbt_library = load_cbt_library()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.warning(
    "⚠️ **This is a student project, not a medical device.**\n\n"
    "It is not a replacement for therapy or professional care. "
    "If you're in crisis, contact 988 (US) or your local emergency services."
)

if not HAS_API_KEY:
    st.sidebar.info("No ANTHROPIC_API_KEY set — live chat replies are disabled, "
                     "but everything else works.")

page = st.sidebar.radio("Navigate", ["Chat", "Mood Check-In & Journal", "Trends"])


# ---------- Chat page ----------

if page == "Chat":
    st.title("🌱 Mind Companion")
    st.caption("A supportive listening space. Not therapy. Not a crisis service.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("What's on your mind?")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        risk = classify_risk(user_input, model=model)

        with st.chat_message("assistant"):
            if risk == "crisis":
                log_crisis_event()
                reply = get_crisis_response()
                st.write(reply)
            elif not HAS_API_KEY:
                reply = (
                    "(Live chat is disabled — no ANTHROPIC_API_KEY set. "
                    f"Your message was classified as **{risk}**. "
                    "Set the API key to enable real supportive replies.)"
                )
                st.write(reply)
            else:
                from llm_chat import get_supportive_response
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                try:
                    reply = get_supportive_response(user_input, history)
                except Exception as e:
                    reply = (
                        "I'm having trouble responding right now "
                        f"(technical error: {e}). If you need to talk to "
                        "someone, 988 (US) is available anytime."
                    )
                st.write(reply)

                if risk == "distress":
                    matching = [c for c in cbt_library if any(
                        tag in user_input.lower() for tag in c["tags"]
                    )]
                    if matching:
                        ex = matching[0]
                        st.info(f"**Try this: {ex['title']}**\n\n{ex['exercise']}")

        st.session_state.messages.append({"role": "assistant", "content": reply})


# ---------- Mood check-in & journal ----------

elif page == "Mood Check-In & Journal":
    st.title("📓 Mood Check-In & Journal")

    mood = st.slider("How are you feeling right now? (1 = very low, 10 = great)", 1, 10, 5)
    entry = st.text_area("Write about your day (optional)")

    if st.button("Save entry"):
        risk = classify_risk(entry, model=model) if entry.strip() else "normal"

        if risk == "crisis":
            log_crisis_event()
            st.error(get_crisis_response())
        else:
            sentiment = score_entry(entry) if entry.strip() else 0.0
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT INTO journal (timestamp, mood_score, entry, sentiment) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), mood, entry, sentiment),
            )
            conn.commit()
            conn.close()
            st.success("Entry saved.")

            if mood <= 3 or risk == "distress":
                matching = [c for c in cbt_library if any(
                    tag in entry.lower() for tag in c["tags"]
                )] or cbt_library[:1]
                ex = matching[0]
                st.info(f"**Try this: {ex['title']}**\n\n{ex['exercise']}")


# ---------- Trends ----------

elif page == "Trends":
    st.title("📈 Your Trends")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM journal ORDER BY timestamp", conn)
    conn.close()

    if df.empty:
        st.info("No journal entries yet. Add some on the Mood Check-In page.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig1 = px.line(df, x="timestamp", y="mood_score", title="Mood over time (1-10)")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(df, x="timestamp", y="sentiment", title="Journal sentiment over time (-1 to 1)")
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df[["timestamp", "mood_score", "sentiment", "entry"]])