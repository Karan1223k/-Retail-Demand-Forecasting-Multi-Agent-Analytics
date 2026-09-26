"""
Streamlit UI
------------------
Chat interface for the multi-agent retail analyst. Shows each agent's
step live as the question moves through the graph (Planner -> Query/
Retrieve/Forecast -> Insight -> Critic), then displays the final
validated answer.

Rate-limit failures are caught and shown as a clean warning instead of
crashing the page.

Run with: streamlit run app.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import streamlit as st
from src.graph import build_graph

st.set_page_config(page_title="Retail Multi-Agent Analyst", page_icon="📊", layout="centered")

st.title("📊 Retail Multi-Agent Analyst")
st.caption(
    "Ask a question about the Rossmann sales dataset. A Planner, Query, "
    "Forecast, Insight, and Critic agent collaborate to answer it — "
    "grounded in real SQL results, the project's own documented "
    "findings, and pre-trained forecast models."
)


@st.cache_resource
def get_app():
    return build_graph()


app = get_app()

if "history" not in st.session_state:
    st.session_state.history = []

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = 0

# One question can use up to 5 Gemini calls (planner/query/forecast/
# insight/critic). Free tier rate limits apply, so we space questions out.
MIN_SECONDS_BETWEEN_QUESTIONS = 65

with st.sidebar:
    st.subheader("Try asking:")
    st.markdown(
        "- What was the average sales amount for store 1 in January 2013?\n"
        "- Is a MAPE of 0.078 a good result for this project?\n"
        "- What bug was found in the rolling average feature?\n"
        "- What will store 1's sales be on 2015-08-20?\n"
        "- What will store 2's sales be in December 2026?"
    )

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander("See agent steps"):
            st.markdown(f"**Plan:** `{entry['plan']}`")
            if entry["sql"]:
                st.code(entry["sql"], language="sql")
            if entry["sql_result"]:
                st.text(entry["sql_result"])
            if entry["context"]:
                st.markdown("**Retrieved context:**")
                st.text(entry["context"][:500])
            if entry.get("forecast_result"):
                st.markdown("**Forecast result:**")
                st.text(entry["forecast_result"])
            st.markdown(
                f"**Critic verdict:** `{entry['verdict']}` "
                f"({entry['revision_count'] - 1} revision(s))"
            )

question = st.chat_input("Ask a question about the retail data...")

if question:
    elapsed = time.time() - st.session_state.last_run_time
    if elapsed < MIN_SECONDS_BETWEEN_QUESTIONS:
        wait = int(MIN_SECONDS_BETWEEN_QUESTIONS - elapsed)
        st.warning(
            f"Please wait {wait}s before asking another question "
            f"(free-tier API rate limit)."
        )
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            status = st.status("Working through the agent pipeline...", expanded=True)

            initial_state = {
                "question": question,
                "plan": "",
                "sql": "",
                "sql_result": "",
                "context": "",
                "forecast_result": "",
                "answer": "",
                "verdict": "",
                "issue": "",
                "revision_count": 0,
            }

            try:
                status.write("🧭 Planner deciding what's needed...")
                final_state = app.invoke(initial_state)

                status.write(f"📋 Plan: `{final_state['plan']}`")
                if final_state["sql"]:
                    status.write("🔍 Query Agent ran SQL against the database")
                if final_state["context"]:
                    status.write("📚 Retrieved relevant context from knowledge base")
                if final_state.get("forecast_result"):
                    status.write("🔮 Forecast Agent generated a prediction")
                status.write("✍️ Insight Agent drafted an answer")
                status.write(f"✅ Critic Agent verdict: {final_state['verdict']}")

                status.update(label="Done", state="complete", expanded=False)

                st.write(final_state["answer"])

                with st.expander("See agent steps"):
                    st.markdown(f"**Plan:** `{final_state['plan']}`")
                    if final_state["sql"]:
                        st.code(final_state["sql"], language="sql")
                    if final_state["sql_result"]:
                        st.text(final_state["sql_result"])
                    if final_state["context"]:
                        st.markdown("**Retrieved context:**")
                        st.text(final_state["context"][:500])
                    if final_state.get("forecast_result"):
                        st.markdown("**Forecast result:**")
                        st.text(final_state["forecast_result"])
                    st.markdown(
                        f"**Critic verdict:** `{final_state['verdict']}` "
                        f"({final_state['revision_count'] - 1} revision(s))"
                    )

                st.session_state.history.append(final_state)
                st.session_state.last_run_time = time.time()

            except RuntimeError as e:
                status.update(label="Rate limit hit", state="error", expanded=False)
                st.error(
                    f"⏳ {str(e)}\n\n"
                    f"This is a known limitation of the free tier. "
                    f"Please wait and try again."
                )
                st.session_state.last_run_time = time.time()