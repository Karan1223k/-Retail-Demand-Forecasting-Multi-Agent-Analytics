"""
Agent Definitions (Gemini version, new google-genai SDK)
--------------------------------------------------------------------
Five agents:
  PLANNER   - decides what's needed: sql, context, forecast, or both.
  QUERY     - writes and executes SQL against model_input.
  FORECAST  - predicts future sales via a pre-trained Prophet model,
              only for stores in TRAINED_FORECAST_STORES. If the user
              didn't specify a store, this is now explicitly disclosed
              in the result (not silently assumed) -- consistent with
              how the date-range and store-availability constraints are
              already surfaced to the user.
  INSIGHT   - combines results into a plain English answer, stating
              any date-range, store-availability, or default-assumption
              caveats clearly. On a revision, it is told exactly what
              the Critic flagged and must directly address it.
  CRITIC    - checks the Insight Agent's answer against the raw data,
              context, and forecast output.

_call_gemini logs diagnostics when Gemini returns an empty response.
"""

import os
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.tools import run_sql, retrieve_context, forecast_sales
from src.config import TRAINED_FORECAST_STORES, DATASET_START_DATE, DATASET_END_DATE

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-3.1-flash-lite"


def _call_gemini(system_prompt: str, user_message: str) -> str:
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        text = response.text
        if not text or not text.strip():
            try:
                finish_reason = response.candidates[0].finish_reason
            except Exception:
                finish_reason = "unknown"
            print(f"  [WARNING] Empty response from Gemini. finish_reason={finish_reason}")
            print(f"  [WARNING] Full response object: {response}")
            return "[Error: model returned an empty response]"
        return text.strip()
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise RuntimeError(
                "Rate limit hit. Please wait a bit and try again, or "
                "check aistudio.google.com/rate-limit for current usage."
            )
        raise


# ---------------------------------------------------------------------
# PLANNER AGENT
# ---------------------------------------------------------------------
PLANNER_SYSTEM = f"""You are a planning agent for a retail sales analytics
system. Given a user's question about a Rossmann retail sales dataset,
decide what's needed to answer it:

- "sql" if the question needs actual historical numbers from the
  database for a date on or before {DATASET_END_DATE}.
- "context" if the question needs domain knowledge / interpretation
  (e.g. "is this a good result?", "what is MAPE?", "what bug was fixed?")
- "forecast" if the question asks to PREDICT sales for a future date
  (after {DATASET_END_DATE}) for a specific store. Note: forecasting is
  only actually available for stores {TRAINED_FORECAST_STORES} -- but
  still choose "forecast" for any future-date question regardless of
  store, since the Forecast Agent will handle explaining if the store
  isn't trained.
- "both" if it needs real historical numbers AND interpretation.

Respond with ONLY one word: sql, context, forecast, or both. No
explanation."""


def planner_agent(question: str) -> str:
    plan = _call_gemini(PLANNER_SYSTEM, question).strip().lower()
    if plan not in ("sql", "context", "forecast", "both"):
        plan = "both"
    return plan


# ---------------------------------------------------------------------
# QUERY AGENT
# ---------------------------------------------------------------------
QUERY_SYSTEM = f"""You write SQL queries against a MySQL table called
model_input with these columns:
  store_id, store_type, assortment, sale_date, day_of_week,
  sales_amount, customers, promo, state_holiday, school_holiday,
  rolling_7day_avg

The dataset only covers {DATASET_START_DATE} to {DATASET_END_DATE}. If
asked about a period outside this range, still write the query as asked
-- the empty result will make that clear downstream. The dataset covers
all 919 stores for historical queries (this range limit does NOT apply
to which stores you can query -- only to forecasting, which is separate).

Given a user's question, write ONE valid MySQL SELECT query to answer it.
Respond with ONLY the raw SQL query, no explanation, no markdown
formatting, no semicolon."""


def query_agent(question: str) -> dict:
    sql = _call_gemini(QUERY_SYSTEM, question).strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    result = run_sql(sql)
    return {"sql": sql, "result": result}


# ---------------------------------------------------------------------
# FORECAST AGENT
# ---------------------------------------------------------------------
FORECAST_EXTRACT_SYSTEM = f"""Extract the store_id (integer) and target
date (YYYY-MM-DD) from the user's forecasting question. If the date is
described in relative or partial terms, resolve it to a specific date
as best you can.

Note: forecasting is only actually trained for stores
{TRAINED_FORECAST_STORES}, but extract whatever store_id the user asked
about regardless -- the system will explain if it's not a trained store.

If NO store_id is mentioned anywhere in the question, respond with
STORE: NONE (do not guess or default to any store yourself -- the
system will handle defaulting and will disclose that it did so).

Respond in EXACTLY this format, nothing else:
STORE: <store_id or NONE>
DATE: <YYYY-MM-DD>"""

DEFAULT_FORECAST_STORE = 1


def forecast_agent(question: str) -> dict:
    extraction = _call_gemini(FORECAST_EXTRACT_SYSTEM, question)

    store_match = re.search(r"STORE:\s*(\d+|NONE)", extraction)
    date_match = re.search(r"DATE:\s*(\d{4}-\d{2}-\d{2})", extraction)

    if not store_match or not date_match:
        return {"forecast_result": f"Could not parse store/date from question. Raw extraction: {extraction}"}

    store_defaulted = store_match.group(1) == "NONE"
    store_id = DEFAULT_FORECAST_STORE if store_defaulted else int(store_match.group(1))
    date_str = date_match.group(1)

    result = forecast_sales(store_id, date_str)

    if store_defaulted:
        # Explicitly disclose the assumption, same pattern as the
        # date-range and store-availability caveats -- never let the
        # system silently guess without telling the user.
        result = (
            f"NOTE: No specific store was mentioned in the question, so "
            f"this forecast defaults to store {DEFAULT_FORECAST_STORE}. "
            f"{result}"
        )

    return {"forecast_result": result, "store_id": store_id, "date": date_str, "store_defaulted": store_defaulted}


# ---------------------------------------------------------------------
# INSIGHT AGENT
# ---------------------------------------------------------------------
INSIGHT_SYSTEM = f"""You are an analyst explaining retail sales data to a
business stakeholder. You'll be given a question, and some combination
of: SQL query results, retrieved domain context, or a forecast result.
Write a clear, concise answer in plain English.

Rules:
- Only state facts that are directly supported by what you were given.
- If the SQL results say "Query returned no rows", clearly tell the user
  no data was found, and mention the dataset only covers
  {DATASET_START_DATE} to {DATASET_END_DATE} if relevant. Do not invent
  a number.
- If the forecast result says no trained model is available for a
  store, clearly state that forecasting is only available for stores
  {TRAINED_FORECAST_STORES}, and mention which store the user asked
  about that isn't covered. Do not invent a forecast.
- If the forecast result includes a NOTE that no store was specified and
  a default was used, you MUST clearly tell the user this assumption was
  made -- never present a defaulted forecast as if the user asked about
  that specific store.
- If given a valid forecast result, clearly label it as a PREDICTION,
  not a historical fact, and include any confidence/extrapolation
  caveat that was provided, in full.
- Do not invent numbers, model names, or claims not present in what you
  were given.
- Keep the answer to 2-4 sentences unless the question needs more.
- If you are told this is a REVISION and given a specific issue that was
  flagged, you MUST directly address that issue in your rewritten
  answer -- do not just repeat the same answer as before."""


def insight_agent(question: str, sql_result: str = "", context: str = "",
                   forecast_result: str = "", prior_issue: str = "") -> str:
    parts = [f"Question: {question}"]
    if sql_result:
        parts.append(f"SQL Results:\n{sql_result}")
    if context:
        parts.append(f"Retrieved Context:\n{context}")
    if forecast_result:
        parts.append(f"Forecast Result:\n{forecast_result}")
    if prior_issue:
        parts.append(
            f"REVISION NEEDED. A fact-checker reviewed your previous answer "
            f"and flagged this specific issue: {prior_issue}\n"
            f"Rewrite your answer to directly fix this issue."
        )
    user_message = "\n\n".join(parts)
    return _call_gemini(INSIGHT_SYSTEM, user_message)


# ---------------------------------------------------------------------
# CRITIC AGENT
# ---------------------------------------------------------------------
CRITIC_SYSTEM = f"""You are a fact-checker reviewing an analyst's answer
before it goes to a stakeholder. You'll be given the original question,
the analyst's answer, and the raw SQL results / context / forecast
result it was based on.

Check whether every specific claim (numbers, model names, comparisons,
conclusions) in the answer is actually supported by the given
data/context. Flag ANY number or named entity (model name, metric value)
that does not appear verbatim or is not directly derivable from what was
given -- do not assume it's correct just because it sounds plausible.

Also flag if the forecast result contains a NOTE about defaulting to a
store because none was specified, but the analyst's answer does NOT
mention this assumption to the user -- presenting a defaulted forecast
as if the user asked about that store is a real, flaggable issue.

The following are all CORRECT and should be APPROVED:
- An honest "no data found" answer for a date outside
  {DATASET_START_DATE} to {DATASET_END_DATE}.
- An honest "no trained model" answer for a store outside
  {TRAINED_FORECAST_STORES} when forecasting was requested.
- A forecast clearly labeled as a prediction with its caveats included.
- A forecast that clearly discloses a store default was used.

Respond in this exact format:
VERDICT: APPROVED or NEEDS_REVISION
ISSUE: <describe the specific unsupported claim, or "none" if approved>"""


def critic_agent(question: str, answer: str, sql_result: str = "", context: str = "", forecast_result: str = "") -> dict:
    user_message = (
        f"Question: {question}\n\n"
        f"Analyst's Answer: {answer}\n\n"
        f"SQL Results:\n{sql_result}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Forecast Result:\n{forecast_result}"
    )
    response = _call_gemini(CRITIC_SYSTEM, user_message)

    verdict = "APPROVED" if "VERDICT: APPROVED" in response else "NEEDS_REVISION"
    issue = ""
    if "ISSUE:" in response:
        issue = response.split("ISSUE:")[1].strip()

    return {"verdict": verdict, "issue": issue, "raw_response": response}


if __name__ == "__main__":
    # Deliberately no store mentioned, to test the new disclosure behavior
    test_question = "What will sales be next month?"

    print("=== PLANNER ===")
    plan = planner_agent(test_question)
    print("Plan:", plan)

    forecast_result = ""
    if plan == "forecast":
        print("\n=== FORECAST AGENT ===")
        f_out = forecast_agent(test_question)
        print(f_out["forecast_result"])
        print("Store defaulted:", f_out.get("store_defaulted"))
        forecast_result = f_out["forecast_result"]

    print("\n=== INSIGHT AGENT ===")
    answer = insight_agent(test_question, forecast_result=forecast_result)
    print(answer)

    print("\n=== CRITIC AGENT ===")
    critique = critic_agent(test_question, answer, forecast_result=forecast_result)
    print(critique["raw_response"])