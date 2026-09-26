"""
Tools Layer
-------------
Three callable tools the agents use:
  - run_sql(query): executes a read-only SQL query against model_input
  - retrieve_context(query): semantic search over the knowledge base
  - forecast_sales(store_id, date): predicts sales using a pre-trained
    Prophet model, only for stores in TRAINED_FORECAST_STORES
"""

import os
import pickle
from datetime import datetime
from pathlib import Path

import mysql.connector
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from src.config import TRAINED_FORECAST_STORES, DATASET_END_DATE

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_store"
FORECAST_MODEL_DIR = Path(__file__).parent.parent / "data" / "forecast_models"
LAST_TRAINING_DATE = datetime.strptime(DATASET_END_DATE, "%Y-%m-%d")

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _chroma_client.get_collection(
    name="rossmann_knowledge",
    embedding_function=_embedding_fn,
)

_forecast_model_cache = {}


def run_sql(query: str) -> str:
    """
    Executes a read-only SQL query against the model_input view and
    returns results as a formatted string. Blocks anything that isn't a
    SELECT.
    """
    cleaned = query.strip().rstrip(";")
    if not cleaned.lower().startswith("select"):
        return "Error: only SELECT queries are allowed."

    if "limit" not in cleaned.lower():
        cleaned += " LIMIT 200"

    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
        cursor = conn.cursor()
        cursor.execute(cleaned)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        return f"SQL Error: {e}"

    if not rows:
        return "Query returned no rows."

    formatted_rows = [[str(val) for val in row] for row in rows]
    header = " | ".join(columns)
    separator = "-" * len(header)
    body = "\n".join(" | ".join(row) for row in formatted_rows)
    return f"{header}\n{separator}\n{body}"


def retrieve_context(query: str, n_results: int = 2) -> str:
    """
    Semantic search over the knowledge base (leakage bug fix, MAPE
    findings, metric definitions). Returns the top matching chunks.
    """
    results = _collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    sources = results["metadatas"][0]

    if not docs:
        return "No relevant context found."

    parts = []
    for doc, meta in zip(docs, sources):
        parts.append(f"[Source: {meta['source']}]\n{doc}")
    return "\n\n".join(parts)


def forecast_sales(store_id: int, date_str: str) -> str:
    """
    Predicts sales for a given store and future date using a pre-trained
    Prophet model. Only works for stores in TRAINED_FORECAST_STORES --
    returns a clear message listing which stores ARE available
    otherwise, the same way an out-of-range date returns a clear
    message about the dataset's actual date range.

    Flags long-range extrapolations (far beyond the last real training
    date) so downstream agents don't present a low-confidence guess as
    a solid number.
    """
    if store_id not in TRAINED_FORECAST_STORES:
        return (
            f"No trained forecast model available for store {store_id}. "
            f"Forecasting is only available for these trained stores: "
            f"{TRAINED_FORECAST_STORES}."
        )

    model_path = FORECAST_MODEL_DIR / f"store_{store_id}.pkl"
    if not model_path.exists():
        return (
            f"Store {store_id} is in the trained store list but its model "
            f"file is missing. Run 04_train_forecast_models.py to regenerate it."
        )

    if store_id not in _forecast_model_cache:
        with open(model_path, "rb") as f:
            _forecast_model_cache[store_id] = pickle.load(f)
    model = _forecast_model_cache[store_id]

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date '{date_str}'. Expected format YYYY-MM-DD."

    future = pd.DataFrame({"ds": [date_str], "promo": [0]})
    pred = model.predict(future)
    row = pred.iloc[0]

    days_beyond_training = (target_date - LAST_TRAINING_DATE).days
    confidence_note = ""
    if days_beyond_training > 180:
        confidence_note = (
            f" NOTE: this date is {days_beyond_training} days beyond the "
            f"last real training data ({DATASET_END_DATE}). This is a "
            f"long-range trend extrapolation (the model's fitted trend and "
            f"seasonality curves are being projected far past where they "
            f"were actually fit), so it should be treated as low-confidence, "
            f"not a reliable forecast."
        )

    return (
        f"Predicted sales for store {store_id} on {date_str}: "
        f"{row['yhat']:.2f} (range: {row['yhat_lower']:.2f} to {row['yhat_upper']:.2f}). "
        f"Assumes no active promo (promo=0) since future promo status is unknown."
        f"{confidence_note}"
    )


if __name__ == "__main__":
    print("=== run_sql test ===")
    print(run_sql("SELECT store_id, sale_date, sales_amount FROM model_input WHERE store_id = 1 ORDER BY sale_date LIMIT 3"))

    print("\n=== retrieve_context test ===")
    print(retrieve_context("what is a good MAPE score"))

    print("\n=== forecast_sales test (trained store) ===")
    print(forecast_sales(1, "2015-08-15"))

    print("\n=== forecast_sales test (untrained store) ===")
    print(forecast_sales(42, "2015-08-15"))