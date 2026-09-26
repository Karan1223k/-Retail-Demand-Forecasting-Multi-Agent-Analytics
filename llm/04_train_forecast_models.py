"""
Step 8b: Pre-train Forecast Models
------------------------------------
One-time setup script. Trains a Prophet model (promo as a regressor,
same approach as the original retail_forcasting project) for a small,
deliberate set of stores, and pickles each trained model to disk.

Why pre-train instead of training live per question:
  - Training is the expensive part; prediction from an already-fit
    model is near-instant, regardless of how far out the requested
    date is.
  - Keeping this to a handful of stores keeps setup time reasonable
    and keeps the Forecast Agent honest: it can only forecast for
    stores it actually has a trained model for.

Uses your live MySQL connection, same as 01_test_connection.py.

Run once: python 04_train_forecast_models.py
"""

import os
import pandas as pd
import pickle
import mysql.connector
from pathlib import Path
from dotenv import load_dotenv
from prophet import Prophet

load_dotenv()

TRAINED_STORES = [1, 2, 3, 4, 5]
MODEL_DIR = Path(__file__).parent / "data" / "forecast_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )
    query = """
        SELECT store_id, sale_date, sales_amount, promo
        FROM model_input
        WHERE store_id IN (1, 2, 3, 4, 5)
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


def train_store_model(df: pd.DataFrame, store_id: int) -> Prophet:
    """
    Trains one Prophet model for a single store, using promo as a
    regressor -- the same best-performing configuration found in the
    original project (MAPE 0.078 with Prophet + walk-forward + promo).

    Note: no walk-forward loop here. Walk-forward was a VALIDATION
    strategy used previously to measure model quality on historical
    data where the true answer is already known. For an actual forecast
    on a new/future date, you fit once on all available history and
    call predict() -- fast regardless of how far into the future the
    date is.
    """
    store_df = df[df["store_id"] == store_id].copy()
    store_df = store_df.sort_values("sale_date")

    prophet_df = store_df.rename(columns={"sale_date": "ds", "sales_amount": "y"})
    prophet_df = prophet_df[["ds", "y", "promo"]].dropna(subset=["y"])

    model = Prophet()
    model.add_regressor("promo")
    model.fit(prophet_df)

    return model


def build_all_models():
    print("Loading data from MySQL ...")
    df = load_data()

    for store_id in TRAINED_STORES:
        print(f"\nTraining model for store {store_id} ...")
        model = train_store_model(df, store_id)

        out_path = MODEL_DIR / f"store_{store_id}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved to {out_path}")

    print(f"\nDone. Trained and cached {len(TRAINED_STORES)} store models in {MODEL_DIR}")


if __name__ == "__main__":
    build_all_models()