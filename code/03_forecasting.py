import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

scale_factor = 1000
load_dotenv()
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
df = pd.read_sql("SELECT * FROM model_input", engine)
df['sale_date'] = pd.to_datetime(df['sale_date'])
df = df.dropna(subset=['rolling_7day_avg'])

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# WALK-FORWARD FUNCTION
# Predicts one day at a time instead of guessing all 42 days at once.
# After each prediction, the REAL outcome for that day is revealed
# before predicting the next — like checking today's actual weather
# before forecasting tomorrow's, instead of guessing 6 weeks blind.
# exog_col toggles promo on/off; seasonal_order toggles seasonality
# on/off — so this one function can act as ARIMA, SARIMA, or SARIMAX.

def walk_forward_general(train, test, exog_col=None, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)):
    history_y = list(train['sales_amount'])
    history_exog = list(train[exog_col]) if exog_col else None
    predictions = []

    for t in range(len(test)):
        if exog_col:
            model = SARIMAX(
                history_y, exog=history_exog,
                order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False
            )
        else:
            model = SARIMAX(
                history_y,
                order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False
            )
        fit = model.fit(disp=False)

        if exog_col:
            next_exog = [test[exog_col].iloc[t]]
            yhat = fit.forecast(steps=1, exog=[next_exog])[0]
        else:
            yhat = fit.forecast(steps=1)[0]

        predictions.append(yhat)
        history_y.append(test['sales_amount'].iloc[t])
        if exog_col:
            history_exog.append(test[exog_col].iloc[t])

    return predictions


def evaluate_store(store_id, df, test_size=42):
    """Runs all 6 comparison-table rows for one store."""
    store_data = df[df['store_id'] == store_id].sort_values('sale_date').reset_index(drop=True)
    if len(store_data) < test_size + 100:
        return None

    train = store_data.iloc[:-test_size]
    test = store_data.iloc[-test_size:]
    results = {"store_id": store_id}

    # Row 1: Rolling average (walk-forward naturally, no seasonality/promo model)
    results["rolling_avg_mape"] = mean_absolute_percentage_error(
        test['sales_amount'], test['rolling_7day_avg']
    )

    # Row 2: ARIMA one-shot
    try:
        arima_fit = ARIMA(train['sales_amount'], order=(5, 0, 1)).fit()
        arima_forecast = arima_fit.forecast(steps=test_size)
        results["arima_oneshot_mape"] = mean_absolute_percentage_error(test['sales_amount'], arima_forecast)
    except Exception:
        results["arima_oneshot_mape"] = None

    # Row 3: SARIMA one-shot
    try:
        train_scaled_local = train['sales_amount'] / scale_factor
        sarima_fit = SARIMAX(
            train_scaled_local, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False, maxiter=200)
        sarima_forecast = sarima_fit.forecast(steps=test_size) * scale_factor
        results["sarima_oneshot_mape"] = mean_absolute_percentage_error(test['sales_amount'], sarima_forecast)
    except Exception:
        results["sarima_oneshot_mape"] = None

    # Row 4: ARIMA walk-forward (no seasonality, no promo)
    try:
        preds = walk_forward_general(train, test, exog_col=None, order=(5, 0, 1), seasonal_order=(0, 0, 0, 0))
        results["arima_wf_mape"] = mean_absolute_percentage_error(test['sales_amount'], preds)
    except Exception:
        results["arima_wf_mape"] = None

    # Row 5: SARIMA walk-forward (seasonality, no promo)
    try:
        preds = walk_forward_general(train, test, exog_col=None, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
        results["sarima_wf_mape"] = mean_absolute_percentage_error(test['sales_amount'], preds)
    except Exception:
        results["sarima_wf_mape"] = None

    # Row 6: SARIMAX walk-forward (seasonality + promo)
    try:
        preds = walk_forward_general(train, test, exog_col='promo', order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
        results["sarimax_wf_mape"] = mean_absolute_percentage_error(test['sales_amount'], preds)
    except Exception:
        results["sarimax_wf_mape"] = None

    return results


# Run on 3 stores first
sample_stores = [1, 87, 789,150, 837, 969]

all_results = []
for sid in sample_stores:
    print(f"\nEvaluating store {sid}...")
    r = evaluate_store(sid, df)
    if r:
        all_results.append(r)
        print(r)

results_df = pd.DataFrame(all_results)
print("\n--- FULL 6-ROW COMPARISON (3 stores) ---")
print(results_df)