import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
df = pd.read_sql("SELECT * FROM model_input", engine)
df['sale_date'] = pd.to_datetime(df['sale_date'])
df = df.dropna(subset=['rolling_7day_avg'])

from prophet import Prophet

# Focus on store 1 first, same store/window used throughout 03_forecasting.py
store1 = df[df['store_id'] == 1].sort_values('sale_date').reset_index(drop=True)

test_size = 42
train = store1.iloc[:-test_size]
test = store1.iloc[-test_size:]


# Prophet requires a specific format: columns must be named
# 'ds' (date) and 'y' (the value to forecast) — no exceptions.
prophet_train = train[['sale_date', 'sales_amount']].rename(columns={'sale_date': 'ds', 'sales_amount': 'y'})


# Basic Prophet model — no promo, no custom seasonality yet.
# Prophet automatically detects weekly/yearly seasonality on its own,
# unlike ARIMA/SARIMA where we had to specify seasonal_order manually.

model = Prophet()
model.fit(prophet_train)

future = model.make_future_dataframe(periods=test_size)
forecast = model.predict(future)

prophet_forecast = forecast.iloc[-test_size:]['yhat'].values

prophet_mape = mean_absolute_percentage_error(test['sales_amount'], prophet_forecast)
prophet_rmse = np.sqrt(mean_squared_error(test['sales_amount'], prophet_forecast))

print("\nProphet (basic, one-shot) performance on test set:")
print("MAPE:", prophet_mape)
print("RMSE:", prophet_rmse)


# Prophet with promo as an external regressor
prophet_train_promo = train[['sale_date', 'sales_amount', 'promo']].rename(
    columns={'sale_date': 'ds', 'sales_amount': 'y'}
)

model_promo = Prophet()
model_promo.add_regressor('promo')
model_promo.fit(prophet_train_promo)

    # Future dataframe needs promo values too, not just dates
future_promo = model_promo.make_future_dataframe(periods=test_size)
future_promo = future_promo.merge(
    store1[['sale_date', 'promo']].rename(columns={'sale_date': 'ds'}),
    on='ds', how='left'
)


    # DIAGNOSTIC + FIX: make_future_dataframe() generates a continuous daily
    # calendar, but our actual data has gaps (closed days are filtered out
    # of model_input remember on sundays most stores were closed still sales high hence 
    # ignored that finding in sql). Any generated date that doesn't exist in store1 gets
    # NaN for promo after the merge, which Prophet rejects outright.
nan_count = future_promo['promo'].isna().sum()
print(f"\nNaN count in promo column after merge: {nan_count}")
if nan_count > 0:
    print(future_promo[future_promo['promo'].isna()])

    # Fill any gap days with 0 (treat unknown/missing days as non-promo)
future_promo['promo'] = future_promo['promo'].fillna(0)

forecast_promo = model_promo.predict(future_promo)
prophet_promo_forecast = forecast_promo.iloc[-test_size:]['yhat'].values

prophet_promo_mape = mean_absolute_percentage_error(test['sales_amount'], prophet_promo_forecast)
prophet_promo_rmse = np.sqrt(mean_squared_error(test['sales_amount'], prophet_promo_forecast))

print("\nProphet (with promo regressor) performance on test set:")
print("MAPE:", prophet_promo_mape)
print("RMSE:", prophet_promo_rmse)



# Walk-forward Prophet (with promo)

def walk_forward_prophet(train, test, use_promo=True):
    history = train[['sale_date', 'sales_amount', 'promo']].rename(
        columns={'sale_date': 'ds', 'sales_amount': 'y'}
    ).copy()
    predictions = []

    for t in range(len(test)):
        model = Prophet()
        if use_promo:
            model.add_regressor('promo')
        model.fit(history)

        next_date = test['sale_date'].iloc[t]
        next_promo = test['promo'].iloc[t]

        future_row = pd.DataFrame({'ds': [next_date], 'promo': [next_promo]})
        forecast = model.predict(future_row)
        yhat = forecast['yhat'].values[0]
        predictions.append(yhat)

        new_row = pd.DataFrame({
            'ds': [next_date],
            'y': [test['sales_amount'].iloc[t]],
            'promo': [next_promo]
        })
        history = pd.concat([history, new_row], ignore_index=True)

    return predictions


wf_prophet_predictions = walk_forward_prophet(train, test, use_promo=True)

wf_prophet_mape = mean_absolute_percentage_error(test['sales_amount'], wf_prophet_predictions)
wf_prophet_rmse = np.sqrt(mean_squared_error(test['sales_amount'], wf_prophet_predictions))

print("\nWalk-forward Prophet (with promo) performance on test set:")
print("MAPE:", wf_prophet_mape)
print("RMSE:", wf_prophet_rmse)


# --- SUMMARY ---
print("\n--- SUMMARY (Store 1, Prophet) ---")
print(f"Prophet (basic, one-shot):      MAPE={prophet_mape:.4f}  RMSE={prophet_rmse:.2f}")
print(f"Prophet (promo, one-shot):      MAPE={prophet_promo_mape:.4f}  RMSE={prophet_promo_rmse:.2f}")
print(f"Prophet (walk-forward, promo):  MAPE={wf_prophet_mape:.4f}  RMSE={wf_prophet_rmse:.2f}")