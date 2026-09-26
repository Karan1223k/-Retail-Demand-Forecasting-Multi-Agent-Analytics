# Data Pipeline & Dataset Notes

## Dataset Coverage
The `model_input` view covers daily sales for 919 Rossmann stores from
2013-01-01 to 2015-07-31. It excludes:
- Store type 'b' — excluded due to low-confidence/sparse data for this
  store type during exploratory analysis.
- Any day where the store was closed (`is_open = 0`) — a closed day has
  sales_amount = 0, which is not a real demand signal and would distort
  any average or forecast if included.
- 180 stores affected by a "renovation gap" (long contiguous stretches of
  missing/zero sales, likely due to store renovation) were excluded via
  the `clean_sales` filtering stage before this view was built.

Store type breakdown in the final dataset: type 'a' (518 stores), type 'c'
(134 stores), type 'd' (267 stores).

## Critical Bug Fix: Data Leakage in rolling_7day_avg

During model development, a data leakage bug was discovered in the
`rolling_7day_avg` feature. The original SQL window function was:

```sql
AVG(sales_amount) OVER (
    PARTITION BY store_id
    ORDER BY sale_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
)
```

This window includes the CURRENT day's own sales_amount in its own rolling
average. When this feature was used as a forecasting baseline, it meant
the "prediction" for a given day partially contained that day's real
answer — an unfair advantage that made this naive baseline outperform
ARIMA/SARIMA in every test, which was a red flag rather than a real result.

A genuine forecast can only use information available BEFORE the day
being predicted. The fix was to shift the window to exclude the current
row:

```sql
AVG(sales_amount) OVER (
    PARTITION BY store_id
    ORDER BY sale_date
    ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
)
```

After the fix, the row count of distinct stores stayed the same (919) —
confirming the fix only changed the rolling average calculation itself,
not which rows/stores were included. Any analysis question involving
"rolling average" or "baseline comparison" should account for the fact
that the corrected (leak-free) version is what's in this dataset.

## Validation Methodology: Walk-Forward vs One-Shot

Two validation strategies were compared for the forecasting models:
- One-shot validation: a single train/test split (e.g., train on
  2013-2014, test on 2015).
- Walk-forward validation: the model is retrained on an expanding window
  and evaluated on each subsequent time step, better simulating how the
  model would actually be used in production.

Finding: walk-forward validation consistently outperformed one-shot
validation across the models tested (ARIMA, SARIMA, SARIMAX, Prophet).

## Best Model Result

The best-performing configuration found was:
Prophet + walk-forward validation + promo as a regressor -> MAPE of 0.078
(7.8% mean absolute percentage error).

This was the best result across all tested models (ARIMA, SARIMA,
SARIMAX, Prophet) and validation strategies (one-shot, walk-forward).