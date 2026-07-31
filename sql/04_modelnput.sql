-- ============================================
-- PHASE C: FINAL MODEL-READY DATASET
-- ============================================
-- GOAL: Build the exact daily sales table that gets exported to Python
-- for forecasting. This applies everything learned in Phases A and B:
-- excludes the 180 renovation-gap stores (via clean_sales), excludes
-- the low-confidence store type 'b', keeps only actually-open days
-- (since a closed-day 0 isn't a real demand signal), and adds a 7-day
-- rolling average as both a feature and a baseline to beat.
CREATE VIEW model_input AS
SELECT
    cs.store_id,
    s.store_type,
    s.assortment,
    cs.sale_date,
    cs.day_of_week,
    cs.sales_amount,
    cs.customers,
    cs.promo,
    cs.state_holiday,
    cs.school_holiday,
    AVG(cs.sales_amount) OVER (
        PARTITION BY cs.store_id
        ORDER BY cs.sale_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7day_avg
FROM clean_sales cs
JOIN stores s ON cs.store_id = s.store_id
WHERE cs.is_open = 1
  AND s.store_type != 'b';

SELECT COUNT(DISTINCT store_id) FROM model_input;
-- Returned 919, as expected.

-- ============================================
-- BUG FOUND DURING MODELING: DATA LEAKAGE IN rolling_7day_avg
-- ============================================
-- The original window "6 PRECEDING AND CURRENT ROW" includes the current
-- day's own sales_amount in its own rolling average. When this was later
-- used as a forecasting baseline, it meant the "prediction" for a given
-- day partially contained that day's real answer — an unfair advantage
-- that made the baseline beat ARIMA/SARIMA in every test. A genuine
-- forecast can only use information available BEFORE the day being
-- predicted. Fixed by shifting the window to exclude the current row.

DROP VIEW IF EXISTS model_input;

CREATE VIEW model_input AS
SELECT
    cs.store_id,
    s.store_type,
    s.assortment,
    cs.sale_date,
    cs.day_of_week,
    cs.sales_amount,
    cs.customers,
    cs.promo,
    cs.state_holiday,
    cs.school_holiday,
    AVG(cs.sales_amount) OVER (
        PARTITION BY cs.store_id
        ORDER BY cs.sale_date
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS rolling_7day_avg
FROM clean_sales cs
JOIN stores s ON cs.store_id = s.store_id
WHERE cs.is_open = 1
  AND s.store_type != 'b';

SELECT COUNT(DISTINCT store_id) FROM model_input;
-- Should still return 919 — exclusions unchanged, only the rolling
-- average calculation itself was corrected.