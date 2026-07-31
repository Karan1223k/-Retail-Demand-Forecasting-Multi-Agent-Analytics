-- PHASE A: DATA QUALITY AUDIT
-- ============================================
-- GOAL: Before building any forecasting model, verify the sales data
-- is trustworthy. Time series models (ARIMA/Prophet/SARIMA) assume a
-- continuous daily record per store — silent gaps or duplicates would
-- either break the model or produce misleadingly bad/good results
-- without any error being thrown. Better to catch this now than debug
-- a mysterious model failure later.

-- Query 1: Check for missing dates in each store's sales history
-- LOGIC: For each store, compare how many days *should* exist between
-- its first and last recorded sale (expected_days) against how many
-- rows actually exist (actual_days). A mismatch means days are missing.
SELECT
    store_id,
    MIN(sale_date) AS first_date,
    MAX(sale_date) AS last_date,
    DATEDIFF(MAX(sale_date), MIN(sale_date)) + 1 AS expected_days,
    COUNT(*) AS actual_days,
    (DATEDIFF(MAX(sale_date), MIN(sale_date)) + 1) - COUNT(*) AS missing_days
FROM sales
GROUP BY store_id
HAVING missing_days > 0
ORDER BY missing_days DESC;
-- RESULT: 180 out of 1,115 stores had missing dates.



-- --------------------------------------------
-- Query 2: Check whether the gaps are random or follow a pattern
-- --------------------------------------------
-- WHY: A random scatter of missing days per store would suggest a data
-- loading/import problem. Instead, if many stores share the exact same
-- gap size, that points to one real, shared event rather than corrupted
-- data — worth confirming before deciding how to handle it.
SELECT missing_days, COUNT(*) AS num_stores
FROM (
    SELECT
        store_id,
        (DATEDIFF(MAX(sale_date), MIN(sale_date)) + 1) - COUNT(*) AS missing_days
    FROM sales
    GROUP BY store_id
) AS gaps
WHERE missing_days > 0
GROUP BY missing_days
ORDER BY num_stores DESC;
-- RESULT: All 180 flagged stores share the exact same gap — 184 missing
-- days each. This is a strong signal of one real-world event (a known
-- ~6-month renovation closure affecting a batch of Rossmann stores),
-- not scattered data corruption.

-- --------------------------------------------
-- Query 3: Get the exact list of the 180 affected store IDs
-- --------------------------------------------
SELECT store_id
FROM sales
GROUP BY store_id
HAVING (DATEDIFF(MAX(sale_date), MIN(sale_date)) + 1) - COUNT(*) = 184;
-- --------------------------------------------
-- DECISION: Exclude these 180 stores from forecasting
-- --------------------------------------------
-- A 184-day gap is too large to safely interpolate — filling it in would
-- mean inventing 6 months of data for a period when the store genuinely
-- wasn't operating. Interpolating it risks a model learning a fake trend
-- from the artificial jump when sales resume post-renovation.
-- Instead of repeating this exclusion logic in every future query, it's
-- captured once in a reusable VIEW.
CREATE VIEW clean_sales AS
SELECT sa.*
FROM sales sa
WHERE sa.store_id NOT IN (
    SELECT store_id
    FROM sales
    GROUP BY store_id
    HAVING (DATEDIFF(MAX(sale_date), MIN(sale_date)) + 1) - COUNT(*) = 184
);

-- VERIFIED: clean_sales contains 935 stores (1,115 - 180), confirmed via
-- SELECT COUNT(DISTINCT store_id) FROM clean_sales;
-- From this point forward, all queries use clean_sales, not sales.

-- --------------------------------------------
-- Query 5: Check for duplicate store+date combinations
-- --------------------------------------------
-- WHY: If a store somehow has two rows for the same date, any SUM/AVG
-- aggregation later would silently double-count that day's sales.
SELECT store_id, sale_date, COUNT(*) AS row_count
FROM clean_sales
GROUP BY store_id, sale_date
HAVING row_count > 1;
-- RESULT: 0 rows returned. No duplicates. Clean.

-- --------------------------------------------
-- Query 6: Check for inconsistent open/sales data
-- --------------------------------------------
-- WHY: A store marked "closed" with recorded sales, or "open" with zero
-- sales, could indicate a data entry error worth knowing about before
-- it skews an average.
SELECT store_id, sale_date, is_open, sales_amount
FROM clean_sales
WHERE (is_open = 0 AND sales_amount > 0)
   OR (is_open = 1 AND sales_amount = 0);
   
-- RESULT: 54 rows out of ~880,000+ total rows (under 0.01%). All were
-- the "open, zero sales" pattern (no "closed but has sales" cases found)
-- — plausibly just very low-traffic days. Volume is too small to
-- meaningfully affect analysis; noted and not corrected.

-- ============================================
-- PHASE A SUMMARY
-- ============================================
-- Started with 1,115 stores / 1,017,209 sales rows.
-- Found and excluded 180 stores with a systematic 184-day gap
-- (renovation closures) → clean_sales view now covers 935 stores.
-- Verified no duplicate records.
-- Found 54 minor open/sales mismatches — immaterial, left as-is.
-- Dataset is now verified clean and ready for exploratory analysis.