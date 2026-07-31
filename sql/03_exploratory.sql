-- ============================================
-- PHASE B: EXPLORATORY ANALYSIS
-- ============================================
-- GOAL: Unlike Phase A (which checked data trustworthiness), these
-- queries answer real modeling decisions — they tell us HOW to build
-- the forecasting stage, not just confirm the data is clean.

-- --------------------------------------------
-- Query 1: Average sales by day of week
-- --------------------------------------------
-- WHY: If sales vary meaningfully by day of week, that's a real
-- seasonal pattern worth modeling. This directly informs whether
-- SARIMA (which explicitly models seasonality) earns its added
-- complexity over plain ARIMA, or whether a simpler model is enough.
SELECT
    day_of_week,
    ROUND(AVG(sales_amount), 2) AS avg_sales
FROM clean_sales
WHERE is_open = 1
GROUP BY day_of_week
ORDER BY day_of_week;

-- Query 1b: Check how many stores are actually open on each day
SELECT
    day_of_week,
    COUNT(*) AS num_open_records
FROM clean_sales
WHERE is_open = 1
GROUP BY day_of_week
ORDER BY day_of_week;

-- NOTE: Sunday's average (day_of_week=7) is NOT reliable — only 3,487
-- open records vs ~120,000 for other days (most German stores are
-- closed Sundays; only a small exempt subset show up as open here).
-- Real, trustworthy seasonality: Monday highest, Saturday lowest,
-- Tue-Fri moderate and stable.
-- This supports using SARIMA's seasonal component rather than settling for plain ARIMA, once we exclude the Sunday noise.

-- Query 2: Average sales by month
-- --------------------------------------------
-- WHY: Checks for yearly seasonality (holiday shopping, summer lulls,
-- etc.) — informs whether SARIMA's seasonal terms should be tuned for
-- a 12-month cycle, and whether Prophet's holiday-effects feature is
-- worth using.
SELECT
    MONTH(sale_date) AS month_num,
    ROUND(AVG(sales_amount), 2) AS avg_sales,
    COUNT(*) AS num_open_records
FROM clean_sales
WHERE is_open = 1
GROUP BY MONTH(sale_date)
ORDER BY month_num;

-- RESULT: Clear yearly seasonality. December average (8,659) is ~20-30%
-- above the typical range (6,600-7,200) — Christmas shopping effect.
-- Mild Aug-Oct dip. Supports using SARIMA's seasonal terms and/or
-- Prophet's holiday-effects feature rather than a non-seasonal model.

-- Query 3: Average sales on promo days vs non-promo days
-- --------------------------------------------
-- WHY: Determines whether "promo" is worth adding as an external
-- regressor in Prophet/SARIMAX. If the effect is negligible, we don't
-- need the added complexity of wiring it into the model — this query
-- is what tells us that, not a guess.
SELECT
    promo,
    ROUND(AVG(sales_amount), 2) AS avg_sales,
    COUNT(*) AS num_records
FROM clean_sales
WHERE is_open = 1
GROUP BY promo;

-- RESULT: Strong promo effect — 8,312.91 avg (promo=1) vs 5,946.86
-- avg (promo=0), a ~40% lift. Both sample sizes large and comparable
-- (327K vs 404K records) — no distortion risk. This is a real, material
-- effect. Promo will be included as an external regressor in
-- SARIMAX/Prophet, not left out.

-- --------------------------------------------
-- Query 4: Average sales by store type and assortment
-- --------------------------------------------
SELECT
    s.store_type,
    s.assortment,
    ROUND(AVG(cs.sales_amount), 2) AS avg_sales,
    COUNT(*) AS num_records
FROM clean_sales cs
JOIN stores s ON cs.store_id = s.store_id
WHERE cs.is_open = 1
GROUP BY s.store_type, s.assortment
ORDER BY avg_sales DESC;

-- RESULT: Store type 'b' has by far the highest average (17,969) but
-- represents only ~16 stores total (942-7,467 records vs 51K-244K for
-- other types) — too small a sample to trust, same issue as Sunday.
-- Reliable comparison is across types a/c/d (all large samples,
-- 6,500-7,600 range) — close enough to support one global model.
-- Consistent finding: assortment 'c' outperforms assortment 'a' within
-- every store type — worth including assortment as a categorical
-- feature/regressor. Store type 'b' should be excluded or flagged as
-- low-confidence due to sample size.



-- Query 7: Average sales by state holiday type
-- --------------------------------------------
-- WHY: Phase B only tested 'promo' as a regressor candidate. state_holiday
-- is right there in the schema and untested — this checks whether it has
-- its own independent effect (separate from the December/promo seasonality
-- already found) worth adding as a regressor too.
-- CASE WHEN translates Rossmann's raw codes ('0','a','b','c') into
-- readable labels, since 'a'/'b'/'c' mean nothing on their own.
SELECT
    CASE state_holiday
        WHEN '0' THEN 'None'
        WHEN 'a' THEN 'Public Holiday'
        WHEN 'b' THEN 'Easter Holiday'
        WHEN 'c' THEN 'Christmas'
        ELSE 'Unknown'
    END AS holiday_type,
    ROUND(AVG(sales_amount), 2) AS avg_sales,
    COUNT(*) AS num_records
FROM clean_sales
WHERE is_open = 1
GROUP BY state_holiday
ORDER BY num_records DESC;

-- RESULT: state_holiday categories (Public/Easter/Christmas) have very
-- small sample sizes (69-643 records vs 731,582 for 'None') since most
-- stores are legally closed on actual holidays. The stores that DO stay
-- open on these days are likely an unusual, non-representative subset —
-- e.g. stores in train stations, airports, or tourist/exemption zones
-- that see high foot traffic specifically because they're one of the
-- few options open that day (captive/convenience demand), not because
-- holidays generally boost sales. This inflates their average and makes
-- it unrepresentative of the other 918 stores' behavior.
-- Same distortion pattern as Sunday (day_of_week=7) and store type 'b'
-- earlier — small, unusual sample producing a misleadingly high average.
-- CONCLUSION: state_holiday should NOT be used as a regressor. The real
-- holiday-driven signal is already captured via the December monthly
-- spike (Phase B), which reflects pre-holiday shopping on normal open
-- days with a solid, trustworthy sample size.

-- --------------------------------------------
-- Query 8: Stores selling below their store_type/assortment average
-- --------------------------------------------
-- WHY: Phase B found real differences by store_type/assortment segment.
-- This identifies specific underperforming stores within their own
-- segment — useful for flagging stores worth investigating, and a
-- natural place to use a CTE instead of a subquery.
WITH segment_avg AS (
    SELECT
        s.store_type,
        s.assortment,
        AVG(cs.sales_amount) AS avg_segment_sales
    FROM clean_sales cs
    JOIN stores s ON cs.store_id = s.store_id
    WHERE cs.is_open = 1
    GROUP BY s.store_type, s.assortment
),
store_avg AS (
    SELECT
        cs.store_id,
        s.store_type,
        s.assortment,
        AVG(cs.sales_amount) AS avg_store_sales
    FROM clean_sales cs
    JOIN stores s ON cs.store_id = s.store_id
    WHERE cs.is_open = 1
    GROUP BY cs.store_id, s.store_type, s.assortment
)
SELECT
    sa.store_id,
    sa.store_type,
    sa.assortment,
    ROUND(sa.avg_store_sales, 2) AS avg_store_sales,
    ROUND(seg.avg_segment_sales, 2) AS avg_segment_sales
FROM store_avg sa
JOIN segment_avg seg
    ON sa.store_type = seg.store_type
    AND sa.assortment = seg.assortment
WHERE sa.avg_store_sales < seg.avg_segment_sales
ORDER BY (seg.avg_segment_sales - sa.avg_store_sales) DESC
LIMIT 20;


-- RESULT: 20 underperforming stores identified. Store 789 stands out
-- most (3,380 vs 7,571 segment avg, less than half). Notable cluster:
-- 6+ underperformers within type a/assortment c specifically (789, 837,
-- 969, 764, 681, 931, 1076, 488, 520) — reliable segment benchmark
-- (161K records), so this cluster is a real signal worth investigating.
-- Caution: store type 'b' results (1081, 274, 85) are compared against
-- an unreliable small-sample segment average (~16 stores total, same
-- issue flagged in Phase B) — don't treat these as confirmed findings.