# Retail Demand Forecasting

A forecasting project built on the Rossmann Store Sales dataset — but honestly, the interesting part of this project isn't the final forecast. It's everything that went wrong along the way and what fixing it taught me.

I set out to compare a few forecasting models against a simple baseline. The baseline kept winning. That was annoying, so I dug into why — and found two real methodology problems hiding underneath. Fixing them changed the whole story. This README walks through what I built and what I actually learned.

## What's in here

- A cleaned, audited SQL data warehouse (MySQL)
- A live Python pipeline pulling from that warehouse (no CSV exports)
- A PySpark stage that independently re-validates the SQL/Python results
- Multiple forecasting models — ARIMA, SARIMA, SARIMAX, Prophet — compared fairly against each other and against two different baselines

## Dataset

**Rossmann Store Sales** (Kaggle) — daily sales records for 1,115 European retail stores, spanning January 2013 to July 2015. Each row is one store's total sales for one day (already aggregated, not individual transactions).

Source: [Rossmann Store Sales on Kaggle](https://www.kaggle.com/c/rossmann-store-sales)

The raw CSVs aren't included in this repo (they're a few hundred MB and freely available at the link above) — download `store.csv` and `train.csv` and place them wherever your local setup points to.

## The pipeline

```
MySQL (schema, cleaning, exploratory SQL)
        │
        ▼
Python (pandas — sanity checks, stats, visual EDA)
        │
        ▼
PySpark (independent re-validation via JDBC — same numbers, different tool)
        │
        ▼
Forecasting (baseline comparisons → ARIMA → SARIMA → SARIMAX → Prophet)
```

## Data cleaning — what I found before touching any model

I checked every store's date range against its actual row count and found 180 stores had a gap. Not a random, scattered gap — all 180 shared the *exact same* 184-day hole. That precision was the giveaway: it wasn't corrupted data, it was a real event. A batch of stores had been closed for extended renovations, and unlike an ordinary closed day (which still logs a row with zero sales), these months just have no rows at all. I excluded those 180 stores rather than guessing six months of fake numbers to fill the gap — left with 935 clean stores.

I also caught the same trap twice more while exploring the data: an average that looked exciting turned out to be based on almost no supporting records. Sunday appeared to be the best sales day — until I noticed only ~3,500 records backed that number, versus ~120,000 for every other day (most German stores are simply closed on Sundays). One store type appeared to sell nearly triple everyone else — until I found it represented only 16 stores total. Both got flagged and excluded rather than trusted.

What *did* hold up under scrutiny: a real, statistically confirmed ~40% sales lift from promotions, and a genuine December seasonal spike, both backed by large, comparable sample sizes.

**📊 Chart to add here: Sales Distribution Histogram**
*(the `sales_amount` histogram from your visual EDA — shows the right-skewed distribution with the long December/promo tail)*

## The baseline that wouldn't lose

I built a simple 7-day rolling average as a baseline and tested ARIMA and SARIMA against it. The baseline won every time. That felt wrong — if a naive average always beats the "real" models, what was the point of building them?

**Turned out the baseline was cheating.** The rolling average, as I'd built it in SQL, included the *current day's own sales* in its own prediction. It wasn't forecasting — it was peeking at the answer. I fixed the SQL window to only use the 7 days strictly before the day being predicted.

That helped, but the baseline still won in most stores. Something else was off.

**The comparison itself was unfair.** The rolling baseline naturally updates every day with real, fresh data. ARIMA and SARIMA, meanwhile, were forecasting all 42 test days in one blind shot, with zero updates along the way — like asking someone to guess six weeks of weather today, versus a forecaster who checks this morning's real temperature before predicting the afternoon. I rebuilt the evaluation as **walk-forward validation**: refit the model daily, reveal the real outcome, then predict the next day. I also added `promo` as an input for the first time, since I already knew from a t-test that it had a large, real effect — and nothing had actually used it yet.

**📊 Chart to add here: Store 1 Daily Sales Trend**
*(the line plot of store 1's sales over 2013–2015 — good spot to show the visual seasonality/stationarity story before the modeling section)*

Once evaluation was fair, walk-forward SARIMAX beat the baseline in 4 of 6 tested stores, by 25–43%.

## Isolating what actually caused the win

I'd bundled two changes together — the fairer evaluation *and* the promo feature — so I couldn't tell which one deserved the credit. I built a "frozen" baseline (the honest one-shot equivalent: just repeat the last known average for all 42 days, no updates) so ARIMA/SARIMA had a fair reference point too. Then I split walk-forward into pieces — ARIMA alone, SARIMA alone, SARIMAX with promo — and ran all of it across all 6 stores.

**📊 Chart to add here: Promo vs. Non-Promo Boxplot**
*(the boxplot comparing sales_amount distributions by promo status — pairs well with this section since it visually backs up the t-test finding referenced above)*

The real, precise answer: **walk-forward validation beat one-shot forecasting in every single store, without exception.** Promo's added value on top of that was inconsistent — it helped in half the stores, did nothing (or slightly hurt) in the other half.

## Prophet — the same lesson, a sharper result

I tried Prophet expecting something similar, and it repeated almost exactly. One-shot Prophet with promo actually got *worse* than basic Prophet. But walk-forward Prophet with promo produced the single best result across the entire project — beating every ARIMA/SARIMA/SARIMAX variant and both baselines. Same lesson, different tool: the model needed to see real outcomes unfold day by day to use the promo signal well. Handing it the whole future schedule upfront wasn't enough.

## The actual finding

The headline isn't "this model won." It's that **how you evaluate a forecast matters more than which model you pick or which features you add to it.** Two completely different model families — ARIMA-based and Prophet — landed on that same conclusion independently, which is what makes it a real finding rather than a fluke of one algorithm.

## What I'd do next with more time

- Test a November–December window specifically, to see whether the models handle a real, unseen holiday spike differently than the summer window I tested
- Extend Prophet's walk-forward validation to the full 6-store sample (currently confirmed strong on one store)
- Extend the stationarity check beyond the current 11-store sample to the full 919

## Tech stack

MySQL · Python (pandas, SQLAlchemy, statsmodels, pmdarima, scikit-learn, Prophet) · PySpark · matplotlib

## Setup

```bash
# Clone the repo
git clone https://github.com/Karan1223k/retail-demand-forecasting.git
cd retail-demand-forecasting

# Set up the Python environment
cd code
python3 -m venv retail_env
source retail_env/bin/activate
pip install pandas sqlalchemy pymysql python-dotenv scipy statsmodels pmdarima prophet scikit-learn matplotlib pyspark

# Create a .env file (not included, for obvious reasons) with:
# DB_HOST=localhost
# DB_USER=root
# DB_PASSWORD=your_password
# DB_NAME=retail_forecasting

# Run the SQL files in sql/ in order (01 through 04) against your MySQL instance,
# then run the Python files in code/ in order (01 through 04)
```

For the PySpark stage specifically, you'll also need the [MySQL Connector/J](https://dev.mysql.com/downloads/connector/j/) jar placed in `code/jars/`.
