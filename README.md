# Retail Demand Forecasting

A forecasting project built on the Rossmann Store Sales dataset — but honestly, the interesting part of this project isn't the final forecast. It's everything that went wrong along the way and what fixing it taught me.

I set out to compare a few forecasting models against a simple baseline. The baseline kept winning. That was annoying, so I dug into why — and found two real methodology problems hiding underneath. Fixing them changed the whole story. This README walks through what I built and what I actually learned.

## What's in here

- A cleaned, audited SQL data warehouse (MySQL)
- A live Python pipeline pulling straight from that warehouse
- A PySpark stage that independently re-validates the SQL/Python results
- Multiple forecasting models — ARIMA, SARIMA, SARIMAX, Prophet — compared fairly against each other and against two different baselines
- A multi-agent AI analyst (LangGraph + Gemini) that answers plain-English questions about the data, grounded in live SQL, the project's own documented findings, and pre-trained Prophet models — with a Critic agent that fact-checks every answer before you see it

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
        │
        ▼
Multi-agent AI analyst (LangGraph — ask questions in plain English, Streamlit UI)
```

## Data cleaning — what I found before touching any model

I checked every store's date range against its actual row count and found 180 stores had a gap. Not a random, scattered gap — all 180 shared the *exact same* 184-day hole. That precision was the giveaway: it wasn't corrupted data, it was a real event. A batch of stores had been closed for extended renovations, and unlike an ordinary closed day (which still logs a row with zero sales), these months just have no rows at all. I excluded those 180 stores rather than guessing six months of fake numbers to fill the gap — left with 935 clean stores.

I also caught the same trap twice more while exploring the data: an average that looked exciting turned out to be based on almost no supporting records. Sunday appeared to be the best sales day — until I noticed only ~3,500 records backed that number, versus ~120,000 for every other day (most German stores are simply closed on Sundays). One store type appeared to sell nearly triple everyone else — until I found it represented only 16 stores total. Both got flagged and excluded rather than trusted.

What *did* hold up under scrutiny: a real, statistically confirmed ~40% sales lift from promotions, and a genuine December seasonal spike, both backed by large, comparable sample sizes.

<img width="759" height="486" alt="Screenshot 2026-07-31 at 11 50 45 AM" src="https://github.com/user-attachments/assets/e629b00c-65aa-4ecb-ac63-ff3e53e56236" />

*Distribution of daily sales — right-skewed, with a long tail from December and promo-driven spikes.*

## The baseline that wouldn't lose

I built a simple 7-day rolling average as a baseline and tested ARIMA and SARIMA against it. The baseline won every time. That felt wrong — if a naive average always beats the "real" models, what was the point of building them?

**Turned out the baseline was cheating.** The rolling average, as I'd built it in SQL, included the *current day's own sales* in its own prediction. It wasn't forecasting — it was peeking at the answer. I fixed the SQL window to only use the 7 days strictly before the day being predicted.

That helped, but the baseline still won in most stores. Something else was off.

**The comparison itself was unfair.** The rolling baseline naturally updates every day with real, fresh data. ARIMA and SARIMA, meanwhile, were forecasting all 42 test days in one blind shot, with zero updates along the way — like asking someone to guess six weeks of weather today, versus a forecaster who checks this morning's real temperature before predicting the afternoon. I rebuilt the evaluation as **walk-forward validation**: refit the model daily, reveal the real outcome, then predict the next day. I also added `promo` as an input for the first time, since I already knew from a t-test that it had a large, real effect — and nothing had actually used it yet.

<img width="1088" height="390" alt="image" src="https://github.com/user-attachments/assets/6b07f11a-c14f-46e2-8ee8-3ea3af1b53a4" />

Once evaluation was fair, walk-forward SARIMAX beat the baseline in 4 of 6 tested stores, by 25–43%.

## Isolating what actually caused the win

I'd bundled two changes together — the fairer evaluation *and* the promo feature — so I couldn't tell which one deserved the credit. I built a "frozen" baseline (the honest one-shot equivalent: just repeat the last known average for all 42 days, no updates) so ARIMA/SARIMA had a fair reference point too. Then I split walk-forward into pieces — ARIMA alone, SARIMA alone, SARIMAX with promo — and ran all of it across all 6 stores.

<img width="617" height="465" alt="image" src="https://github.com/user-attachments/assets/bf9363f2-fcbb-4a7d-ae30-d33d7b1f3e98" />

The real, precise answer: **walk-forward validation beat one-shot forecasting in every single store, without exception.** Promo's added value on top of that was inconsistent — it helped in half the stores, did nothing (or slightly hurt) in the other half.

## Prophet — the same lesson, a sharper result

I tried Prophet expecting something similar, and it repeated almost exactly. One-shot Prophet with promo actually got *worse* than basic Prophet. But walk-forward Prophet with promo produced the single best result across the entire project — beating every ARIMA/SARIMA/SARIMAX variant and both baselines. Same lesson, different tool: the model needed to see real outcomes unfold day by day to use the promo signal well. Handing it the whole future schedule upfront wasn't enough.

## The actual finding

The headline isn't "this model won." It's that **how you evaluate a forecast matters more than which model you pick or which features you add to it.** Two completely different model families — ARIMA-based and Prophet — landed on that same conclusion independently, which is what makes it a real finding rather than a fluke of one algorithm.

## Phase 2 — a multi-agent AI analyst on top of the pipeline

Once the forecasting work was done, I wanted anyone — not just someone who can write SQL — to be able to ask questions about the data. So I built an AI analyst on top of the same MySQL warehouse (it lives in `llm/`).

The obvious risk with an LLM answering data questions is that it confidently makes numbers up. The whole design is built around stopping that: every number in an answer has to come from a real SQL result, the project's own notes, or a real Prophet model — and a separate agent checks that before the answer is shown.

<!-- SCREENSHOT 1: full app view -->
> 📸 **[Screenshot: the Streamlit app home screen]**

### How a question flows through the agents

```
Question
   │
   ▼
Planner ──── decides what's needed: sql / context / forecast / both
   │
   ├──► Query agent     writes a SQL SELECT and runs it on MySQL (read-only, 200-row cap)
   ├──► Retrieve        semantic search over the project's own notes (ChromaDB, local embeddings)
   └──► Forecast agent  runs a pre-trained Prophet model (stores 1–5)
   │
   ▼
Insight agent ──── writes a 2–4 sentence plain-English answer from ONLY what it was given
   │
   ▼
Critic agent ───── checks every number and claim against the raw data
   │                 APPROVED → done
   └── NEEDS_REVISION → back to Insight once, with the exact issue flagged
```

| Agent | Job |
|---|---|
| **Planner** | Classifies the question so only the tools that are actually needed run |
| **Query** | Text-to-SQL against the `model_input` table. Anything that isn't a `SELECT` is blocked |
| **Retrieve** | Pulls the 3 most relevant chunks from the knowledge base (the bug fixes, MAPE findings and metric definitions from Phase 1) |
| **Forecast** | Extracts store + date from the question and calls a saved Prophet model. Prediction is instant because the models are trained once, up front |
| **Insight** | Turns the raw results into an answer a non-technical stakeholder can read |
| **Critic** | Fact-checker. Flags any number or claim that isn't directly supported by the data it was given |

### Example: a historical question (SQL path)

<!-- SCREENSHOT 2: ask "What was the average sales amount for store 1 in January 2013?" and expand "See agent steps" -->
> 📸 **[Screenshot: SQL question with agent steps expanded]**

### Example: a question about the project itself (knowledge-base path)

<!-- SCREENSHOT 3: ask "What bug was found in the rolling average feature?" -->
> 📸 **[Screenshot: knowledge-base question]**

### Example: a forecast

<!-- SCREENSHOT 4: ask "What will store 1's sales be on 2015-08-20?" and expand "See agent steps" -->
> 📸 **[Screenshot: forecast question with agent steps expanded]**

### Being honest about what it *can't* do

The part I care most about isn't the happy path — it's what happens when the system doesn't have the answer. It never quietly guesses:

- Ask about a date outside the data (Jan 2013 – Jul 2015) → it says no data exists for that period, instead of inventing a number
- Ask for a forecast for a store without a trained model → it says which stores *are* available
- Ask for a forecast without naming a store → it defaults to store 1 and **tells you it did**
- Ask for a date far in the future → the forecast is labelled a low-confidence long-range extrapolation

<!-- SCREENSHOT 5: ask "What will store 42's sales be next month?" -->
> 📸 **[Screenshot: the system declining to forecast an untrained store]**

### Evaluation

`llm/03_eval_harness.py` runs a fixed set of test questions through the full pipeline and checks each answer against ground truth pulled directly from SQL (or against expected keywords for knowledge-base questions).

| Question type | Correct |
|---|---|
| SQL (historical numbers) | 6 / 6 |
| Knowledge base | 5 / 5 |
| Forecast (incl. untrained-store and no-store edge cases) | 5 / 5 |
| **Total** | **16 / 16** |

The Critic sent 1 of the 16 answers back for revision; median latency was ~3.8s per question. To be clear about the limits: 16 questions is a small test set, and a perfect score says more about how easy the set is than about the system being perfect. Building a larger, harder benchmark is the next step (see below).

## What I'd do next with more time

- Test a November–December window specifically, to see whether the models handle a real, unseen holiday spike differently than the summer window I tested
- Extend Prophet's walk-forward validation to the full 6-store sample (currently confirmed strong on one store)
- Extend the stationarity check beyond the current 11-store sample to the full 919
- Grow the AI analyst's evaluation set to a few hundred questions, including deliberately tricky ones, so it can actually tell good versions of the system from bad ones
- Train forecast models for more than 5 stores

## Tech stack

MySQL · Python (pandas, SQLAlchemy, statsmodels, pmdarima, scikit-learn, Prophet) · PySpark · matplotlib

**AI analyst:** LangGraph · Google Gemini · ChromaDB · sentence-transformers · Streamlit

## Setup

```bash
# Clone the repo
git clone https://github.com/Karan1223k/retail-demand-forecasting.git
cd retail-demand-forecasting

# Set up the Python environment
cd code
python3 -m venv retail_env
source retail_env/bin/activate
pip install pandas sqlalchemy pymysql mysql-connector-python python-dotenv scipy statsmodels pmdarima prophet scikit-learn matplotlib pyspark

# Create a .env file (not included, for obvious reasons) with:
# DB_HOST=localhost
# DB_USER=root
# DB_PASSWORD=your_password
# DB_NAME=retail_forecasting

# Run the SQL files in sql/ in order (01 through 04) against your MySQL instance,
# then run the Python files in code/ in order (01 through 04)
```

For the PySpark stage specifically, you'll also need the [MySQL Connector/J](https://dev.mysql.com/downloads/connector/j/) jar placed in `code/jars/`.

### AI analyst (`llm/`)

Needs the MySQL database from above to be running, plus a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
cd llm
pip install -r requirements.txt

# Create llm/.env with the same DB_* values as above, plus:
# GEMINI_API_KEY=your_key

python 01_test_connection.py        # check the database connection
python 02_build_vector_store.py     # one-time: build the knowledge base (already included in the repo)
python 04_train_forecast_models.py  # one-time: train Prophet for stores 1–5 (already included in the repo)

streamlit run app.py                # launch the chat UI
python 03_eval_harness.py           # optional: run the evaluation
```

The free Gemini tier has rate limits, so the app spaces questions ~65 seconds apart.
