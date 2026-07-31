import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(connection_string)

df = pd.read_sql("SELECT * FROM model_input", engine)

print(df.shape)
print(df.head())

# Check 1: sales date should be date type 
print(df.dtypes)

# Check 2: missing values 
print(df.isnull().sum())

# Check 3: SQL data consistency (should be 919)
print(df['store_id'].nunique())

# Check 4
print(df[['sales_amount', 'customers', 'rolling_7day_avg']].describe())

#MySQL's DATE column type SQL != pandas date often gets loaded as plain text 
df['sale_date'] = pd.to_datetime(df['sale_date'])
print(df['sale_date'].dtype)
print(df['sale_date'].min(), "to", df['sale_date'].max())


#T TEST
from scipy import stats

    # Split sales into two groups: promo days vs non-promo days
promo_sales = df[df['promo'] == 1]['sales_amount']
non_promo_sales = df[df['promo'] == 0]['sales_amount']

    # Run an independent two-sample t-test
t_stat, p_value = stats.ttest_ind(promo_sales, non_promo_sales)

print("Promo group mean:", promo_sales.mean())
print("Non-promo group mean:", non_promo_sales.mean())
print("t-statistic:", t_stat)
print("p-value:", p_value)
# FINDING: Promo days average significantly higher sales than non-promo
# days (8,255.51 vs 5,858.07, a ~41% lift). t-stat=357.20, p≈0.0 —
# confirms this is a real effect, not random chance. Formally validates
# the ~40% promo lift already found informally in SQL.


# Test stationarity across a sample of stores, not just store 1
from statsmodels.tsa.stattools import adfuller

sample_store_ids = df['store_id'].drop_duplicates().sample(10, random_state=42)

print("Testing stationarity for 10 sample stores:\n")

for sid in sample_store_ids:
    store_data = df[df['store_id'] == sid].sort_values('sale_date')
    result = adfuller(store_data['sales_amount'])
    p_value = result[1]
    verdict = "Stationary" if p_value < 0.05 else "NOT stationary"
    print(f"Store {sid}: p-value = {p_value:.6f} -> {verdict}")
# Stationarity was confirmed for Store 1 and a random sample of 10 stores,
# with all ADF test p-values below 0.05 (most below 0.0001). This suggests
# the sales series is already stationary, so differencing is not required
# before applying ARIMA or SARIMA models. Since only 10 out of 919 stores
# were tested, validating the full dataset could be explored as a future
# enhancement.



# Check the top 10 highest sales days

top_sales = df.nlargest(10, 'sales_amount')[['store_id', 'sale_date', 'sales_amount', 'customers', 'promo', 'state_holiday']]
print(top_sales)
# The top 10 sales outliers were all found to have valid business
# explanations rather than being data errors. Eight occurred during
# December, reflecting expected seasonal demand, while the remaining
# two were linked to promotional campaigns outside December. Interestingly,
# several December spikes happened even without promotions, indicating
# that the increase in sales is driven by seasonality itself rather than
# promotional activity.



# Verify that the SQL and Python implementations produce the same
# 7-day rolling average by independently recalculating the metric
# in pandas for Store 1 and comparing the results.
store1 = df[df['store_id'] == 1].sort_values('sale_date').copy()
store1['python_rolling_avg'] = store1['sales_amount'].rolling(window=7, min_periods=1).mean()

# Compare SQL's version against pandas' version
comparison = store1[['sale_date', 'sales_amount', 'rolling_7day_avg', 'python_rolling_avg']].head(10)
print(comparison)




#visual 

# FINDING: Visual EDA confirms all prior numerical findings.
# - Store 1 trend: stable/stationary with clear seasonal spikes (Dec 2013, Dec 2014)
# - Promo boxplot: visibly higher median/IQR for promo=1, matches t-test
# - Distribution: right-skewed, long tail from Dec/promo outliers already
#   explained in outlier check — may warrant log-transform if forecast
#   residuals show non-normal error patterns later
import matplotlib.pyplot as plt

# Plot 1: Daily sales trend for store 1 (visualize the stationarity finding)
plt.figure(figsize=(12, 4))
plt.plot(store1['sale_date'], store1['sales_amount'])
plt.title('Store 1 - Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.show()

# Plot 2: Promo vs non-promo boxplot (visualize the t-test finding)
plt.figure(figsize=(6, 5))
df.boxplot(column='sales_amount', by='promo')
plt.title('Sales Distribution: Promo vs Non-Promo')
plt.suptitle('')
plt.xlabel('Promo (0 = No, 1 = Yes)')
plt.ylabel('Sales Amount')
plt.show()

# Plot 3: Overall sales distribution histogram
plt.figure(figsize=(8, 5))
df['sales_amount'].hist(bins=50)
plt.title('Distribution of Daily Sales Amount')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.show()

print(df[df['store_id'] == 1][['sale_date', 'sales_amount', 'rolling_7day_avg']].head(10))