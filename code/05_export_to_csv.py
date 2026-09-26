import mysql.connector
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

query = """
SELECT
    store_id, store_type, assortment, sale_date, day_of_week,
    sales_amount, customers, promo, state_holiday, school_holiday,
    rolling_7day_avg
FROM model_input
"""

df = pd.read_sql(query, conn)
conn.close()

print(df.shape)
print(df.dtypes)
print(df.head())

df.to_csv("rossmann_model_input.csv", index=False)
print("Saved rossmann_model_input.csv")