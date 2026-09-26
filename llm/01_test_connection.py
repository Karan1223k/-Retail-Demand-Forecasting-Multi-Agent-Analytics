"""
Step 1: Connection Test
------------------------
Standalone sanity check — confirms the llm environment can connect to the
same MySQL database the original retail_forcasting pipeline uses, and can
read from the model_input view. This script is not part of the final
agent system; it's a one-time verification step.
"""

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
)

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*), COUNT(DISTINCT store_id) FROM model_input")
row_count, store_count = cursor.fetchone()
print(f"Connected successfully.")
print(f"Rows in model_input: {row_count:,}")
print(f"Distinct stores: {store_count}")

cursor.execute("SELECT MIN(sale_date), MAX(sale_date) FROM model_input")
min_date, max_date = cursor.fetchone()
print(f"Date range: {min_date} to {max_date}")

cursor.execute(
    "SELECT store_id, sale_date, sales_amount, rolling_7day_avg "
    "FROM model_input WHERE store_id = 1 ORDER BY sale_date LIMIT 3"
)
print("Sample rows:")
for row in cursor.fetchall():
    print(" ", row)

cursor.close()
conn.close()
print("Connection closed cleanly.")