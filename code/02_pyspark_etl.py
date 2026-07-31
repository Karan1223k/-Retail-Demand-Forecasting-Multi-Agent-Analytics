from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count

# creating a spark session
spark = SparkSession.builder \
    .appName("RetailForecastingETL") \
    .config("spark.jars", "jars/mysql-connector-j-26.7.0.jar") \
    .getOrCreate()

# Spark's JDBC connecto reads mysql not pymysql
df_spark = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:mysql://localhost:3306/retail_forecasting") \
    .option("driver", "com.mysql.cj.jdbc.Driver") \
    .option("dbtable", "model_input") \
    .option("user", "root") \
    .option("password", "Karan0207") \
    .load()


print("Row count:", df_spark.count())
df_spark.show(5)



# Calculate the average sales for promotional and non-promotional days
# using PySpark's DataFrame API. This provides an independent check of
# the SQL analysis and helps confirm that promotions are associated
# with higher average sales.
promo_summary = df_spark.groupBy("promo") \
    .agg(
        avg("sales_amount").alias("avg_sales"),
        count("*").alias("num_records")
    )

promo_summary.show()

JDBC_URL = "jdbc:mysql://localhost:3306/retail_forecasting?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC"



# Read the two raw tables separately (not the pre-joined model_input view)
stores_spark = spark.read \
    .format("jdbc") \
    .option("url", JDBC_URL) \
    .option("driver", "com.mysql.cj.jdbc.Driver") \
    .option("dbtable", "stores") \
    .option("user", "root") \
    .option("password", "Karan0207") \
    .load()

sales_spark = spark.read \
    .format("jdbc") \
    .option("url", JDBC_URL) \
    .option("driver", "com.mysql.cj.jdbc.Driver") \
    .option("dbtable", "clean_sales") \
    .option("user", "root") \
    .option("password", "Karan0207") \
    .load()

# Join them — direct equivalent of your SQL JOIN ... ON clean_sales.store_id = stores.store_id
joined = sales_spark.join(stores_spark, on="store_id", how="inner")

# Recompute the store_type/assortment averages (same as SQL Phase B query 4)
segment_summary = joined.filter(col("is_open") == 1) \
    .groupBy("store_type", "assortment") \
    .agg(
        avg("sales_amount").alias("avg_sales"),
        count("*").alias("num_records")
    ) \
    .orderBy(col("avg_sales").desc())

segment_summary.show()